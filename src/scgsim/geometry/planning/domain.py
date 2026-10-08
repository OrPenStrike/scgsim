"""Resolve solution occupancy, auto-vacuum hosts and domain boundary surfaces from source facts. No native-coordinate ownership guesses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from math import hypot, isfinite
from typing import Any

from scgsim.geometry._primitives.constants import (
    _INTERFACE_KIND_ORDER,
    _TOPOLOGY_EPS_UM,
)
from scgsim.geometry._primitives.entities import (
    _entity_by_id,
    _entity_z_range_um,
    _unique_ids,
)
from scgsim.geometry._primitives.entity_validation import (
    _is_solution_entity,
    _is_vacuum_solution_entity,
)
from scgsim.geometry._primitives.geometry_refs import (
    _route_entity_geometry_ref,
    _sidewall_geometry_refs,
)
from scgsim.geometry._primitives.loops import (
    _boolean_gdstk_region,
    _clean_loop,
    _edge_is_covered_by_loop_edge,
    _filter_gdstk_polygons,
    _gdstk_surface_region,
    _geometry_ref_from_gdstk_polygon,
    _geometry_refs_from_gdstk_region,
    _loop_centroid,
    _loop_signature,
    _loops_share_edge_overlap,
    _polygon_area,
    _ring_edges,
    _simple_interior_hole_loops,
    _split_gdstk_cutline_loop,
)
from scgsim.geometry._primitives.spatial import (
    _bounds_contains_point,
    _geometry_ref_surface_z_um,
    _interpolate_2d,
    _intersect_bounds,
    _interval_complement,
    _range_within_bounds,
    _same_z,
    _segment_overlap_interval,
    _undirected_xy_edge,
)
from scgsim.geometry._primitives.surface_records import (
    _canonical_face_signature_3d,
    _interface_kinds,
    _is_route_a_sheet_interface,
    _RouteASheetPatch,
    _surface_boundary_volume_ids,
    _surface_contribution_provenance,
)
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import InterfacePlanRecord, SurfacePlanRecord
from scgsim.geometry.planning.evidence import solution_solution_evidence
from scgsim.geometry.planning.topology import _surface_ring3d_specs
from scgsim.semantics import (
    EvidenceResult,
    SemanticEvidenceFacade,
    conductor_solution_interface_kind,
)
from scgsim.semantics.route_a import geometry_z_range, record_geometry, same_z


def verified_route_a_substrate_support(
    build_input: GeometryBuildInput,
    stack: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Bind each face metal to one positive-area dielectric contact plane.

    The normalized face polygons and modeled substrate region (rectangle or
    explicit outer/hole loops) are intersected as regions. A metal overhang
    remains exposed; neither bounds nor a point/edge touch establishes contact.
    """
    import gdstk

    layers = stack.get("layers")
    if isinstance(layers, str | bytes) or not isinstance(layers, Sequence):
        raise TypeError("Route A requires a sequence of stack layers.")
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    substrates = tuple(
        entity
        for entity in build_input.entities
        if _is_solution_entity(entity) and entity.material_kind == "dielectric"
    )
    result: dict[str, dict[str, str]] = {}
    for record in layers:
        if not isinstance(record, Mapping):
            raise TypeError("Route A stack layers must contain mappings.")
        if record.get("part_role") != "face_metal":
            continue
        source_id = record.get("semantic_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("face_metal requires semantic_id for substrate support.")
        face_z_min, face_z_max = geometry_z_range(
            record_geometry(record, source_id), source_id
        )
        if face_z_max <= face_z_min:
            raise ValueError(f"{source_id} physical face-metal thickness must be > 0.")
        face_entities = tuple(
            entity
            for entity in build_input.entities
            if entity.semantic_id == source_id
            or entity.metadata.get("semantic_group_id") == source_id
            or entity.metadata.get("source_semantic_id") == source_id
        )
        face_regions: list[Any] = []
        for entity in face_entities:
            for polygon_id in entity.polygon_ids:
                polygon = polygons.get(polygon_id)
                if polygon is None:
                    raise ValueError(
                        f"{source_id} references missing normalized polygon {polygon_id!r}."
                    )
                exterior = (gdstk.Polygon(_clean_loop(polygon.exterior)),)
                holes = tuple(
                    gdstk.Polygon(_clean_loop(hole)) for hole in polygon.holes
                )
                face_regions.extend(
                    _boolean_gdstk_region(gdstk, exterior, holes, "not")
                )
        if not face_regions:
            raise ValueError(
                f"{source_id} has no normalized face-metal polygon for substrate attachment."
            )
        matches: list[tuple[str, str]] = []
        for substrate in substrates:
            sub_z_min, sub_z_max = _entity_z_range_um(substrate)
            sides = (
                *(("lower",) if same_z(sub_z_max, face_z_min) else ()),
                *(("upper",) if same_z(sub_z_min, face_z_max) else ()),
            )
            if not sides:
                continue
            substrate_region = _solution_entity_xy_region(gdstk, substrate)
            if _boolean_gdstk_region(gdstk, face_regions, substrate_region, "and"):
                matches.extend((side, substrate.semantic_id) for side in sides)
        if len(matches) != 1:
            raise ValueError(
                f"{source_id} requires exactly one positive-area dielectric substrate "
                f"contact at its physical face; matched {matches!r} among "
                f"{tuple(entity.semantic_id for entity in substrates)!r}."
            )
        side, substrate_id = matches[0]
        result[source_id] = {"side": side, "substrate_id": substrate_id}
    return result


def _prepare_auto_vacuum_solution_regions(
    build_input: GeometryBuildInput,
    route: RouteLiteral,
    *,
    native_complement: bool = False,
) -> GeometryBuildInput:
    """Replace auto VACUUM_REGION with planner-side complement components."""
    auto_region = _auto_vacuum_solution_region(build_input)
    if auto_region is None:
        return build_input
    if auto_region.material_kind != "vacuum":
        raise ValueError(
            "auto vacuum region must have vacuum material_kind before planning"
        )

    auto_metadata = dict(auto_region.metadata)
    auto_padding = _auto_vacuum_padding(auto_metadata, auto_region.semantic_id)
    auto_bounds = _auto_vacuum_envelope_bounds(build_input, route=route)
    auto_bounds = {
        "x_min_um": auto_bounds["x_min_um"] - auto_padding["x_minus_um"],
        "y_min_um": auto_bounds["y_min_um"] - auto_padding["y_minus_um"],
        "x_max_um": auto_bounds["x_max_um"] + auto_padding["x_plus_um"],
        "y_max_um": auto_bounds["y_max_um"] + auto_padding["y_plus_um"],
        "z_min_um": auto_bounds["z_min_um"] - auto_padding["z_minus_um"],
        "z_max_um": auto_bounds["z_max_um"] + auto_padding["z_plus_um"],
    }
    if not auto_bounds["x_min_um"] < auto_bounds["x_max_um"]:
        raise ValueError("auto VACUUM_REGION has non-positive padded x extent")
    if not auto_bounds["y_min_um"] < auto_bounds["y_max_um"]:
        raise ValueError("auto VACUUM_REGION has non-positive padded y extent")
    if not auto_bounds["z_min_um"] < auto_bounds["z_max_um"]:
        raise ValueError("auto VACUUM_REGION has non-positive padded z extent")
    auto_region = replace(
        auto_region,
        geometry={
            **dict(auto_region.geometry),
            "domain_bounds_um": {
                "x_min_um": auto_bounds["x_min_um"],
                "y_min_um": auto_bounds["y_min_um"],
                "x_max_um": auto_bounds["x_max_um"],
                "y_max_um": auto_bounds["y_max_um"],
            },
            "z_min_um": auto_bounds["z_min_um"],
            "z_max_um": auto_bounds["z_max_um"],
            "outer_loop": _domain_bounds_loop(auto_bounds),
            "hole_loops": (),
        },
    )
    envelope_loop = _domain_bounds_loop(auto_bounds)
    if native_complement:
        # The curved compiler retains the same authored envelope and padding,
        # but computes its complement in the shared native planar arrangement.
        return replace(
            build_input,
            entities=tuple(
                auto_region if entity.semantic_id == auto_region.semantic_id else entity
                for entity in build_input.entities
            ),
        )
    auto_z_min_um = float(auto_bounds["z_min_um"])
    auto_z_max_um = float(auto_bounds["z_max_um"])

    all_entities = tuple(build_input.entities)
    obstacle_entities = tuple(
        entity
        for entity in all_entities
        if _is_auto_vacuum_subtractor(entity, route=route)
        and entity.semantic_id != auto_region.semantic_id
        and not bool(entity.metadata.get("is_auto_vacuum_region"))
    )

    z_events = {auto_z_min_um, auto_z_max_um}
    for entity in obstacle_entities:
        entity_z_min_um, entity_z_max_um = _entity_z_range_um(entity)
        if not (
            isfinite(entity_z_min_um)
            and isfinite(entity_z_max_um)
            and entity_z_max_um > entity_z_min_um
        ):
            raise ValueError(f"{entity.semantic_id} has non-positive z extent")
        z_events.update((entity_z_min_um, entity_z_max_um))
    z_slices = sorted(z_events)
    if len(z_slices) < 2:
        raise ValueError("auto VACUUM_REGION has no valid z sweep")

    import gdstk

    components: list[SemanticEntitySpec] = []
    component_index = 0
    for z_start, z_end in pairwise(z_slices):
        if z_end - z_start <= _TOPOLOGY_EPS_UM:
            continue
        if (
            z_start < auto_z_min_um - _TOPOLOGY_EPS_UM
            or z_end > auto_z_max_um + _TOPOLOGY_EPS_UM
        ):
            continue

        base_region = _solution_entity_xy_region(
            gdstk,
            auto_region,
            z_min_um=z_start,
            z_max_um=z_end,
        )
        if not base_region:
            continue

        active_subtractor_region: tuple[Any, ...] = ()
        active_subtractors: list[SemanticEntitySpec] = []
        for entity in obstacle_entities:
            obstacle_z_min_um, obstacle_z_max_um = _entity_z_range_um(entity)
            if (
                obstacle_z_min_um >= z_end - _TOPOLOGY_EPS_UM
                or obstacle_z_max_um <= z_start + _TOPOLOGY_EPS_UM
            ):
                continue
            subtractor_regions = _solution_entity_xy_region(
                gdstk,
                entity,
                z_min_um=z_start,
                z_max_um=z_end,
            )
            if not subtractor_regions:
                continue
            active_subtractors.append(entity)
            if active_subtractor_region:
                active_subtractor_region = _boolean_gdstk_region(
                    gdstk,
                    active_subtractor_region,
                    subtractor_regions,
                    "or",
                )
            else:
                active_subtractor_region = subtractor_regions

        vacuum_region_refs = _geometry_refs_from_gdstk_region(
            {"outer_loop": envelope_loop},
            _boolean_gdstk_region(
                gdstk,
                base_region,
                active_subtractor_region,
                "not",
            ),
        )
        if not vacuum_region_refs:
            continue

        for geometry_ref in sorted(
            vacuum_region_refs,
            key=lambda region: _loop_signature(region["outer_loop"]),
        ):
            component_region = _gdstk_surface_region(geometry_ref)
            component_subtractor_ids = tuple(
                sorted(
                    entity.semantic_id
                    for entity in active_subtractors
                    if _auto_vacuum_component_contacts_subtractor(
                        gdstk,
                        component_geometry_ref=geometry_ref,
                        subtractor=entity,
                    )
                )
            )
            boundary_entity_ids = {"bottom": [], "top": []}
            for entity in obstacle_entities:
                entity_z_min_um, entity_z_max_um = _entity_z_range_um(entity)
                boundary_key: str | None = None
                if _same_z(entity_z_max_um, z_start):
                    boundary_key = "bottom"
                elif _same_z(entity_z_min_um, z_end):
                    boundary_key = "top"
                if boundary_key is None:
                    continue
                entity_region = _solution_entity_xy_region(gdstk, entity)
                if _boolean_gdstk_region(
                    gdstk,
                    component_region,
                    entity_region,
                    "and",
                ):
                    boundary_entity_ids[boundary_key].append(entity.semantic_id)
            component_index += 1
            component_id = (
                auto_region.semantic_id
                if component_index == 1
                else f"{auto_region.semantic_id}__{component_index:04d}"
            )
            components.append(
                SemanticEntitySpec(
                    semantic_id=component_id,
                    role=auto_region.role,
                    material_id=auto_region.material_id,
                    material_kind=auto_region.material_kind,
                    priority=auto_region.priority,
                    geometry_kind=auto_region.geometry_kind,
                    host_void_semantic_id=auto_region.host_void_semantic_id,
                    route_representations=auto_region.route_representations,
                    geometry={
                        "geometry_kind": auto_region.geometry.get(
                            "geometry_kind",
                            auto_region.geometry_kind,
                        ),
                        "outer_loop": geometry_ref["outer_loop"],
                        "hole_loops": geometry_ref.get("hole_loops", ()),
                        "z_min_um": float(z_start),
                        "z_max_um": float(z_end),
                        "domain_bounds_um": _loop_domain_bounds(
                            geometry_ref["outer_loop"],
                            geometry_ref.get("hole_loops", ()),
                        ),
                        "from_soln_region": auto_region.semantic_id,
                    },
                    metadata={
                        **auto_metadata,
                        "is_auto_vacuum_region": True,
                        "auto_vacuum_group_id": auto_region.semantic_id,
                        "auto_vacuum_envelope_outer_loop": envelope_loop,
                        "auto_vacuum_component_index": component_index,
                        "auto_vacuum_z_range_um": (float(z_start), float(z_end)),
                        "auto_vacuum_subtracting_entity_ids": tuple(
                            component_subtractor_ids
                        ),
                        "auto_vacuum_boundary_entity_ids": {
                            boundary_key: tuple(sorted(set(entity_ids)))
                            for boundary_key, entity_ids in boundary_entity_ids.items()
                        },
                    },
                    polygon_ids=(),
                    labels=(),
                )
            )

    if not components:
        raise ValueError("auto VACUUM_REGION leaves no finite sweep vacuum geometry")

    retained_entities = tuple(
        entity
        for entity in all_entities
        if entity.semantic_id != auto_region.semantic_id
    )
    return replace(
        build_input,
        entities=tuple(components) + tuple(retained_entities),
        solution_regions=build_input.solution_regions,
    )


def _is_auto_vacuum_subtractor(
    entity: SemanticEntitySpec,
    route: RouteLiteral,
) -> bool:
    if bool(entity.metadata.get("is_auto_vacuum_region")):
        return False
    if _is_solution_entity(entity):
        return True

    representation = entity.route_representations.get(route)
    return representation in {"cutout_boundary_shell", "material_volume"}


def _auto_vacuum_surface_sheet_for_xy_envelope(
    entity: SemanticEntitySpec,
    route: RouteLiteral,
) -> bool:
    return route == "A" and entity.route_representations.get(route) == "surface_sheet"


def _auto_vacuum_entity_xy_bounds(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
) -> dict[str, float]:
    geometry = entity.geometry
    domain_bounds = geometry.get("domain_bounds_um")
    if isinstance(domain_bounds, Mapping):
        required = ("x_min_um", "x_max_um", "y_min_um", "y_max_um")
        missing = [name for name in required if name not in domain_bounds]
        if missing:
            raise TypeError(
                f"{entity.semantic_id} has missing {tuple(missing)} in domain_bounds_um for auto vacuum envelope."
            )
        values = [domain_bounds.get(name) for name in required]
        if any(not isfinite(float(value)) for value in values):
            raise ValueError(
                f"{entity.semantic_id} has non-finite domain_bounds_um for auto vacuum envelope."
            )
        x_min_um, x_max_um, y_min_um, y_max_um = (float(value) for value in values)
        if not (x_min_um < x_max_um and y_min_um < y_max_um):
            raise ValueError(
                f"{entity.semantic_id} has non-positive XY extent for auto vacuum envelope."
            )
        return {
            "x_min_um": x_min_um,
            "x_max_um": x_max_um,
            "y_min_um": y_min_um,
            "y_max_um": y_max_um,
        }

    if "outer_loop" in geometry:
        outer_loop = _clean_loop(geometry["outer_loop"])
        hole_loops = tuple(_clean_loop(loop) for loop in geometry.get("hole_loops", ()))
        if not all(
            isfinite(coordinate)
            for loop in (outer_loop, *hole_loops)
            for point in loop
            for coordinate in point
        ):
            raise ValueError(
                f"{entity.semantic_id} has non-finite loop geometry for auto vacuum envelope."
            )
        return _loop_domain_bounds(outer_loop, hole_loops)

    polygon_ids = tuple(entity.polygon_ids)
    if not polygon_ids:
        raise ValueError(
            f"{entity.semantic_id} must define domain_bounds_um, outer_loop, or polygon_ids for auto vacuum envelope."
        )
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    points: list[tuple[float, float]] = []
    for polygon_id in polygon_ids:
        try:
            polygon = polygons[polygon_id]
        except KeyError as exc:
            raise ValueError(
                f"{entity.semantic_id} references an unknown layout polygon for auto vacuum envelope."
            ) from exc
        loop = _clean_loop(polygon.exterior)
        if len(loop) < 3:
            raise ValueError(
                f"{entity.semantic_id} references degenerate polygon {polygon_id!r} for auto vacuum envelope."
            )
        points.extend(loop)
        for hole in polygon.holes:
            hole_loop = _clean_loop(hole)
            if not all(
                isfinite(coordinate) for point in hole_loop for coordinate in point
            ):
                raise ValueError(
                    f"{entity.semantic_id} references non-finite polygon {polygon_id!r} for auto vacuum envelope."
                )
            points.extend(hole_loop)
    if not points:
        raise ValueError(
            f"{entity.semantic_id} has no geometry points for auto vacuum envelope."
        )
    if not all(isfinite(coordinate) for point in points for coordinate in point):
        raise ValueError(
            f"{entity.semantic_id} has non-finite polygon geometry for auto vacuum envelope."
        )
    return {
        "x_min_um": min(point[0] for point in points),
        "x_max_um": max(point[0] for point in points),
        "y_min_um": min(point[1] for point in points),
        "y_max_um": max(point[1] for point in points),
    }


def _auto_vacuum_padding(
    auto_metadata: Mapping[str, Any],
    auto_region_id: str,
) -> dict[str, float]:
    raw = auto_metadata.get("vacuum_region_padding_um")
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"{auto_region_id} requires metadata vacuum_region_padding_um for auto envelope."
        )
    required = (
        "x_minus_um",
        "x_plus_um",
        "y_minus_um",
        "y_plus_um",
        "z_minus_um",
        "z_plus_um",
    )
    values = {}
    for key in required:
        value = raw.get(key)
        if (
            value is None
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise ValueError(
                f"{auto_region_id} vacuum padding {key!r} must be a finite non-negative number."
            )
        value = float(value)
        if not isfinite(value) or value < 0.0:
            raise ValueError(
                f"{auto_region_id} vacuum padding {key!r} must be a finite non-negative number."
            )
        values[key] = value
    return values


def _auto_vacuum_envelope_bounds(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
) -> dict[str, float]:
    bounds: list[dict[str, float]] = []
    for entity in build_input.entities:
        is_subtractor = _is_auto_vacuum_subtractor(entity, route=route)
        include_for_xy = _auto_vacuum_surface_sheet_for_xy_envelope(entity, route)
        if not is_subtractor and not include_for_xy:
            continue
        if bool(entity.metadata.get("is_auto_vacuum_region")):
            continue
        xy_bounds = _auto_vacuum_entity_xy_bounds(build_input, entity)
        entry = {
            **xy_bounds,
            "z_min_um": float("nan"),
            "z_max_um": float("nan"),
        }
        if is_subtractor:
            z_min_um, z_max_um = _entity_z_range_um(entity)
            if not (isfinite(z_min_um) and isfinite(z_max_um) and z_max_um > z_min_um):
                raise ValueError(
                    f"{entity.semantic_id} has non-positive or non-finite z extent."
                )
            entry["z_min_um"] = float(z_min_um)
            entry["z_max_um"] = float(z_max_um)
        elif (
            route == "A"
            and entity.route_representations.get(route) == "surface_sheet"
            and not is_subtractor
        ):
            sheet_z_um = _auto_vacuum_sheet_z(build_input, entity)
            entry["z_min_um"] = sheet_z_um
            entry["z_max_um"] = sheet_z_um

        bounds.append(entry)

    if not bounds:
        raise ValueError(
            "Cannot auto-compute vacuum envelope without route-aware positive occupancy."
        )

    finite_z_bounds = tuple(bound for bound in bounds if isfinite(bound["z_min_um"]))
    if not finite_z_bounds:
        raise ValueError(
            "Cannot auto-compute vacuum envelope without route-aware finite z occupancy."
        )
    return {
        "x_min_um": min(item["x_min_um"] for item in bounds),
        "x_max_um": max(item["x_max_um"] for item in bounds),
        "y_min_um": min(item["y_min_um"] for item in bounds),
        "y_max_um": max(item["y_max_um"] for item in bounds),
        "z_min_um": min(item["z_min_um"] for item in finite_z_bounds),
        "z_max_um": max(item["z_max_um"] for item in finite_z_bounds),
    }


def _auto_vacuum_sheet_z(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec
) -> float:
    provenance = build_input.metadata.get("route_a_thin_film")
    if not isinstance(provenance, Mapping):
        raise ValueError(
            "auto VACUUM_REGION Route-A sheets require thin-film provenance"
        )
    ranges = provenance.get("physical_face_metal_z_ranges_um")
    sheet_positions = provenance.get("effective_sheet_z_um")
    if not isinstance(ranges, Mapping) or not isinstance(sheet_positions, Mapping):
        raise ValueError("auto VACUUM_REGION Route-A sheet provenance is incomplete")
    source_id = entity.metadata.get(
        "source_semantic_id",
        entity.metadata.get("semantic_group_id", entity.semantic_id),
    )
    matches = (
        side
        for side, record in ranges.items()
        if isinstance(record, Mapping) and source_id in record.get("semantic_ids", ())
    )
    sides = tuple(matches)
    if len(sides) != 1 or not isfinite(
        float(sheet_positions.get(sides[0], float("nan")))
    ):
        raise ValueError(
            f"{entity.semantic_id} has no unique effective Route-A sheet position"
        )
    return float(sheet_positions[sides[0]])


def _auto_vacuum_solution_region(
    build_input: GeometryBuildInput,
) -> SemanticEntitySpec | None:
    candidates = tuple(
        entity
        for entity in build_input.entities
        if _is_solution_entity(entity)
        and bool(entity.metadata.get("is_auto_vacuum_region"))
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError("exactly one auto VACUUM_REGION solution entity is required")
    return candidates[0]


def _solution_entity_xy_region(
    gdstk: Any,
    entity: SemanticEntitySpec,
    *,
    z_min_um: float | None = None,
    z_max_um: float | None = None,
) -> tuple[Any, ...]:
    del z_min_um, z_max_um
    geometry = entity.geometry
    if "outer_loop" in geometry:
        outer_loop = _clean_loop(geometry["outer_loop"])
        split_outer_loop, *split_hole_loops = _split_gdstk_cutline_loop(outer_loop)
        holes = tuple(
            _clean_loop(loop)
            for loop in (
                tuple(geometry.get("hole_loops", ())) + tuple(split_hole_loops)
            )
            if len(loop) >= 3
        )
        if not holes:
            return (gdstk.Polygon(split_outer_loop),)
        return _filter_gdstk_polygons(
            gdstk.boolean(
                (gdstk.Polygon(split_outer_loop),),
                tuple(gdstk.Polygon(hole_loop) for hole_loop in holes),
                "not",
                precision=1e-9,
            )
        )
    bounds = _solution_bounds(entity)
    return (gdstk.Polygon(_domain_bounds_loop(bounds)),)


def _loop_domain_bounds(
    outer_loop: tuple[tuple[float, float], ...],
    hole_loops: tuple[tuple[tuple[float, float], ...], ...] = (),
) -> dict[str, float]:
    points = tuple(
        point for loop in (outer_loop, *hole_loops) for point in _clean_loop(loop)
    )
    return {
        "x_min_um": min(point[0] for point in points),
        "y_min_um": min(point[1] for point in points),
        "x_max_um": max(point[0] for point in points),
        "y_max_um": max(point[1] for point in points),
    }


def _planar_side_solution_regions(
    build_input: GeometryBuildInput,
    *,
    owner_id: str,
    occupied_region: tuple[Any, ...],
    plane_z_um: float,
    side: str,
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    """Resolve exact nonoverlapping solution coverage on one planar side."""
    import gdstk

    records: list[tuple[str, tuple[Any, ...]]] = []
    for solution in _solution_entities(build_input):
        z_min_um = float(solution.geometry["z_min_um"])
        z_max_um = float(solution.geometry["z_max_um"])
        on_boundary = (
            _same_z(z_max_um, plane_z_um)
            if side == "bottom"
            else _same_z(z_min_um, plane_z_um)
        )
        contains_plane = z_min_um < plane_z_um < z_max_um
        if not on_boundary and not contains_plane:
            continue
        overlap = _boolean_gdstk_region(
            gdstk,
            occupied_region,
            _solution_entity_xy_region(gdstk, solution),
            "and",
        )
        if overlap:
            records.append((solution.semantic_id, overlap))

    for index, (left_id, left_region) in enumerate(records):
        for right_id, right_region in records[index + 1 :]:
            if _boolean_gdstk_region(gdstk, left_region, right_region, "and"):
                raise ValueError(
                    f"{owner_id} {side} local adjacency is ambiguous "
                    f"between {left_id!r} and {right_id!r}."
                )
    covered: tuple[Any, ...] = ()
    for _, region in records:
        covered = (
            region
            if not covered
            else _boolean_gdstk_region(gdstk, covered, region, "or")
        )
    uncovered = _boolean_gdstk_region(gdstk, occupied_region, covered, "not")
    if uncovered:
        raise ValueError(f"{owner_id} {side} has no local solution coverage.")
    return tuple(records)


def _conductor_face_solution_pieces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    entity: SemanticEntitySpec,
    shell_part: str,
    geometry_refs: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Partition Route-A/B planar faces by exact local solution adjacency."""
    if route not in {"A", "B"}:
        adjacent_id = _conductor_face_adjacent_solution_id(
            build_input,
            entity,
            shell_part,
        )
        return tuple(
            (adjacent_id, dict(geometry_ref)) for geometry_ref in geometry_refs
        )

    z_min_um, z_max_um = _entity_z_range_um(entity)
    plane_z_um = z_min_um if shell_part == "bottom" else z_max_um
    records: list[tuple[str, dict[str, Any]]] = []
    for geometry_ref in geometry_refs:
        occupied_region = _gdstk_surface_region(geometry_ref)
        adjacent_regions = _planar_side_solution_regions(
            build_input,
            owner_id=entity.semantic_id,
            occupied_region=occupied_region,
            plane_z_um=plane_z_um,
            side=shell_part,
        )
        for adjacent_id, region in adjacent_regions:
            records.extend(
                (adjacent_id, child)
                for child in _geometry_refs_from_gdstk_region(geometry_ref, region)
            )
    return tuple(
        sorted(
            records,
            key=lambda item: (
                item[0],
                _loop_signature(item[1]["outer_loop"]),
            ),
        )
    )


def _active_route_conductor_entities(
    build_input: GeometryBuildInput,
    route: RouteLiteral,
) -> tuple[SemanticEntitySpec, ...]:
    return tuple(
        entity
        for entity in build_input.entities
        if not _is_solution_entity(entity)
        and entity.route_representations.get(route) is not None
        and "outer_loop" in entity.geometry
    )


def _entity_loop_bounds(entity: SemanticEntitySpec) -> Mapping[str, float]:
    points = tuple(
        point
        for loop in (
            entity.geometry["outer_loop"],
            *entity.geometry.get("hole_loops", ()),
        )
        for point in _clean_loop(loop)
    )
    return {
        "x_min_um": min(point[0] for point in points),
        "y_min_um": min(point[1] for point in points),
        "x_max_um": max(point[0] for point in points),
        "y_max_um": max(point[1] for point in points),
    }


def _route_a_sheet_plane_z_um_from_solutions(
    entity: SemanticEntitySpec,
    solution_entities: Sequence[SemanticEntitySpec],
) -> float:
    z_min_um, z_max_um = _entity_z_range_um(entity)
    point = _loop_centroid(entity.geometry["outer_loop"])
    for face_z_um, solution_edge_key in (
        (z_min_um, "z_max_um"),
        (z_max_um, "z_min_um"),
    ):
        for solution in solution_entities:
            if not _bounds_contains_point(_solution_bounds(solution), point):
                continue
            if not _same_z(float(solution.geometry[solution_edge_key]), face_z_um):
                continue
            if not _is_vacuum_solution_entity(solution):
                return face_z_um
    return z_min_um


def _route_a_sheet_boundary_volume_ids_from_solutions(
    entity: SemanticEntitySpec,
    solution_entities: Sequence[SemanticEntitySpec],
) -> tuple[str, ...]:
    z_min_um, z_max_um = _entity_z_range_um(entity)
    point = _loop_centroid(entity.geometry["outer_loop"])
    ids: list[str] = []
    for face_z_um, solution_edge_key in (
        (z_min_um, "z_max_um"),
        (z_max_um, "z_min_um"),
    ):
        for solution in solution_entities:
            if not _bounds_contains_point(_solution_bounds(solution), point):
                continue
            if _same_z(float(solution.geometry[solution_edge_key]), face_z_um):
                ids.append(solution.semantic_id)
                break
    return _unique_ids(ids)


def _interface_surface_kinds(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interface: InterfacePlanRecord,
    route_a_evidence: tuple[EvidenceResult, ...] = (),
) -> tuple[str, ...]:
    raw_kinds = interface.metadata.get("interface_kinds")
    if raw_kinds is not None:
        if isinstance(raw_kinds, str):
            raw_kinds = (raw_kinds,)
        return tuple(kind for kind in _INTERFACE_KIND_ORDER if kind in set(raw_kinds))
    if _is_route_a_sheet_interface(route, interface):
        derived = (
            tuple(result.classification for result in route_a_evidence)
            if route_a_evidence
            else tuple(
                _conductor_solution_interface_kind(
                    _entity_by_id(build_input, boundary_id)
                )
                for boundary_id in _route_a_sheet_boundary_volume_ids(
                    build_input,
                    _entity_by_id(build_input, interface.owner_semantic_ids[0]),
                )
            )
        )
        return tuple(kind for kind in _INTERFACE_KIND_ORDER if kind in set(derived))
    kinds = set(_interface_kinds(interface))
    return tuple(kind for kind in _INTERFACE_KIND_ORDER if kind in kinds)


def _plan_substrate_air_surfaces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    semantic_facts: SemanticEvidenceFacade | None = None,
    route_a_sheet_patches: Sequence[_RouteASheetPatch] = (),
) -> tuple[SurfacePlanRecord, ...]:
    import gdstk

    records: list[SurfacePlanRecord] = list(
        _plan_solution_domain_boundary_surfaces(build_input, route=route)
    )
    surface_index = 0
    for lower, upper, z_um, bounds in _solution_interface_planes(build_input):
        interface_region = _boolean_gdstk_region(
            gdstk,
            _solution_entity_xy_region(gdstk, lower),
            _solution_entity_xy_region(gdstk, upper),
            "and",
        )
        if not interface_region:
            continue
        material_pair = (lower.material_kind, upper.material_kind)
        if material_pair == ("dielectric", "vacuum"):
            physical_side = "top"
        elif material_pair == ("vacuum", "dielectric"):
            physical_side = "bottom"
        else:
            physical_side = "shared_plane"
        interface_evidence = solution_solution_evidence(
            semantic_facts,
            contribution_id=(
                f"solution-solution:{lower.semantic_id}:{upper.semantic_id}:z={z_um}"
            ),
            patch_id=(f"planned:{lower.semantic_id}:{upper.semantic_id}:z={z_um}"),
            lower_id=lower.semantic_id,
            upper_id=upper.semantic_id,
            side=physical_side,
        )
        kind = interface_evidence.classification
        owner_ids = interface_evidence.source_owner_ids
        interface_region = tuple(
            sorted(
                interface_region,
                key=lambda polygon: (
                    round(_loop_centroid(_clean_loop(polygon.points))[0], 9),
                    round(_loop_centroid(_clean_loop(polygon.points))[1], 9),
                    round(abs(_polygon_area(polygon.points)), 9),
                ),
            )
        )
        for patch in interface_region:
            base_geometry_ref = _geometry_ref_from_gdstk_polygon(
                {"plane": {"axis": "z", "value_um": z_um}},
                patch,
            )
            plane_conductors = _conductor_entities_on_solution_plane(
                build_input,
                route=route,
                lower=lower,
                upper=upper,
                z_um=z_um,
                base_region=(patch,),
            )
            sheet_geometry_refs = tuple(
                sheet_patch.geometry_ref
                for sheet_patch in route_a_sheet_patches
                if sheet_patch.boundary_volume_ids
                == (lower.semantic_id, upper.semantic_id)
                and _same_z(
                    _geometry_ref_surface_z_um(sheet_patch.geometry_ref),
                    z_um,
                )
                and _boolean_gdstk_region(
                    gdstk,
                    _gdstk_surface_region(sheet_patch.geometry_ref),
                    (patch,),
                    "and",
                )
            )
            solution_geometry_refs = _solution_interface_geometry_refs(
                base_geometry_ref,
                plane_conductors,
                conductor_geometry_refs=sheet_geometry_refs,
            )
            for geometry_ref in solution_geometry_refs:
                interface_id = (
                    f"{kind}__{owner_ids[0]}__{owner_ids[1]}__{surface_index:04d}"
                )
                records.append(
                    SurfacePlanRecord(
                        surface_id=f"SURF__{interface_id}",
                        owner_semantic_id=owner_ids[0],
                        surface_role="solution_interface",
                        geometry_ref=geometry_ref,
                        interface_id=interface_id,
                        valid_routes=(route,),
                        solver_use="solver_active",
                        metadata={
                            "interface_kinds": (kind,),
                            "owner_semantic_ids": owner_ids,
                            "boundary_volume_ids": (
                                lower.semantic_id,
                                upper.semantic_id,
                            ),
                            "source_provenance": (
                                _surface_contribution_provenance(
                                    parent_interface_id=(
                                        f"solution-interface:{lower.semantic_id}:"
                                        f"{upper.semantic_id}"
                                    ),
                                    patch_id=interface_evidence.contribution_id,
                                    contributions=(interface_evidence,),
                                )
                            ),
                        },
                    )
                )
                surface_index += 1
    return tuple(records)


def _solution_interface_geometry_refs(
    parent_geometry_ref: Mapping[str, Any],
    plane_conductors: Sequence[SemanticEntitySpec],
    *,
    conductor_geometry_refs: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], ...]:
    """Create live solution-interface patches after removing conductors."""
    if not plane_conductors and not conductor_geometry_refs:
        return (dict(parent_geometry_ref),)

    import gdstk

    hole_loops = _simple_interior_hole_loops(
        parent_geometry_ref,
        plane_conductors,
        conductor_geometry_refs=conductor_geometry_refs,
    )
    if hole_loops is not None:
        return ({**dict(parent_geometry_ref), "hole_loops": hole_loops},)

    base_region = _gdstk_surface_region(parent_geometry_ref)
    conductor_region = tuple(
        polygon
        for entity in plane_conductors
        for polygon in _entity_occupied_region(gdstk, entity)
    ) + tuple(
        polygon
        for geometry_ref in conductor_geometry_refs
        for polygon in _gdstk_surface_region(geometry_ref)
    )
    live_region = _boolean_gdstk_region(
        gdstk,
        base_region,
        conductor_region,
        "not",
    )
    return _geometry_refs_from_gdstk_region(parent_geometry_ref, live_region)


def _auto_vacuum_component_contacts_subtractor(
    gdstk: Any,
    *,
    component_geometry_ref: Mapping[str, Any],
    subtractor: SemanticEntitySpec,
) -> bool:
    """Whether a subtractor owns an exact boundary on one vacuum component.

    A z-slab can have multiple disconnected complement components.  The ledger
    must therefore record only a subtractor whose canonical planar boundary
    shares a nonzero segment with this component, rather than every obstacle
    active in the slab.  This remains geometry/topology bookkeeping; semantic
    identity comes from the pre-existing structured entity record.
    """
    subtractor_region = _solution_entity_xy_region(gdstk, subtractor)
    if not subtractor_region:
        return False
    base_geometry_ref: dict[str, Any]
    if "outer_loop" in subtractor.geometry:
        base_geometry_ref = {"outer_loop": subtractor.geometry["outer_loop"]}
    else:
        base_geometry_ref = {
            "outer_loop": _domain_bounds_loop(_solution_bounds(subtractor))
        }
    subtractor_refs = _geometry_refs_from_gdstk_region(
        base_geometry_ref,
        subtractor_region,
    )
    component_loops = (
        _clean_loop(component_geometry_ref["outer_loop"]),
        *(_clean_loop(loop) for loop in component_geometry_ref.get("hole_loops", ())),
    )
    return any(
        _loops_share_edge_overlap(component_loop, subtractor_loop)
        for component_loop in component_loops
        for subtractor_ref in subtractor_refs
        for subtractor_loop in (
            _clean_loop(subtractor_ref["outer_loop"]),
            *(_clean_loop(loop) for loop in subtractor_ref.get("hole_loops", ())),
        )
    )


def _entity_occupied_region(gdstk: Any, entity: SemanticEntitySpec) -> tuple[Any, ...]:
    if "outer_loop" not in entity.geometry:
        return ()
    outer = gdstk.Polygon(_clean_loop(entity.geometry["outer_loop"]))
    holes = tuple(
        gdstk.Polygon(_clean_loop(hole_loop))
        for hole_loop in entity.geometry.get("hole_loops", ())
    )
    if not holes:
        return _filter_gdstk_polygons((outer,))
    return _boolean_gdstk_region(gdstk, (outer,), holes, "not")


def _solution_domain_sidewall_geometry_refs(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    solution: SemanticEntitySpec,
    outer_loop: tuple[tuple[float, float], ...],
    z_min_um: float,
    z_max_um: float,
) -> tuple[dict[str, Any], ...]:
    refs: list[dict[str, Any]] = []
    for ring_role, ring_loop in (
        ("outer", _clean_loop(outer_loop)),
        *(
            (f"hole_{index:04d}", _clean_loop(hole_loop))
            for index, hole_loop in enumerate(solution.geometry.get("hole_loops", ()))
        ),
    ):
        for edge_index, (start, end) in enumerate(_ring_edges(ring_loop)):
            parameters = _solution_boundary_edge_parameters(
                build_input,
                route=route,
                solution=solution,
                start=start,
                end=end,
            )
            for first, second in pairwise(parameters):
                segment_start = _interpolate_2d(start, end, first)
                segment_end = _interpolate_2d(start, end, second)
                if (
                    hypot(
                        segment_end[0] - segment_start[0],
                        segment_end[1] - segment_start[1],
                    )
                    <= _TOPOLOGY_EPS_UM
                ):
                    continue
                edge_z_min_um = _solution_boundary_edge_z_min_um(
                    build_input,
                    route=route,
                    solution=solution,
                    start=segment_start,
                    end=segment_end,
                    default_z_min_um=z_min_um,
                )
                edge_z_max_um = _solution_boundary_edge_z_max_um(
                    build_input,
                    route=route,
                    solution=solution,
                    start=segment_start,
                    end=segment_end,
                    default_z_max_um=z_max_um,
                )
                if edge_z_max_um - edge_z_min_um <= _TOPOLOGY_EPS_UM:
                    continue
                refs.append(
                    {
                        "quad_points": (
                            (*segment_start, edge_z_min_um),
                            (*segment_end, edge_z_min_um),
                            (*segment_end, edge_z_max_um),
                            (*segment_start, edge_z_max_um),
                        ),
                        "sidewall_ring_role": ring_role,
                        "sidewall_edge_index": edge_index,
                    }
                )
    return tuple(refs)


def _solution_boundary_edge_parameters(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    solution: SemanticEntitySpec,
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[float, ...]:
    """Split a vacuum exterior edge at exact finite-conductor endpoints."""
    if route != "B" or not _is_vacuum_solution_entity(solution):
        return (0.0, 1.0)
    parameters = {0.0, 1.0}
    solution_z_min = float(solution.geometry["z_min_um"])
    solution_z_max = float(solution.geometry["z_max_um"])
    for entity in _active_route_conductor_entities(build_input, route):
        if entity.route_representations[route] not in {
            "cutout_boundary_shell",
            "material_volume",
        }:
            continue
        entity_z_min, entity_z_max = _entity_z_range_um(entity)
        if not (
            _same_z(entity_z_min, solution_z_min)
            or _same_z(entity_z_max, solution_z_max)
        ):
            continue
        for edge_start, edge_end in _ring_edges(
            _clean_loop(entity.geometry["outer_loop"])
        ):
            interval = _segment_overlap_interval(start, end, edge_start, edge_end)
            if interval is not None:
                parameters.update(interval)
    return tuple(sorted(parameters))


def _solution_boundary_edge_z_min_um(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    solution: SemanticEntitySpec,
    start: tuple[float, float],
    end: tuple[float, float],
    default_z_min_um: float,
) -> float:
    if not _is_vacuum_solution_entity(solution):
        return default_z_min_um
    z_min_um = default_z_min_um
    for entity in build_input.entities:
        if (
            _is_solution_entity(entity)
            or entity.route_representations.get(route)
            not in {
                "cutout_boundary_shell",
                "material_volume",
            }
            or "outer_loop" not in entity.geometry
        ):
            continue
        entity_z_min_um, entity_z_max_um = _entity_z_range_um(entity)
        if not _same_z(entity_z_min_um, default_z_min_um):
            continue
        if _edge_is_covered_by_loop_edge(start, end, entity.geometry["outer_loop"]):
            z_min_um = max(z_min_um, entity_z_max_um)
    return z_min_um


def _solution_boundary_edge_z_max_um(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    solution: SemanticEntitySpec,
    start: tuple[float, float],
    end: tuple[float, float],
    default_z_max_um: float,
) -> float:
    if not _is_vacuum_solution_entity(solution):
        return default_z_max_um
    z_max_um = default_z_max_um
    for entity in build_input.entities:
        if (
            _is_solution_entity(entity)
            or entity.route_representations.get(route)
            not in {
                "cutout_boundary_shell",
                "material_volume",
            }
            or "outer_loop" not in entity.geometry
        ):
            continue
        entity_z_min_um, entity_z_max_um = _entity_z_range_um(entity)
        if not _same_z(entity_z_max_um, default_z_max_um):
            continue
        if _edge_is_covered_by_loop_edge(start, end, entity.geometry["outer_loop"]):
            z_max_um = min(z_max_um, entity_z_min_um)
    return z_max_um


def _merge_solution_sidewall_interfaces(
    build_input: GeometryBuildInput,
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
    semantic_facts: SemanticEvidenceFacade | None = None,
) -> tuple[SurfacePlanRecord, ...]:
    """Merge coincident solution sidewalls into one shared interface surface."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    solution_ids = {
        entity.semantic_id
        for entity in entities.values()
        if _is_solution_entity(entity)
    }
    non_merge: list[SurfacePlanRecord] = []
    sidewall_groups: dict[
        tuple[tuple[float, float, float], ...],
        list[SurfacePlanRecord],
    ] = {}

    for surface in surfaces:
        if surface.construction_only:
            non_merge.append(surface)
            continue
        if (
            surface.surface_role != "domain_boundary"
            or surface.metadata.get("boundary_role") != "sidewall"
        ):
            non_merge.append(surface)
            continue
        if "quad_points" not in surface.geometry_ref:
            non_merge.append(surface)
            continue
        boundary_ids = _surface_boundary_volume_ids(
            surface,
            known_entity_ids=solution_ids,
        )
        if len(boundary_ids) != 1:
            non_merge.append(surface)
            continue
        owner_id = boundary_ids[0]
        owner = entities.get(owner_id)
        if owner is None or not _is_solution_entity(owner):
            non_merge.append(surface)
            continue
        specs = _surface_ring3d_specs(surface)
        if len(specs) != 1:
            non_merge.append(surface)
            continue
        signature = _canonical_face_signature_3d(specs[0][2])
        if signature is None:
            non_merge.append(surface)
            continue
        sidewall_groups.setdefault(signature, []).append(surface)

    grouped_surfaces: list[SurfacePlanRecord] = []
    for signature, group in sidewall_groups.items():
        if len(group) == 1:
            grouped_surfaces.append(group[0])
            continue
        all_ids = _unique_ids(
            value
            for surface in group
            for value in _surface_boundary_volume_ids(
                surface,
                known_entity_ids=solution_ids,
            )
        )
        if len(all_ids) != 2:
            names = tuple(surface.surface_id for surface in group)
            raise ValueError(
                "duplicate solution sidewall cannot define a 2-owner interface: "
                + f"{signature} => {names!r}"
            )
        lower = entities[all_ids[0]]
        upper = entities[all_ids[1]]
        interface_evidence = solution_solution_evidence(
            semantic_facts,
            contribution_id=(
                f"solution-sidewall:{lower.semantic_id}:{upper.semantic_id}:"
                f"{group[0].surface_id.rsplit('__', maxsplit=1)[-1]}"
            ),
            patch_id=f"planned:merged:{group[0].surface_id}",
            lower_id=lower.semantic_id,
            upper_id=upper.semantic_id,
            side="sidewall",
        )
        kind = interface_evidence.classification
        owner_ids = interface_evidence.source_owner_ids
        edge_suffix = group[0].surface_id.rsplit("__", maxsplit=1)[-1]
        interface_id = f"{kind}__{owner_ids[0]}__{owner_ids[1]}__{edge_suffix}"
        merged = replace(
            group[0],
            owner_semantic_id=owner_ids[0],
            interface_id=interface_id,
            metadata={
                **group[0].metadata,
                "owner_semantic_ids": owner_ids,
                "boundary_volume_ids": owner_ids,
                "interface_kinds": (kind,),
                "interface_type": kind,
                "source_provenance": _surface_contribution_provenance(
                    parent_interface_id=(
                        f"solution-sidewall:{lower.semantic_id}:{upper.semantic_id}"
                    ),
                    patch_id=interface_evidence.contribution_id,
                    contributions=(interface_evidence,),
                ),
            },
        )
        grouped_surfaces.append(merged)
    grouped_surfaces.extend(non_merge)
    return tuple(grouped_surfaces)


def _reconcile_solution_domain_boundaries(
    build_input: GeometryBuildInput,
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
) -> tuple[SurfacePlanRecord, ...]:
    """Reuse live structured interfaces in place of duplicate domain faces.

    A domain boundary is only replaced by a surface that explicitly names the
    same solution volume in its structured boundary ownership. Horizontal
    regions are reconciled by exact planar Boolean residuals; sidewalls are
    removed only for one-for-one canonical face equality. Partial vertical
    overlap remains an error for the existing topology validators to expose.
    """
    import gdstk

    solution_ids = {
        entity.semantic_id
        for entity in build_input.entities
        if _is_solution_entity(entity)
    }
    retained: list[SurfacePlanRecord] = []
    non_domain_surfaces = tuple(
        surface
        for surface in surfaces
        if not surface.construction_only
        and surface.solver_use == "solver_active"
        and surface.surface_role != "domain_boundary"
    )
    for surface in surfaces:
        if surface.construction_only or surface.surface_role != "domain_boundary":
            retained.append(surface)
            continue
        boundary_ids = _surface_boundary_volume_ids(
            surface,
            known_entity_ids=solution_ids,
        )
        if len(boundary_ids) != 1:
            retained.append(surface)
            continue
        owner_id = boundary_ids[0]
        candidates = tuple(
            candidate
            for candidate in non_domain_surfaces
            if owner_id
            in _surface_boundary_volume_ids(
                candidate,
                known_entity_ids=solution_ids,
            )
        )
        if not candidates:
            retained.append(surface)
            continue
        if "quad_points" in surface.geometry_ref:
            domain_specs = _surface_ring3d_specs(surface)
            domain_signature = (
                _canonical_face_signature_3d(domain_specs[0][2])
                if len(domain_specs) == 1
                else None
            )
            if domain_signature is not None and any(
                len(candidate_specs := _surface_ring3d_specs(candidate)) == 1
                and _canonical_face_signature_3d(candidate_specs[0][2])
                == domain_signature
                for candidate in candidates
            ):
                continue
            retained.append(surface)
            continue
        if "outer_loop" not in surface.geometry_ref:
            retained.append(surface)
            continue
        plane_z_um = _geometry_ref_surface_z_um(surface.geometry_ref)
        planar_candidates = tuple(
            candidate
            for candidate in candidates
            if "quad_points" not in candidate.geometry_ref
            and "outer_loop" in candidate.geometry_ref
            and _same_z(
                _geometry_ref_surface_z_um(candidate.geometry_ref),
                plane_z_um,
            )
        )
        if not planar_candidates:
            retained.append(surface)
            continue
        replacement_region: tuple[Any, ...] = ()
        for candidate in sorted(planar_candidates, key=lambda item: item.surface_id):
            candidate_region = _gdstk_surface_region(candidate.geometry_ref)
            replacement_region = (
                candidate_region
                if not replacement_region
                else _boolean_gdstk_region(
                    gdstk,
                    replacement_region,
                    candidate_region,
                    "or",
                )
            )
        residual_region = _boolean_gdstk_region(
            gdstk,
            _gdstk_surface_region(surface.geometry_ref),
            replacement_region,
            "not",
        )
        residual_refs = tuple(
            sorted(
                _geometry_refs_from_gdstk_region(surface.geometry_ref, residual_region),
                key=lambda geometry_ref: _loop_signature(geometry_ref["outer_loop"]),
            )
        )
        if not residual_refs:
            continue
        for index, geometry_ref in enumerate(residual_refs):
            surface_id = (
                surface.surface_id
                if len(residual_refs) == 1
                else f"{surface.surface_id}__R{index:04d}"
            )
            retained.append(
                replace(
                    surface,
                    surface_id=surface_id,
                    geometry_ref=geometry_ref,
                )
            )
    return tuple(retained)


def _plan_solution_domain_boundary_surfaces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
) -> tuple[SurfacePlanRecord, ...]:
    records: list[SurfacePlanRecord] = []
    for entity in build_input.entities:
        if not _is_solution_entity(entity):
            continue
        if "outer_loop" in entity.geometry:
            loop = _clean_loop(entity.geometry["outer_loop"])
            z_min_um = float(entity.geometry["z_min_um"])
            z_max_um = float(entity.geometry["z_max_um"])
        else:
            bounds = entity.geometry.get("domain_bounds_um")
            if not isinstance(bounds, Mapping):
                continue
            loop = _domain_bounds_loop(bounds)
            z_min_um = float(entity.geometry["z_min_um"])
            z_max_um = float(entity.geometry["z_max_um"])
        if len(loop) < 3:
            continue
        for boundary_role in ("bottom", "top"):
            geometry_refs = _solution_exterior_face_geometry_refs(
                build_input,
                entity,
                boundary_role,
            )
            for component_index, geometry_ref in enumerate(geometry_refs):
                component_suffix = (
                    "" if len(geometry_refs) == 1 else f"__{component_index:04d}"
                )
                records.append(
                    SurfacePlanRecord(
                        surface_id=(
                            f"SURF__BOUNDARY__{entity.semantic_id}__"
                            f"{boundary_role.upper()}{component_suffix}"
                        ),
                        owner_semantic_id=entity.semantic_id,
                        surface_role="domain_boundary",
                        geometry_ref=geometry_ref,
                        valid_routes=(route,),
                        metadata={
                            "owner_semantic_ids": (entity.semantic_id,),
                            "boundary_volume_ids": (entity.semantic_id,),
                            "boundary_role": boundary_role,
                        },
                    )
                )
        for edge_index, geometry_ref in enumerate(
            _solution_domain_sidewall_geometry_refs(
                build_input,
                route=route,
                solution=entity,
                outer_loop=loop,
                z_min_um=z_min_um,
                z_max_um=z_max_um,
            )
        ):
            records.append(
                SurfacePlanRecord(
                    surface_id=(
                        f"SURF__BOUNDARY__{entity.semantic_id}__"
                        f"SIDEWALL__{edge_index:04d}"
                    ),
                    owner_semantic_id=entity.semantic_id,
                    surface_role="domain_boundary",
                    geometry_ref=geometry_ref,
                    valid_routes=(route,),
                    metadata={
                        "owner_semantic_ids": (entity.semantic_id,),
                        "boundary_volume_ids": (entity.semantic_id,),
                        "boundary_role": "sidewall",
                    },
                )
            )
    return tuple(records)


def _domain_bounds_loop(bounds: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    return (
        (float(bounds["x_min_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_max_um"])),
        (float(bounds["x_min_um"]), float(bounds["y_max_um"])),
    )


def _route_a_sheet_boundary_volume_ids(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
) -> tuple[str, ...]:
    return _unique_ids(
        (
            _conductor_face_adjacent_solution_id(build_input, entity, "bottom"),
            _conductor_face_adjacent_solution_id(build_input, entity, "top"),
        )
    )


def _route_a_sheet_plane_z_um(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
) -> float:
    if any(
        solution.metadata.get("auto_vacuum_group_id") == entity.host_void_semantic_id
        for solution in _solution_entities(build_input)
    ):
        return _auto_vacuum_sheet_z(build_input, entity)
    z_min_um, z_max_um = _entity_z_range_um(entity)
    for boundary_id, z_um in (
        (
            _conductor_face_adjacent_solution_id(build_input, entity, "bottom"),
            z_min_um,
        ),
        (
            _conductor_face_adjacent_solution_id(build_input, entity, "top"),
            z_max_um,
        ),
    ):
        if not _is_vacuum_solution_entity(_entity_by_id(build_input, boundary_id)):
            return z_um
    return z_min_um


def _component_is_boundary_attached(
    build_input: GeometryBuildInput,
    *,
    solution: SemanticEntitySpec,
    surface: SurfacePlanRecord,
) -> bool:
    """Whether this PEC surface is on a solution-domain boundary plane.

    Boundary-attached Route-A conductors form part of a solution exterior
    shell; only a component wholly internal to a solution becomes an inner
    PEC void shell.
    """
    if surface.surface_role not in {
        "cutout_boundary_shell",
        "route_a_sheet_contact_cap",
    }:
        return False
    if "quad_points" in surface.geometry_ref:
        z_values = tuple(
            float(point[2]) for point in surface.geometry_ref["quad_points"]
        )
        return any(
            _same_z(z_um, float(solution.geometry["z_min_um"]))
            or _same_z(z_um, float(solution.geometry["z_max_um"]))
            for z_um in z_values
        )
    if solution.semantic_id not in _surface_boundary_volume_ids(
        surface,
        known_entity_ids={entity.semantic_id for entity in build_input.entities},
    ):
        return False
    z_min = float(solution.geometry["z_min_um"])
    z_max = float(solution.geometry["z_max_um"])
    try:
        z_um = _geometry_ref_surface_z_um(surface.geometry_ref)
    except (KeyError, TypeError, ValueError):
        return False
    return _same_z(z_um, z_min) or _same_z(z_um, z_max)


def _conductor_face_adjacent_solution_id(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
    face: str,
) -> str:
    z_min_um, z_max_um = _entity_z_range_um(entity)
    face_z_um = z_min_um if face == "bottom" else z_max_um
    auto_parent, auto_component = _auto_vacuum_host_component_id(
        build_input,
        entity=entity,
        relation=face,
        z_um=face_z_um,
    )
    if auto_component is not None:
        return auto_component
    if auto_parent:
        non_auto_solution = _solution_id_for_entity_coverage(
            build_input,
            entity,
            z_um=face_z_um,
            mode=face,
            include_auto_vacuum=False,
        )
        if non_auto_solution is not None:
            return non_auto_solution
        adjacent_solution = _solution_id_for_entity_coverage(
            build_input,
            entity,
            z_um=face_z_um,
            mode=face,
        )
        if (
            adjacent_solution is not None
            and _entity_by_id(build_input, adjacent_solution).metadata.get(
                "auto_vacuum_group_id"
            )
            == entity.host_void_semantic_id
        ):
            return adjacent_solution
        # A Route-A sheet may retain its physical film thickness in source
        # coordinates while its solution side lies inside one generated vacuum
        # component. Require full polygonal coverage by that exact component.
        containing_solution = _solution_id_for_entity_coverage(
            build_input,
            entity,
            z_um=face_z_um,
            mode="containing",
        )
        if (
            containing_solution is not None
            and _entity_by_id(build_input, containing_solution).metadata.get(
                "auto_vacuum_group_id"
            )
            == entity.host_void_semantic_id
        ):
            return containing_solution
        raise ValueError(
            f"{entity.semantic_id} {face} has no local auto-vacuum component."
        )
    exact = _solution_id_for_entity_coverage(
        build_input,
        entity,
        z_um=face_z_um,
        mode=face,
    )
    if exact is not None:
        return exact
    containing = _solution_id_for_entity_coverage(
        build_input,
        entity,
        z_um=face_z_um,
        mode="containing",
    )
    if containing is not None:
        return containing
    raise ValueError(f"{entity.semantic_id} {face} face has no adjacent solution")


def _conductor_sidewall_adjacent_solution_id(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
) -> str | None:
    z_min_um, z_max_um = _entity_z_range_um(entity)
    mid_z_um = (z_min_um + z_max_um) / 2.0
    auto_parent, auto_component = _auto_vacuum_host_component_id(
        build_input,
        entity=entity,
        relation="sidewall",
        z_um=mid_z_um,
    )
    if auto_component is not None:
        return auto_component
    if auto_parent:
        if _auto_vacuum_sidewall_is_exterior_only(build_input, entity):
            return None
        raise ValueError(
            f"{entity.semantic_id} sidewall has no local auto-vacuum component."
        )
    solution_id = _solution_id_for_entity_coverage(
        build_input,
        entity,
        z_um=mid_z_um,
        mode="containing",
    )
    if solution_id is not None:
        return solution_id
    raise ValueError(f"{entity.semantic_id} sidewall has no adjacent solution")


def _auto_vacuum_host_component_id(
    build_input: GeometryBuildInput,
    *,
    entity: SemanticEntitySpec,
    relation: str,
    z_um: float,
) -> tuple[bool, str | None]:
    """Resolve an explicitly authored auto-vacuum parent to one child only.

    This uses only the parent group recorded on an auto-vacuum component, the
    exact subtractor entity ledger for that component, and the declared z
    relation. It intentionally does not sample geometry, physical labels, or
    bounding boxes, and fails rather than choosing across multiple components.
    """
    host_id = entity.host_void_semantic_id
    if not isinstance(host_id, str) or not host_id:
        raise ValueError(f"{entity.semantic_id} requires host_void_semantic_id")
    components = []
    has_parent = False
    for solution in _solution_entities(build_input):
        metadata = solution.metadata
        if metadata.get("auto_vacuum_group_id") != host_id:
            continue
        has_parent = True
        if relation == "sidewall":
            entity_ids = metadata.get("auto_vacuum_subtracting_entity_ids")
            if isinstance(entity_ids, str | bytes) or not isinstance(
                entity_ids, Sequence
            ):
                raise TypeError(
                    f"auto-vacuum component {solution.semantic_id!r} lacks an entity subtractor ledger"
                )
        elif relation in {"bottom", "top"}:
            boundary_entity_ids = metadata.get("auto_vacuum_boundary_entity_ids")
            if not isinstance(boundary_entity_ids, Mapping):
                raise TypeError(
                    f"auto-vacuum component {solution.semantic_id!r} lacks a boundary entity ledger"
                )
            boundary_key = "top" if relation == "bottom" else "bottom"
            entity_ids = boundary_entity_ids.get(boundary_key)
            if isinstance(entity_ids, str | bytes) or not isinstance(
                entity_ids, Sequence
            ):
                raise TypeError(
                    f"auto-vacuum component {solution.semantic_id!r} has invalid {boundary_key!r} boundary ledger"
                )
        else:
            raise ValueError(f"unsupported auto-vacuum face relation {relation!r}")
        if entity.semantic_id not in entity_ids:
            continue
        z_min = float(solution.geometry["z_min_um"])
        z_max = float(solution.geometry["z_max_um"])
        if relation == "sidewall":
            matches = z_min < z_um < z_max
        elif relation == "bottom":
            matches = _same_z(z_max, z_um)
        else:
            matches = _same_z(z_min, z_um)
        if matches:
            components.append(solution.semantic_id)
    if not has_parent:
        return False, None
    if len(components) > 1:
        raise ValueError(
            f"{entity.semantic_id} {relation} requires exactly one structured "
            f"auto-vacuum component for parent {host_id!r}; got {components!r}."
        )
    return True, components[0] if components else None


def _auto_vacuum_sidewall_ref_solution_id(
    build_input: GeometryBuildInput,
    *,
    entity: SemanticEntitySpec,
    geometry_ref: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Resolve one retained sidewall segment to one explicit vacuum child.

    Auto-vacuum parents may deliberately contain disconnected components.  A
    finite conductor can then have opposite sidewalls adjacent to different
    components.  This resolver uses the per-component subtractor ledger and
    complete canonical boundary-segment coverage for *this* sidewall, never a
    centroid, bounding box, physical label, or parent-level choice.
    """
    host_id = entity.host_void_semantic_id
    if not isinstance(host_id, str) or not host_id:
        raise ValueError(f"{entity.semantic_id} requires host_void_semantic_id")
    points = geometry_ref.get("quad_points", ())
    if len(points) != 4:
        raise ValueError(f"{entity.semantic_id} sidewall requires four quad points")
    start = (float(points[0][0]), float(points[0][1]))
    end = (float(points[1][0]), float(points[1][1]))
    if _segment_overlap_interval(start, end, start, end) is None:
        raise ValueError(f"{entity.semantic_id} sidewall has degenerate XY segment")
    z_values = tuple(float(point[2]) for point in points)
    z_min_um, z_max_um = min(z_values), max(z_values)
    if not z_max_um > z_min_um:
        raise ValueError(f"{entity.semantic_id} sidewall has non-positive z extent")
    matches: list[str] = []
    has_parent = False
    for solution in _solution_entities(build_input):
        metadata = solution.metadata
        if metadata.get("auto_vacuum_group_id") != host_id:
            continue
        has_parent = True
        entity_ids = metadata.get("auto_vacuum_subtracting_entity_ids")
        if isinstance(entity_ids, str | bytes) or not isinstance(entity_ids, Sequence):
            raise TypeError(
                f"auto-vacuum component {solution.semantic_id!r} lacks an entity subtractor ledger"
            )
        if entity.semantic_id not in entity_ids:
            continue
        solution_z_min_um = float(solution.geometry["z_min_um"])
        solution_z_max_um = float(solution.geometry["z_max_um"])
        if not (
            _same_z(solution_z_min_um, z_min_um)
            and _same_z(solution_z_max_um, z_max_um)
        ):
            continue
        if _sidewall_segment_is_component_boundary(start, end, solution):
            matches.append(solution.semantic_id)
    if not has_parent:
        return False, None
    if len(matches) != 1:
        raise ValueError(
            f"{entity.semantic_id} sidewall segment {start!r}->{end!r} requires "
            f"exactly one auto-vacuum child for parent {host_id!r}; got {matches!r}."
        )
    return True, matches[0]


def _adjacent_auto_vacuum_sidewall_component_id(
    build_input: GeometryBuildInput,
    *,
    entity: SemanticEntitySpec,
    geometry_ref: Mapping[str, Any],
) -> str | None:
    """Resolve an exact auto-vacuum child across an authored host boundary."""
    points = geometry_ref.get("quad_points", ())
    if len(points) != 4:
        raise ValueError(f"{entity.semantic_id} sidewall requires four quad points")
    start = (float(points[0][0]), float(points[0][1]))
    end = (float(points[1][0]), float(points[1][1]))
    z_values = tuple(float(point[2]) for point in points)
    z_min_um, z_max_um = min(z_values), max(z_values)
    matches: list[str] = []
    for solution in _solution_entities(build_input):
        if not bool(solution.metadata.get("is_auto_vacuum_region")):
            continue
        entity_ids = solution.metadata.get("auto_vacuum_subtracting_entity_ids")
        if isinstance(entity_ids, str | bytes) or not isinstance(entity_ids, Sequence):
            raise TypeError(
                f"auto-vacuum component {solution.semantic_id!r} lacks an entity subtractor ledger"
            )
        if entity.semantic_id not in entity_ids:
            continue
        if not (
            _same_z(float(solution.geometry["z_min_um"]), z_min_um)
            and _same_z(float(solution.geometry["z_max_um"]), z_max_um)
            and _sidewall_segment_is_component_boundary(start, end, solution)
        ):
            continue
        matches.append(solution.semantic_id)
    if len(matches) > 1:
        raise ValueError(
            f"{entity.semantic_id} sidewall segment {start!r}->{end!r} has "
            f"ambiguous auto-vacuum adjacency: {matches!r}."
        )
    return matches[0] if matches else None


def _sidewall_segment_is_component_boundary(
    start: tuple[float, float],
    end: tuple[float, float],
    solution: SemanticEntitySpec,
) -> bool:
    """Require a component's canonical outer/hole edges to cover one segment."""
    outer_loop = solution.geometry.get("outer_loop")
    if outer_loop is None:
        raise ValueError(
            f"auto-vacuum component {solution.semantic_id!r} lacks outer_loop."
        )
    intervals = tuple(
        interval
        for loop in (outer_loop, *solution.geometry.get("hole_loops", ()))
        for edge_start, edge_end in _ring_edges(_clean_loop(loop))
        if (interval := _segment_overlap_interval(start, end, edge_start, edge_end))
        is not None
    )
    return not _interval_complement(intervals)


def _auto_vacuum_sidewall_is_exterior_only(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
) -> bool:
    """Whether every unmatched finite sidewall is an exact parent-envelope edge."""
    z_min, z_max = _entity_z_range_um(entity)
    if z_max <= z_min:
        raise ValueError(f"{entity.semantic_id} has non-positive sidewall extent.")
    base_ref = _route_entity_geometry_ref(
        build_input,
        "B",
        entity,
        representation="cutout_boundary_shell",
    )
    sidewalls = _sidewall_geometry_refs(base_ref)
    if not sidewalls:
        raise ValueError(f"{entity.semantic_id} has no sidewall edge references.")
    for sidewall in sidewalls:
        if not _sidewall_is_auto_vacuum_envelope_edge(
            build_input,
            entity=entity,
            geometry_ref=sidewall,
        ):
            return False
    return True


def _sidewall_is_auto_vacuum_envelope_edge(
    build_input: GeometryBuildInput,
    *,
    entity: SemanticEntitySpec,
    geometry_ref: Mapping[str, Any],
) -> bool:
    """Return whether one exact sidewall base edge is on an auto envelope."""
    host_id = entity.host_void_semantic_id
    envelopes = {
        tuple(tuple(float(value) for value in point) for point in loop)
        for solution in _solution_entities(build_input)
        if solution.metadata.get("auto_vacuum_group_id") == host_id
        for loop in (solution.metadata.get("auto_vacuum_envelope_outer_loop"),)
        if isinstance(loop, Sequence) and not isinstance(loop, str | bytes)
    }
    if not envelopes:
        return False
    if len(envelopes) != 1:
        raise ValueError(
            f"{entity.semantic_id} auto-vacuum parent {host_id!r} lacks one envelope loop."
        )
    quad = geometry_ref.get("quad_points")
    if not isinstance(quad, Sequence) or len(quad) != 4:
        raise ValueError(f"{entity.semantic_id} sidewall lacks exact quad geometry.")
    start = (float(quad[0][0]), float(quad[0][1]))
    end = (float(quad[1][0]), float(quad[1][1]))
    envelope = next(iter(envelopes))
    return _undirected_xy_edge(start, end) in {
        _undirected_xy_edge(envelope_start, envelope_end)
        for envelope_start, envelope_end in _ring_edges(envelope)
    }


def _solution_id_for_entity_coverage(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
    *,
    z_um: float,
    mode: str,
    include_auto_vacuum: bool = True,
) -> str | None:
    """Resolve one exact solution cover for a structured conductor footprint.

    This is topology coverage, not a centroid or bounds proxy: every occupied
    polygon point must be inside one candidate solution region at the declared
    z relation. Disjoint covers therefore fail instead of assigning one part
    of a conductor to an arbitrary solution volume.
    """
    import gdstk

    occupied_region = _entity_occupied_region(gdstk, entity)
    if not occupied_region:
        raise ValueError(f"{entity.semantic_id} has no occupied geometry region.")
    candidates: list[str] = []
    for solution in _solution_entities(build_input):
        if not include_auto_vacuum and bool(
            solution.metadata.get("is_auto_vacuum_region")
        ):
            continue
        if mode == "bottom":
            matches = _same_z(float(solution.geometry["z_max_um"]), z_um)
        elif mode == "top":
            matches = _same_z(float(solution.geometry["z_min_um"]), z_um)
        elif mode == "containing":
            matches = (
                float(solution.geometry["z_min_um"])
                < z_um
                < float(solution.geometry["z_max_um"])
            )
        else:
            raise ValueError(f"unsupported solution adjacency mode {mode!r}")
        if not matches:
            continue
        uncovered_region = _boolean_gdstk_region(
            gdstk,
            occupied_region,
            _solution_entity_xy_region(gdstk, solution),
            "not",
        )
        if not uncovered_region:
            candidates.append(solution.semantic_id)
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError(
            f"{entity.semantic_id} {mode} footprint has ambiguous exact solution coverage: {candidates!r}"
        )
    return candidates[0]


def _conductor_solution_interface_kind(
    solution: SemanticEntitySpec,
    *,
    semantic_facts: SemanticEvidenceFacade | None = None,
) -> str:
    material_kind = (
        semantic_facts.material_kind(solution.semantic_id)
        if semantic_facts is not None
        else solution.material_kind
    )
    if not isinstance(material_kind, str):
        raise ValueError(f"{solution.semantic_id} has no snapshotted material kind")
    return conductor_solution_interface_kind(material_kind)


def _solution_interface_planes(
    build_input: GeometryBuildInput,
) -> tuple[
    tuple[SemanticEntitySpec, SemanticEntitySpec, float, Mapping[str, float]],
    ...,
]:
    records: list[
        tuple[SemanticEntitySpec, SemanticEntitySpec, float, Mapping[str, float]]
    ] = []
    solutions = _solution_entities(build_input)
    for index, first in enumerate(solutions):
        for second in solutions[index + 1 :]:
            first_bounds = _solution_bounds(first)
            second_bounds = _solution_bounds(second)
            overlap = _intersect_bounds(first_bounds, second_bounds)
            if overlap is None:
                continue
            if _same_z(
                float(first.geometry["z_max_um"]),
                float(second.geometry["z_min_um"]),
            ):
                records.append(
                    (first, second, float(first.geometry["z_max_um"]), overlap)
                )
            elif _same_z(
                float(second.geometry["z_max_um"]),
                float(first.geometry["z_min_um"]),
            ):
                records.append(
                    (second, first, float(second.geometry["z_max_um"]), overlap)
                )
    return tuple(records)


def _conductor_entities_on_solution_plane(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    z_um: float,
    base_loop: tuple[tuple[float, float], ...] | None = None,
    base_region: tuple[Any, ...] | None = None,
) -> tuple[SemanticEntitySpec, ...]:
    import gdstk

    pair_ids = {lower.semantic_id, upper.semantic_id}
    records: list[SemanticEntitySpec] = []
    if base_loop is not None and base_region is not None:
        raise ValueError("pass at most one of base_loop or base_region")
    if base_loop is None and base_region is None:
        raise ValueError("base_loop or base_region is required")
    patch_region = (
        base_region
        if base_region is not None
        else (gdstk.Polygon(_clean_loop(base_loop)),)
    )
    for entity in build_input.entities:
        if (
            _is_solution_entity(entity)
            or entity.route_representations.get(route) is None
            or "outer_loop" not in entity.geometry
            or not _boolean_gdstk_region(
                gdstk,
                _entity_occupied_region(gdstk, entity),
                patch_region,
                "and",
            )
        ):
            continue
        representation = entity.route_representations.get(route)
        if representation == "surface_sheet":
            # Route-A sheets are removed by exact local patch geometry in
            # `_plan_substrate_air_surfaces`; whole-entity coverage cannot
            # represent a sheet spanning supported and exposed regions.
            if (
                route != "A"
                and set(_route_a_sheet_boundary_volume_ids(build_input, entity))
                == pair_ids
                and _same_z(_route_a_sheet_plane_z_um(build_input, entity), z_um)
            ):
                records.append(entity)
            continue
        entity_region = _entity_occupied_region(gdstk, entity)
        local_overlap = _boolean_gdstk_region(gdstk, entity_region, patch_region, "and")
        if not local_overlap:
            continue
        for face, face_z in zip(
            ("bottom", "top"),
            _entity_z_range_um(entity),
            strict=True,
        ):
            if not _same_z(face_z, z_um):
                continue
            local_domains = _planar_side_solution_regions(
                build_input,
                owner_id=entity.semantic_id,
                occupied_region=local_overlap,
                plane_z_um=face_z,
                side=face,
            )
            if any(domain_id in pair_ids for domain_id, _ in local_domains):
                records.append(entity)
                break
    return tuple(records)


def _solution_exterior_face_geometry_refs(
    build_input: GeometryBuildInput,
    entity: SemanticEntitySpec,
    face: str,
) -> tuple[dict[str, Any], ...]:
    import gdstk

    z_key = "z_max_um" if face == "top" else "z_min_um"
    z_um = float(entity.geometry[z_key])
    base_region = _solution_entity_xy_region(gdstk, entity)
    if not base_region:
        return ()
    subtractor_region: tuple[Any, ...] = ()
    for other in _solution_entities(build_input):
        if other.semantic_id == entity.semantic_id:
            continue
        touches_face = (
            face == "top" and _same_z(float(other.geometry["z_min_um"]), z_um)
        ) or (face == "bottom" and _same_z(float(other.geometry["z_max_um"]), z_um))
        if touches_face:
            candidate = _solution_entity_xy_region(gdstk, other)
        else:
            continue
        if not candidate:
            continue
        if subtractor_region:
            subtractor_region = _boolean_gdstk_region(
                gdstk,
                subtractor_region,
                candidate,
                "or",
            )
        else:
            subtractor_region = candidate
    residual_region = _boolean_gdstk_region(
        gdstk,
        base_region,
        subtractor_region,
        "not",
    )
    if not residual_region:
        return ()
    geometry_refs = _geometry_refs_from_gdstk_region(
        {"plane": {"axis": "z", "value_um": z_um}},
        residual_region,
    )
    return tuple(
        sorted(
            (
                geometry_ref
                for geometry_ref in geometry_refs
                if geometry_ref["outer_loop"]
            ),
            key=lambda geometry_ref: _loop_signature(geometry_ref["outer_loop"]),
        )
    )


def _solution_entities(
    build_input: GeometryBuildInput,
) -> tuple[SemanticEntitySpec, ...]:
    return tuple(
        entity for entity in build_input.entities if _is_solution_entity(entity)
    )


def _solution_bounds(entity: SemanticEntitySpec) -> Mapping[str, float]:
    bounds = entity.geometry.get("domain_bounds_um")
    if not isinstance(bounds, Mapping):
        raise TypeError(f"{entity.semantic_id} requires domain_bounds_um")
    return bounds


def _sidewall_on_solution_outer_boundary(
    geometry_ref: Mapping[str, Any],
    solution: SemanticEntitySpec,
) -> bool:
    points = geometry_ref.get("quad_points", ())
    if len(points) != 4:
        return False
    start = (float(points[0][0]), float(points[0][1]))
    end = (float(points[1][0]), float(points[1][1]))
    bounds = _solution_bounds(solution)
    x_min = float(bounds["x_min_um"])
    x_max = float(bounds["x_max_um"])
    y_min = float(bounds["y_min_um"])
    y_max = float(bounds["y_max_um"])
    if _same_z(start[0], x_min) and _same_z(end[0], x_min):
        return _range_within_bounds(start[1], end[1], y_min, y_max)
    if _same_z(start[0], x_max) and _same_z(end[0], x_max):
        return _range_within_bounds(start[1], end[1], y_min, y_max)
    if _same_z(start[1], y_min) and _same_z(end[1], y_min):
        return _range_within_bounds(start[0], end[0], x_min, x_max)
    if _same_z(start[1], y_max) and _same_z(end[1], y_max):
        return _range_within_bounds(start[0], end[0], x_min, x_max)
    return False


def _required_host_solution_id(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec
) -> str:
    """Return one authored conductor host, never an inferred global vacuum."""
    host_id = entity.host_void_semantic_id
    if not isinstance(host_id, str) or not host_id:
        raise ValueError(f"{entity.semantic_id} requires host_void_semantic_id")
    host = _entity_by_id(build_input, host_id)
    if not _is_solution_entity(host):
        raise ValueError(
            f"{entity.semantic_id} host_void_semantic_id {host_id!r} is not a solution_region"
        )
    return host_id
