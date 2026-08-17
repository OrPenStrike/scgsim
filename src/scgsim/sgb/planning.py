"""Route-first topology planning for semantic geometry v1.

This module owns semantic planning only. It recognizes interface intent,
canonicalizes points/curves/surfaces, plans Route A/B construction cuts, Route C
retained volumes, and physical tag intent before any backend entity tag exists.

The intended compiler stages are:

1. normalize GDS/stack geometry onto one canonical grid, fix ring orientation,
   split multipolygons, and reject self-intersections or slivers;
2. build a 2D planar arrangement by splitting intersections, shared overlaps,
   and T-junctions into atomic points, edges, and cells;
3. sweep those cells through stack z-events to resolve material/domain
   occupancy for each 3D cell;
4. derive `InterfacePlanRecord`s from horizontal and vertical adjacency;
5. emit canonical point, curve, surface, volume, and tag plans.

`interface_intents_2d` can seed or request interface behavior, but it must not
be the only source of solver-relevant interfaces. Actual interface plans come
from topology adjacency.

Route special cases to preserve during implementation:

- Indium bump to ground-plane contact is not a fake ground-volume void. The
  route plan must recognize the XY contact patch before OCC creation, split the
  affected ground face into contact patch plus remainder surface, and make both
  touching volumes/shells reference the same live contact surface when the
  selected route needs retained material or PEC shell geometry.
Route outputs:

- Route A: interface-owned `surface_sheet` conductors for face metal and
  airbridge decks, plus `cutout_boundary_shell` PEC surfaces for bumps/posts.
- Route B: `cutout_boundary_shell` PEC surfaces from construction bodies.
- Route C: retained `material_volume` records assembled from planned shared
  surfaces.

High-count local conductors such as indium bumps and airbridges keep
per-instance topology records, but physical groups are planned at the semantic
family/context level. Instance ids like `_0000` may remain in `SurfacePlanRecord`
or `VolumePlanRecord` source ids; they must not leak into final physical names
when `semantic_group_id` identifies the die-to-die conductor family.

Volume construction contract:

All routes build topology surface-first. The planner must create or reuse every
boundary surface before it emits a backend-live volume. `AIR`, substrate, and
Route C material volumes are then assembled only from `SurfaceRefRecord`s with
`addSurfaceLoop()` followed by `addVolume()`. Domain bounds may be kept for
audit and for planning outer boundary faces, but they must not become direct
box/extrude fallback volumes.

Canonical topology contract:

The compiler, not the Gmsh backend, owns conformal topology identity. Planned
surface candidates must be converted into canonical `PointPlanRecord`s,
`CurvePlanRecord`s, and `SurfaceLoopRecord`s before backend lowering. That
registry is where shared vertices, shared edges,
duplicate surfaces, and volume shell closure become reviewable metadata.
Backend point/line caches may still exist as a lowering optimization, but they
must not be the proof that two surfaces share topology.

Hard invariants for this registry:

- no planned curve may contain another planned point in its interior;
- every MA/MS/MM/SA/SS/AA surface must trace back to `InterfacePlanRecord`;
- live surfaces on the same plane must not have duplicate or overlapping area;
- volume shells must be checkable from planned curve incidence before OCC runs.
- internal shared surfaces must be referenced by both adjacent volumes; exterior
  surfaces must be referenced by exactly one live volume.
"""

from __future__ import annotations

import time
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from math import hypot, isfinite, sqrt
from typing import Any

from scgsim.sgb.models import (
    HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES,
    ConstructionBodyPlanRecord,
    ConstructionPlanRecord,
    CurvePlanRecord,
    CurveRefRecord,
    CutHostOperationRecord,
    GeometryBuildInput,
    InnerPecVoidShellRecord,
    InterfacePlanRecord,
    MMContactRecord,
    PointPlanRecord,
    PortSheetRegionRecord,
    RouteABConstructionPlanRecord,
    RouteABVolumePlanRecord,
    RouteLiteral,
    SemanticEntitySpec,
    SurfaceLoopRecord,
    SurfaceOrientationLiteral,
    SurfacePartitionRecord,
    SurfacePlanRecord,
    SurfaceRefRecord,
    TagPlanRecord,
    VolumePlanRecord,
)
from scgsim.sgb.validation import (
    _is_solution_entity,
    _is_vacuum_solution_entity,
    _resolve_contact_pad_attachment,
    validate_curve_plan_coverage,
    validate_interface_surface_source_of_truth,
    validate_no_surface_overlap,
    validate_route_operation_coverage,
    validate_route_volume_surface_refs,
    validate_selected_route,
    validate_surface_deduplication,
    validate_surface_partition_coverage,
    validate_surface_sheet_interface_coverage,
    validate_surface_use_counts,
    validate_tag_plan_coverage,
    validate_volume_surface_closure,
)

_GEOMETRY_REF_METADATA_KEYS = (
    "plane",
    "contact_plane",
    "footprint",
    "outer_loop",
    "hole_loops",
    "loop_geometry_ref",
    "z_um",
    "thickness_um",
)
_INTERFACE_KIND_ORDER = ("MM", "SS", "AA", "MS", "MA", "SA")
_TOPOLOGY_EPS_UM = 1e-9


def _prepare_auto_vacuum_solution_regions(
    build_input: GeometryBuildInput,
    route: RouteLiteral,
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
    auto_z_min_um = float(auto_bounds["z_min_um"])
    auto_z_max_um = float(auto_bounds["z_max_um"])
    if not (auto_z_max_um > auto_z_min_um):
        raise ValueError("auto VACUUM_REGION z_max_um must exceed z_min_um")

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
    if set(values.keys()) != set(required):
        raise ValueError(
            f"{auto_region_id} metadata vacuum_region_padding_um must define exact six-face keys."
        )
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
            entry["z_min_um"] = float("nan")
            entry["z_max_um"] = float("nan")

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


def build_route_construction_plan(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
) -> ConstructionPlanRecord:
    """Build the route-aware plan consumed by bottom-up OCC construction.

    The plan is the semantic center of v1. Interface recognition, surface
    ownership, partition ownership, and tag ownership are decided before OCC
    geometry exists.

    The backend must be able to build from this record without global
    `occ.fragment()`: surfaces carry loop geometry, volumes reference those
    surfaces, Route A/B cuts are explicit `CutHostOperationRecord`s, and
    `TagPlanRecord`s define the physical names before dim-tags exist.

    Tags are planned before backend construction. Every backend-live
    `SurfacePlanRecord` or `VolumePlanRecord` must either get a `TagPlanRecord`
    or be explicitly marked `construction_only`.
    """
    timings: list[dict[str, Any]] = []
    build_input = _prepare_auto_vacuum_solution_regions(build_input, route=route)
    _timed(
        timings,
        "validate_selected_route",
        lambda: validate_selected_route(build_input, route),
    )
    interfaces = _timed(
        timings,
        "recognize_route_interfaces",
        lambda: recognize_route_interfaces(build_input, route=route),
    )
    interfaces = _timed(
        timings,
        "plan_conductor_contact_patches",
        lambda: plan_conductor_contact_patches(
            build_input,
            route=route,
            interfaces=interfaces,
        ),
    )
    if route in {"A", "B"}:
        interfaces, mm_contacts = _timed(
            timings,
            "plan_mm_contact_records",
            lambda: plan_mm_contact_records(
                build_input,
                route=route,
                interfaces=interfaces,
            ),
        )
    else:
        mm_contacts = ()
    surface_partitions = _timed(
        timings,
        "plan_surface_partitions",
        lambda: plan_surface_partitions(
            build_input,
            route=route,
            interfaces=interfaces,
        ),
    )
    construction_bodies = _timed(
        timings,
        "plan_route_construction_bodies",
        lambda: plan_route_construction_bodies(
            build_input,
            route=route,
            interfaces=interfaces,
        ),
    )
    surfaces = _timed(
        timings,
        "plan_route_surfaces",
        lambda: plan_route_surfaces(
            build_input,
            route=route,
            interfaces=interfaces,
            surface_partitions=surface_partitions,
            construction_bodies=construction_bodies,
            mm_contacts=mm_contacts,
        ),
    )
    surfaces = _timed(
        timings,
        "reconcile_solution_domain_boundaries",
        lambda: _reconcile_solution_domain_boundaries(
            build_input,
            surfaces=surfaces,
        ),
    )
    if route in {"A", "B"} and build_input.port_sheet_regions:
        surfaces = _timed(
            timings,
            "lower_port_sheet_regions",
            lambda: _lower_port_sheet_regions(
                build_input,
                route=route,
                mm_contacts=mm_contacts,
                planned_surfaces=surfaces,
            ),
        )
    _timed(
        timings,
        "validate_surface_sheet_interface_coverage",
        lambda: validate_surface_sheet_interface_coverage(
            build_input,
            route=route,
            surfaces=surfaces,
        ),
    )
    surfaces = _timed(
        timings,
        "merge_solution_sidewall_interfaces",
        lambda: _merge_solution_sidewall_interfaces(
            build_input,
            surfaces=surfaces,
        ),
    )
    interfaces = _timed(
        timings,
        "complete_interface_plan_from_surfaces",
        lambda: complete_interface_plan_from_surfaces(
            interfaces=interfaces,
            surfaces=surfaces,
        ),
    )
    points, curves, surface_loops, surfaces = _timed(
        timings,
        "plan_canonical_topology",
        lambda: plan_canonical_topology(surfaces=surfaces),
    )
    _timed(
        timings,
        "validate_route_b_port_sheet_sidewall_topology",
        lambda: _validate_route_b_port_sheet_sidewall_topology(
            route=route,
            points=points,
            curves=curves,
            surface_loops=surface_loops,
            surfaces=surfaces,
        ),
    )
    _timed(
        timings,
        "validate_curve_plan_coverage",
        lambda: validate_curve_plan_coverage(
            points=points,
            curves=curves,
            surface_loops=surface_loops,
            surfaces=surfaces,
        ),
    )
    _timed(
        timings,
        "validate_surface_deduplication",
        lambda: validate_surface_deduplication(surfaces=surfaces),
    )
    _timed(
        timings,
        "validate_no_surface_overlap",
        lambda: validate_no_surface_overlap(surfaces=surfaces),
    )
    volumes = _timed(
        timings,
        "plan_route_volumes",
        lambda: plan_route_volumes(
            build_input,
            route=route,
            surfaces=surfaces,
            mm_contacts=mm_contacts,
        ),
    )
    _timed(
        timings,
        "validate_route_volume_surface_refs",
        lambda: validate_route_volume_surface_refs(route=route, volumes=volumes),
    )
    _timed(
        timings,
        "validate_volume_surface_closure",
        lambda: validate_volume_surface_closure(
            volumes=volumes,
            surfaces=surfaces,
            surface_loops=surface_loops,
        ),
    )
    _timed(
        timings,
        "validate_surface_use_counts",
        lambda: validate_surface_use_counts(volumes=volumes, surfaces=surfaces),
    )
    cut_operations = _timed(
        timings,
        "plan_cut_host_operations",
        lambda: plan_cut_host_operations(
            route=route,
            construction_bodies=construction_bodies,
        ),
    )
    tags = _timed(
        timings,
        "plan_route_tags",
        lambda: plan_route_tags(route=route, surfaces=surfaces, volumes=volumes),
    )
    _timed(
        timings,
        "validate_route_operation_coverage",
        lambda: validate_route_operation_coverage(
            construction_bodies=construction_bodies,
            cut_operations=cut_operations,
            surfaces=surfaces,
        ),
    )
    _timed(
        timings,
        "validate_surface_partition_coverage",
        lambda: validate_surface_partition_coverage(
            interfaces=interfaces,
            surface_partitions=surface_partitions,
            surfaces=surfaces,
        ),
    )
    _timed(
        timings,
        "validate_interface_surface_source_of_truth",
        lambda: validate_interface_surface_source_of_truth(
            interfaces=interfaces,
            surfaces=surfaces,
        ),
    )
    _timed(
        timings,
        "validate_tag_plan_coverage",
        lambda: validate_tag_plan_coverage(
            surfaces=surfaces,
            volumes=volumes,
            tags=tags,
        ),
    )
    plan_type = (
        RouteABConstructionPlanRecord if route in {"A", "B"} else ConstructionPlanRecord
    )
    plan_kwargs = {
        "route": route,
        "interfaces": interfaces,
        "surface_partitions": surface_partitions,
        "points": points,
        "curves": curves,
        "surface_loops": surface_loops,
        "surfaces": surfaces,
        "volumes": volumes,
        "construction_bodies": construction_bodies,
        "cut_operations": cut_operations,
        "port_sheet_regions": build_input.port_sheet_regions,
        "tags": tags,
        "metadata": {
            "backend_strategy": "surface_plan_first_bottom_up_occ",
            "port_sheet_region_layer": {
                "source": "GeometryBuildInput.port_sheet_regions",
                "backend_live_surface_records": (
                    bool(build_input.port_sheet_regions) and route in {"A", "B"}
                ),
                "allowed_overlap": "palace_lumped_port_sheet_only",
                "lowering_status": (
                    "lowered" if build_input.port_sheet_regions else "none"
                ),
            },
            "timings": timings,
            **(
                {"conductor_components": _component_metadata(mm_contacts)}
                if route in {"A", "B"}
                else {}
            ),
        },
    }
    if route in {"A", "B"}:
        plan_kwargs["mm_contacts"] = mm_contacts
    return plan_type(**plan_kwargs)


def _timed(
    timings: list[dict[str, Any]],
    stage: str,
    fn: Any,
) -> Any:
    started = time.perf_counter()
    try:
        result = fn()
    except Exception:
        timings.append(
            {
                "stage": stage,
                "seconds": round(time.perf_counter() - started, 6),
                "status": "failed",
            }
        )
        raise
    timings.append(
        {
            "stage": stage,
            "seconds": round(time.perf_counter() - started, 6),
            "status": "done",
        }
    )
    return result


def complete_interface_plan_from_surfaces(
    *,
    interfaces: tuple[InterfacePlanRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> tuple[InterfacePlanRecord, ...]:
    """Backfill InterfacePlan records for planned adjacency surfaces."""
    known = {interface.interface_id for interface in interfaces}
    completed = list(interfaces)
    for surface in surfaces:
        if surface.interface_id is None or surface.interface_id in known:
            continue
        kind = surface.interface_id.split("__", 1)[0]
        if kind not in _INTERFACE_KIND_ORDER:
            raise ValueError(f"invalid surface interface id: {surface.interface_id}")
        owners = _surface_owner_ids(surface)
        record_owners = _surface_interface_record_owners(surface, owners)
        if len(record_owners) != 2:
            raise ValueError(
                f"{surface.surface_id} interface needs two owners, got {owners!r}"
            )
        completed.append(
            InterfacePlanRecord(
                interface_id=surface.interface_id,
                kind=kind,  # type: ignore[arg-type]
                owner_semantic_ids=(record_owners[0], record_owners[1]),
                recognition_rule=str(
                    surface.metadata.get(
                        "recognition_rule",
                        "planned_surface_adjacency",
                    )
                ),
                solver_use=surface.solver_use,
                metadata={
                    "generated_from_surface_id": surface.surface_id,
                    **dict(surface.metadata),
                },
            )
        )
        known.add(surface.interface_id)
    return tuple(completed)


def _surface_interface_record_owners(
    surface: SurfacePlanRecord,
    owners: tuple[str, ...],
) -> tuple[str, ...]:
    """Project structured sheet ownership onto the legacy two-owner record."""
    if len(owners) <= 2:
        return owners
    return _structured_surface_boundary_volume_ids(surface, owners)


def _structured_surface_boundary_volume_ids(
    surface: SurfacePlanRecord,
    owners: tuple[str, ...],
) -> tuple[str, str]:
    raw_boundary_ids = surface.metadata.get("boundary_volume_ids", ())
    boundary_ids = (
        ()
        if isinstance(raw_boundary_ids, str)
        else tuple(str(value) for value in raw_boundary_ids)
    )
    if (
        len(boundary_ids) != 2
        or len(set(boundary_ids)) != 2
        or not set(boundary_ids).issubset(owners)
    ):
        raise ValueError(
            f"{surface.surface_id} structured interface requires exactly two "
            "distinct boundary_volume_ids from its owners"
        )
    return (boundary_ids[0], boundary_ids[1])


def plan_canonical_topology(
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
) -> tuple[
    tuple[PointPlanRecord, ...],
    tuple[CurvePlanRecord, ...],
    tuple[SurfaceLoopRecord, ...],
    tuple[SurfacePlanRecord, ...],
]:
    """Canonicalize planned surface boundaries into compiler-owned topology.

    This is the v1 implementation boundary that turns surface geometry into a
    topology registry. It must:

    - collect every outer/hole/quad boundary from planned surfaces;
    - create one `PointPlanRecord` per unique live coordinate;
    - split collinear overlapping edges and T-junctions into shared atomic
      curves;
    - create `CurvePlanRecord`s and ordered `SurfaceLoopRecord`s;
    - assign `outer_loop_ref` / `hole_loop_refs` on each surface;
    - reject parent-plus-child live overlap after surface partitioning;
    - ensure every interface surface is backed by `InterfacePlanRecord`;
    - reject duplicate live surfaces unless they are intentionally merged into
      one surface id before tagging; and
    - make volume closure checkable without asking OCC to discover topology.

    Raw `geometry_ref` lowering is not a v1 conformal-geometry contract; the
    backend consumes the planned point/curve/surface-loop refs produced here.
    """
    point_ids: dict[tuple[float, float, float], str] = {}
    point_coordinates: dict[str, tuple[float, float, float]] = {}
    point_curve_ids: dict[str, set[str]] = {}
    curve_ids: dict[tuple[str, str], str] = {}
    curve_owner_ids: dict[str, set[str]] = {}
    curve_interface_ids: dict[str, set[str]] = {}
    curve_surface_ids: dict[str, set[str]] = {}
    curve_volume_ids: dict[str, set[str]] = {}
    loop_ids: dict[tuple[str, tuple[tuple[str, int], ...]], str] = {}
    loops: dict[str, SurfaceLoopRecord] = {}
    canonical_surfaces: list[SurfacePlanRecord] = []
    surface_specs: list[
        tuple[
            SurfacePlanRecord,
            tuple[tuple[str, int, tuple[tuple[float, float, float], ...]], ...],
        ]
    ] = []

    def point_id(coordinate: tuple[float, float, float]) -> str:
        key = _coordinate_key(coordinate)
        existing = point_ids.get(key)
        if existing is not None:
            return existing
        new_id = f"P__{len(point_ids):06d}"
        point_ids[key] = new_id
        point_coordinates[new_id] = key
        point_curve_ids[new_id] = set()
        return new_id

    for surface in surfaces:
        if surface.construction_only:
            canonical_surfaces.append(surface)
            continue
        specs = _surface_ring3d_specs(surface)
        surface_specs.append((surface, specs))
        for _, _, ring in specs:
            for coordinate in ring:
                point_id(coordinate)

    point_axis_index = _point_axis_index(point_coordinates.values())
    point_line_index = _axis_aligned_point_index(point_coordinates.values())

    def split_edge_points(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], ...]:
        points_on_edge = [
            (parameter, coordinate)
            for coordinate in _segment_candidate_points(
                start,
                end,
                point_axis_index,
                point_line_index,
            )
            for parameter in (_segment_parameter(coordinate, start, end),)
            if parameter is not None
        ]
        return tuple(coordinate for _, coordinate in sorted(points_on_edge))

    def curve_ref(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        surface: SurfacePlanRecord,
    ) -> CurveRefRecord:
        start_id = point_id(start)
        end_id = point_id(end)
        if start_id == end_id:
            raise ValueError(f"{surface.surface_id} has zero-length curve")
        key = tuple(sorted((start_id, end_id)))
        curve_id = curve_ids.get(key)
        if curve_id is None:
            curve_id = f"C__{len(curve_ids):06d}"
            curve_ids[key] = curve_id
        point_curve_ids[start_id].add(curve_id)
        point_curve_ids[end_id].add(curve_id)
        curve_owner_ids.setdefault(curve_id, set()).update(
            str(owner_id) for owner_id in _surface_owner_ids(surface)
        )
        if surface.interface_id is not None:
            curve_interface_ids.setdefault(curve_id, set()).add(surface.interface_id)
        curve_surface_ids.setdefault(curve_id, set()).add(surface.surface_id)
        curve_volume_ids.setdefault(curve_id, set()).update(
            str(volume_id)
            for volume_id in surface.metadata.get("boundary_volume_ids", ())
        )
        return CurveRefRecord(
            curve_id=curve_id,
            orientation=1 if key == (start_id, end_id) else -1,
            role="boundary",
        )

    for surface, specs in surface_specs:
        loop_refs: list[str] = []
        for role, index, ring in specs:
            raw_curve_refs = tuple(
                curve_ref(segment_start, segment_end, surface)
                for start, end in _ring3d_edges(ring)
                for split_points in (split_edge_points(start, end),)
                for segment_start, segment_end in pairwise(split_points)
            )
            curve_refs = _cancel_backtracking_curve_refs(raw_curve_refs)
            if not curve_refs:
                raise ValueError(f"{surface.surface_id} has empty canonical loop")
            loop_key = (role, _ordered_loop_signature(curve_refs))
            loop_id = loop_ids.get(loop_key)
            if loop_id is None:
                suffix = "OUTER" if role == "outer" else f"HOLE_{index:04d}"
                loop_id = f"LOOP__{surface.surface_id}__{suffix}"
                loop_ids[loop_key] = loop_id
                loops[loop_id] = SurfaceLoopRecord(
                    loop_id=loop_id,
                    curve_refs=curve_refs,
                    role=role,
                    surface_id=surface.surface_id,
                )
            loop_refs.append(loop_id)
        if not loop_refs:
            raise ValueError(f"{surface.surface_id} has no planned loops")
        canonical_surfaces.append(
            replace(
                surface,
                outer_loop_ref=loop_refs[0],
                hole_loop_refs=tuple(loop_refs[1:]),
            )
        )

    retained_loop_surface_ids: dict[str, set[str]] = {}
    for surface in canonical_surfaces:
        if surface.construction_only:
            continue
        for loop_id in (surface.outer_loop_ref, *surface.hole_loop_refs):
            if loop_id is not None:
                retained_loop_surface_ids.setdefault(loop_id, set()).add(
                    surface.surface_id
                )
    retained_curve_surface_ids: dict[str, set[str]] = {}
    for loop_id, loop in loops.items():
        for ref in loop.curve_refs:
            retained_curve_surface_ids.setdefault(ref.curve_id, set()).update(
                retained_loop_surface_ids.get(loop_id, ())
            )
    retained_curve_ids = set(retained_curve_surface_ids)
    retained_surfaces = {surface.surface_id: surface for surface in canonical_surfaces}
    curve_owner_ids = {
        curve_id: {
            owner_id
            for surface_id in surface_ids
            for owner_id in _surface_owner_ids(retained_surfaces[surface_id])
        }
        for curve_id, surface_ids in retained_curve_surface_ids.items()
    }
    curve_interface_ids = {
        curve_id: {
            retained_surfaces[surface_id].interface_id
            for surface_id in surface_ids
            if retained_surfaces[surface_id].interface_id is not None
        }
        for curve_id, surface_ids in retained_curve_surface_ids.items()
    }
    curve_volume_ids = {
        curve_id: {
            str(volume_id)
            for surface_id in surface_ids
            for volume_id in retained_surfaces[surface_id].metadata.get(
                "boundary_volume_ids", ()
            )
        }
        for curve_id, surface_ids in retained_curve_surface_ids.items()
    }
    # `curve_ref()` records provisional ownership before Boolean residual
    # cancellation.  Rebuild every public curve claim from the retained loops,
    # then make the invariant explicit so canceled cut lines cannot leak into
    # the canonical ledger.
    for curve_id, surface_ids in retained_curve_surface_ids.items():
        retained = tuple(retained_surfaces[surface_id] for surface_id in surface_ids)
        expected_owners = {
            owner_id for surface in retained for owner_id in _surface_owner_ids(surface)
        }
        expected_interfaces = {
            surface.interface_id
            for surface in retained
            if surface.interface_id is not None
        }
        expected_volumes = {
            str(volume_id)
            for surface in retained
            for volume_id in surface.metadata.get("boundary_volume_ids", ())
        }
        if (
            curve_owner_ids[curve_id] != expected_owners
            or curve_interface_ids[curve_id] != expected_interfaces
            or curve_volume_ids[curve_id] != expected_volumes
        ):
            raise AssertionError(
                f"{curve_id} retains claims absent from its retained surfaces"
            )
    point_curve_ids = {
        point_id_: {
            curve_id for curve_id in curve_ids_ if curve_id in retained_curve_ids
        }
        for point_id_, curve_ids_ in point_curve_ids.items()
    }
    points = tuple(
        PointPlanRecord(
            point_id=point_id_,
            coordinate=point_coordinates[point_id_],
            used_by_curve_ids=tuple(sorted(point_curve_ids[point_id_])),
        )
        for point_id_ in sorted(point_coordinates)
    )
    curves = tuple(
        CurvePlanRecord(
            curve_id=curve_id,
            curve_kind="line_segment",
            start_point_id=start_id,
            end_point_id=end_id,
            owner_semantic_ids=tuple(sorted(curve_owner_ids.get(curve_id, ()))),
            interface_ids=tuple(sorted(curve_interface_ids.get(curve_id, ()))),
            used_by_surface_ids=tuple(
                sorted(retained_curve_surface_ids.get(curve_id, ()))
            ),
            boundary_volume_ids=tuple(sorted(curve_volume_ids.get(curve_id, ()))),
        )
        for (start_id, end_id), curve_id in sorted(
            curve_ids.items(),
            key=lambda item: item[1],
        )
        if curve_id in retained_curve_ids
    )
    return points, curves, tuple(loops.values()), tuple(canonical_surfaces)


def _cancel_backtracking_curve_refs(
    refs: tuple[CurveRefRecord, ...],
) -> tuple[CurveRefRecord, ...]:
    """Remove immediate A→B→A artifacts from Boolean residual splitting."""
    result: list[CurveRefRecord] = []
    for ref in refs:
        if (
            result
            and result[-1].curve_id == ref.curve_id
            and result[-1].orientation == -ref.orientation
        ):
            result.pop()
        else:
            result.append(ref)
    if (
        len(result) > 1
        and result[0].curve_id == result[-1].curve_id
        and result[0].orientation == -result[-1].orientation
    ):
        result = result[1:-1]
    return tuple(result)


def _surface_ring3d_specs(
    surface: SurfacePlanRecord,
) -> tuple[tuple[str, int, tuple[tuple[float, float, float], ...]], ...]:
    geometry_ref = surface.geometry_ref
    if "quad_points" in geometry_ref:
        return (("outer", 0, _clean_ring3d(geometry_ref["quad_points"])),)
    if "outer_loop" not in geometry_ref:
        raise ValueError(f"{surface.surface_id} requires outer_loop or quad_points")
    z_um = _geometry_ref_surface_z_um(geometry_ref)
    outer_loop = _canonical_planar_loop_orientation(
        _clean_loop(geometry_ref["outer_loop"])
    )
    specs = [
        (
            "outer",
            0,
            tuple((x, y, z_um) for x, y in outer_loop),
        )
    ]
    specs.extend(
        (
            "hole",
            index,
            tuple(
                (x, y, z_um)
                for x, y in _canonical_planar_loop_orientation(_clean_loop(hole_loop))
            ),
        )
        for index, hole_loop in enumerate(geometry_ref.get("hole_loops", ()))
    )
    return tuple(specs)


def _canonical_planar_loop_orientation(
    loop: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Use one XY loop direction for OCC plane surfaces and their holes."""
    if _polygon_area(loop) < 0:
        return tuple(reversed(loop))
    return loop


def _geometry_ref_surface_z_um(geometry_ref: Mapping[str, Any]) -> float:
    z_min_um = float(geometry_ref.get("z_min_um", geometry_ref.get("z_um", 0.0)))
    if geometry_ref.get("shell_part") == "top":
        return z_min_um + float(geometry_ref.get("thickness_um", 0.0))
    if geometry_ref.get("shell_part") == "bottom":
        return z_min_um
    plane = geometry_ref.get("plane") or geometry_ref.get("contact_plane")
    if isinstance(plane, Mapping) and plane.get("axis") == "z":
        return float(plane["value_um"])
    return z_min_um


def _clean_ring3d(ring: Any) -> tuple[tuple[float, float, float], ...]:
    points = tuple(
        (float(point[0]), float(point[1]), float(point[2])) for point in ring
    )
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("3D loop requires at least 3 unique points")
    return points


def _ring3d_edges(
    ring: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    return tuple(
        (ring[index], ring[(index + 1) % len(ring)]) for index in range(len(ring))
    )


def _coordinate_key(
    coordinate: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(round(float(value), 9) for value in coordinate)


def _segment_parameter(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float | None:
    vector = tuple(end[index] - start[index] for index in range(3))
    offset = tuple(point[index] - start[index] for index in range(3))
    length_sq = sum(value * value for value in vector)
    if length_sq <= 1e-18:
        return None
    parameter = sum(offset[index] * vector[index] for index in range(3)) / length_sq
    if parameter < -1e-9 or parameter > 1.0 + 1e-9:
        return None
    closest = tuple(start[index] + parameter * vector[index] for index in range(3))
    distance_sq = sum((point[index] - closest[index]) ** 2 for index in range(3))
    if distance_sq > 1e-18:
        return None
    return max(0.0, min(1.0, parameter))


def _point_axis_index(
    coordinates: Sequence[tuple[float, float, float]],
) -> tuple[
    tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]],
    ...,
]:
    indexes: list[tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]]] = []
    for axis in range(3):
        items = sorted((coordinate[axis], coordinate) for coordinate in coordinates)
        indexes.append(
            (
                tuple(value for value, _ in items),
                tuple(coordinate for _, coordinate in items),
            )
        )
    return tuple(indexes)


def _axis_aligned_point_index(
    coordinates: Sequence[tuple[float, float, float]],
) -> dict[
    tuple[int, tuple[float, float]],
    tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]],
]:
    records: dict[
        tuple[int, tuple[float, float]],
        list[tuple[float, tuple[float, float, float]]],
    ] = {}
    for coordinate in coordinates:
        key = _coordinate_key(coordinate)
        for varying_axis in range(3):
            fixed_key = tuple(key[axis] for axis in range(3) if axis != varying_axis)
            records.setdefault((varying_axis, fixed_key), []).append(
                (key[varying_axis], key)
            )
    return {
        line_key: (
            tuple(value for value, _ in sorted_items),
            tuple(coordinate for _, coordinate in sorted_items),
        )
        for line_key, items in records.items()
        for sorted_items in (tuple(sorted(items)),)
    }


def _segment_candidate_points(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    point_axis_index: Sequence[
        tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]]
    ],
    point_line_index: Mapping[
        tuple[int, tuple[float, float]],
        tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]],
    ],
) -> tuple[tuple[float, float, float], ...]:
    start_key = _coordinate_key(start)
    end_key = _coordinate_key(end)
    varying_axes = tuple(
        axis
        for axis in range(3)
        if abs(start_key[axis] - end_key[axis]) > _TOPOLOGY_EPS_UM
    )
    if len(varying_axes) == 1:
        varying_axis = varying_axes[0]
        fixed_key = tuple(start_key[axis] for axis in range(3) if axis != varying_axis)
        values, coordinates = point_line_index.get(
            (varying_axis, fixed_key),
            ((), ()),
        )
        lower = min(start_key[varying_axis], end_key[varying_axis]) - _TOPOLOGY_EPS_UM
        upper = max(start_key[varying_axis], end_key[varying_axis]) + _TOPOLOGY_EPS_UM
        left = bisect_left(values, lower)
        right = bisect_right(values, upper)
        candidates = set(coordinates[left:right])
        candidates.update((start_key, end_key))
        return tuple(sorted(candidates))
    bounds = tuple(
        (
            min(start_key[axis], end_key[axis]) - _TOPOLOGY_EPS_UM,
            max(start_key[axis], end_key[axis]) + _TOPOLOGY_EPS_UM,
        )
        for axis in range(3)
    )
    ranges: list[tuple[int, int, int]] = []
    for axis, (values, _) in enumerate(point_axis_index):
        lower, upper = bounds[axis]
        left = bisect_left(values, lower)
        right = bisect_right(values, upper)
        ranges.append((right - left, left, axis))
    _, left, axis = min(ranges)
    right = left + min(ranges)[0]
    coordinates = point_axis_index[axis][1][left:right]
    candidates = {
        coordinate
        for coordinate in coordinates
        if all(
            bounds[coordinate_axis][0]
            <= coordinate[coordinate_axis]
            <= bounds[coordinate_axis][1]
            for coordinate_axis in range(3)
        )
    }
    candidates.update((start_key, end_key))
    return tuple(sorted(candidates))


def _surface_owner_ids(surface: SurfacePlanRecord) -> tuple[str, ...]:
    owner_ids = surface.metadata.get("owner_semantic_ids", (surface.owner_semantic_id,))
    if isinstance(owner_ids, str):
        return (owner_ids,)
    if isinstance(owner_ids, Sequence):
        return tuple(str(owner_id) for owner_id in owner_ids)
    return (surface.owner_semantic_id,)


def recognize_route_interfaces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
) -> tuple[InterfacePlanRecord, ...]:
    """Recognize interfaces before creating OCC geometry.

    Route-specific recognizers should own each interface rule: draw/ground
    shared edges, XY footprint overlap contacts, stack domain boundaries, ports,
    and exposed EM boundaries. Current fixture metadata may seed those rules
    through `interface_intents_2d`, including generic `interfaces` entries with
    explicit `kind`.
    """
    intents = dict(build_input.metadata.get("interface_intents_2d", {}))
    records: list[InterfacePlanRecord] = []
    for index, intent in enumerate(intents.get("interfaces", ())):
        if not isinstance(intent, Mapping):
            raise TypeError("interfaces entries must be mappings")
        if not _intent_supports_route(intent, route):
            continue
        kind = str(intent.get("kind", ""))
        if kind not in _INTERFACE_KIND_ORDER:
            raise ValueError(f"invalid interface kind: {kind!r}")
        owners = tuple(str(owner) for owner in intent.get("owner_semantic_ids", ()))
        if len(owners) != 2:
            raise ValueError(f"invalid interface owners: {intent!r}")
        records.append(
            InterfacePlanRecord(
                interface_id=str(
                    intent.get("interface_id") or f"{kind}__INTENT__{index:04d}"
                ),
                kind=kind,
                owner_semantic_ids=(owners[0], owners[1]),
                recognition_rule=str(intent.get("recognition_rule", "explicit")),
                source_polygon_ids=tuple(
                    str(value) for value in intent.get("source_polygon_ids", ())
                ),
                metadata=dict(intent),
            )
        )
    for index, intent in enumerate(intents.get("metal_metal_contact_edges", ())):
        if not isinstance(intent, Mapping):
            raise TypeError("metal_metal_contact_edges entries must be mappings")
        if not _intent_supports_route(intent, route):
            continue
        owners = tuple(str(owner) for owner in intent.get("owner_semantic_ids", ()))
        if len(owners) != 2:
            raise ValueError(f"invalid MM edge intent owners: {intent!r}")
        records.append(
            InterfacePlanRecord(
                interface_id=f"MM__CONTACT_EDGE__{index:04d}",
                kind="MM",
                owner_semantic_ids=(owners[0], owners[1]),
                recognition_rule="draw_edge_overlaps_ground_mask_cutout_edge",
                source_polygon_ids=tuple(
                    str(intent[key])
                    for key in ("source_polygon_id", "ground_polygon_id")
                    if key in intent
                ),
                metadata=dict(intent),
            )
        )
    for index, intent in enumerate(intents.get("metal_ground_contact_patches", ())):
        if not isinstance(intent, Mapping):
            raise TypeError("metal_ground_contact_patches entries must be mappings")
        if not _intent_supports_route(intent, route):
            continue
        owners = tuple(str(owner) for owner in intent.get("owner_semantic_ids", ()))
        if len(owners) != 2:
            raise ValueError(f"invalid contact patch intent owners: {intent!r}")
        records.append(
            InterfacePlanRecord(
                interface_id=f"MM__CONTACT_PATCH__{index:04d}",
                kind="MM",
                owner_semantic_ids=(owners[0], owners[1]),
                recognition_rule="projected_xy_footprint_overlap",
                source_polygon_ids=tuple(
                    str(intent[key])
                    for key in ("source_polygon_id", "ground_polygon_id")
                    if key in intent
                ),
                metadata=dict(intent),
            )
        )
    return tuple(records)


def plan_conductor_contact_patches(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
) -> tuple[InterfacePlanRecord, ...]:
    """Recognize coplanar conductor contact patches from planned conductors.

    This is the small v1 contact planner: it partitions each opposite-face
    overlap into connected conductor islands and emits one interface per
    validated contact component.
    """
    if route == "C":
        return _plan_route_c_conductor_contact_patches(
            build_input, route=route, interfaces=interfaces
        )
    import gdstk

    generated: list[InterfacePlanRecord] = []
    seen = {_contact_signature(interface) for interface in interfaces}
    solution_entities = _solution_entities(build_input)
    top_faces: dict[
        float,
        list[tuple[SemanticEntitySpec, Any, Mapping[str, float]]],
    ] = {}
    bottom_faces: dict[
        float,
        list[tuple[SemanticEntitySpec, Any, Mapping[str, float]]],
    ] = {}

    for entity in _active_route_conductor_entities(build_input, route):
        region = _entity_occupied_region(gdstk, entity)
        if not region:
            continue
        bounds = _entity_loop_bounds(entity)
        z_min_um, z_max_um = _entity_z_range_um(entity)
        bottom_faces.setdefault(_z_key(z_min_um), []).append((entity, region, bounds))
        top_faces.setdefault(_z_key(z_max_um), []).append((entity, region, bounds))

    index = 0
    for z_key, lower_faces in top_faces.items():
        for lower, lower_region, lower_bounds in lower_faces:
            for upper, upper_region, upper_bounds in bottom_faces.get(z_key, ()):
                if lower.semantic_id == upper.semantic_id:
                    continue
                if not _bounds_overlap(lower_bounds, upper_bounds):
                    if _loops_touch_without_area(
                        lower.geometry["outer_loop"], upper.geometry["outer_loop"]
                    ):
                        raise ValueError(
                            f"{lower.semantic_id}/{upper.semantic_id} has "
                            "edge/point-only conductor contact"
                        )
                    continue
                overlap_region = _boolean_gdstk_region(
                    gdstk,
                    lower_region,
                    upper_region,
                    "and",
                )
                if not overlap_region:
                    if _loops_touch_without_area(
                        lower.geometry["outer_loop"], upper.geometry["outer_loop"]
                    ):
                        raise ValueError(
                            f"{lower.semantic_id}/{upper.semantic_id} has "
                            "edge/point-only conductor contact"
                        )
                    continue
                contact_loops = _contact_patch_loops(
                    overlap_region,
                    lower.semantic_id,
                    upper.semantic_id,
                )
                # When the upper footprint is wholly the contact patch, keep
                # its authored ring verbatim.  The lower top hole and upper
                # authored sidewall then share canonical curves without
                # replacing either body's sidewall geometry.
                upper_remainder = _boolean_gdstk_region(
                    gdstk, upper_region, overlap_region, "not"
                )
                if not upper_remainder and not upper.geometry.get("hole_loops"):
                    contact_loops = (_clean_loop(upper.geometry["outer_loop"]),)
                for contact_loop in contact_loops:
                    signature = (
                        lower.semantic_id,
                        upper.semantic_id,
                        _z_key(float(z_key)),
                        _loop_signature(contact_loop),
                    )
                    if signature in seen:
                        continue
                    seen.add(signature)
                    metadata = _contact_patch_metadata(
                        build_input,
                        route=route,
                        solution_entities=solution_entities,
                        lower=lower,
                        upper=upper,
                        contact_z_um=float(z_key),
                        contact_loop=contact_loop,
                        contact_index=index,
                    )
                    generated.append(
                        InterfacePlanRecord(
                            interface_id=(
                                f"MM__CONTACT__{lower.semantic_id}__"
                                f"{upper.semantic_id}__{index:04d}"
                            ),
                            kind="MM",
                            owner_semantic_ids=(lower.semantic_id, upper.semantic_id),
                            recognition_rule="coplanar_conductor_contact_patch",
                            source_polygon_ids=(
                                *lower.polygon_ids,
                                *upper.polygon_ids,
                            ),
                            metadata=metadata,
                        )
                    )
                    index += 1
    return (*interfaces, *generated)


def _plan_route_c_conductor_contact_patches(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
) -> tuple[InterfacePlanRecord, ...]:
    """Preserve the pre-A/B Route-C rectangular-contact behavior exactly."""
    import gdstk

    generated: list[InterfacePlanRecord] = []
    seen = {_contact_signature(interface) for interface in interfaces}
    solution_entities = _solution_entities(build_input)
    top_faces: dict[
        float, list[tuple[SemanticEntitySpec, Any, Mapping[str, float]]]
    ] = {}
    bottom_faces: dict[
        float, list[tuple[SemanticEntitySpec, Any, Mapping[str, float]]]
    ] = {}
    for entity in _active_route_conductor_entities(build_input, route):
        region = _entity_occupied_region(gdstk, entity)
        if not region:
            continue
        bounds = _entity_loop_bounds(entity)
        z_min_um, z_max_um = _entity_z_range_um(entity)
        bottom_faces.setdefault(_z_key(z_min_um), []).append((entity, region, bounds))
        top_faces.setdefault(_z_key(z_max_um), []).append((entity, region, bounds))

    index = 0
    for z_key, lower_faces in top_faces.items():
        for lower, lower_region, lower_bounds in lower_faces:
            for upper, upper_region, upper_bounds in bottom_faces.get(z_key, ()):
                if lower.semantic_id == upper.semantic_id or not _bounds_overlap(
                    lower_bounds, upper_bounds
                ):
                    continue
                overlap_region = _boolean_gdstk_region(
                    gdstk, lower_region, upper_region, "and"
                )
                if not overlap_region:
                    continue
                contact_loop = _single_rectangular_contact_loop(
                    overlap_region, lower.semantic_id, upper.semantic_id
                )
                signature = (
                    lower.semantic_id,
                    upper.semantic_id,
                    _z_key(float(z_key)),
                    _loop_signature(contact_loop),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                metadata = _route_c_contact_patch_metadata(
                    build_input,
                    solution_entities=solution_entities,
                    lower=lower,
                    upper=upper,
                    contact_z_um=float(z_key),
                    contact_loop=contact_loop,
                )
                generated.append(
                    InterfacePlanRecord(
                        interface_id=(
                            f"MM__CONTACT__{lower.semantic_id}__"
                            f"{upper.semantic_id}__{index:04d}"
                        ),
                        kind="MM",
                        owner_semantic_ids=(lower.semantic_id, upper.semantic_id),
                        recognition_rule="coplanar_conductor_contact_patch",
                        source_polygon_ids=(*lower.polygon_ids, *upper.polygon_ids),
                        metadata=metadata,
                    )
                )
                index += 1
    return (*interfaces, *generated)


def _route_c_contact_patch_metadata(
    build_input: GeometryBuildInput,
    *,
    solution_entities: Sequence[SemanticEntitySpec],
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    contact_z_um: float,
    contact_loop: tuple[tuple[float, float], ...],
) -> Mapping[str, Any]:
    """Exact pre-A/B Route-C contact metadata shape."""
    del build_input, solution_entities
    return {
        "recognition_rule": "coplanar_conductor_contact_patch",
        "contact_policy": "retained_material_contact",
        "export_surface": True,
        "lower_entity_id": lower.semantic_id,
        "upper_entity_id": upper.semantic_id,
        "lower_face": "top",
        "upper_face": "bottom",
        "contact_z_um": contact_z_um,
        "contact_plane": {"axis": "z", "value_um": contact_z_um},
        "plane": {"axis": "z", "value_um": contact_z_um},
        "outer_loop": contact_loop,
        "hole_loops": (),
        "interface_kinds": ("MM",),
        "surface_owner_semantic_ids": (lower.semantic_id, upper.semantic_id),
        "boundary_volume_ids": (lower.semantic_id, upper.semantic_id),
        "valid_routes": ("C",),
    }


def _loops_touch_without_area(left: Any, right: Any) -> bool:
    """Reject a coplanar contact that has no finite-area footprint."""
    left_loop, right_loop = _clean_loop(left), _clean_loop(right)
    return any(
        _point_on_segment(point, start, end)
        for point in left_loop
        for start, end in _ring_edges(right_loop)
    ) or any(
        _point_on_segment(point, start, end)
        for point in right_loop
        for start, end in _ring_edges(left_loop)
    )


def plan_mm_contact_records(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
) -> tuple[tuple[InterfacePlanRecord, ...], tuple[MMContactRecord, ...]]:
    """Turn finite-area same-net contact into hidden component provenance.

    Gmsh/OCC must never infer this graph from coincident bodies.  The contact
    record remains present for both Route A and Route B while its internal face
    is deliberately absent from solver physical groups.
    """
    entities = {
        entity.semantic_id: entity
        for entity in _active_route_conductor_entities(build_input, route)
    }
    contacts: list[
        tuple[InterfacePlanRecord, SemanticEntitySpec, SemanticEntitySpec]
    ] = []
    for interface in interfaces:
        if interface.recognition_rule != "coplanar_conductor_contact_patch":
            continue
        lower_id, upper_id = interface.owner_semantic_ids
        lower = entities.get(lower_id)
        upper = entities.get(upper_id)
        if lower is None or upper is None:
            raise ValueError(f"{interface.interface_id} references non-live conductors")
        if not lower.net_id or not upper.net_id:
            raise ValueError(f"{interface.interface_id} contact requires resolved nets")
        if lower.net_id != upper.net_id:
            raise ValueError(
                f"{interface.interface_id} shorts different nets "
                f"{lower.net_id!r} and {upper.net_id!r}"
            )
        contacts.append((interface, lower, upper))

    normalizations = _validate_volumetric_conductor_contacts(
        build_input, route=route, entities=entities
    )
    parent = {entity_id: entity_id for entity_id in entities}

    def find(entity_id: str) -> str:
        while parent[entity_id] != entity_id:
            parent[entity_id] = parent[parent[entity_id]]
            entity_id = parent[entity_id]
        return entity_id

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for _, lower, upper in contacts:
        union(lower.semantic_id, upper.semantic_id)
    for lower, upper, _ in normalizations:
        union(lower.semantic_id, upper.semantic_id)

    component_ids = {entity_id: f"COMP__{find(entity_id)}" for entity_id in entities}
    component_values: dict[str, tuple[str | None, str | None]] = {}
    for component_id in set(component_ids.values()):
        members = [
            entity
            for entity_id, entity in entities.items()
            if component_ids[entity_id] == component_id
        ]
        nets = {entity.net_id for entity in members if entity.net_id}
        equipotentials = {
            str(entity.metadata["equipotential_id"])
            for entity in members
            if entity.metadata.get("equipotential_id") is not None
        }
        if len(nets) > 1 or len(equipotentials) > 1:
            raise ValueError(f"{component_id} has conflicting net/equipotential ids")
        component_values[component_id] = (
            next(iter(nets), None),
            next(iter(equipotentials), None),
        )
    records: list[MMContactRecord] = []
    enriched: list[InterfacePlanRecord] = []
    for interface in interfaces:
        if interface.recognition_rule != "coplanar_conductor_contact_patch":
            enriched.append(interface)
            continue
        lower_id, upper_id = interface.owner_semantic_ids
        lower, upper = entities[lower_id], entities[upper_id]
        loop = _clean_loop(interface.metadata["outer_loop"])
        component_id = component_ids[lower_id]
        component_net, component_equipotential = component_values[component_id]
        contact_id = str(interface.metadata.get("contact_id") or interface.interface_id)
        lower_face = str(interface.metadata.get("lower_face", "top"))
        upper_face = str(interface.metadata.get("upper_face", "bottom"))
        records.append(
            MMContactRecord(
                contact_id=contact_id,
                lower_entity_id=lower_id,
                upper_entity_id=upper_id,
                lower_source_face_id=f"{lower_id}__{lower_face}",
                upper_source_face_id=f"{upper_id}__{upper_face}",
                lower_source_fragment_ids=lower.polygon_ids,
                upper_source_fragment_ids=upper.polygon_ids,
                outer_loop=loop,
                area_um2=abs(_polygon_area(loop)),
                normal=(0.0, 0.0, 1.0),
                conductor_component_id=component_id,
                net_id=component_net,
                equipotential_id=component_equipotential,
                layer_provenance={
                    "lower": _source_layer_provenance(lower),
                    "upper": _source_layer_provenance(upper),
                },
                material_provenance={
                    "lower": lower.material_id,
                    "upper": upper.material_id,
                },
                source_provenance={
                    "interface_id": interface.interface_id,
                    "route": route,
                    "recognition_rule": interface.recognition_rule,
                    "source_polygon_ids": interface.source_polygon_ids,
                },
            )
        )
        enriched.append(
            replace(
                interface,
                metadata={
                    **dict(interface.metadata),
                    "conductor_component_id": component_id,
                    "net_id": component_net,
                    "equipotential_id": component_equipotential,
                    "hidden_solver_contact": route in {"A", "B"},
                },
            )
        )
    for lower, upper, loops in normalizations:
        for index, loop in enumerate(loops):
            records.append(
                MMContactRecord(
                    contact_id=(
                        f"NORMALIZED__{lower.semantic_id}__{upper.semantic_id}__"
                        f"{index:04d}"
                    ),
                    lower_entity_id=lower.semantic_id,
                    upper_entity_id=upper.semantic_id,
                    lower_source_face_id=f"{lower.semantic_id}__normalized_overlap",
                    upper_source_face_id=f"{upper.semantic_id}__normalized_overlap",
                    lower_source_fragment_ids=lower.polygon_ids,
                    upper_source_fragment_ids=upper.polygon_ids,
                    outer_loop=loop,
                    area_um2=abs(_polygon_area(loop)),
                    normal=(0.0, 0.0, 1.0),
                    conductor_component_id=component_ids[lower.semantic_id],
                    net_id=component_values[component_ids[lower.semantic_id]][0],
                    equipotential_id=component_values[component_ids[lower.semantic_id]][
                        1
                    ],
                    layer_provenance={
                        "lower": _source_layer_provenance(lower),
                        "upper": _source_layer_provenance(upper),
                    },
                    material_provenance={
                        "lower": lower.material_id,
                        "upper": upper.material_id,
                    },
                    source_provenance={
                        "route": route,
                        "source_polygon_ids": (*lower.polygon_ids, *upper.polygon_ids),
                    },
                )
            )
    return tuple(enriched), tuple(records)


def _source_layer_provenance(entity: SemanticEntitySpec) -> dict[str, Any]:
    """Exact authored source layer identity; no layer/datatype collapsing."""
    return {
        "source_layer_name": entity.metadata.get("source_layer_name"),
        "gds_layer": entity.geometry.get("gds_layer"),
        "gds_datatype": entity.geometry.get("gds_datatype"),
    }


def _validate_volumetric_conductor_contacts(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    entities: Mapping[str, SemanticEntitySpec],
) -> tuple[
    tuple[
        SemanticEntitySpec,
        SemanticEntitySpec,
        tuple[tuple[tuple[float, float], ...], ...],
    ],
    ...,
]:
    """Reject ambiguous body overlap before a route can hide it as a PEC void."""
    import gdstk

    ordered = tuple(sorted(entities.values(), key=lambda entity: entity.semantic_id))
    normalized: list[
        tuple[
            SemanticEntitySpec,
            SemanticEntitySpec,
            tuple[tuple[tuple[float, float], ...], ...],
        ]
    ] = []
    # An authored contact_pad is not merely a label: it must normalize a real,
    # finite same-z overlap with the explicitly attached face-metal entity (or
    # semantic group).  A disconnected pad would otherwise be silently kept
    # as an unrelated PEC body under an attachment provenance claim.
    for pad in ordered:
        if pad.part_role != "contact_pad":
            continue
        _resolve_contact_pad_attachment(pad, ordered)
    for index, lower in enumerate(ordered):
        lower_region = _entity_occupied_region(gdstk, lower)
        lower_min, lower_max = _entity_z_range_um(lower)
        for upper in ordered[index + 1 :]:
            upper_min, upper_max = _entity_z_range_um(upper)
            if (
                min(lower_max, upper_max) - max(lower_min, upper_min)
                <= _TOPOLOGY_EPS_UM
            ):
                continue
            overlap = _boolean_gdstk_region(
                gdstk, lower_region, _entity_occupied_region(gdstk, upper), "and"
            )
            if not overlap:
                continue
            if not lower.net_id or not upper.net_id:
                raise ValueError(
                    f"{lower.semantic_id}/{upper.semantic_id} volumetric overlap "
                    "requires resolved nets"
                )
            if lower.net_id != upper.net_id:
                raise ValueError(
                    f"{lower.semantic_id}/{upper.semantic_id} volumetric overlap "
                    "shorts different nets"
                )
            if not _has_explicit_volumetric_normalization(lower, upper, ordered):
                raise ValueError(
                    f"{lower.semantic_id}/{upper.semantic_id} volumetric overlap "
                    "requires explicit same-net UBM/M1/In normalization provenance"
                )
            normalized.append(
                (
                    lower,
                    upper,
                    _contact_patch_loops(overlap, lower.semantic_id, upper.semantic_id),
                )
            )
    return tuple(normalized)


def _has_explicit_volumetric_normalization(
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    entities: Sequence[SemanticEntitySpec],
) -> bool:
    lower_z_min, lower_z_max = _entity_z_range_um(lower)
    upper_z_min, upper_z_max = _entity_z_range_um(upper)
    if not (_same_z(lower_z_min, upper_z_min) and _same_z(lower_z_max, upper_z_max)):
        return False
    return any(
        pad.part_role == "contact_pad"
        and face.part_role == "face_metal"
        and _resolve_contact_pad_attachment(pad, entities) is face
        for pad, face in ((lower, upper), (upper, lower))
    )


def _is_fully_normalized_face_metal(
    build_input: GeometryBuildInput, entity: SemanticEntitySpec
) -> bool:
    if entity.part_role != "face_metal":
        return False
    import gdstk

    face = _entity_occupied_region(gdstk, entity)
    pads = tuple(
        _entity_occupied_region(gdstk, pad)
        for pad in build_input.entities
        if pad.part_role == "contact_pad"
        and _resolve_contact_pad_attachment(pad, build_input.entities) is entity
    )
    covered = _boolean_gdstk_region(
        gdstk, face, tuple(polygon for region in pads for polygon in region), "not"
    )
    return not covered


def _component_metadata(
    records: tuple[MMContactRecord, ...],
) -> tuple[dict[str, Any], ...]:
    """Derive one complete transitive component ledger from MM contacts."""
    grouped: dict[str, list[MMContactRecord]] = {}
    for record in records:
        grouped.setdefault(record.conductor_component_id, []).append(record)
    return tuple(
        {
            "conductor_component_id": component_id,
            "members": tuple(
                sorted(
                    {
                        entity_id
                        for record in component_records
                        for entity_id in (
                            record.lower_entity_id,
                            record.upper_entity_id,
                        )
                    }
                )
            ),
            "contact_ids": tuple(
                sorted(record.contact_id for record in component_records)
            ),
            "net_id": _one_component_value(
                component_id,
                "net_id",
                (record.net_id for record in component_records),
            ),
            "equipotential_id": _one_component_value(
                component_id,
                "equipotential_id",
                (record.equipotential_id for record in component_records),
            ),
        }
        for component_id, component_records in sorted(grouped.items())
    )


def _one_component_value(
    component_id: str,
    field: str,
    values: Sequence[str | None],
) -> str | None:
    distinct = {value for value in values if value is not None}
    if len(distinct) > 1:
        raise ValueError(f"{component_id} has ambiguous {field}")
    return next(iter(distinct), None)


def plan_surface_partitions(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
) -> tuple[SurfacePartitionRecord, ...]:
    """Plan parent-interface partitions before live surfaces are created.

    This function only consumes explicit `build_input.metadata["surface_partitions"]`
    records.

    Child regions are partition intent, not backend geometry. Each returned
    `SurfacePartitionRecord` must point to a child `SurfacePlanRecord` that
    will be created directly by the backend. The parent interface itself is a
    semantic aggregate and must not be cut by OCC after creation.
    """
    interface_ids = {interface.interface_id for interface in interfaces}
    raw_partitions = build_input.metadata.get("surface_partitions", ())
    if raw_partitions in (None, ()):
        return ()
    if not isinstance(raw_partitions, tuple | list):
        raise TypeError("surface_partitions metadata must be a sequence")

    records: list[SurfacePartitionRecord] = []
    for index, intent in enumerate(raw_partitions):
        if not isinstance(intent, Mapping):
            raise TypeError("surface_partitions entries must be mappings")
        valid_routes = tuple(
            str(value) for value in intent.get("valid_routes", (route,))
        )
        if route not in valid_routes:
            continue
        parent_interface_id = str(intent.get("parent_interface_id", ""))
        if parent_interface_id not in interface_ids:
            raise ValueError(
                f"surface partition references unknown interface "
                f"{parent_interface_id!r}"
            )
        label = str(intent.get("label", "")).strip()
        if not label:
            raise ValueError("surface partition label must be non-empty")
        child_surface_id = str(
            intent.get("child_surface_id") or f"SURF__{parent_interface_id}__{label}"
        )
        records.append(
            SurfacePartitionRecord(
                partition_id=str(
                    intent.get("partition_id")
                    or f"PART__{parent_interface_id}__{index:04d}"
                ),
                parent_interface_id=parent_interface_id,
                child_surface_id=child_surface_id,
                label=label,
                valid_routes=valid_routes,
                metadata=dict(intent),
            )
        )
    return tuple(records)


def plan_route_construction_bodies(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...] = (),
) -> tuple[ConstructionBodyPlanRecord, ...]:
    """Plan Route A/B cutter bodies without making them final geometry.

    Route A only gives construction bodies to conductors represented as
    `cutout_boundary_shell` such as bumps/posts. Route A `surface_sheet`
    conductors are not cutters. Route B gives every `cutout_boundary_shell`
    conductor a construction body. Route C has no construction bodies because
    material volumes survive.
    """
    if route == "C":
        return ()

    contact_faces = _contact_patches_by_entity_face(interfaces)
    records: list[ConstructionBodyPlanRecord] = []
    for entity in build_input.entities:
        if _is_solution_entity(entity):
            continue
        representation = entity.route_representations.get(route)
        if representation != "cutout_boundary_shell":
            continue

        host_id = _required_host_solution_id(build_input, entity)
        records.append(
            ConstructionBodyPlanRecord(
                construction_body_id=f"CBODY__{route}__{entity.semantic_id}",
                owner_semantic_id=entity.semantic_id,
                host_semantic_id=host_id,
                representation=representation,
                geometry_ref=_route_entity_geometry_ref(
                    build_input,
                    route,
                    entity,
                    representation=representation,
                    interfaces=interfaces,
                ),
                expected_surface_ids=(
                    ()
                    if route == "B"
                    and _is_fully_normalized_face_metal(build_input, entity)
                    else _cutout_shell_surface_ids(
                        build_input,
                        route,
                        entity,
                        interfaces=interfaces,
                        contact_faces=contact_faces,
                    )
                ),
                valid_routes=(route,),
            )
        )
    return tuple(records)


def plan_route_surfaces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
    surface_partitions: tuple[SurfacePartitionRecord, ...],
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...],
    mm_contacts: tuple[MMContactRecord, ...] = (),
) -> tuple[SurfacePlanRecord, ...]:
    """Plan route-specific surfaces without building geometry.

    Route A should plan thin sheet interfaces and PEC shell/contact surfaces. Route B
    should plan host-owned cutout shell surfaces. Route C should plan retained
    material top/bottom/sidewall/contact surfaces.

    Partitioned interfaces must already be represented as child live surfaces
    in this stage. The backend should only receive surfaces it can build
    directly from point/curve/loop metadata.

    Route A `surface_sheet` conductors are not standalone surfaces here. They
    must appear as interface-owned `MS`, `MA`, `MM`, or `SA` surfaces, so metal
    coverage replaces the bare substrate-air interface instead of overlapping it.
    """
    partitions_by_interface: dict[str, list[SurfacePartitionRecord]] = {}
    for partition in surface_partitions:
        partitions_by_interface.setdefault(
            partition.parent_interface_id,
            [],
        ).append(partition)

    records: list[SurfacePlanRecord] = []
    records.extend(_plan_substrate_air_surfaces(build_input, route=route))
    contact_faces = _contact_patches_by_entity_face(interfaces)
    sheet_contacts_by_face = (
        _route_a_sheet_contacts_by_face_metal(build_input, mm_contacts)
        if route == "A"
        else {}
    )
    normalized_sheet_loops = _sheet_contact_loops_by_face_metal(
        build_input,
        mm_contacts,
        include_direct_bump_contacts=False,
    )
    for interface in interfaces:
        # Same-net conductor contacts are component provenance only.  Neither
        # Route A nor Route B may lower an internal MM face as solver geometry.
        if _is_hidden_contact_interface(route, interface):
            continue
        geometry_ref = {
            "from_interface_id": interface.interface_id,
            "source_polygon_ids": interface.source_polygon_ids,
            **_geometry_ref_from_metadata(interface.metadata),
        }
        surface_interface_id = _surface_interface_id(
            build_input,
            route=route,
            interface=interface,
        )
        if _is_route_a_sheet_interface(route, interface):
            sheet_entity = _entity_by_id(
                build_input,
                interface.owner_semantic_ids[0],
            )
            geometry_ref = {
                **geometry_ref,
                "plane": {
                    "axis": "z",
                    "value_um": _route_a_sheet_plane_z_um(
                        build_input,
                        sheet_entity,
                    ),
                },
            }
            sheet_contacts = sheet_contacts_by_face.get(sheet_entity.semantic_id, ())
            sheet_contact_loops = tuple(record.outer_loop for record in sheet_contacts)
            if sheet_contacts:
                # Keep the sheet contact footprint as the one live MS cap of
                # the finite PEC void.  Its MM relation remains only in the
                # MMContactRecord; it is never an MM physical group.
                for contact_index, contact in enumerate(sheet_contacts):
                    contact_loop = contact.outer_loop
                    cap_face = _route_a_sheet_contact_cap_face(
                        build_input,
                        sheet_entity=sheet_entity,
                        contact=contact,
                    )
                    cap_solution_id = _conductor_face_adjacent_solution_id(
                        build_input,
                        sheet_entity,
                        cap_face,
                    )
                    cap_owner_ids = (sheet_entity.semantic_id, cap_solution_id)
                    records.append(
                        SurfacePlanRecord(
                            surface_id=(
                                f"SURF__MS__{sheet_entity.semantic_id}__"
                                f"SHEET_CONTACT_CAP__{contact_index:04d}"
                            ),
                            owner_semantic_id=sheet_entity.semantic_id,
                            surface_role="route_a_sheet_contact_cap",
                            geometry_ref={
                                "outer_loop": contact_loop,
                                "hole_loops": (),
                                "plane": geometry_ref["plane"],
                                "representation": "surface_sheet",
                                "source_polygon_ids": _unique_ids(
                                    (
                                        *contact.lower_source_fragment_ids,
                                        *contact.upper_source_fragment_ids,
                                    )
                                ),
                            },
                            interface_id=(
                                f"MS__{sheet_entity.semantic_id}__"
                                f"SHEET_CONTACT_CAP__{contact_index:04d}"
                            ),
                            valid_routes=(route,),
                            metadata={
                                "physical_name": (
                                    f"MS__{_entity_physical_group_id(sheet_entity)}"
                                    "__SHEET_CONTACT_CAP"
                                ),
                                "interface_kinds": ("MS",),
                                "owner_semantic_ids": cap_owner_ids,
                                "physical_owner_semantic_ids": (
                                    _physical_group_owner_ids(
                                        build_input, cap_owner_ids
                                    )
                                ),
                                "boundary_volume_ids": (cap_solution_id,),
                                "exposed_surface_role": "sheet_contact_cap",
                                "sheet_contact_cap": True,
                                "source_contact_id": contact.contact_id,
                                "source_contact_owner_semantic_ids": (
                                    contact.lower_entity_id,
                                    contact.upper_entity_id,
                                ),
                            },
                        )
                    )
                geometry_refs = _subtract_contact_patches_from_face(
                    geometry_ref,
                    sheet_contact_loops,
                )
                if not geometry_refs:
                    continue
                if len(geometry_refs) != 1:
                    raise ValueError(
                        f"{sheet_entity.semantic_id} Route A sheet contact "
                        "split produced multiple sheet remainders"
                    )
                geometry_ref = geometry_refs[0]
        interface_kinds = _interface_surface_kinds(
            build_input,
            route=route,
            interface=interface,
        )
        owner_semantic_ids = _interface_surface_owner_ids(
            build_input,
            route=route,
            interface=interface,
        )
        boundary_volume_ids = _interface_boundary_volume_ids(
            build_input,
            route=route,
            interface=interface,
        )
        physical_owner_semantic_ids = _physical_group_owner_ids(
            build_input,
            owner_semantic_ids,
        )
        embedded_sheet = _is_route_a_sheet_interface(route, interface)
        partitions = partitions_by_interface.get(interface.interface_id, ())
        if partitions:
            parent_surface_id = f"SURF__{surface_interface_id}"
            for partition in partitions:
                child_geometry_ref = {
                    **geometry_ref,
                    "partition_id": partition.partition_id,
                    "parent_interface_id": partition.parent_interface_id,
                    **_geometry_ref_from_metadata(partition.metadata),
                }
                records.append(
                    SurfacePlanRecord(
                        surface_id=partition.child_surface_id,
                        owner_semantic_id=interface.owner_semantic_ids[0],
                        surface_role=f"{route}_planned_interface_partition",
                        geometry_ref=child_geometry_ref,
                        interface_id=surface_interface_id,
                        parent_surface_id=parent_surface_id,
                        partition_label=partition.label,
                        solver_use=interface.solver_use or "solver_active",
                        valid_routes=(route,),
                        metadata={
                            "interface_kinds": interface_kinds,
                            "owner_semantic_ids": owner_semantic_ids,
                            "physical_owner_semantic_ids": physical_owner_semantic_ids,
                            "boundary_volume_ids": boundary_volume_ids,
                            "embedded_surface_sheet": embedded_sheet,
                        },
                    )
                )
            continue
        records.append(
            SurfacePlanRecord(
                surface_id=f"SURF__{surface_interface_id}",
                owner_semantic_id=interface.owner_semantic_ids[0],
                surface_role=f"{route}_planned_interface",
                geometry_ref=geometry_ref,
                interface_id=surface_interface_id,
                solver_use=interface.solver_use or "solver_active",
                valid_routes=(route,),
                metadata={
                    "interface_kinds": interface_kinds,
                    "owner_semantic_ids": owner_semantic_ids,
                    "physical_owner_semantic_ids": physical_owner_semantic_ids,
                    "boundary_volume_ids": boundary_volume_ids,
                    "embedded_surface_sheet": embedded_sheet,
                },
            )
        )

    construction_body_by_surface_id = {
        surface_id: body
        for body in construction_bodies
        for surface_id in body.expected_surface_ids
    }
    normalized_pad_loops = _sheet_contact_loops_by_pad(build_input, mm_contacts)
    for entity in build_input.entities:
        if _is_solution_entity(entity):
            continue
        representation = entity.route_representations.get(route)
        if representation in {"cutout_boundary_shell", "material_volume"}:
            for shell_part in ("top", "bottom"):
                base_surface_id = _conductor_boundary_surface_id(
                    route,
                    entity,
                    representation,
                    shell_part,
                )
                base_geometry_ref = {
                    **_route_entity_geometry_ref(
                        build_input,
                        route,
                        entity,
                        representation=representation,
                        interfaces=interfaces,
                    ),
                    "shell_part": shell_part,
                }
                face_geometry_refs = _subtract_contact_patches_from_face(
                    base_geometry_ref,
                    (
                        *contact_faces.get((entity.semantic_id, shell_part), ()),
                        *(
                            normalized_sheet_loops.get(entity.semantic_id, ())
                            if route == "B" and entity.part_role == "face_metal"
                            else normalized_pad_loops.get(entity.semantic_id, ())
                            if route == "A" and shell_part == "bottom"
                            else ()
                        ),
                    ),
                )
                if not face_geometry_refs:
                    continue
                adjacent_id = _conductor_face_adjacent_solution_id(
                    build_input,
                    entity,
                    shell_part,
                )
                interface_kind = _conductor_solution_interface_kind(
                    _entity_by_id(build_input, adjacent_id)
                )
                for face_index, face_geometry_ref in enumerate(face_geometry_refs):
                    surface_id = (
                        base_surface_id
                        if len(face_geometry_refs) == 1
                        else f"{base_surface_id}__P{face_index:04d}"
                    )
                    body = construction_body_by_surface_id.get(surface_id)
                    if body is None:
                        body = construction_body_by_surface_id.get(base_surface_id)
                    face_owner_ids = (entity.semantic_id, adjacent_id)
                    records.append(
                        SurfacePlanRecord(
                            surface_id=surface_id,
                            owner_semantic_id=entity.semantic_id,
                            surface_role=(
                                "cutout_boundary_shell"
                                if representation == "cutout_boundary_shell"
                                else "material_interface"
                            ),
                            geometry_ref={
                                **face_geometry_ref,
                                "construction_body_id": (
                                    body.construction_body_id
                                    if body is not None
                                    else None
                                ),
                            },
                            interface_id=_conductor_face_interface_id(
                                interface_kind,
                                entity.semantic_id,
                                adjacent_id,
                                shell_part,
                                None if len(face_geometry_refs) == 1 else face_index,
                            ),
                            valid_routes=(route,),
                            solver_use="solver_active",
                            metadata={
                                "interface_kinds": (interface_kind,),
                                "owner_semantic_ids": face_owner_ids,
                                "physical_owner_semantic_ids": (
                                    _physical_group_owner_ids(
                                        build_input,
                                        face_owner_ids,
                                    )
                                ),
                                "boundary_volume_ids": (
                                    _route_conductor_boundary_volume_ids(
                                        build_input,
                                        route=route,
                                        entity=entity,
                                        face=shell_part,
                                        adjacent_solution_id=adjacent_id,
                                    )
                                ),
                                "exposed_surface_role": shell_part,
                            },
                        )
                    )
            sidewall_adjacent_id = (
                _conductor_sidewall_adjacent_solution_id(build_input, entity)
                if route == "C"
                else None
            )
            sidewall_geometry_refs = (
                ()
                if route == "C" and sidewall_adjacent_id is None
                else _conductor_sidewall_geometry_refs(
                    build_input,
                    route=route,
                    entity=entity,
                    representation=representation,
                    adjacent_solution_id=sidewall_adjacent_id,
                    interfaces=interfaces,
                    excluded_footprints=(
                        normalized_sheet_loops.get(entity.semantic_id, ())
                        if route == "B" and entity.part_role == "face_metal"
                        else ()
                    ),
                )
            )
            for edge_index, geometry_ref in enumerate(sidewall_geometry_refs):
                shell_part = f"sidewall_{edge_index:04d}"
                surface_id = _conductor_boundary_surface_id(
                    route,
                    entity,
                    representation,
                    shell_part,
                )
                body = construction_body_by_surface_id.get(surface_id)
                sidewall_adjacent_owner_id = str(
                    geometry_ref.get(
                        "adjacent_conductor_semantic_id",
                        geometry_ref.get("adjacent_solution_id", sidewall_adjacent_id),
                    )
                )
                if (
                    not sidewall_adjacent_owner_id
                    or sidewall_adjacent_owner_id == "None"
                ):
                    raise ValueError(
                        f"{entity.semantic_id} sidewall lacks exact adjacent solution provenance."
                    )
                sidewall_interface_kind = (
                    "MM"
                    if "adjacent_conductor_semantic_id" in geometry_ref
                    else _conductor_solution_interface_kind(
                        _entity_by_id(build_input, sidewall_adjacent_owner_id)
                    )
                )
                sidewall_interface_id = (
                    None
                    if geometry_ref.get("solution_exterior_boundary")
                    else (
                        f"{sidewall_interface_kind}__{entity.semantic_id}__"
                        f"{sidewall_adjacent_owner_id}__"
                        f"{shell_part.upper()}"
                    )
                )
                sidewall_boundary_volume_ids = (
                    (entity.semantic_id,)
                    if geometry_ref.get("solution_exterior_boundary")
                    else _conductor_boundary_volume_ids(
                        route,
                        entity,
                        sidewall_adjacent_owner_id,
                    )
                )
                sidewall_owner_ids = (
                    (entity.semantic_id,)
                    if geometry_ref.get("solution_exterior_boundary")
                    else (entity.semantic_id, sidewall_adjacent_owner_id)
                )
                sidewall_physical_owner_ids = _physical_group_owner_ids(
                    build_input,
                    sidewall_owner_ids,
                )
                sidewall_physical_name = (
                    f"MA__{'__'.join(sidewall_physical_owner_ids)}__SIDEWALL"
                    if entity.part_role == "bump_body"
                    and sidewall_interface_kind == "MA"
                    else None
                )
                records.append(
                    SurfacePlanRecord(
                        surface_id=surface_id,
                        owner_semantic_id=entity.semantic_id,
                        surface_role=(
                            "cutout_boundary_shell"
                            if representation == "cutout_boundary_shell"
                            else "material_interface"
                        ),
                        geometry_ref={
                            **geometry_ref,
                            "from_semantic_id": entity.semantic_id,
                            "geometry_kind": entity.geometry_kind,
                            "part_role": entity.part_role,
                            "representation": representation,
                            "source_polygon_ids": entity.polygon_ids,
                            "shell_part": shell_part,
                            "construction_body_id": (
                                body.construction_body_id if body is not None else None
                            ),
                        },
                        interface_id=sidewall_interface_id,
                        valid_routes=(route,),
                        solver_use="solver_active",
                        metadata={
                            "physical_name": sidewall_physical_name,
                            "interface_kinds": (
                                ()
                                if sidewall_interface_id is None
                                else (sidewall_interface_kind,)
                            ),
                            "owner_semantic_ids": sidewall_owner_ids,
                            "physical_owner_semantic_ids": (
                                sidewall_physical_owner_ids
                            ),
                            "boundary_volume_ids": sidewall_boundary_volume_ids,
                            "exposed_surface_role": shell_part,
                        },
                    )
                )
    if route == "C":
        return tuple(records)
    return _with_surface_contract_metadata(
        build_input,
        route=route,
        interfaces=interfaces,
        mm_contacts=mm_contacts,
        surfaces=tuple(records),
    )


def _lower_port_sheet_regions(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    mm_contacts: tuple[MMContactRecord, ...],
    planned_surfaces: Sequence[SurfacePlanRecord] = (),
) -> tuple[SurfacePlanRecord, ...]:
    """Lower the one accepted Palace port-sheet shape before canonical topology."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    component_by_entity = {
        entity_id: str(component["conductor_component_id"])
        for component in _component_metadata(mm_contacts)
        for entity_id in component["members"]
    }
    records: list[SurfacePlanRecord] = list(planned_surfaces)
    for region in build_input.port_sheet_regions:
        metadata = region.metadata
        source_name = _port_sheet_string(metadata, "source_name", region)
        target_layer = _port_sheet_string(metadata, "target_layer", region)
        port_index = metadata.get("port_index")
        if (
            isinstance(port_index, bool)
            or not isinstance(port_index, int)
            or port_index < 1
        ):
            raise ValueError(f"{region.port_sheet_id} requires 1-based port_index")
        direction = _port_sheet_direction(region)
        raw_direction = _port_sheet_raw_direction(region)
        sign_convention = _port_sheet_string(
            metadata, "direction_sign_convention", region
        )
        if len(region.overlaps) != 2:
            raise ValueError(
                f"{region.port_sheet_id} requires exactly two host overlap records"
            )
        host_ids = _unique_ids(overlap.host_semantic_id for overlap in region.overlaps)
        if len(host_ids) != 2:
            raise ValueError(
                f"{region.port_sheet_id} requires one overlap per distinct "
                "face_metal host"
            )
        hosts = tuple(entities.get(host_id) for host_id in host_ids)
        if any(host is None for host in hosts):
            raise ValueError(f"{region.port_sheet_id} references an unknown port host")
        host_entities = tuple(host for host in hosts if host is not None)
        if any(
            host.part_role != "face_metal"
            or not _port_sheet_target_layer_matches(host, target_layer)
            or host.route_representations.get("A") != "surface_sheet"
            or host.route_representations.get("B") != "cutout_boundary_shell"
            for host in host_entities
        ):
            raise ValueError(
                f"{region.port_sheet_id} hosts must be active A/B face_metal "
                f"on {target_layer}"
            )
        z_values = tuple(
            float(host.geometry.get("z_um", float("nan"))) for host in host_entities
        )
        if not all(isfinite(value) for value in z_values) or not _same_z(
            z_values[0], z_values[1]
        ):
            raise ValueError(
                f"{region.port_sheet_id} hosts require the same finite z_um"
            )
        thickness_values: tuple[float, ...] = ()
        if route == "B":
            thickness_values = tuple(
                float(host.geometry.get("thickness_um", float("nan")))
                for host in host_entities
            )
            if (
                not all(isfinite(value) for value in thickness_values)
                or any(value <= 0.0 for value in thickness_values)
                or not _same_z(thickness_values[0], thickness_values[1])
            ):
                raise ValueError(
                    f"{region.port_sheet_id} Route B hosts require the same "
                    "finite positive thickness_um"
                )
        host_void_ids = tuple(host.host_void_semantic_id for host in host_entities)
        if route == "B" and (
            any(not host_void_id for host_void_id in host_void_ids)
            or len(set(host_void_ids)) != 1
            or host_void_ids[0] not in entities
            or not _is_solution_entity(entities[str(host_void_ids[0])])
        ):
            raise ValueError(
                f"{region.port_sheet_id} hosts require one common host_void solution"
            )
        if any(overlap.host_polygon_id not in polygons for overlap in region.overlaps):
            raise ValueError(
                f"{region.port_sheet_id} references an unknown host polygon"
            )
        remainder = _port_sheet_remainder_geometry(region)
        if not all(
            _port_sheet_boundary_shares_segment(remainder, overlap.overlap_loop)
            for host_id in host_ids
            for overlap in region.overlaps
            if overlap.host_semantic_id == host_id
        ):
            raise ValueError(
                f"{region.port_sheet_id} remainder must share a finite segment "
                "with each host"
            )
        host_polygon_ids = tuple(overlap.host_polygon_id for overlap in region.overlaps)
        owner_provenance = tuple(
            {
                "semantic_id": host.semantic_id,
                "net_id": host.net_id,
                "equipotential_id": host.metadata.get("equipotential_id"),
                "conductor_component_id": component_by_entity.get(
                    host.semantic_id,
                    f"COMP__{host.semantic_id}",
                ),
            }
            for host in host_entities
        )
        source_provenance = {
            "source_name": source_name,
            "port_index": port_index,
            "source_layer": region.source_layer,
            "source_polygon_id": region.source_polygon_id,
            "overlap_ids": tuple(overlap.overlap_id for overlap in region.overlaps),
            "overlaps": tuple(
                {
                    "overlap_id": overlap.overlap_id,
                    "host_semantic_id": overlap.host_semantic_id,
                    "host_polygon_id": overlap.host_polygon_id,
                    "overlap_loop": overlap.overlap_loop,
                }
                for overlap in region.overlaps
            ),
            "host_polygon_ids": host_polygon_ids,
            "route": route,
            "target_layer": target_layer,
            "direction": direction,
            "direction_raw": raw_direction,
            "direction_sign_convention": sign_convention,
        }
        if route == "A":
            plane_z_um, boundary_volume_ids = _route_a_port_sheet_sheet_contract(
                host_ids,
                planned_surfaces,
            )
            embedded_volume_id = _route_a_port_sheet_vacuum_volume_id(
                boundary_volume_ids,
                entities,
            )
        else:
            plane_z_um = z_values[0] + thickness_values[0] / 2.0
            boundary_volume_ids = (host_void_ids[0],)
            embedded_volume_id = str(host_void_ids[0])
        port_surface = SurfacePlanRecord(
            surface_id=f"SURF__LUMPED_PORT__{_safe_port_sheet_name(source_name)}",
            owner_semantic_id=host_ids[0],
            surface_role="lumped_port",
            geometry_ref={
                "outer_loop": remainder["outer_loop"],
                "hole_loops": remainder["hole_loops"],
                "plane": {
                    "axis": "z",
                    "value_um": plane_z_um,
                },
                "representation": "lumped_port_sheet",
            },
            valid_routes=(route,),
            metadata={
                "route": route,
                "representation": "lumped_port_sheet",
                "interface_type": "lumped_port",
                "face_kind": "sheet",
                "owner_semantic_ids": host_ids,
                "physical_owner_semantic_ids": host_ids,
                "boundary_volume_ids": boundary_volume_ids,
                "route_a_boundary_port": route == "A",
                "embedded_surface": True,
                "embedded_volume_id": embedded_volume_id,
                "physical_name": (f"LUMPED_PORT__{_safe_port_sheet_name(source_name)}"),
                "source_provenance": source_provenance,
                "physical_attribute": {
                    "port_index": port_index,
                    "port_name": source_name,
                    "source_layer": region.source_layer,
                    "target_layer": target_layer,
                    "embedded_volume_id": embedded_volume_id,
                    "direction": direction,
                    "owner_semantic_ids": host_ids,
                    "owner_provenance": owner_provenance,
                },
            },
        )
        if route == "A":
            records = _carve_route_a_port_sheet_from_host_plane(
                records,
                port_surface=port_surface,
            )
        elif records:
            records = _partition_route_b_port_sheet_sidewalls(
                records,
                port_surface=port_surface,
                remainder=remainder,
                overlaps_by_host={
                    overlap.host_semantic_id: overlap for overlap in region.overlaps
                },
            )
        records.append(port_surface)
    return tuple(records)


def _partition_route_b_port_sheet_sidewalls(
    surfaces: Sequence[SurfacePlanRecord],
    *,
    port_surface: SurfacePlanRecord,
    remainder: Mapping[str, Any],
    overlaps_by_host: Mapping[str, Any],
) -> list[SurfacePlanRecord]:
    """Split only the two overlap-bound Route-B PEC sidewalls at the port plane."""
    port_z_um = _geometry_ref_surface_z_um(port_surface.geometry_ref)
    terminal_segments_by_host = {
        host_id: tuple(
            (start, end)
            for boundary in (remainder["outer_loop"], *remainder["hole_loops"])
            for start, end in _ring_edges(_clean_loop(boundary))
            if any(
                _segment_overlap_interval(start, end, overlap_start, overlap_end)
                is not None
                for overlap_start, overlap_end in _ring_edges(
                    _clean_loop(overlap.overlap_loop)
                )
            )
        )
        for host_id, overlap in overlaps_by_host.items()
    }
    if any(not segments for segments in terminal_segments_by_host.values()):
        missing = tuple(
            host_id
            for host_id, segments in terminal_segments_by_host.items()
            if not segments
        )
        raise ValueError(
            f"{port_surface.surface_id} has no finite terminal segment for {missing!r}"
        )
    result: list[SurfacePlanRecord] = []
    split_hosts: set[str] = set()
    for surface in surfaces:
        shell_part = str(surface.geometry_ref.get("shell_part", ""))
        if (
            surface.surface_role != "cutout_boundary_shell"
            or surface.owner_semantic_id not in overlaps_by_host
            or not shell_part.startswith("sidewall_")
        ):
            result.append(surface)
            continue
        points = tuple(surface.geometry_ref.get("quad_points", ()))
        if len(points) != 4:
            raise ValueError(
                f"{surface.surface_id} Route B port host sidewall requires a quad"
            )
        quad = tuple(
            (float(point[0]), float(point[1]), float(point[2])) for point in points
        )
        if not any(
            _segment_overlap_interval(quad[0][:2], quad[1][:2], start, end) is not None
            for start, end in terminal_segments_by_host[surface.owner_semantic_id]
        ):
            result.append(surface)
            continue
        z_min_um = min(point[2] for point in quad)
        z_max_um = max(point[2] for point in quad)
        if not z_min_um < z_max_um:
            raise ValueError(
                f"{surface.surface_id} Route B port host sidewall has empty height"
            )
        overlap = overlaps_by_host[surface.owner_semantic_id]
        binding = {
            "port_surface_id": port_surface.surface_id,
            "overlap_id": str(overlap.overlap_id),
            "host_semantic_id": surface.owner_semantic_id,
        }
        if _same_z(port_z_um, z_min_um) or _same_z(port_z_um, z_max_um):
            if not surface.metadata.get("route_b_port_sheet_bindings"):
                raise ValueError(
                    f"{port_surface.surface_id} requires an imprinted "
                    f"sidewall child for {surface.surface_id}"
                )
            result.append(
                replace(
                    surface,
                    metadata=_route_b_port_sheet_partition_metadata(
                        surface.metadata,
                        binding=binding,
                        port_z_um=port_z_um,
                    ),
                )
            )
            split_hosts.add(surface.owner_semantic_id)
            continue
        if not z_min_um < port_z_um < z_max_um:
            raise ValueError(
                f"{port_surface.surface_id} must cut {surface.surface_id} at its "
                "finite sidewall interior"
            )
        lower_quad = (
            quad[0],
            quad[1],
            (quad[2][0], quad[2][1], port_z_um),
            (quad[3][0], quad[3][1], port_z_um),
        )
        upper_quad = (
            (quad[0][0], quad[0][1], port_z_um),
            (quad[1][0], quad[1][1], port_z_um),
            quad[2],
            quad[3],
        )
        parent_surface_id = surface.parent_surface_id or surface.surface_id
        partition_metadata = _route_b_port_sheet_partition_metadata(
            surface.metadata,
            binding=binding,
            port_z_um=port_z_um,
        )
        result.extend(
            (
                replace(
                    surface,
                    surface_id=(
                        f"{surface.surface_id}__{port_surface.surface_id}__LOWER"
                    ),
                    parent_surface_id=parent_surface_id,
                    partition_label="port_lower",
                    geometry_ref={
                        **dict(surface.geometry_ref),
                        "quad_points": lower_quad,
                    },
                    metadata=partition_metadata,
                ),
                replace(
                    surface,
                    surface_id=(
                        f"{surface.surface_id}__{port_surface.surface_id}__UPPER"
                    ),
                    parent_surface_id=parent_surface_id,
                    partition_label="port_upper",
                    geometry_ref={
                        **dict(surface.geometry_ref),
                        "quad_points": upper_quad,
                    },
                    metadata=partition_metadata,
                ),
            )
        )
        split_hosts.add(surface.owner_semantic_id)
    if set(overlaps_by_host) != split_hosts:
        missing = tuple(
            host_id for host_id in overlaps_by_host if host_id not in split_hosts
        )
        raise ValueError(
            f"{port_surface.surface_id} requires finite Route B sidewalls "
            f"for {missing!r}"
        )
    return result


def _route_b_port_sheet_partition_metadata(
    metadata: Mapping[str, Any],
    *,
    binding: Mapping[str, str],
    port_z_um: float,
) -> dict[str, Any]:
    bindings = tuple(
        dict(existing)
        for existing in metadata.get("route_b_port_sheet_bindings", ())
        if isinstance(existing, Mapping)
    )
    if len(bindings) != len(metadata.get("route_b_port_sheet_bindings", ())):
        raise ValueError("Route B port sidewall bindings must be structured records")
    if dict(binding) not in bindings:
        bindings = (*bindings, dict(binding))
    return {
        **dict(metadata),
        "route_b_port_sheet_sidewall_partition": True,
        "route_b_port_sheet_plane_z_um": port_z_um,
        "route_b_port_sheet_bindings": bindings,
    }


def _validate_route_b_port_sheet_sidewall_topology(
    *,
    route: RouteLiteral,
    points: Sequence[PointPlanRecord],
    curves: Sequence[CurvePlanRecord],
    surface_loops: Sequence[SurfaceLoopRecord],
    surfaces: Sequence[SurfacePlanRecord],
) -> None:
    """Require each Route-B port overlap to reuse just its bound PEC curve set."""
    if route != "B":
        return
    loops_by_id = {loop.loop_id: loop for loop in surface_loops}
    surfaces_by_id = {surface.surface_id: surface for surface in surfaces}
    points_by_id = {point.point_id: point.coordinate for point in points}
    curves_by_id = {curve.curve_id: curve for curve in curves}
    for port_surface in (
        surface for surface in surfaces if surface.surface_role == "lumped_port"
    ):
        owners = tuple(
            str(owner) for owner in port_surface.metadata["owner_semantic_ids"]
        )
        overlaps_by_host = {
            str(overlap["host_semantic_id"]): overlap
            for overlap in port_surface.metadata["source_provenance"]["overlaps"]
            if isinstance(overlap, Mapping)
        }
        port_curve_ids = {
            curve_ref.curve_id
            for loop_id in (port_surface.outer_loop_ref, *port_surface.hole_loop_refs)
            for curve_ref in loops_by_id[loop_id].curve_refs
        }
        if not port_curve_ids:
            raise ValueError(
                f"{port_surface.surface_id} has no canonical terminal curves"
            )
        for host_id in owners:
            overlap = overlaps_by_host.get(host_id)
            if overlap is None:
                raise ValueError(
                    f"{port_surface.surface_id} lacks exact overlap provenance "
                    f"for {host_id}"
                )
            expected_terminal_curve_ids = {
                curve_id
                for curve_id in port_curve_ids
                if _route_b_port_terminal_curve_matches_overlap(
                    curves_by_id[curve_id],
                    points_by_id=points_by_id,
                    overlap_loop=overlap["overlap_loop"],
                )
            }
            if not expected_terminal_curve_ids:
                raise ValueError(
                    f"{port_surface.surface_id} has no expected Route B terminal "
                    f"curves for {host_id}"
                )
            credited_terminal_curve_ids = {
                curve.curve_id
                for curve in curves
                if curve.curve_id in port_curve_ids
                and any(
                    candidate.surface_role == "cutout_boundary_shell"
                    and candidate.owner_semantic_id == host_id
                    and str(candidate.geometry_ref.get("shell_part", "")).startswith(
                        "sidewall_"
                    )
                    and candidate.metadata.get("route_b_port_sheet_sidewall_partition")
                    and _route_b_port_sheet_binding_matches(
                        candidate,
                        port_surface_id=port_surface.surface_id,
                        overlap_id=str(overlap["overlap_id"]),
                        host_semantic_id=host_id,
                    )
                    for candidate in (
                        surfaces_by_id[surface_id]
                        for surface_id in curve.used_by_surface_ids
                        if surface_id in surfaces_by_id
                    )
                )
            }
            if credited_terminal_curve_ids != expected_terminal_curve_ids:
                missing = sorted(
                    expected_terminal_curve_ids - credited_terminal_curve_ids
                )
                extra = sorted(
                    credited_terminal_curve_ids - expected_terminal_curve_ids
                )
                raise ValueError(
                    f"{port_surface.surface_id} Route B terminal curves for "
                    f"{host_id} PEC sidewall differ; missing={missing!r}, "
                    f"extra={extra!r}"
                )


def _route_b_port_sheet_binding_matches(
    surface: SurfacePlanRecord,
    *,
    port_surface_id: str,
    overlap_id: str | None,
    host_semantic_id: str,
) -> bool:
    return any(
        isinstance(binding, Mapping)
        and binding.get("port_surface_id") == port_surface_id
        and binding.get("overlap_id") == overlap_id
        and binding.get("host_semantic_id") == host_semantic_id
        for binding in surface.metadata.get("route_b_port_sheet_bindings", ())
    )


def _route_b_port_terminal_curve_matches_overlap(
    curve: CurvePlanRecord,
    *,
    points_by_id: Mapping[str, tuple[float, float, float]],
    overlap_loop: Any,
) -> bool:
    start = points_by_id.get(curve.start_point_id)
    end = points_by_id.get(curve.end_point_id)
    if start is None or end is None:
        return False
    return any(
        _segment_overlap_interval(start[:2], end[:2], overlap_start, overlap_end)
        is not None
        for overlap_start, overlap_end in _ring_edges(_clean_loop(overlap_loop))
    )


def _route_a_port_sheet_sheet_contract(
    host_ids: Sequence[str],
    planned_surfaces: Sequence[SurfacePlanRecord],
) -> tuple[float, tuple[str, str]]:
    """Use the actual Route-A host sheets, never stack-name inference."""
    host_planes: list[float] = []
    boundary_volume_ids: tuple[str, str] | None = None
    for host_id in host_ids:
        host_sheets = tuple(
            surface
            for surface in planned_surfaces
            if surface.metadata.get("representation") == "surface_sheet"
            # Route-A contact caps are per bump/face contact and bound a
            # single solution volume.  A layout junction sheet replaces the
            # actual two-volume face interface, never one of those caps.
            and surface.surface_role == "A_planned_interface"
            and host_id in _surface_owner_ids(surface)
        )
        if len(host_sheets) != 1:
            raise ValueError(
                f"{host_id} requires exactly one planned Route A surface_sheet"
            )
        plane_z_um = _geometry_ref_surface_z_um(host_sheets[0].geometry_ref)
        if not isfinite(plane_z_um):
            raise ValueError(f"{host_id} Route A surface_sheet plane must be finite")
        host_planes.append(plane_z_um)
        raw_boundary_ids = host_sheets[0].metadata.get("boundary_volume_ids")
        if isinstance(raw_boundary_ids, str):
            candidate_boundary_ids = ()
        else:
            candidate_boundary_ids = tuple(str(value) for value in raw_boundary_ids)
        if len(candidate_boundary_ids) != 2 or len(set(candidate_boundary_ids)) != 2:
            raise ValueError(
                f"{host_id} Route A surface_sheet requires two boundary volumes"
            )
        if boundary_volume_ids is None:
            boundary_volume_ids = candidate_boundary_ids
        elif boundary_volume_ids != candidate_boundary_ids:
            raise ValueError(
                "Route A lumped-port hosts require common boundary volumes"
            )
    if not _same_z(host_planes[0], host_planes[1]):
        raise ValueError("Route A lumped-port hosts require one common sheet plane")
    if boundary_volume_ids is None:
        raise ValueError("Route A lumped-port hosts require boundary volumes")
    return host_planes[0], boundary_volume_ids


def _route_a_port_sheet_vacuum_volume_id(
    boundary_volume_ids: Sequence[str],
    entities: Mapping[str, SemanticEntitySpec],
) -> str:
    """Select the typed vacuum member of the actual Route-A sheet adjacency."""
    boundary_entities: list[SemanticEntitySpec] = []
    for volume_id in boundary_volume_ids:
        entity = entities.get(volume_id)
        if entity is None or not _is_solution_entity(entity):
            raise ValueError(
                "Route A lumped-port boundary volumes must be typed solution regions"
            )
        boundary_entities.append(entity)
    vacuum_ids = tuple(
        volume_id
        for volume_id, entity in zip(
            boundary_volume_ids, boundary_entities, strict=True
        )
        if _is_vacuum_solution_entity(entity)
    )
    if len(vacuum_ids) != 1:
        raise ValueError(
            "Route A lumped-port boundary volumes require exactly one vacuum "
            "and one non-vacuum solution region"
        )
    return vacuum_ids[0]


def _carve_route_a_port_sheet_from_host_plane(
    surfaces: Sequence[SurfacePlanRecord],
    *,
    port_surface: SurfacePlanRecord,
) -> list[SurfacePlanRecord]:
    """Replace the co-planar solution face under a Route-A port with its hole."""
    import gdstk

    port_region = _gdstk_surface_region(port_surface.geometry_ref)
    port_z_um = _geometry_ref_surface_z_um(port_surface.geometry_ref)
    result: list[SurfacePlanRecord] = []
    for surface in surfaces:
        if (
            surface.construction_only
            or surface.metadata.get("representation") == "surface_sheet"
            or "outer_loop" not in surface.geometry_ref
            or not _same_z(_geometry_ref_surface_z_um(surface.geometry_ref), port_z_um)
        ):
            result.append(surface)
            continue
        overlap = _boolean_gdstk_region(
            gdstk,
            _gdstk_surface_region(surface.geometry_ref),
            port_region,
            "and",
        )
        if not overlap:
            result.append(surface)
            continue
        remainder = _boolean_gdstk_region(
            gdstk,
            _gdstk_surface_region(surface.geometry_ref),
            port_region,
            "not",
        )
        refs = _geometry_refs_from_gdstk_region(surface.geometry_ref, remainder)
        if len(refs) != 1:
            raise ValueError(
                f"{port_surface.surface_id} requires one host-plane remainder "
                f"for {surface.surface_id}"
            )
        result.append(replace(surface, geometry_ref=refs[0]))
    return result


def _port_sheet_remainder_geometry(region: PortSheetRegionRecord) -> dict[str, Any]:
    import gdstk

    authored = _gdstk_surface_region(
        {"outer_loop": region.exterior, "hole_loops": region.holes}
    )
    overlaps = tuple(
        gdstk.Polygon(_clean_loop(overlap.overlap_loop)) for overlap in region.overlaps
    )
    remainder = _boolean_gdstk_region(gdstk, authored, overlaps, "not")
    refs = _geometry_refs_from_gdstk_region(
        {"outer_loop": region.exterior, "hole_loops": region.holes}, remainder
    )
    if len(refs) != 1:
        raise ValueError(
            f"{region.port_sheet_id} requires exactly one finite active remainder"
        )
    result = refs[0]
    area = abs(_polygon_area(_clean_loop(result["outer_loop"]))) - sum(
        abs(_polygon_area(_clean_loop(hole))) for hole in result["hole_loops"]
    )
    if not isfinite(area) or area <= _TOPOLOGY_EPS_UM:
        raise ValueError(f"{region.port_sheet_id} active remainder must be finite")
    return result


def _port_sheet_boundary_shares_segment(
    remainder: Mapping[str, Any], overlap_loop: Any
) -> bool:
    return any(
        _loops_share_edge_overlap(boundary, overlap_loop)
        for boundary in (remainder["outer_loop"], *remainder["hole_loops"])
    )


def _port_sheet_string(
    metadata: Mapping[str, Any], key: str, region: PortSheetRegionRecord
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{region.port_sheet_id} requires {key}")
    return value


def _port_sheet_direction(region: PortSheetRegionRecord) -> tuple[float, float, float]:
    raw = region.metadata.get("direction")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"{region.port_sheet_id} requires a 3D direction")
    direction = tuple(float(value) for value in raw)
    if (
        not all(isfinite(value) for value in direction)
        or direction[2] != 0.0
        or direction[0] == direction[1] == 0.0
    ):
        raise ValueError(
            f"{region.port_sheet_id} direction must be finite, XY, and nonzero"
        )
    length = sqrt(direction[0] ** 2 + direction[1] ** 2)
    return (direction[0] / length, direction[1] / length, 0.0)


def _port_sheet_raw_direction(
    region: PortSheetRegionRecord,
) -> tuple[float, float, float]:
    raw = region.metadata.get("direction_raw")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"{region.port_sheet_id} requires raw direction provenance")
    direction = tuple(float(value) for value in raw)
    if (
        not all(isfinite(value) for value in direction)
        or direction[2] != 0.0
        or direction[0] == direction[1] == 0.0
    ):
        raise ValueError(
            f"{region.port_sheet_id} raw direction must be finite, XY, and nonzero"
        )
    return direction


def _port_sheet_target_layer_matches(
    entity: SemanticEntitySpec, target_layer: str
) -> bool:
    logical_layer = entity.metadata.get("logical_layer_id")
    return target_layer in {
        str(entity.metadata.get("semantic_group_id", "")),
        logical_layer if isinstance(logical_layer, str) else "",
        f"{entity.geometry.get('gds_layer')}/{entity.geometry.get('gds_datatype')}",
    }


def _safe_port_sheet_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _with_surface_contract_metadata(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interfaces: tuple[InterfacePlanRecord, ...],
    mm_contacts: tuple[MMContactRecord, ...],
    surfaces: tuple[SurfacePlanRecord, ...],
) -> tuple[SurfacePlanRecord, ...]:
    """Attach the explicit fields required by final surface export."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    interface_by_id = {interface.interface_id: interface for interface in interfaces}
    # This is a projection of the completed MMContactRecord ledger, including
    # same-z contact-pad normalizations.  Do not infer a second union graph
    # from exposed surface adjacency.
    component_ledger = _component_metadata(mm_contacts)
    component_by_entity = {
        entity_id: str(component["conductor_component_id"])
        for component in component_ledger
        for entity_id in component["members"]
    }
    component_net = {
        str(component["conductor_component_id"]): component["net_id"]
        for component in component_ledger
    }
    component_equipotential = {
        str(component["conductor_component_id"]): component["equipotential_id"]
        for component in component_ledger
    }

    result: list[SurfacePlanRecord] = []
    for surface in surfaces:
        owner_ids = _surface_owner_ids(surface)
        structured_owner_ids = _surface_physical_owner_ids(surface)
        conductors = tuple(
            entities[owner_id]
            for owner_id in owner_ids
            if owner_id in entities and not _is_solution_entity(entities[owner_id])
        )
        net_ids = {entity.net_id for entity in conductors if entity.net_id}
        if len(net_ids) > 1:
            raise ValueError(f"{surface.surface_id} has ambiguous conductor net")
        component_ids = {
            component_by_entity[entity.semantic_id]
            for entity in conductors
            if entity.semantic_id in component_by_entity
        }
        if conductors and not component_ids:
            component_ids = {f"COMP__{conductors[0].semantic_id}"}
        if len(component_ids) > 1:
            raise ValueError(
                f"{surface.surface_id} spans independent conductor components"
            )
        equipotential_ids = {
            str(entity.metadata["equipotential_id"])
            for entity in conductors
            if entity.metadata.get("equipotential_id") is not None
        }
        if len(equipotential_ids) > 1:
            raise ValueError(f"{surface.surface_id} has ambiguous equipotential id")
        interface = interface_by_id.get(surface.interface_id or "")
        raw_interface_kinds = surface.metadata.get("interface_kinds", ())
        interface_kinds = (
            (str(raw_interface_kinds),)
            if isinstance(raw_interface_kinds, str)
            else tuple(str(kind) for kind in raw_interface_kinds)
        )
        face_kind = str(
            surface.metadata.get(
                "exposed_surface_role",
                surface.metadata.get("boundary_role", "interface"),
            )
        )
        if face_kind.startswith("sidewall_"):
            face_kind = "sidewall"
        source_layer_names = {
            str(entity.metadata["source_layer_name"])
            for entity in conductors
            if entity.metadata.get("source_layer_name") is not None
        }
        if len(source_layer_names) > 1:
            raise ValueError(
                f"{surface.surface_id} has ambiguous conductor source_layer_name"
            )
        result.append(
            replace(
                surface,
                metadata={
                    **dict(surface.metadata),
                    "route": route,
                    "representation": str(
                        surface.geometry_ref.get(
                            "representation",
                            "surface_sheet" if conductors else "solution_surface",
                        )
                    ),
                    "interface_type": "_".join(interface_kinds)
                    if interface_kinds
                    else "boundary",
                    "contact_kind": (
                        str(interface.metadata.get("contact_kind", "MM"))
                        if interface is not None
                        and interface.recognition_rule
                        == "coplanar_conductor_contact_patch"
                        else None
                    ),
                    "face_kind": face_kind,
                    # Physical groups can intentionally aggregate split
                    # entities under a stable semantic group id.  Keep the
                    # exact split owners in source provenance, while the
                    # solver-live owner field names the actual group owner.
                    "owner_semantic_ids": structured_owner_ids,
                    "net_id": (
                        component_net[next(iter(component_ids))]
                        if component_ids and next(iter(component_ids)) in component_net
                        else next(iter(net_ids), None)
                    ),
                    "conductor_component_id": next(iter(component_ids), None),
                    "equipotential_id": (
                        component_equipotential[next(iter(component_ids))]
                        if component_ids
                        and next(iter(component_ids)) in component_equipotential
                        else next(iter(equipotential_ids), None)
                    ),
                    "source_provenance": {
                        "source_polygon_ids": _normalized_source_polygon_ids(
                            build_input,
                            owner_ids,
                            surface.geometry_ref,
                            mm_contacts=mm_contacts,
                        ),
                        "interface_id": surface.interface_id,
                        "route": route,
                        "conductor_source_layer_name": next(
                            iter(source_layer_names), None
                        ),
                    },
                },
            )
        )
    return tuple(result)


def _normalized_source_polygon_ids(
    build_input: GeometryBuildInput,
    owner_ids: Sequence[str],
    geometry_ref: Mapping[str, Any],
    *,
    mm_contacts: Sequence[MMContactRecord],
) -> tuple[str, ...]:
    """Keep only spatially participating M1 contact provenance."""
    ids = list(geometry_ref.get("source_polygon_ids", ()))
    owner_set = set(owner_ids)
    component_id = geometry_ref.get("conductor_component_id")
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    for record in mm_contacts:
        if not _is_face_pad_mm_contact(record, entities):
            continue
        if component_id is not None and record.conductor_component_id != component_id:
            continue
        if not owner_set.intersection((record.lower_entity_id, record.upper_entity_id)):
            continue
        ids.extend(record.lower_source_fragment_ids)
        ids.extend(record.upper_source_fragment_ids)
    return _unique_ids(ids)


def plan_route_volumes(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    surfaces: tuple[SurfacePlanRecord, ...],
    mm_contacts: tuple[MMContactRecord, ...] = (),
) -> tuple[VolumePlanRecord, ...]:
    """Plan volumes only after all boundary surfaces have stable ids.

    The same rule applies to Route A/B/C: solution domains and retained Route C
    conductors are closed by planned surfaces, then lowered through
    `addSurfaceLoop()` and `addVolume()`. This function must not create a
    volume from `domain_bounds_um`, `outer_loop`, or `thickness_um`; those values
    are only audit/planning metadata once surface ids exist.
    """
    entity_ids = {entity.semantic_id for entity in build_input.entities}
    surfaces_by_owner: dict[str, list[SurfacePlanRecord]] = {}
    for surface in surfaces:
        owner_ids = (
            _structured_surface_boundary_volume_ids(
                surface,
                _surface_owner_ids(surface),
            )
            if surface.metadata.get("embedded_surface_sheet")
            else _surface_boundary_volume_ids(
                surface,
                known_entity_ids=entity_ids,
            )
        )
        for owner_id in owner_ids:
            surfaces_by_owner.setdefault(str(owner_id), []).append(surface)

    records: list[VolumePlanRecord] = []
    for entity in build_input.entities:
        if _is_solution_entity(entity):
            physical_owner_semantic_id = (
                "VACUUM_REGION"
                if bool(entity.metadata.get("is_auto_vacuum_region"))
                and entity.material_kind == "vacuum"
                else None
            )
            all_refs = tuple(
                SurfaceRefRecord(
                    surface_id=surface.surface_id,
                    orientation=_surface_orientation_for_volume(surface, entity),
                    role="planned_boundary",
                )
                for surface in surfaces_by_owner.get(entity.semantic_id, ())
            )
            void_surfaces: dict[str, list[SurfaceRefRecord]] = {}
            component_members: dict[str, set[str]] = {}
            contact_ids_by_component: dict[str, list[str]] = {}
            for contact in mm_contacts:
                component_members.setdefault(
                    contact.conductor_component_id, set()
                ).update((contact.lower_entity_id, contact.upper_entity_id))
                contact_ids_by_component.setdefault(
                    contact.conductor_component_id, []
                ).append(contact.contact_id)
            boundary_components = {
                str(surface.metadata["conductor_component_id"])
                for surface in surfaces_by_owner.get(entity.semantic_id, ())
                if surface.metadata.get("conductor_component_id") is not None
                and _component_is_boundary_attached(
                    build_input, solution=entity, surface=surface
                )
            }
            exterior_refs: list[SurfaceRefRecord] = []
            for surface, ref in zip(
                surfaces_by_owner.get(entity.semantic_id, ()), all_refs, strict=True
            ):
                is_sheet_cap = (
                    route == "A"
                    and bool(surface.metadata.get("sheet_contact_cap"))
                    and _is_vacuum_solution_entity(entity)
                )
                component_id = str(
                    surface.metadata.get(
                        "conductor_component_id",
                        f"COMP__{surface.owner_semantic_id}",
                    )
                )
                is_component_boundary = component_id in boundary_components
                if (
                    surface.surface_role == "cutout_boundary_shell" or is_sheet_cap
                ) and not is_component_boundary:
                    component_members.setdefault(component_id, set()).add(
                        surface.owner_semantic_id
                    )
                    void_surfaces.setdefault(component_id, []).append(
                        replace(
                            ref,
                            orientation=_inner_void_orientation(
                                build_input,
                                surface=surface,
                                component_members=component_members[component_id],
                            ),
                        )
                    )
                else:
                    exterior_refs.append(ref)
            void_shells = tuple(
                InnerPecVoidShellRecord(
                    shell_id=f"VOID__{entity.semantic_id}__{component_id}",
                    surface_refs=tuple(refs),
                    conductor_component_id=component_id,
                    owner_semantic_ids=tuple(sorted(component_members[component_id])),
                )
                for component_id, refs in void_surfaces.items()
            )
            if route in {"A", "B"}:
                records.append(
                    RouteABVolumePlanRecord(
                        volume_id=f"VOL__{entity.semantic_id}",
                        owner_semantic_id=entity.semantic_id,
                        material_id=entity.material_id,
                        surface_refs=all_refs,
                        exterior_surface_refs=tuple(exterior_refs),
                        inner_pec_void_shells=void_shells,
                        valid_routes=(route,),
                        metadata={
                            "representation": "solution_volume",
                            "material_kind": entity.material_kind,
                            "geometry_ref": dict(entity.geometry),
                            "physical_owner_semantic_id": physical_owner_semantic_id,
                            "inner_pec_void_shell_contact_ids": {
                                component_id: tuple(
                                    contact_ids_by_component.get(component_id, ())
                                )
                                for component_id in void_surfaces
                            },
                        },
                    )
                )
            else:
                records.append(
                    VolumePlanRecord(
                        volume_id=f"VOL__{entity.semantic_id}",
                        owner_semantic_id=entity.semantic_id,
                        material_id=entity.material_id,
                        surface_refs=all_refs,
                        valid_routes=(route,),
                        metadata={
                            "representation": "solution_volume",
                            "material_kind": entity.material_kind,
                            "geometry_ref": dict(entity.geometry),
                            "physical_owner_semantic_id": physical_owner_semantic_id,
                        },
                    )
                )
            continue
        representation = entity.route_representations.get(route)
        if route == "C" and representation == "material_volume":
            physical_owner_semantic_id = _entity_physical_group_id(entity)
            records.append(
                VolumePlanRecord(
                    volume_id=f"VOL__{entity.semantic_id}",
                    owner_semantic_id=entity.semantic_id,
                    material_id=entity.material_id,
                    surface_refs=tuple(
                        SurfaceRefRecord(
                            surface_id=surface.surface_id,
                            orientation=_surface_orientation_for_volume(
                                surface,
                                entity,
                            ),
                            role="planned_boundary",
                        )
                        for surface in surfaces_by_owner.get(
                            entity.semantic_id,
                            (),
                        )
                    ),
                    valid_routes=(route,),
                    metadata={
                        "representation": representation,
                        "material_kind": entity.material_kind,
                        "geometry_ref": _entity_geometry_ref(
                            entity,
                            representation=representation,
                        ),
                        "physical_owner_semantic_id": physical_owner_semantic_id,
                    },
                )
            )
    return tuple(records)


def _inner_void_orientation(
    build_input: GeometryBuildInput,
    *,
    surface: SurfacePlanRecord,
    component_members: set[str],
) -> SurfaceOrientationLiteral:
    """Orient from conductor center, then reverse for a solution inner shell."""
    centers = tuple(
        _entity_volume_center_um(_entity_by_id(build_input, member))
        for member in component_members
    )
    component_center = tuple(
        sum(center[index] for center in centers) / len(centers) for index in range(3)
    )
    normal = _surface_normal_vector(surface)
    centroid = _surface_centroid(surface)
    dot = sum(
        normal[index] * (centroid[index] - component_center[index])
        for index in range(3)
    )
    if abs(dot) <= 1e-9:
        # A symmetric perforated conductor can put a sidewall centroid exactly
        # on the component center.  Classify the normal locally against the
        # component's occupied planar region instead of inventing a global
        # orientation convention for that degenerate center-vector case.
        normal_xy_length = hypot(normal[0], normal[1])
        if normal_xy_length > _TOPOLOGY_EPS_UM:
            import gdstk

            regions = tuple(
                region
                for member in component_members
                for region in _entity_occupied_region(
                    gdstk,
                    _entity_by_id(build_input, member),
                )
            )
            if regions:
                epsilon = max(1e-6, normal_xy_length * 1e-6)
                plus = (
                    centroid[0] + epsilon * normal[0] / normal_xy_length,
                    centroid[1] + epsilon * normal[1] / normal_xy_length,
                )
                minus = (
                    centroid[0] - epsilon * normal[0] / normal_xy_length,
                    centroid[1] - epsilon * normal[1] / normal_xy_length,
                )
                plus_inside, minus_inside = gdstk.inside((plus, minus), regions)
                if plus_inside != minus_inside:
                    # A forward normal points out of the conductor precisely
                    # when its positive local probe is outside the component.
                    return "reversed" if not plus_inside else "forward"
        raise ValueError(
            f"{surface.surface_id} has ambiguous component void orientation"
        )
    return "reversed" if dot > 0 else "forward"


def plan_cut_host_operations(
    *,
    route: RouteLiteral,
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...],
) -> tuple[CutHostOperationRecord, ...]:
    """Group Route A/B construction bodies into host-cut operation plans.

    This is semantic grouping and provenance in the current v1 backend. Exposed
    shell surfaces are already planned as `SurfacePlanRecord`s; these records
    explain which construction bodies belong to each host exclusion policy
    without asking the backend to discover new surfaces through boolean cuts.
    """
    if route == "C":
        return ()

    bodies_by_host: dict[str, list[ConstructionBodyPlanRecord]] = {}
    for body in construction_bodies:
        bodies_by_host.setdefault(body.host_semantic_id, []).append(body)

    return tuple(
        CutHostOperationRecord(
            operation_id=f"CUT__{route}__{host_id}",
            host_semantic_id=host_id,
            construction_body_ids=tuple(body.construction_body_id for body in bodies),
            exposed_surface_ids=tuple(
                surface_id
                for body in bodies
                for surface_id in body.expected_surface_ids
            ),
            valid_routes=(route,),
        )
        for host_id, bodies in bodies_by_host.items()
    )


def plan_route_tags(
    *,
    route: RouteLiteral,
    surfaces: tuple[SurfacePlanRecord, ...],
    volumes: tuple[VolumePlanRecord, ...],
) -> tuple[TagPlanRecord, ...]:
    """Plan physical names before backend entity tags exist.

    Interface surfaces and domain exterior boundary surfaces are exported as
    solver-visible surface physical groups. Other construction-only surfaces
    remain backend topology.
    """
    tags: list[TagPlanRecord] = []
    tags.extend(
        TagPlanRecord(
            physical_name=_volume_physical_name(volume),
            dimension=3,
            source_record_kind="volume",
            source_record_id=volume.volume_id,
            role="material_volume",
        )
        for volume in volumes
        if not volume.construction_only
    )
    tags.extend(
        TagPlanRecord(
            physical_name=_surface_physical_name(surface),
            dimension=2,
            source_record_kind="surface",
            source_record_id=surface.surface_id,
            role=surface.surface_role,
            solver_use=surface.solver_use,
        )
        for surface in surfaces
        if not surface.construction_only
        and not surface.metadata.get("hidden_solver_contact")
        and (
            surface.interface_id is not None
            or surface.surface_role == "domain_boundary"
            or surface.surface_role == "lumped_port"
        )
    )
    return tuple(tags)


def _geometry_ref_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key] for key in _GEOMETRY_REF_METADATA_KEYS if key in metadata
    }


def _intent_supports_route(intent: Mapping[str, Any], route: RouteLiteral) -> bool:
    valid_routes = intent.get("valid_routes")
    if valid_routes is None:
        return True
    if isinstance(valid_routes, str):
        return route == valid_routes
    return route in {str(value) for value in valid_routes}


def _contact_signature(interface: InterfacePlanRecord) -> tuple[Any, ...]:
    if interface.recognition_rule != "coplanar_conductor_contact_patch":
        return ()
    return (
        str(interface.metadata.get("lower_entity_id", "")),
        str(interface.metadata.get("upper_entity_id", "")),
        _z_key(float(interface.metadata.get("contact_z_um", 0.0))),
        _loop_signature(interface.metadata.get("outer_loop", ())),
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


def _bounds_overlap(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> bool:
    return not (
        float(first["x_max_um"]) <= float(second["x_min_um"]) + _TOPOLOGY_EPS_UM
        or float(second["x_max_um"]) <= float(first["x_min_um"]) + _TOPOLOGY_EPS_UM
        or float(first["y_max_um"]) <= float(second["y_min_um"]) + _TOPOLOGY_EPS_UM
        or float(second["y_max_um"]) <= float(first["y_min_um"]) + _TOPOLOGY_EPS_UM
    )


def _z_key(value_um: float) -> float:
    return round(float(value_um), 9)


def _contact_patch_loops(
    overlap_region: Sequence[Any],
    lower_id: str,
    upper_id: str,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    import gdstk

    polygons = _boolean_gdstk_region(gdstk, overlap_region, (), "or")
    if not polygons:
        raise ValueError(
            f"{lower_id} and {upper_id} contact must produce at least one "
            f"polygon, got {len(polygons)}"
        )
    loops: list[tuple[tuple[float, float], ...]] = []
    for polygon in polygons:
        loop = _clean_loop(polygon.points)
        loops.append(loop)
    return _sort_contact_loops(tuple(loops))


def _single_rectangular_contact_loop(
    overlap_region: Sequence[Any],
    lower_id: str,
    upper_id: str,
) -> tuple[tuple[float, float], ...]:
    """Route-C legacy contact recognizer: exactly one rectangular patch."""
    import gdstk

    polygons = _boolean_gdstk_region(gdstk, overlap_region, (), "or")
    if len(polygons) != 1:
        raise ValueError(
            f"{lower_id} and {upper_id} contact must produce one polygon, got "
            f"{len(polygons)}"
        )
    loop = _clean_loop(polygons[0].points)
    if not _is_axis_aligned_rectangle(loop):
        raise ValueError(
            f"{lower_id} and {upper_id} contact is not a rectangular patch"
        )
    return loop


def _is_axis_aligned_rectangle(loop: tuple[tuple[float, float], ...]) -> bool:
    return len(loop) == 4 and all(
        _same_z(start[0], end[0]) or _same_z(start[1], end[1])
        for start, end in _ring_edges(loop)
    )


def _sort_contact_loops(
    loops: tuple[tuple[tuple[float, float], ...], ...],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    def _stable_loop_key(loop: tuple[tuple[float, float], ...]) -> tuple:
        clean = _clean_loop(loop)
        xs = tuple(point[0] for point in clean)
        ys = tuple(point[1] for point in clean)
        canonical = _canonical_loop_sort_key(clean)
        return (
            min(xs),
            max(xs),
            min(ys),
            max(ys),
            round(abs(_polygon_area(clean)), 12),
            canonical,
        )

    return tuple(
        sorted(
            {_loop_signature(loop): loop for loop in loops}.values(),
            key=_stable_loop_key,
        )
    )


def _canonical_loop_sort_key(loop: tuple[tuple[float, float], ...]) -> tuple:
    clean = _clean_loop(loop)
    candidates: list[tuple[tuple[float, float], ...]] = []
    reversed_loop = tuple(reversed(clean))
    for candidate in (clean, reversed_loop):
        for offset in range(len(candidate)):
            rotated = candidate[offset:] + candidate[:offset]
            candidates.append(rotated)
    return min(candidates)


def _loop_signature(loop: Any) -> tuple[tuple[float, float], ...]:
    try:
        points = _clean_loop(loop)
    except Exception:  # noqa: BLE001 - malformed optional loop has no signature
        return ()
    return tuple(sorted(_coordinate_2d_key(point) for point in points))


def _contact_patch_metadata(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    solution_entities: Sequence[SemanticEntitySpec],
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
    contact_z_um: float,
    contact_loop: tuple[tuple[float, float], ...],
    contact_index: int,
) -> Mapping[str, Any]:
    plane_z_um = contact_z_um
    boundary_volume_ids: tuple[str, ...]
    interface_kinds: tuple[str, ...] = ("MM",)
    export_surface = route != "B"

    if route == "A":
        sheet = next(
            (
                entity
                for entity in (lower, upper)
                if entity.route_representations.get(route) == "surface_sheet"
            ),
            None,
        )
        if sheet is not None:
            plane_z_um = _route_a_sheet_plane_z_um_from_solutions(
                sheet,
                solution_entities,
            )
            boundary_volume_ids = _route_a_sheet_boundary_volume_ids_from_solutions(
                sheet,
                solution_entities,
            )
            interface_kinds = ("MM", "MS")
        else:
            boundary_volume_ids = ()
            export_surface = False
    elif route == "C":
        boundary_volume_ids = (lower.semantic_id, upper.semantic_id)
    else:
        boundary_volume_ids = _unique_ids(
            (
                _required_host_solution_id(build_input, lower),
                _required_host_solution_id(build_input, upper),
            )
        )

    contact_policy = {
        "A": "sheet_contact_patch",
        "B": "hidden_cutout_contact",
        "C": "retained_material_contact",
    }[route]
    interface_type = "_".join(interface_kinds)
    return {
        "recognition_rule": "coplanar_conductor_contact_patch",
        "contact_policy": contact_policy,
        "export_surface": export_surface,
        "contact_kind": "MM",
        "contact_index": contact_index,
        "lower_entity_id": lower.semantic_id,
        "upper_entity_id": upper.semantic_id,
        "lower_face": "top",
        "upper_face": "bottom",
        "contact_z_um": contact_z_um,
        "face_kinds": ("top", "bottom"),
        "route": route,
        "contact_representation": "circuit_contact_patch",
        "contact_id": (
            f"{lower.semantic_id}__{upper.semantic_id}__"
            f"{contact_policy}__{contact_index:04d}"
        ),
        "contact_plane": {"axis": "z", "value_um": plane_z_um},
        "plane": {"axis": "z", "value_um": plane_z_um},
        "outer_loop": contact_loop,
        "hole_loops": (),
        "interface_type": interface_type,
        "interface_kinds": interface_kinds,
        "surface_owner_semantic_ids": (lower.semantic_id, upper.semantic_id),
        "boundary_volume_ids": boundary_volume_ids,
        "conductor_component_id": f"{lower.semantic_id}__{upper.semantic_id}",
        "valid_routes": (route,),
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


def _contact_patches_by_entity_face(
    interfaces: tuple[InterfacePlanRecord, ...],
) -> dict[tuple[str, str], tuple[tuple[tuple[float, float], ...], ...]]:
    records: dict[tuple[str, str], list[tuple[tuple[float, float], ...]]] = {}
    for interface in interfaces:
        if interface.recognition_rule != "coplanar_conductor_contact_patch":
            continue
        loop = _clean_loop(interface.metadata["outer_loop"])
        lower_id = str(interface.metadata["lower_entity_id"])
        upper_id = str(interface.metadata["upper_entity_id"])
        lower_face = str(interface.metadata.get("lower_face", "top"))
        upper_face = str(interface.metadata.get("upper_face", "bottom"))
        records.setdefault((lower_id, lower_face), []).append(loop)
        records.setdefault((upper_id, upper_face), []).append(loop)
    return {key: tuple(value) for key, value in records.items()}


def _sheet_contact_loops_by_pad(
    build_input: GeometryBuildInput,
    records: Sequence[MMContactRecord],
) -> dict[str, tuple[tuple[tuple[float, float], ...], ...]]:
    """Return finite MM ledger loops for the contact-pad side of an M1 join."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    result: dict[str, list[tuple[tuple[float, float], ...]]] = {}
    for record in records:
        if not _is_face_pad_mm_contact(record, entities):
            continue
        lower = entities[record.lower_entity_id]
        upper = entities[record.upper_entity_id]
        pad = lower if lower.part_role == "contact_pad" else upper
        result.setdefault(pad.semantic_id, []).append(record.outer_loop)
    return {entity_id: tuple(loops) for entity_id, loops in result.items()}


def _is_face_pad_mm_contact(
    record: MMContactRecord,
    entities: Mapping[str, SemanticEntitySpec],
) -> bool:
    return {
        entities[record.lower_entity_id].part_role,
        entities[record.upper_entity_id].part_role,
    } == {"contact_pad", "face_metal"}


def _sheet_contact_loops_by_face_metal(
    build_input: GeometryBuildInput,
    records: Sequence[MMContactRecord],
    *,
    include_direct_bump_contacts: bool,
) -> dict[str, tuple[tuple[tuple[float, float], ...], ...]]:
    """Return finite M1 contact loops that require Route-A sheet cap topology."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    result: dict[str, list[tuple[tuple[float, float], ...]]] = {}
    for record in records:
        lower = entities[record.lower_entity_id]
        upper = entities[record.upper_entity_id]
        roles = {lower.part_role, upper.part_role}
        if "face_metal" not in roles or (
            roles != {"face_metal", "contact_pad"}
            and not (
                include_direct_bump_contacts and roles == {"face_metal", "bump_body"}
            )
        ):
            continue
        face = lower if lower.part_role == "face_metal" else upper
        result.setdefault(face.semantic_id, []).append(record.outer_loop)
    return {entity_id: tuple(loops) for entity_id, loops in result.items()}


def _route_a_sheet_contacts_by_face_metal(
    build_input: GeometryBuildInput,
    records: Sequence[MMContactRecord],
) -> dict[str, tuple[MMContactRecord, ...]]:
    """Project typed finite MM records into Route-A sheet cap authority."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    result: dict[str, list[MMContactRecord]] = {}
    for record in records:
        lower = entities[record.lower_entity_id]
        upper = entities[record.upper_entity_id]
        roles = {lower.part_role, upper.part_role}
        if roles not in ({"face_metal", "contact_pad"}, {"face_metal", "bump_body"}):
            continue
        face = lower if lower.part_role == "face_metal" else upper
        result.setdefault(face.semantic_id, []).append(record)
    return {
        entity_id: tuple(
            sorted(
                contacts,
                key=lambda contact: (
                    _canonical_loop_sort_key(_clean_loop(contact.outer_loop)),
                    contact.contact_id,
                ),
            )
        )
        for entity_id, contacts in result.items()
    }


def _route_a_sheet_contact_cap_face(
    build_input: GeometryBuildInput,
    *,
    sheet_entity: SemanticEntitySpec,
    contact: MMContactRecord,
) -> str:
    """Select the sheet side opposite its typed finite MM contact direction."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    lower = entities.get(contact.lower_entity_id)
    upper = entities.get(contact.upper_entity_id)
    if (
        lower is None
        or upper is None
        or sheet_entity.semantic_id
        not in {
            contact.lower_entity_id,
            contact.upper_entity_id,
        }
    ):
        raise ValueError(f"{contact.contact_id} has ambiguous sheet contact members")
    normal = tuple(float(value) for value in contact.normal)
    if len(normal) != 3 or not all(isfinite(value) for value in normal):
        raise ValueError(f"{contact.contact_id} has ambiguous sheet contact normal")
    roles = {lower.part_role, upper.part_role}
    if roles == {"face_metal", "contact_pad"}:
        # A typed contact_pad is the same-z under-sheet normalization authority.
        return "bottom"
    if roles != {"face_metal", "bump_body"}:
        raise ValueError(f"{contact.contact_id} has incompatible sheet contact roles")
    if (
        abs(normal[0]) > _TOPOLOGY_EPS_UM
        or abs(normal[1]) > _TOPOLOGY_EPS_UM
        or abs(normal[2]) <= _TOPOLOGY_EPS_UM
    ):
        raise ValueError(f"{contact.contact_id} has ambiguous sheet contact normal")
    if sheet_entity.semantic_id == lower.semantic_id:
        contact_face = "top" if normal[2] > 0 else "bottom"
        source_face_id = contact.lower_source_face_id
        expected_source_face_id = f"{lower.semantic_id}__{contact_face}"
    else:
        contact_face = "bottom" if normal[2] > 0 else "top"
        source_face_id = contact.upper_source_face_id
        expected_source_face_id = f"{upper.semantic_id}__{contact_face}"
    expected_lower_face_id = (
        f"{lower.semantic_id}__{'top' if normal[2] > 0 else 'bottom'}"
    )
    expected_upper_face_id = (
        f"{upper.semantic_id}__{'bottom' if normal[2] > 0 else 'top'}"
    )
    if (
        source_face_id != expected_source_face_id
        or contact.lower_source_face_id != expected_lower_face_id
        or contact.upper_source_face_id != expected_upper_face_id
    ):
        raise ValueError(f"{contact.contact_id} has inconsistent sheet contact face")
    return "bottom" if contact_face == "top" else "top"


def _contact_loops_for_entity(
    contact_faces: Mapping[
        tuple[str, str],
        tuple[tuple[tuple[float, float], ...], ...],
    ],
    semantic_id: str,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    return tuple(
        loop
        for (entity_id, _), loops in contact_faces.items()
        if entity_id == semantic_id
        for loop in loops
    )


def _subtract_contact_patches_from_face(
    geometry_ref: Mapping[str, Any],
    contact_loops: Sequence[tuple[tuple[float, float], ...]],
) -> tuple[dict[str, Any], ...]:
    if not contact_loops:
        return (dict(geometry_ref),)
    outer_loop = _clean_loop(geometry_ref["outer_loop"])
    existing_holes = tuple(
        _clean_loop(hole_loop) for hole_loop in geometry_ref.get("hole_loops", ())
    )
    remaining_holes: list[tuple[tuple[float, float], ...]] = list(existing_holes)
    for contact_loop in contact_loops:
        if _same_loop_geometry(contact_loop, outer_loop):
            return ()
        remaining_holes.append(_clean_loop(contact_loop))
    simple_ref = {
        **dict(geometry_ref),
        "hole_loops": tuple(remaining_holes),
        "contact_hole_loops": tuple(_clean_loop(loop) for loop in contact_loops),
    }
    if _contact_holes_are_simple(outer_loop, remaining_holes):
        return (simple_ref,)

    import gdstk

    live_region = _boolean_gdstk_region(
        gdstk,
        _gdstk_surface_region(geometry_ref),
        tuple(gdstk.Polygon(loop) for loop in contact_loops),
        "not",
    )
    return _geometry_refs_from_gdstk_region(geometry_ref, live_region)


def _same_loop_geometry(left: Any, right: Any) -> bool:
    return _loop_signature(left) == _loop_signature(right)


def _contact_holes_are_simple(
    outer_loop: tuple[tuple[float, float], ...],
    hole_loops: Sequence[tuple[tuple[float, float], ...]],
) -> bool:
    for hole_loop in hole_loops:
        if not _loop_strictly_inside_loop(hole_loop, outer_loop):
            return False
    hole_records = tuple(
        (_loop_bounds_tuple(hole_loop), hole_loop) for hole_loop in hole_loops
    )
    for index, (left_bounds, left) in enumerate(hole_records):
        for right_bounds, right in hole_records[index + 1 :]:
            if not _bounds_tuple_may_overlap(left_bounds, right_bounds):
                continue
            if _loops_share_edge_overlap(left, right):
                return False
    return True


def _loop_strictly_inside_loop(loop: Any, container: Any) -> bool:
    """Require a real interior hole, excluding any boundary touch."""
    if not _loop_inside_loop(loop, container):
        return False
    loop_points = _clean_loop(loop)
    container_points = _clean_loop(container)
    if any(
        _point_on_segment(point, start, end)
        for point in loop_points
        for start, end in _ring_edges(container_points)
    ):
        return False
    return not _loops_share_edge_overlap(loop_points, container_points)


def _point_on_segment(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> bool:
    dx, dy = end[0] - start[0], end[1] - start[1]
    cross = (point[0] - start[0]) * dy - (point[1] - start[1]) * dx
    if abs(cross) > _TOPOLOGY_EPS_UM:
        return False
    return (
        min(start[0], end[0]) - _TOPOLOGY_EPS_UM
        <= point[0]
        <= max(start[0], end[0]) + _TOPOLOGY_EPS_UM
        and min(start[1], end[1]) - _TOPOLOGY_EPS_UM
        <= point[1]
        <= max(start[1], end[1]) + _TOPOLOGY_EPS_UM
    )


def _loop_bounds_tuple(
    loop: tuple[tuple[float, float], ...],
) -> tuple[float, float, float, float]:
    return (
        min(point[0] for point in loop),
        min(point[1] for point in loop),
        max(point[0] for point in loop),
        max(point[1] for point in loop),
    )


def _bounds_tuple_may_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    """Cheap candidate filter; contact/hole validity is checked on loops."""
    return not (
        left[2] <= right[0] + _TOPOLOGY_EPS_UM
        or right[2] <= left[0] + _TOPOLOGY_EPS_UM
        or left[3] <= right[1] + _TOPOLOGY_EPS_UM
        or right[3] <= left[1] + _TOPOLOGY_EPS_UM
    )


def _interface_kinds(interface: InterfacePlanRecord) -> tuple[str, ...]:
    raw = interface.metadata.get("interface_kinds", (interface.kind,))
    if isinstance(raw, str):
        raw_values = (raw,)
    elif isinstance(raw, tuple | list | set):
        raw_values = raw
    else:
        raw_values = (interface.kind,)

    seen = {str(value) for value in raw_values}
    seen.add(interface.kind)
    return tuple(kind for kind in _INTERFACE_KIND_ORDER if kind in seen)


def _interface_surface_kinds(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interface: InterfacePlanRecord,
) -> tuple[str, ...]:
    del build_input
    raw_kinds = interface.metadata.get("interface_kinds")
    if raw_kinds is not None:
        if isinstance(raw_kinds, str):
            raw_kinds = (raw_kinds,)
        return tuple(kind for kind in _INTERFACE_KIND_ORDER if kind in set(raw_kinds))
    if _is_route_a_sheet_interface(route, interface):
        return ("MS", "MA")
    kinds = set(_interface_kinds(interface))
    return tuple(kind for kind in _INTERFACE_KIND_ORDER if kind in kinds)


def _interface_surface_owner_ids(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interface: InterfacePlanRecord,
) -> tuple[str, ...]:
    raw_owner_ids = interface.metadata.get("surface_owner_semantic_ids")
    if raw_owner_ids is not None:
        if isinstance(raw_owner_ids, str):
            return (raw_owner_ids,)
        return _unique_ids(raw_owner_ids)
    if _is_route_a_sheet_interface(route, interface):
        entity = _entity_by_id(build_input, interface.owner_semantic_ids[0])
        return _unique_ids(
            (
                entity.semantic_id,
                *_route_a_sheet_boundary_volume_ids(build_input, entity),
            )
        )
    return tuple(interface.owner_semantic_ids)


def _interface_boundary_volume_ids(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interface: InterfacePlanRecord,
) -> tuple[str, ...]:
    raw_boundary_ids = interface.metadata.get("boundary_volume_ids")
    if raw_boundary_ids is not None:
        if isinstance(raw_boundary_ids, str):
            return (raw_boundary_ids,)
        return _unique_ids(raw_boundary_ids)
    if _is_route_a_sheet_interface(route, interface):
        return _route_a_sheet_boundary_volume_ids(
            build_input,
            _entity_by_id(build_input, interface.owner_semantic_ids[0]),
        )
    return _unique_ids(interface.owner_semantic_ids)


def _surface_interface_id(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    interface: InterfacePlanRecord,
) -> str:
    if not _is_route_a_sheet_interface(route, interface):
        return interface.interface_id
    entity = _entity_by_id(build_input, interface.owner_semantic_ids[0])
    boundary_ids = _route_a_sheet_boundary_volume_ids(build_input, entity)
    suffix = interface.interface_id.rsplit("__", 1)[-1]
    return f"MA__{entity.semantic_id}__{'__'.join(boundary_ids)}__{suffix}"


def _is_route_a_sheet_interface(
    route: RouteLiteral,
    interface: InterfacePlanRecord,
) -> bool:
    return (
        route == "A"
        and interface.metadata.get("recognition_rule")
        == "route_a_surface_sheet_polygon"
    )


def _is_hidden_contact_interface(
    route: RouteLiteral,
    interface: InterfacePlanRecord,
) -> bool:
    return (
        interface.recognition_rule == "coplanar_conductor_contact_patch"
        and route in {"A", "B"}
        and bool(interface.metadata.get("hidden_solver_contact"))
    )


def _volume_physical_name(volume: VolumePlanRecord) -> str:
    override = volume.metadata.get("physical_name")
    if isinstance(override, str) and override:
        return override
    physical_owner_id = volume.metadata.get("physical_owner_semantic_id")
    if isinstance(physical_owner_id, str) and physical_owner_id:
        return physical_owner_id
    return volume.owner_semantic_id


def _surface_physical_name(surface: SurfacePlanRecord) -> str:
    override = surface.metadata.get("physical_name")
    if isinstance(override, str) and override:
        return override
    raw = surface.metadata.get("interface_kinds", ())
    if isinstance(raw, str):
        kinds = (raw,)
    elif isinstance(raw, tuple | list | set):
        kinds = tuple(str(kind) for kind in raw)
    else:
        kinds = ()
    if len(kinds) > 2:
        raise ValueError(
            f"{surface.surface_id} has too many interface kinds: {kinds!r}"
        )
    kind_prefix = "_".join(kinds)
    exposed_role = surface.metadata.get("exposed_surface_role")
    boundary_role = surface.metadata.get("boundary_role")
    owner_ids = _surface_owner_ids(surface)
    physical_owner_ids = _surface_physical_owner_ids(surface)
    if physical_owner_ids != owner_ids:
        return _grouped_surface_physical_name(
            surface,
            kind_prefix=kind_prefix,
            physical_owner_ids=physical_owner_ids,
        )
    if surface.interface_id is not None:
        parts = surface.interface_id.split("__")
        if isinstance(exposed_role, str) and exposed_role.startswith("sidewall_"):
            parts[-1] = "SIDEWALL"
        if parts and parts[0] in _INTERFACE_KIND_ORDER:
            return "__".join((kind_prefix or parts[0], *parts[1:]))
        return surface.interface_id
    if kind_prefix:
        owner_ids = surface.metadata.get(
            "owner_semantic_ids",
            (surface.owner_semantic_id,),
        )
        if isinstance(owner_ids, str):
            owner_ids = (owner_ids,)
        suffix = exposed_role
        parts = [kind_prefix, *(str(owner_id) for owner_id in owner_ids)]
        if suffix:
            parts.append(str(suffix).upper())
        return "__".join(parts)
    if boundary_role == "sidewall":
        return surface.surface_id.removeprefix("SURF__").rsplit("__", 1)[0]
    return surface.surface_id.removeprefix("SURF__")


def _grouped_surface_physical_name(
    surface: SurfacePlanRecord,
    *,
    kind_prefix: str,
    physical_owner_ids: tuple[str, ...],
) -> str:
    exposed_role = surface.metadata.get("exposed_surface_role")
    suffix = _surface_role_suffix(exposed_role)
    if surface.interface_id is not None:
        parts = surface.interface_id.split("__")
        if len(parts) >= 2 and parts[1] == "CONTACT":
            return "__".join((kind_prefix or parts[0], "CONTACT", *physical_owner_ids))
        if suffix is None and parts:
            suffix = _surface_role_suffix(parts[-1])
        return "__".join(
            (
                kind_prefix or (parts[0] if parts else ""),
                *physical_owner_ids,
                *((suffix,) if suffix else ()),
            )
        )
    return "__".join(
        (
            kind_prefix,
            *physical_owner_ids,
            *((suffix,) if suffix else ()),
        )
    )


def _surface_role_suffix(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith(("sidewall_", "SIDEWALL_")):
        return "SIDEWALL"
    if value.lower() in {"top", "bottom", "sidewall"}:
        return value.upper()
    if value.upper() in {"TOP", "BOTTOM", "SIDEWALL"}:
        return value.upper()
    return None


def _surface_physical_owner_ids(surface: SurfacePlanRecord) -> tuple[str, ...]:
    owner_ids = surface.metadata.get("physical_owner_semantic_ids")
    if isinstance(owner_ids, str):
        return (owner_ids,)
    if isinstance(owner_ids, Sequence):
        return tuple(str(owner_id) for owner_id in owner_ids)
    return _surface_owner_ids(surface)


def _physical_group_owner_ids(
    build_input: GeometryBuildInput,
    owner_ids: Sequence[str],
) -> tuple[str, ...]:
    entities_by_id = {entity.semantic_id: entity for entity in build_input.entities}
    return _unique_ids(
        _entity_physical_group_id(entities_by_id[owner_id])
        if owner_id in entities_by_id
        else owner_id
        for owner_id in owner_ids
    )


def _entity_physical_group_id(entity: SemanticEntitySpec) -> str:
    auto_group_id = entity.metadata.get("auto_vacuum_group_id")
    if isinstance(auto_group_id, str) and auto_group_id:
        return auto_group_id
    if entity.part_role not in HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES:
        return entity.semantic_id
    for key in ("physical_group_name", "physical_group_id", "semantic_group_id"):
        value = entity.metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return entity.semantic_id


def _plan_substrate_air_surfaces(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
) -> tuple[SurfacePlanRecord, ...]:
    import gdstk

    records: list[SurfacePlanRecord] = list(
        _plan_solution_domain_boundary_surfaces(build_input, route=route)
    )
    surface_index = 0
    for lower, upper, z_um, bounds in _solution_interface_planes(build_input):
        kind = _solution_interface_kind(lower, upper)
        owner_ids = _solution_interface_owner_ids(kind, lower, upper)
        interface_region = _boolean_gdstk_region(
            gdstk,
            _solution_entity_xy_region(gdstk, lower),
            _solution_entity_xy_region(gdstk, upper),
            "and",
        )
        if not interface_region:
            continue
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
            solution_geometry_refs = _solution_interface_geometry_refs(
                base_geometry_ref,
                plane_conductors,
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
                        },
                    )
                )
                surface_index += 1
    return tuple(records)


def _solution_interface_geometry_refs(
    parent_geometry_ref: Mapping[str, Any],
    plane_conductors: Sequence[SemanticEntitySpec],
) -> tuple[dict[str, Any], ...]:
    """Create live solution-interface patches after removing conductors."""
    if not plane_conductors:
        return (dict(parent_geometry_ref),)

    import gdstk

    hole_loops = _simple_interior_hole_loops(parent_geometry_ref, plane_conductors)
    if hole_loops is not None:
        return ({**dict(parent_geometry_ref), "hole_loops": hole_loops},)

    base_region = _gdstk_surface_region(parent_geometry_ref)
    conductor_region = tuple(
        polygon
        for entity in plane_conductors
        for polygon in _entity_occupied_region(gdstk, entity)
    )
    live_region = _boolean_gdstk_region(
        gdstk,
        base_region,
        conductor_region,
        "not",
    )
    return _geometry_refs_from_gdstk_region(parent_geometry_ref, live_region)


def _simple_interior_hole_loops(
    parent_geometry_ref: Mapping[str, Any],
    plane_conductors: Sequence[SemanticEntitySpec],
) -> tuple[tuple[tuple[float, float], ...], ...] | None:
    if any(entity.geometry.get("hole_loops") for entity in plane_conductors):
        return None
    base_loop = _clean_loop(parent_geometry_ref["outer_loop"])
    hole_loops = tuple(
        _clean_loop(entity.geometry["outer_loop"])
        for entity in plane_conductors
        if "outer_loop" in entity.geometry
    )
    if len(hole_loops) != len(plane_conductors):
        return None
    if any(not _loop_inside_loop(hole_loop, base_loop) for hole_loop in hole_loops):
        return None
    # Overlapping conductor footprints must be unioned before they become
    # solution-interface holes; separate overlapping holes create a third
    # coincident rim when a normalized contact-pad cap is added.
    import gdstk

    for index, left in enumerate(hole_loops):
        for right in hole_loops[index + 1 :]:
            if _boolean_gdstk_region(
                gdstk,
                (gdstk.Polygon(left),),
                (gdstk.Polygon(right),),
                "and",
            ):
                return None
    all_loops = (base_loop, *hole_loops)
    for index, left in enumerate(all_loops):
        for right in all_loops[index + 1 :]:
            if _loops_share_edge_overlap(left, right):
                return None
    return hole_loops


def _loops_share_edge_overlap(
    left: tuple[tuple[float, float], ...],
    right: tuple[tuple[float, float], ...],
) -> bool:
    return any(
        _segment_overlap_interval(left_start, left_end, right_start, right_end)
        is not None
        for left_start, left_end in _ring_edges(left)
        for right_start, right_end in _ring_edges(right)
    )


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


def _interpolate_2d(
    start: tuple[float, float],
    end: tuple[float, float],
    parameter: float,
) -> tuple[float, float]:
    return (
        start[0] + (end[0] - start[0]) * parameter,
        start[1] + (end[1] - start[1]) * parameter,
    )


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


def _edge_is_covered_by_loop_edge(
    start: tuple[float, float],
    end: tuple[float, float],
    loop: Any,
) -> bool:
    return any(
        interval is not None
        and interval[0] <= _TOPOLOGY_EPS_UM
        and interval[1] >= 1.0 - _TOPOLOGY_EPS_UM
        for edge_start, edge_end in _ring_edges(_clean_loop(loop))
        for interval in (_segment_overlap_interval(start, end, edge_start, edge_end),)
    )


def _canonical_face_signature_3d(
    ring: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...] | None:
    if len(ring) < 3:
        return None
    normalized = tuple(_coordinate_key(point) for point in ring)
    candidates: list[tuple[tuple[float, float, float], ...]] = []
    for candidate in (normalized, tuple(reversed(normalized))):
        rotations = tuple(
            candidate[index:] + candidate[:index] for index in range(len(candidate))
        )
        candidates.append(min(rotations))
    return min(candidates)


def _merge_solution_sidewall_interfaces(
    build_input: GeometryBuildInput,
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
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
        kind = _solution_interface_kind(lower, upper)
        owner_ids = _solution_interface_owner_ids(kind, lower, upper)
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


def _cutout_shell_surface_ids(
    build_input: GeometryBuildInput,
    route: RouteLiteral,
    entity: SemanticEntitySpec,
    *,
    interfaces: tuple[InterfacePlanRecord, ...] = (),
    contact_faces: Mapping[
        tuple[str, str],
        tuple[tuple[tuple[float, float], ...], ...],
    ]
    | None = None,
) -> tuple[str, ...]:
    contact_faces = contact_faces or {}
    top_bottom_ids: list[str] = []
    for shell_part in ("top", "bottom"):
        base_surface_id = _conductor_boundary_surface_id(
            route,
            entity,
            "cutout_boundary_shell",
            shell_part,
        )
        base_geometry_ref = {
            **_route_entity_geometry_ref(
                build_input,
                route,
                entity,
                representation="cutout_boundary_shell",
                interfaces=interfaces,
            ),
            "shell_part": shell_part,
        }
        face_geometry_refs = _subtract_contact_patches_from_face(
            base_geometry_ref,
            contact_faces.get((entity.semantic_id, shell_part), ()),
        )
        if len(face_geometry_refs) == 1:
            top_bottom_ids.append(base_surface_id)
        else:
            top_bottom_ids.extend(
                f"{base_surface_id}__P{index:04d}"
                for index, _ in enumerate(face_geometry_refs)
            )
    sidewall_adjacent_id = (
        _conductor_sidewall_adjacent_solution_id(build_input, entity)
        if route == "C"
        else None
    )
    sidewall_refs = (
        ()
        if route == "C" and sidewall_adjacent_id is None
        else _conductor_sidewall_geometry_refs(
            build_input,
            route=route,
            entity=entity,
            representation="cutout_boundary_shell",
            adjacent_solution_id=sidewall_adjacent_id,
            interfaces=interfaces,
        )
    )
    sidewall_ids = tuple(
        _conductor_boundary_surface_id(
            route,
            entity,
            "cutout_boundary_shell",
            f"sidewall_{edge_index:04d}",
        )
        for edge_index, _ in enumerate(sidewall_refs)
    )
    return (*top_bottom_ids, *sidewall_ids)


def _conductor_boundary_surface_id(
    route: RouteLiteral,
    entity: SemanticEntitySpec,
    representation: str,
    shell_part: str,
) -> str:
    surface_kind = "SHELL" if representation == "cutout_boundary_shell" else "MAT"
    return f"SURF__{route}__{surface_kind}__{entity.semantic_id}__{shell_part.upper()}"


def _conductor_face_interface_id(
    interface_kind: str,
    semantic_id: str,
    adjacent_id: str,
    shell_part: str,
    face_index: int | None,
) -> str:
    suffix = "" if face_index is None else f"__P{face_index:04d}"
    return (
        f"{interface_kind}__{semantic_id}__{adjacent_id}__{shell_part.upper()}{suffix}"
    )


def _entity_by_id(
    build_input: GeometryBuildInput,
    semantic_id: str,
) -> SemanticEntitySpec:
    for entity in build_input.entities:
        if entity.semantic_id == semantic_id:
            return entity
    raise ValueError(f"unknown semantic entity: {semantic_id}")


def _domain_bounds_loop(bounds: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    return (
        (float(bounds["x_min_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_max_um"])),
        (float(bounds["x_min_um"]), float(bounds["y_max_um"])),
    )


def _loop_inside_loop(loop: Any, container: Any) -> bool:
    import gdstk

    clean_loop = _clean_loop(loop)
    clean_container = _clean_loop(container)
    loop_area = abs(_polygon_area(clean_loop))
    intersection = _boolean_gdstk_region(
        gdstk,
        (gdstk.Polygon(clean_loop),),
        (gdstk.Polygon(clean_container),),
        "and",
    )
    intersection_area = sum(
        abs(_polygon_area(polygon.points)) for polygon in intersection
    )
    return loop_area - intersection_area <= max(_TOPOLOGY_EPS_UM, loop_area * 1e-9)


def _loop_centroid(loop: Any) -> tuple[float, float]:
    points = tuple((float(point[0]), float(point[1])) for point in loop)
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _surface_boundary_volume_ids(
    surface: SurfacePlanRecord,
    *,
    known_entity_ids: set[str],
) -> tuple[str, ...]:
    if surface.surface_role == "lumped_port" and not surface.metadata.get(
        "route_a_boundary_port"
    ):
        return ()
    if surface.metadata.get("embedded_surface_sheet") and not surface.metadata.get(
        "sheet_contact_cap"
    ):
        return ()
    raw = surface.metadata.get(
        "boundary_volume_ids",
        surface.metadata.get("owner_semantic_ids", (surface.owner_semantic_id,)),
    )
    if isinstance(raw, str):
        values = (raw,)
    else:
        values = tuple(str(value) for value in raw)
    result = _unique_ids(value for value in values if value in known_entity_ids)
    if len(result) > 2:
        raise ValueError(
            f"{surface.surface_id} belongs to more than two volumes: {result!r}"
        )
    return result


def _surface_orientation_for_volume(
    surface: SurfacePlanRecord,
    entity: SemanticEntitySpec,
) -> SurfaceOrientationLiteral:
    """Orient a planned surface as an outward face of one owning volume."""
    normal = _surface_normal_vector(surface)
    surface_centroid = _surface_centroid(surface)
    volume_center = _entity_volume_center_um(entity)
    outward = tuple(
        surface_centroid[index] - volume_center[index] for index in range(3)
    )
    dot = sum(normal[index] * outward[index] for index in range(3))
    if abs(dot) <= 1e-9:
        raise ValueError(
            f"{surface.surface_id} has ambiguous orientation for {entity.semantic_id}"
        )
    return "forward" if dot > 0 else "reversed"


def _surface_normal_vector(surface: SurfacePlanRecord) -> tuple[float, float, float]:
    if surface.normal_hint is not None:
        return tuple(float(value) for value in surface.normal_hint)
    ring = _surface_ring3d_specs(surface)[0][2]
    origin = ring[0]
    for first_index in range(1, len(ring) - 1):
        first = _vector_subtract(ring[first_index], origin)
        for second_index in range(first_index + 1, len(ring)):
            second = _vector_subtract(ring[second_index], origin)
            normal = _vector_cross(first, second)
            length_sq = sum(value * value for value in normal)
            if length_sq > 1e-18:
                return normal
    raise ValueError(f"{surface.surface_id} has no nondegenerate normal")


def _surface_centroid(surface: SurfacePlanRecord) -> tuple[float, float, float]:
    points = tuple(
        coordinate
        for _, _, ring in _surface_ring3d_specs(surface)
        for coordinate in ring
    )
    if not points:
        raise ValueError(f"{surface.surface_id} has no coordinates")
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    )


def _entity_volume_center_um(
    entity: SemanticEntitySpec,
) -> tuple[float, float, float]:
    if _is_solution_entity(entity):
        bounds = _solution_bounds(entity)
        return (
            (float(bounds["x_min_um"]) + float(bounds["x_max_um"])) / 2.0,
            (float(bounds["y_min_um"]) + float(bounds["y_max_um"])) / 2.0,
            (float(entity.geometry["z_min_um"]) + float(entity.geometry["z_max_um"]))
            / 2.0,
        )
    if "outer_loop" not in entity.geometry:
        raise ValueError(f"{entity.semantic_id} requires outer_loop for volume center")
    loop = _clean_loop(entity.geometry["outer_loop"])
    z_min_um, z_max_um = _entity_z_range_um(entity)
    return (
        (min(point[0] for point in loop) + max(point[0] for point in loop)) / 2.0,
        (min(point[1] for point in loop) + max(point[1] for point in loop)) / 2.0,
        (z_min_um + z_max_um) / 2.0,
    )


def _vector_subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _vector_cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _unique_ids(values: Any) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if item in seen:
            continue
        result.append(item)
        seen.add(item)
    return tuple(result)


def _ordered_loop_signature(
    curve_refs: tuple[CurveRefRecord, ...],
) -> tuple[tuple[str, int], ...]:
    """Return a rotation-stable loop signature without discarding orientation."""
    signature = tuple((ref.curve_id, ref.orientation) for ref in curve_refs)
    if not signature:
        return ()
    rotations = (
        signature[index:] + signature[:index] for index in range(len(signature))
    )
    return min(rotations)


def _surface_planar_bounds(surface: SurfacePlanRecord) -> dict[str, float]:
    points = [
        point
        for loop in (
            surface.geometry_ref["outer_loop"],
            *surface.geometry_ref.get("hole_loops", ()),
        )
        for point in _clean_loop(loop)
    ]
    return {
        "x_min_um": min(point[0] for point in points),
        "y_min_um": min(point[1] for point in points),
        "x_max_um": max(point[0] for point in points),
        "y_max_um": max(point[1] for point in points),
    }


def _merge_bounds(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> dict[str, float]:
    return {
        "x_min_um": min(float(first["x_min_um"]), float(second["x_min_um"])),
        "y_min_um": min(float(first["y_min_um"]), float(second["y_min_um"])),
        "x_max_um": max(float(first["x_max_um"]), float(second["x_max_um"])),
        "y_max_um": max(float(first["y_max_um"]), float(second["y_max_um"])),
    }


def _bounds_touch_or_overlap(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> bool:
    return not (
        float(first["x_max_um"]) < float(second["x_min_um"]) - _TOPOLOGY_EPS_UM
        or float(second["x_max_um"]) < float(first["x_min_um"]) - _TOPOLOGY_EPS_UM
        or float(first["y_max_um"]) < float(second["y_min_um"]) - _TOPOLOGY_EPS_UM
        or float(second["y_max_um"]) < float(first["y_min_um"]) - _TOPOLOGY_EPS_UM
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


def _conductor_boundary_volume_ids(
    route: RouteLiteral,
    entity: SemanticEntitySpec,
    adjacent_solution_id: str,
) -> tuple[str, ...]:
    if route == "C":
        return (entity.semantic_id, adjacent_solution_id)
    return (adjacent_solution_id,)


def _route_conductor_boundary_volume_ids(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    entity: SemanticEntitySpec,
    face: str,
    adjacent_solution_id: str,
) -> tuple[str, ...]:
    """Keep each finite conductor face with its directly adjacent solution.

    Route-A boundary-attached components join an exterior shell through their
    exposed side/top faces; their retained bottom MS face belongs only to the
    substrate-side solution and is never duplicated into air.
    """
    del build_input, face
    return _conductor_boundary_volume_ids(route, entity, adjacent_solution_id)


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


def _undirected_xy_edge(
    start: Sequence[float], end: Sequence[float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    first = _coordinate_2d_key((float(start[0]), float(start[1])))
    second = _coordinate_2d_key((float(end[0]), float(end[1])))
    return tuple(sorted((first, second)))


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


def _conductor_solution_interface_kind(solution: SemanticEntitySpec) -> str:
    return "MA" if _is_vacuum_solution_entity(solution) else "MS"


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
    for entity in build_input.entities:
        if (
            _is_solution_entity(entity)
            or entity.route_representations.get(route) is None
            or "outer_loop" not in entity.geometry
            or (
                (
                    base_loop is not None
                    and not _loop_inside_loop(entity.geometry["outer_loop"], base_loop)
                )
                or (
                    base_region is not None
                    and not _boolean_gdstk_region(
                        gdstk,
                        _entity_occupied_region(gdstk, entity),
                        base_region,
                        "and",
                    )
                )
            )
        ):
            continue
        representation = entity.route_representations.get(route)
        if representation == "surface_sheet":
            if set(
                _route_a_sheet_boundary_volume_ids(build_input, entity)
            ) == pair_ids and _same_z(
                _route_a_sheet_plane_z_um(build_input, entity), z_um
            ):
                records.append(entity)
            continue
        if any(
            _same_z(face_z, z_um)
            and _conductor_face_adjacent_solution_id(
                build_input,
                entity,
                face,
            )
            in pair_ids
            for face, face_z in zip(
                ("bottom", "top"),
                _entity_z_range_um(entity),
                strict=True,
            )
        ):
            records.append(entity)
    return tuple(records)


def _solution_interface_kind(
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
) -> str:
    if _is_vacuum_solution_entity(lower) or _is_vacuum_solution_entity(upper):
        return (
            "AA"
            if (_is_vacuum_solution_entity(lower) and _is_vacuum_solution_entity(upper))
            else "SA"
        )
    return "SS"


def _solution_interface_owner_ids(
    kind: str,
    lower: SemanticEntitySpec,
    upper: SemanticEntitySpec,
) -> tuple[str, str]:
    if kind == "SA":
        if _is_vacuum_solution_entity(lower):
            return (upper.semantic_id, lower.semantic_id)
        return (lower.semantic_id, upper.semantic_id)
    return (lower.semantic_id, upper.semantic_id)


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


def _intersect_bounds(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, float] | None:
    bounds = {
        "x_min_um": max(float(first["x_min_um"]), float(second["x_min_um"])),
        "y_min_um": max(float(first["y_min_um"]), float(second["y_min_um"])),
        "x_max_um": min(float(first["x_max_um"]), float(second["x_max_um"])),
        "y_max_um": min(float(first["y_max_um"]), float(second["y_max_um"])),
    }
    if (
        bounds["x_min_um"] >= bounds["x_max_um"]
        or bounds["y_min_um"] >= bounds["y_max_um"]
    ):
        return None
    return bounds


def _same_bounds(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return all(
        abs(float(first[key]) - float(second[key])) <= _TOPOLOGY_EPS_UM
        for key in ("x_min_um", "y_min_um", "x_max_um", "y_max_um")
    )


def _bounds_contains_point(
    bounds: Mapping[str, Any],
    point: tuple[float, float],
) -> bool:
    x, y = point
    return float(bounds["x_min_um"]) <= x <= float(bounds["x_max_um"]) and float(
        bounds["y_min_um"]
    ) <= y <= float(bounds["y_max_um"])


def _entity_z_range_um(entity: SemanticEntitySpec) -> tuple[float, float]:
    z_min_um = float(entity.geometry.get("z_min_um", entity.geometry.get("z_um", 0.0)))
    z_max_um = entity.geometry.get("z_max_um")
    if z_max_um is not None:
        z_max_um = float(z_max_um)
        if not isfinite(z_max_um):
            raise ValueError(f"{entity.semantic_id} has non-finite z_max_um")
        if z_max_um > z_min_um:
            return z_min_um, z_max_um

    thickness_um = float(entity.geometry.get("thickness_um", 0.0))
    return z_min_um, z_min_um + thickness_um


def _same_z(left: float, right: float) -> bool:
    return abs(left - right) <= _TOPOLOGY_EPS_UM


def _entity_geometry_ref(
    entity: SemanticEntitySpec,
    *,
    representation: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "from_semantic_id": entity.semantic_id,
        "geometry_kind": entity.geometry_kind,
        "part_role": entity.part_role,
        "representation": representation,
        "source_polygon_ids": entity.polygon_ids,
    }
    result.update(_geometry_ref_from_metadata(entity.geometry))
    result.update(_geometry_ref_from_metadata(entity.metadata))
    if "z_um" in entity.geometry and "plane" not in result:
        result["plane"] = {"axis": "z", "value_um": entity.geometry["z_um"]}
    return result


def _route_entity_geometry_ref(
    build_input: GeometryBuildInput,
    route: RouteLiteral,
    entity: SemanticEntitySpec,
    *,
    representation: str,
    interfaces: tuple[InterfacePlanRecord, ...] = (),
) -> dict[str, Any]:
    del build_input
    geometry_ref = _entity_geometry_ref(entity, representation=representation)
    if route != "A" or representation != "cutout_boundary_shell":
        return geometry_ref

    z_min_um, z_max_um = _entity_z_range_um(entity)
    for interface in interfaces:
        if interface.recognition_rule != "coplanar_conductor_contact_patch":
            continue
        metadata = interface.metadata
        plane = metadata.get("contact_plane") or metadata.get("plane")
        if not isinstance(plane, Mapping) or plane.get("axis") != "z":
            continue
        plane_z_um = float(plane["value_um"])
        if (
            metadata.get("upper_entity_id") == entity.semantic_id
            and metadata.get("upper_face") == "bottom"
        ):
            z_min_um = min(z_min_um, plane_z_um)
        if (
            metadata.get("lower_entity_id") == entity.semantic_id
            and metadata.get("lower_face") == "top"
        ):
            z_max_um = max(z_max_um, plane_z_um)

    if z_max_um <= z_min_um:
        raise ValueError(f"{entity.semantic_id} Route A cutout body has empty z range")
    if not _same_z(z_min_um, _entity_z_range_um(entity)[0]) or not _same_z(
        z_max_um,
        _entity_z_range_um(entity)[1],
    ):
        geometry_ref["z_um"] = z_min_um
        geometry_ref["z_min_um"] = z_min_um
        geometry_ref["thickness_um"] = z_max_um - z_min_um
        geometry_ref["route_a_cutout_z_range_um"] = (z_min_um, z_max_um)
    return geometry_ref


def _sidewall_geometry_refs(
    geometry_ref: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    z_min_um = float(geometry_ref.get("z_min_um", geometry_ref.get("z_um", 0.0)))
    thickness_um = float(geometry_ref.get("thickness_um", 0.0))
    if thickness_um <= 0:
        return ()
    z_max_um = z_min_um + thickness_um
    refs: list[dict[str, Any]] = []
    for ring_role, ring in (
        ("outer", geometry_ref["outer_loop"]),
        *(
            (f"hole_{index:04d}", hole_loop)
            for index, hole_loop in enumerate(geometry_ref.get("hole_loops", ()))
        ),
    ):
        for edge_index, (start, end) in enumerate(_ring_edges(_clean_loop(ring))):
            refs.append(
                {
                    "quad_points": (
                        (start[0], start[1], z_min_um),
                        (end[0], end[1], z_min_um),
                        (end[0], end[1], z_max_um),
                        (start[0], start[1], z_max_um),
                    ),
                    "sidewall_ring_role": ring_role,
                    "sidewall_edge_index": edge_index,
                }
            )
    return tuple(refs)


def _conductor_sidewall_geometry_refs(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    entity: SemanticEntitySpec,
    representation: str,
    adjacent_solution_id: str | None,
    interfaces: tuple[InterfacePlanRecord, ...] = (),
    excluded_footprints: Sequence[tuple[tuple[float, float], ...]] = (),
) -> tuple[dict[str, Any], ...]:
    base_ref = _route_entity_geometry_ref(
        build_input,
        route,
        entity,
        representation=representation,
        interfaces=interfaces,
    )
    face_refs = _subtract_contact_patches_from_face(base_ref, excluded_footprints)
    refs = tuple(
        side_ref
        for face_ref in face_refs
        for side_ref in _sidewall_geometry_refs(face_ref)
    )
    edge_index = _conductor_boundary_edge_index(
        build_input,
        route=route,
        entity=entity,
    )
    if route == "C":
        if adjacent_solution_id is None:
            raise ValueError(
                f"{entity.semantic_id} Route C sidewall requires a solution."
            )
        return _route_c_conductor_sidewall_geometry_refs(
            entity=entity,
            adjacent_solution=_entity_by_id(build_input, adjacent_solution_id),
            refs=refs,
            edge_index=edge_index,
        )
    exposed_refs: list[dict[str, Any]] = []
    for raw_geometry_ref in refs:
        if _sidewall_is_auto_vacuum_envelope_edge(
            build_input,
            entity=entity,
            geometry_ref=raw_geometry_ref,
        ):
            continue
        # Exact same-z conductor contacts remove their shared MM segment
        # before solution adjacency is selected.  One raw sidewall can span
        # contact and vacuum portions; only each retained topology segment is
        # eligible for a unique auto-vacuum child.
        for geometry_ref in _trim_sidewall_ref_against_route_conductors(
            geometry_ref=raw_geometry_ref,
            edge_index=edge_index,
        ):
            auto_parent, ref_adjacent_solution_id = (
                _auto_vacuum_sidewall_ref_solution_id(
                    build_input,
                    entity=entity,
                    geometry_ref=geometry_ref,
                )
            )
            if auto_parent:
                if ref_adjacent_solution_id is None:
                    raise ValueError(
                        f"{entity.semantic_id} sidewall has no exact auto-vacuum child."
                    )
            else:
                if adjacent_solution_id is None:
                    adjacent_solution_id = _conductor_sidewall_adjacent_solution_id(
                        build_input,
                        entity,
                    )
                if adjacent_solution_id is None:
                    continue
                ref_adjacent_solution_id = adjacent_solution_id
            adjacent_solution = _entity_by_id(build_input, ref_adjacent_solution_id)
            # The explicit parent-envelope check above is the only exterior rule
            # for auto-vacuum children. A disconnected child's rectangular bounds
            # can coincide with a conductor hole edge, which is not an exterior
            # boundary and must retain its exact child adjacency.
            if not auto_parent and _sidewall_on_solution_outer_boundary(
                geometry_ref, adjacent_solution
            ):
                continue
            exposed_refs.append(
                {
                    **geometry_ref,
                    "adjacent_solution_id": ref_adjacent_solution_id,
                }
            )
    return tuple(exposed_refs)


def _route_c_conductor_sidewall_geometry_refs(
    *,
    entity: SemanticEntitySpec,
    adjacent_solution: SemanticEntitySpec,
    refs: tuple[dict[str, Any], ...],
    edge_index: Mapping[
        tuple[tuple[float, float], float],
        Sequence[tuple[tuple[float, float], tuple[float, float], str]],
    ],
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for geometry_ref in refs:
        points = geometry_ref.get("quad_points", ())
        if len(points) != 4:
            result.append(dict(geometry_ref))
            continue
        start = (float(points[0][0]), float(points[0][1]))
        end = (float(points[1][0]), float(points[1][1]))
        exterior_intervals = (
            ((0.0, 1.0),)
            if _sidewall_on_solution_outer_boundary(geometry_ref, adjacent_solution)
            else ()
        )
        contact_intervals = _route_c_contact_intervals(
            entity=entity,
            start=start,
            end=end,
            edge_index=edge_index,
        )
        result.extend(
            _sidewall_subsegment_geometry_ref(
                geometry_ref,
                interval_start,
                interval_end,
                extra={"solution_exterior_boundary": True},
            )
            for interval_start, interval_end in exterior_intervals
        )
        result.extend(
            _sidewall_subsegment_geometry_ref(
                geometry_ref,
                interval_start,
                interval_end,
                extra={"adjacent_conductor_semantic_id": adjacent_id},
            )
            for (
                interval_start,
                interval_end,
                adjacent_id,
                create_surface,
            ) in contact_intervals
            if create_surface
        )
        blocked = (
            *exterior_intervals,
            *((start, end) for start, end, _, _ in contact_intervals),
        )
        result.extend(
            _sidewall_subsegment_geometry_ref(
                geometry_ref,
                interval_start,
                interval_end,
            )
            for interval_start, interval_end in _interval_complement(blocked)
        )
    return tuple(result)


def _conductor_boundary_edge_index(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    entity: SemanticEntitySpec,
) -> dict[
    tuple[tuple[float, float], float],
    tuple[tuple[tuple[float, float], tuple[float, float], str], ...],
]:
    if entity.part_role in HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES:
        return {}
    index: dict[
        tuple[tuple[float, float], float],
        list[tuple[tuple[float, float], tuple[float, float], str]],
    ] = {}
    for other in build_input.entities:
        if not _can_trim_against_route_conductor(
            route,
            entity=entity,
            other=other,
        ):
            continue
        for other_start, other_end in _entity_boundary_edges(other):
            line_key = _line_key_2d(other_start, other_end)
            if line_key is None:
                continue
            index.setdefault(line_key, []).append(
                (other_start, other_end, other.semantic_id)
            )
    return {line_key: tuple(records) for line_key, records in index.items()}


def _candidate_boundary_edges(
    edge_index: Mapping[
        tuple[tuple[float, float], float],
        Sequence[tuple[tuple[float, float], tuple[float, float], str]],
    ],
    start: tuple[float, float],
    end: tuple[float, float],
) -> Sequence[tuple[tuple[float, float], tuple[float, float], str]]:
    line_key = _line_key_2d(start, end)
    if line_key is None:
        return ()
    return edge_index.get(line_key, ())


def _route_c_contact_intervals(
    entity: SemanticEntitySpec,
    start: tuple[float, float],
    end: tuple[float, float],
    edge_index: Mapping[
        tuple[tuple[float, float], float],
        Sequence[tuple[tuple[float, float], tuple[float, float], str]],
    ],
) -> tuple[tuple[float, float, str, bool], ...]:
    records: list[tuple[float, float, str, bool]] = []
    for other_start, other_end, other_semantic_id in _candidate_boundary_edges(
        edge_index,
        start,
        end,
    ):
        interval = _segment_overlap_interval(start, end, other_start, other_end)
        if interval is None:
            continue
        records.append(
            (
                interval[0],
                interval[1],
                other_semantic_id,
                entity.semantic_id < other_semantic_id,
            )
        )
    return tuple(records)


def _trim_sidewall_ref_against_route_conductors(
    *,
    geometry_ref: Mapping[str, Any],
    edge_index: Mapping[
        tuple[tuple[float, float], float],
        Sequence[tuple[tuple[float, float], tuple[float, float], str]],
    ],
) -> tuple[dict[str, Any], ...]:
    points = geometry_ref.get("quad_points", ())
    if len(points) != 4:
        return (dict(geometry_ref),)
    start = (float(points[0][0]), float(points[0][1]))
    end = (float(points[1][0]), float(points[1][1]))
    covered_intervals: list[tuple[float, float]] = []
    for other_start, other_end, _ in _candidate_boundary_edges(
        edge_index,
        start,
        end,
    ):
        interval = _segment_overlap_interval(start, end, other_start, other_end)
        if interval is None:
            continue
        covered_intervals.append(interval)
    if not covered_intervals:
        return (dict(geometry_ref),)
    return tuple(
        _sidewall_subsegment_geometry_ref(
            geometry_ref,
            start_parameter,
            end_parameter,
        )
        for start_parameter, end_parameter in _interval_complement(
            covered_intervals,
        )
        if end_parameter - start_parameter > _TOPOLOGY_EPS_UM
    )


def _can_trim_against_route_conductor(
    route: RouteLiteral,
    *,
    entity: SemanticEntitySpec,
    other: SemanticEntitySpec,
) -> bool:
    if (
        other.semantic_id == entity.semantic_id
        or _is_solution_entity(other)
        or other.route_representations.get(route) is None
        or "outer_loop" not in other.geometry
    ):
        return False
    z_min_um, z_max_um = _entity_z_range_um(entity)
    other_z_min_um, other_z_max_um = _entity_z_range_um(other)
    return _same_z(z_min_um, other_z_min_um) and _same_z(z_max_um, other_z_max_um)


def _entity_boundary_edges(
    entity: SemanticEntitySpec,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return tuple(
        edge
        for loop in (
            entity.geometry["outer_loop"],
            *entity.geometry.get("hole_loops", ()),
        )
        for edge in _ring_edges(_clean_loop(loop))
    )


def _segment_overlap_interval(
    start: tuple[float, float],
    end: tuple[float, float],
    other_start: tuple[float, float],
    other_end: tuple[float, float],
) -> tuple[float, float] | None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-18:
        return None

    def parameter(point: tuple[float, float]) -> float | None:
        cross = (point[0] - start[0]) * dy - (point[1] - start[1]) * dx
        if abs(cross) > 1e-9:
            return None
        return ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq

    first = parameter(other_start)
    second = parameter(other_end)
    if first is None or second is None:
        return None
    overlap_start = max(0.0, min(first, second))
    overlap_end = min(1.0, max(first, second))
    if overlap_end - overlap_start <= _TOPOLOGY_EPS_UM:
        return None
    return overlap_start, overlap_end


def _line_key_2d(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], float] | None:
    start_key = tuple(round(float(value), 9) for value in start)
    end_key = tuple(round(float(value), 9) for value in end)
    direction = (
        end_key[0] - start_key[0],
        end_key[1] - start_key[1],
    )
    scale = max(abs(value) for value in direction)
    if scale <= 1e-18:
        return None
    unit = tuple(round(value / scale, 9) for value in direction)
    for value in unit:
        if abs(value) <= 1e-18:
            continue
        if value < 0:
            unit = (-unit[0], -unit[1])
        break
    offset = round(start_key[0] * unit[1] - start_key[1] * unit[0], 9)
    return unit, offset


def _interval_complement(
    intervals: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        start = max(0.0, min(1.0, start))
        end = max(0.0, min(1.0, end))
        if end - start <= _TOPOLOGY_EPS_UM:
            continue
        if not merged or start > merged[-1][1] + _TOPOLOGY_EPS_UM:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    exposed: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start - cursor > _TOPOLOGY_EPS_UM:
            exposed.append((cursor, start))
        cursor = max(cursor, end)
    if 1.0 - cursor > _TOPOLOGY_EPS_UM:
        exposed.append((cursor, 1.0))
    return tuple(exposed)


def _sidewall_subsegment_geometry_ref(
    geometry_ref: Mapping[str, Any],
    start_parameter: float,
    end_parameter: float,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    points = tuple(
        (float(point[0]), float(point[1]), float(point[2]))
        for point in geometry_ref["quad_points"]
    )
    return {
        **dict(geometry_ref),
        "quad_points": (
            _interpolate_3d(points[0], points[1], start_parameter),
            _interpolate_3d(points[0], points[1], end_parameter),
            _interpolate_3d(points[3], points[2], end_parameter),
            _interpolate_3d(points[3], points[2], start_parameter),
        ),
        "trimmed_by_conductor_contact": True,
        **dict(extra or {}),
    }


def _interpolate_3d(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    parameter: float,
) -> tuple[float, float, float]:
    return (
        start[0] + (end[0] - start[0]) * parameter,
        start[1] + (end[1] - start[1]) * parameter,
        start[2] + (end[2] - start[2]) * parameter,
    )


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


def _range_within_bounds(
    start: float,
    end: float,
    lower: float,
    upper: float,
) -> bool:
    return lower - _TOPOLOGY_EPS_UM <= min(start, end) and max(start, end) <= (
        upper + _TOPOLOGY_EPS_UM
    )


def _clean_loop(loop: Any) -> tuple[tuple[float, float], ...]:
    points = tuple(
        _coordinate_2d_key((float(point[0]), float(point[1]))) for point in loop
    )
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    points = _drop_near_duplicate_loop_points(points)
    points = _drop_redundant_collinear_loop_points(points)
    if len(points) < 3:
        raise ValueError("loop requires at least 3 unique points")
    if abs(_polygon_area(points)) <= _TOPOLOGY_EPS_UM:
        raise ValueError("loop area is below topology tolerance")
    return points


def _drop_near_duplicate_loop_points(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if cleaned and cleaned[-1] == point:
            continue
        cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return tuple(cleaned)


def _drop_redundant_collinear_loop_points(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    points = _drop_near_duplicate_loop_points(points)
    if len(points) <= 3:
        return points

    protected = _duplicate_edge_point_indices(points)
    changed = True
    while changed and len(points) > 3:
        changed = False
        for index, point in enumerate(points):
            if index in protected:
                continue
            previous = points[index - 1]
            following = points[(index + 1) % len(points)]
            if _point_on_segment_2d(point, previous, following):
                points = points[:index] + points[index + 1 :]
                protected = _duplicate_edge_point_indices(points)
                changed = True
                break
    return points


def _duplicate_edge_point_indices(
    points: tuple[tuple[float, float], ...],
) -> set[int]:
    edge_counts = Counter(
        tuple(sorted((start, end))) for start, end in _ring_edges(points)
    )
    protected: set[int] = set()
    for index, edge in enumerate(_ring_edges(points)):
        if edge_counts[tuple(sorted(edge))] > 1:
            protected.update((index, (index + 1) % len(points)))
    return protected


def _point_on_segment_2d(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= _TOPOLOGY_EPS_UM * _TOPOLOGY_EPS_UM:
        return False
    cross = (point[0] - start[0]) * dy - (point[1] - start[1]) * dx
    if abs(cross) > _TOPOLOGY_EPS_UM * sqrt(length_sq):
        return False
    dot = (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    return _TOPOLOGY_EPS_UM < dot < length_sq - _TOPOLOGY_EPS_UM


def _ring_edges(
    ring: tuple[tuple[float, float], ...],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return tuple(
        (ring[index], ring[(index + 1) % len(ring)]) for index in range(len(ring))
    )


def _gdstk_surface_region(geometry_ref: Mapping[str, Any]) -> tuple[Any, ...]:
    import gdstk

    outer = gdstk.Polygon(_clean_loop(geometry_ref["outer_loop"]))
    holes = tuple(
        gdstk.Polygon(_clean_loop(hole_loop))
        for hole_loop in geometry_ref.get("hole_loops", ())
    )
    if not holes:
        return (outer,)
    return _boolean_gdstk_region(gdstk, (outer,), holes, "not")


def _boolean_gdstk_region(
    gdstk: Any,
    left: Sequence[Any],
    right: Sequence[Any],
    operation: str,
) -> tuple[Any, ...]:
    if not left:
        return ()
    if operation == "not" and not right:
        return _filter_gdstk_polygons(left)
    result = gdstk.boolean(
        left,
        right,
        operation,
        precision=1e-9,
    )
    return _filter_gdstk_polygons(result or ())


def _filter_gdstk_polygons(polygons: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(
        polygon
        for polygon in polygons
        if abs(_polygon_area(polygon.points)) > _TOPOLOGY_EPS_UM
    )


def _geometry_ref_from_gdstk_polygon(
    parent_geometry_ref: Mapping[str, Any],
    polygon: Any,
) -> dict[str, Any]:
    geometry_ref = dict(parent_geometry_ref)
    outer_loop, hole_loops = _split_gdstk_cutline_loop(_clean_loop(polygon.points))
    geometry_ref["outer_loop"] = outer_loop
    geometry_ref["hole_loops"] = hole_loops
    return geometry_ref


def _split_gdstk_cutline_loop(
    loop: tuple[tuple[float, float], ...],
) -> tuple[
    tuple[tuple[float, float], ...],
    tuple[tuple[tuple[float, float], ...], ...],
]:
    """Normalize gdstk's cutline-encoded boundary into simple outer/hole loops.

    Gdstk represents a polygon with holes as one walk with paired, opposite
    cutline segments.  Atomic splitting and cancellation leaves exactly the
    independent boundary cycles; that representation is suitable for both the
    adapter's fused selector components and Boolean residual lowering.
    """
    atomic_edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for start, end in _ring_edges(loop):
        points = sorted(
            {point for point in loop if _point_on_planar_segment(point, start, end)},
            key=lambda point: _planar_segment_parameter(point, start, end),
        )
        atomic_edges.extend((left, right) for left, right in pairwise(points))
    loops = _simple_planar_loops_from_edges(_cancel_reversed_planar_edges(atomic_edges))
    if not loops:
        raise ValueError("gdstk cutline polygon has no simple boundary loop")
    outer_index = max(
        range(len(loops)), key=lambda index: abs(_polygon_area(loops[index]))
    )
    outer = loops[outer_index]
    holes = tuple(
        sorted(
            (loop_ for index, loop_ in enumerate(loops) if index != outer_index),
            key=lambda loop_: (
                min(point[0] for point in loop_),
                min(point[1] for point in loop_),
            ),
        )
    )
    if any(not _loop_inside_loop(hole, outer) for hole in holes):
        raise ValueError("gdstk cutline polygon has a non-interior boundary loop")
    return outer, holes


def _point_on_planar_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    dx, dy = end[0] - start[0], end[1] - start[1]
    if abs((point[0] - start[0]) * dy - (point[1] - start[1]) * dx) > _TOPOLOGY_EPS_UM:
        return False
    return (
        min(start[0], end[0]) - _TOPOLOGY_EPS_UM
        <= point[0]
        <= max(start[0], end[0]) + _TOPOLOGY_EPS_UM
        and min(start[1], end[1]) - _TOPOLOGY_EPS_UM
        <= point[1]
        <= max(start[1], end[1]) + _TOPOLOGY_EPS_UM
    )


def _planar_segment_parameter(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    return ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (
        dx * dx + dy * dy
    )


def _cancel_reversed_planar_edges(
    edges: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    counts: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
    for edge in edges:
        counts[edge] = counts.get(edge, 0) + 1
    for edge in tuple(counts):
        reverse = (edge[1], edge[0])
        cancelled = min(counts.get(edge, 0), counts.get(reverse, 0))
        if cancelled:
            counts[edge] -= cancelled
            counts[reverse] -= cancelled
    return tuple(
        edge
        for edge, count in counts.items()
        for _ in range(count)
        if edge[0] != edge[1]
    )


def _simple_planar_loops_from_edges(
    edges: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    neighbors: dict[tuple[float, float], set[tuple[float, float]]] = {}
    unused: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for start, end in edges:
        edge_key = tuple(sorted((start, end)))
        if edge_key in unused:
            raise ValueError("gdstk cutline repeats a retained boundary edge")
        unused.add(edge_key)
        neighbors.setdefault(start, set()).add(end)
        neighbors.setdefault(end, set()).add(start)
    if any(len(points) != 2 for points in neighbors.values()):
        raise ValueError("gdstk cutline boundary is not a simple loop")
    loops: list[tuple[tuple[float, float], ...]] = []
    while unused:
        start, current = next(iter(unused))
        loop = [start]
        previous = start
        while current != start:
            loop.append(current)
            choices = neighbors[current] - {previous}
            if len(choices) != 1:
                raise ValueError("gdstk cutline boundary branches")
            following = next(iter(choices))
            unused.discard(tuple(sorted((previous, current))))
            previous, current = current, following
        unused.discard(tuple(sorted((previous, current))))
        loops.append(tuple(loop))
    return tuple(loops)


def _geometry_refs_from_gdstk_region(
    parent_geometry_ref: Mapping[str, Any],
    region: tuple[Any, ...],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        _geometry_ref_from_gdstk_polygon(parent_geometry_ref, polygon)
        for polygon in _filter_gdstk_polygons(region)
    )


def _coordinate_2d_key(point: tuple[float, float]) -> tuple[float, float]:
    return tuple(round(float(value), 9) for value in point)


def _polygon_area(points: Sequence[Sequence[float]]) -> float:
    clean_points = tuple((float(point[0]), float(point[1])) for point in points)
    if len(clean_points) < 3:
        return 0.0
    return 0.5 * sum(
        x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in _ring_edges(clean_points)
    )


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
