"""Planar source preparation and HFSS-native body/selection binding for EPR."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from scgsim.semantics.route_a import (
    apply_thin_film_profile_with_provenance,
    derive_thin_film_facts,
    geometry_z_range,
)
from scgsim.sgb import (
    GeometryBuildInput,
    validate_geometry_input,
    validate_selected_route,
)
from scgsim.sgb.planning import (
    plan_surface_contribution_patches,
    verified_route_a_substrate_support,
)

from ._epr_models import (
    PlanarJunction,
    PreparedPlanarGeometry,
    SurfaceEprSpec,
    canonical_sha256,
    detached,
)
from ._native_common import (
    _native_boundary_type,
    _native_object_boolean_property,
    _native_object_evidence,
    native_object_property,
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"geometry source contains unsupported {type(value).__name__}")


def _native_name(kind: str, *identity: Any) -> str:
    """Return a stable AEDT-safe name while retaining raw ids only as metadata."""

    if not isinstance(kind, str) or not kind.isidentifier():
        raise ValueError("native name kind must be an identifier")
    digest = canonical_sha256({"kind": kind, "identity": list(identity)})[:24]
    return f"scgsim_{kind}_{digest}"


def _source_payload(
    build_input: GeometryBuildInput,
    prepared_stack: Mapping[str, Any],
    *,
    route: str,
) -> dict[str, Any]:
    materials = prepared_stack.get("materials")
    solution_regions = prepared_stack.get("solution_regions")
    layers = prepared_stack.get("layers")
    metadata = prepared_stack.get("metadata", {})
    if (
        not isinstance(materials, Mapping)
        or not isinstance(solution_regions, Mapping)
        or isinstance(layers, (str, bytes))
        or not isinstance(layers, Sequence)
        or not isinstance(metadata, Mapping)
    ):
        raise TypeError(
            "prepared_stack must contain materials, solution_regions, layers, and metadata"
        )
    material_catalog = _plain(materials)
    for entity in build_input.entities:
        material = materials.get(entity.material_id)
        if not isinstance(material, Mapping):
            raise ValueError(
                f"geometry entity {entity.semantic_id!r} material is absent from prepared_stack"
            )
        if material.get("kind") != entity.material_kind:
            raise ValueError(
                f"geometry entity {entity.semantic_id!r} material kind contradicts prepared_stack"
            )
    source_layers = {
        str(record.get("semantic_id")): record
        for record in layers
        if isinstance(record, Mapping) and isinstance(record.get("semantic_id"), str)
    }
    conductors = []
    for item in build_input.entities:
        if item.material_kind != "conductor":
            continue
        source_id = str(item.metadata.get("semantic_group_id", item.semantic_id))
        if source_id not in source_layers:
            raise ValueError(
                f"geometry conductor {item.semantic_id!r} has no prepared-stack source record"
            )
        source_record = source_layers[source_id]
        if source_record.get("material_id") != item.material_id:
            raise ValueError(
                f"geometry conductor {item.semantic_id!r} material contradicts prepared_stack"
            )
        conductors.append(
            {
                "semantic_id": item.semantic_id,
                "source_semantic_id": source_id,
                "role": item.role,
                "material_id": item.material_id,
                "material_kind": item.material_kind,
                "part_role": item.part_role,
                "representation": item.route_representations.get(route),
                "net_id": item.net_id,
                "polygon_ids": list(item.polygon_ids),
                "geometry": _plain(item.geometry),
                "metadata": {
                    key: _plain(item.metadata[key])
                    for key in (
                        "semantic_group_id",
                        "split_polygon_index",
                        "ground_bump_id",
                    )
                    if key in item.metadata
                },
            }
        )
    normalized_regions = []
    for entity in build_input.entities:
        if entity.material_kind not in {"vacuum", "dielectric"}:
            continue
        semantic_id = entity.semantic_id
        record = solution_regions.get(semantic_id)
        if record is not None and not isinstance(record, Mapping):
            raise TypeError("prepared_stack solution regions must be mappings")
        if record is not None and record.get("material_id", semantic_id) != entity.material_id:
            raise ValueError(
                f"solution region {semantic_id!r} material contradicts normalized geometry"
            )
        generated_auto = bool(entity.metadata.get("is_auto_vacuum_region"))
        geometry = (
            entity.geometry
            if record is None or generated_auto
            else record.get("geometry", record)
        )
        geometry_z_range(geometry, semantic_id)
        normalized_regions.append(
            {
                "semantic_id": semantic_id,
                "material_id": entity.material_id,
                "material_kind": entity.material_kind,
                "representation": entity.route_representations.get(route),
                "geometry": _plain(geometry),
                "metadata": _plain(entity.metadata),
            }
        )
    missing_regions = set(solution_regions) - {item["semantic_id"] for item in normalized_regions}
    if missing_regions:
        raise ValueError(
            f"prepared-stack solution regions lack normalized geometry: {sorted(missing_regions)!r}"
        )
    profile = metadata.get("route_a_thin_film")
    if profile is not None and not isinstance(profile, Mapping):
        raise TypeError("prepared_stack route_a_thin_film provenance must be a mapping")
    return {
        "schema_version": "scgsim.aedt.epr-planar-source.v1",
        "prepared_stack_sha256": canonical_sha256(_plain(prepared_stack)),
        "materials": material_catalog,
        "solution_regions": normalized_regions,
        "conductors": conductors,
        "route_a_thin_film": _plain(profile) if profile is not None else None,
        "junction_regions": [
            {
                "port_sheet_id": item.port_sheet_id,
                "source_polygon_id": item.source_polygon_id,
                "source_layer": item.source_layer,
                "exterior": [list(point) for point in item.exterior],
                "holes": [[list(point) for point in ring] for ring in item.holes],
                "host_semantic_ids": list(
                    dict.fromkeys(overlap.host_semantic_id for overlap in item.overlaps)
                ),
                "metadata": _plain(item.metadata),
            }
            for item in build_input.port_sheet_regions
        ],
        "polygons": [
            {
                "polygon_id": item.polygon_id,
                "layer": item.layer,
                "exterior": [list(point) for point in item.exterior],
                "holes": [[list(point) for point in ring] for ring in item.holes],
                "object_name": item.object_name,
                "net_name": item.net_name,
                "port_name": item.port_name,
            }
            for item in build_input.polygons
        ],
    }


def _geometry_with_prepared_z(
    geometry: Mapping[str, Any], prepared: Mapping[str, Any], context: str
) -> dict[str, Any]:
    z_min, z_max = geometry_z_range(prepared.get("geometry", prepared), context)
    result = dict(geometry)
    if "z_min_um" in result or "z_max_um" in result:
        result["z_min_um"], result["z_max_um"] = z_min, z_max
    else:
        result["z_um"], result["thickness_um"] = z_min, z_max - z_min
    return result


def _profiled_geometry_input(
    build_input: GeometryBuildInput, prepared_stack: Mapping[str, Any]
) -> GeometryBuildInput:
    regions = prepared_stack["solution_regions"]
    layers = {
        str(item.get("semantic_id")): item
        for item in prepared_stack["layers"]
        if isinstance(item, Mapping) and isinstance(item.get("semantic_id"), str)
    }
    entities = []
    for entity in build_input.entities:
        if entity.material_kind in {"vacuum", "dielectric"}:
            record = regions.get(entity.semantic_id)
        else:
            source_id = str(entity.metadata.get("semantic_group_id", entity.semantic_id))
            record = layers.get(source_id)
        if not isinstance(record, Mapping):
            raise ValueError(
                f"prepared stack lacks geometry for {entity.semantic_id!r}"
            )
        entities.append(
            replace(
                entity,
                geometry=_geometry_with_prepared_z(
                    entity.geometry, record, entity.semantic_id
                ),
            )
        )
    metadata = prepared_stack.get("metadata", {})
    return replace(
        build_input,
        entities=tuple(entities),
        solution_regions=_plain(regions),
        metadata={**dict(build_input.metadata), **dict(metadata)},
    )


def _prepare_stack_and_geometry(
    build_input: GeometryBuildInput,
    prepared_stack: Mapping[str, Any],
    *,
    route: str,
    route_a_profile: str | None,
) -> tuple[GeometryBuildInput, Mapping[str, Any]]:
    if route == "B":
        if route_a_profile is not None:
            raise ValueError("Route B does not accept route_a_profile")
        return _profiled_geometry_input(build_input, prepared_stack), prepared_stack
    if route_a_profile not in {"substrate_face", "metal_gap_equivalent"}:
        raise ValueError(
            "Route A requires route_a_profile='substrate_face' or "
            "'metal_gap_equivalent'"
        )
    metadata = prepared_stack.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("prepared_stack metadata must be a mapping")
    existing = metadata.get("route_a_thin_film")
    if existing is not None:
        if not isinstance(existing, Mapping) or existing.get("variant") != route_a_profile:
            raise ValueError("prepared_stack Route A profile contradicts request")
        profiled = prepared_stack
    else:
        support = verified_route_a_substrate_support(build_input, prepared_stack)
        facts = derive_thin_film_facts(
            prepared_stack,
            allow_single_face=route_a_profile == "substrate_face",
            substrate_support=support,
        )
        source_digest = canonical_sha256(_plain(prepared_stack))
        profiled = apply_thin_film_profile_with_provenance(
            prepared_stack,
            profile=route_a_profile,
            facts=facts,
            source_revision=f"sha256:{source_digest}",
        )
    return _profiled_geometry_input(build_input, profiled), profiled


def _surface_mask_plane(geometry_ref: Mapping[str, Any]) -> dict[str, Any]:
    outer = geometry_ref.get("outer_loop")
    plane = geometry_ref.get("plane")
    if isinstance(outer, Sequence) and not isinstance(outer, (str, bytes)):
        if not isinstance(plane, Mapping) or plane.get("axis") != "z":
            raise ValueError("planar EPR surface loop requires an explicit Z plane")
        z_um = plane.get("value_um")
        if isinstance(z_um, bool) or not isinstance(z_um, (int, float)):
            raise ValueError("planar EPR surface requires finite plane Z")
        return {
            "origin_um": [0.0, 0.0, float(z_um)],
            "u": [1.0, 0.0, 0.0],
            "v": [0.0, 1.0, 0.0],
            "exterior": _plain(outer),
            "holes": _plain(geometry_ref.get("hole_loops", ())),
        }
    quad = geometry_ref.get("quad_points")
    if isinstance(quad, Sequence) and not isinstance(quad, (str, bytes)):
        points = tuple(tuple(float(value) for value in point) for point in quad)
        if len(points) < 3 or any(len(point) != 3 for point in points):
            raise ValueError("sidewall EPR surface requires at least three 3D points")
        origin = points[0]
        first = next(
            (
                tuple(points[index][axis] - origin[axis] for axis in range(3))
                for index in range(1, len(points))
                if points[index] != origin
            ),
            None,
        )
        if first is None:
            raise ValueError("sidewall EPR surface has no nonzero edge")
        norm = math.sqrt(sum(value * value for value in first))
        u = tuple(value / norm for value in first)
        second = next(
            (
                tuple(point[axis] - origin[axis] for axis in range(3))
                for point in points[1:]
                if math.sqrt(
                    sum(
                        (
                            (point[axis] - origin[axis])
                            - sum(
                                (point[k] - origin[k]) * u[k] for k in range(3)
                            )
                            * u[axis]
                        )
                        ** 2
                        for axis in range(3)
                    )
                )
                > 1e-12
            ),
            None,
        )
        if second is None:
            raise ValueError("sidewall EPR surface points are collinear")
        projection = sum(second[index] * u[index] for index in range(3))
        v_raw = tuple(second[index] - projection * u[index] for index in range(3))
        v_norm = math.sqrt(sum(value * value for value in v_raw))
        v = tuple(value / v_norm for value in v_raw)
        local = []
        for point in points:
            delta = tuple(point[index] - origin[index] for index in range(3))
            local_u = sum(delta[index] * u[index] for index in range(3))
            local_v = sum(delta[index] * v[index] for index in range(3))
            residual = tuple(
                delta[index] - local_u * u[index] - local_v * v[index]
                for index in range(3)
            )
            if math.sqrt(sum(value * value for value in residual)) > 1e-9:
                raise ValueError("sidewall EPR surface points are not coplanar")
            local.append([local_u, local_v])
        return {
            "origin_um": list(origin),
            "u": list(u),
            "v": list(v),
            "exterior": local,
            "holes": [],
        }
    raise ValueError("EPR surface has no supported local planar geometry")


def _plane_group_key(binding: Mapping[str, Any]) -> tuple[Any, ...]:
    plane = binding["mask_plane"]
    u = tuple(float(value) for value in plane["u"])
    v = tuple(float(value) for value in plane["v"])
    normal = (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )
    length = math.sqrt(sum(value * value for value in normal))
    normal = tuple(value / length for value in normal)
    first = next((value for value in normal if abs(value) > 1e-12), 1.0)
    if first < 0.0:
        normal = tuple(-value for value in normal)
    origin = tuple(float(value) for value in plane["origin_um"])
    offset = sum(normal[index] * origin[index] for index in range(3))
    evidence = binding["contribution"]
    return (
        *(round(value, 12) for value in normal),
        round(offset, 12),
        evidence["classification"],
        evidence["side"],
        binding["effective_material_id"],
    )


def _project_plane_region(
    source: Mapping[str, Any], target: Mapping[str, Any]
) -> dict[str, Any]:
    source_origin = tuple(float(value) for value in source["origin_um"])
    source_u = tuple(float(value) for value in source["u"])
    source_v = tuple(float(value) for value in source["v"])
    target_origin = tuple(float(value) for value in target["origin_um"])
    target_u = tuple(float(value) for value in target["u"])
    target_v = tuple(float(value) for value in target["v"])

    def project(point: Sequence[float]) -> list[float]:
        global_point = tuple(
            source_origin[index]
            + float(point[0]) * source_u[index]
            + float(point[1]) * source_v[index]
            for index in range(3)
        )
        delta = tuple(
            global_point[index] - target_origin[index] for index in range(3)
        )
        return [
            sum(delta[index] * target_u[index] for index in range(3)),
            sum(delta[index] * target_v[index] for index in range(3)),
        ]

    return {
        "exterior": [project(point) for point in source["exterior"]],
        "holes": [
            [project(point) for point in ring] for ring in source.get("holes", ())
        ],
    }


def _gdstk_polygons(regions: Sequence[Mapping[str, Any]]) -> list[Any]:
    import gdstk

    polygons: list[Any] = []
    for region in regions:
        current = [gdstk.Polygon(region["exterior"])]
        holes = [gdstk.Polygon(ring) for ring in region.get("holes", ())]
        if holes:
            current = list(gdstk.boolean(current, holes, "not", precision=1e-9))
        polygons.extend(current)
    return polygons


def _cycles_from_gdstk_polygon(
    points: Sequence[Sequence[float]],
) -> list[list[list[float]]]:
    vertices = [tuple(round(float(value), 12) for value in point) for point in points]
    edges: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        key = tuple(sorted((first, second)))
        edges[key] = edges.get(key, 0) + 1
    adjacency: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for (first, second), count in edges.items():
        if count == 2:
            continue
        if count != 1:
            raise RuntimeError("mask Boolean produced ambiguous boundary edges")
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise RuntimeError("mask Boolean produced a non-manifold boundary")
    remaining = {tuple(sorted(edge)) for edge, count in edges.items() if count == 1}
    cycles: list[list[list[float]]] = []
    while remaining:
        start, current = next(iter(remaining))
        previous = start
        cycle = [start]
        remaining.remove(tuple(sorted((start, current))))
        while current != start:
            cycle.append(current)
            next_vertex = next(
                candidate
                for candidate in adjacency[current]
                if candidate != previous
            )
            edge = tuple(sorted((current, next_vertex)))
            if edge not in remaining and next_vertex != start:
                raise RuntimeError("mask Boolean boundary walk is inconsistent")
            remaining.discard(edge)
            previous, current = current, next_vertex
        cycles.append([[value[0], value[1]] for value in cycle])
    return cycles


def _regions_from_gdstk(polygons: Sequence[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for polygon in polygons:
        cycles = _cycles_from_gdstk_polygon(polygon.points)
        cycles.sort(
            key=lambda ring: abs(
                sum(
                    ring[index][0] * ring[(index + 1) % len(ring)][1]
                    - ring[(index + 1) % len(ring)][0] * ring[index][1]
                    for index in range(len(ring))
                )
            ),
            reverse=True,
        )
        result.append({"exterior": cycles[0], "holes": cycles[1:]})
    return result


def _bind_physical_support_masks(
    bindings: list[dict[str, Any]],
    support_bindings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not bindings:
        return []
    import gdstk

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for binding in support_bindings:
        groups.setdefault(_plane_group_key(binding), []).append(binding)
    support_by_group: dict[
        tuple[Any, ...],
        tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]],
    ] = {}
    for key, group in groups.items():
        plane = group[0]["mask_plane"]
        projected = [
            _project_plane_region(item["mask_plane"], plane) for item in group
        ]
        union = list(gdstk.boolean(_gdstk_polygons(projected), [], "or", precision=1e-9))
        if not union:
            raise RuntimeError("physical EPR support union is empty")
        support_by_group[key] = (plane, _regions_from_gdstk(union), group)
    result: list[dict[str, Any]] = []
    for binding in bindings:
        support = support_by_group.get(_plane_group_key(binding))
        if support is None:
            raise RuntimeError("selected EPR surface lacks complete physical support")
        plane, support_regions, group = support
        result.append(
            {
                **binding,
                "mask_plane": {**plane, "exterior": [], "holes": []},
                "mask_support": {
                    "group_identity_sha256": canonical_sha256(
                        {
                            "plane": plane,
                            "classification": binding["contribution"]["classification"],
                            "side": binding["contribution"]["side"],
                            "material_id": binding["effective_material_id"],
                            "members": sorted(item["binding_id"] for item in group),
                        }
                    ),
                    "attribution_regions": [
                        _project_plane_region(binding["mask_plane"], plane)
                    ],
                    # The original union boundary is authoritative.  The
                    # field compiler applies exact finite-segment distance
                    # for each margin; polygon offset geometry would change
                    # concave and hole-corner semantics.
                    "support_regions": support_regions,
                },
            }
        )
    return result


def prepare_planar_geometry_input(
    build_input: GeometryBuildInput,
    *,
    prepared_stack: Mapping[str, Any],
    route: str,
    route_a_profile: str | None = None,
    junctions: Sequence[PlanarJunction] = (),
    contributions: Sequence[SurfaceEprSpec] = (),
) -> PreparedPlanarGeometry:
    """Validate and detach the shared Route A/B source facts used by HFSS EPR."""

    if not isinstance(build_input, GeometryBuildInput):
        raise TypeError("build_input must be GeometryBuildInput")
    if route not in {"A", "B"}:
        raise ValueError("HFSS planar EPR supports only Route A or Route B")
    if not isinstance(prepared_stack, Mapping):
        raise TypeError("prepared_stack must be a mapping")
    validate_geometry_input(build_input)
    build_input, prepared_stack = _prepare_stack_and_geometry(
        build_input,
        prepared_stack,
        route=route,
        route_a_profile=route_a_profile,
    )
    validate_geometry_input(build_input)
    validate_selected_route(build_input, route)  # type: ignore[arg-type]
    junction_tuple = tuple(junctions)
    contribution_tuple = tuple(contributions)
    if any(not isinstance(item, PlanarJunction) for item in junction_tuple):
        raise TypeError("junctions must contain PlanarJunction records")
    if any(not isinstance(item, SurfaceEprSpec) for item in contribution_tuple):
        raise TypeError("contributions must contain SurfaceEprSpec records")

    polygons = {item.polygon_id for item in build_input.polygons}
    junction_region_ids = {
        item.source_polygon_id for item in build_input.port_sheet_regions
    }
    ids = [item.junction_id for item in junction_tuple]
    if len(ids) != len(set(ids)):
        raise ValueError("junction ids must be unique")
    contribution_ids = [item.contribution_id for item in contribution_tuple]
    if len(contribution_ids) != len(set(contribution_ids)):
        raise ValueError("surface contribution ids must be unique")
    for item in junction_tuple:
        if item.source_polygon_id not in polygons | junction_region_ids:
            raise ValueError(
                f"junction {item.junction_id!r} references unknown polygon "
                f"{item.source_polygon_id!r}"
            )
    build_input, planned_surfaces = plan_surface_contribution_patches(
        build_input, route=route  # type: ignore[arg-type]
    )
    surfaces: list[dict[str, Any]] = []
    for surface in planned_surfaces:
        provenance = surface.metadata.get("source_provenance")
        if not isinstance(provenance, Mapping):
            continue
        ledger = provenance.get("surface_contribution_ledger")
        if not isinstance(ledger, Sequence) or isinstance(ledger, (str, bytes)):
            continue
        for record in ledger:
            if not isinstance(record, Mapping):
                raise TypeError("surface contribution ledger entries must be mappings")
            surfaces.append(
                {
                    "surface_id": surface.surface_id,
                    "owner_semantic_id": surface.owner_semantic_id,
                    "geometry_ref": _plain(surface.geometry_ref),
                    "normal_hint": _plain(surface.normal_hint),
                    "surface_role": surface.surface_role,
                    "mask_plane": _surface_mask_plane(surface.geometry_ref),
                    "contribution": _plain(record),
                }
            )
    catalog_by_id: dict[str, dict[str, Any]] = {}
    for item in surfaces:
        evidence = item["contribution"]
        contribution_id = evidence.get("contribution_id")
        classification = evidence.get("classification")
        side = evidence.get("side")
        if (
            not isinstance(contribution_id, str)
            or classification not in {"MA", "MS", "SA", "MM"}
            or side not in {"top", "bottom", "sidewall"}
        ):
            continue
        record = catalog_by_id.setdefault(
            contribution_id,
            {
                "contribution_id": contribution_id,
                "interface_kind": classification,
                "field_side": side,
                "source_polygon_ids": [],
                "effective_domain_ids": [],
                "surface_roles": [],
            },
        )
        if (
            record["interface_kind"] != classification
            or record["field_side"] != side
        ):
            raise RuntimeError(
                f"surface contribution {contribution_id!r} has contradictory topology"
            )
        source_ids = tuple(evidence.get("outer_source_ids") or ()) + tuple(
            evidence.get("hole_source_ids") or ()
        )
        if not source_ids and classification == "SA":
            source_ids = (item["surface_id"],)
        for source_id in source_ids:
            if source_id not in record["source_polygon_ids"]:
                record["source_polygon_ids"].append(source_id)
        for domain_id in evidence.get("effective_domain_ids", ()):
            if domain_id not in record["effective_domain_ids"]:
                record["effective_domain_ids"].append(domain_id)
        if item["surface_role"] not in record["surface_roles"]:
            record["surface_roles"].append(item["surface_role"])
    contribution_catalog = tuple(
        catalog_by_id[key] for key in sorted(catalog_by_id)
    )
    source = _source_payload(build_input, prepared_stack, route=route)
    regions_by_id = {
        item["semantic_id"]: item for item in source["solution_regions"]
    }
    materials = source["materials"]
    support_bindings: list[dict[str, Any]] = []
    requested_support_sides = {
        (item.interface_kind, item.field_side)
        for item in contribution_tuple
    }
    for surface in surfaces:
        evidence = surface["contribution"]
        if (
            evidence.get("classification"), evidence.get("side")
        ) not in requested_support_sides:
            continue
        domain_ids = tuple(evidence.get("effective_domain_ids", ()))
        expected_count = 2 if evidence["classification"] == "SA" else 1
        if len(domain_ids) != expected_count:
            raise ValueError("physical EPR support has inconsistent adjacent domains")
        domains = [regions_by_id.get(domain_id) for domain_id in domain_ids]
        if any(domain is None for domain in domains):
            raise ValueError("physical EPR support domain is absent from prepared_stack")
        if evidence["classification"] == "SA":
            field_domains = [
                domain for domain in domains
                if materials.get(domain["material_id"], {}).get("kind") == "vacuum"
            ]
            if len(field_domains) != 1:
                raise ValueError("physical SA support needs one vacuum field domain")
            field_domain = field_domains[0]
        else:
            field_domain = domains[0]
        support_bindings.append(
            {
                **surface,
                "binding_id": (
                    f"{evidence['contribution_id']}__"
                    f"{canonical_sha256({'surface_id': surface['surface_id'], 'geometry_ref': surface['geometry_ref']})[:16]}"
                ),
                "effective_material_id": field_domain["material_id"],
            }
        )
    resolved_surfaces: list[dict[str, Any]] = []
    for requested in contribution_tuple:
        matches = [
            item
            for item in surfaces
            if item["contribution"].get("contribution_id")
            == requested.contribution_id
        ]
        if not matches:
            raise ValueError(
                f"contribution {requested.contribution_id!r} is not established by "
                "the shared source topology"
            )
        for match in matches:
            evidence = match["contribution"]
            if evidence.get("classification") != requested.interface_kind:
                raise ValueError(
                    f"contribution {requested.contribution_id!r} classification "
                    "contradicts shared source evidence"
                )
            if evidence.get("side") != requested.field_side:
                raise ValueError(
                    f"contribution {requested.contribution_id!r} side contradicts "
                    "shared source evidence"
                )
            source_polygon_ids = tuple(evidence.get("outer_source_ids") or ()) + tuple(
                evidence.get("hole_source_ids") or ()
            )
            if not source_polygon_ids and requested.interface_kind == "SA":
                source_polygon_ids = (match["surface_id"],)
            if requested.source_polygon_id not in source_polygon_ids:
                raise ValueError(
                    f"contribution {requested.contribution_id!r} does not retain "
                    f"source polygon {requested.source_polygon_id!r}"
                )
            effective_domains = tuple(evidence.get("effective_domain_ids", ()))
            expected_domain_count = 2 if requested.interface_kind == "SA" else 1
            if len(effective_domains) != expected_domain_count:
                raise ValueError(
                    f"contribution {requested.contribution_id!r} does not have "
                    f"{expected_domain_count} physical adjacent domain(s)"
                )
            adjacent_domains = [regions_by_id.get(item) for item in effective_domains]
            if any(item is None for item in adjacent_domains):
                raise ValueError(
                    f"contribution {requested.contribution_id!r} adjacent domain "
                    "is absent from prepared_stack"
                )
            if requested.interface_kind == "SA":
                vacuum_domains = [
                    item
                    for item in adjacent_domains
                    if materials.get(item["material_id"], {}).get("kind") == "vacuum"
                ]
                dielectric_domains = [
                    item
                    for item in adjacent_domains
                    if materials.get(item["material_id"], {}).get("kind")
                    == "dielectric"
                ]
                if len(vacuum_domains) != 1 or len(dielectric_domains) != 1:
                    raise ValueError(
                        f"contribution {requested.contribution_id!r} must bind one "
                        "vacuum and one dielectric domain"
                    )
                domain = vacuum_domains[0]
                substrate_domain = dielectric_domains[0]
            else:
                domain = adjacent_domains[0]
                substrate_domain = domain if requested.interface_kind == "MS" else None
            if domain is None:
                raise ValueError(
                    f"contribution {requested.contribution_id!r} field-side domain "
                    "is absent from prepared_stack"
                )
            material = materials.get(domain["material_id"])
            if not isinstance(material, Mapping):
                raise ValueError(
                    f"contribution {requested.contribution_id!r} field-side material "
                    "is absent from prepared_stack"
                )
            expected_kind = {
                "MA": "vacuum",
                "MS": "dielectric",
                "SA": "vacuum",
            }.get(requested.interface_kind)
            if expected_kind is not None and material.get("kind") != expected_kind:
                raise ValueError(
                    f"contribution {requested.contribution_id!r} field-side material "
                    f"must be {expected_kind}"
                )
            substrate_material = (
                materials.get(substrate_domain["material_id"])
                if substrate_domain is not None
                else None
            )
            if requested.interface_kind in {"MS", "SA"}:
                permittivity = (
                    substrate_material.get("permittivity")
                    if isinstance(substrate_material, Mapping)
                    else None
                )
                if (
                    isinstance(permittivity, bool)
                    or not isinstance(permittivity, (int, float))
                    or not math.isfinite(float(permittivity))
                    or float(permittivity) <= 0.0
                ):
                    raise ValueError(
                        f"contribution {requested.contribution_id!r} requires positive "
                        "substrate permittivity from prepared_stack"
                    )
            resolved_surfaces.append(
                {
                    **match,
                    "binding_id": (
                        f"{requested.contribution_id}__"
                        f"{canonical_sha256({'surface_id': match['surface_id'], 'geometry_ref': match['geometry_ref']})[:16]}"
                    ),
                    "effective_domain_id": domain["semantic_id"],
                    "substrate_domain_id": (
                        substrate_domain["semantic_id"]
                        if substrate_domain is not None else None
                    ),
                    "effective_material_id": domain["material_id"],
                    "effective_material": _plain(material),
                    "substrate_material": (
                        _plain(substrate_material)
                        if substrate_material is not None
                        else None
                    ),
                    "margins_um": list(requested.margins_um),
                    "film_thickness_m": requested.film_thickness_m,
                    "film_relative_permittivity": requested.film_relative_permittivity,
                }
            )

    resolved_surfaces = _bind_physical_support_masks(
        resolved_surfaces, support_bindings
    )

    if route == "A":
        profile = source.get("route_a_thin_film")
        if not isinstance(profile, Mapping):
            raise ValueError(
                "Route A planar EPR requires prepared thin-film profile provenance"
            )
        if profile.get("variant") not in {"substrate_face", "metal_gap_equivalent"}:
            raise ValueError("Route A thin-film profile variant is invalid")
    digest_input = {
        "route": route,
        "source": source,
        "junctions": [item.to_payload() for item in junction_tuple],
        "contribution_catalog": contribution_catalog,
        "contributions": [item.to_payload() for item in contribution_tuple],
        "surface_bindings": resolved_surfaces,
    }
    return PreparedPlanarGeometry(
        route=route,  # type: ignore[arg-type]
        source=source,
        junctions=junction_tuple,
        contribution_catalog=contribution_catalog,
        contributions=contribution_tuple,
        surface_bindings=tuple(resolved_surfaces),
        model_sha256=canonical_sha256(
            {
                "route": route,
                "source": source,
                "junctions": [item.to_payload() for item in junction_tuple],
            }
        ),
        source_sha256=canonical_sha256(digest_input),
    )


def _polygon_sheet(app: Any, polygon: Mapping[str, Any], *, name: str, z_um: float) -> Any:
    points = [[float(x), float(y), z_um] for x, y in polygon["exterior"]]
    value = app.modeler.create_polyline(
        points,
        cover_surface=True,
        close_surface=True,
        name=name,
        non_model=False,
    )
    if value is False or value is None:
        raise RuntimeError(f"failed to create planar source polygon {name!r}")
    for index, ring in enumerate(polygon["holes"]):
        hole_name = f"{name}__hole_{index}"
        hole = app.modeler.create_polyline(
            [[float(x), float(y), z_um] for x, y in ring],
            cover_surface=True,
            close_surface=True,
            name=hole_name,
            non_model=False,
        )
        if hole is False or hole is None:
            raise RuntimeError(f"failed to create planar hole {hole_name!r}")
        if not app.modeler.subtract(name, hole_name, keep_originals=False):
            raise RuntimeError(f"failed to subtract planar hole {hole_name!r}")
    return app.modeler[name]


def _swept_z_solid(
    app: Any,
    polygon: Mapping[str, Any],
    *,
    name: str,
    z_min_um: float,
    z_max_um: float,
) -> Any:
    """Sweep a planar loop along explicit global +Z, independent of its normal."""

    if z_max_um <= z_min_um:
        raise ValueError(f"solid {name!r} has empty Z extent")
    sheet = _polygon_sheet(app, polygon, name=name, z_um=z_min_um)
    body = app.modeler.sweep_along_vector(
        sheet.name, ["0um", "0um", f"{z_max_um - z_min_um:.17g}um"]
    )
    if body is False or body is None:
        raise RuntimeError(f"failed to sweep solid {name!r} along positive Z")
    bounds = body.bounding_box
    scale = max(1.0, abs(z_min_um), abs(z_max_um))
    if (
        len(bounds) != 6
        or not math.isclose(float(bounds[2]), z_min_um, rel_tol=0.0, abs_tol=1e-9 * scale)
        or not math.isclose(float(bounds[5]), z_max_um, rel_tol=0.0, abs_tol=1e-9 * scale)
    ):
        raise RuntimeError(f"solid {name!r} native Z range differs from source")
    return body


def _analysis_surface_sheet(
    app: Any, geometry_ref: Mapping[str, Any], *, name: str
) -> tuple[Any, list[list[float]]]:
    outer = geometry_ref.get("outer_loop")
    plane = geometry_ref.get("plane")
    if isinstance(outer, Sequence) and not isinstance(outer, (str, bytes)):
        if not isinstance(plane, Mapping) or plane.get("axis") != "z":
            raise ValueError("analysis surface loop requires an explicit Z plane")
        z_um = float(plane["value_um"])
        points = [[float(x), float(y), z_um] for x, y in outer]
        holes = geometry_ref.get("hole_loops", ())
    else:
        quad = geometry_ref.get("quad_points")
        if not isinstance(quad, Sequence) or isinstance(quad, (str, bytes)):
            raise ValueError("analysis surface requires a loop or sidewall quad")
        points = [[float(value) for value in point] for point in quad]
        holes = ()
    if len(points) < 3 or any(
        len(point) != 3 or not all(math.isfinite(value) for value in point)
        for point in points
    ):
        raise ValueError("analysis surface points are invalid")
    sheet = app.modeler.create_polyline(
        points,
        cover_surface=True,
        close_surface=True,
        name=name,
        non_model=True,
    )
    if sheet is False or sheet is None:
        raise RuntimeError(f"failed to create analysis surface {name!r}")
    for index, ring in enumerate(holes):
        hole_name = f"{name}__hole_{index}"
        hole = app.modeler.create_polyline(
            [[float(x), float(y), points[0][2]] for x, y in ring],
            cover_surface=True,
            close_surface=True,
            name=hole_name,
            non_model=True,
        )
        if hole is False or hole is None:
            raise RuntimeError(f"failed to create analysis hole {hole_name!r}")
        if not app.modeler.subtract(name, hole_name, keep_originals=False):
            raise RuntimeError(f"failed to subtract analysis hole {hole_name!r}")
    return app.modeler[name], points


def _domain_center(source: Mapping[str, Any], semantic_id: str) -> tuple[float, float, float]:
    matches = [
        item
        for item in source["solution_regions"]
        if item["semantic_id"] == semantic_id
    ]
    if len(matches) != 1:
        raise ValueError(f"effective domain {semantic_id!r} is not unique")
    geometry = matches[0]["geometry"]
    bounds = geometry.get("domain_bounds_um")
    if not isinstance(bounds, Mapping):
        outer = geometry.get("outer_loop")
        if not isinstance(outer, Sequence) or isinstance(outer, (str, bytes)):
            raise ValueError(f"effective domain {semantic_id!r} lacks planar bounds")
        xs = [float(point[0]) for point in outer]
        ys = [float(point[1]) for point in outer]
        x_center = (min(xs) + max(xs)) / 2.0
        y_center = (min(ys) + max(ys)) / 2.0
    else:
        x_center = (float(bounds["x_min_um"]) + float(bounds["x_max_um"])) / 2.0
        y_center = (float(bounds["y_min_um"]) + float(bounds["y_max_um"])) / 2.0
    z_min, z_max = geometry_z_range(geometry, semantic_id)
    return x_center, y_center, (z_min + z_max) / 2.0


def _bounds_box(app: Any, entity: Mapping[str, Any]) -> Any:
    geometry = entity["geometry"]
    bounds = geometry.get("domain_bounds_um")
    if not isinstance(bounds, Mapping):
        raise ValueError(
            f"solution domain {entity['semantic_id']!r} lacks domain_bounds_um"
        )
    required = ("x_min_um", "x_max_um", "y_min_um", "y_max_um")
    if any(key not in bounds for key in required):
        raise ValueError(
            f"solution domain {entity['semantic_id']!r} bounds are incomplete"
        )
    z_min = geometry.get("z_min_um", geometry.get("z_um"))
    z_max = geometry.get("z_max_um")
    if z_max is None and z_min is not None and geometry.get("thickness_um") is not None:
        z_max = float(z_min) + float(geometry["thickness_um"])
    if z_min is None or z_max is None:
        raise ValueError(
            f"solution domain {entity['semantic_id']!r} lacks a finite z range"
        )
    origin = [float(bounds["x_min_um"]), float(bounds["y_min_um"]), float(z_min)]
    size = [
        float(bounds["x_max_um"]) - origin[0],
        float(bounds["y_max_um"]) - origin[1],
        float(z_max) - origin[2],
    ]
    if any(value <= 0.0 for value in size):
        raise ValueError(f"solution domain {entity['semantic_id']!r} has empty bounds")
    native_material = str(entity["material_id"])
    value = app.modeler.create_box(
        origin,
        size,
        name=_native_name("domain", entity["semantic_id"]),
        material=native_material,
    )
    if value is False or value is None:
        raise RuntimeError(f"failed to create solution domain {entity['semantic_id']!r}")
    return value


def _solution_body(app: Any, entity: Mapping[str, Any]) -> Any:
    geometry = entity["geometry"]
    if "outer_loop" not in geometry:
        return _bounds_box(app, entity)
    z_min, z_max = geometry_z_range(geometry, entity["semantic_id"])
    if z_max <= z_min:
        raise ValueError(
            f"solution domain {entity['semantic_id']!r} has empty Z extent"
        )
    polygon = {
        "exterior": geometry["outer_loop"],
        "holes": geometry.get("hole_loops", ()),
    }
    name = _native_name("domain", entity["semantic_id"])
    body = _swept_z_solid(
        app, polygon, name=name, z_min_um=z_min, z_max_um=z_max
    )
    body.material_name = str(entity["material_id"])
    return body


def _material_readback(app: Any, materials: Mapping[str, Any]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for material_id, record in materials.items():
        if not isinstance(record, Mapping) or record.get("kind") not in {
            "vacuum",
            "dielectric",
        }:
            continue
        material = app.materials.exists_material(material_id)
        if material is False or material is None:
            raise RuntimeError(f"native material {material_id!r} is unavailable")
        observed[material_id] = {
            "relative_permittivity": float(material.permittivity.value),
            "relative_permeability": float(material.permeability.value),
        }
    return observed


def _install_material_catalog(app: Any, materials: Mapping[str, Any]) -> dict[str, Any]:
    for material_id, record in materials.items():
        if not isinstance(material_id, str) or not isinstance(record, Mapping):
            raise TypeError("prepared material catalog must be string-keyed mappings")
        kind = record.get("kind")
        if kind == "conductor":
            continue
        if kind not in {"vacuum", "dielectric"}:
            raise ValueError(f"unsupported prepared material kind {kind!r}")
        permittivity = record.get("permittivity", 1.0 if kind == "vacuum" else None)
        if (
            isinstance(permittivity, bool)
            or not isinstance(permittivity, (int, float))
            or not math.isfinite(float(permittivity))
            or float(permittivity) <= 0.0
        ):
            raise ValueError(
                f"prepared material {material_id!r} requires positive permittivity"
            )
        properties: dict[str, float] = {"permittivity": float(permittivity)}
        loss = record.get("loss_tangent")
        if loss is not None:
            if (
                isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(float(loss))
                or float(loss) < 0.0
            ):
                raise ValueError(
                    f"prepared material {material_id!r} has invalid loss tangent"
                )
            properties["dielectric_loss_tangent"] = float(loss)
        observed = app.materials.exists_material(material_id)
        material = observed or app.materials.add_material(material_id, properties)
        if material is False or material is None:
            raise RuntimeError(f"failed to define native material {material_id!r}")
        native_permittivity = float(material.permittivity.value)
        if not math.isclose(
            native_permittivity, float(permittivity), rel_tol=0.0, abs_tol=0.0
        ):
            raise RuntimeError(
                f"native material {material_id!r} permittivity readback differs"
            )
    return _material_readback(app, materials)


def _route_a_sheet_z(source: Mapping[str, Any], entity: Mapping[str, Any]) -> float:
    profile = source.get("route_a_thin_film")
    if not isinstance(profile, Mapping):
        raise ValueError("Route A conductor lacks thin-film profile provenance")
    ranges = profile.get("physical_face_metal_z_ranges_um")
    positions = profile.get("effective_sheet_z_um")
    if not isinstance(ranges, Mapping) or not isinstance(positions, Mapping):
        raise ValueError("Route A thin-film profile is incomplete")
    source_id = entity["source_semantic_id"]
    sides = tuple(
        str(side)
        for side, record in ranges.items()
        if isinstance(record, Mapping) and source_id in record.get("semantic_ids", ())
    )
    if len(sides) != 1:
        raise ValueError(
            f"Route A conductor {entity['semantic_id']!r} has no unique sheet side"
        )
    value = positions.get(sides[0])
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(
            f"Route A conductor {entity['semantic_id']!r} has no finite effective sheet Z"
        )
    return float(value)


def _entity_z_range(entity: Mapping[str, Any]) -> tuple[float, float]:
    return geometry_z_range(entity.get("geometry"), str(entity.get("semantic_id")))


def _junction_terminal_line(
    polygon: Mapping[str, Any], junction: PlanarJunction, z_um: float
) -> tuple[list[list[float]], float]:
    exterior = tuple(
        (float(point[0]), float(point[1])) for point in polygon["exterior"]
    )
    if len(exterior) != 4 or polygon.get("holes"):
        raise ValueError(
            f"junction {junction.junction_id!r} requires one rectangular polygon without holes"
        )
    dx, dy = junction.direction_xy
    px, py = -dy, dx
    along = tuple(x * dx + y * dy for x, y in exterior)
    across = tuple(x * px + y * py for x, y in exterior)
    along_min, along_max = min(along), max(along)
    across_min, across_max = min(across), max(across)
    tolerance = 1e-9 * max(1.0, abs(along_min), abs(along_max), abs(across_min), abs(across_max))
    if along_max - along_min <= tolerance or across_max - across_min <= tolerance:
        raise ValueError(f"junction {junction.junction_id!r} polygon has empty projected extent")
    expected = {
        (along_min, across_min),
        (along_min, across_max),
        (along_max, across_min),
        (along_max, across_max),
    }
    if any(
        not any(abs(u - eu) <= tolerance and abs(v - ev) <= tolerance for eu, ev in expected)
        for u, v in zip(along, across)
    ):
        raise ValueError(
            f"junction {junction.junction_id!r} polygon is not rectangular in its declared direction"
        )
    width = across_max - across_min
    if not math.isclose(width, junction.width_um, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"junction {junction.junction_id!r} width disagrees with its polygon"
        )
    across_mid = (across_min + across_max) / 2.0
    line = [
        [along_min * dx + across_mid * px, along_min * dy + across_mid * py, z_um],
        [along_max * dx + across_mid * px, along_max * dy + across_mid * py, z_um],
    ]
    return line, along_max - along_min


def _point_on_segment(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> bool:
    scale = max(1.0, *(abs(value) for value in (*point, *start, *end)))
    tolerance = 1e-9 * scale
    cross = (point[0] - start[0]) * (end[1] - start[1]) - (
        point[1] - start[1]
    ) * (end[0] - start[0])
    if abs(cross) > tolerance:
        return False
    return (
        min(start[0], end[0]) - tolerance
        <= point[0]
        <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance
        <= point[1]
        <= max(start[1], end[1]) + tolerance
    )


def _assign_closed_enclosure(
    app: Any, source: Mapping[str, Any], bindings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    auto_regions = [
        item
        for item in source["solution_regions"]
        if item["material_kind"] == "vacuum"
        and bool(item.get("metadata", {}).get("is_auto_vacuum_region"))
    ]
    loops = {
        tuple(tuple(float(value) for value in point) for point in item["metadata"].get(
            "auto_vacuum_envelope_outer_loop", ()
        ))
        for item in auto_regions
    }
    if len(loops) != 1 or len(next(iter(loops), ())) < 3:
        raise ValueError(
            "closed EPR model requires one explicit auto-vacuum enclosure loop"
        )
    loop = next(iter(loops))
    segments = tuple(zip(loop, (*loop[1:], loop[0])))
    z_ranges = [
        geometry_z_range(item["geometry"], item["semantic_id"])
        for item in source["solution_regions"]
    ]
    z_min = min(item[0] for item in z_ranges)
    z_max = max(item[1] for item in z_ranges)
    face_records: list[dict[str, Any]] = []
    for binding in bindings:
        if binding["kind"] != "solution_domain":
            continue
        obj = app.modeler.get_object_from_name(binding["object_name"])
        if obj is None:
            raise RuntimeError("native enclosure solution object is unavailable")
        for face in obj.faces:
            center = tuple(float(value) for value in face.center)
            scale = max(1.0, *(abs(value) for value in center), abs(z_min), abs(z_max))
            tolerance = 1e-9 * scale
            plane = None
            if math.isclose(center[2], z_min, rel_tol=0.0, abs_tol=tolerance):
                plane = "bottom"
            elif math.isclose(center[2], z_max, rel_tol=0.0, abs_tol=tolerance):
                plane = "top"
            elif any(_point_on_segment(center[:2], start, end) for start, end in segments):
                plane = "side"
            if plane is not None:
                face_records.append(
                    {
                        "face_id": int(face.id),
                        "object_name": binding["object_name"],
                        "center_um": list(center),
                        "enclosure_plane": plane,
                    }
                )
    planes = {item["enclosure_plane"] for item in face_records}
    if not face_records or planes != {"bottom", "top", "side"}:
        raise RuntimeError("native closed-enclosure face selection is incomplete")
    face_ids = [item["face_id"] for item in face_records]
    if len(face_ids) != len(set(face_ids)):
        raise RuntimeError("native closed-enclosure face selection repeats a face")
    boundary_name = "SCGSimClosedEnclosure"
    boundary = app.assign_perfect_e(face_ids, name=boundary_name)
    if boundary is False or boundary is None:
        raise RuntimeError("native closed-enclosure Perfect E assignment failed")
    if _native_boundary_type(app, boundary_name) != "Perfect E":
        raise RuntimeError("native closed-enclosure boundary readback differs")
    return {
        "boundary_name": boundary_name,
        "native_boundary_type": "Perfect E",
        "face_records": face_records,
        "envelope_outer_loop_um": [list(point) for point in loop],
        "z_range_um": [z_min, z_max],
    }


def prepare_native_planar_geometry(app: Any, prepared: PreparedPlanarGeometry) -> dict[str, Any]:
    """Create and read back the body-first HFSS geometry for one prepared request."""

    if not isinstance(prepared, PreparedPlanarGeometry):
        raise TypeError("prepared must be PreparedPlanarGeometry")
    source = detached(prepared.source)
    polygons = {item["polygon_id"]: item for item in source["polygons"]}
    junction_regions = {
        item["source_polygon_id"]: item for item in source["junction_regions"]
    }
    polygons.update(
        {
            source_id: {
                "polygon_id": source_id,
                "layer": item["source_layer"],
                "exterior": item["exterior"],
                "holes": item["holes"],
                "object_name": None,
                "net_name": None,
                "port_name": item["port_sheet_id"],
            }
            for source_id, item in junction_regions.items()
        }
    )
    entities = source["conductors"]
    bindings: list[dict[str, Any]] = []

    material_readback = _install_material_catalog(app, source["materials"])

    for entity in source["solution_regions"]:
        obj = _solution_body(app, entity)
        evidence = _native_object_evidence(app, obj.name)
        if evidence["native_object_type"] != "Solid":
            raise RuntimeError(f"solution domain {entity['semantic_id']!r} is not solid")
        bindings.append(
            {
                "kind": "solution_domain",
                "semantic_id": entity["semantic_id"],
                "material_id": entity["material_id"],
                "object_name": obj.name,
                **evidence,
            }
        )

    enclosure = _assign_closed_enclosure(app, source, bindings)

    junction_polygons = {item.source_polygon_id for item in prepared.junctions}
    for entity in entities:
        z_min_um, z_max_um = _entity_z_range(entity)
        is_route_a_sheet = entity["representation"] == "surface_sheet"
        z_um = _route_a_sheet_z(source, entity) if is_route_a_sheet else z_min_um
        thickness_um = z_max_um - z_min_um
        for polygon_id in entity["polygon_ids"]:
            if polygon_id in junction_polygons:
                continue
            name = _native_name("conductor", entity["semantic_id"], polygon_id)
            boundary_name: str | None = None
            if is_route_a_sheet:
                obj = _polygon_sheet(app, polygons[polygon_id], name=name, z_um=z_um)
                boundary_name = _native_name(
                    "pec_boundary", entity["semantic_id"], polygon_id
                )
                boundary = app.assign_perfect_e(obj.name, name=boundary_name)
                if boundary is False or boundary is None:
                    raise RuntimeError(f"failed to assign Perfect E to {obj.name!r}")
                if _native_boundary_type(app, boundary_name) != "Perfect E":
                    raise RuntimeError(f"Perfect E readback failed for {obj.name!r}")
            else:
                if thickness_um <= 0.0:
                    raise ValueError(
                        f"finite conductor {entity['semantic_id']!r} requires positive thickness"
                    )
                obj = _swept_z_solid(
                    app,
                    polygons[polygon_id],
                    name=name,
                    z_min_um=z_min_um,
                    z_max_um=z_max_um,
                )
                obj.material_name = "pec"
                obj.solve_inside = False
            evidence = _native_object_evidence(app, obj.name)
            expected_type = "Sheet" if is_route_a_sheet else "Solid"
            if evidence["native_object_type"] != expected_type:
                raise RuntimeError(f"native conductor type mismatch for {obj.name!r}")
            observed: dict[str, Any] = {}
            if not is_route_a_sheet:
                observed = {
                    "native_material_name": native_object_property(
                        obj, "Material"
                    ).strip('"'),
                    "native_solve_inside": _native_object_boolean_property(
                        obj, "Solve Inside"
                    ),
                }
                if (
                    observed["native_material_name"].casefold() != "pec"
                    or observed["native_solve_inside"] is not False
                ):
                    raise RuntimeError(
                        f"native finite PEC readback mismatch for {obj.name!r}"
                    )
            bindings.append(
                {
                    "kind": "conductor",
                    "semantic_id": entity["semantic_id"],
                    "source_polygon_id": polygon_id,
                    "route": prepared.route,
                    "object_name": obj.name,
                    "boundary_name": boundary_name,
                    "observed": observed,
                    **evidence,
                }
            )

    junction_bindings: list[dict[str, Any]] = []
    for junction in prepared.junctions:
        polygon = polygons[junction.source_polygon_id]
        region = junction_regions.get(junction.source_polygon_id)
        if region is None:
            owner_ids = [
                item["semantic_id"]
                for item in entities
                if junction.source_polygon_id in item["polygon_ids"]
            ]
        else:
            owner_ids = list(region["host_semantic_ids"])
        owners = [item for item in entities if item["semantic_id"] in owner_ids]
        if not owners or {item["net_id"] for item in owners} != {
            junction.terminal_a_net,
            junction.terminal_b_net,
        }:
            raise RuntimeError(
                f"junction {junction.junction_id!r} source owners do not match its nets"
            )
        z_values = [
            (
                _route_a_sheet_z(source, owner)
                if owner["representation"] == "surface_sheet"
                else _entity_z_range(owner)[0]
            )
            for owner in owners
        ]
        if max(z_values) - min(z_values) > 1e-9:
            raise RuntimeError(f"junction {junction.junction_id!r} owners are not coplanar")
        z_um = z_values[0]
        name = _native_name("junction", junction.junction_id)
        sheet = _polygon_sheet(app, polygon, name=name, z_um=z_um)
        line, span_um = _junction_terminal_line(polygon, junction, z_um)
        boundary_name = _native_name("junction_rlc", junction.junction_id)
        boundary = app.assign_lumped_rlc_to_sheet(
            sheet.name,
            line,
            name=boundary_name,
            rlc_type="Parallel",
            inductance=junction.inductance_h,
            capacitance=(junction.capacitance_f or None),
        )
        if boundary is False or boundary is None:
            raise RuntimeError(f"failed to assign junction {junction.junction_id!r}")
        evidence = _native_object_evidence(app, sheet.name)
        junction_bindings.append(
            {
                "junction_id": junction.junction_id,
                "source_polygon_id": junction.source_polygon_id,
                "object_name": sheet.name,
                "boundary_name": boundary_name,
                "terminal_a_net": junction.terminal_a_net,
                "terminal_b_net": junction.terminal_b_net,
                "integration_line_um": line,
                "terminal_span_um": span_um,
                "width_um": junction.width_um,
                "inductance_h": junction.inductance_h,
                "capacitance_f": junction.capacitance_f,
                **evidence,
            }
        )

    surface_selections: list[dict[str, Any]] = []
    for binding in prepared.surface_bindings:
        binding_id = str(binding["binding_id"])
        name = _native_name("epr_surface", binding_id)
        sheet, points = _analysis_surface_sheet(
            app, binding["geometry_ref"], name=name
        )
        native = _native_object_evidence(app, sheet.name)
        if native["native_object_type"] != "Sheet" or len(sheet.faces) != 1:
            raise RuntimeError(f"analysis surface {name!r} is not one native sheet")
        normal = sheet.faces[0].normal
        if (
            normal is None
            or len(normal) != 3
            or not all(math.isfinite(float(value)) for value in normal)
        ):
            raise RuntimeError(f"analysis surface {name!r} normal is unavailable")
        unit_normal = tuple(float(value) for value in normal)
        centroid = tuple(
            sum(point[axis] for point in points) / len(points) for axis in range(3)
        )
        field_side = str(binding["contribution"]["side"])
        if field_side == "top":
            desired_normal = (0.0, 0.0, 1.0)
        elif field_side == "bottom":
            desired_normal = (0.0, 0.0, -1.0)
        elif field_side == "sidewall":
            hint = binding.get("normal_hint")
            if (
                not isinstance(hint, Sequence)
                or isinstance(hint, (str, bytes))
                or len(hint) != 3
            ):
                raise RuntimeError(
                    f"analysis surface {name!r} lacks an explicit sidewall normal"
                )
            length = math.sqrt(sum(float(value) ** 2 for value in hint))
            if length == 0.0 or not math.isfinite(length):
                raise RuntimeError(
                    f"analysis surface {name!r} has an invalid sidewall normal"
                )
            desired_normal = tuple(float(value) / length for value in hint)
        else:
            raise RuntimeError(f"analysis surface {name!r} has unknown field side")
        orientation = sum(
            unit_normal[axis] * desired_normal[axis] for axis in range(3)
        )
        if abs(orientation) <= 1e-12:
            raise RuntimeError(
                f"analysis surface {name!r} has ambiguous effective-domain side"
            )
        surface_selections.append(
            {
                "binding_id": binding_id,
                "contribution_id": binding["contribution"]["contribution_id"],
                "selection_name": sheet.name,
                "effective_domain_id": binding["effective_domain_id"],
                "adjacent_side": orientation < 0.0,
                "field_side": field_side,
                "native_normal": list(unit_normal),
                "selection_centroid_um": list(centroid),
                "desired_field_side_normal": list(desired_normal),
                **native,
            }
        )

    object_names = [item["object_name"] for item in bindings] + [
        item["object_name"] for item in junction_bindings
    ]
    if len(object_names) != len(set(object_names)):
        raise RuntimeError("native planar object names are not unique")
    return {
        "schema_version": "scgsim.aedt.epr-native-binding.v1",
        "route": prepared.route,
        "source_sha256": prepared.source_sha256,
        "objects": bindings,
        "junctions": junction_bindings,
        "surface_selections": surface_selections,
        "material_readback": material_readback,
        "closed_enclosure": enclosure,
    }


def bind_saved_planar_geometry(app: Any, prepared: PreparedPlanarGeometry) -> dict[str, Any]:
    """Bind deterministic native objects in a saved project without creating CAD."""

    if not isinstance(prepared, PreparedPlanarGeometry):
        raise TypeError("prepared must be PreparedPlanarGeometry")
    source = detached(prepared.source)
    objects: list[dict[str, Any]] = []
    for domain in source["solution_regions"]:
        name = _native_name("domain", domain["semantic_id"])
        evidence = _native_object_evidence(app, name)
        if evidence["native_object_type"] != "Solid":
            raise RuntimeError(f"saved solution domain {name!r} is not solid")
        objects.append(
            {
                "kind": "solution_domain",
                "semantic_id": domain["semantic_id"],
                "material_id": domain["material_id"],
                "object_name": name,
                **evidence,
            }
        )

    junctions: list[dict[str, Any]] = []
    for junction in prepared.junctions:
        name = _native_name("junction", junction.junction_id)
        evidence = _native_object_evidence(app, name)
        if evidence["native_object_type"] != "Sheet":
            raise RuntimeError(f"saved junction {name!r} is not a sheet")
        junctions.append(
            {
                "junction_id": junction.junction_id,
                "source_polygon_id": junction.source_polygon_id,
                "object_name": name,
                **evidence,
            }
        )

    selections: list[dict[str, Any]] = []
    for binding in prepared.surface_bindings:
        binding_id = str(binding["binding_id"])
        name = _native_name("epr_surface", binding_id)
        obj = app.modeler.get_object_from_name(name)
        if obj is None or len(obj.faces) != 1:
            raise RuntimeError(f"saved analysis surface {name!r} is unavailable")
        evidence = _native_object_evidence(app, name)
        if evidence["native_object_type"] != "Sheet":
            raise RuntimeError(f"saved analysis surface {name!r} is not a sheet")
        normal = obj.faces[0].normal
        if (
            normal is None
            or len(normal) != 3
            or not all(math.isfinite(float(value)) for value in normal)
        ):
            raise RuntimeError(f"saved analysis surface {name!r} normal is unavailable")
        native_normal = tuple(float(value) for value in normal)
        side = str(binding["contribution"]["side"])
        if side == "top":
            desired = (0.0, 0.0, 1.0)
        elif side == "bottom":
            desired = (0.0, 0.0, -1.0)
        elif side == "sidewall":
            hint = binding.get("normal_hint")
            if (
                not isinstance(hint, Sequence)
                or isinstance(hint, (str, bytes))
                or len(hint) != 3
            ):
                raise RuntimeError(f"saved sidewall {name!r} lacks a normal hint")
            length = math.sqrt(sum(float(value) ** 2 for value in hint))
            if not math.isfinite(length) or length == 0.0:
                raise RuntimeError(f"saved sidewall {name!r} normal hint is invalid")
            desired = tuple(float(value) / length for value in hint)
        else:
            raise RuntimeError(f"saved analysis surface {name!r} has unknown side")
        orientation = sum(a * b for a, b in zip(native_normal, desired))
        if abs(orientation) <= 1e-12:
            raise RuntimeError(f"saved analysis surface {name!r} side is ambiguous")
        selections.append(
            {
                "binding_id": binding_id,
                "contribution_id": binding["contribution"]["contribution_id"],
                "selection_name": name,
                "effective_domain_id": binding["effective_domain_id"],
                "adjacent_side": orientation < 0.0,
                "field_side": side,
                "native_normal": list(native_normal),
                "desired_field_side_normal": list(desired),
                **evidence,
            }
        )
    return {
        "schema_version": "scgsim.aedt.epr-native-binding.v1",
        "route": prepared.route,
        "source_sha256": prepared.source_sha256,
        "objects": objects,
        "junctions": junctions,
        "surface_selections": selections,
        "material_readback": _material_readback(app, source["materials"]),
        "closed_enclosure": {
            "boundary_name": "SCGSimClosedEnclosure",
            "native_boundary_type": _native_boundary_type(
                app, "SCGSimClosedEnclosure"
            ),
            "binding_stage": "saved_project_readback_without_cad_creation",
        },
        "binding_stage": "saved_project_readback_without_cad_creation",
    }


__all__ = [
    "bind_saved_planar_geometry",
    "prepare_native_planar_geometry",
    "prepare_planar_geometry_input",
]
