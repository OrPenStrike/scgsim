"""Planar preparation and native binding using the same final source boundaries.

Authored inputs remain immutable. Junction binding includes closed edge contact;
prepared arm subdivisions, PEC ends and central rectangles share model vertices.
Native junction observations use the recorded absolute coordinate comparison
policy without moving CAD, changing prepared geometry or asserting exact contact
below the readback resolution.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from fractions import Fraction
from importlib.metadata import version
from multiprocessing import get_context
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
    surface_evaluations,
)
from ._junction_partition import (
    _closed_contact,
    _cross,
    _inside_ring,
    _section,
    partition_junctions,
)
from ._native_common import (
    _native_boundary_type,
    _native_object_boolean_property,
    _native_object_evidence,
    _resolve_native_assignment,
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


def _native_entity_name(source: Mapping[str, Any], kind: str, *identity: Any) -> str:
    """Keep historical hash names while giving new Plan entities readable names."""

    policy = source.get("native_entity_name_policy", "hash.v1")
    if policy == "hash.v1":
        return _native_name(kind, *identity)
    if policy != "readable.v1":
        raise ValueError("unknown native entity name policy")
    raw = "__".join(str(value) for value in identity)
    label = "".join(
        character if character.isascii() and character.isalnum() else "_"
        for character in raw
    ).strip("_")
    if not label:
        raise ValueError("native entity name has no readable identity")
    digest = canonical_sha256({"kind": kind, "identity": list(identity)})[:10]
    prefix = f"SCGSim_{kind}_"
    label = label[: 60 - len(prefix) - len(digest) - 1].rstrip("_")
    return f"{prefix}{label}_{digest}"


def _sheet_member(binding: Mapping[str, Any], margin_label: str) -> tuple[str, ...]:
    contribution = binding["contribution"]
    return (
        str(binding["owner_semantic_id"]),
        str(binding["surface_role"]),
        str(contribution["classification"]),
        str(contribution["side"]),
        margin_label,
        str(contribution["contribution_id"]),
        str(binding["binding_id"]),
    )


def _sheet_name(
    members: set[tuple[str, ...]], geometry_sha256: str, *, parts: set[int]
) -> str:
    ordered = sorted(members)
    if not ordered:
        raise ValueError("analysis sheet has no physical membership")

    def label(values: set[str]) -> str:
        raw = "_".join(sorted(values))
        safe = "".join(
            character if character.isascii() and character.isalnum() else "_"
            for character in raw
        ).strip("_")
        return safe or "Unknown"

    owners = {member[0] for member in ordered}
    owner = label(owners) if len(ordered) == 1 else (
        f"Shared_{label(owners)}" if len(owners) == 1 else "Shared"
    )
    classification, side, margin = (
        label({member[index] for member in ordered}) for index in (2, 3, 4)
    )
    digest = canonical_sha256(
        {"geometry_sha256": geometry_sha256, "members": ordered, "parts": sorted(parts)}
    )[:12]
    part_label = "" if not parts else "_" + "_".join(
        f"Part{index:02d}" for index in sorted(parts)
    )
    tail = f"_{classification}_{side}_{margin}{part_label}_{digest}"
    owner_budget = 60 - len("EPR_") - len(tail)
    if owner_budget < 1:
        raise ValueError("EPR analysis sheet semantic labels exceed AEDT's 60-character limit")
    return f"EPR_{owner[:owner_budget].rstrip('_')}{tail}"


def _base_sheet_plan(
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_geometry: dict[str, set[tuple[str, ...]]] = {}
    keys: dict[str, str] = {}
    for binding in bindings:
        binding_id = str(binding["binding_id"])
        key = canonical_sha256(binding["geometry_ref"])
        keys[binding_id] = key
        by_geometry.setdefault(key, set()).add(_sheet_member(binding, "Unmasked"))
    names = {key: _sheet_name(members, key, parts=set()) for key, members in by_geometry.items()}
    if len(set(names.values())) != len(names):
        raise RuntimeError("distinct EPR base sheet geometries share a native name")
    return {
        binding_id: {
            "name": names[key],
            "members": sorted(by_geometry[key]),
            "geometry_sha256": key,
        }
        for binding_id, key in keys.items()
    }


def plan_inset_sheet_names(
    bindings: Sequence[Mapping[str, Any]],
    inset_plan: Mapping[tuple[str, float], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Name each actual inset contour from all of its physical memberships."""

    grouped: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        binding_id = str(binding["binding_id"])
        dbu_um = float(binding["mask_support"]["source_dbu_um"])
        for _, margin in surface_evaluations(
            binding["margins_um"], policy="unmasked_plus_requested.v1"
        ):
            key = (binding_id, float(margin))
            if key not in inset_plan:
                continue
            planned = inset_plan[key]
            effective_nm = float(planned["effective_margin_um"]) * 1000.0
            margin_label = (
                "Unmasked" if effective_nm == 0.0
                else f"Margin_{effective_nm:.12g}nm"
            )
            regions = planned["regions"]
            for index, region in enumerate(regions):
                key = canonical_sha256(
                    {
                        "plane": binding["mask_plane"],
                        "region": region,
                        "method": _INSET_METHOD,
                        "source_dbu_um": dbu_um,
                        "klayout_version": version("klayout"),
                    }
                )
                record = grouped.setdefault(key, {"members": set(), "parts": set()})
                record["members"].add(_sheet_member(binding, margin_label))
                if len(regions) > 1:
                    record["parts"].add(index + 1)
    names = {
        key: _sheet_name(record["members"], key, parts=record["parts"])
        for key, record in grouped.items()
    }
    if len(set(names.values())) != len(names):
        raise RuntimeError("distinct EPR inset sheet geometries share a native name")
    return {
        key: {
            "name": names[key],
            "members": sorted(record["members"]),
            "geometry_sha256": key,
        }
        for key, record in grouped.items()
    }


def _source_payload(
    build_input: GeometryBuildInput,
    prepared_stack: Mapping[str, Any],
    *,
    route: str,
    source_dbu_um: float,
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
        source_id = str(
            item.metadata.get("source_semantic_id", item.semantic_id)
            if "split_polygon_index" in item.metadata
            else item.semantic_id
        )
        if "split_polygon_index" in item.metadata and "source_semantic_id" not in item.metadata:
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
                        "source_semantic_id",
                        "source_occurrence_path",
                        "source_local_entity_id",
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
    if any(item["semantic_id"] == "Region" for item in normalized_regions):
        raise ValueError("source solution-domain id 'Region' is reserved for native EPR CAD")
    vacuum_regions = [
        item for item in normalized_regions if item["material_kind"] == "vacuum"
    ]
    auto_vacuum_regions = [
        item for item in vacuum_regions if item["metadata"].get("is_auto_vacuum_region")
    ]
    if not auto_vacuum_regions:
        raise ValueError("EPR Region requires planner-owned auto vacuum components")
    vacuum_material_ids = {item["material_id"] for item in auto_vacuum_regions}
    vacuum_groups = {
        item["metadata"].get("auto_vacuum_group_id") for item in auto_vacuum_regions
    }
    padding_records = {
        canonical_sha256(item["metadata"].get("vacuum_region_padding_um"))
        for item in auto_vacuum_regions
    }
    if (
        len(vacuum_material_ids) != 1
        or len(vacuum_groups) != 1
        or None in vacuum_groups
        or len(padding_records) != 1
    ):
        raise ValueError("EPR auto vacuum components disagree on Region identity")
    padding = auto_vacuum_regions[0]["metadata"].get("vacuum_region_padding_um")
    if not isinstance(padding, Mapping) or set(padding) != {
        "x_plus_um", "x_minus_um", "y_plus_um", "y_minus_um",
        "z_plus_um", "z_minus_um",
    }:
        raise ValueError("EPR Region requires exact six-face source padding")
    padding_um = [
        float(padding[key]) for key in (
            "x_plus_um", "x_minus_um", "y_plus_um", "y_minus_um",
            "z_plus_um", "z_minus_um",
        )
    ]
    if any(not math.isfinite(value) or value < 0.0 for value in padding_um):
        raise ValueError("EPR Region padding must be finite and nonnegative")
    loops = {
        canonical_sha256(item["metadata"].get("auto_vacuum_envelope_outer_loop"))
        for item in auto_vacuum_regions
    }
    if len(loops) != 1:
        raise ValueError("EPR auto vacuum components disagree on envelope")
    envelope_loop = auto_vacuum_regions[0]["metadata"].get(
        "auto_vacuum_envelope_outer_loop"
    )
    if not isinstance(envelope_loop, Sequence) or len(envelope_loop) < 3:
        raise ValueError("EPR Region requires an auto vacuum envelope loop")
    z_ranges = [
        geometry_z_range(item["geometry"], item["semantic_id"])
        for item in auto_vacuum_regions
    ]
    native_region = {
        "method": "single_region_absolute_offset.v1",
        "outer_boundary_policy": "hfss_default.v1",
        "name": "Region",
        "material_id": next(iter(vacuum_material_ids)),
        "logical_vacuum_ids": sorted(item["semantic_id"] for item in auto_vacuum_regions),
        "padding_um": padding_um,
        "envelope_outer_loop_um": _plain(envelope_loop),
        "z_range_um": [min(item[0] for item in z_ranges), max(item[1] for item in z_ranges)],
    }
    profile = metadata.get("route_a_thin_film")
    if profile is not None and not isinstance(profile, Mapping):
        raise TypeError("prepared_stack route_a_thin_film provenance must be a mapping")
    return {
        "schema_version": "scgsim.aedt.epr-planar-source.v1",
        "source_dbu_um": source_dbu_um,
        "prepared_stack_sha256": canonical_sha256(_plain(prepared_stack)),
        "materials": material_catalog,
        "solution_regions": normalized_regions,
        "native_region": native_region,
        "native_entity_name_policy": (
            "readable.v1"
            if metadata.get("component_semantics_schema_version") == 2
            else "hash.v1"
        ),
        "surface_evaluation_policy": "unmasked_plus_requested.v1",
        **(
            {
                "junction_partitions": _plain(
                    build_input.metadata["aedt_junction_partitions"]
                ),
                "authored_input_sha256": build_input.metadata[
                    "aedt_authored_input_sha256"
                ],
            }
            if "aedt_junction_partitions" in build_input.metadata
            else {}
        ),
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
            source_id = str(
                entity.metadata.get("source_semantic_id", entity.semantic_id)
                if "split_polygon_index" in entity.metadata
                else entity.semantic_id
            )
            if "split_polygon_index" in entity.metadata and "source_semantic_id" not in entity.metadata:
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
    raise ValueError("horizontal EPR surface requires a planar loop")


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


_KLAYOUT_COORD_MIN = -(2**31)
_KLAYOUT_COORD_MAX = 2**31 - 1
_INSET_CIRCLE_POINTS = 128
_INSET_METHOD = "klayout_complement_minkowski.v1"


def _klayout_point(point: Sequence[float], dbu_um: float, k: Any) -> Any:
    if len(point) != 2:
        raise ValueError("EPR mask point must have two local coordinates")
    values = [float(value) for value in point]
    if any(
        not math.isfinite(value)
        or not _KLAYOUT_COORD_MIN < value / dbu_um < _KLAYOUT_COORD_MAX
        for value in values
    ):
        raise OverflowError("EPR mask coordinate exceeds KLayout integer range")
    return k.DPoint(*values).to_itype(dbu_um)


def _klayout_region(
    regions: Sequence[Mapping[str, Any]], dbu_um: float, k: Any
) -> Any:
    result = k.Region()
    for region in regions:
        exterior = [_klayout_point(point, dbu_um, k) for point in region["exterior"]]
        polygon = k.Polygon(exterior)
        for hole in region.get("holes", ()):
            polygon.insert_hole(
                [_klayout_point(point, dbu_um, k) for point in hole]
            )
        result.insert(polygon)
    return result.merged()


def _regions_from_klayout(region: Any, dbu_um: float) -> list[dict[str, Any]]:
    # Split point contacts without changing the integer polygon set. Iterate
    # stored pieces so implicit merging cannot recreate a self-touching contour.
    simple = region.merged(True, 1)

    def points(iterator: Any) -> list[list[float]]:
        return [[point.x * dbu_um, point.y * dbu_um] for point in iterator]

    result = [
        {
            "exterior": points(polygon.each_point_hull()),
            "holes": [
                points(polygon.each_point_hole(index))
                for index in range(polygon.holes())
            ],
        }
        for polygon in simple.each()
    ]
    return sorted(result, key=canonical_sha256)


def _klayout_inset(support: Any, radius: int, k: Any) -> Any:
    if radius == 0:
        return support
    bounds = support.bbox()
    padding = 2 * radius + 1
    if any(
        coordinate < _KLAYOUT_COORD_MIN + radius
        or coordinate > _KLAYOUT_COORD_MAX - radius
        for coordinate in (
            bounds.left - padding,
            bounds.bottom - padding,
            bounds.right + padding,
            bounds.top + padding,
        )
    ):
        raise OverflowError("EPR inset complement exceeds KLayout integer range")
    complement = k.Region(
        k.Box(
            bounds.left - padding,
            bounds.bottom - padding,
            bounds.right + padding,
            bounds.top + padding,
        )
    ) - support
    circle = k.Polygon.ellipse(
        k.Box(-radius, -radius, radius, radius), _INSET_CIRCLE_POINTS
    )
    return (support - complement.minkowski_sum(circle).merged()).merged()


def _bind_physical_support_masks(
    bindings: list[dict[str, Any]],
    support_bindings: list[dict[str, Any]],
    *,
    source_dbu_um: float,
) -> list[dict[str, Any]]:
    if not bindings:
        return []
    import klayout.db as k

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
        union = _klayout_region(projected, source_dbu_um, k)
        if union.is_empty():
            raise RuntimeError("physical EPR support union is empty")
        support_by_group[key] = (
            plane, _regions_from_klayout(union, source_dbu_um), group
        )
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
                    # Offset the complete union before intersecting owner
                    # attribution; internal owner seams are not boundaries.
                    "support_regions": support_regions,
                    "source_dbu_um": source_dbu_um,
                    "inset_method": _INSET_METHOD,
                    "circle_points": _INSET_CIRCLE_POINTS,
                    "klayout_version": version("klayout"),
                },
            }
        )
    return result


def inset_surface_regions(
    binding: Mapping[str, Any],
    margin_um: float,
    support_insets: dict[tuple[str, int], Any],
) -> list[dict[str, Any]]:
    """Inset complete physical support, then attribute its surviving pieces."""

    import klayout.db as k

    if not math.isfinite(margin_um) or margin_um < 0.0:
        raise ValueError("surface EPR margin must be finite and nonnegative")
    support = binding["mask_support"]
    if support["klayout_version"] != version("klayout"):
        raise RuntimeError("prepared EPR KLayout version differs from runtime")
    dbu_um = float(support["source_dbu_um"])
    if margin_um / dbu_um >= _KLAYOUT_COORD_MAX:
        raise OverflowError("EPR margin exceeds KLayout integer range")
    radius = k.DPoint(margin_um, 0).to_itype(dbu_um).x
    key = (str(support["group_identity_sha256"]), radius)
    if key not in support_insets:
        physical = _klayout_region(support["support_regions"], dbu_um, k)
        support_insets[key] = _klayout_inset(physical, radius, k)
    attribution = _klayout_region(support["attribution_regions"], dbu_um, k)
    return _regions_from_klayout(support_insets[key] & attribution, dbu_um)


def _inset_group_task(
    task: tuple[dict[str, Any], list[dict[str, Any]]],
) -> dict[tuple[str, float], dict[str, Any]]:
    """Pure integer geometry for one physical support and all its owners/margins."""

    import klayout.db as k

    support, members = task
    if support["klayout_version"] != version("klayout"):
        raise RuntimeError("prepared EPR KLayout version differs from worker runtime")
    dbu_um = float(support["source_dbu_um"])
    physical = _klayout_region(support["support_regions"], dbu_um, k)
    insets: dict[int, Any] = {0: physical}
    result: dict[tuple[str, float], dict[str, Any]] = {}
    for member in members:
        attribution = _klayout_region(member["attribution_regions"], dbu_um, k)
        for requested in member["margins_um"]:
            margin_um = float(requested)
            if not math.isfinite(margin_um) or margin_um < 0:
                raise ValueError("surface EPR margin must be finite and nonnegative")
            if margin_um / dbu_um >= _KLAYOUT_COORD_MAX:
                raise OverflowError("EPR margin exceeds KLayout integer range")
            radius = k.DPoint(margin_um, 0).to_itype(dbu_um).x
            if radius not in insets:
                insets[radius] = _klayout_inset(physical, radius, k)
            regions = _regions_from_klayout(insets[radius] & attribution, dbu_um)
            result[(member["binding_id"], margin_um)] = {
                "regions": regions,
                "requested_margin_um": margin_um,
                "integer_margin_dbu": radius,
                "effective_margin_um": radius * dbu_um,
                "reuse_source_sheet": radius == 0 and len(regions) == 1
                and (insets[radius] & attribution ^ attribution).is_empty(),
            }
    return result


def validate_geometry_workers(geometry_workers: int | None) -> None:
    if geometry_workers is not None and (
        isinstance(geometry_workers, bool)
        or not isinstance(geometry_workers, int)
        or geometry_workers <= 0
    ):
        raise ValueError("geometry_workers must be a positive integer or None")


def precompute_inset_surfaces(
    geometry: PreparedPlanarGeometry,
    request: Any,
    geometry_workers: int | None,
) -> dict[tuple[str, float], dict[str, Any]]:
    """Finish spawned pure mask geometry before an owned Desktop is constructed."""

    validate_geometry_workers(geometry_workers)
    selected = (
        None
        if request.surface_contribution_ids is None
        else set(request.surface_contribution_ids)
    )
    grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for binding in geometry.surface_bindings:
        if binding["contribution"]["classification"] == "MM" or (
            selected is not None
            and binding["contribution"]["contribution_id"] not in selected
        ):
            continue
        support = _plain(binding["mask_support"])
        key = support["group_identity_sha256"]
        if key not in grouped:
            grouped[key] = (support, [])
        grouped[key][1].append(
            {
                "binding_id": binding["binding_id"],
                "attribution_regions": support["attribution_regions"],
                "margins_um": list(dict.fromkeys(
                    margin for _, margin in surface_evaluations(
                        binding["margins_um"],
                        policy=geometry.source.get("surface_evaluation_policy"),
                    )
                )),
            }
        )
    tasks = [grouped[key] for key in sorted(grouped)]
    if not tasks:
        return {}
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    )
    workers = min(geometry_workers or available, len(tasks))
    if workers == 1:
        parts = [_inset_group_task(task) for task in tasks]
    else:
        parts = []
        task_iter = iter(tasks)
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn")
        ) as pool:
            pending = {
                pool.submit(_inset_group_task, task)
                for _, task in zip(range(2 * workers), task_iter)
            }
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    parts.append(future.result())
                    try:
                        pending.add(pool.submit(_inset_group_task, next(task_iter)))
                    except StopIteration:
                        pass
    result: dict[tuple[str, float], dict[str, Any]] = {}
    for part in parts:
        if result.keys() & part.keys():
            raise RuntimeError("EPR inset plan has duplicate binding/margin identity")
        result.update(part)
    return result


def bind_inset_surface_selections(
    app: Any,
    binding: Mapping[str, Any],
    base_selection: Mapping[str, Any],
    margin_um: float,
    inset_plan: Mapping[tuple[str, float], Mapping[str, Any]],
    sheet_facts: dict[str, dict[str, Any]],
    sheet_names: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create or rebind non-model inset sheets without changing solver CAD."""

    plane = binding["mask_plane"]
    desired = tuple(float(value) for value in base_selection["desired_field_side_normal"])
    planned = inset_plan[(str(binding["binding_id"]), margin_um)]
    regions = planned["regions"]
    dbu_um = float(binding["mask_support"]["source_dbu_um"])
    result: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        contour_sha256 = canonical_sha256(
            {
                "plane": plane,
                "region": region,
                "method": _INSET_METHOD,
                "source_dbu_um": dbu_um,
                "klayout_version": version("klayout"),
            }
        )
        planned_name = sheet_names[contour_sha256]
        name = str(planned_name["name"])
        facts = sheet_facts.get(name)
        if facts is None:
            sheet = app.modeler.get_object_from_name(name)
            if sheet is None:
                sheet, _ = _analysis_surface_sheet(
                    app,
                    {"plane_basis": plane, "local_region": region},
                    name=name,
                )
            native = _native_object_evidence(app, name)
            faces = sheet.faces
            if native["native_object_type"] != "Sheet" or len(faces) != 1:
                raise RuntimeError(f"inset EPR selection {name!r} is not one sheet")
            if _native_object_boolean_property(sheet, "Model"):
                raise RuntimeError(f"inset EPR selection {name!r} became model geometry")
            normal = faces[0].normal
            if normal is None or len(normal) != 3 or not all(
                math.isfinite(float(value)) for value in normal
            ):
                raise RuntimeError(f"inset EPR selection {name!r} has no native normal")
            facts = {
                "contour_sha256": contour_sha256,
                "shared_sheet_members": planned_name["members"],
                "native": native,
                "native_normal": [float(value) for value in normal],
            }
            sheet_facts[name] = facts
        if facts["contour_sha256"] != contour_sha256:
            raise RuntimeError(f"inset EPR selection name collision: {name!r}")
        native = facts["native"]
        native_normal = facts["native_normal"]
        orientation = sum(a * b for a, b in zip(native_normal, desired))
        if abs(orientation) <= 1e-12:
            raise RuntimeError(f"inset EPR selection {name!r} has ambiguous field side")
        result.append(
            {
                "selection_name": name,
                "component_index": index,
                "contour_sha256": contour_sha256,
                "contour": region,
                "adjacent_side": orientation < 0.0,
                "native_normal": native_normal,
                "polygon_approximation": {
                    "method": _INSET_METHOD,
                    "source_dbu_um": dbu_um,
                    "points_per_circle": _INSET_CIRCLE_POINTS,
                    "requested_margin_um": margin_um,
                    "integer_margin_dbu": planned["integer_margin_dbu"],
                    "effective_margin_um": planned["effective_margin_um"],
                    "klayout_version": version("klayout"),
                },
                "reused_source_sheet": False,
                **native,
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
    source_dbu_um: float | None = None,
) -> PreparedPlanarGeometry:
    """Validate and detach the shared Route A/B source facts used by HFSS EPR."""

    if not isinstance(build_input, GeometryBuildInput):
        raise TypeError("build_input must be GeometryBuildInput")
    if route not in {"A", "B"}:
        raise ValueError("HFSS planar EPR supports only Route A or Route B")
    if not isinstance(prepared_stack, Mapping):
        raise TypeError("prepared_stack must be a mapping")
    recorded_dbu = build_input.metadata.get("source_dbu_um")
    if source_dbu_um is None:
        source_dbu_um = recorded_dbu
    elif recorded_dbu is not None and source_dbu_um != recorded_dbu:
        raise ValueError("explicit source_dbu_um differs from geometry source DBU")
    if (
        isinstance(source_dbu_um, bool)
        or not isinstance(source_dbu_um, (int, float))
        or not math.isfinite(float(source_dbu_um))
        or source_dbu_um <= 0
    ):
        raise ValueError("planar EPR requires a finite positive source_dbu_um")
    source_dbu_um = float(source_dbu_um)
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
    if any(item.field_side == "sidewall" for item in contribution_tuple):
        raise ValueError(
            "sidewall Surface-EPR is excluded; reprepare with horizontal "
            "top/bottom contributions"
        )

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
        authored = next(
            (p for p in build_input.polygons if p.polygon_id == item.source_polygon_id),
            None,
        )
        if authored is None:
            authored = next(
                r
                for r in build_input.port_sheet_regions
                if r.source_polygon_id == item.source_polygon_id
            )
        _junction_terminal_line(
            {"exterior": authored.exterior, "holes": authored.holes}, item, 0.0
        )
    build_input = partition_junctions(build_input, junction_tuple, source_dbu_um)
    if junction_tuple:
        final_polygons = {p.polygon_id: p for p in build_input.polygons}
        for junction in junction_tuple:
            record = build_input.metadata["aedt_junction_partitions"][junction.junction_id]
            polygon = final_polygons[record["central_polygon_id"]]
            _junction_terminal_line(
                {"exterior": polygon.exterior, "holes": polygon.holes}, junction, 0.0
            )
        partition_source = {
            "route_a_thin_film": prepared_stack.get("metadata", {}).get(
                "route_a_thin_film"
            ),
            "conductors": [
                {
                    "semantic_id": e.semantic_id,
                    "source_semantic_id": e.metadata.get(
                        "source_semantic_id",
                        e.metadata.get("semantic_group_id", e.semantic_id),
                    )
                    if "split_polygon_index" in e.metadata
                    else e.semantic_id,
                    "representation": e.route_representations.get(route),
                    "geometry": e.geometry,
                }
                for e in build_input.entities
                if e.material_kind == "conductor"
            ],
        }
        records = _plain(build_input.metadata["aedt_junction_partitions"])
        for record in records.values():
            record["z_um"] = _junction_partition_plane(partition_source, record)
        build_input = replace(
            build_input,
            metadata={**build_input.metadata, "aedt_junction_partitions": records},
        )
        validate_geometry_input(build_input)
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
            if record.get("side") == "sidewall":
                continue
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
            or side not in {"top", "bottom"}
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
    source = _source_payload(
        build_input, prepared_stack, route=route, source_dbu_um=source_dbu_um
    )
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
        resolved_surfaces, support_bindings, source_dbu_um=source_dbu_um
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
        hole_name = _native_name("hole", name, index)
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


def _lift_inset_ring(
    plane: Mapping[str, Any], ring: Sequence[Sequence[float]]
) -> list[list[float]]:
    origin = tuple(float(value) for value in plane["origin_um"])
    u = tuple(float(value) for value in plane["u"])
    v = tuple(float(value) for value in plane["v"])
    return [
        [
            origin[axis] + float(point[0]) * u[axis] + float(point[1]) * v[axis]
            for axis in range(3)
        ]
        for point in ring
    ]


def _analysis_surface_sheet(
    app: Any, geometry_ref: Mapping[str, Any], *, name: str
) -> tuple[Any, list[list[float]]]:
    local_region = geometry_ref.get("local_region")
    plane_basis = geometry_ref.get("plane_basis")
    outer = geometry_ref.get("outer_loop")
    plane = geometry_ref.get("plane")
    if isinstance(local_region, Mapping) and isinstance(plane_basis, Mapping):
        points = _lift_inset_ring(plane_basis, local_region["exterior"])
        holes = [
            _lift_inset_ring(plane_basis, ring)
            for ring in local_region.get("holes", ())
        ]
    elif isinstance(outer, Sequence) and not isinstance(outer, (str, bytes)):
        if not isinstance(plane, Mapping) or plane.get("axis") != "z":
            raise ValueError("analysis surface loop requires an explicit Z plane")
        z_um = float(plane["value_um"])
        points = [[float(x), float(y), z_um] for x, y in outer]
        holes = [
            [[float(x), float(y), z_um] for x, y in ring]
            for ring in geometry_ref.get("hole_loops", ())
        ]
    else:
        raise ValueError("horizontal analysis surface requires a planar loop")
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
        hole_name = _native_name("hole", name, index)
        hole = app.modeler.create_polyline(
            ring,
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


def _bounds_box(app: Any, entity: Mapping[str, Any], source: Mapping[str, Any]) -> Any:
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
        name=_native_entity_name(source, "domain", entity["semantic_id"]),
        material=native_material,
    )
    if value is False or value is None:
        raise RuntimeError(f"failed to create solution domain {entity['semantic_id']!r}")
    return value


def _solution_body(app: Any, entity: Mapping[str, Any], source: Mapping[str, Any]) -> Any:
    geometry = entity["geometry"]
    if "outer_loop" not in geometry:
        return _bounds_box(app, entity, source)
    z_min, z_max = geometry_z_range(geometry, entity["semantic_id"])
    if z_max <= z_min:
        raise ValueError(
            f"solution domain {entity['semantic_id']!r} has empty Z extent"
        )
    polygon = {
        "exterior": geometry["outer_loop"],
        "holes": geometry.get("hole_loops", ()),
    }
    name = _native_entity_name(source, "domain", entity["semantic_id"])
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


def _native_solution_materials(source: Mapping[str, Any]) -> dict[str, Any]:
    """Install only materials carried by native solution-domain bodies."""

    catalog = source["materials"]
    material_ids = {
        str(item["material_id"]) for item in source["solution_regions"]
    }
    material_ids.add(str(source["native_region"]["material_id"]))
    missing = material_ids - set(catalog)
    if missing:
        raise ValueError(f"native solution materials are missing: {sorted(missing)!r}")
    return {material_id: catalog[material_id] for material_id in sorted(material_ids)}


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


def _junction_partition_plane(
    source: Mapping[str, Any], record: Mapping[str, Any]
) -> float:
    entities = {item["semantic_id"]: item for item in source["conductors"]}
    owners = [entities[end["source_entity_id"]] for end in record["ends"]]
    positions = [
        _route_a_sheet_z(source, owner)
        if owner["representation"] == "surface_sheet"
        else _entity_z_range(owner)[0]
        for owner in owners
    ]
    if len(set(positions)) != 1:
        raise ValueError(
            "junction source Entity planes differ; no thickness or plane fallback"
        )
    for owner in owners:
        if (
            owner["representation"] != "surface_sheet"
            and _entity_z_range(owner)[1] <= positions[0]
        ):
            raise ValueError(
                "junction Route B source Entity has no positive solid thickness"
            )
    return positions[0]


def _junction_polygon(
    source: Mapping[str, Any], junction: PlanarJunction, polygons: Mapping[str, Any]
):
    record = source.get("junction_partitions", {}).get(junction.junction_id)
    if record is None:
        # Historical prepared geometry retains its recorded authored sheet.
        return polygons[junction.source_polygon_id], None
    if record.get("method") != "scgsim.aedt.junction-partition.v1":
        raise ValueError("unknown junction partition method")
    if record["authored_polygon_id"] != junction.source_polygon_id:
        raise ValueError("junction partition authored source identity differs")
    return polygons[record["central_polygon_id"]], record


_NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM = 1e-9


def _native_coordinates_close(left, right) -> bool:
    return len(left) == len(right) and all(
        math.isfinite(float(a))
        and math.isfinite(float(b))
        and abs(float(a) - float(b)) <= _NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM
        for a, b in zip(left, right)
    )


def _native_polygon_correspondence(loops, polygon, name, deviations):
    """Verify a unique vertex/edge bijection; return actual, ordered vertices."""
    expected = [
        tuple(Fraction(str(float(v))) for v in point)
        for ring in (polygon["exterior"], *polygon["holes"])
        for point in ring
    ]
    actual = [point for rings in loops for ring in rings for point in ring]
    matches = {}
    for point in actual:
        candidates = [p for p in expected if _native_coordinates_close(point, p)]
        if len(candidates) != 1 or candidates[0] in matches:
            raise RuntimeError(
                f"native junction vertex correspondence is missing or ambiguous: {name!r}"
            )
        matches[candidates[0]] = point
        deviations.append(max(abs(float(a - b)) for a, b in zip(point, candidates[0])))
    if len(actual) != len(expected) or set(matches) != set(expected):
        raise RuntimeError(f"native junction vertex topology differs: {name!r}")
    reverse = {point: wanted for wanted, point in matches.items()}
    expected_edges = {
        tuple(
            sorted(
                (
                    tuple(Fraction(str(float(v))) for v in a),
                    tuple(Fraction(str(float(v))) for v in b),
                )
            )
        )
        for ring in (polygon["exterior"], *polygon["holes"])
        for a, b in zip(ring, (*ring[1:], ring[0]))
    }
    actual_edges = {
        tuple(sorted((reverse[a], reverse[b])))
        for rings in loops
        for ring in rings
        for a, b in zip(ring, (*ring[1:], ring[0]))
    }
    if (
        actual_edges != expected_edges
        or len(loops) != 1
        or len(loops[0]) != 1 + len(polygon["holes"])
    ):
        raise RuntimeError(
            f"native junction conductor shape/holes differ from source: {name!r}"
        )
    return matches


def _native_edge_projection(point, start, end):
    delta = tuple(b - a for a, b in zip(start, end))
    norm = sum(v * v for v in delta)
    if not norm:
        raise RuntimeError("native junction boundary has a collapsed edge")
    parameter = sum((p - a) * v for p, a, v in zip(point, start, delta)) / norm
    projected = tuple(a + parameter * v for a, v in zip(start, delta))
    return parameter, projected


def _native_on_segment(point, start, end) -> bool:
    parameter, projected = _native_edge_projection(point, start, end)
    nearest = start if parameter < 0 else end if parameter > 1 else projected
    return _native_coordinates_close(point, nearest)


def _native_edge_section(bodies, start, end, *, boundary_only=False):
    """Sections of actual footprints, with only bounded boundary comparisons."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = dx * dx + dy * dy
    if not norm or _native_coordinates_close(start, end):
        raise RuntimeError("native junction edge is unresolved at readback resolution")
    intervals, contacts = [], set()

    def clamp(value):
        point = (start[0] + value * dx, start[1] + value * dy)
        if _native_coordinates_close(point, start):
            return Fraction(0)
        if _native_coordinates_close(point, end):
            return Fraction(1)
        return value

    for rings in bodies:
        if not boundary_only:
            transformed = tuple(
                tuple(
                    (
                        ((p[0] - start[0]) * dx + (p[1] - start[1]) * dy) / norm,
                        dx * (p[1] - start[1]) - dy * (p[0] - start[0]),
                    )
                    for p in ring
                )
                for ring in rings
            )
            spans, points = _section(transformed, 1, 0, 0, 1)
            intervals.extend((clamp(a), clamp(b)) for a, b in spans)
            contacts.update(clamp(v) for v in points)
        for ring in rings:
            for a, b in zip(ring, (*ring[1:], ring[0])):
                ta, pa = _native_edge_projection(a, start, end)
                tb, pb = _native_edge_projection(b, start, end)
                near_a = _native_coordinates_close(a, pa)
                near_b = _native_coordinates_close(b, pb)
                if near_a and 0 <= clamp(ta) <= 1:
                    contacts.add(clamp(ta))
                if near_b and 0 <= clamp(tb) <= 1:
                    contacts.add(clamp(tb))
                if near_a and near_b:
                    lo, hi = (
                        max(Fraction(0), clamp(min(ta, tb))),
                        min(Fraction(1), clamp(max(ta, tb))),
                    )
                    if lo < hi:
                        intervals.append((lo, hi))
    spans = [(a, b) for a, b in intervals if a < b]
    contacts.update(a for a, b in intervals if a == b)
    return spans, contacts


def _native_edge_covered(bodies, start, end) -> bool:
    intervals, _ = _native_edge_section(bodies, start, end, boundary_only=True)
    reached = Fraction(0)
    for lo, hi in sorted(intervals):
        left = tuple(a + lo * (b - a) for a, b in zip(start, end))
        right = tuple(a + reached * (b - a) for a, b in zip(start, end))
        if lo > reached and not _native_coordinates_close(left, right):
            return False
        reached = max(reached, hi)
    return reached == 1


def _native_boundary_contact(left, right) -> bool:
    for rings in left:
        for ring in rings:
            for start, end in zip(ring, (*ring[1:], ring[0])):
                spans, _ = _native_edge_section(right, start, end, boundary_only=True)
                for lo, hi in spans:
                    a = tuple(p + lo * (q - p) for p, q in zip(start, end))
                    b = tuple(p + hi * (q - p) for p, q in zip(start, end))
                    if not _native_coordinates_close(a, b):
                        return True
    return False


def _native_closed_contact(left, right) -> bool:
    return _closed_contact(left, right) or any(
        _native_on_segment(point, a, b)
        for regions, others in ((left, right), (right, left))
        for rings in regions
        for ring in rings
        for point in ring
        for other in others
        for boundary in other
        for a, b in zip(boundary, (*boundary[1:], boundary[0]))
    )


def _native_strict_inside(point, rings) -> bool:
    return (
        _inside_ring(point, rings[0]) == 1
        and all(_inside_ring(point, hole) == -1 for hole in rings[1:])
        and not any(
            _native_on_segment(point, a, b)
            for ring in rings
            for a, b in zip(ring, (*ring[1:], ring[0]))
        )
    )


def _native_cross_sign(a, b, point):
    _, projected = _native_edge_projection(point, a, b)
    value = _cross(a, b, point)
    return (
        0 if _native_coordinates_close(point, projected) else (1 if value > 0 else -1)
    )


def _native_z_may_contact(left, right) -> bool:
    """Exclude forbidden contact only when actual Z ranges are separated."""
    low, high = max(left[0], right[0]), min(left[1], right[1])
    return low <= high or _native_coordinates_close((low,), (high,))


def _native_z_contact(left, left_solid, right, right_solid) -> bool:
    """Required contact compatibility; sheets need full consistent planes."""
    if not left_solid and not right_solid:
        return all(_native_coordinates_close((a,), (b,)) for a in left for b in right)
    if left_solid and right_solid:
        return _native_z_may_contact(left, right)
    solid, sheet = (left, right) if left_solid else (right, left)
    return all(
        (z >= solid[0] or _native_coordinates_close((z,), (solid[0],)))
        and (z <= solid[1] or _native_coordinates_close((z,), (solid[1],)))
        for z in sheet
    )


def _native_planar_loops(obj: Any, z_um: float, observations=None) -> list[tuple]:
    """Read actual planar boundaries at absolute 1e-9 um coordinate resolution.

    Independent vertex/midpoint/length getters need not serialize identically.
    Midpoint and actual native length jointly check straight edges; ambiguity
    below this comparison resolution is not an exact CAD contact guarantee.
    """
    from scgsim.sgb.planning import (
        _cancel_reversed_planar_edges,
        _simple_planar_loops_from_edges,
    )

    loops = []
    for face in obj.faces:
        positions = [vertex.position for vertex in face.vertices]
        if not positions or any(p is None or len(p) != 3 for p in positions):
            raise RuntimeError(
                f"native junction face vertices unavailable: {obj.name!r}"
            )
        if any(not _native_coordinates_close((p[2],), (z_um,)) for p in positions):
            continue
        if not _native_coordinates_close(
            (min(float(p[2]) for p in positions),),
            (max(float(p[2]) for p in positions),),
        ):
            raise RuntimeError(
                "native junction face has inconsistent actual Z coordinates"
            )
        edges = []
        edge_observations = []
        for edge in face.edges:
            vertices = [vertex.position for vertex in edge.vertices]
            if len(vertices) != 2 or any(p is None or len(p) != 3 for p in vertices):
                raise RuntimeError(
                    f"native junction boundary is not polygonal: {obj.name!r}"
                )
            midpoint = edge.midpoint
            expected_midpoint = [(float(a) + float(b)) / 2 for a, b in zip(*vertices)]
            length = edge.length
            chord = math.dist(*vertices)
            if (
                midpoint is None
                or not _native_coordinates_close(midpoint, expected_midpoint)
                or isinstance(length, bool)
                or not isinstance(length, (int, float))
                or not _native_coordinates_close((length,), (chord,))
            ):
                raise RuntimeError(
                    f"native junction edge is curved or its linear readback differs: {obj.name!r}"
                )
            a, b = [tuple(Fraction(str(float(v))) for v in p[:2]) for p in vertices]
            edges.append((a, b))
            edge_observations.append(
                {
                    "vertices_um": [list(map(float, point)) for point in vertices],
                    "midpoint_um": list(map(float, midpoint)),
                    "native_length_um": float(length),
                    "chord_length_um": chord,
                    "maximum_midpoint_coordinate_deviation_um": max(
                        abs(float(a) - float(b))
                        for a, b in zip(midpoint, expected_midpoint)
                    ),
                }
            )
        if observations is not None:
            observations.append(
                {
                    "object_name": obj.name,
                    "requested_plane_z_um": z_um,
                    "maximum_plane_coordinate_deviation_um": max(
                        abs(float(point[2]) - z_um) for point in positions
                    ),
                    "face_vertices_um": [
                        list(map(float, point)) for point in positions
                    ],
                    "edges": edge_observations,
                }
            )
        # Edge direction is not a native face-loop ordering contract.
        face_loops = _simple_planar_loops_from_edges(
            _cancel_reversed_planar_edges(edges)
        )
        area = lambda ring: abs(
            sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, (*ring[1:], ring[0])))
        )
        outer = max(face_loops, key=area)
        loops.append((outer, *(ring for ring in face_loops if ring is not outer)))
    if not loops:
        raise RuntimeError(
            f"native junction conductor has no planar face at source Z: {obj.name!r}"
        )
    return loops


def _native_junction_readback(
    app: Any, source: Mapping[str, Any], junction: PlanarJunction
) -> dict[str, Any]:
    """Verify partitioned CAD and live boundary targets before expression caches.

    Face edges supply complete planar footprints, rather than bounding-box
    contact guesses. Actual returned geometry is preserved. Coordinate identity,
    straightness, Z and contact comparisons use absolute 1e-9 um, rel_tol=0;
    correspondence never substitutes source vertices into contact geometry.
    Below-resolution gaps/curvature remain numerically unresolved. Missing live
    RLC properties or ambiguous topology are explicit readback failures.
    """
    from ansys.aedt.core.generic.constants import AEDT_UNITS, unit_converter
    from ansys.aedt.core.generic.numbers_utils import decompose_variable_value
    from ansys.aedt.core.modules.boundary.common import BoundaryObject

    record = source["junction_partitions"][junction.junction_id]
    if app.modeler.model_units != "um":
        raise RuntimeError(
            "junction native readback requires source micrometre model units"
        )
    polygons = {p["polygon_id"]: p for p in source["polygons"]}
    z_um = float(record["z_um"])
    central_name = _native_name("junction", junction.junction_id)
    central = app.modeler.get_object_from_name(central_name)
    if central is None or len(central.faces) != 1 or len(central.faces[0].edges) != 4:
        raise RuntimeError("native central junction is not one four-edge face")
    observations, deviations = [], []
    central_loops = _native_planar_loops(central, z_um, observations)
    expected = polygons[record["central_polygon_id"]]
    central_matches = _native_polygon_correspondence(
        central_loops, expected, central_name, deviations
    )
    central_z = tuple(
        operation(p[2] for p in observations[0]["face_vertices_um"])
        for operation in (min, max)
    )

    def assignment(boundary_name, target, kind):
        if _native_boundary_type(app, boundary_name) != kind:
            raise RuntimeError(
                f"native junction boundary type differs: {boundary_name!r}"
            )
        raw = app.oboundary.GetBoundaryAssignment(boundary_name)
        if raw is None:
            raise RuntimeError(
                f"native junction boundary target unavailable: {boundary_name!r}"
            )
        ids = [int(v) for v in raw]
        if not ids or len(ids) != len(set(ids)):
            raise RuntimeError("native junction boundary assignment IDs are invalid")
        _, _, names = _resolve_native_assignment(app, ids)
        if names != {target}:
            raise RuntimeError(
                f"native junction boundary has wrong targets: {boundary_name!r}"
            )

    native_by_net: dict[str, list] = {}
    potential_by_net: dict[str, list] = {}
    native_by_polygon: dict[tuple[str, str], list] = {}
    native_z_by_polygon = {}
    entity_nets = {
        entity["semantic_id"]: entity["net_id"] for entity in source["conductors"]
    }
    authored_ids = {
        partition["authored_polygon_id"]
        for partition in source["junction_partitions"].values()
    }
    for entity in source["conductors"]:
        plane = (
            _route_a_sheet_z(source, entity)
            if entity["representation"] == "surface_sheet"
            else _entity_z_range(entity)[0]
        )
        for pid in entity["polygon_ids"]:
            if pid in authored_ids:
                continue
            name = _native_entity_name(source, "conductor", entity["semantic_id"], pid)
            obj = app.modeler.get_object_from_name(name)
            if obj is None:
                raise RuntimeError(
                    f"native junction source conductor unavailable: {name!r}"
                )
            loops = _native_planar_loops(obj, plane, observations)
            wanted = polygons[pid]
            bottom_matches = _native_polygon_correspondence(
                loops, wanted, name, deviations
            )
            bottom_observation = observations[-1]
            bottom_xyz = {
                tuple(Fraction(str(v)) for v in p[:2]): p
                for p in bottom_observation["face_vertices_um"]
            }
            if entity["representation"] == "surface_sheet":
                assignment(
                    _native_name("pec_boundary", entity["semantic_id"], pid),
                    name,
                    "Perfect E",
                )
            else:
                if (
                    native_object_property(obj, "Material").strip('"').casefold()
                    != "pec"
                    or _native_object_boolean_property(obj, "Solve Inside") is not False
                ):
                    raise RuntimeError(
                        f"native junction conductor lacks finite PEC lowering: {name!r}"
                    )
                zs = [float(vertex.position[2]) for vertex in obj.vertices]
                if not zs or not _native_coordinates_close(
                    (min(zs), max(zs)), _entity_z_range(entity)
                ):
                    raise RuntimeError(
                        f"native junction source material/Z lineage differs: {name!r}"
                    )
                top = _native_planar_loops(obj, max(zs), observations)
                top_matches = _native_polygon_correspondence(
                    top, wanted, name, deviations
                )
                top_xyz = {
                    tuple(Fraction(str(v)) for v in p[:2]): p
                    for p in observations[-1]["face_vertices_um"]
                }
                thickness = _entity_z_range(entity)[1] - _entity_z_range(entity)[0]
                if any(
                    not _native_coordinates_close(
                        bottom_matches[point], top_matches[point]
                    )
                    or not _native_coordinates_close(
                        (
                            top_xyz[top_matches[point]][2]
                            - bottom_xyz[bottom_matches[point]][2],
                        ),
                        (thickness,),
                    )
                    for point in bottom_matches
                ):
                    raise RuntimeError(
                        "native junction actual top/bottom extrusion differs"
                    )
                expected_xyz = [
                    (*point, z)
                    for ring in (wanted["exterior"], *wanted["holes"])
                    for point in ring
                    for z in _entity_z_range(entity)
                ]
                actual_xyz = [vertex.position for vertex in obj.vertices]
                matches = []
                for position in actual_xyz:
                    candidates = [
                        n
                        for n, point in enumerate(expected_xyz)
                        if _native_coordinates_close(position, point)
                    ]
                    if len(candidates) != 1 or candidates[0] in matches:
                        raise RuntimeError(
                            "native solid vertex correspondence is missing or ambiguous"
                        )
                    matches.append(candidates[0])
                    deviations.append(
                        max(
                            abs(float(a) - float(b))
                            for a, b in zip(position, expected_xyz[candidates[0]])
                        )
                    )
                edge_count = sum(
                    len(ring) for ring in (wanted["exterior"], *wanted["holes"])
                )
                if (
                    len(matches) != len(expected_xyz)
                    or len(obj.faces) != edge_count + 2
                ):
                    raise RuntimeError(
                        f"native junction solid is not the source polygon extrusion: {name!r}"
                    )
            solid = entity["representation"] != "surface_sheet"
            native_z = (
                (min(zs), max(zs))
                if solid
                else tuple(
                    operation(p[2] for p in bottom_observation["face_vertices_um"])
                    for operation in (min, max)
                )
            )
            key = entity["semantic_id"], pid
            native_by_polygon[key] = loops
            native_z_by_polygon[key] = native_z, solid
            if _native_z_contact(central_z, False, native_z, solid):
                native_by_net.setdefault(entity["net_id"], []).extend(loops)
            if _native_z_may_contact(central_z, native_z):
                potential_by_net.setdefault(entity["net_id"], []).extend(loops)

    # Correspondence orders actual vertices in the authored A->B frame.
    points = [
        central_matches[tuple(Fraction(str(float(v))) for v in p)]
        for p in expected["exterior"]
    ]
    central_midpoint = tuple(sum(p[i] for p in points) / 4 for i in (0, 1))
    for bodies in potential_by_net.values():
        for rings in bodies:
            if any(_native_strict_inside(p, (points,)) for p in rings[0]) or any(
                _native_strict_inside(p, rings) for p in (*points, central_midpoint)
            ):
                raise RuntimeError("native PEC occupies central RLC interior")
            for ring in rings:
                for a, b in zip(ring, (*ring[1:], ring[0])):
                    for c, d in zip(points, (*points[1:], points[0])):
                        if (
                            _native_cross_sign(a, b, c) * _native_cross_sign(a, b, d)
                            < 0
                            and _native_cross_sign(c, d, a)
                            * _native_cross_sign(c, d, b)
                            < 0
                        ):
                            raise RuntimeError(
                                "native PEC crosses central RLC interior"
                            )
    derived_ids = {
        end["polygon_id"]
        for partition in source["junction_partitions"].values()
        for end in partition["ends"]
        if end["polygon_id"] is not None
    }

    for end in record["ends"]:
        if end["polygon_id"] is None:
            continue
        end_loops = native_by_polygon[end["source_entity_id"], end["polygon_id"]]
        end_z, end_solid = native_z_by_polygon[
            end["source_entity_id"], end["polygon_id"]
        ]
        arms = [
            rings
            for (eid, pid), loops in native_by_polygon.items()
            if eid == end["source_entity_id"]
            and pid not in derived_ids
            and _native_z_contact(end_z, end_solid, *native_z_by_polygon[eid, pid])
            for rings in loops
        ]
        if not _native_boundary_contact(end_loops, arms):
            raise RuntimeError(
                "native derived end lacks positive-length source-arm contact"
            )
        foreign = [
            rings
            for (eid, pid), loops in native_by_polygon.items()
            if entity_nets[eid] != end["net_id"]
            and _native_z_may_contact(end_z, native_z_by_polygon[eid, pid][0])
            for rings in loops
        ]
        if _native_closed_contact(end_loops, foreign):
            raise RuntimeError("native derived end contacts another net")
    edge_specs = (
        (points[3], points[0], junction.terminal_a_net),
        (points[1], points[2], junction.terminal_b_net),
        (points[0], points[1], None),
        (points[2], points[3], None),
    )
    edge_evidence = []
    for start, end, correct_net in edge_specs:
        if correct_net is not None and not _native_edge_covered(
            native_by_net.get(correct_net, []), start, end
        ):
            raise RuntimeError(
                "native designated junction end edge is not wholly covered"
            )
        for net, bodies in potential_by_net.items():
            for rings in bodies:
                intervals, contacts = _native_edge_section([rings], start, end)
                if correct_net is not None:
                    forbidden = net != correct_net and bool(intervals or contacts)
                else:
                    forbidden = bool(intervals) or any(0 < v < 1 for v in contacts)
                    # Open side corners belong to their respective end edge.
                    for v in contacts:
                        point = start if v == 0 else end
                        own_corner = (
                            junction.terminal_a_net
                            if point in (points[0], points[3])
                            else junction.terminal_b_net
                        )
                        forbidden |= net != own_corner
                if forbidden:
                    raise RuntimeError(
                        "native junction has forbidden side/wrong-net corner contact"
                    )
        edge_evidence.append(
            {
                "start_um": list(map(float, start)),
                "end_um": list(map(float, end)),
                "terminal_net": correct_net,
            }
        )

    boundary_name = _native_name("junction_rlc", junction.junction_id)
    assignment(boundary_name, central_name, "Lumped RLC")
    live = BoundaryObject(app, boundary_name, auto_update=False)._child_object
    if live is None:
        raise RuntimeError("native live junction RLC property object unavailable")
    available = [str(v) for v in live.GetPropNames()]
    required = (
        "RLC Type",
        "UseInduct",
        "Inductance",
        "UseCap",
        "UseResist",
        "CurrentLine",
    )
    if any(name not in available for name in required):
        raise RuntimeError(
            f"native live RLC readback lacks required properties; available={available!r}"
        )
    values = {name: live.GetPropValue(name) for name in required}

    def enabled(value):
        if value in (True, "true", "True", 1):
            return True
        if value in (False, "false", "False", 0):
            return False
        raise RuntimeError("native RLC enabled property is invalid")

    def scalar(value, unit):
        number, observed_unit = decompose_variable_value(value)
        system = {"H": "Inductance", "F": "Capacitance", "um": "Length"}[unit]
        if (
            observed_unit not in AEDT_UNITS[system]
            or not isinstance(number, (int, float))
            or isinstance(number, bool)
            or not math.isfinite(number)
        ):
            raise RuntimeError("native RLC scalar unit/value unavailable")
        return float(
            unit_converter(
                number, unit_system=system, input_units=observed_unit, output_units=unit
            )
        )

    if (
        values["RLC Type"] != "Parallel"
        or not enabled(values["UseInduct"])
        or enabled(values["UseResist"])
        or scalar(values["Inductance"], "H") != junction.inductance_h
    ):
        raise RuntimeError("native junction parallel L/R parameters differ")
    if enabled(values["UseCap"]) != bool(junction.capacitance_f):
        raise RuntimeError("native junction capacitance enable state differs")
    if junction.capacitance_f and (
        "Capacitance" not in available
        or scalar(live.GetPropValue("Capacitance"), "F") != junction.capacitance_f
    ):
        raise RuntimeError("native junction capacitance differs")
    line, _ = _junction_terminal_line(expected, junction, z_um)
    actual_line = values["CurrentLine"]
    if not isinstance(actual_line, Mapping) or set(actual_line) != {"Start", "End"}:
        raise RuntimeError("native live RLC CurrentLine format is unavailable")
    observed_line = [
        [scalar(value, "um") for value in actual_line[key]] for key in ("Start", "End")
    ]
    if any(
        not _native_coordinates_close(actual, wanted)
        for actual, wanted in zip(observed_line, line)
    ) or len(observed_line) != len(line):
        raise RuntimeError("native junction integration line differs")
    actual_midpoints = []
    for a, b in ((points[3], points[0]), (points[1], points[2])):
        matches = [
            edge["midpoint_um"]
            for edge in observations[0]["edges"]
            if {
                tuple(Fraction(str(v)) for v in point[:2])
                for point in edge["vertices_um"]
            }
            == {a, b}
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "native junction actual terminal-edge midpoint is ambiguous"
            )
        actual_midpoints.append(matches[0])
    if any(
        not _native_coordinates_close(actual, midpoint)
        for actual, midpoint in zip(observed_line, actual_midpoints)
    ):
        raise RuntimeError(
            "native junction integration line differs from actual terminal edges"
        )
    deviations.extend(
        abs(a - b)
        for actual, wanted in zip(observed_line, line)
        for a, b in zip(actual, wanted)
    )
    deviations.extend(
        observation["maximum_plane_coordinate_deviation_um"]
        for observation in observations
    )
    return {
        "method": "native_face_edges_and_live_boundary_properties.v2",
        "coordinate_comparison_policy": {
            "absolute_tolerance_um": _NATIVE_JUNCTION_COORDINATE_ABS_TOL_UM,
            "relative_tolerance": 0.0,
            "scope": "SCGSim native junction readback; not an AEDT precision guarantee",
            "geometry_moved": False,
        },
        "maximum_source_coordinate_deviation_um": max(deviations, default=0.0),
        "native_geometry_observations": observations,
        "edge_contacts": edge_evidence,
        "integration_line_um": observed_line,
        "actual_terminal_edge_midpoints_um": actual_midpoints,
        "raw_integration_line": detached(actual_line),
        "boundary_name": boundary_name,
    }
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


def planar_junction_from_port(
    build_input: GeometryBuildInput,
    *,
    port_name: str,
    junction_id: str,
    terminal_a_net: str,
    terminal_b_net: str,
    inductance_h: float,
    capacitance_f: float,
) -> PlanarJunction:
    """Bind an explicit circuit junction to one normalized authored port sheet."""

    if not isinstance(build_input, GeometryBuildInput):
        raise TypeError("build_input must be GeometryBuildInput")
    if not isinstance(port_name, str) or not port_name:
        raise ValueError("port_name must be non-empty text")
    validate_geometry_input(build_input)
    matches = tuple(
        region for region in build_input.port_sheet_regions
        if region.metadata.get("source_name") == port_name
    )
    if len(matches) != 1:
        raise ValueError(
            f"port {port_name!r} requires exactly one normalized source sheet; "
            f"found {len(matches)}"
        )
    region = matches[0]
    # Adapter overlap inventories contain positive area only. Binding also owns
    # complete terminal-edge contact, and therefore checks the closed XY source.
    def source_rings(exterior, holes):
        return tuple(
            tuple(tuple(Fraction(str(float(v))) for v in point) for point in ring)
            for ring in (exterior, *holes)
        )

    sheet = source_rings(region.exterior, region.holes)
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    overlap_nets = set()
    for entity in build_input.entities:
        if entity.material_kind != "conductor":
            continue
        for pid in entity.polygon_ids:
            if pid == region.source_polygon_id:
                continue
            polygon = polygons[pid]
            if _closed_contact([sheet], [source_rings(polygon.exterior, polygon.holes)]):
                overlap_nets.add(entity.net_id)
    if overlap_nets != {terminal_a_net, terminal_b_net}:
        raise ValueError(
            f"port {port_name!r} overlaps must identify exactly the explicit "
            "terminal A and B Nets"
        )
    direction = region.metadata["direction"]
    dx, dy = float(direction[0]), float(direction[1])
    magnitude = math.hypot(dx, dy)
    dx, dy = dx / magnitude, dy / magnitude
    across = tuple(-dy * x + dx * y for x, y in region.exterior)
    width_um = max(across) - min(across)
    junction = PlanarJunction(
        junction_id=junction_id,
        source_polygon_id=region.source_polygon_id,
        terminal_a_net=terminal_a_net,
        terminal_b_net=terminal_b_net,
        direction_xy=(dx, dy),
        width_um=width_um,
        inductance_h=inductance_h,
        capacitance_f=capacitance_f,
        metadata={"source_port_name": port_name},
    )
    _junction_terminal_line(
        {"exterior": region.exterior, "holes": region.holes}, junction, 0.0
    )
    return junction


def _region_bounds(region: Any) -> tuple[float, ...]:
    bounds = tuple(float(value) for value in region.bounding_box)
    if (
        len(bounds) != 6
        or not all(math.isfinite(value) for value in bounds)
        or any(bounds[index] >= bounds[index + 3] for index in range(3))
    ):
        raise RuntimeError("native EPR Region bounds are invalid")
    return bounds


def _verified_region_bounds(region: Any, plan: Mapping[str, Any]) -> tuple[float, ...]:
    bounds = _region_bounds(region)
    loop = tuple(
        tuple(float(value) for value in point)
        for point in plan["envelope_outer_loop_um"]
    )
    z_min, z_max = plan["z_range_um"]
    expected = (
        min(point[0] for point in loop), min(point[1] for point in loop),
        float(z_min), max(point[0] for point in loop),
        max(point[1] for point in loop), float(z_max),
    )
    if any(
        not math.isclose(
            actual, wanted, rel_tol=0.0,
            abs_tol=max(1e-6, 1e-9 * max(abs(actual), abs(wanted))),
        )
        for actual, wanted in zip(bounds, expected)
    ):
        raise RuntimeError("native EPR Region bounds differ from source envelope")
    return bounds


def _create_epr_region(app: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    plan = source["native_region"]
    if app.modeler.get_object_from_name("Region") is not None:
        raise RuntimeError("new EPR model already has Region")
    region = app.modeler.create_region(
        pad_value=list(plan["padding_um"]),
        pad_type="Absolute Offset",
        name="Region",
    )
    if region is False or region is None or region.name != "Region":
        raise RuntimeError("native EPR Region creation failed")
    # HFSS partitions overlapping interior bodies from Region in solver and
    # Calculator volume support; its special Region object rejects CAD subtract.
    region.material_name = str(plan["material_id"])
    observed_material = native_object_property(region, "Material").strip('"')
    if observed_material.casefold() != str(plan["material_id"]).casefold():
        raise RuntimeError("native EPR Region vacuum material differs")
    if not _native_object_boolean_property(region, "Solve Inside"):
        raise RuntimeError("native EPR Region must solve inside vacuum")
    evidence = _native_object_evidence(app, "Region")
    if evidence["native_object_type"] != "Solid":
        raise RuntimeError("native EPR Region is not solid")
    return {
        "kind": "solution_domain",
        "semantic_id": "Region",
        "material_id": plan["material_id"],
        "logical_vacuum_ids": list(plan["logical_vacuum_ids"]),
        "object_name": "Region",
        "padding_um": list(plan["padding_um"]),
        "native_solve_inside": True,
        "native_bounding_box_um": list(_verified_region_bounds(region, plan)),
        **evidence,
    }


def _enclosure_faces(app: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the Region's six outer faces without assigning a boundary."""

    region = app.modeler.get_object_from_name("Region")
    if region is None:
        raise RuntimeError("native EPR Region is unavailable")
    bounds = _verified_region_bounds(region, source["native_region"])
    z_min, z_max = source["native_region"]["z_range_um"]
    face_records: list[dict[str, Any]] = []
    plane_labels = ("x_minus", "y_minus", "bottom", "x_plus", "y_plus", "top")
    for face in region.faces:
        center = tuple(float(value) for value in face.center)
        matches = [
            plane_labels[index]
            for index in range(6)
            if math.isclose(
                center[index % 3], bounds[index], rel_tol=0.0, abs_tol=1e-6
            )
        ]
        if len(matches) == 1:
            face_records.append(
                {
                    "face_id": int(face.id),
                    "object_name": "Region",
                    "center_um": list(center),
                    "enclosure_plane": matches[0],
                }
            )
    planes = [item["enclosure_plane"] for item in face_records]
    if sorted(planes) != sorted(plane_labels):
        raise RuntimeError("native closed-enclosure face selection is incomplete")
    face_ids = [item["face_id"] for item in face_records]
    if len(face_ids) != len(set(face_ids)):
        raise RuntimeError("native closed-enclosure face selection repeats a face")
    return {
        "face_records": face_records,
        "envelope_outer_loop_um": _plain(
            source["native_region"]["envelope_outer_loop_um"]
        ),
        "z_range_um": [z_min, z_max],
        "native_bounding_box_um": list(bounds),
    }


def _verify_no_explicit_region_boundary(app: Any) -> None:
    """Require native assignment evidence that leaves Region to HFSS defaults."""

    raw = [str(value) for value in app.oboundary.GetBoundaries()]
    if len(raw) % 2:
        raise RuntimeError("native HFSS boundary list is invalid")
    names = raw[::2]
    if len(names) != len(set(names)):
        raise RuntimeError("native HFSS boundary names are not unique")
    for name in names:
        assigned = app.oboundary.GetBoundaryAssignment(name)
        if assigned is None:
            raise RuntimeError(f"native boundary assignment is unavailable for {name!r}")
        try:
            raw_ids = [int(value) for value in assigned]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"native boundary assignment IDs are invalid for {name!r}") from exc
        if not raw_ids or len(raw_ids) != len(set(raw_ids)):
            raise RuntimeError(f"native boundary assignment IDs are invalid for {name!r}")
        _, _, objects = _resolve_native_assignment(app, raw_ids)
        if "Region" in objects:
            raise RuntimeError(f"Region has an explicit native boundary {name!r}")


def _implicit_closed_enclosure(app: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    if source["native_region"].get("outer_boundary_policy") != "hfss_default.v1":
        raise ValueError("EPR Region boundary policy requires re-preparation")
    faces = _enclosure_faces(app, source)
    _verify_no_explicit_region_boundary(app)
    return {
        "boundary_policy": "hfss_default.v1",
        "explicit_assignment": False,
        **faces,
    }


def prepare_native_planar_geometry(app: Any, prepared: PreparedPlanarGeometry) -> dict[str, Any]:
    """Create and read back the body-first HFSS geometry for one prepared request."""

    if not isinstance(prepared, PreparedPlanarGeometry):
        raise TypeError("prepared must be PreparedPlanarGeometry")
    source = detached(prepared.source)
    if (
        source.get("native_region", {}).get("method")
        != "single_region_absolute_offset.v1"
    ):
        raise ValueError("EPR geometry predates single Region; reprepare the handoff")
    if source["native_region"].get("outer_boundary_policy") != "hfss_default.v1":
        raise ValueError("EPR Region boundary policy requires re-preparation")
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
    phase_seconds: dict[str, float] = {}

    started = time.perf_counter()
    material_readback = _install_material_catalog(app, _native_solution_materials(source))
    phase_seconds["materials_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    for entity in source["solution_regions"]:
        if entity["metadata"].get("is_auto_vacuum_region"):
            continue
        obj = _solution_body(app, entity, source)
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
    phase_seconds["interior_solution_bodies_seconds"] = time.perf_counter() - started

    junction_polygons = {item.source_polygon_id for item in prepared.junctions}
    started = time.perf_counter()
    for entity in entities:
        z_min_um, z_max_um = _entity_z_range(entity)
        is_route_a_sheet = entity["representation"] == "surface_sheet"
        z_um = _route_a_sheet_z(source, entity) if is_route_a_sheet else z_min_um
        thickness_um = z_max_um - z_min_um
        for polygon_id in entity["polygon_ids"]:
            if polygon_id in junction_polygons:
                continue
            name = _native_entity_name(
                source, "conductor", entity["semantic_id"], polygon_id
            )
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
    phase_seconds["conductors_seconds"] = time.perf_counter() - started

    junction_bindings: list[dict[str, Any]] = []
    started = time.perf_counter()
    for junction in prepared.junctions:
        polygon, partition = _junction_polygon(source, junction, polygons)
        region = junction_regions.get(junction.source_polygon_id)
        if partition is not None:
            owner_ids = [end["source_entity_id"] for end in partition["ends"]]
        elif region is None:
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
        z_um = partition["z_um"] if partition is not None else z_values[0]
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
        partition_readback = (
            _native_junction_readback(app, source, junction)
            if partition is not None
            else None
        )
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
                **(
                    {"partition_readback": partition_readback}
                    if partition_readback is not None
                    else {}
                ),
                **(
                    {
                        "central_polygon_id": partition["central_polygon_id"],
                        "partition_method": partition["method"],
                    }
                    if partition is not None
                    else {}
                ),
                **evidence,
            }
        )
    phase_seconds["junctions_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    bindings.insert(0, _create_epr_region(app, source))
    enclosure = _implicit_closed_enclosure(app, source)
    phase_seconds["region_boundary_seconds"] = time.perf_counter() - started

    surface_selections: list[dict[str, Any]] = []
    started = time.perf_counter()
    base_names = _base_sheet_plan(prepared.surface_bindings)
    base_points: dict[str, list[list[float]]] = {}
    for binding in prepared.surface_bindings:
        binding_id = str(binding["binding_id"])
        planned_name = base_names[binding_id]
        name = planned_name["name"]
        if name not in base_points:
            sheet, points = _analysis_surface_sheet(
                app, binding["geometry_ref"], name=name
            )
            base_points[name] = points
        else:
            sheet = app.modeler.get_object_from_name(name)
            if sheet is None:
                raise RuntimeError(f"shared analysis surface {name!r} vanished")
            points = base_points[name]
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
                "shared_sheet_members": planned_name["members"],
                "effective_domain_id": binding["effective_domain_id"],
                "adjacent_side": orientation < 0.0,
                "field_side": field_side,
                "native_normal": list(unit_normal),
                "selection_centroid_um": list(centroid),
                "desired_field_side_normal": list(desired_normal),
                **native,
            }
        )
    phase_seconds["base_analysis_sheets_seconds"] = time.perf_counter() - started

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
        "geometry_phases": {
            key: round(value, 6) for key, value in phase_seconds.items()
        },
    }


def bind_saved_planar_geometry(app: Any, prepared: PreparedPlanarGeometry) -> dict[str, Any]:
    """Bind deterministic native objects in a saved project without creating CAD."""

    if not isinstance(prepared, PreparedPlanarGeometry):
        raise TypeError("prepared must be PreparedPlanarGeometry")
    source = detached(prepared.source)
    if (
        source.get("native_region", {}).get("method")
        != "single_region_absolute_offset.v1"
    ):
        raise ValueError("saved EPR geometry predates single Region; reprepare")
    objects: list[dict[str, Any]] = []
    for domain in source["solution_regions"]:
        if domain["metadata"].get("is_auto_vacuum_region"):
            continue
        name = _native_entity_name(source, "domain", domain["semantic_id"])
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
    region_plan = source["native_region"]
    saved_region = app.modeler.get_object_from_name("Region")
    if saved_region is None:
        raise RuntimeError("saved EPR Region is unavailable")
    region_evidence = _native_object_evidence(app, "Region")
    if region_evidence["native_object_type"] != "Solid":
        raise RuntimeError("saved EPR Region is not solid")
    saved_material = native_object_property(saved_region, "Material").strip('"')
    if saved_material.casefold() != str(region_plan["material_id"]).casefold():
        raise RuntimeError("saved EPR Region vacuum material differs")
    if not _native_object_boolean_property(saved_region, "Solve Inside"):
        raise RuntimeError("saved EPR Region does not solve inside vacuum")
    objects.insert(
        0,
        {
            "kind": "solution_domain",
            "semantic_id": "Region",
            "material_id": region_plan["material_id"],
            "logical_vacuum_ids": list(region_plan["logical_vacuum_ids"]),
            "object_name": "Region",
            "padding_um": list(region_plan["padding_um"]),
            "native_solve_inside": True,
            "native_bounding_box_um": list(
                _verified_region_bounds(
                    saved_region, region_plan
                )
            ),
            **region_evidence,
        },
    )

    junction_polygon_ids = {item.source_polygon_id for item in prepared.junctions}
    for entity in source["conductors"]:
        is_sheet = entity["representation"] == "surface_sheet"
        for polygon_id in entity["polygon_ids"]:
            if polygon_id in junction_polygon_ids:
                continue
            name = _native_entity_name(
                source, "conductor", entity["semantic_id"], polygon_id
            )
            evidence = _native_object_evidence(app, name)
            expected_type = "Sheet" if is_sheet else "Solid"
            if evidence["native_object_type"] != expected_type:
                raise RuntimeError(f"saved conductor {name!r} has wrong object type")
            boundary_name: str | None = None
            observed: dict[str, Any] = {}
            if is_sheet:
                boundary_name = _native_name(
                    "pec_boundary", entity["semantic_id"], polygon_id
                )
                if _native_boundary_type(app, boundary_name) != "Perfect E":
                    raise RuntimeError(f"saved conductor {name!r} lacks Perfect E")
                assigned = app.oboundary.GetBoundaryAssignment(boundary_name)
                if assigned is None:
                    raise RuntimeError(f"saved conductor {name!r} lacks PEC assignment")
                try:
                    raw_ids = [int(value) for value in assigned]
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(f"saved conductor {name!r} has invalid PEC IDs") from exc
                if not raw_ids or len(raw_ids) != len(set(raw_ids)):
                    raise RuntimeError(f"saved conductor {name!r} has invalid PEC IDs")
                _, _, assigned_objects = _resolve_native_assignment(app, raw_ids)
                if assigned_objects != {name}:
                    raise RuntimeError(f"saved conductor {name!r} PEC target differs")
            else:
                obj = app.modeler.get_object_from_name(name)
                if obj is None:
                    raise RuntimeError(f"saved conductor {name!r} is unavailable")
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
                    raise RuntimeError(f"saved conductor {name!r} PEC readback differs")
            objects.append({
                "kind": "conductor",
                "semantic_id": entity["semantic_id"],
                "source_polygon_id": polygon_id,
                "route": prepared.route,
                "object_name": name,
                "boundary_name": boundary_name,
                "observed": observed,
                **evidence,
            })

    junctions: list[dict[str, Any]] = []
    for junction in prepared.junctions:
        name = _native_name("junction", junction.junction_id)
        evidence = _native_object_evidence(app, name)
        if evidence["native_object_type"] != "Sheet":
            raise RuntimeError(f"saved junction {name!r} is not a sheet")
        partition_readback = (
            _native_junction_readback(app, source, junction)
            if junction.junction_id in source.get("junction_partitions", {})
            else None
        )
        junctions.append(
            {
                "junction_id": junction.junction_id,
                "source_polygon_id": junction.source_polygon_id,
                "object_name": name,
                **(
                    {"partition_readback": partition_readback}
                    if partition_readback is not None
                    else {}
                ),
                **evidence,
            }
        )

    selections: list[dict[str, Any]] = []
    base_names = _base_sheet_plan(prepared.surface_bindings)
    for binding in prepared.surface_bindings:
        binding_id = str(binding["binding_id"])
        planned_name = base_names[binding_id]
        name = planned_name["name"]
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
                "shared_sheet_members": planned_name["members"],
                "effective_domain_id": binding["effective_domain_id"],
                "adjacent_side": orientation < 0.0,
                "field_side": side,
                "native_normal": list(native_normal),
                "desired_field_side_normal": list(desired),
                **evidence,
            }
        )
    boundary_policy = region_plan.get("outer_boundary_policy")
    if boundary_policy == "hfss_default.v1":
        enclosure = {
            **_implicit_closed_enclosure(app, source),
            "binding_stage": "saved_project_readback_without_cad_creation",
        }
    elif "outer_boundary_policy" not in region_plan:
        # Historical prepared source records an explicit named enclosure.
        closed_type = _native_boundary_type(app, "SCGSimClosedEnclosure")
        if closed_type != "Perfect E":
            raise RuntimeError("saved EPR closed enclosure boundary differs")
        enclosure = {
            "boundary_name": "SCGSimClosedEnclosure",
            "native_boundary_type": closed_type,
            "binding_stage": "saved_project_readback_without_cad_creation",
        }
    else:
        raise ValueError("saved EPR Region boundary policy is unknown")
    return {
        "schema_version": "scgsim.aedt.epr-native-binding.v1",
        "route": prepared.route,
        "source_sha256": prepared.source_sha256,
        "objects": objects,
        "junctions": junctions,
        "surface_selections": selections,
        "material_readback": _material_readback(app, _native_solution_materials(source)),
        "closed_enclosure": enclosure,
        "binding_stage": "saved_project_readback_without_cad_creation",
    }


__all__ = [
    "bind_saved_planar_geometry",
    "prepare_native_planar_geometry",
    "prepare_planar_geometry_input",
]
