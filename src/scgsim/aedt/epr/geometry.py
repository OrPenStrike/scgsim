"""Source-level planar EPR geometry preparation and junction ownership."""

from __future__ import annotations


import math

import os

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

from scgsim.geometry import GeometryBuildInput
from scgsim.geometry.compiler.validation import validate_selected_route
from scgsim.geometry.source.validation import validate_geometry_input

from scgsim.geometry._primitives.spatial import _geometry_ref_surface_z_um
from scgsim.geometry.planning.domain import verified_route_a_substrate_support
from scgsim.geometry.planning.surfaces import plan_surface_contribution_patches

from scgsim.aedt.epr.junction_partition import _closed_contact, partition_junctions

from scgsim.aedt.epr.models import (
    PlanarJunction,
    PreparedPlanarGeometry,
    SurfaceEprSpec,
    canonical_sha256,
    surface_evaluations,
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
    owner = (
        label(owners)
        if len(ordered) == 1
        else (f"Shared_{label(owners)}" if len(owners) == 1 else "Shared")
    )
    classification, side, margin = (
        label({member[index] for member in ordered}) for index in (2, 3, 4)
    )
    digest = canonical_sha256(
        {"geometry_sha256": geometry_sha256, "members": ordered, "parts": sorted(parts)}
    )[:12]
    part_label = (
        ""
        if not parts
        else "_" + "_".join(f"Part{index:02d}" for index in sorted(parts))
    )
    tail = f"_{classification}_{side}_{margin}{part_label}_{digest}"
    owner_budget = 60 - len("EPR_") - len(tail)
    if owner_budget < 1:
        raise ValueError(
            "EPR analysis sheet semantic labels exceed AEDT's 60-character limit"
        )
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
    names = {
        key: _sheet_name(members, key, parts=set())
        for key, members in by_geometry.items()
    }
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
                "Unmasked" if effective_nm == 0.0 else f"Margin_{effective_nm:.12g}nm"
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
        if (
            "split_polygon_index" in item.metadata
            and "source_semantic_id" not in item.metadata
        ):
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
        if (
            record is not None
            and record.get("material_id", semantic_id) != entity.material_id
        ):
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
    missing_regions = set(solution_regions) - {
        item["semantic_id"] for item in normalized_regions
    }
    if missing_regions:
        raise ValueError(
            f"prepared-stack solution regions lack normalized geometry: {sorted(missing_regions)!r}"
        )
    if any(item["semantic_id"] == "Region" for item in normalized_regions):
        raise ValueError(
            "source solution-domain id 'Region' is reserved for native EPR CAD"
        )
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
        "x_plus_um",
        "x_minus_um",
        "y_plus_um",
        "y_minus_um",
        "z_plus_um",
        "z_minus_um",
    }:
        raise ValueError("EPR Region requires exact six-face source padding")
    padding_um = [
        float(padding[key])
        for key in (
            "x_plus_um",
            "x_minus_um",
            "y_plus_um",
            "y_minus_um",
            "z_plus_um",
            "z_minus_um",
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
        "logical_vacuum_ids": sorted(
            item["semantic_id"] for item in auto_vacuum_regions
        ),
        "padding_um": padding_um,
        "envelope_outer_loop_um": _plain(envelope_loop),
        "z_range_um": [
            min(item[0] for item in z_ranges),
            max(item[1] for item in z_ranges),
        ],
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
            if (
                "split_polygon_index" in entity.metadata
                and "source_semantic_id" not in entity.metadata
            ):
                source_id = str(
                    entity.metadata.get("semantic_group_id", entity.semantic_id)
                )
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
        if (
            not isinstance(existing, Mapping)
            or existing.get("variant") != route_a_profile
        ):
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
        if (
            not isinstance(plane, Mapping) or plane.get("axis") != "z"
        ) and geometry_ref.get("shell_part") not in {"top", "bottom"}:
            raise ValueError("planar EPR surface loop requires an explicit Z plane")
        if geometry_ref.get("shell_part") not in {"top", "bottom"} and (
            isinstance(plane.get("value_um"), bool)
            or not isinstance(plane.get("value_um"), (int, float))
        ):
            raise ValueError("planar EPR surface requires finite plane Z")
        z_um = _geometry_ref_surface_z_um(geometry_ref)
        if not math.isfinite(z_um):
            raise ValueError("planar EPR surface requires finite plane Z")
        return {
            "origin_um": [0.0, 0.0, z_um],
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
        delta = tuple(global_point[index] - target_origin[index] for index in range(3))
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
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"Route A conductor {entity['semantic_id']!r} has no finite effective sheet Z"
        )
    return float(value)


def _entity_z_range(entity: Mapping[str, Any]) -> tuple[float, float]:
    return geometry_z_range(entity.get("geometry"), str(entity.get("semantic_id")))


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


def _klayout_region(regions: Sequence[Mapping[str, Any]], dbu_um: float, k: Any) -> Any:
    result = k.Region()
    for region in regions:
        exterior = [_klayout_point(point, dbu_um, k) for point in region["exterior"]]
        polygon = k.Polygon(exterior)
        for hole in region.get("holes", ()):
            polygon.insert_hole([_klayout_point(point, dbu_um, k) for point in hole])
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
    complement = (
        k.Region(
            k.Box(
                bounds.left - padding,
                bounds.bottom - padding,
                bounds.right + padding,
                bounds.top + padding,
            )
        )
        - support
    )
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
        projected = [_project_plane_region(item["mask_plane"], plane) for item in group]
        union = _klayout_region(projected, source_dbu_um, k)
        if union.is_empty():
            raise RuntimeError("physical EPR support union is empty")
        support_by_group[key] = (
            plane,
            _regions_from_klayout(union, source_dbu_um),
            group,
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
                "reuse_source_sheet": radius == 0
                and len(regions) == 1
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
                "margins_um": list(
                    dict.fromkeys(
                        margin
                        for _, margin in surface_evaluations(
                            binding["margins_um"],
                            policy=geometry.source.get("surface_evaluation_policy"),
                        )
                    )
                ),
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
    if build_input.boundary_curves or build_input.boundary_reconstruction:
        raise NotImplementedError(
            "AEDT planar lowering does not support active source curve intent"
        )
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
            record = build_input.metadata["aedt_junction_partitions"][
                junction.junction_id
            ]
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
        build_input,
        route=route,  # type: ignore[arg-type]
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
        if record["interface_kind"] != classification or record["field_side"] != side:
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
    contribution_catalog = tuple(catalog_by_id[key] for key in sorted(catalog_by_id))
    source = _source_payload(
        build_input, prepared_stack, route=route, source_dbu_um=source_dbu_um
    )
    regions_by_id = {item["semantic_id"]: item for item in source["solution_regions"]}
    materials = source["materials"]
    support_bindings: list[dict[str, Any]] = []
    requested_support_sides = {
        (item.interface_kind, item.field_side) for item in contribution_tuple
    }
    for surface in surfaces:
        evidence = surface["contribution"]
        if (
            evidence.get("classification"),
            evidence.get("side"),
        ) not in requested_support_sides:
            continue
        domain_ids = tuple(evidence.get("effective_domain_ids", ()))
        expected_count = 2 if evidence["classification"] == "SA" else 1
        if len(domain_ids) != expected_count:
            raise ValueError("physical EPR support has inconsistent adjacent domains")
        domains = [regions_by_id.get(domain_id) for domain_id in domain_ids]
        if any(domain is None for domain in domains):
            raise ValueError(
                "physical EPR support domain is absent from prepared_stack"
            )
        if evidence["classification"] == "SA":
            field_domains = [
                domain
                for domain in domains
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
            if item["contribution"].get("contribution_id") == requested.contribution_id
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
                        if substrate_domain is not None
                        else None
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
    tolerance = 1e-9 * max(
        1.0, abs(along_min), abs(along_max), abs(across_min), abs(across_max)
    )
    if along_max - along_min <= tolerance or across_max - across_min <= tolerance:
        raise ValueError(
            f"junction {junction.junction_id!r} polygon has empty projected extent"
        )
    expected = {
        (along_min, across_min),
        (along_min, across_max),
        (along_max, across_min),
        (along_max, across_max),
    }
    if any(
        not any(
            abs(u - eu) <= tolerance and abs(v - ev) <= tolerance for eu, ev in expected
        )
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
        region
        for region in build_input.port_sheet_regions
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
            if _closed_contact(
                [sheet], [source_rings(polygon.exterior, polygon.holes)]
            ):
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
