"""Pure Route-A facts and copied profile transformations."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

RouteAProfile = Literal["substrate_face", "metal_gap_equivalent"]
_Z_TOLERANCE_UM = 1e-9


def normalize_optional_profile(value: str | None) -> RouteAProfile | None:
    if value is None:
        return None
    if value in {"substrate_face", "metal_gap_equivalent"}:
        return value  # type: ignore[return-value]
    raise ValueError("route_a profile must be substrate_face, metal_gap_equivalent, or None")


def derive_thin_film_facts(
    stack: Mapping[str, Any],
    *,
    allow_single_face: bool = False,
    substrate_support: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Derive validated physical facts from an already-shaped stack mapping."""
    layers = stack.get("layers")
    regions = stack.get("solution_regions")
    materials = stack.get("materials")
    if (
        isinstance(layers, str | bytes)
        or not isinstance(layers, Sequence)
        or not isinstance(regions, Mapping)
        or not isinstance(materials, Mapping)
    ):
        raise TypeError(
            "Route-A thin-film lowering requires layers, solution_regions, and materials."
        )
    faces: list[tuple[str, str, float, float]] = []
    face_records: dict[str, Mapping[str, Any]] = {}
    for record in layers:
        if not isinstance(record, Mapping):
            raise TypeError("stack layers must contain mappings.")
        if record.get("part_role") != "face_metal":
            continue
        semantic_id = _nonempty(record.get("semantic_id"), "face semantic_id")
        host_id = _nonempty(
            record.get("host_void_semantic_id"),
            f"{semantic_id} host_void_semantic_id",
        )
        z_min, z_max = geometry_z_range(
            record_geometry(record, semantic_id), semantic_id
        )
        if z_max <= z_min:
            raise ValueError(
                f"{semantic_id} physical face-metal thickness must be > 0."
            )
        faces.append((semantic_id, host_id, z_min, z_max))
        face_records[semantic_id] = record
    if not faces:
        raise ValueError("Route A requires typed face_metal layers.")
    host_ids = {record[1] for record in faces}
    if len(host_ids) != 1:
        raise ValueError(
            "Route A face_metal layers must share one explicit host solution."
        )
    host_id = next(iter(host_ids))
    origins = set()
    for semantic_id, record in face_records.items():
        metadata = record.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError(f"{semantic_id} metadata must be a mapping.")
        origins.add(metadata.get("host_reference_origin"))
    if len(origins) != 1:
        raise ValueError("Route A face-metal hosts mix authored and generated references.")
    generated = origins == {"generated_background"}
    if generated and substrate_support is None:
        raise ValueError("generated-background Route A requires verified substrate coverage.")
    if generated and host_id != "VACUUM_REGION":
        raise ValueError("generated-background Route A requires its internal vacuum reference.")
    legacy_support = (
        None
        if generated
        else _legacy_host_substrate_support(
            faces,
            regions=regions,
            materials=materials,
            host_id=host_id,
            allow_single_face=allow_single_face,
        )
    )
    if substrate_support is not None and legacy_support is not None:
        if dict(substrate_support) != legacy_support:
            raise ValueError(
                "Route A authored host and full-planar substrate coverage disagree."
            )
    return _derive_thin_film_facts_from_substrates(
        faces,
        face_records=face_records,
        regions=regions,
        materials=materials,
        support=substrate_support if substrate_support is not None else legacy_support,
        allow_single_face=allow_single_face,
        host_id=host_id,
        generated=generated,
    )


def _legacy_host_substrate_support(
    faces: Sequence[tuple[str, str, float, float]],
    *,
    regions: Mapping[str, Any],
    materials: Mapping[str, Any],
    host_id: str,
    allow_single_face: bool,
) -> dict[str, dict[str, str]]:
    """Retain explicit-host input checks, then project to shared support facts."""
    groups = group_z_ranges(faces)
    host = regions.get(host_id)
    if host is None:
        raise ValueError(f"Route A host solution {host_id!r} is missing.")
    if not isinstance(host, Mapping):
        raise TypeError(f"Route A host solution {host_id!r} must be a mapping.")
    host_min, host_max = geometry_z_range(record_geometry(host, host_id), host_id)
    material_id = host.get("material_id", host_id)
    host_material = materials.get(material_id)
    if not isinstance(host_material, Mapping) or host_material.get("kind") != "vacuum":
        raise ValueError(
            f"Route A host solution {host_id!r} must reference an explicit vacuum material."
        )
    if len(groups) == 1 and allow_single_face:
        z_min, z_max, semantic_ids = groups[0]
        if z_min < host_min - _Z_TOLERANCE_UM or z_max > host_max + _Z_TOLERANCE_UM:
            raise ValueError(
                "Route A single face-metal interval must be contained within its host solution."
            )
        lower = same_z(host_min, z_min)
        upper = same_z(host_max, z_max)
        if lower == upper:
            raise ValueError(
                "Route A single face-metal interval must share exactly one host solution boundary."
            )
        side = "lower" if lower else "upper"
        substrate = adjacent_dielectric_region(
            regions,
            materials,
            host_id=host_id,
            z_um=host_min if lower else host_max,
            side=side,
        )
        return {
            semantic_id: {"side": side, "substrate_id": substrate["semantic_id"]}
            for semantic_id in semantic_ids
        }
    if len(groups) != 2:
        raise ValueError("Route A requires exactly two physical face-metal Z ranges.")
    lower, upper = groups
    if lower[1] >= upper[0] - _Z_TOLERANCE_UM:
        raise ValueError("Route A face-metal intervals must enclose a positive cavity.")
    if not same_z(host_min, lower[0]) or not same_z(host_max, upper[1]):
        raise ValueError(
            "Route A host solution boundaries must equal the two substrate faces."
        )
    support = {}
    for side, semantic_ids, face_z in (
        ("lower", lower[2], host_min),
        ("upper", upper[2], host_max),
    ):
        substrate = adjacent_dielectric_region(
            regions, materials, host_id=host_id, z_um=face_z, side=side
        )
        support.update(
            {
                semantic_id: {"side": side, "substrate_id": substrate["semantic_id"]}
                for semantic_id in semantic_ids
            }
        )
    return support


def _derive_thin_film_facts_from_substrates(
    faces: Sequence[tuple[str, str, float, float]],
    *,
    face_records: Mapping[str, Mapping[str, Any]],
    regions: Mapping[str, Any],
    materials: Mapping[str, Any],
    support: Mapping[str, Mapping[str, str]],
    allow_single_face: bool,
    host_id: str,
    generated: bool,
) -> dict[str, Any]:
    """Use verified full-planar support, never background bounds, for Route A."""
    face_ids = {semantic_id for semantic_id, _, _, _ in faces}
    if set(support) != face_ids:
        raise ValueError(
            f"Route A substrate support must cover exactly {sorted(face_ids)!r}; "
            f"received {sorted(support)!r}."
        )
    groups = group_z_ranges(faces)
    if len(groups) not in ({1, 2} if allow_single_face else {2}):
        raise ValueError("Route A requires one or two physical face-metal Z ranges for this profile.")

    grouped: list[tuple[float, float, list[str], str, dict[str, Any]]] = []
    for z_min, z_max, semantic_ids in groups:
        sides = {support[semantic_id]["side"] for semantic_id in semantic_ids}
        substrate_ids = {
            support[semantic_id]["substrate_id"] for semantic_id in semantic_ids
        }
        if len(sides) != 1 or len(substrate_ids) != 1:
            raise ValueError(
                f"Route A face-metal group {semantic_ids!r} has ambiguous substrate support."
            )
        side = next(iter(sides))
        substrate_id = next(iter(substrate_ids))
        if side not in {"lower", "upper"}:
            raise ValueError(f"Route A substrate support for {semantic_ids!r} has invalid side.")
        substrate = regions.get(substrate_id)
        if not isinstance(substrate, Mapping):
            raise ValueError(f"Route A supporting substrate {substrate_id!r} is missing.")
        material_id = substrate.get("material_id", substrate_id)
        material = materials.get(material_id)
        if not isinstance(material, Mapping) or material.get("kind") != "dielectric":
            raise ValueError(f"Route A supporting substrate {substrate_id!r} must be dielectric.")
        sub_min, sub_max = geometry_z_range(
            record_geometry(substrate, substrate_id), substrate_id
        )
        expected_z = sub_max if side == "lower" else sub_min
        face_z = z_min if side == "lower" else z_max
        if not same_z(expected_z, face_z):
            raise ValueError(
                f"Route A face-metal group {semantic_ids!r} does not touch substrate {substrate_id!r} on {side} side."
            )
        validate_single_face_metal_records(
            semantic_ids, records=face_records, materials=materials
        )
        grouped.append(
            (
                z_min,
                z_max,
                semantic_ids,
                side,
                {"semantic_id": substrate_id, "z_min_um": sub_min, "z_max_um": sub_max},
            )
        )

    if len(grouped) == 1:
        z_min, z_max, semantic_ids, side, substrate = grouped[0]
        return {
            "host_solution_volume_id": host_id,
            **({"host_reference_origin": "generated_background"} if generated else {}),
            "physical_face_z_um": z_min if side == "lower" else z_max,
            "physical_substrate_z_ranges_um": {side: substrate},
            "physical_face_metal_z_ranges_um": {
                side: {
                    "semantic_ids": semantic_ids,
                    "z_min_um": z_min,
                    "z_max_um": z_max,
                }
            },
        }

    lower, upper = grouped
    if lower[3] != "lower" or upper[3] != "upper":
        raise ValueError("Route A opposing face-metal groups require lower and upper substrates.")
    if lower[4]["semantic_id"] == upper[4]["semantic_id"]:
        raise ValueError("Route A opposing faces require distinct supporting substrates.")
    if lower[1] >= upper[0] - _Z_TOLERANCE_UM:
        raise ValueError("Route A face-metal intervals must enclose a positive cavity.")
    physical_gap = upper[4]["z_min_um"] - lower[4]["z_max_um"]
    if physical_gap <= _Z_TOLERANCE_UM:
        raise ValueError("Route A supporting substrates must enclose a positive physical gap.")
    return {
        "host_solution_volume_id": host_id,
        **({"host_reference_origin": "generated_background"} if generated else {}),
        "lower_substrate_face_z_um": lower[4]["z_max_um"],
        "upper_substrate_face_z_um": upper[4]["z_min_um"],
        "lower_metal_outer_z_um": lower[1],
        "upper_metal_outer_z_um": upper[0],
        "lower_metal_thickness_um": lower[1] - lower[0],
        "upper_metal_thickness_um": upper[1] - upper[0],
        "physical_substrate_face_gap_um": physical_gap,
        "physical_metal_gap_um": upper[0] - lower[1],
        "physical_substrate_z_ranges_um": {
            "lower": lower[4],
            "upper": upper[4],
        },
        "physical_face_metal_z_ranges_um": {
            "lower": {
                "semantic_ids": lower[2],
                "z_min_um": lower[0],
                "z_max_um": lower[1],
            },
            "upper": {
                "semantic_ids": upper[2],
                "z_min_um": upper[0],
                "z_max_um": upper[1],
            },
        },
    }


def validate_single_face_metal_records(
    semantic_ids: Sequence[str],
    *,
    records: Mapping[str, Mapping[str, Any]],
    materials: Mapping[str, Any],
) -> None:
    for semantic_id in semantic_ids:
        record = records[semantic_id]
        if record.get("role") != "metal":
            raise ValueError(f"{semantic_id} face_metal must have role='metal'.")
        material_id = _nonempty(
            record.get("material_id"), f"{semantic_id} material_id"
        )
        material = materials.get(material_id)
        if not isinstance(material, Mapping) or material.get("kind") != "conductor":
            raise ValueError(
                f"{semantic_id} face_metal must reference an explicit conductor material."
            )


def single_face_thin_film_facts(
    face: tuple[float, float, list[str]],
    *,
    regions: Mapping[str, Any],
    materials: Mapping[str, Any],
    host_id: str,
) -> dict[str, Any]:
    host = regions.get(host_id)
    if host is None:
        raise ValueError(f"Route A host solution {host_id!r} is missing.")
    if not isinstance(host, Mapping):
        raise TypeError(f"Route A host solution {host_id!r} must be a mapping.")
    host_material_id = _nonempty(
        host.get("material_id", host_id),
        f"Route A host solution {host_id!r} material_id",
    )
    host_material = materials.get(host_material_id)
    if not isinstance(host_material, Mapping) or host_material.get("kind") != "vacuum":
        raise ValueError(
            f"Route A host solution {host_id!r} must reference an explicit vacuum material."
        )
    host_min, host_max = geometry_z_range(record_geometry(host, host_id), host_id)
    z_min, z_max, semantic_ids = face
    if z_min < host_min - _Z_TOLERANCE_UM or z_max > host_max + _Z_TOLERANCE_UM:
        raise ValueError(
            "Route A single face-metal interval must be contained within its host solution."
        )
    at_lower_boundary = same_z(host_min, z_min)
    at_upper_boundary = same_z(host_max, z_max)
    if at_lower_boundary == at_upper_boundary:
        raise ValueError(
            "Route A single face-metal interval must share exactly one host solution boundary."
        )
    if at_lower_boundary:
        side: Literal["lower", "upper"] = "lower"
        face_z_um = host_min
    else:
        side = "upper"
        face_z_um = host_max
    substrate = adjacent_dielectric_region(
        regions, materials, host_id=host_id, z_um=face_z_um, side=side
    )
    return {
        "host_solution_volume_id": host_id,
        "physical_face_z_um": face_z_um,
        "physical_substrate_z_ranges_um": {side: substrate},
        "physical_face_metal_z_ranges_um": {
            side: {
                "semantic_ids": semantic_ids,
                "z_min_um": z_min,
                "z_max_um": z_max,
            }
        },
    }


def group_z_ranges(
    faces: Sequence[tuple[str, str, float, float]],
) -> list[tuple[float, float, list[str]]]:
    result: list[tuple[float, float, list[str]]] = []
    for semantic_id, _, z_min, z_max in sorted(faces, key=lambda item: item[0]):
        match = next(
            (
                index
                for index, (left, right, _) in enumerate(result)
                if same_z(left, z_min) and same_z(right, z_max)
            ),
            None,
        )
        if match is None:
            result.append((z_min, z_max, [semantic_id]))
        else:
            result[match][2].append(semantic_id)
    return sorted(result, key=lambda item: (item[0], item[1]))


def adjacent_dielectric_region(
    regions: Mapping[str, Any],
    materials: Mapping[str, Any],
    *,
    host_id: str,
    z_um: float,
    side: Literal["lower", "upper"],
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for semantic_id, region in regions.items():
        if semantic_id == host_id or not isinstance(region, Mapping):
            continue
        material = materials.get(region.get("material_id"))
        if not isinstance(material, Mapping) or material.get("kind") != "dielectric":
            continue
        z_min, z_max = geometry_z_range(
            record_geometry(region, str(semantic_id)), str(semantic_id)
        )
        boundary = z_max if side == "lower" else z_min
        if same_z(boundary, z_um):
            matches.append(
                {
                    "semantic_id": str(semantic_id),
                    "z_min_um": z_min,
                    "z_max_um": z_max,
                }
            )
    if len(matches) != 1:
        raise ValueError(
            f"Route A requires exactly one typed dielectric {side} substrate adjacent to its host."
        )
    return matches[0]


def record_geometry(record: Mapping[str, Any], context: str) -> dict[str, Any]:
    raw = record.get("geometry")
    if raw is None:
        geometry = dict(record)
    elif isinstance(raw, Mapping):
        geometry = dict(raw)
    else:
        raise TypeError(f"{context} geometry must be a mapping.")
    for first, second in (("z_min_um", "z_max_um"), ("z_um", "thickness_um")):
        if first in record or second in record:
            if first not in record or second not in record:
                raise ValueError(f"{context} must define both {first} and {second}.")
            geometry.setdefault(first, record[first])
            geometry.setdefault(second, record[second])
    return geometry


def geometry_z_range(geometry: Any, context: str) -> tuple[float, float]:
    if not isinstance(geometry, Mapping):
        raise TypeError(f"{context} geometry must be a mapping.")
    if "z_min_um" in geometry or "z_max_um" in geometry:
        z_min = _finite_z(geometry.get("z_min_um"), f"{context} z_min_um")
        z_max = _finite_z(geometry.get("z_max_um"), f"{context} z_max_um")
    else:
        z_min = _finite_z(geometry.get("z_um"), f"{context} z_um")
        thickness = _finite_z(geometry.get("thickness_um"), f"{context} thickness_um")
        z_max = z_min + thickness
    if z_max < z_min:
        raise ValueError(f"{context} has a negative Z extent.")
    return z_min, z_max


def apply_thin_film_profile(
    stack: Mapping[str, Any],
    *,
    profile: str | None,
    facts: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return a detached stack after one optional neutral coordinate profile.

    The core does not attach a backend artifact schema or canonical source
    identity. Palace retains that provenance projection in its thin wrapper.
    """
    normalized = normalize_optional_profile(profile)
    if not isinstance(stack, Mapping):
        raise TypeError("Route-A thin-film lowering requires mapping stacks.")
    work = copy.deepcopy(dict(stack))
    if normalized is None:
        return work
    if facts is None:
        raise ValueError("explicit Route-A profile requires facts")
    if normalized == "metal_gap_equivalent":
        map_stack_z_ranges(work, facts)
    return work


def apply_thin_film_profile_with_provenance(
    stack: Mapping[str, Any],
    *,
    profile: str,
    facts: Mapping[str, Any],
    source_revision: str,
) -> Mapping[str, Any]:
    """Apply one Route-A profile and record its effective physical coordinates."""

    normalized = normalize_optional_profile(profile)
    if normalized is None:
        raise ValueError("Route A profile is required")
    if not isinstance(source_revision, str) or not source_revision.startswith("sha256:"):
        raise ValueError("Route A source revision must be a sha256 identity")
    work = dict(apply_thin_film_profile(stack, profile=normalized, facts=facts))
    metadata = work.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("stack metadata must be a mapping")
    if "route_a_thin_film" in metadata:
        raise ValueError("stack already defines route_a_thin_film provenance")
    face_ranges = facts["physical_face_metal_z_ranges_um"]
    if len(face_ranges) == 1:
        side = next(iter(face_ranges))
        provenance = {
            "schema_version": 1,
            "variant": normalized,
            "source_stack": {
                "revision": source_revision,
                "sha256": source_revision.removeprefix("sha256:"),
            },
            "host_solution_volume_id": facts["host_solution_volume_id"],
            **(
                {"host_reference_origin": facts["host_reference_origin"]}
                if "host_reference_origin" in facts
                else {}
            ),
            "physical_substrate_z_ranges_um": facts[
                "physical_substrate_z_ranges_um"
            ],
            "physical_face_metal_z_ranges_um": face_ranges,
            "effective_sheet_z_um": {side: facts["physical_face_z_um"]},
            "collapsed_thickness_um": 0.0,
        }
    else:
        collapsed = (
            facts["lower_metal_thickness_um"]
            + facts["upper_metal_thickness_um"]
            if normalized == "metal_gap_equivalent"
            else 0.0
        )
        effective_gap = facts["physical_substrate_face_gap_um"] - collapsed
        provenance = {
            "schema_version": 1,
            "variant": normalized,
            "source_stack": {
                "revision": source_revision,
                "sha256": source_revision.removeprefix("sha256:"),
            },
            "host_solution_volume_id": facts["host_solution_volume_id"],
            **(
                {"host_reference_origin": facts["host_reference_origin"]}
                if "host_reference_origin" in facts
                else {}
            ),
            "physical_substrate_z_ranges_um": facts[
                "physical_substrate_z_ranges_um"
            ],
            "physical_face_metal_z_ranges_um": face_ranges,
            "physical_substrate_face_gap_um": facts[
                "physical_substrate_face_gap_um"
            ],
            "physical_metal_gap_um": facts["physical_metal_gap_um"],
            "effective_sheet_z_um": {
                "lower": facts["lower_substrate_face_z_um"],
                "upper": facts["lower_substrate_face_z_um"] + effective_gap,
            },
            "effective_gap_um": effective_gap,
            "collapsed_thickness_um": collapsed,
        }
    work["metadata"] = {**dict(metadata), "route_a_thin_film": provenance}
    return work


def map_stack_z_ranges(stack: dict[str, Any], facts: Mapping[str, Any]) -> None:
    for section in ("solution_regions", "layers"):
        records = stack.get(section)
        if isinstance(records, Mapping):
            rewritten_records: dict[Any, Any] | list[Any] = dict(records)
        elif isinstance(records, Sequence) and not isinstance(records, str | bytes):
            rewritten_records = list(records)
        else:
            raise TypeError(f"stack {section} must contain structured records.")
        items = (
            rewritten_records.items()
            if isinstance(rewritten_records, dict)
            else enumerate(rewritten_records)
        )
        for key, record in items:
            if not isinstance(record, Mapping):
                raise TypeError(f"stack {section} record {key!r} must be a mapping.")
            rewritten = _mapped_record(record, facts, str(key))
            geometry = record.get("geometry")
            if geometry is not None:
                if not isinstance(geometry, Mapping):
                    raise TypeError(
                        f"stack {section} record {key!r} geometry must be a mapping."
                    )
                rewritten["geometry"] = _mapped_record(geometry, facts, str(key))
            if isinstance(rewritten_records, dict):
                rewritten_records[key] = rewritten
            else:
                rewritten_records[int(key)] = rewritten
        stack[section] = rewritten_records


def map_z(z_um: float, facts: Mapping[str, Any]) -> float:
    lower_face = float(facts["lower_substrate_face_z_um"])
    lower_outer = float(facts["lower_metal_outer_z_um"])
    upper_outer = float(facts["upper_metal_outer_z_um"])
    upper_face = float(facts["upper_substrate_face_z_um"])
    lower_thickness = float(facts["lower_metal_thickness_um"])
    collapsed = lower_thickness + float(facts["upper_metal_thickness_um"])
    if z_um <= lower_face + _Z_TOLERANCE_UM:
        return z_um
    if z_um <= lower_outer + _Z_TOLERANCE_UM:
        return lower_face
    if z_um <= upper_outer + _Z_TOLERANCE_UM:
        return z_um - lower_thickness
    if z_um <= upper_face + _Z_TOLERANCE_UM:
        return upper_outer - lower_thickness
    return z_um - collapsed


def mapped_geometry(
    geometry: Mapping[str, Any],
    facts: Mapping[str, Any],
    *,
    context: str,
    z_range: tuple[float, float],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(geometry))
    z_min, z_max = z_range
    if "z_min_um" in result or "z_max_um" in result:
        result["z_min_um"] = map_z(z_min, facts)
        result["z_max_um"] = map_z(z_max, facts)
    elif "z_um" in result or "thickness_um" in result:
        mapped_min, mapped_max = map_z(z_min, facts), map_z(z_max, facts)
        result["z_um"] = mapped_min
        result["thickness_um"] = mapped_max - mapped_min
    return result


def mapped_record(
    geometry: Mapping[str, Any], facts: Mapping[str, Any], context: str
) -> dict[str, Any]:
    """Map one record while preserving exact paired-field validation."""
    return _mapped_record(geometry, facts, context)


def _mapped_record(
    geometry: Mapping[str, Any], facts: Mapping[str, Any], context: str
) -> dict[str, Any]:
    result = copy.deepcopy(dict(geometry))
    if "z_min_um" in result or "z_max_um" in result:
        if "z_min_um" not in result or "z_max_um" not in result:
            raise ValueError(f"{context} must define both z_min_um and z_max_um.")
        z_range = geometry_z_range(result, context)
        return mapped_geometry(result, facts, context=context, z_range=z_range)
    if "z_um" in result or "thickness_um" in result:
        if "z_um" not in result or "thickness_um" not in result:
            raise ValueError(f"{context} must define both z_um and thickness_um.")
        z_range = geometry_z_range(result, context)
        return mapped_geometry(result, facts, context=context, z_range=z_range)
    return result


def same_z(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=_Z_TOLERANCE_UM)


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty.")
    return text


def _finite_z(value: Any, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number.")
    return float(value)
