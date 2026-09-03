"""Shared staged simulation primitives for Palace candidates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from scgsim.sgb.models import VacuumRegionSpec

SCHEMA_VERSION = "v0.16.0"

RouteAThinFilm = Literal["substrate_face", "metal_gap_equivalent"]
_Z_TOLERANCE_UM = 1e-9


def validate_nonempty_string(value: Any, field: str) -> str:
    """Return a non-empty trimmed string."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty.")
    return text


def validate_positive_number(value: Any, field: str) -> float:
    """Validate finite positive real values used by the mesh controls."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be a finite number.")
    if float(value) <= 0.0:
        raise ValueError(f"{field} must be > 0.")
    return float(value)


def validate_non_negative_int(value: Any, field: str) -> int:
    """Validate a non-negative integer control."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an int.")
    if value < 0:
        raise ValueError(f"{field} must be >= 0.")
    return int(value)


def normalize_route_a_thin_film(route: str, value: str | None) -> RouteAThinFilm | None:
    """Validate the explicit Route-A thin-film lowering choice."""
    if route == "B":
        if value is not None:
            raise ValueError("Route B does not accept route_a_thin_film.")
        return None
    if route != "A":
        raise ValueError("route must be either 'A' or 'B'.")
    if value == "substrate_face":
        return "substrate_face"
    if value == "metal_gap_equivalent":
        return "metal_gap_equivalent"
    raise ValueError(
        "Route A requires route_a_thin_film='substrate_face' or 'metal_gap_equivalent'."
    )


def apply_route_a_thin_film_to_stack(
    stack: Mapping[str, Any],
    *,
    source_stack: Mapping[str, Any],
    variant: str | None,
) -> Mapping[str, Any]:
    """Return a Route-A stack with one explicit thin-film coordinate contract.

    ``substrate_face`` preserves physical coordinates and lets Route A lower
    face metal to the adjacent substrate faces. ``metal_gap_equivalent``
    collapses both face-metal intervals and applies the same monotone Z map to
    every solution and conductor range. The source mappings are never mutated.
    """
    normalized = normalize_route_a_thin_film("A", variant)
    if not isinstance(stack, Mapping) or not isinstance(source_stack, Mapping):
        raise TypeError("Route-A thin-film lowering requires mapping stacks.")
    facts = _route_a_thin_film_facts(stack)
    work = copy.deepcopy(dict(stack))
    collapsed = 0.0
    if normalized == "metal_gap_equivalent":
        collapsed = (
            facts["lower_metal_thickness_um"] + facts["upper_metal_thickness_um"]
        )
        _map_stack_z_ranges(work, facts)

    effective_gap = facts["physical_substrate_face_gap_um"] - collapsed
    source_hash = _canonical_mapping_sha256(source_stack)
    provenance = {
        "schema_version": 1,
        "variant": normalized,
        "display_label": "A_PRIME" if normalized == "substrate_face" else "A",
        "source_stack": {
            "revision": f"sha256:{source_hash}",
            "sha256": source_hash,
        },
        "host_solution_volume_id": facts["host_solution_volume_id"],
        "physical_substrate_z_ranges_um": facts["physical_substrate_z_ranges_um"],
        "physical_face_metal_z_ranges_um": facts["physical_face_metal_z_ranges_um"],
        "physical_substrate_face_gap_um": facts["physical_substrate_face_gap_um"],
        "physical_metal_gap_um": facts["physical_metal_gap_um"],
        "effective_sheet_z_um": {
            "lower": facts["lower_substrate_face_z_um"],
            "upper": facts["lower_substrate_face_z_um"] + effective_gap,
        },
        "effective_gap_um": effective_gap,
        "collapsed_thickness_um": collapsed,
        "mapping": (
            {
                "kind": "identity_thin_sheet",
                "summary": "Physical Z coordinates are preserved; face films lower to sheets at their substrate faces.",
            }
            if normalized == "substrate_face"
            else {
                "kind": "piecewise_coordinate_normalization",
                "summary": "Both face-metal intervals collapse to sheets; the inter-metal cavity becomes the effective gap and all upper material shifts by the collapsed thickness.",
                "source_breakpoints_um": [
                    facts["lower_substrate_face_z_um"],
                    facts["lower_metal_outer_z_um"],
                    facts["upper_metal_outer_z_um"],
                    facts["upper_substrate_face_z_um"],
                ],
                "target_breakpoints_um": [
                    facts["lower_substrate_face_z_um"],
                    facts["lower_substrate_face_z_um"],
                    facts["lower_substrate_face_z_um"] + effective_gap,
                    facts["lower_substrate_face_z_um"] + effective_gap,
                ],
                "exposed_opening_statement": "The exposed opening silicon gap is coordinate-normalized with the same map.",
                "model_scope": "Sensitivity model; not full physical finite-thickness geometry.",
            }
        ),
    }
    metadata = work.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("stack metadata must be a mapping.")
    if "route_a_thin_film" in metadata:
        raise ValueError("stack already defines route_a_thin_film provenance.")
    work["metadata"] = {**dict(metadata), "route_a_thin_film": provenance}
    return work


def _route_a_thin_film_facts(stack: Mapping[str, Any]) -> dict[str, Any]:
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
    for record in layers:
        if not isinstance(record, Mapping):
            raise TypeError("stack layers must contain mappings.")
        if record.get("part_role") != "face_metal":
            continue
        semantic_id = validate_nonempty_string(
            record.get("semantic_id"), "face semantic_id"
        )
        host_id = validate_nonempty_string(
            record.get("host_void_semantic_id"),
            f"{semantic_id} host_void_semantic_id",
        )
        z_min, z_max = _geometry_z_range(
            _record_geometry(record, semantic_id), semantic_id
        )
        if z_max <= z_min:
            raise ValueError(
                f"{semantic_id} physical face-metal thickness must be > 0."
            )
        faces.append((semantic_id, host_id, z_min, z_max))
    if not faces:
        raise ValueError("Route A requires typed face_metal layers.")
    host_ids = {record[1] for record in faces}
    if len(host_ids) != 1:
        raise ValueError(
            "Route A face_metal layers must share one explicit host solution."
        )
    host_id = next(iter(host_ids))
    face_ranges = _group_z_ranges(faces)
    if len(face_ranges) != 2:
        raise ValueError("Route A requires exactly two physical face-metal Z ranges.")
    lower, upper = face_ranges
    if lower[1] >= upper[0] - _Z_TOLERANCE_UM:
        raise ValueError("Route A face-metal intervals must enclose a positive cavity.")

    host = regions.get(host_id)
    if host is None:
        raise ValueError(f"Route A host solution {host_id!r} is missing.")
    if not isinstance(host, Mapping):
        raise TypeError(f"Route A host solution {host_id!r} must be a mapping.")
    host_min, host_max = _geometry_z_range(_record_geometry(host, host_id), host_id)
    if not _same_z(host_min, lower[0]) or not _same_z(host_max, upper[1]):
        raise ValueError(
            "Route A host solution boundaries must equal the two substrate faces."
        )
    lower_substrate = _adjacent_dielectric_region(
        regions, materials, host_id=host_id, z_um=host_min, side="lower"
    )
    upper_substrate = _adjacent_dielectric_region(
        regions, materials, host_id=host_id, z_um=host_max, side="upper"
    )
    return {
        "host_solution_volume_id": host_id,
        "lower_substrate_face_z_um": host_min,
        "upper_substrate_face_z_um": host_max,
        "lower_metal_outer_z_um": lower[1],
        "upper_metal_outer_z_um": upper[0],
        "lower_metal_thickness_um": lower[1] - lower[0],
        "upper_metal_thickness_um": upper[1] - upper[0],
        "physical_substrate_face_gap_um": host_max - host_min,
        "physical_metal_gap_um": upper[0] - lower[1],
        "physical_substrate_z_ranges_um": {
            "lower": lower_substrate,
            "upper": upper_substrate,
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


def _group_z_ranges(
    faces: Sequence[tuple[str, str, float, float]],
) -> list[tuple[float, float, list[str]]]:
    result: list[tuple[float, float, list[str]]] = []
    for semantic_id, _, z_min, z_max in sorted(faces, key=lambda item: item[0]):
        match = next(
            (
                index
                for index, (left, right, _) in enumerate(result)
                if _same_z(left, z_min) and _same_z(right, z_max)
            ),
            None,
        )
        if match is None:
            result.append((z_min, z_max, [semantic_id]))
        else:
            result[match][2].append(semantic_id)
    return sorted(result, key=lambda item: (item[0], item[1]))


def _adjacent_dielectric_region(
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
        z_min, z_max = _geometry_z_range(
            _record_geometry(region, str(semantic_id)), str(semantic_id)
        )
        boundary = z_max if side == "lower" else z_min
        if _same_z(boundary, z_um):
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


def _map_stack_z_ranges(stack: dict[str, Any], facts: Mapping[str, Any]) -> None:
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
            rewritten = _mapped_geometry(record, facts, str(key))
            geometry = record.get("geometry")
            if geometry is not None:
                if not isinstance(geometry, Mapping):
                    raise TypeError(
                        f"stack {section} record {key!r} geometry must be a mapping."
                    )
                rewritten["geometry"] = _mapped_geometry(geometry, facts, str(key))
            if isinstance(rewritten_records, dict):
                rewritten_records[key] = rewritten
            else:
                rewritten_records[int(key)] = rewritten
        stack[section] = rewritten_records


def _record_geometry(record: Mapping[str, Any], context: str) -> dict[str, Any]:
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


def _mapped_geometry(
    geometry: Mapping[str, Any], facts: Mapping[str, Any], context: str
) -> dict[str, Any]:
    result = copy.deepcopy(dict(geometry))
    if "z_min_um" in result or "z_max_um" in result:
        if "z_min_um" not in result or "z_max_um" not in result:
            raise ValueError(f"{context} must define both z_min_um and z_max_um.")
        z_min, z_max = _geometry_z_range(result, context)
        result["z_min_um"] = _map_z(z_min, facts)
        result["z_max_um"] = _map_z(z_max, facts)
    elif "z_um" in result or "thickness_um" in result:
        if "z_um" not in result or "thickness_um" not in result:
            raise ValueError(f"{context} must define both z_um and thickness_um.")
        z_min, z_max = _geometry_z_range(result, context)
        mapped_min, mapped_max = _map_z(z_min, facts), _map_z(z_max, facts)
        result["z_um"] = mapped_min
        result["thickness_um"] = mapped_max - mapped_min
    return result


def _map_z(z_um: float, facts: Mapping[str, Any]) -> float:
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


def _geometry_z_range(geometry: Any, context: str) -> tuple[float, float]:
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


def _finite_z(value: Any, field: str) -> float:
    if not _is_finite_float(value):
        raise ValueError(f"{field} must be a finite number.")
    return float(value)


def _same_z(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=_Z_TOLERANCE_UM)


def _canonical_mapping_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def apply_airbox_to_stack(
    stack: Mapping[str, Any], airbox: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return a copy of stack with requested airbox envelope updates applied."""
    if not airbox:
        return dict(stack)
    if not isinstance(stack, Mapping):
        raise TypeError("stack must be a mapping.")

    solution_regions = stack.get("solution_regions")
    if not isinstance(solution_regions, Mapping):
        raise TypeError("stack must provide 'solution_regions' mapping.")

    matched = []
    for key, region in solution_regions.items():
        if not isinstance(region, Mapping):
            raise TypeError("each solution region must be a mapping.")
        marker = region.get("is_airbox", False)
        if not isinstance(marker, bool):
            raise TypeError(f"solution region {key!r} is_airbox must be a bool.")
        if marker:
            if region.get("role") != "solution_region":
                raise ValueError(
                    f"airbox marker {key!r} must be on a solution_region record."
                )
            matched.append((str(key), region))

    work = copy.deepcopy(dict(stack))
    work_regions = dict(work["solution_regions"])
    if len(matched) == 1:
        targets = ((matched[0][0], matched[0][1], None),)
    elif len(matched) == 2:
        targets = tuple(
            (key, region, region.get("airbox_side")) for key, region in matched
        )
        sides = tuple(side for _, _, side in targets)
        if set(sides) != {"below", "above"} or len(set(sides)) != len(sides):
            raise ValueError(
                "Two airbox solution regions require one explicit airbox_side=below "
                "and one explicit airbox_side=above."
            )
    else:
        raise TypeError(
            "Airbox requires either one legacy marker or explicit below/above markers."
        )

    has_x = "margin_x" in airbox
    has_y = "margin_y" in airbox
    if has_x or has_y:
        if not (has_x and has_y):
            raise ValueError("margin_x and margin_y must both be provided together.")
        margin_x = validate_positive_number(airbox["margin_x"], "margin_x")
        margin_y = validate_positive_number(airbox["margin_y"], "margin_y")
        if margin_x != margin_y:
            raise ValueError(
                "margin_x and margin_y must be equal for scalar padding updates."
            )
    shared_domain_bounds: dict[str, float] | None = None
    if len(targets) == 2 and has_x:
        marker_bounds = tuple(
            _explicit_domain_bounds(region, key) for key, region, _ in targets
        )
        if marker_bounds[0] != marker_bounds[1]:
            raise ValueError(
                "Two-sided airbox markers must have identical explicit domain_bounds_um."
            )
        common_bounds = marker_bounds[0]
        shared_regions = tuple(
            (str(key), region)
            for key, region in solution_regions.items()
            if region.get("role") == "solution_region"
        )
        for key, region in shared_regions:
            if _explicit_domain_bounds(region, key) != common_bounds:
                raise ValueError(
                    "Two-sided airbox requires every solution_region to share its "
                    "explicit domain_bounds_um footprint."
                )
        shared_domain_bounds = {
            "x_min_um": common_bounds["x_min_um"] - margin_x,
            "y_min_um": common_bounds["y_min_um"] - margin_y,
            "x_max_um": common_bounds["x_max_um"] + margin_x,
            "y_max_um": common_bounds["y_max_um"] + margin_y,
        }
        for key, region in shared_regions:
            work_region = dict(region)
            work_geometry = dict(region["geometry"])
            work_geometry["domain_bounds_um"] = dict(shared_domain_bounds)
            work_region["geometry"] = work_geometry
            work_regions[key] = work_region
    z_below = (
        validate_positive_number(airbox["z_below"], "z_below")
        if "z_below" in airbox
        else None
    )
    z_above = (
        validate_positive_number(airbox["z_above"], "z_above")
        if "z_above" in airbox
        else None
    )
    for key, region, side in targets:
        geometry = region.get("geometry")
        if not isinstance(geometry, Mapping):
            raise TypeError(f"airbox solution region {key!r} must define geometry.")
        work_region = dict(region)
        work_geometry = dict(geometry)
        if has_x:
            if shared_domain_bounds is not None:
                work_geometry["domain_bounds_um"] = dict(shared_domain_bounds)
            else:
                bounds = work_geometry.get("domain_bounds_um")
                if isinstance(bounds, Mapping):
                    required_bounds = ("x_min_um", "y_min_um", "x_max_um", "y_max_um")
                    if any(
                        not isinstance(bounds.get(name), (int, float))
                        or isinstance(bounds.get(name), bool)
                        for name in required_bounds
                    ):
                        raise TypeError(
                            f"airbox solution region {key!r} has invalid domain_bounds_um."
                        )
                    work_geometry["domain_bounds_um"] = {
                        "x_min_um": float(bounds["x_min_um"]) - margin_x,
                        "y_min_um": float(bounds["y_min_um"]) - margin_y,
                        "x_max_um": float(bounds["x_max_um"]) + margin_x,
                        "y_max_um": float(bounds["y_max_um"]) + margin_y,
                    }
                else:
                    padding = work_geometry.get("padding_um")
                    if not isinstance(padding, (int, float)) or isinstance(
                        padding, bool
                    ):
                        raise TypeError(
                            f"airbox solution region {key!r} must define padding_um or domain_bounds_um."
                        )
                    if float(padding) < 0.0:
                        raise ValueError(
                            f"airbox solution region {key!r} has invalid padding_um."
                        )
                    work_geometry["padding_um"] = float(padding) + margin_x
        if z_below is not None and side in {None, "below"}:
            z_min = work_geometry.get("z_min_um")
            if not isinstance(z_min, (int, float)) or isinstance(z_min, bool):
                raise TypeError(f"airbox solution region {key!r} must define z_min_um.")
            work_geometry["z_min_um"] = float(z_min) - z_below
        if z_above is not None and side in {None, "above"}:
            z_max = work_geometry.get("z_max_um")
            if not isinstance(z_max, (int, float)) or isinstance(z_max, bool):
                raise TypeError(f"airbox solution region {key!r} must define z_max_um.")
            work_geometry["z_max_um"] = float(z_max) + z_above
        work_region["geometry"] = work_geometry
        work_regions[key] = work_region
    work["solution_regions"] = work_regions
    return work


def apply_vacuum_region_to_stack(
    stack: Mapping[str, Any],
    vacuum_region: VacuumRegionSpec,
) -> Mapping[str, Any]:
    """Return a copy of stack with one canonical VACUUM_REGION solution region.

    Explicit vacuum solution regions are preserved as independent semantic+physical
    records. The generated VACUUM_REGION envelope is derived from non-vacuum
    positive regions and then padded by the normalized vacuum specification.
    """
    if not isinstance(stack, Mapping):
        raise TypeError("stack must be a mapping.")
    if not isinstance(vacuum_region, VacuumRegionSpec):
        raise TypeError("vacuum_region must be a VacuumRegionSpec")

    solution_regions_raw = stack.get("solution_regions")
    if not isinstance(solution_regions_raw, Mapping):
        raise TypeError("stack must define solution_regions mapping.")
    for semantic_id, region in solution_regions_raw.items():
        if not isinstance(region, Mapping):
            raise TypeError(f"solution region {semantic_id!r} must be a mapping.")
        is_airbox = region.get("is_airbox", False)
        if not isinstance(is_airbox, bool):
            raise TypeError(
                f"solution region {semantic_id!r} is_airbox must be a bool."
            )
        if is_airbox:
            raise ValueError(
                "set_vacuum_region cannot be used with explicit airbox solution regions."
            )

    materials = stack.get("materials")
    if not isinstance(materials, Mapping):
        raise TypeError("stack must define a materials mapping.")
    solution_regions = {
        str(key): dict(value) for key, value in solution_regions_raw.items()
    }

    work = dict(stack)
    work_materials = dict(materials)

    vacuum_material_id = _materialize_locked_vacuum_material(
        solution_regions=solution_regions,
        materials=work_materials,
    )

    auto_bounds = _auto_vacuum_bounds(
        solution_regions,
        materials=work_materials,
    )

    region = solution_regions.get("VACUUM_REGION")
    if region is None:
        region = {
            "role": "solution_region",
            "material_id": vacuum_material_id,
            "geometry_kind": "domain",
            "metadata": {},
            "geometry": {},
        }
    else:
        if not isinstance(region, Mapping):
            raise TypeError("VACUUM_REGION solution record must be a mapping.")
        current_material_id = _material_id_for_region(region)
        if current_material_id != vacuum_material_id:
            raise ValueError(
                "VACUUM_REGION material_id conflicts with existing vacuum material"
            )

    region.update(
        {
            "material_id": vacuum_material_id,
            "material_kind": "vacuum",
            "is_auto_vacuum_region": True,
            "geometry": {
                **dict(region.get("geometry", {})),
                "domain": "VACUUM_REGION",
                "domain_bounds_um": {
                    "x_min_um": auto_bounds["x_min_um"] - vacuum_region.x_minus_um,
                    "y_min_um": auto_bounds["y_min_um"] - vacuum_region.y_minus_um,
                    "x_max_um": auto_bounds["x_max_um"] + vacuum_region.x_plus_um,
                    "y_max_um": auto_bounds["y_max_um"] + vacuum_region.y_plus_um,
                },
                "z_min_um": auto_bounds["z_min_um"] - vacuum_region.z_minus_um,
                "z_max_um": auto_bounds["z_max_um"] + vacuum_region.z_plus_um,
            },
            "metadata": {
                **dict(region.get("metadata", {})),
                "source": "set_vacuum_region",
                "is_auto_vacuum_region": True,
                "vacuum_region_padding_um": {
                    "x_minus_um": vacuum_region.x_minus_um,
                    "x_plus_um": vacuum_region.x_plus_um,
                    "y_minus_um": vacuum_region.y_minus_um,
                    "y_plus_um": vacuum_region.y_plus_um,
                    "z_minus_um": vacuum_region.z_minus_um,
                    "z_plus_um": vacuum_region.z_plus_um,
                },
            },
        }
    )

    solution_regions["VACUUM_REGION"] = region
    work["solution_regions"] = solution_regions
    work["materials"] = work_materials
    return {
        **work,
        "solution_regions": solution_regions,
    }


def _materialize_locked_vacuum_material(
    solution_regions: Mapping[str, Mapping[str, Any]],
    materials: dict[str, Any],
) -> str:
    vacuum_id = "vacuum"
    if vacuum_id in materials:
        vacuum_material = materials[vacuum_id]
        if not isinstance(vacuum_material, Mapping):
            raise TypeError("material 'vacuum' must be a mapping.")
        vacuum_kind = vacuum_material.get("kind")
        vacuum_permittivity = vacuum_material.get("permittivity")
        vacuum_loss_tangent = vacuum_material.get("loss_tangent")
        if str(vacuum_kind) != "vacuum":
            raise ValueError("material 'vacuum' must have kind 'vacuum'.")
        if vacuum_permittivity is None:
            vacuum_material = dict(vacuum_material)
            vacuum_permittivity = 1.0
            vacuum_material["permittivity"] = vacuum_permittivity
            materials[vacuum_id] = vacuum_material
        if not isinstance(vacuum_permittivity, (int, float)):
            raise ValueError("material 'vacuum' must define permittivity=1.0.")
        if float(vacuum_permittivity) != 1.0:
            raise ValueError("material 'vacuum' must define permittivity=1.0.")
        if vacuum_loss_tangent is None:
            vacuum_material = dict(vacuum_material)
            vacuum_loss_tangent = 0.0
            vacuum_material["loss_tangent"] = vacuum_loss_tangent
            materials[vacuum_id] = vacuum_material
        if not isinstance(vacuum_loss_tangent, (int, float)):
            raise ValueError("material 'vacuum' must define loss_tangent=0.0.")
        if float(vacuum_loss_tangent) != 0.0:
            raise ValueError("material 'vacuum' must define loss_tangent=0.0.")
    else:
        materials[vacuum_id] = {
            "kind": "vacuum",
            "permittivity": 1.0,
            "loss_tangent": 0.0,
        }

    vacuum_regions: dict[str, str] = {}
    for semantic_id, region in solution_regions.items():
        material_id = _material_id_for_region(region)
        kind = _material_kind(material_id, materials)
        if kind == "vacuum":
            if material_id != vacuum_id:
                raise ValueError(
                    "explicit vacuum solution region uses non-canonical material_id "
                    f"{material_id!r}; expected {vacuum_id!r}."
                )
            vacuum_regions[semantic_id] = material_id

    if len({*vacuum_regions.values()}) > 1:
        raise ValueError(
            "Conflicting vacuum material definitions across existing solution regions."
        )

    if "VACUUM_REGION" in vacuum_regions:
        return vacuum_regions["VACUUM_REGION"]

    return vacuum_id


def _auto_vacuum_bounds(
    solution_regions: Mapping[str, Mapping[str, Any]],
    materials: Mapping[str, Any],
) -> dict[str, float]:
    bounds: list[dict[str, float]] = []
    for semantic_id, region in solution_regions.items():
        semantic_id = str(semantic_id)
        material_id = _material_id_for_region(region)
        kind = _material_kind(material_id, materials)
        metadata = region.get("metadata")
        if (
            kind == "vacuum"
            and isinstance(metadata, Mapping)
            and bool(metadata.get("is_auto_vacuum_region"))
        ):
            continue

        geometry = region.get("geometry")
        if not isinstance(geometry, Mapping):
            raise TypeError(
                f"solution region {semantic_id!r} must define geometry for auto vacuum envelope."
            )

        domain = geometry.get("domain_bounds_um")
        if not isinstance(domain, Mapping):
            raise TypeError(
                f"solution region {semantic_id!r} must define domain_bounds_um for auto vacuum envelope."
            )

        required_domain = ("x_min_um", "x_max_um", "y_min_um", "y_max_um")
        if any(not _is_finite_float(domain.get(name)) for name in required_domain):
            missing = [
                name
                for name in required_domain
                if not _is_finite_float(domain.get(name))
            ]
            raise ValueError(
                f"solution region {semantic_id!r} has invalid {missing!r} for domain_bounds_um."
            )

        z_min = geometry.get("z_min_um", geometry.get("z_um"))
        if z_min is None or not _is_finite_float(z_min):
            raise ValueError(
                f"solution region {semantic_id!r} has missing or non-finite z_min_um."
            )
        z_min = float(z_min)

        z_max = geometry.get("z_max_um")
        if z_max is None:
            thickness = geometry.get("thickness_um")
            if thickness is None or not _is_finite_float(thickness):
                raise ValueError(
                    f"solution region {semantic_id!r} requires finite z_max_um or thickness_um."
                )
            z_max = z_min + float(thickness)
        else:
            if not _is_finite_float(z_max):
                raise ValueError(
                    f"solution region {semantic_id!r} has non-finite z_max_um."
                )
            z_max = float(z_max)

        if not (z_min < z_max):
            raise ValueError(
                f"solution region {semantic_id!r} has non-positive thickness for envelope aggregation."
            )

        bounds.append(
            {
                "x_min_um": float(domain["x_min_um"]),
                "x_max_um": float(domain["x_max_um"]),
                "y_min_um": float(domain["y_min_um"]),
                "y_max_um": float(domain["y_max_um"]),
                "z_min_um": z_min,
                "z_max_um": z_max,
            }
        )

    if not bounds:
        raise ValueError(
            "Cannot auto-compute vacuum envelope without non-vacuum solution regions."
        )

    return {
        "x_min_um": min(item["x_min_um"] for item in bounds),
        "x_max_um": max(item["x_max_um"] for item in bounds),
        "y_min_um": min(item["y_min_um"] for item in bounds),
        "y_max_um": max(item["y_max_um"] for item in bounds),
        "z_min_um": min(item["z_min_um"] for item in bounds),
        "z_max_um": max(item["z_max_um"] for item in bounds),
    }


def _material_id_for_region(region: Mapping[str, Any]) -> str:
    material_id = region.get("material_id", region.get("name"))
    if not isinstance(material_id, str) or not material_id:
        raise ValueError("solution region material_id must be a non-empty string.")
    return material_id


def _material_kind(material_id: str, materials: Mapping[str, Any]) -> str:
    material = materials.get(material_id)
    if not isinstance(material, Mapping):
        raise TypeError(
            f"solution region material {material_id!r} must resolve a material mapping."
        )
    material_kind = material.get("kind")
    if material_kind not in {"vacuum", "dielectric", "conductor"}:
        raise ValueError(f"solution region material {material_id!r} has invalid kind.")
    return str(material_kind)


def _is_finite_float(raw: Any) -> bool:
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        return False
    value = float(raw)
    return math.isfinite(value)


def _explicit_domain_bounds(region: Mapping[str, Any], key: str) -> dict[str, float]:
    geometry = region.get("geometry")
    if not isinstance(geometry, Mapping):
        raise TypeError(f"solution region {key!r} must define geometry.")
    bounds = geometry.get("domain_bounds_um")
    if not isinstance(bounds, Mapping):
        raise TypeError(
            f"solution region {key!r} must define explicit domain_bounds_um."
        )
    required = ("x_min_um", "y_min_um", "x_max_um", "y_max_um")
    if any(
        not isinstance(bounds.get(name), (int, float))
        or isinstance(bounds.get(name), bool)
        or not math.isfinite(float(bounds[name]))
        for name in required
    ):
        raise TypeError(f"solution region {key!r} has invalid domain_bounds_um.")
    return {name: float(bounds[name]) for name in required}
