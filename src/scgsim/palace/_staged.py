"""Shared staged simulation primitives for Palace candidates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

from scgsim.semantics.route_a import (
    adjacent_dielectric_region as semantic_adjacent_dielectric_region,
    apply_thin_film_profile,
    derive_thin_film_facts,
    geometry_z_range as semantic_geometry_z_range,
    group_z_ranges as semantic_group_z_ranges,
    map_z as semantic_map_z,
    map_stack_z_ranges as semantic_map_stack_z_ranges,
    mapped_record as semantic_mapped_record,
    normalize_optional_profile,
    record_geometry as semantic_record_geometry,
    same_z as semantic_same_z,
    single_face_thin_film_facts,
    validate_single_face_metal_records,
)
from scgsim.sgb.vacuum import apply_vacuum_region_to_stack

SCHEMA_VERSION = "v0.16.0"

RouteAThinFilm = Literal["substrate_face", "metal_gap_equivalent"]


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
    try:
        normalized = normalize_optional_profile(value)
    except ValueError:
        normalized = None
    if normalized is not None:
        return normalized
    raise ValueError(
        "Route A requires route_a_thin_film='substrate_face' or 'metal_gap_equivalent'."
    )


def apply_route_a_thin_film_to_stack(
    stack: Mapping[str, Any],
    *,
    source_stack: Mapping[str, Any],
    variant: str | None,
    component: Any | None = None,
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
    layers = stack.get("layers", ())
    if isinstance(layers, str | bytes) or not isinstance(layers, Sequence):
        raise TypeError("Route-A thin-film lowering requires structured layers.")
    substrate_support = (
        _normalized_route_a_support(component, stack) if component is not None else None
    )
    facts = _route_a_thin_film_facts(
        stack,
        allow_single_face=normalized == "substrate_face",
        substrate_support=substrate_support,
    )
    work = dict(
        apply_thin_film_profile(
            stack,
            profile=normalized,
            facts=facts,
        )
    )
    source_hash = _canonical_mapping_sha256(source_stack)
    provenance = _route_a_thin_film_provenance(
        normalized,
        facts=facts,
        source_hash=source_hash,
    )
    metadata = work.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("stack metadata must be a mapping.")
    if "route_a_thin_film" in metadata:
        raise ValueError("stack already defines route_a_thin_film provenance.")
    work["metadata"] = {**dict(metadata), "route_a_thin_film": provenance}
    return work


def _route_a_thin_film_provenance(
    variant: RouteAThinFilm,
    *,
    facts: Mapping[str, Any],
    source_hash: str,
) -> dict[str, Any]:
    face_ranges = facts["physical_face_metal_z_ranges_um"]
    if len(face_ranges) == 1:
        side = next(iter(face_ranges))
        return {
            "schema_version": 1,
            "variant": variant,
            "display_label": "A_PRIME",
            "source_stack": {
                "revision": f"sha256:{source_hash}",
                "sha256": source_hash,
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
            "physical_substrate_face_gap_um": None,
            "physical_metal_gap_um": None,
            "effective_sheet_z_um": {side: facts["physical_face_z_um"]},
            "effective_gap_um": None,
            "collapsed_thickness_um": 0.0,
            "mapping": {
                "kind": "identity_thin_sheet",
                "summary": "Physical Z coordinates are preserved; face films lower to sheets at their substrate faces.",
            },
        }
    collapsed = 0.0
    if variant == "metal_gap_equivalent":
        collapsed = (
            facts["lower_metal_thickness_um"] + facts["upper_metal_thickness_um"]
        )
    effective_gap = facts["physical_substrate_face_gap_um"] - collapsed
    return {
        "schema_version": 1,
        "variant": variant,
        "display_label": "A_PRIME" if variant == "substrate_face" else "A",
        "source_stack": {
            "revision": f"sha256:{source_hash}",
            "sha256": source_hash,
        },
        "host_solution_volume_id": facts["host_solution_volume_id"],
        **(
            {"host_reference_origin": facts["host_reference_origin"]}
            if "host_reference_origin" in facts
            else {}
        ),
        "physical_substrate_z_ranges_um": facts["physical_substrate_z_ranges_um"],
        "physical_face_metal_z_ranges_um": face_ranges,
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
            if variant == "substrate_face"
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


def _route_a_thin_film_facts(
    stack: Mapping[str, Any],
    *,
    allow_single_face: bool = False,
    substrate_support: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    return derive_thin_film_facts(
        stack,
        allow_single_face=allow_single_face,
        substrate_support=substrate_support,
    )


def _normalized_route_a_support(
    component: Any | None,
    stack: Mapping[str, Any],
) -> Mapping[str, Mapping[str, str]]:
    if component is None:
        raise ValueError(
            "generated-background Route A requires the component's normalized layout polygons."
        )
    from scgsim.sgb import build_gds_stack_geometry_input
    from scgsim.sgb.planning import verified_route_a_substrate_support

    from ._mesh import _component_gds_top_cell_name, _write_component_gds

    with TemporaryDirectory(prefix="scgsim-auto-route-a-") as directory:
        gds_path = Path(directory) / "design.gds"
        stack_path = Path(directory) / "design.stack.json"
        _write_component_gds(component, gds_path)
        stack_path.write_text(json.dumps(dict(stack), indent=2) + "\n", encoding="utf-8")
        build_input = build_gds_stack_geometry_input(
            gds_file=gds_path,
            stack_file=stack_path,
            top_cell_name=_component_gds_top_cell_name(
                component=component, gds_path=gds_path
            ),
        )
        return verified_route_a_substrate_support(build_input, stack)


def _validate_single_face_metal_records(
    semantic_ids: Sequence[str],
    *,
    records: Mapping[str, Mapping[str, Any]],
    materials: Mapping[str, Any],
) -> None:
    validate_single_face_metal_records(
        semantic_ids, records=records, materials=materials
    )


def _single_face_route_a_thin_film_facts(
    face: tuple[float, float, list[str]],
    *,
    regions: Mapping[str, Any],
    materials: Mapping[str, Any],
    host_id: str,
) -> dict[str, Any]:
    return single_face_thin_film_facts(
        face, regions=regions, materials=materials, host_id=host_id
    )


def _group_z_ranges(
    faces: Sequence[tuple[str, str, float, float]],
) -> list[tuple[float, float, list[str]]]:
    return semantic_group_z_ranges(faces)


def _adjacent_dielectric_region(
    regions: Mapping[str, Any],
    materials: Mapping[str, Any],
    *,
    host_id: str,
    z_um: float,
    side: Literal["lower", "upper"],
) -> dict[str, Any]:
    return semantic_adjacent_dielectric_region(
        regions, materials, host_id=host_id, z_um=z_um, side=side
    )


def _map_stack_z_ranges(stack: dict[str, Any], facts: Mapping[str, Any]) -> None:
    semantic_map_stack_z_ranges(stack, facts)


def _record_geometry(record: Mapping[str, Any], context: str) -> dict[str, Any]:
    return semantic_record_geometry(record, context)


def _mapped_geometry(
    geometry: Mapping[str, Any], facts: Mapping[str, Any], context: str
) -> dict[str, Any]:
    return semantic_mapped_record(geometry, facts, context)


def _map_z(z_um: float, facts: Mapping[str, Any]) -> float:
    return semantic_map_z(z_um, facts)


def _geometry_z_range(geometry: Any, context: str) -> tuple[float, float]:
    return semantic_geometry_z_range(geometry, context)


def _same_z(left: float, right: float) -> bool:
    return semantic_same_z(left, right)


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
