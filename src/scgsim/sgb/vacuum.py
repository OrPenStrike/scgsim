"""Public preparation of the canonical automatic vacuum solution region."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.semantics.route_a import (
    geometry_z_range,
    record_geometry,
)
from scgsim.sgb.models import VacuumRegionSpec


def apply_vacuum_region_to_stack(
    stack: Mapping[str, Any], vacuum_region: VacuumRegionSpec
) -> Mapping[str, Any]:
    """Return a detached stack with one canonical automatic vacuum region."""

    if not isinstance(stack, Mapping):
        raise TypeError("stack must be a mapping.")
    if not isinstance(vacuum_region, VacuumRegionSpec):
        raise TypeError("vacuum_region must be a VacuumRegionSpec")
    raw_regions = stack.get("solution_regions")
    materials = stack.get("materials")
    if not isinstance(raw_regions, Mapping):
        raise TypeError("stack must define solution_regions mapping.")
    if not isinstance(materials, Mapping):
        raise TypeError("stack must define a materials mapping.")
    for semantic_id, region in raw_regions.items():
        if not isinstance(region, Mapping):
            raise TypeError(f"solution region {semantic_id!r} must be a mapping.")
        if not isinstance(region.get("is_airbox", False), bool):
            raise TypeError(
                f"solution region {semantic_id!r} is_airbox must be a bool."
            )
        if region.get("is_airbox", False):
            raise ValueError(
                "set_vacuum_region cannot be used with explicit airbox solution regions."
            )
    regions = {str(key): dict(value) for key, value in raw_regions.items()}
    work = dict(stack)
    material_catalog = dict(materials)
    vacuum_id = _materialize_locked_vacuum_material(regions, material_catalog)
    bounds = _auto_vacuum_bounds(regions, material_catalog, stack.get("layers", ()))
    region = regions.get("VACUUM_REGION")
    if region is None:
        region = {
            "role": "solution_region",
            "material_id": vacuum_id,
            "geometry_kind": "domain",
            "metadata": {},
            "geometry": {},
        }
    else:
        if not isinstance(region, Mapping):
            raise TypeError("VACUUM_REGION solution record must be a mapping.")
        if _material_id_for_region(region) != vacuum_id:
            raise ValueError(
                "VACUUM_REGION material_id conflicts with existing vacuum material"
            )
        region = dict(region)
    region.update({
        "material_id": vacuum_id,
        "material_kind": "vacuum",
        "is_auto_vacuum_region": True,
        "geometry": {
            **dict(region.get("geometry", {})),
            "domain": "VACUUM_REGION",
            "domain_bounds_um": {
                "x_min_um": bounds["x_min_um"] - vacuum_region.x_minus_um,
                "x_max_um": bounds["x_max_um"] + vacuum_region.x_plus_um,
                "y_min_um": bounds["y_min_um"] - vacuum_region.y_minus_um,
                "y_max_um": bounds["y_max_um"] + vacuum_region.y_plus_um,
            },
            "z_min_um": bounds["z_min_um"] - vacuum_region.z_minus_um,
            "z_max_um": bounds["z_max_um"] + vacuum_region.z_plus_um,
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
    })
    regions["VACUUM_REGION"] = region
    work["solution_regions"] = regions
    work["materials"] = material_catalog
    return {**work, "solution_regions": regions}


def _materialize_locked_vacuum_material(
    regions: Mapping[str, Mapping[str, Any]], materials: dict[str, Any]
) -> str:
    record = materials.get("vacuum")
    if record is None:
        materials["vacuum"] = {
            "kind": "vacuum",
            "permittivity": 1.0,
            "loss_tangent": 0.0,
        }
    elif not isinstance(record, Mapping):
        raise TypeError("material 'vacuum' must be a mapping.")
    else:
        normalized = dict(record)
        if normalized.get("permittivity") is None:
            normalized["permittivity"] = 1.0
        if normalized.get("loss_tangent") is None:
            normalized["loss_tangent"] = 0.0
        if normalized.get("kind") != "vacuum":
            raise ValueError("material 'vacuum' must have kind 'vacuum'.")
        if normalized["permittivity"] != 1.0:
            raise ValueError("material 'vacuum' must define permittivity=1.0.")
        if normalized["loss_tangent"] != 0.0:
            raise ValueError("material 'vacuum' must define loss_tangent=0.0.")
        materials["vacuum"] = normalized
    for semantic_id, region in regions.items():
        material_id = _material_id_for_region(region)
        material = materials.get(material_id)
        if not isinstance(material, Mapping):
            raise TypeError(f"solution region material {material_id!r} is missing.")
        if material.get("kind") == "vacuum" and material_id != "vacuum":
            raise ValueError(
                f"vacuum solution region {semantic_id!r} uses noncanonical material"
            )
    return "vacuum"


def _auto_vacuum_bounds(
    regions: Mapping[str, Mapping[str, Any]],
    materials: Mapping[str, Any],
    layers: Any,
) -> dict[str, float]:
    bounds: list[dict[str, float]] = []
    for semantic_id, region in regions.items():
        semantic_id = str(semantic_id)
        material_id = _material_id_for_region(region)
        kind = _material_kind(material_id, materials)
        metadata = region.get("metadata")
        if kind == "vacuum" and isinstance(metadata, Mapping) and bool(
            metadata.get("is_auto_vacuum_region")
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
        names = ("x_min_um", "x_max_um", "y_min_um", "y_max_um")
        if any(not _finite(domain.get(name)) for name in names):
            missing = [name for name in names if not _finite(domain.get(name))]
            raise ValueError(
                f"solution region {semantic_id!r} has invalid {missing!r} for domain_bounds_um."
            )
        z_min = geometry.get("z_min_um", geometry.get("z_um"))
        if z_min is None or not _finite(z_min):
            raise ValueError(
                f"solution region {semantic_id!r} has missing or non-finite z_min_um."
            )
        z_min = float(z_min)
        z_max = geometry.get("z_max_um")
        if z_max is None:
            thickness = geometry.get("thickness_um")
            if thickness is None or not _finite(thickness):
                raise ValueError(
                    f"solution region {semantic_id!r} requires finite z_max_um or thickness_um."
                )
            z_max = z_min + float(thickness)
        elif not _finite(z_max):
            raise ValueError(
                f"solution region {semantic_id!r} has non-finite z_max_um."
            )
        else:
            z_max = float(z_max)
        if not z_min < z_max:
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
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
        raise TypeError("stack layers must be a sequence for auto vacuum envelope.")
    layer_ranges = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise TypeError(
                f"stack layer {index} must be a mapping for auto vacuum envelope."
            )
        layer_ranges.append(
            geometry_z_range(
                record_geometry(layer, str(layer.get("semantic_id", index))),
                str(layer.get("semantic_id", index)),
            )
        )
    return {
        "x_min_um": min(item["x_min_um"] for item in bounds),
        "x_max_um": max(item["x_max_um"] for item in bounds),
        "y_min_um": min(item["y_min_um"] for item in bounds),
        "y_max_um": max(item["y_max_um"] for item in bounds),
        "z_min_um": min(
            *(item["z_min_um"] for item in bounds),
            *(item[0] for item in layer_ranges),
        ),
        "z_max_um": max(
            *(item["z_max_um"] for item in bounds),
            *(item[1] for item in layer_ranges),
        ),
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
    kind = material.get("kind")
    if kind not in {"vacuum", "dielectric", "conductor"}:
        raise ValueError(f"solution region material {material_id!r} has invalid kind.")
    return str(kind)


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


__all__ = ["apply_vacuum_region_to_stack"]
