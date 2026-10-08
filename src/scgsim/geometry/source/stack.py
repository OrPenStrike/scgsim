"""Build stack inputs from component/PDK declarations. Shared normalization is defined once below this entrypoint."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.source._normalization import (
    _coupon_domain_bounds,
    _finite_number,
    _mapping,
    _materials,
    _semantic_layer_records,
    _solution_regions,
)


def build_component_stack(
    *,
    component: Any,
    layer_stack: Any,
    material_records: Mapping[str, Mapping[str, Any]],
    coupon_padding_um: float,
) -> dict[str, Any]:
    """Compile PDK process facts and component topology into one SGB stack.

    The PDK layer stack owns material, fabrication, host-volume, and 3D
    integration semantics. The component owns conductor identities, nets, and
    topology selectors. Geometry bounds size the coupon only.
    """
    padding = _finite_number(coupon_padding_um, "coupon_padding_um")
    if padding < 0.0:
        raise ValueError("coupon_padding_um must be a finite non-negative number.")
    bounds = _coupon_domain_bounds(component, padding)
    try:
        spec = _mapping(
            component.info["component_semantics"], "component component_semantics"
        )
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            "component must provide an authored info['component_semantics'] annotation."
        ) from exc
    if spec.get("schema_version") != 1:
        raise ValueError("component_semantics schema_version must be 1.")
    unknown = set(spec) - {
        "schema_version",
        "conductor_regions",
        "boundary_curves",
        "boundary_reconstruction",
        "metadata",
    }
    if unknown:
        raise ValueError(
            f"component_semantics has unsupported fields {sorted(unknown)!r}."
        )
    conductor_regions = spec.get("conductor_regions")
    metadata = spec.get("metadata")
    levels = getattr(layer_stack, "layers", None)
    if not isinstance(levels, Mapping):
        raise TypeError("layer_stack must expose a mapping 'layers'.")
    if isinstance(conductor_regions, str | bytes) or not isinstance(
        conductor_regions, Sequence
    ):
        raise TypeError("component conductor_regions must be a sequence of mappings.")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping when provided.")

    materials = _materials(material_records)
    regions = _solution_regions(levels, materials=materials, bounds=bounds)
    if not regions:
        raise ValueError(
            "PDK layer_stack must declare at least one included solution region."
        )
    layers = _semantic_layer_records(
        conductor_regions,
        levels=levels,
        materials=materials,
        solution_region_ids=regions,
    )
    return {
        "solution_regions": regions,
        "materials": materials,
        "layers": layers,
        "metadata": {
            **dict(metadata or {}),
            "adapter": "scgsim.sgb.stack.build_component_stack",
            "boundary_curves": copy.deepcopy(spec.get("boundary_curves", ())),
            "boundary_reconstruction": copy.deepcopy(
                spec.get("boundary_reconstruction", ())
            ),
            "coupon_domain_bounds_um": dict(bounds),
            "coupon_padding_um": padding,
        },
    }
