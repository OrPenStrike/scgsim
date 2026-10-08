"""Private Palace problem-input parsing shared by the public problem classes."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _non_negative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return result


def _load_stack(stack: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(stack, Mapping):
        return dict(stack)
    if isinstance(stack, Path):
        if not stack.is_file():
            raise FileNotFoundError(f"stack JSON path does not exist: {stack}")
        payload = json.loads(stack.read_text(encoding="utf-8"))
    elif isinstance(stack, str):
        stripped = stack.strip()
        if stripped.startswith(("{", "[")):
            payload = json.loads(stripped)
        else:
            path = Path(stack)
            try:
                exists = path.is_file()
            except OSError as exc:
                raise ValueError(
                    "stack string must be JSON text or an existing JSON path."
                ) from exc
            if not exists:
                raise FileNotFoundError(
                    "stack string must be JSON text or an existing JSON path."
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise TypeError("stack must be a mapping, JSON string, or JSON path.")
    if not isinstance(payload, Mapping):
        raise TypeError("stack payload must be a mapping.")
    return dict(payload)


def _validate_stack_material_kinds(
    stack: Mapping[str, Any], materials: Mapping[str, Mapping[str, Any]]
) -> None:
    def material_kind(material_id: Any, owner: str) -> str:
        if not isinstance(material_id, str) or material_id not in materials:
            raise ValueError(f"{owner} must reference an explicit stack material_id.")
        kind = materials[material_id].get("kind")
        if kind not in {"vacuum", "dielectric", "conductor"}:
            raise ValueError(
                f"material {material_id!r} requires kind vacuum, dielectric, or conductor."
            )
        return str(kind)

    regions = stack.get("solution_regions")
    if not isinstance(regions, Mapping) or not regions:
        raise ValueError("stack must define explicit solution_regions.")
    for name, region in regions.items():
        if not isinstance(region, Mapping) or material_kind(
            region.get("material_id"), f"solution region {name!r}"
        ) not in {"vacuum", "dielectric"}:
            raise ValueError(
                f"solution region {name!r} must use vacuum or dielectric material kind."
            )
    layers = stack.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("stack must define explicit layers.")
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise TypeError("stack layers must be mappings.")
        if (
            layer.get("role") == "metal"
            and material_kind(layer.get("material_id"), "metal layer") != "conductor"
        ):
            raise ValueError("metal layer material must have conductor kind.")


__all__ = ["_load_stack", "_non_negative_number", "_validate_stack_material_kinds"]
