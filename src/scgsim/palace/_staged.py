"""Shared staged simulation primitives for Palace candidates."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

RUNTIME_VERSION = "v0.16.1"
SCHEMA_VERSION = "v0.16.0"


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

    if len(matched) != 1:
        raise ValueError(
            "Airbox requires exactly one explicit is_airbox=true solution_region."
        )

    key, region = matched[0]
    geometry = region.get("geometry")
    if not isinstance(geometry, Mapping):
        raise TypeError(f"airbox solution region {key!r} must define geometry.")

    work = copy.deepcopy(dict(stack))
    work_regions = dict(work["solution_regions"])
    work_region = dict(region)
    work_geometry = dict(geometry)

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
        padding = work_geometry.get("padding_um")
        if not isinstance(padding, (int, float)) or isinstance(padding, bool):
            raise ValueError(f"airbox solution region {key!r} must define padding_um.")
        if float(padding) < 0.0:
            raise ValueError(f"airbox solution region {key!r} has invalid padding_um.")
        work_geometry["padding_um"] = float(padding) + margin_x

    if "z_below" in airbox:
        z_below = validate_positive_number(airbox["z_below"], "z_below")
        z_min = work_geometry.get("z_min_um")
        if not isinstance(z_min, (int, float)) or isinstance(z_min, bool):
            raise ValueError(f"airbox solution region {key!r} must define z_min_um.")
        work_geometry["z_min_um"] = float(z_min) - z_below

    if "z_above" in airbox:
        z_above = validate_positive_number(airbox["z_above"], "z_above")
        z_max = work_geometry.get("z_max_um")
        if not isinstance(z_max, (int, float)) or isinstance(z_max, bool):
            raise ValueError(f"airbox solution region {key!r} must define z_max_um.")
        work_geometry["z_max_um"] = float(z_max) + z_above

    work_region["geometry"] = work_geometry
    work_regions[key] = work_region
    work["solution_regions"] = work_regions
    return work
