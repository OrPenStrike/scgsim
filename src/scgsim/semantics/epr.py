"""Pure film assumptions shared by solver-specific EPR readers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


FILM_FIELDS = frozenset(
    {
        "film_thickness_m",
        "film_relative_permittivity",
        "loss_tangent",
        "source",
        "preset",
    }
)


def film_assumptions(
    value: Mapping[str, Any], *, partial: bool = False, allow_unknown: bool = False
) -> dict[str, Any]:
    """Validate an SI film snapshot or a partial offline update."""

    if not isinstance(value, Mapping) or set(value) - FILM_FIELDS:
        raise ValueError("film assumptions contain unsupported fields")
    if not partial and set(value) != FILM_FIELDS:
        raise ValueError("film assumptions require all snapshot fields")
    result = dict(value)
    for name in ("film_thickness_m", "film_relative_permittivity"):
        if name not in result:
            continue
        number = result[name]
        if number is None and allow_unknown:
            continue
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
        result[name] = float(number)
    if "loss_tangent" in result and result["loss_tangent"] is not None:
        number = result["loss_tangent"]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number < 0
        ):
            raise ValueError("loss_tangent must be finite and non-negative or None")
        result["loss_tangent"] = float(number)
    for name in ("source", "preset"):
        if (
            name in result
            and result[name] is not None
            and (not isinstance(result[name], str) or not result[name].strip())
        ):
            raise ValueError(f"{name} must be non-empty text or None")
    return result


def inverse_quality_factor(
    participation: float | None, loss_tangent: float | None
) -> float | None:
    """Return p tan(delta), preserving an unknown input as unknown."""

    if participation is None or loss_tangent is None:
        return None
    if (
        isinstance(participation, bool)
        or not isinstance(participation, (int, float))
        or not math.isfinite(participation)
        or participation < 0
    ):
        raise ValueError("participation must be finite and non-negative or None")
    film_assumptions({"loss_tangent": loss_tangent}, partial=True)
    return float(participation) * float(loss_tangent)
