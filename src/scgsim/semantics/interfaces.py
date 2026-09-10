"""Pure interface classification and solution-owner ordering."""

from __future__ import annotations

from .materials import is_supported_material_kind, is_vacuum_material_kind


def conductor_solution_interface_kind(solution_material_kind: str) -> str:
    if not is_supported_material_kind(solution_material_kind):
        raise ValueError("solution material kind must be vacuum, dielectric, or conductor")
    return "MA" if is_vacuum_material_kind(solution_material_kind) else "MS"


def solution_interface_kind(lower_material_kind: str, upper_material_kind: str) -> str:
    if not is_supported_material_kind(lower_material_kind) or not is_supported_material_kind(
        upper_material_kind
    ):
        raise ValueError("solution material kinds must be vacuum, dielectric, or conductor")
    lower_vacuum = is_vacuum_material_kind(lower_material_kind)
    upper_vacuum = is_vacuum_material_kind(upper_material_kind)
    if lower_vacuum or upper_vacuum:
        return "AA" if lower_vacuum and upper_vacuum else "SA"
    return "SS"


def solution_interface_owner_ids(
    kind: str,
    lower_id: str,
    upper_id: str,
    *,
    lower_is_vacuum: bool,
) -> tuple[str, str]:
    if kind == "SA" and lower_is_vacuum:
        return (upper_id, lower_id)
    return (lower_id, upper_id)
