"""Valid-value material predicates; source adapters retain error contracts."""

from __future__ import annotations

from typing import Final

MATERIAL_KINDS: Final = frozenset({"vacuum", "dielectric", "conductor"})


def is_supported_material_kind(value: object) -> bool:
    return isinstance(value, str) and value in MATERIAL_KINDS


def is_vacuum_material_kind(value: object) -> bool:
    return value == "vacuum"
