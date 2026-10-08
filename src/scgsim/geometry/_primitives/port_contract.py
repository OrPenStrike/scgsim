"""Single existing Route B sheet-binding predicate shared by planning and conformity evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _route_b_port_sheet_binding_matches(
    surface: Any,
    *,
    port_surface_id: str,
    overlap_id: str | None,
    host_semantic_id: str,
) -> bool:
    return any(
        isinstance(binding, Mapping)
        and binding.get("port_surface_id") == port_surface_id
        and binding.get("overlap_id") == overlap_id
        and binding.get("host_semantic_id") == host_semantic_id
        for binding in surface.metadata.get("route_b_port_sheet_bindings", ())
    )
