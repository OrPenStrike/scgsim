"""Project generated sheet interface intent from current source polygons. Compiler refresh uses this exact source projection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec


def _route_a_sheet_interfaces(
    entities: Sequence[SemanticEntitySpec],
    polygons: Sequence[LayoutPolygonSpec],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Build automatic Route A sheet intents from current source polygons."""
    polygons_by_id = {polygon.polygon_id: polygon for polygon in polygons}
    interfaces: list[Mapping[str, Any]] = []
    for entity in entities:
        if entity.route_representations.get("A") != "surface_sheet":
            continue
        for index, polygon_id in enumerate(entity.polygon_ids):
            polygon = polygons_by_id.get(polygon_id)
            if polygon is None:
                raise ValueError(
                    f"{entity.semantic_id} Route A sheet references missing "
                    f"polygon {polygon_id!r}"
                )
            z_um = float(entity.geometry.get("z_um", 0.0))
            interfaces.append(
                {
                    "interface_id": f"MA__{entity.semantic_id}__AIR__{index:04d}",
                    "kind": "MA",
                    "owner_semantic_ids": (entity.semantic_id, "AIR"),
                    "interface_kinds": ("MS", "MA"),
                    "recognition_rule": "route_a_surface_sheet_polygon",
                    "intent_origin": "generated_route_a_surface_sheet",
                    "source_polygon_ids": (polygon_id,),
                    "valid_routes": ("A",),
                    "plane": {"axis": "z", "value_um": z_um},
                    "outer_loop": polygon.exterior,
                    "hole_loops": polygon.holes,
                }
            )
    return {"interfaces": tuple(interfaces)}
