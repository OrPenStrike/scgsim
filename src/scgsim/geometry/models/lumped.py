"""Backend-neutral source support. Treatments never own or infer this geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LumpedSupport:
    """Occurrence-local nonmetal polygon and ordered, explicitly owned contacts.

    GeometryPlan resolves the original polygon and records the occurrence affine
    once. Coordinates remain local source facts; consumers derive world/effective
    coordinates rather than rewriting the source into a solver boundary.
    """

    support_id: str
    source_polygon_id: str
    source_occurrence_path: str
    source_layer: tuple[int, int]
    source_level: str
    contact_physical_layer_id: str
    normal: tuple[float, float, float]
    contact_points_um: tuple[tuple[float, float, float], tuple[float, float, float]]
    terminal_a_entity_ids: tuple[str, ...]
    terminal_b_entity_ids: tuple[str, ...]
    exterior: tuple[tuple[float, float], ...]
    holes: tuple[tuple[tuple[float, float], ...], ...]
    source_z_um: float | None = None
    source_transform: tuple[float, float, float, float, float, float] = (
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    )

    def __post_init__(self) -> None:
        for key in ("support_id", "source_polygon_id", "source_occurrence_path", "source_level", "contact_physical_layer_id"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{key} requires explicit source identity")
        for key, size in (("source_layer", 2), ("normal", 3), ("source_transform", 6)):
            values = tuple(getattr(self, key))
            if len(values) != size:
                raise ValueError(f"{key} requires {size} values")
            object.__setattr__(self, key, values)
        points = tuple(tuple(point) for point in self.contact_points_um)
        if len(points) != 2 or any(len(point) != 3 for point in points):
            raise ValueError("contact_points_um requires two ordered XYZ points")
        object.__setattr__(self, "contact_points_um", points)
        for key in ("terminal_a_entity_ids", "terminal_b_entity_ids"):
            values = tuple(getattr(self, key))
            if not values or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{key} requires qualified Entity identities")
            object.__setattr__(self, key, values)
        object.__setattr__(self, "exterior", tuple(tuple(p) for p in self.exterior))
        object.__setattr__(self, "holes", tuple(tuple(tuple(p) for p in ring) for ring in self.holes))
        for ring in (self.exterior, *self.holes):
            if len(ring) < 3 or any(len(point) != 2 for point in ring):
                raise ValueError("lumped support requires complete local XY polygon rings")

    def world_point(self, point: tuple[float, ...]) -> tuple[float, ...]:
        a, b, c, d, x, y = self.source_transform
        result = (a * point[0] + b * point[1] + x, c * point[0] + d * point[1] + y)
        return result if len(point) == 2 else (*result, point[2])

    def to_payload(self) -> dict[str, Any]:
        return {
            "support_id": self.support_id, "source_polygon_id": self.source_polygon_id,
            "source_occurrence_path": self.source_occurrence_path,
            "source_layer": list(self.source_layer), "source_level": self.source_level,
            "contact_physical_layer_id": self.contact_physical_layer_id,
            "normal": list(self.normal),
            "contact_points_um": [list(point) for point in self.contact_points_um],
            "terminal_a_entity_ids": list(self.terminal_a_entity_ids),
            "terminal_b_entity_ids": list(self.terminal_b_entity_ids),
            "exterior": [list(point) for point in self.exterior],
            "holes": [[list(point) for point in ring] for ring in self.holes],
            "source_transform": list(self.source_transform),
            "source_z_um": self.source_z_um,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LumpedSupport:
        return cls(**dict(payload))
