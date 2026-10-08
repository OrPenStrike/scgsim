"""Canonical geometry records. Keep source facts distinct from compiler records; reexports share these exact definitions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from scgsim.geometry.models.common import (
    ConductorPartRoleLiteral,
    ConductorRepresentationLiteral,
    PolygonRing,
    RouteLiteral,
)
from scgsim.geometry.models.regions import PortSheetRegionRecord


@dataclass(frozen=True)
class LayoutPolygonSpec:
    """Adapter-normalized polygon with stable frontend provenance."""

    polygon_id: str
    layer: str
    exterior: PolygonRing
    holes: tuple[PolygonRing, ...] = ()
    object_name: str | None = None
    net_name: str | None = None
    port_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceCurveSpec:
    """Component-authored XY geometry; control points are not topology vertices.

    A circular arc uses start, on-arc, and end points. B-spline knots are
    distinct values with corresponding multiplicities; an optional interval
    trims the native source parameter domain without changing its geometry.
    """

    kind: str
    points_um: tuple[tuple[float, float], ...]
    degree: int | None = None
    weights: tuple[float, ...] = ()
    knots: tuple[float, ...] = ()
    multiplicities: tuple[int, ...] = ()
    parameter_interval: tuple[float, float] | None = None


@dataclass(frozen=True)
class BoundaryCurveChainSpec:
    """Ordered source boundary bound to one Entity-owned source polygon.

    The selector chooses a polygon in that Entity, not an arbitrary surface.
    Hole indices address that selected polygon's recorded hole order.
    """

    boundary_id: str
    entity_id: str
    selector_point_um: tuple[float, float]
    curves: tuple[SourceCurveSpec, ...]
    role: str = "outer"
    hole_index: int | None = None
    source_occurrence_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GdsBoundaryReconstructionSpec:
    """Explicit caller selection of a polygon boundary reconstruction.

    Circular reconstruction uses three caller-selected vertex indices. Spline
    reconstruction uses the recorded boundary vertices, or explicit selected
    indices. No polygon-only input is implicitly reconstructed.
    """

    boundary_id: str
    entity_id: str
    selector_point_um: tuple[float, float]
    kind: str
    role: str = "outer"
    hole_index: int | None = None
    segment_indices: tuple[int, ...] = ()
    closed: bool = False
    anchor_indices: tuple[int, ...] = ()
    degree: int | None = None
    parameter_interval: tuple[float, float] | None = None
    source_occurrence_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VacuumRegionSpec:
    """Cross-layer vacuum padding specification for auto envelope derivation."""

    x_minus_um: float
    x_plus_um: float
    y_minus_um: float
    y_plus_um: float
    z_minus_um: float
    z_plus_um: float

    @classmethod
    def from_padding(
        cls,
        value: float | Sequence[float] | Mapping[str, Any],
    ) -> VacuumRegionSpec:
        """Normalize scalar, 3-sequence, or sparse six-face vacuum padding."""
        if isinstance(value, bool) or value is None:
            raise TypeError(
                "vacuum padding must be a number, 3-sequence, or six-face mapping."
            )

        if isinstance(value, (int, float)):
            base = _non_negative_float(value, "padding")
            return cls(base, base, base, base, base, base)

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 3:
                raise ValueError(
                    "vacuum 3-sequence padding must contain exactly 3 items for x/y/z."
                )
            x_padding, y_padding, z_padding = (
                _non_negative_float(value[0], "x_padding"),
                _non_negative_float(value[1], "y_padding"),
                _non_negative_float(value[2], "z_padding"),
            )
            return cls(
                x_padding,
                x_padding,
                y_padding,
                y_padding,
                z_padding,
                z_padding,
            )

        if not isinstance(value, Mapping):
            raise TypeError("vacuum padding must be a number, 3-sequence, or mapping.")

        required = {
            "x_minus_um",
            "x_plus_um",
            "y_minus_um",
            "y_plus_um",
            "z_minus_um",
            "z_plus_um",
        }
        keys = set(value.keys())
        if not keys <= required:
            extras = tuple(sorted(keys - required))
            raise ValueError(
                f"vacuum face padding mapping has unsupported keys: {extras!r}."
            )

        face_map = {
            key: _non_negative_float(value.get(key, 0.0), key) for key in required
        }
        return cls(**{key: value for key, value in face_map.items()})


def _non_negative_float(value: Any, field: str) -> float:
    if value is None:
        raise ValueError(f"vacuum padding field {field!r} is required.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"vacuum padding field {field!r} must be numeric.")
    text = float(value)
    if not math.isfinite(text):
        raise ValueError(f"vacuum padding field {field!r} must be finite.")
    if text < 0.0:
        raise ValueError(f"vacuum padding field {field!r} must be non-negative.")
    return text


@dataclass(frozen=True)
class SemanticEntitySpec:
    """Stable semantic object before route-aware construction planning."""

    semantic_id: str
    role: str
    material_id: str
    material_kind: str
    priority: int
    geometry_kind: str
    part_role: ConductorPartRoleLiteral | None = None
    attached_face_metal_semantic_id: str | None = None
    net_id: str | None = None
    polygon_ids: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    host_void_semantic_id: str | None = None
    requires_construction_body: bool = False
    route_representations: Mapping[
        RouteLiteral,
        ConductorRepresentationLiteral,
    ] = field(default_factory=dict)
    geometry: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryBuildInput:
    """Adapter boundary for route-aware semantic geometry construction.

    `port_sheet_regions` are pre-planning 2D region-layer records for Palace
    lumped-port sheets. They are not backend topology and must not become live
    surfaces without an explicit local-fragment lowering step.
    """

    polygons: tuple[LayoutPolygonSpec, ...]
    entities: tuple[SemanticEntitySpec, ...]
    solution_regions: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    port_sheet_regions: tuple[PortSheetRegionRecord, ...] = ()
    boundary_curves: tuple[BoundaryCurveChainSpec, ...] = ()
    boundary_reconstruction: tuple[GdsBoundaryReconstructionSpec, ...] = ()
