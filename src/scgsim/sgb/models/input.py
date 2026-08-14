"""Adapter-boundary records.

These records are produced before route planning. They preserve frontend
provenance and semantic stack intent, but they do not contain backend tags or
canonical topology.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from scgsim.sgb.models.common import (
    ConductorPartRoleLiteral,
    ConductorRepresentationLiteral,
    PolygonRing,
    RouteLiteral,
)
from scgsim.sgb.models.regions import PortSheetRegionRecord


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
