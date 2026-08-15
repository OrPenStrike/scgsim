"""2D region-layer records owned by SGB before surface planning.

These records capture the small set of intentional 2D overlaps that are legal
input semantics. Solver-live surfaces must still be non-overlapping after route
planning; region records only explain what the planner caught before that stage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from scgsim.sgb.models.common import PolygonRing


@dataclass(frozen=True)
class PortSheetOverlapRecord:
    """One overlap cell between a Palace lumped-port sheet and a host polygon.

    Palace lumped-port sheets are the only intentional 2D overlay currently
    supported. Each overlap is explicit so later lowering can perform a local
    fragment operation without turning global overlap into accepted geometry.
    """

    overlap_id: str
    port_sheet_id: str
    port_polygon_id: str
    host_semantic_id: str
    host_polygon_id: str
    overlap_loop: PolygonRing
    operation: str = "local_fragment_required"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortSheetRegionRecord:
    """Adapter-owned Palace lumped-port source polygon and its host overlaps.

    This is not a backend-live surface and must not be exported as a physical
    surface group by itself. It is the 2D region-layer contract that says which
    source polygons are allowed to overlap and where local fragmentation will be
    needed when port-sheet lowering is implemented.
    """

    port_sheet_id: str
    source_layer: str
    source_polygon_id: str
    exterior: PolygonRing
    holes: tuple[PolygonRing, ...] = ()
    overlaps: tuple[PortSheetOverlapRecord, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
