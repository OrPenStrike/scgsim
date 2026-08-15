"""Construction-plan records.

These records explain how route-specific surfaces, construction bodies,
physical-tag intent, and the aggregate handoff plan are assembled before
backend lowering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from scgsim.sgb.models.common import (
    ConductorRepresentationLiteral,
    RouteLiteral,
)
from scgsim.sgb.models.regions import PortSheetRegionRecord
from scgsim.sgb.models.tags import BackendEntityTagRecord, TagPlanRecord
from scgsim.sgb.models.topology import (
    CurvePlanRecord,
    InterfacePlanRecord,
    MMContactRecord,
    PointPlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
    VolumePlanRecord,
)


@dataclass(frozen=True)
class SurfacePartitionRecord:
    """Partition intent for one child surface of a parent interface.

    This is not a backend geometry object and is not a physical group source.
    It tells the planner why a parent interface is represented by one or more
    child live surfaces. The child must be directly buildable; overlay masks are
    not supported.
    """

    partition_id: str
    parent_interface_id: str
    child_surface_id: str
    label: str
    valid_routes: tuple[RouteLiteral, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstructionBodyPlanRecord:
    """Route A/B temporary body used to derive final cutout shell surfaces.

    This is not final solver geometry and must not receive a physical group.
    It records which semantic conductor blocks a host solution region and which
    final shell surfaces are expected to survive after the backend cut.
    """

    construction_body_id: str
    owner_semantic_id: str
    host_semantic_id: str
    representation: ConductorRepresentationLiteral
    geometry_ref: Mapping[str, Any]
    expected_surface_ids: tuple[str, ...] = ()
    valid_routes: tuple[RouteLiteral, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CutHostOperationRecord:
    """Route A/B operation that cuts one host with construction bodies.

    Compatible operation records may later be batched by the backend, but the
    semantic operation identity and member construction bodies must remain
    recoverable for provenance.
    """

    operation_id: str
    host_semantic_id: str
    construction_body_ids: tuple[str, ...]
    exposed_surface_ids: tuple[str, ...]
    valid_routes: tuple[RouteLiteral, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstructionPlanRecord:
    """Route-aware handoff from semantic planning to bottom-up OCC creation.

    This is the complete input to the backend. `interfaces` explain why shared
    surfaces exist, `surface_partitions` explain parent-to-child ownership,
    `points`, `curves`, and `surface_loops` make shared topology canonical,
    `surfaces` and `volumes` describe geometry to build,
    `construction_bodies` and `cut_operations` describe Route A/B host cuts,
    `port_sheet_regions` carry explicit 2D lumped-port overlap intent, and
    `tags` define physical names before backend dim-tags exist.

    After OCC construction, the backend returns the same plan with
    `backend_entity_tags` populated. `FinalPhysicalGroupRecord`s are then a
    deterministic projection of `tags + backend_entity_tags`.
    """

    route: RouteLiteral
    interfaces: tuple[InterfacePlanRecord, ...] = ()
    surface_partitions: tuple[SurfacePartitionRecord, ...] = ()
    points: tuple[PointPlanRecord, ...] = ()
    curves: tuple[CurvePlanRecord, ...] = ()
    surface_loops: tuple[SurfaceLoopRecord, ...] = ()
    surfaces: tuple[SurfacePlanRecord, ...] = ()
    volumes: tuple[VolumePlanRecord, ...] = ()
    construction_bodies: tuple[ConstructionBodyPlanRecord, ...] = ()
    cut_operations: tuple[CutHostOperationRecord, ...] = ()
    tags: tuple[TagPlanRecord, ...] = ()
    backend_entity_tags: tuple[BackendEntityTagRecord, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    port_sheet_regions: tuple[PortSheetRegionRecord, ...] = ()


@dataclass(frozen=True)
class RouteABConstructionPlanRecord(ConstructionPlanRecord):
    """A/B-only extension; base Route-C record remains legacy-shaped."""

    mm_contacts: tuple[MMContactRecord, ...] = ()
