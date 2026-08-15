"""Interface, topology, surface, and volume records.

These records are the main semantic-to-topology ladder. They remain together
because surfaces and volumes are not useful without the interface and loop
identity that makes them conformal before backend lowering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from scgsim.sgb.models.common import (
    DEFAULT_INTERFACE_SOLVER_USE,
    CurveKindLiteral,
    CurveOrientationLiteral,
    InterfaceKindLiteral,
    RouteLiteral,
    SolverUseLiteral,
    SurfaceLoopRoleLiteral,
    SurfaceOrientationLiteral,
    Vector3D,
)


@dataclass(frozen=True)
class InterfacePlanRecord:
    """Pre-construction declaration that two semantic owners share a surface.

    `interface_id` is the stable semantic id and must start with the interface
    kind directly, such as `MM__`, `MS__`, `MA__`, `SA__`, `SS__`, or `AA__`.
    There is no extra `IF__` prefix.

    Every interface must be recognized before OCC geometry creation. This
    record only states the semantic fact: which two owners meet, what kind of
    interface it is, and which recognition rule proved it. Backend-live
    geometry is created only from `SurfacePlanRecord`.

    Required metadata for bottom-up OCC planning should be attached here as
    soon as it is recognized: `contact_plane` for the physical plane of contact,
    `footprint` for projected contact geometry, and `loop_geometry_ref` when the
    interface is already known as an explicit loop source.

    Route A surface sheets may need side-aware semantics on the same geometric
    loop. Use metadata such as `surface_side`, `adjacent_domain_id`,
    `excluded_owner_semantic_id`, and `interface_kinds` to say which side is
    MA/MS/MM without creating overlapping backend surfaces.
    """

    interface_id: str
    kind: InterfaceKindLiteral
    owner_semantic_ids: tuple[str, str]
    recognition_rule: str
    source_polygon_ids: tuple[str, ...] = ()
    host_domain_id: str | None = None
    solver_use: SolverUseLiteral | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.interface_id.startswith(f"{self.kind}__"):
            raise ValueError(
                f"interface_id {self.interface_id!r} must start with {self.kind}__"
            )
        if self.solver_use is None:
            object.__setattr__(
                self,
                "solver_use",
                DEFAULT_INTERFACE_SOLVER_USE[self.kind],
            )


@dataclass(frozen=True)
class SurfacePlanRecord:
    """One backend-live planned surface to create before volume assembly.

    Shared interfaces are represented by one `surface_id`, and both owning
    volumes must reference that same id. Partitioned interfaces are already
    expanded into child `SurfacePlanRecord`s before backend lowering. The
    backend must create each child surface directly from its loops and hole
    loops; it must not cut a parent surface later.

    v1 backend lowering should consume `outer_loop_ref` and `hole_loop_refs`.
    `geometry_ref` remains as audit/source metadata: it may carry `plane` or
    `contact_plane`, `outer_loop`, optional `hole_loops`, optional `footprint`,
    and optional `loop_geometry_ref`. A surface that cannot be tied to canonical
    loop refs must fail during canonicalization rather than falling back to
    global fragment-based discovery.

    `metadata` may carry side/exposure semantics such as `interface_kinds`,
    `surface_side`, `owner_semantic_ids`, `adjacent_domain_id`,
    and `excluded_owner_semantic_id`. Route A surface-sheet conductors must be
    represented by these interface surfaces, not by extra standalone sheet
    surfaces.
    When a surface belongs to more than one geometry interface kind, physical
    names must include the joined interface kinds, for example `MM_MS__...`.
    """

    surface_id: str
    owner_semantic_id: str
    surface_role: str
    geometry_ref: Mapping[str, Any]
    outer_loop_ref: str | None = None
    hole_loop_refs: tuple[str, ...] = ()
    interface_id: str | None = None
    parent_surface_id: str | None = None
    partition_label: str | None = None
    normal_hint: Vector3D | None = None
    valid_routes: tuple[RouteLiteral, ...] = ()
    solver_use: SolverUseLiteral = "solver_active"
    construction_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SurfaceRefRecord:
    """Reference to one planned surface from a volume surface loop."""

    surface_id: str
    orientation: SurfaceOrientationLiteral
    role: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MMContactRecord:
    """Hidden finite-area same-net conductor contact provenance.

    Contacts are topology records, never physical groups.  They preserve the
    original faces/fragments that were normalized into one PEC component while
    keeping Route A/B solver geometry free of an internal contact boundary.
    """

    contact_id: str
    lower_entity_id: str
    upper_entity_id: str
    lower_source_face_id: str
    upper_source_face_id: str
    lower_source_fragment_ids: tuple[str, ...]
    upper_source_fragment_ids: tuple[str, ...]
    outer_loop: tuple[tuple[float, float], ...]
    area_um2: float
    normal: Vector3D
    conductor_component_id: str
    net_id: str | None
    equipotential_id: str | None = None
    layer_provenance: Mapping[str, Any] = field(default_factory=dict)
    material_provenance: Mapping[str, Any] = field(default_factory=dict)
    source_provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InnerPecVoidShellRecord:
    """One oriented inner PEC shell excluded from a solution volume."""

    shell_id: str
    surface_refs: tuple[SurfaceRefRecord, ...]
    conductor_component_id: str
    owner_semantic_ids: tuple[str, ...]


@dataclass(frozen=True)
class PointPlanRecord:
    """One canonical planned point shared by all touching curves.

    A conformal v1 plan may not contain two live point records at the same
    coordinate. Points are topology/audit records, not physical groups.
    """

    point_id: str
    coordinate: Vector3D
    owner_semantic_ids: tuple[str, ...] = ()
    interface_ids: tuple[str, ...] = ()
    used_by_curve_ids: tuple[str, ...] = ()
    boundary_volume_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurvePlanRecord:
    """One canonical planned curve shared by all touching surfaces.

    Curves are topology/audit/mesh-refinement contract records, not physical
    groups. `used_by_surface_ids`, `interface_ids`, and `boundary_volume_ids`
    describe how the curve participates in the final topology without forcing a
    single semantic owner for shared edges.
    """

    curve_id: str
    curve_kind: CurveKindLiteral
    start_point_id: str
    end_point_id: str
    owner_semantic_ids: tuple[str, ...] = ()
    interface_ids: tuple[str, ...] = ()
    used_by_surface_ids: tuple[str, ...] = ()
    boundary_volume_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurveRefRecord:
    """Oriented reference to one canonical curve from a surface loop."""

    curve_id: str
    orientation: CurveOrientationLiteral
    role: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SurfaceLoopRecord:
    """Canonical loop used by a planned surface.

    A surface owns one outer loop and zero or more hole loops by reference.
    The loop owns ordered `CurveRefRecord`s so shared boundaries are explicit
    before OCC lowering.
    """

    loop_id: str
    curve_refs: tuple[CurveRefRecord, ...]
    role: SurfaceLoopRoleLiteral
    surface_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VolumePlanRecord:
    """One planned volume assembled from already-planned surfaces.

    Backend-live volumes must be created from `surface_refs`: the planner first
    controls every boundary surface, and the backend then calls
    `addSurfaceLoop()` plus `addVolume()` on those surfaces. `metadata` may keep
    domain bounds, material, or stack provenance for audit, but those records
    are not permission to create a box or extruded fallback volume.

    Route C material volumes are backend-live solver geometry. Route A/B
    construction cutters are not represented here; they use
    `ConstructionBodyPlanRecord` so retained material volumes and temporary
    cutter bodies cannot be confused.
    """

    volume_id: str
    owner_semantic_id: str
    material_id: str
    surface_refs: tuple[SurfaceRefRecord, ...]
    valid_routes: tuple[RouteLiteral, ...] = ()
    construction_only: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteABVolumePlanRecord(VolumePlanRecord):
    """A/B-only solution-shell extension; base Route-C shape is unchanged."""

    exterior_surface_refs: tuple[SurfaceRefRecord, ...] = ()
    inner_pec_void_shells: tuple[InnerPecVoidShellRecord, ...] = ()
