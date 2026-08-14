"""Physical group and backend tag ledger records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from scgsim.sgb.models.common import (
    DimensionLiteral,
    GmshDimTag,
    RouteLiteral,
    SolverUseLiteral,
    TagSourceKindLiteral,
)


@dataclass(frozen=True)
class TagPlanRecord:
    """Solver-neutral physical tag plan before backend entity tags exist.

    Tags are planned before OCC construction. `source_record_kind` and
    `source_record_id` must point to a live `SurfacePlanRecord` or
    `VolumePlanRecord`. After backend construction, the same source id is used
    to recover OCC dim-tags and create final physical groups. Together with
    `BackendEntityTagRecord`, this is the tag ledger: plan ids are stable,
    OCC tags are lowering results, and physical names are solver-facing labels.
    """

    physical_name: str
    dimension: DimensionLiteral
    source_record_kind: TagSourceKindLiteral
    source_record_id: str
    role: str
    solver_use: SolverUseLiteral = "solver_active"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_dimension = 2 if self.source_record_kind == "surface" else 3
        if self.dimension != expected_dimension:
            raise ValueError(
                f"{self.source_record_kind} tag {self.physical_name!r} "
                f"must use dimension {expected_dimension}"
            )


@dataclass(frozen=True)
class BackendEntityTagRecord:
    """Backend dim-tag recovered for one live planned source record.

    Two live source ids mapping to the same backend dim-tag means
    canonicalization failed earlier; do not accept that as a backend shortcut.
    """

    source_record_kind: TagSourceKindLiteral
    source_record_id: str
    dim_tag: GmshDimTag
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalPhysicalGroupRecord:
    """Final solver-neutral group plan after backend tags are known.

    `physical_name` is part of the downstream contract. Interface surface names
    currently carry the reviewed grammar consumed by gsim, for example
    `MA__...`, `MS__...`, `SA__...`, and `MS_MA__...`.
    """

    physical_name: str
    dimension: int
    route: RouteLiteral
    role: str
    source_record_id: str
    net_id: str | None = None
    solver_use: SolverUseLiteral | None = None
    entity_tags: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.solver_use is None:
            object.__setattr__(self, "solver_use", "solver_active")
        if not self.entity_tags:
            raise ValueError("physical groups require backend entity tags")


@dataclass(frozen=True)
class StructuredFinalPhysicalGroupRecord(FinalPhysicalGroupRecord):
    """A/B-only solver-live structured surface authority."""

    representation: str = ""
    surface_id: str | None = None
    interface_type: str | None = None
    contact_kind: str | None = None
    face_kind: str | None = None
    owner_semantic_ids: tuple[str, ...] = ()
    adjacent_solution_volume_ids: tuple[str, ...] = ()
    conductor_component_id: str | None = None
    equipotential_id: str | None = None
    source_provenance: Mapping[str, Any] = field(default_factory=dict)
    physical_attribute: Mapping[str, Any] = field(default_factory=dict)
