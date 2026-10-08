"""Project semantic physical names and tag records from planned surfaces/volumes, preserving grouped ownership."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from scgsim.geometry._primitives.constants import _INTERFACE_KIND_ORDER
from scgsim.geometry.models.common import (
    HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES,
    RouteLiteral,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.tags import TagPlanRecord
from scgsim.geometry.models.topology import SurfacePlanRecord, VolumePlanRecord
from scgsim.geometry.planning.topology import _surface_owner_ids
from scgsim.semantics.ownership import (
    physical_group_owner_ids,
    surface_physical_owner_ids,
)


def plan_route_tags(
    *,
    route: RouteLiteral,
    surfaces: tuple[SurfacePlanRecord, ...],
    volumes: tuple[VolumePlanRecord, ...],
) -> tuple[TagPlanRecord, ...]:
    """Plan physical names before backend entity tags exist.

    Interface surfaces and domain exterior boundary surfaces are exported as
    solver-visible surface physical groups. Other construction-only surfaces
    remain backend topology.
    """
    tags: list[TagPlanRecord] = []
    tags.extend(
        TagPlanRecord(
            physical_name=_volume_physical_name(volume),
            dimension=3,
            source_record_kind="volume",
            source_record_id=volume.volume_id,
            role="material_volume",
        )
        for volume in volumes
        if not volume.construction_only
    )
    tags.extend(
        TagPlanRecord(
            physical_name=_surface_physical_name(surface),
            dimension=2,
            source_record_kind="surface",
            source_record_id=surface.surface_id,
            role=surface.surface_role,
            solver_use=surface.solver_use,
        )
        for surface in surfaces
        if not surface.construction_only
        and not surface.metadata.get("hidden_solver_contact")
        and (
            surface.interface_id is not None
            or surface.surface_role == "domain_boundary"
            or surface.surface_role == "lumped_port"
        )
    )
    return tuple(tags)


def _volume_physical_name(volume: VolumePlanRecord) -> str:
    override = volume.metadata.get("physical_name")
    if isinstance(override, str) and override:
        return override
    physical_owner_id = volume.metadata.get("physical_owner_semantic_id")
    if isinstance(physical_owner_id, str) and physical_owner_id:
        return physical_owner_id
    return volume.owner_semantic_id


def _surface_physical_name(surface: SurfacePlanRecord) -> str:
    override = surface.metadata.get("physical_name")
    if isinstance(override, str) and override:
        return override
    raw = surface.metadata.get("interface_kinds", ())
    if isinstance(raw, str):
        kinds = (raw,)
    elif isinstance(raw, tuple | list | set):
        kinds = tuple(str(kind) for kind in raw)
    else:
        kinds = ()
    if len(kinds) > 2:
        raise ValueError(
            f"{surface.surface_id} has too many interface kinds: {kinds!r}"
        )
    kind_prefix = "_".join(kinds)
    exposed_role = surface.metadata.get("exposed_surface_role")
    boundary_role = surface.metadata.get("boundary_role")
    owner_ids = _surface_owner_ids(surface)
    physical_owner_ids = _surface_physical_owner_ids(surface)
    if physical_owner_ids != owner_ids:
        return _grouped_surface_physical_name(
            surface,
            kind_prefix=kind_prefix,
            physical_owner_ids=physical_owner_ids,
        )
    if surface.interface_id is not None:
        parts = surface.interface_id.split("__")
        if isinstance(exposed_role, str) and exposed_role.startswith("sidewall_"):
            parts[-1] = "SIDEWALL"
        if parts and parts[0] in _INTERFACE_KIND_ORDER:
            return "__".join((kind_prefix or parts[0], *parts[1:]))
        return surface.interface_id
    if kind_prefix:
        owner_ids = surface.metadata.get(
            "owner_semantic_ids",
            (surface.owner_semantic_id,),
        )
        if isinstance(owner_ids, str):
            owner_ids = (owner_ids,)
        suffix = exposed_role
        parts = [kind_prefix, *(str(owner_id) for owner_id in owner_ids)]
        if suffix:
            parts.append(str(suffix).upper())
        return "__".join(parts)
    if boundary_role == "sidewall":
        return surface.surface_id.removeprefix("SURF__").rsplit("__", 1)[0]
    return surface.surface_id.removeprefix("SURF__")


def _grouped_surface_physical_name(
    surface: SurfacePlanRecord,
    *,
    kind_prefix: str,
    physical_owner_ids: tuple[str, ...],
) -> str:
    exposed_role = surface.metadata.get("exposed_surface_role")
    suffix = _surface_role_suffix(exposed_role)
    if surface.interface_id is not None:
        parts = surface.interface_id.split("__")
        if len(parts) >= 2 and parts[1] == "CONTACT":
            return "__".join((kind_prefix or parts[0], "CONTACT", *physical_owner_ids))
        if suffix is None and parts:
            suffix = _surface_role_suffix(parts[-1])
        return "__".join(
            (
                kind_prefix or (parts[0] if parts else ""),
                *physical_owner_ids,
                *((suffix,) if suffix else ()),
            )
        )
    return "__".join(
        (
            kind_prefix,
            *physical_owner_ids,
            *((suffix,) if suffix else ()),
        )
    )


def _surface_role_suffix(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith(("sidewall_", "SIDEWALL_")):
        return "SIDEWALL"
    if value.lower() in {"top", "bottom", "sidewall"}:
        return value.upper()
    if value.upper() in {"TOP", "BOTTOM", "SIDEWALL"}:
        return value.upper()
    return None


def _surface_physical_owner_ids(surface: SurfacePlanRecord) -> tuple[str, ...]:
    owner_ids = surface.metadata.get("physical_owner_semantic_ids")
    return surface_physical_owner_ids(owner_ids, _surface_owner_ids(surface))


def _physical_group_owner_ids(
    build_input: GeometryBuildInput,
    owner_ids: Sequence[str],
) -> tuple[str, ...]:
    entities_by_id = {entity.semantic_id: entity for entity in build_input.entities}
    resolved = {
        owner_id: _entity_physical_group_id(entities_by_id[owner_id])
        for owner_id in owner_ids
        if owner_id in entities_by_id
    }
    return physical_group_owner_ids(
        owner_ids,
        resolved,
    )


def _entity_physical_group_id(entity: SemanticEntitySpec) -> str:
    auto_group_id = entity.metadata.get("auto_vacuum_group_id")
    if isinstance(auto_group_id, str) and auto_group_id:
        return auto_group_id
    if entity.part_role not in HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES:
        return entity.semantic_id
    for key in ("physical_group_name", "physical_group_id", "semantic_group_id"):
        value = entity.metadata.get(key)
        if isinstance(value, str) and value:
            return value
    # Plan v2 keeps each occurrence as an Entity. Only explicitly net-bound,
    # same-family local conductors share a computational physical group.
    occurrence = entity.metadata.get("source_occurrence_path")
    local_id = entity.metadata.get("source_local_entity_id")
    if (
        isinstance(occurrence, str)
        and occurrence
        and isinstance(local_id, str)
        and local_id
        and isinstance(entity.net_id, str)
        and entity.net_id
    ):
        parent = occurrence.rpartition("/")[0]
        key = (parent, local_id, entity.part_role, entity.material_id, entity.net_id)
        digest = hashlib.sha256(
            json.dumps(key, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return f"{local_id}__{digest}"
    return entity.semantic_id
