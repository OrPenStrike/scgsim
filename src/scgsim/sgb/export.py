"""Final physical-group export helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.sgb.models import (
    BackendEntityTagRecord,
    ConstructionPlanRecord,
    FinalPhysicalGroupRecord,
    StructuredFinalPhysicalGroupRecord,
    SurfacePlanRecord,
    TagPlanRecord,
    VolumePlanRecord,
)


def export_physical_group_records(
    plan: ConstructionPlanRecord,
) -> tuple[FinalPhysicalGroupRecord, ...]:
    """Export exactly one record for each backend physical group.

    OCC groups sources by physical name, dimension, role, and solver use.  The
    final handoff mirrors that grouping instead of publishing a partial record
    per source tag.  Route C retains its established grouped export shape;
    A/B additionally require complete structured surface authority.
    """
    backend_tags = _backend_tags_by_source(plan.backend_entity_tags)
    surfaces = {surface.surface_id: surface for surface in plan.surfaces}
    volumes = {volume.volume_id: volume for volume in plan.volumes}
    records: list[FinalPhysicalGroupRecord] = []
    for tags in _group_tag_plans(plan.tags):
        first = tags[0]
        entity_tags = _physical_entity_tags(tags, backend_tags)
        if not entity_tags:
            raise NotImplementedError(
                f"{first.physical_name} has no backend entity tags yet"
            )
        if plan.route == "C":
            records.append(_route_c_group(plan, tags, entity_tags, volumes))
        elif first.source_record_kind == "surface":
            sources = tuple(surfaces[tag.source_record_id] for tag in tags)
            records.append(_structured_surface_group(plan, tags, sources, entity_tags))
        else:
            sources = tuple(volumes[tag.source_record_id] for tag in tags)
            records.append(_structured_volume_group(plan, tags, sources, entity_tags))
    return tuple(records)


def _route_c_group(
    plan: ConstructionPlanRecord,
    tags: tuple[TagPlanRecord, ...],
    entity_tags: tuple[int, ...],
    volumes: Mapping[str, VolumePlanRecord],
) -> FinalPhysicalGroupRecord:
    first = tags[0]
    return FinalPhysicalGroupRecord(
        physical_name=first.physical_name,
        dimension=first.dimension,
        route=plan.route,
        role=first.role,
        source_record_id=first.physical_name,
        solver_use=first.solver_use,
        entity_tags=entity_tags,
        metadata={"source_record_ids": tuple(tag.source_record_id for tag in tags)},
    )


def _structured_surface_group(
    plan: ConstructionPlanRecord,
    tags: tuple[TagPlanRecord, ...],
    sources: tuple[SurfacePlanRecord, ...],
    entity_tags: tuple[int, ...],
) -> FinalPhysicalGroupRecord:
    first_tag, first = tags[0], sources[0]
    required = (
        "route",
        "representation",
        "interface_type",
        "face_kind",
        "owner_semantic_ids",
        "boundary_volume_ids",
        "source_provenance",
    )
    for source in sources:
        missing = tuple(key for key in required if key not in source.metadata)
        if missing or source.metadata["route"] != plan.route:
            raise ValueError(
                f"{source.surface_id} lacks unambiguous structured export fields "
                f"{missing!r}"
            )
    same_fields = (
        "representation",
        "interface_type",
        "contact_kind",
        "face_kind",
        "owner_semantic_ids",
        "boundary_volume_ids",
        "conductor_component_id",
        "net_id",
        "equipotential_id",
        "physical_attribute",
    )
    for field in same_fields:
        _require_same_metadata_field(field, sources)
    provenance = (
        dict(first.metadata["source_provenance"])
        if first.surface_role == "lumped_port" and len(sources) == 1
        else _combined_provenance(sources)
    )
    return StructuredFinalPhysicalGroupRecord(
        physical_name=first_tag.physical_name,
        dimension=first_tag.dimension,
        route=plan.route,
        representation=str(first.metadata["representation"]),
        role=first_tag.role,
        source_record_id=first_tag.physical_name,
        net_id=_optional_string(first.metadata.get("net_id"), "net_id", first),
        solver_use=first_tag.solver_use,
        entity_tags=entity_tags,
        surface_id=first_tag.physical_name,
        interface_type=str(first.metadata["interface_type"]),
        contact_kind=_optional_string(
            first.metadata.get("contact_kind"), "contact_kind", first
        ),
        face_kind=str(first.metadata["face_kind"]),
        owner_semantic_ids=_nonempty_tuple(
            first.metadata["owner_semantic_ids"], "owner_semantic_ids", first
        ),
        adjacent_solution_volume_ids=_nonempty_tuple(
            first.metadata["boundary_volume_ids"], "boundary_volume_ids", first
        ),
        conductor_component_id=_optional_string(
            first.metadata.get("conductor_component_id"),
            "conductor_component_id",
            first,
        ),
        equipotential_id=_optional_string(
            first.metadata.get("equipotential_id"), "equipotential_id", first
        ),
        source_provenance=provenance,
        physical_attribute={
            "solver_use": first.solver_use,
            "surface_role": first.surface_role,
            **_custom_physical_attribute(first),
        },
    )


def _structured_volume_group(
    plan: ConstructionPlanRecord,
    tags: tuple[TagPlanRecord, ...],
    sources: tuple[VolumePlanRecord, ...],
    entity_tags: tuple[int, ...],
) -> FinalPhysicalGroupRecord:
    first_tag = tags[0]
    if any(source.construction_only for source in sources):
        raise ValueError(f"{first_tag.physical_name} includes construction-only volume")
    representations = {
        str(source.metadata.get("representation", "solution_volume"))
        for source in sources
    }
    if len(representations) != 1:
        raise ValueError(
            f"{first_tag.physical_name} groups incompatible volume representations"
        )
    material_ids = tuple(source.material_id for source in sources)
    material_kinds = tuple(_volume_material_kind(source) for source in sources)
    if len(set(material_ids)) != 1 or len(set(material_kinds)) != 1:
        raise ValueError(
            f"{first_tag.physical_name} grouped material metadata is not uniform"
        )
    return StructuredFinalPhysicalGroupRecord(
        physical_name=first_tag.physical_name,
        dimension=first_tag.dimension,
        route=plan.route,
        representation=representations.pop(),
        role=first_tag.role,
        source_record_id=first_tag.physical_name,
        solver_use=first_tag.solver_use,
        entity_tags=entity_tags,
        interface_type="material_volume",
        face_kind="volume",
        owner_semantic_ids=tuple(source.owner_semantic_id for source in sources),
        adjacent_solution_volume_ids=(),
        source_provenance={
            "volume_ids": tuple(source.volume_id for source in sources),
            "route": plan.route,
        },
        physical_attribute={
            "material_ids": (material_ids[0],),
            "material_kinds": (material_kinds[0],),
        },
    )


def _volume_material_kind(source: VolumePlanRecord) -> str:
    kind = source.metadata.get("material_kind")
    if kind not in {"vacuum", "dielectric", "conductor"}:
        raise ValueError(
            f"{source.volume_id} needs explicit material_kind in volume metadata"
        )
    return str(kind)


def _require_same_metadata_field(
    field: str, sources: Sequence[SurfacePlanRecord]
) -> None:
    values = {repr(source.metadata.get(field)) for source in sources}
    if len(values) != 1:
        raise ValueError(
            f"{sources[0].surface_id} physical group has ambiguous {field} "
            "across sources"
        )


def _combined_provenance(sources: Sequence[SurfacePlanRecord]) -> dict[str, Any]:
    provenance = [source.metadata["source_provenance"] for source in sources]
    if any(not isinstance(item, Mapping) or not item for item in provenance):
        raise ValueError(f"{sources[0].surface_id} has no source provenance")
    return {
        "source_record_ids": tuple(source.surface_id for source in sources),
        "sources": tuple(dict(item) for item in provenance),
    }


def _backend_tags_by_source(
    backend_tags: tuple[BackendEntityTagRecord, ...],
) -> dict[tuple[str, str], tuple[tuple[int, int], ...]]:
    result: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for record in backend_tags:
        result.setdefault(
            (record.source_record_kind, record.source_record_id), []
        ).append(record.dim_tag)
    return {source: tuple(dim_tags) for source, dim_tags in result.items()}


def _group_tag_plans(
    tags: tuple[TagPlanRecord, ...],
) -> tuple[tuple[TagPlanRecord, ...], ...]:
    grouped: dict[tuple[str, int, str, str], list[TagPlanRecord]] = {}
    contracts: dict[tuple[str, int], tuple[str, str]] = {}
    for tag in tags:
        key = (tag.physical_name, tag.dimension)
        contract = (tag.role, tag.solver_use)
        if key in contracts and contracts[key] != contract:
            raise ValueError(f"{tag.physical_name} has heterogeneous tag sources")
        contracts[key] = contract
        grouped.setdefault(
            (tag.physical_name, tag.dimension, tag.role, tag.solver_use), []
        ).append(tag)
    return tuple(tuple(items) for items in grouped.values())


def _physical_entity_tags(
    tags: tuple[TagPlanRecord, ...],
    backend_tags: Mapping[tuple[str, str], tuple[tuple[int, int], ...]],
) -> tuple[int, ...]:
    result: list[int] = []
    for tag in tags:
        for dimension, entity_tag in backend_tags.get(
            (tag.source_record_kind, tag.source_record_id), ()
        ):
            if dimension == tag.dimension and entity_tag not in result:
                result.append(entity_tag)
    return tuple(result)


def _nonempty_tuple(
    raw: object, field: str, surface: SurfacePlanRecord
) -> tuple[str, ...]:
    values = (
        (raw,)
        if isinstance(raw, str)
        else tuple(raw)
        if isinstance(raw, tuple | list)
        else ()
    )
    values = tuple(str(value) for value in values)
    if not values or any(not value for value in values):
        raise ValueError(f"{surface.surface_id} has ambiguous {field}")
    return values


def _optional_string(raw: object, field: str, surface: SurfacePlanRecord) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{surface.surface_id} has ambiguous {field}")
    return raw


def _custom_physical_attribute(surface: SurfacePlanRecord) -> dict[str, Any]:
    raw = surface.metadata.get("physical_attribute", {})
    if not isinstance(raw, Mapping):
        raise TypeError(f"{surface.surface_id} physical_attribute must be a mapping")
    return dict(raw)
