"""SGB-to-core projection; legacy DTOs and serialization remain SGB-owned."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, Literal

from scgsim.semantics import (
    EvidenceResult,
    SemanticEvidenceFacade,
    SourcedPatch,
    canonical_sha256,
    create_snapshot,
)

from .models import GeometryBuildInput

PROJECTION_VERSION = "scgsim.semantic-facts.v1"


def _optional_revision(metadata: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _string_tuple(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return None


def _upstream_plain(value: Any) -> Any:
    """Preserve legacy source fingerprinting outside strict core mappings."""
    if isinstance(value, Mapping):
        return {str(key): _upstream_plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_upstream_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_upstream_plain(item) for item in value), key=repr)
    return value


def _project_semantic_facts(
    build_input: GeometryBuildInput,
    *,
    route: str,
) -> dict[str, Any]:
    polygon_by_id = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    entities: dict[str, dict[str, Any]] = {}
    for entity in build_input.entities:
        metadata = entity.metadata
        aggregate_id = next(
            (
                value
                for key in ("physical_group_name", "physical_group_id", "semantic_group_id")
                if isinstance((value := metadata.get(key)), str) and value
            ),
            entity.semantic_id,
        )
        effective_domains = (
            (entity.semantic_id,)
            if entity.role == "solution_region"
            else (entity.host_void_semantic_id,)
            if entity.host_void_semantic_id
            else None
        )
        polygon_ids = tuple(entity.polygon_ids)
        hole_ids = tuple(
            polygon_id
            for polygon_id in polygon_ids
            if polygon_id in polygon_by_id and polygon_by_id[polygon_id].holes
        )
        sheet_owner = (
            entity.semantic_id
            if entity.route_representations.get(route) == "surface_sheet"
            else None
        )
        side = (
            metadata.get("surface_side")
            if isinstance(metadata.get("surface_side"), str)
            else None
        )
        seam_ids = _string_tuple(metadata.get("seam_source_ids"))
        entities[entity.semantic_id] = {
            "semantic_id": entity.semantic_id,
            "role": entity.role,
            "part_role": entity.part_role,
            "material_id": entity.material_id,
            "material_kind": entity.material_kind,
            "net_id": entity.net_id,
            "source_owner_ids": [entity.semantic_id],
            "aggregate_owner_ids": [aggregate_id],
            "effective_domain_ids": list(effective_domains) if effective_domains else None,
            "side": side,
            "sheet_owner_id": sheet_owner,
            "outer_source_ids": list(polygon_ids) if polygon_ids else None,
            "hole_source_ids": list(hole_ids) if polygon_ids else None,
            "seam_source_ids": list(seam_ids) if seam_ids is not None else None,
            "route_representation": entity.route_representations.get(route),
            "geometry": entity.geometry,
            "polygon_ids": polygon_ids,
        }
    polygons = {
        polygon.polygon_id: {
            "layer": polygon.layer,
            "exterior": polygon.exterior,
            "holes": polygon.holes,
            "object_name": polygon.object_name,
            "net_name": polygon.net_name,
            "port_name": polygon.port_name,
        }
        for polygon in build_input.polygons
    }
    ports = {
        port.port_sheet_id: {
            "source_layer": port.source_layer,
            "source_polygon_id": port.source_polygon_id,
            "exterior": port.exterior,
            "holes": port.holes,
            "overlaps": tuple(
                {
                    "overlap_id": overlap.overlap_id,
                    "host_semantic_id": overlap.host_semantic_id,
                    "host_polygon_id": overlap.host_polygon_id,
                    "overlap_loop": overlap.overlap_loop,
                    "operation": overlap.operation,
                }
                for overlap in port.overlaps
            ),
        }
        for port in build_input.port_sheet_regions
    }
    return {
        "route": route,
        "entities": entities,
        "polygons": polygons,
        "solution_regions": build_input.solution_regions,
        "port_sheet_regions": ports,
    }


def _source_revision(
    build_input: GeometryBuildInput,
) -> tuple[str, Literal["declared_revision", "content_hash_fallback"]]:
    declared_source_revision = _optional_revision(
        build_input.metadata, "source_revision", "upstream_revision"
    )
    if declared_source_revision is not None:
        return declared_source_revision, "declared_revision"
    return (
        f"sha256:{canonical_sha256(_upstream_plain(asdict(build_input)))}",
        "content_hash_fallback",
    )


def build_semantic_evidence_facade(
    build_input: GeometryBuildInput,
    *,
    route: str,
) -> SemanticEvidenceFacade:
    """Snapshot one effective build input for reuse during one planning call."""
    projected = _project_semantic_facts(build_input, route=route)
    source_revision, source_revision_kind = _source_revision(build_input)
    snapshot = create_snapshot(
        projected,
        source_revision=source_revision,
        geometry_revision=_optional_revision(
            build_input.metadata, "geometry_revision", "model_revision"
        ),
        projection_version=PROJECTION_VERSION,
        source_revision_kind=source_revision_kind,
    )
    return SemanticEvidenceFacade(snapshot)


def require_semantic_evidence_facade(
    build_input: GeometryBuildInput,
    *,
    route: str,
    facade: SemanticEvidenceFacade | None,
) -> SemanticEvidenceFacade:
    """Create once when absent, otherwise verify exact projected build identity."""
    if facade is None:
        return build_semantic_evidence_facade(build_input, route=route)
    if not isinstance(facade, SemanticEvidenceFacade):
        raise TypeError("semantic_facts must be a SemanticEvidenceFacade")
    projected = _project_semantic_facts(build_input, route=route)
    expected_digest = canonical_sha256(projected)
    snapshot = facade.snapshot
    if snapshot.projection_version != PROJECTION_VERSION:
        raise ValueError("semantic_facts projection version does not match SGB")
    if snapshot.canonical_sha256 != expected_digest:
        raise ValueError("semantic_facts do not match the supplied build input and route")
    expected_source, expected_source_kind = _source_revision(build_input)
    if (
        snapshot.source_revision != expected_source
        or snapshot.source_revision_kind != expected_source_kind
    ):
        raise ValueError("semantic_facts source revision does not match build input")
    declared_geometry = _optional_revision(
        build_input.metadata, "geometry_revision", "model_revision"
    )
    expected_geometry = declared_geometry or f"sha256:{expected_digest}"
    expected_geometry_kind = (
        "declared_revision" if declared_geometry else "content_hash_fallback"
    )
    if (
        snapshot.geometry_revision != expected_geometry
        or snapshot.geometry_revision_kind != expected_geometry_kind
    ):
        raise ValueError("semantic_facts geometry revision does not match build input")
    return facade


def conductor_solution_evidence(
    facade: SemanticEvidenceFacade,
    *,
    contribution_id: str,
    patch_id: str,
    conductor_id: str,
    solution_id: str,
    side: str | None,
    evidence_stage: Literal["declared", "planned", "observed"] = "planned",
) -> EvidenceResult:
    """Project one proven conductor/domain adjacency and classify it in core."""
    conductor = facade.snapshot.facts["entities"][conductor_id]
    solution = facade.snapshot.facts["entities"][solution_id]
    return facade.evaluate(
        SourcedPatch(
            patch_id=patch_id,
            contribution_id=contribution_id,
            interface_intent="conductor_solution",
            source_owner_ids=(conductor_id, solution_id),
            aggregate_owner_ids=(
                *tuple(conductor["aggregate_owner_ids"]),
                *tuple(solution["aggregate_owner_ids"]),
            ),
            material_id=conductor["material_id"],
            material_kind=conductor["material_kind"],
            effective_domain_ids=(solution_id,),
            side=side,
            sheet_owner_id=conductor["sheet_owner_id"],
            outer_source_ids=conductor["outer_source_ids"],
            hole_source_ids=conductor["hole_source_ids"],
            seam_source_ids=conductor["seam_source_ids"],
            evidence_stage=evidence_stage,
            snapshot_reference=facade.snapshot.reference(),
        )
    )


def solution_solution_evidence(
    facade: SemanticEvidenceFacade,
    *,
    contribution_id: str,
    patch_id: str,
    lower_id: str,
    upper_id: str,
    side: str | None,
    evidence_stage: Literal["declared", "planned", "observed"] = "planned",
) -> EvidenceResult:
    """Project one proven ordered solution adjacency and classify it in core."""
    lower = facade.snapshot.facts["entities"][lower_id]
    upper = facade.snapshot.facts["entities"][upper_id]
    return facade.evaluate(
        SourcedPatch(
            patch_id=patch_id,
            contribution_id=contribution_id,
            interface_intent="solution_solution",
            source_owner_ids=(lower_id, upper_id),
            aggregate_owner_ids=(
                *tuple(lower["aggregate_owner_ids"]),
                *tuple(upper["aggregate_owner_ids"]),
            ),
            material_id=None,
            material_kind=None,
            effective_domain_ids=(lower_id, upper_id),
            side=side,
            sheet_owner_id=None,
            outer_source_ids=None,
            hole_source_ids=None,
            seam_source_ids=None,
            evidence_stage=evidence_stage,
            snapshot_reference=facade.snapshot.reference(),
        )
    )


def metal_metal_evidence(
    facade: SemanticEvidenceFacade,
    *,
    contribution_id: str,
    patch_id: str,
    lower_id: str,
    upper_id: str,
    side: str | None,
    evidence_stage: Literal["declared", "planned", "observed"] = "planned",
) -> EvidenceResult:
    """Project one proven conductor contact without moving contact search."""
    lower = facade.snapshot.facts["entities"][lower_id]
    upper = facade.snapshot.facts["entities"][upper_id]
    return facade.evaluate(
        SourcedPatch(
            patch_id=patch_id,
            contribution_id=contribution_id,
            interface_intent="metal_metal_contact",
            source_owner_ids=(lower_id, upper_id),
            aggregate_owner_ids=(
                *tuple(lower["aggregate_owner_ids"]),
                *tuple(upper["aggregate_owner_ids"]),
            ),
            material_id=None,
            material_kind="conductor",
            effective_domain_ids=None,
            side=side,
            sheet_owner_id=lower["sheet_owner_id"] or upper["sheet_owner_id"],
            outer_source_ids=lower["outer_source_ids"],
            hole_source_ids=lower["hole_source_ids"],
            seam_source_ids=lower["seam_source_ids"],
            evidence_stage=evidence_stage,
            snapshot_reference=facade.snapshot.reference(),
        )
    )
