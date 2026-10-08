"""Shared source Entity lookup/metadata/range operations. No guessed native bindings or simulation state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from scgsim.geometry._primitives.constants import _GEOMETRY_REF_METADATA_KEYS
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import MMContactRecord


def _component_metadata(
    records: tuple[MMContactRecord, ...],
) -> tuple[dict[str, Any], ...]:
    """Derive one complete transitive component ledger from MM contacts."""
    grouped: dict[str, list[MMContactRecord]] = {}
    for record in records:
        grouped.setdefault(record.conductor_component_id, []).append(record)
    return tuple(
        {
            "conductor_component_id": component_id,
            "members": tuple(
                sorted(
                    {
                        entity_id
                        for record in component_records
                        for entity_id in (
                            record.lower_entity_id,
                            record.upper_entity_id,
                        )
                    }
                )
            ),
            "contact_ids": tuple(
                sorted(record.contact_id for record in component_records)
            ),
            "net_id": _one_component_value(
                component_id,
                "net_id",
                (record.net_id for record in component_records),
            ),
            "equipotential_id": _one_component_value(
                component_id,
                "equipotential_id",
                (record.equipotential_id for record in component_records),
            ),
        }
        for component_id, component_records in sorted(grouped.items())
    )


def _one_component_value(
    component_id: str,
    field: str,
    values: Sequence[str | None],
) -> str | None:
    distinct = {value for value in values if value is not None}
    if len(distinct) > 1:
        raise ValueError(f"{component_id} has ambiguous {field}")
    return next(iter(distinct), None)


def _geometry_ref_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key] for key in _GEOMETRY_REF_METADATA_KEYS if key in metadata
    }


def _intent_supports_route(intent: Mapping[str, Any], route: RouteLiteral) -> bool:
    valid_routes = intent.get("valid_routes")
    if valid_routes is None:
        return True
    if isinstance(valid_routes, str):
        return route == valid_routes
    return route in {str(value) for value in valid_routes}


def _entity_by_id(
    build_input: GeometryBuildInput,
    semantic_id: str,
) -> SemanticEntitySpec:
    for entity in build_input.entities:
        if entity.semantic_id == semantic_id:
            return entity
    raise ValueError(f"unknown semantic entity: {semantic_id}")


def _unique_ids(values: Any) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if item in seen:
            continue
        result.append(item)
        seen.add(item)
    return tuple(result)


def _entity_z_range_um(entity: SemanticEntitySpec) -> tuple[float, float]:
    z_min_um = float(entity.geometry.get("z_min_um", entity.geometry.get("z_um", 0.0)))
    z_max_um = entity.geometry.get("z_max_um")
    if z_max_um is not None:
        z_max_um = float(z_max_um)
        if not isfinite(z_max_um):
            raise ValueError(f"{entity.semantic_id} has non-finite z_max_um")
        if z_max_um > z_min_um:
            return z_min_um, z_max_um

    thickness_um = float(entity.geometry.get("thickness_um", 0.0))
    return z_min_um, z_min_um + thickness_um
