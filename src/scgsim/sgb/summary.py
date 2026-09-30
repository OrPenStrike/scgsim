"""Detached, read-only view of normalized source inputs before backend work."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from scgsim.semantics.route_a import geometry_z_range

from .models import GeometryBuildInput
from .validation import validate_geometry_input


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"input summary cannot detach {type(value).__name__}")


@dataclass(frozen=True, init=False)
class InputSummary:
    """Immutable snapshot whose ``data`` access returns a detached JSON tree."""

    _payload_json: str

    def __init__(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("InputSummary data must be a mapping")
        object.__setattr__(
            self, "_payload_json",
            json.dumps(_plain(data), allow_nan=False, sort_keys=True),
        )

    @property
    def data(self) -> dict[str, Any]:
        return json.loads(self._payload_json)

    def _repr_html_(self) -> str:
        return (
            "<section><h3>Pre-solve input summary</h3>"
            "<p>Configured source and settings; native geometry and solve unobserved.</p>"
            "<pre>" + html.escape(json.dumps(self.data, indent=2)) + "</pre></section>"
        )


def summarize_geometry_input(
    build_input: GeometryBuildInput, *, prepared_stack: Mapping[str, Any]
) -> InputSummary:
    """Summarize an already-normalized GDS/stack pair without preparing it."""

    if not isinstance(build_input, GeometryBuildInput):
        raise TypeError("build_input must be GeometryBuildInput")
    if not isinstance(prepared_stack, Mapping):
        raise TypeError("prepared_stack must be a mapping")
    validate_geometry_input(build_input)
    stack_metadata = prepared_stack.get("metadata", {})
    if not isinstance(stack_metadata, Mapping):
        raise TypeError("prepared_stack metadata must be a mapping")
    source_hash = build_input.metadata.get("gds_sha256")
    stack_hash = stack_metadata.get("gds_sha256")
    if (source_hash is not None or stack_hash is not None) and (
        not isinstance(source_hash, str) or source_hash != stack_hash
    ):
        raise ValueError("normalized input and prepared stack GDS identities differ")
    materials = prepared_stack.get("materials")
    if not isinstance(materials, Mapping):
        raise ValueError("prepared_stack requires a materials mapping")
    missing = {entity.material_id for entity in build_input.entities} - set(materials)
    if missing:
        raise ValueError(f"source entities reference unknown materials: {sorted(missing)!r}")
    nets: dict[str, list[str]] = {}
    entities = []
    by_id = {entity.semantic_id: entity for entity in build_input.entities}
    for entity in build_input.entities:
        z_min, z_max = geometry_z_range(entity.geometry, entity.semantic_id)
        if entity.net_id is not None:
            nets.setdefault(entity.net_id, []).append(entity.semantic_id)
        entities.append({
            "semantic_id": entity.semantic_id,
            "role": entity.role,
            "part_role": entity.part_role,
            "material_id": entity.material_id,
            "material_kind": entity.material_kind,
            "net_id": entity.net_id,
            "z_min_um": z_min,
            "z_max_um": z_max,
            "polygon_ids": list(entity.polygon_ids),
            "source_occurrence_path": entity.metadata.get("source_occurrence_path"),
        })
    ports = []
    for port in build_input.port_sheet_regions:
        ports.append({
            "source_name": port.metadata["source_name"],
            "source_polygon_id": port.source_polygon_id,
            "target_layer": port.metadata["target_layer"],
            "direction": port.metadata["direction"],
            "source_occurrence_path": port.metadata.get("source_occurrence_path"),
            "overlap_hosts": [item.host_semantic_id for item in port.overlaps],
            "overlap_nets": sorted({
                by_id[item.host_semantic_id].net_id
                for item in port.overlaps
                if by_id[item.host_semantic_id].net_id is not None
            }),
        })
    source_metadata = build_input.metadata
    top_cell_name = source_metadata.get("selected_cell_name")
    if not isinstance(top_cell_name, str) or not top_cell_name.strip():
        top_cell_name = source_metadata.get("top_cell_name")
    if not isinstance(top_cell_name, str) or not top_cell_name.strip():
        top_cell_name = None
    return InputSummary({
        "source": {
            "status": "normalized",
            "gds_sha256": source_metadata.get("gds_sha256"),
            "top_cell_name": top_cell_name,
            "entity_count": len(build_input.entities),
            "polygon_count": len(build_input.polygons),
            "port_sheet_count": len(build_input.port_sheet_regions),
        },
        "entities": entities,
        "nets": [
            {"net_id": name, "entity_ids": members}
            for name, members in sorted(nets.items())
        ],
        "materials": [
            {"material_id": name, "record": materials[name]}
            for name in sorted({entity.material_id for entity in build_input.entities})
        ],
        "vacuum_domains": [
            {"semantic_id": entity["semantic_id"],
             "z_min_um": entity["z_min_um"], "z_max_um": entity["z_max_um"]}
            for entity in entities if entity["material_kind"] == "vacuum"
        ],
        "ports": ports,
        "native_status": "not_observed",
    })
