"""Detached, read-only view of normalized source inputs before backend work."""

from __future__ import annotations

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
        from scgsim._notebook_presentation import (
            cards, details, display_text, json_details, section, selection_text, table,
        )

        data = self.data
        source = data.get("source", {})
        backend = data.get("backend", {})
        entities = data.get("entities", [])
        ports = data.get("ports", [])
        junctions = backend.get("junctions", [])
        source_cards = cards((
            ("Source cell", source.get("top_cell_name")),
            ("Backend", backend.get("name")),
            ("Route", backend.get("route")),
            ("Entities", source.get("entity_count")),
            ("Polygons", source.get("polygon_count")),
            ("Ports", source.get("port_sheet_count")),
        ))
        entity_rows = [
            (item.get("semantic_id"), item.get("source_occurrence_path"),
             item.get("net_id"), item.get("material_id"), item.get("material_kind"),
             f"{display_text(item.get('z_min_um'))}–{display_text(item.get('z_max_um'))}",
             item.get("role"))
            for item in entities
        ]
        net_rows = [
            (item.get("net_id"), ", ".join(item.get("entity_ids", [])))
            for item in data.get("nets", [])
        ]
        material_rows = [
            (item.get("material_id"), item.get("record"))
            for item in data.get("materials", [])
        ]
        port_rows = [
            (item.get("source_name"), item.get("source_occurrence_path"),
             ", ".join(item.get("overlap_hosts", [])),
             ", ".join(item.get("overlap_nets", [])), item.get("direction"),
             item.get("target_layer"))
            for item in ports
        ]
        junction_rows = [
            (item.get("junction_id"), item.get("terminal_a_net"),
             item.get("terminal_b_net"), item.get("inductance_h"),
             item.get("capacitance_f"), item.get("source_polygon_id"))
            for item in junctions
        ]
        palace_port_rows = [
            (item.get("name"), item.get("layer"), item.get("inductance_h"))
            for item in backend.get("ports", [])
        ]
        surfaces = backend.get("surface_contributions", [])
        surface_rows = [
            (item.get("contribution_id"), item.get("interface_kind"),
             item.get("source_polygon_id"), selection_text(item.get("margins_um")),
             f"{display_text(item.get('film_thickness_m'))} m"
             if item.get("film_thickness_m") is not None else None,
             item.get("film_relative_permittivity"),
             item.get("loss_tangent"), item.get("source"), item.get("preset"))
            for item in surfaces
        ]
        palace_surfaces = backend.get("surface_epr_specs")
        if isinstance(palace_surfaces, Mapping):
            surface_rows.extend(
                (interface, interface, None,
                 selection_text(item.get("inset_margins_um")),
                 (f"{display_text(item['film_thickness_m'])} m"
                  if "film_thickness_m" in item else
                  f"{display_text(item['thickness'])} × Model.L0 (Palace native)"
                  if "thickness" in item else None),
                 item.get("permittivity"), item.get("loss_tangent"),
                 item.get("source"), item.get("preset"))
                for interface, item in palace_surfaces.items()
            )
        request = backend.get("epr_request")
        request_rows = (
            [(key, selection_text(value)) for key, value in request.items()]
            if isinstance(request, Mapping) else [("EPR request", "not requested")]
        )
        settings = [
            (key, display_text(backend.get(key)))
            for key in ("project_name", "design_name", "route_a_profile", "geometry_workers")
            if key in backend
        ]
        content = (
            "<p>Configured source and settings; native geometry and solve unobserved.</p>"
            + source_cards
            + "<h4>Entities, Nets, materials, and source Z (µm)</h4>"
            + table(("Entity", "Occurrence", "Net", "Material", "Kind", "Z range", "Role"), entity_rows)
            + details("Net membership", table(("Net", "Entities"), net_rows))
            + details("Material records", table(("Material", "Recorded properties"), material_rows))
            + "<h4>Port sheets and source hosts</h4>"
            + table(("Port", "Occurrence", "Hosts", "Nets", "Direction", "Layer"), port_rows)
            + "<h4>Junction declarations</h4>"
            + table(("Junction", "Terminal A", "Terminal B", "L (H)", "C (F)", "Source sheet"), junction_rows)
            + (details("Palace lumped ports", table(("Port", "Layer", "L (H)"),
                                                      palace_port_rows)) if palace_port_rows else "")
            + "<h4>Surface films and requested margins</h4>"
            + table(("Contribution", "Interface", "Source", "Margins (µm)", "Film thickness",
                     "εr", "Loss tangent", "Assumption source", "Preset"), surface_rows)
            + "<h4>Analysis choices</h4>" + table(("Selection", "Configured"), request_rows)
            + details("Solver controls and resources", table(
                ("Setting", "Value"), [*settings, ("run_control", backend.get("run_control")),
                                      ("resources", backend.get("resources")),
                                      ("surface_defaults", backend.get("surface_defaults"))]
            ))
            + json_details("Original detached input JSON", data)
        )
        return section("Pre-solve input summary", content, class_name="scgsim-input-summary")


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
