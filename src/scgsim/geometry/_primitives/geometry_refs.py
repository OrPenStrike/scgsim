"""Project source geometry references for planning; preserve existing route-specific source meaning without native inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from scgsim.geometry._primitives.entities import (
    _entity_z_range_um,
    _geometry_ref_from_metadata,
)
from scgsim.geometry._primitives.loops import _clean_loop, _ring_edges
from scgsim.geometry._primitives.spatial import _same_z
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.topology import InterfacePlanRecord


def _entity_geometry_ref(
    entity: SemanticEntitySpec,
    *,
    representation: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "from_semantic_id": entity.semantic_id,
        "geometry_kind": entity.geometry_kind,
        "part_role": entity.part_role,
        "representation": representation,
        "source_polygon_ids": entity.polygon_ids,
    }
    result.update(_geometry_ref_from_metadata(entity.geometry))
    result.update(_geometry_ref_from_metadata(entity.metadata))
    if "z_um" in entity.geometry and "plane" not in result:
        result["plane"] = {"axis": "z", "value_um": entity.geometry["z_um"]}
    return result


def _route_entity_geometry_ref(
    build_input: GeometryBuildInput,
    route: RouteLiteral,
    entity: SemanticEntitySpec,
    *,
    representation: str,
    interfaces: tuple[InterfacePlanRecord, ...] = (),
) -> dict[str, Any]:
    del build_input
    geometry_ref = _entity_geometry_ref(entity, representation=representation)
    if route != "A" or representation != "cutout_boundary_shell":
        return geometry_ref

    z_min_um, z_max_um = _entity_z_range_um(entity)
    for interface in interfaces:
        if interface.recognition_rule != "coplanar_conductor_contact_patch":
            continue
        metadata = interface.metadata
        plane = metadata.get("contact_plane") or metadata.get("plane")
        if not isinstance(plane, Mapping) or plane.get("axis") != "z":
            continue
        plane_z_um = float(plane["value_um"])
        if (
            metadata.get("upper_entity_id") == entity.semantic_id
            and metadata.get("upper_face") == "bottom"
        ):
            z_min_um = min(z_min_um, plane_z_um)
        if (
            metadata.get("lower_entity_id") == entity.semantic_id
            and metadata.get("lower_face") == "top"
        ):
            z_max_um = max(z_max_um, plane_z_um)

    if z_max_um <= z_min_um:
        raise ValueError(f"{entity.semantic_id} Route A cutout body has empty z range")
    if not _same_z(z_min_um, _entity_z_range_um(entity)[0]) or not _same_z(
        z_max_um,
        _entity_z_range_um(entity)[1],
    ):
        geometry_ref["z_um"] = z_min_um
        geometry_ref["z_min_um"] = z_min_um
        geometry_ref["z_max_um"] = z_max_um
        geometry_ref["thickness_um"] = z_max_um - z_min_um
        geometry_ref["route_a_cutout_z_range_um"] = (z_min_um, z_max_um)
    return geometry_ref


def _sidewall_geometry_refs(
    geometry_ref: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    z_min_um = float(geometry_ref.get("z_min_um", geometry_ref.get("z_um", 0.0)))
    thickness_um = float(geometry_ref.get("thickness_um", 0.0))
    if thickness_um <= 0:
        return ()
    z_max_um = z_min_um + thickness_um
    refs: list[dict[str, Any]] = []
    for ring_role, ring in (
        ("outer", geometry_ref["outer_loop"]),
        *(
            (f"hole_{index:04d}", hole_loop)
            for index, hole_loop in enumerate(geometry_ref.get("hole_loops", ()))
        ),
    ):
        for edge_index, (start, end) in enumerate(_ring_edges(_clean_loop(ring))):
            refs.append(
                {
                    "quad_points": (
                        (start[0], start[1], z_min_um),
                        (end[0], end[1], z_min_um),
                        (end[0], end[1], z_max_um),
                        (start[0], start[1], z_max_um),
                    ),
                    "sidewall_ring_role": ring_role,
                    "sidewall_edge_index": edge_index,
                }
            )
    return tuple(refs)
