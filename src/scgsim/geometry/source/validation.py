"""Own existing input-only validation before compilation. Shared Entity predicates are lower operations, not duplicate validators."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from scgsim.geometry._primitives.entity_validation import (
    _is_solution_entity,
    _is_vacuum_solution_entity,
    _requires_route_representation,
    _resolve_contact_pad_attachment,
)
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec


def validate_geometry_input(build_input: GeometryBuildInput) -> GeometryBuildInput:
    """Validate adapter-normalized input before route-aware planning."""
    errors: list[str] = []
    polygon_ids: set[str] = set()
    entity_ids: set[str] = set()
    entities_by_id: dict[str, SemanticEntitySpec] = {}

    for polygon in build_input.polygons:
        if not polygon.polygon_id:
            errors.append("polygon_id must be non-empty")
        elif polygon.polygon_id in polygon_ids:
            errors.append(f"duplicate polygon_id: {polygon.polygon_id}")
        polygon_ids.add(polygon.polygon_id)
        if polygon.exterior and len(polygon.exterior) < 3:
            errors.append(f"{polygon.polygon_id} exterior requires at least 3 points")

    for entity in build_input.entities:
        if not entity.semantic_id:
            errors.append("semantic_id must be non-empty")
        elif entity.semantic_id in entity_ids:
            errors.append(f"duplicate semantic_id: {entity.semantic_id}")
        entity_ids.add(entity.semantic_id)
        if not entity.material_id:
            errors.append(f"{entity.semantic_id} material_id must be non-empty")
        if entity.material_kind not in {"vacuum", "dielectric", "conductor"}:
            errors.append(
                f"{entity.semantic_id} material_kind must be vacuum, dielectric, or conductor"
            )
        if entity.role == "solution_region":
            if entity.material_kind not in {"vacuum", "dielectric"}:
                errors.append(
                    f"{entity.semantic_id} solution_region must be vacuum or dielectric"
                )
        elif entity.material_kind != "conductor":
            errors.append(f"{entity.semantic_id} non-solution entity must be conductor")
        if entity.material_kind == "conductor" and (
            not isinstance(host_id := entity.host_void_semantic_id, str) or not host_id
        ):
            errors.append(
                f"{entity.semantic_id} conductor needs explicit host_void_semantic_id"
            )
        entities_by_id[entity.semantic_id] = entity
        for polygon_id in entity.polygon_ids:
            if polygon_id not in polygon_ids:
                errors.append(
                    f"{entity.semantic_id} references unknown polygon_id: {polygon_id}"
                )

    for entity in build_input.entities:
        if entity.material_kind == "conductor" and entity.host_void_semantic_id:
            host = entities_by_id.get(entity.host_void_semantic_id)
            if host is None or not _is_solution_entity(host):
                errors.append(
                    f"{entity.semantic_id} host_void_semantic_id must resolve a solution_region"
                )
        if entity.part_role != "contact_pad":
            continue
        try:
            _resolve_contact_pad_attachment(
                entity,
                build_input.entities,
                defer_overlap=bool(
                    build_input.boundary_curves or build_input.boundary_reconstruction
                ),
            )
        except ValueError as exc:
            errors.append(str(exc))

    port_sheet_ids: set[str] = set()
    for region in build_input.port_sheet_regions:
        if not region.port_sheet_id:
            errors.append("port_sheet_id must be non-empty")
        elif region.port_sheet_id in port_sheet_ids:
            errors.append(f"duplicate port_sheet_id: {region.port_sheet_id}")
        port_sheet_ids.add(region.port_sheet_id)
        if region.metadata.get("source") != "palace_lumped_port_sheet":
            errors.append(
                f"{region.port_sheet_id} must declare source 'palace_lumped_port_sheet'"
            )
        for key in ("source_name", "target_layer", "direction_sign_convention"):
            if (
                not isinstance(region.metadata.get(key), str)
                or not region.metadata[key]
            ):
                errors.append(f"{region.port_sheet_id} requires {key}")
        port_index = region.metadata.get("port_index")
        if (
            isinstance(port_index, bool)
            or not isinstance(port_index, int)
            or port_index < 1
        ):
            errors.append(f"{region.port_sheet_id} requires 1-based port_index")
        direction = region.metadata.get("direction")
        direction_raw = region.metadata.get("direction_raw")
        if (
            isinstance(direction, str | bytes)
            or not isinstance(direction, Sequence)
            or len(direction) != 3
        ):
            errors.append(f"{region.port_sheet_id} requires a 3D direction")
        else:
            try:
                direction_values = tuple(float(value) for value in direction)
            except (TypeError, ValueError):
                errors.append(f"{region.port_sheet_id} direction must be numeric")
            else:
                if (
                    not all(isfinite(value) for value in direction_values)
                    or direction_values[2] != 0.0
                    or direction_values[0] == direction_values[1] == 0.0
                ):
                    errors.append(
                        f"{region.port_sheet_id} direction must be finite, XY, "
                        "and nonzero"
                    )
        if (
            isinstance(direction_raw, str | bytes)
            or not isinstance(direction_raw, Sequence)
            or len(direction_raw) != 3
        ):
            errors.append(f"{region.port_sheet_id} requires raw direction provenance")
        if len(region.exterior) < 3:
            errors.append(f"{region.port_sheet_id} exterior requires 3 points")
        for overlap in region.overlaps:
            if overlap.port_polygon_id != region.source_polygon_id:
                errors.append(
                    f"{overlap.overlap_id} port polygon "
                    f"{overlap.port_polygon_id!r} does not match region source "
                    f"{region.source_polygon_id!r}"
                )
            if overlap.operation != "local_fragment_required":
                errors.append(
                    f"{overlap.overlap_id} unsupported port-sheet operation "
                    f"{overlap.operation!r}"
                )
            if overlap.port_sheet_id != region.port_sheet_id:
                errors.append(
                    f"{overlap.overlap_id} references port sheet "
                    f"{overlap.port_sheet_id!r}, expected {region.port_sheet_id!r}"
                )
            if overlap.host_semantic_id not in entity_ids:
                errors.append(
                    f"{overlap.overlap_id} references unknown host semantic id "
                    f"{overlap.host_semantic_id}"
                )
            else:
                host_entity = entities_by_id[overlap.host_semantic_id]
                host_is_solver_geometry = not _is_solution_entity(
                    host_entity
                ) and _requires_route_representation(host_entity)
                if not host_is_solver_geometry:
                    errors.append(
                        f"{overlap.overlap_id} host {overlap.host_semantic_id} "
                        "is not solver-relevant geometry"
                    )
                if overlap.host_polygon_id not in host_entity.polygon_ids:
                    errors.append(
                        f"{overlap.overlap_id} host polygon "
                        f"{overlap.host_polygon_id} is not owned by "
                        f"{overlap.host_semantic_id}"
                    )
            if overlap.host_polygon_id not in polygon_ids:
                errors.append(
                    f"{overlap.overlap_id} references unknown host polygon id "
                    f"{overlap.host_polygon_id}"
                )
            if len(overlap.overlap_loop) < 3:
                errors.append(f"{overlap.overlap_id} overlap_loop requires 3 points")

    if not any(_is_vacuum_solution_entity(entity) for entity in build_input.entities):
        errors.append("GeometryBuildInput requires a vacuum solution_region")

    if errors:
        raise ValueError("Invalid GeometryBuildInput: " + "; ".join(errors))
    return build_input
