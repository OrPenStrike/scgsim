"""Lower declared port sheets and bind each piece to final VolumePlan hosts independently of material labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite, sqrt
from typing import Any

from scgsim.geometry._primitives.constants import _TOPOLOGY_EPS_UM
from scgsim.geometry._primitives.entities import (
    _component_metadata,
    _entity_z_range_um,
    _unique_ids,
)
from scgsim.geometry._primitives.entity_validation import (
    _is_solution_entity,
    _is_vacuum_solution_entity,
)
from scgsim.geometry._primitives.loops import (
    _boolean_gdstk_region,
    _clean_loop,
    _gdstk_surface_region,
    _geometry_refs_from_gdstk_region,
    _loops_share_edge_overlap,
    _polygon_area,
    _ring_edges,
)
from scgsim.geometry._primitives.port_contract import (
    _route_b_port_sheet_binding_matches,
)
from scgsim.geometry._primitives.spatial import (
    _geometry_ref_surface_z_um,
    _same_z,
    _segment_overlap_interval,
)
from scgsim.geometry.models.common import RouteLiteral
from scgsim.geometry.models.input import GeometryBuildInput, SemanticEntitySpec
from scgsim.geometry.models.regions import PortSheetRegionRecord
from scgsim.geometry.models.topology import (
    CurvePlanRecord,
    MMContactRecord,
    PointPlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
    VolumePlanRecord,
)
from scgsim.geometry.planning.domain import _solution_entity_xy_region
from scgsim.geometry.planning.topology import _surface_owner_ids


def _lower_port_sheet_regions(
    build_input: GeometryBuildInput,
    *,
    route: RouteLiteral,
    mm_contacts: tuple[MMContactRecord, ...],
    planned_surfaces: Sequence[SurfacePlanRecord] = (),
) -> tuple[SurfacePlanRecord, ...]:
    """Lower the one accepted Palace port-sheet shape before canonical topology."""
    entities = {entity.semantic_id: entity for entity in build_input.entities}
    polygons = {polygon.polygon_id: polygon for polygon in build_input.polygons}
    component_by_entity = {
        entity_id: str(component["conductor_component_id"])
        for component in _component_metadata(mm_contacts)
        for entity_id in component["members"]
    }
    records: list[SurfacePlanRecord] = list(planned_surfaces)
    for region in build_input.port_sheet_regions:
        metadata = region.metadata
        source_name = _port_sheet_string(metadata, "source_name", region)
        target_layer = _port_sheet_string(metadata, "target_layer", region)
        port_index = metadata.get("port_index")
        if (
            isinstance(port_index, bool)
            or not isinstance(port_index, int)
            or port_index < 1
        ):
            raise ValueError(f"{region.port_sheet_id} requires 1-based port_index")
        direction = _port_sheet_direction(region)
        raw_direction = _port_sheet_raw_direction(region)
        sign_convention = _port_sheet_string(
            metadata, "direction_sign_convention", region
        )
        if len(region.overlaps) != 2:
            raise ValueError(
                f"{region.port_sheet_id} requires exactly two host overlap records"
            )
        host_ids = _unique_ids(overlap.host_semantic_id for overlap in region.overlaps)
        if len(host_ids) != 2:
            raise ValueError(
                f"{region.port_sheet_id} requires one overlap per distinct "
                "face_metal host"
            )
        hosts = tuple(entities.get(host_id) for host_id in host_ids)
        if any(host is None for host in hosts):
            raise ValueError(f"{region.port_sheet_id} references an unknown port host")
        host_entities = tuple(host for host in hosts if host is not None)
        if any(
            host.part_role != "face_metal"
            or not _port_sheet_target_layer_matches(host, target_layer)
            or host.route_representations.get("A") != "surface_sheet"
            or host.route_representations.get("B") != "cutout_boundary_shell"
            for host in host_entities
        ):
            raise ValueError(
                f"{region.port_sheet_id} hosts must be active A/B face_metal "
                f"on {target_layer}"
            )
        z_values = tuple(
            float(host.geometry.get("z_um", float("nan"))) for host in host_entities
        )
        if not all(isfinite(value) for value in z_values) or not _same_z(
            z_values[0], z_values[1]
        ):
            raise ValueError(
                f"{region.port_sheet_id} hosts require the same finite z_um"
            )
        thickness_values: tuple[float, ...] = ()
        if route == "B":
            thickness_values = tuple(
                float(host.geometry.get("thickness_um", float("nan")))
                for host in host_entities
            )
            if (
                not all(isfinite(value) for value in thickness_values)
                or any(value <= 0.0 for value in thickness_values)
                or not _same_z(thickness_values[0], thickness_values[1])
            ):
                raise ValueError(
                    f"{region.port_sheet_id} Route B hosts require the same "
                    "finite positive thickness_um"
                )
        host_void_ids = tuple(host.host_void_semantic_id for host in host_entities)
        if route == "B" and (
            any(not host_void_id for host_void_id in host_void_ids)
            or len(set(host_void_ids)) != 1
            or host_void_ids[0] not in entities
            or not _is_solution_entity(entities[str(host_void_ids[0])])
        ):
            raise ValueError(
                f"{region.port_sheet_id} hosts require one common host_void solution"
            )
        if any(overlap.host_polygon_id not in polygons for overlap in region.overlaps):
            raise ValueError(
                f"{region.port_sheet_id} references an unknown host polygon"
            )
        remainder = _port_sheet_remainder_geometry(region)
        if not all(
            _port_sheet_boundary_shares_segment(remainder, overlap.overlap_loop)
            for host_id in host_ids
            for overlap in region.overlaps
            if overlap.host_semantic_id == host_id
        ):
            raise ValueError(
                f"{region.port_sheet_id} remainder must share a finite segment "
                "with each host"
            )
        host_polygon_ids = tuple(overlap.host_polygon_id for overlap in region.overlaps)
        owner_provenance = tuple(
            {
                "semantic_id": host.semantic_id,
                "net_id": host.net_id,
                "equipotential_id": host.metadata.get("equipotential_id"),
                "conductor_component_id": component_by_entity.get(
                    host.semantic_id,
                    f"COMP__{host.semantic_id}",
                ),
            }
            for host in host_entities
        )
        source_provenance = {
            "source_name": source_name,
            "port_index": port_index,
            "source_layer": region.source_layer,
            "source_polygon_id": region.source_polygon_id,
            "overlap_ids": tuple(overlap.overlap_id for overlap in region.overlaps),
            "overlaps": tuple(
                {
                    "overlap_id": overlap.overlap_id,
                    "host_semantic_id": overlap.host_semantic_id,
                    "host_polygon_id": overlap.host_polygon_id,
                    "overlap_loop": overlap.overlap_loop,
                }
                for overlap in region.overlaps
            ),
            "host_polygon_ids": host_polygon_ids,
            "route": route,
            "target_layer": target_layer,
            "direction": direction,
            "direction_raw": raw_direction,
            "direction_sign_convention": sign_convention,
        }
        if route == "A":
            plane_z_um, boundary_volume_ids = _route_a_port_sheet_sheet_contract(
                host_ids,
                planned_surfaces,
                region.overlaps,
            )
            embedded_volume_id = _route_a_port_sheet_vacuum_volume_id(
                boundary_volume_ids,
                entities,
            )
        else:
            plane_z_um = z_values[0] + thickness_values[0] / 2.0
            boundary_volume_ids = (host_void_ids[0],)
            embedded_volume_id = str(host_void_ids[0])
        port_surface = SurfacePlanRecord(
            surface_id=f"SURF__LUMPED_PORT__{_safe_port_sheet_name(source_name)}",
            owner_semantic_id=host_ids[0],
            surface_role="lumped_port",
            geometry_ref={
                "outer_loop": remainder["outer_loop"],
                "hole_loops": remainder["hole_loops"],
                "plane": {
                    "axis": "z",
                    "value_um": plane_z_um,
                },
                "representation": "lumped_port_sheet",
            },
            valid_routes=(route,),
            metadata={
                "route": route,
                "representation": "lumped_port_sheet",
                "interface_type": "lumped_port",
                "face_kind": "sheet",
                "owner_semantic_ids": host_ids,
                "physical_owner_semantic_ids": host_ids,
                "boundary_volume_ids": boundary_volume_ids,
                "route_a_boundary_port": route == "A",
                "embedded_surface": True,
                "embedded_volume_id": embedded_volume_id,
                "physical_name": (f"LUMPED_PORT__{_safe_port_sheet_name(source_name)}"),
                "source_provenance": source_provenance,
                "physical_attribute": {
                    "port_index": port_index,
                    "port_name": source_name,
                    "source_layer": region.source_layer,
                    "target_layer": target_layer,
                    "embedded_volume_id": embedded_volume_id,
                    "direction": direction,
                    "owner_semantic_ids": host_ids,
                    "owner_provenance": owner_provenance,
                },
            },
        )
        if route == "A":
            records = _carve_route_a_port_sheet_from_host_plane(
                records,
                port_surface=port_surface,
            )
        elif records:
            records = _partition_route_b_port_sheet_sidewalls(
                records,
                port_surface=port_surface,
                remainder=remainder,
                overlaps_by_host={
                    overlap.host_semantic_id: overlap for overlap in region.overlaps
                },
            )
        records.append(port_surface)
    return tuple(records)


def _bind_route_b_port_volume_plans(
    build_input: GeometryBuildInput,
    surfaces: tuple[SurfacePlanRecord, ...],
    volumes: tuple[VolumePlanRecord, ...],
) -> tuple[SurfacePlanRecord, ...]:
    """Bind active port remainders to final geometric vacuum pieces.

    Material physical groups may aggregate these pieces; their names cannot
    identify a native mesh embedding host.
    """
    import gdstk

    entities = {entity.semantic_id: entity for entity in build_input.entities}
    result = []
    for surface in surfaces:
        if surface.surface_role != "lumped_port":
            result.append(surface)
            continue
        z = _geometry_ref_surface_z_um(surface.geometry_ref)
        remainder = _gdstk_surface_region(surface.geometry_ref)
        hosts = []
        for volume in volumes:
            entity = entities[volume.owner_semantic_id]
            if entity.material_kind != "vacuum":
                continue
            z_min, z_max = _entity_z_range_um(entity)
            if not z_min < z < z_max:
                continue
            if _boolean_gdstk_region(
                gdstk, remainder, _solution_entity_xy_region(gdstk, entity), "and"
            ):
                hosts.append(volume.volume_id)
        if not hosts:
            raise ValueError(
                f"{surface.surface_id} active port has no final vacuum volume host"
            )
        result.append(
            replace(
                surface,
                metadata={**surface.metadata, "embedded_volume_plan_ids": tuple(hosts)},
            )
        )
    return tuple(result)


def _partition_route_b_port_sheet_sidewalls(
    surfaces: Sequence[SurfacePlanRecord],
    *,
    port_surface: SurfacePlanRecord,
    remainder: Mapping[str, Any],
    overlaps_by_host: Mapping[str, Any],
) -> list[SurfacePlanRecord]:
    """Split only the two overlap-bound Route-B PEC sidewalls at the port plane."""
    port_z_um = _geometry_ref_surface_z_um(port_surface.geometry_ref)
    terminal_segments_by_host = {
        host_id: tuple(
            (start, end)
            for boundary in (remainder["outer_loop"], *remainder["hole_loops"])
            for start, end in _ring_edges(_clean_loop(boundary))
            if any(
                _segment_overlap_interval(start, end, overlap_start, overlap_end)
                is not None
                for overlap_start, overlap_end in _ring_edges(
                    _clean_loop(overlap.overlap_loop)
                )
            )
        )
        for host_id, overlap in overlaps_by_host.items()
    }
    if any(not segments for segments in terminal_segments_by_host.values()):
        missing = tuple(
            host_id
            for host_id, segments in terminal_segments_by_host.items()
            if not segments
        )
        raise ValueError(
            f"{port_surface.surface_id} has no finite terminal segment for {missing!r}"
        )
    result: list[SurfacePlanRecord] = []
    split_hosts: set[str] = set()
    for surface in surfaces:
        shell_part = str(surface.geometry_ref.get("shell_part", ""))
        if (
            surface.surface_role != "cutout_boundary_shell"
            or surface.owner_semantic_id not in overlaps_by_host
            or not shell_part.startswith("sidewall_")
        ):
            result.append(surface)
            continue
        points = tuple(surface.geometry_ref.get("quad_points", ()))
        if len(points) != 4:
            raise ValueError(
                f"{surface.surface_id} Route B port host sidewall requires a quad"
            )
        quad = tuple(
            (float(point[0]), float(point[1]), float(point[2])) for point in points
        )
        if not any(
            _segment_overlap_interval(quad[0][:2], quad[1][:2], start, end) is not None
            for start, end in terminal_segments_by_host[surface.owner_semantic_id]
        ):
            result.append(surface)
            continue
        z_min_um = min(point[2] for point in quad)
        z_max_um = max(point[2] for point in quad)
        if not z_min_um < z_max_um:
            raise ValueError(
                f"{surface.surface_id} Route B port host sidewall has empty height"
            )
        overlap = overlaps_by_host[surface.owner_semantic_id]
        binding = {
            "port_surface_id": port_surface.surface_id,
            "overlap_id": str(overlap.overlap_id),
            "host_semantic_id": surface.owner_semantic_id,
        }
        if _same_z(port_z_um, z_min_um) or _same_z(port_z_um, z_max_um):
            if not surface.metadata.get("route_b_port_sheet_bindings"):
                raise ValueError(
                    f"{port_surface.surface_id} requires an imprinted "
                    f"sidewall child for {surface.surface_id}"
                )
            result.append(
                replace(
                    surface,
                    metadata=_route_b_port_sheet_partition_metadata(
                        surface.metadata,
                        binding=binding,
                        port_z_um=port_z_um,
                    ),
                )
            )
            split_hosts.add(surface.owner_semantic_id)
            continue
        if not z_min_um < port_z_um < z_max_um:
            raise ValueError(
                f"{port_surface.surface_id} must cut {surface.surface_id} at its "
                "finite sidewall interior"
            )
        lower_quad = (
            quad[0],
            quad[1],
            (quad[2][0], quad[2][1], port_z_um),
            (quad[3][0], quad[3][1], port_z_um),
        )
        upper_quad = (
            (quad[0][0], quad[0][1], port_z_um),
            (quad[1][0], quad[1][1], port_z_um),
            quad[2],
            quad[3],
        )
        parent_surface_id = surface.parent_surface_id or surface.surface_id
        partition_metadata = _route_b_port_sheet_partition_metadata(
            surface.metadata,
            binding=binding,
            port_z_um=port_z_um,
        )
        result.extend(
            (
                replace(
                    surface,
                    surface_id=(
                        f"{surface.surface_id}__{port_surface.surface_id}__LOWER"
                    ),
                    parent_surface_id=parent_surface_id,
                    partition_label="port_lower",
                    geometry_ref={
                        **dict(surface.geometry_ref),
                        "quad_points": lower_quad,
                    },
                    metadata=partition_metadata,
                ),
                replace(
                    surface,
                    surface_id=(
                        f"{surface.surface_id}__{port_surface.surface_id}__UPPER"
                    ),
                    parent_surface_id=parent_surface_id,
                    partition_label="port_upper",
                    geometry_ref={
                        **dict(surface.geometry_ref),
                        "quad_points": upper_quad,
                    },
                    metadata=partition_metadata,
                ),
            )
        )
        split_hosts.add(surface.owner_semantic_id)
    if set(overlaps_by_host) != split_hosts:
        missing = tuple(
            host_id for host_id in overlaps_by_host if host_id not in split_hosts
        )
        raise ValueError(
            f"{port_surface.surface_id} requires finite Route B sidewalls "
            f"for {missing!r}"
        )
    return result


def _route_b_port_sheet_partition_metadata(
    metadata: Mapping[str, Any],
    *,
    binding: Mapping[str, str],
    port_z_um: float,
) -> dict[str, Any]:
    bindings = tuple(
        dict(existing)
        for existing in metadata.get("route_b_port_sheet_bindings", ())
        if isinstance(existing, Mapping)
    )
    if len(bindings) != len(metadata.get("route_b_port_sheet_bindings", ())):
        raise ValueError("Route B port sidewall bindings must be structured records")
    if dict(binding) not in bindings:
        bindings = (*bindings, dict(binding))
    return {
        **dict(metadata),
        "route_b_port_sheet_sidewall_partition": True,
        "route_b_port_sheet_plane_z_um": port_z_um,
        "route_b_port_sheet_bindings": bindings,
    }


def _validate_route_b_port_sheet_sidewall_topology(
    *,
    route: RouteLiteral,
    points: Sequence[PointPlanRecord],
    curves: Sequence[CurvePlanRecord],
    surface_loops: Sequence[SurfaceLoopRecord],
    surfaces: Sequence[SurfacePlanRecord],
) -> None:
    """Require each Route-B port overlap to reuse just its bound PEC curve set."""
    if route != "B":
        return
    loops_by_id = {loop.loop_id: loop for loop in surface_loops}
    surfaces_by_id = {surface.surface_id: surface for surface in surfaces}
    points_by_id = {point.point_id: point.coordinate for point in points}
    curves_by_id = {curve.curve_id: curve for curve in curves}
    for port_surface in (
        surface for surface in surfaces if surface.surface_role == "lumped_port"
    ):
        owners = tuple(
            str(owner) for owner in port_surface.metadata["owner_semantic_ids"]
        )
        overlaps_by_host = {
            str(overlap["host_semantic_id"]): overlap
            for overlap in port_surface.metadata["source_provenance"]["overlaps"]
            if isinstance(overlap, Mapping)
        }
        port_curve_ids = {
            curve_ref.curve_id
            for loop_id in (port_surface.outer_loop_ref, *port_surface.hole_loop_refs)
            for curve_ref in loops_by_id[loop_id].curve_refs
        }
        if not port_curve_ids:
            raise ValueError(
                f"{port_surface.surface_id} has no canonical terminal curves"
            )
        for host_id in owners:
            overlap = overlaps_by_host.get(host_id)
            if overlap is None:
                raise ValueError(
                    f"{port_surface.surface_id} lacks exact overlap provenance "
                    f"for {host_id}"
                )
            expected_terminal_curve_ids = {
                curve_id
                for curve_id in port_curve_ids
                if _route_b_port_terminal_curve_matches_overlap(
                    curves_by_id[curve_id],
                    points_by_id=points_by_id,
                    overlap_loop=overlap["overlap_loop"],
                )
            }
            if not expected_terminal_curve_ids:
                raise ValueError(
                    f"{port_surface.surface_id} has no expected Route B terminal "
                    f"curves for {host_id}"
                )
            credited_terminal_curve_ids = {
                curve.curve_id
                for curve in curves
                if curve.curve_id in port_curve_ids
                and any(
                    candidate.surface_role == "cutout_boundary_shell"
                    and candidate.owner_semantic_id == host_id
                    and str(candidate.geometry_ref.get("shell_part", "")).startswith(
                        "sidewall_"
                    )
                    and candidate.metadata.get("route_b_port_sheet_sidewall_partition")
                    and _route_b_port_sheet_binding_matches(
                        candidate,
                        port_surface_id=port_surface.surface_id,
                        overlap_id=str(overlap["overlap_id"]),
                        host_semantic_id=host_id,
                    )
                    for candidate in (
                        surfaces_by_id[surface_id]
                        for surface_id in curve.used_by_surface_ids
                        if surface_id in surfaces_by_id
                    )
                )
            }
            if credited_terminal_curve_ids != expected_terminal_curve_ids:
                missing = sorted(
                    expected_terminal_curve_ids - credited_terminal_curve_ids
                )
                extra = sorted(
                    credited_terminal_curve_ids - expected_terminal_curve_ids
                )
                raise ValueError(
                    f"{port_surface.surface_id} Route B terminal curves for "
                    f"{host_id} PEC sidewall differ; missing={missing!r}, "
                    f"extra={extra!r}"
                )


def _route_b_port_terminal_curve_matches_overlap(
    curve: CurvePlanRecord,
    *,
    points_by_id: Mapping[str, tuple[float, float, float]],
    overlap_loop: Any,
) -> bool:
    start = points_by_id.get(curve.start_point_id)
    end = points_by_id.get(curve.end_point_id)
    if start is None or end is None:
        return False
    return any(
        _segment_overlap_interval(start[:2], end[:2], overlap_start, overlap_end)
        is not None
        for overlap_start, overlap_end in _ring_edges(_clean_loop(overlap_loop))
    )


def _route_a_port_sheet_sheet_contract(
    host_ids: Sequence[str],
    planned_surfaces: Sequence[SurfacePlanRecord],
    overlaps: Sequence[Any],
) -> tuple[float, tuple[str, str]]:
    """Select the local Route-A child intersecting each exact host footprint."""
    import gdstk

    host_planes: list[float] = []
    boundary_volume_ids: tuple[str, str] | None = None
    for host_id in host_ids:
        host_overlaps = tuple(
            overlap for overlap in overlaps if overlap.host_semantic_id == host_id
        )
        if len(host_overlaps) != 1:
            raise ValueError(
                f"{host_id} requires exactly one Route A port overlap record"
            )
        overlap_region = (gdstk.Polygon(_clean_loop(host_overlaps[0].overlap_loop)),)
        candidates = tuple(
            surface
            for surface in planned_surfaces
            if surface.metadata.get("representation") == "surface_sheet"
            # Route-A contact caps are per bump/face contact and bound a
            # single solution volume.  A layout junction sheet replaces the
            # actual two-volume face interface, never one of those caps.
            and surface.surface_role == "A_planned_interface"
            and host_id in _surface_owner_ids(surface)
        )
        host_sheets = tuple(
            surface
            for surface in candidates
            if _boolean_gdstk_region(
                gdstk,
                overlap_region,
                _gdstk_surface_region(surface.geometry_ref),
                "and",
            )
        )
        if len(host_sheets) != 1:
            raise ValueError(
                f"{host_id} port overlap intersects {len(host_sheets)} local "
                "Route A surface_sheet patches"
            )
        if _boolean_gdstk_region(
            gdstk,
            overlap_region,
            _gdstk_surface_region(host_sheets[0].geometry_ref),
            "not",
        ):
            raise ValueError(
                f"{host_id} port overlap crosses incompatible Route A domains"
            )
        plane_z_um = _geometry_ref_surface_z_um(host_sheets[0].geometry_ref)
        if not isfinite(plane_z_um):
            raise ValueError(f"{host_id} Route A surface_sheet plane must be finite")
        host_planes.append(plane_z_um)
        raw_boundary_ids = host_sheets[0].metadata.get("boundary_volume_ids")
        if isinstance(raw_boundary_ids, str):
            candidate_boundary_ids = ()
        else:
            candidate_boundary_ids = tuple(str(value) for value in raw_boundary_ids)
        if len(candidate_boundary_ids) != 2 or len(set(candidate_boundary_ids)) != 2:
            raise ValueError(
                f"{host_id} Route A surface_sheet requires two boundary volumes"
            )
        if boundary_volume_ids is None:
            boundary_volume_ids = candidate_boundary_ids
        elif boundary_volume_ids != candidate_boundary_ids:
            raise ValueError(
                "Route A lumped-port hosts require common boundary volumes"
            )
    if not _same_z(host_planes[0], host_planes[1]):
        raise ValueError("Route A lumped-port hosts require one common sheet plane")
    if boundary_volume_ids is None:
        raise ValueError("Route A lumped-port hosts require boundary volumes")
    return host_planes[0], boundary_volume_ids


def _route_a_port_sheet_vacuum_volume_id(
    boundary_volume_ids: Sequence[str],
    entities: Mapping[str, SemanticEntitySpec],
) -> str:
    """Select the typed vacuum member of the actual Route-A sheet adjacency."""
    boundary_entities: list[SemanticEntitySpec] = []
    for volume_id in boundary_volume_ids:
        entity = entities.get(volume_id)
        if entity is None or not _is_solution_entity(entity):
            raise ValueError(
                "Route A lumped-port boundary volumes must be typed solution regions"
            )
        boundary_entities.append(entity)
    vacuum_ids = tuple(
        volume_id
        for volume_id, entity in zip(
            boundary_volume_ids, boundary_entities, strict=True
        )
        if _is_vacuum_solution_entity(entity)
    )
    if len(vacuum_ids) != 1:
        raise ValueError(
            "Route A lumped-port boundary volumes require exactly one vacuum "
            "and one non-vacuum solution region"
        )
    return vacuum_ids[0]


def _carve_route_a_port_sheet_from_host_plane(
    surfaces: Sequence[SurfacePlanRecord],
    *,
    port_surface: SurfacePlanRecord,
) -> list[SurfacePlanRecord]:
    """Replace the co-planar solution face under a Route-A port with its hole."""
    import gdstk

    port_region = _gdstk_surface_region(port_surface.geometry_ref)
    port_z_um = _geometry_ref_surface_z_um(port_surface.geometry_ref)
    result: list[SurfacePlanRecord] = []
    for surface in surfaces:
        if (
            surface.construction_only
            or surface.metadata.get("representation") == "surface_sheet"
            or "outer_loop" not in surface.geometry_ref
            or not _same_z(_geometry_ref_surface_z_um(surface.geometry_ref), port_z_um)
        ):
            result.append(surface)
            continue
        overlap = _boolean_gdstk_region(
            gdstk,
            _gdstk_surface_region(surface.geometry_ref),
            port_region,
            "and",
        )
        if not overlap:
            result.append(surface)
            continue
        remainder = _boolean_gdstk_region(
            gdstk,
            _gdstk_surface_region(surface.geometry_ref),
            port_region,
            "not",
        )
        refs = _geometry_refs_from_gdstk_region(surface.geometry_ref, remainder)
        parent_surface_id = surface.parent_surface_id or surface.surface_id
        for index, geometry_ref in enumerate(refs):
            result.append(
                replace(
                    surface,
                    surface_id=(
                        surface.surface_id
                        if len(refs) == 1
                        else f"{surface.surface_id}__PORT_REMAINDER_{index:04d}"
                    ),
                    geometry_ref=geometry_ref,
                    parent_surface_id=(
                        surface.parent_surface_id
                        if len(refs) == 1
                        else parent_surface_id
                    ),
                    partition_label=(
                        surface.partition_label
                        if len(refs) == 1
                        else f"PORT_REMAINDER_{index:04d}"
                    ),
                )
            )
    return result


def _port_sheet_remainder_geometry(region: PortSheetRegionRecord) -> dict[str, Any]:
    import gdstk

    authored = _gdstk_surface_region(
        {"outer_loop": region.exterior, "hole_loops": region.holes}
    )
    overlaps = tuple(
        gdstk.Polygon(_clean_loop(overlap.overlap_loop)) for overlap in region.overlaps
    )
    remainder = _boolean_gdstk_region(gdstk, authored, overlaps, "not")
    refs = _geometry_refs_from_gdstk_region(
        {"outer_loop": region.exterior, "hole_loops": region.holes}, remainder
    )
    if len(refs) != 1:
        raise ValueError(
            f"{region.port_sheet_id} requires exactly one finite active remainder"
        )
    result = refs[0]
    area = abs(_polygon_area(_clean_loop(result["outer_loop"]))) - sum(
        abs(_polygon_area(_clean_loop(hole))) for hole in result["hole_loops"]
    )
    if not isfinite(area) or area <= _TOPOLOGY_EPS_UM:
        raise ValueError(f"{region.port_sheet_id} active remainder must be finite")
    return result


def _port_sheet_boundary_shares_segment(
    remainder: Mapping[str, Any], overlap_loop: Any
) -> bool:
    return any(
        _loops_share_edge_overlap(boundary, overlap_loop)
        for boundary in (remainder["outer_loop"], *remainder["hole_loops"])
    )


def _port_sheet_string(
    metadata: Mapping[str, Any], key: str, region: PortSheetRegionRecord
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{region.port_sheet_id} requires {key}")
    return value


def _port_sheet_direction(region: PortSheetRegionRecord) -> tuple[float, float, float]:
    raw = region.metadata.get("direction")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"{region.port_sheet_id} requires a 3D direction")
    direction = tuple(float(value) for value in raw)
    if (
        not all(isfinite(value) for value in direction)
        or direction[2] != 0.0
        or direction[0] == direction[1] == 0.0
    ):
        raise ValueError(
            f"{region.port_sheet_id} direction must be finite, XY, and nonzero"
        )
    length = sqrt(direction[0] ** 2 + direction[1] ** 2)
    return (direction[0] / length, direction[1] / length, 0.0)


def _port_sheet_raw_direction(
    region: PortSheetRegionRecord,
) -> tuple[float, float, float]:
    raw = region.metadata.get("direction_raw")
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"{region.port_sheet_id} requires raw direction provenance")
    direction = tuple(float(value) for value in raw)
    if (
        not all(isfinite(value) for value in direction)
        or direction[2] != 0.0
        or direction[0] == direction[1] == 0.0
    ):
        raise ValueError(
            f"{region.port_sheet_id} raw direction must be finite, XY, and nonzero"
        )
    return direction


def _port_sheet_target_layer_matches(
    entity: SemanticEntitySpec, target_layer: str
) -> bool:
    logical_layer = entity.metadata.get("logical_layer_id")
    return target_layer in {
        str(entity.metadata.get("semantic_group_id", "")),
        logical_layer if isinstance(logical_layer, str) else "",
        f"{entity.geometry.get('gds_layer')}/{entity.geometry.get('gds_datatype')}",
    }


def _safe_port_sheet_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)
