"""Machine-checkable Engine Gates for conformal geometry claims.

An Engine Gate is the evidence boundary between "the compiler produced a
reviewable plan" and "SGB can claim this route output is conformal geometry."
These reports are stricter than ordinary validation because they name the
specific engine-level claim being made, list the records that were checked, and
fail loudly when the claim is not machine-checkable.

The SGB-owned gates are intentionally small:

- `volume_adjacency_conformality` proves planned live surfaces are referenced by
  the expected one or two volumes.
- `gmsh_brep_conformality` proves the lowered Gmsh BRep boundaries still match
  planned curves and surfaces after `occ.synchronize()`.

Downstream tetra mesh topology remains a fourth gate owned by the meshing
consumer. Passing these reports means SGB has produced conformal CAD topology;
it does not mean a particular mesh-size strategy is solver-ready.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scgsim.sgb.models import ConstructionPlanRecord

_COORD_TOL_UM = 1e-9


def assert_engine_gate_pass(report: Mapping[str, Any]) -> None:
    """Fail the build when an Engine Gate report does not pass.

    Engine Gate failures are not warnings. If a report is missing or has
    `status != "pass"`, the route output may still be useful for debugging, but
    it must not be treated as proven conformal geometry.
    """
    if not isinstance(report, Mapping):
        raise TypeError("Engine Gate report is missing")
    if report.get("status") != "pass":
        failures = report.get("failures", ())
        raise ValueError(
            f"Engine Gate {report.get('engine_gate')} failed: {failures!r}"
        )


def engine_gate_volume_adjacency_conformality(
    plan: ConstructionPlanRecord,
) -> dict[str, Any]:
    """Prove every live planned surface has the expected volume incidence.

    This is the plan-level shared-face contract. Each live surface must declare
    `metadata["boundary_volume_ids"]`, and the volumes that actually reference
    that surface must match those ids exactly. Exterior faces have one adjacent
    volume; retained internal interfaces have two. More than two adjacent
    volumes is a non-manifold topology failure.
    """
    volume_refs_by_surface: dict[str, list[str]] = {}
    failures: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for volume in plan.volumes:
        seen: set[str] = set()
        for surface_ref in volume.surface_refs:
            if surface_ref.surface_id in seen:
                failures.append(
                    _failure(
                        "duplicate_surface_ref_in_volume",
                        [volume.volume_id, surface_ref.surface_id],
                        "A volume references the same surface more than once.",
                    )
                )
            seen.add(surface_ref.surface_id)
            volume_refs_by_surface.setdefault(surface_ref.surface_id, []).append(
                volume.owner_semantic_id
            )

    for surface in plan.surfaces:
        if surface.construction_only:
            continue
        if surface.surface_role == "lumped_port" and not surface.metadata.get(
            "route_a_boundary_port"
        ):
            records.append(
                {
                    "surface_id": surface.surface_id,
                    "physical_name": surface.metadata.get("physical_name"),
                    "interface_type": "lumped_port",
                    "expected_adjacent_volume_ids": (),
                    "actual_adjacent_volume_ids": (),
                    "volume_use_count": 0,
                    "expected_use_count": 0,
                    "status": "pass",
                }
            )
            continue
        expected = tuple(
            str(value) for value in surface.metadata.get("boundary_volume_ids", ())
        )
        actual = tuple(sorted(volume_refs_by_surface.get(surface.surface_id, ())))
        expected_sorted = tuple(sorted(expected))
        expected_count = len(expected_sorted)
        actual_count = len(actual)
        status = "pass"
        if not expected_sorted:
            status = "fail"
            failures.append(
                _failure(
                    "missing_boundary_volume_ids",
                    [surface.surface_id],
                    "Live surface lacks explicit boundary_volume_ids metadata.",
                )
            )
        elif expected_count not in {1, 2}:
            status = "fail"
            failures.append(
                _failure(
                    "invalid_expected_volume_count",
                    [surface.surface_id],
                    "Expected surface incidence must be one or two volumes.",
                )
            )
        elif actual_count != expected_count or actual != expected_sorted:
            status = "fail"
            failures.append(
                _failure(
                    "volume_adjacency_mismatch",
                    [surface.surface_id],
                    "Actual volume owners do not match boundary_volume_ids.",
                )
            )
        if actual_count > 2:
            status = "fail"
            failures.append(
                _failure(
                    "surface_used_by_more_than_two_volumes",
                    [surface.surface_id],
                    "A live surface is referenced by more than two volumes.",
                )
            )
        records.append(
            {
                "surface_id": surface.surface_id,
                "physical_name": surface.metadata.get("physical_name"),
                "interface_type": "_".join(surface.metadata.get("interface_kinds", ())),
                "expected_adjacent_volume_ids": expected_sorted,
                "actual_adjacent_volume_ids": actual,
                "volume_use_count": actual_count,
                "expected_use_count": expected_count,
                "status": status,
            }
        )

    return _report(
        plan,
        "volume_adjacency_conformality",
        "after_build_route_construction_plan",
        failures,
        records,
        counts={"surfaces": len(records)},
    )


def _append_lumped_port_mesh_conformality(
    plan: ConstructionPlanRecord,
    *,
    gmsh: Any,
    loops_by_id: Mapping[str, Any],
    curve_tags: Mapping[str, int],
    surface_tag_by_id: Mapping[str, int],
    failures: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> None:
    """Fail closed unless each Route-B terminal uses its exact owner-sidewall edge."""
    if plan.route not in {"A", "B"}:
        return
    port_surfaces = tuple(
        surface for surface in plan.surfaces if surface.surface_role == "lumped_port"
    )
    if not port_surfaces:
        return
    try:
        gmsh.model.mesh.generate(2)
        for port_surface in port_surfaces:
            port_tag = surface_tag_by_id.get(port_surface.surface_id)
            if port_tag is None:
                continue
            port_nodes = _surface_mesh_node_tags(gmsh, port_tag)
            owners = tuple(
                str(owner) for owner in port_surface.metadata["owner_semantic_ids"]
            )
            port_curve_ids = {
                curve_ref.curve_id
                for loop_id in (
                    port_surface.outer_loop_ref,
                    *port_surface.hole_loop_refs,
                )
                for curve_ref in loops_by_id[loop_id].curve_refs
            }
            for owner_id in owners:
                overlap = _route_b_port_overlap(port_surface, owner_id)
                overlap_id = (
                    str(overlap.get("overlap_id"))
                    if isinstance(overlap, Mapping)
                    else None
                )
                owner_pec_surfaces = tuple(
                    surface
                    for surface in plan.surfaces
                    if surface.owner_semantic_id == owner_id
                    and (
                        (
                            plan.route == "A"
                            and surface.metadata.get("representation")
                            == "surface_sheet"
                        )
                        or (
                            plan.route == "B"
                            and surface.surface_role == "cutout_boundary_shell"
                            and str(
                                surface.geometry_ref.get("shell_part", "")
                            ).startswith("sidewall_")
                            and surface.metadata.get(
                                "route_b_port_sheet_sidewall_partition"
                            )
                            and _route_b_port_sheet_binding_matches(
                                surface,
                                port_surface_id=port_surface.surface_id,
                                overlap_id=overlap_id,
                                host_semantic_id=owner_id,
                            )
                        )
                    )
                )
                sidewall_tags = tuple(
                    surface_tag_by_id[surface.surface_id]
                    for surface in owner_pec_surfaces
                    if surface.surface_id in surface_tag_by_id
                )
                owner_surface_ids = {
                    surface.surface_id for surface in owner_pec_surfaces
                }
                credited_terminal_curve_ids = tuple(
                    curve.curve_id
                    for curve in plan.curves
                    if curve.curve_id in port_curve_ids
                    and owner_surface_ids.intersection(curve.used_by_surface_ids)
                )
                owner_sidewall_nodes: set[int] = set()
                for sidewall_tag in sidewall_tags:
                    owner_sidewall_nodes.update(
                        _surface_mesh_node_tags(gmsh, sidewall_tag)
                    )
                shared_nodes = port_nodes & owner_sidewall_nodes
                expected_terminal_curve_ids = (
                    _route_b_expected_terminal_curve_ids(
                        plan,
                        port_curve_ids=port_curve_ids,
                        overlap=overlap,
                    )
                    if plan.route == "B"
                    else credited_terminal_curve_ids
                )
                terminal_curve_node_counts = {
                    curve_id: len(
                        _curve_mesh_node_tags(gmsh, curve_tags[curve_id])
                        & port_nodes
                        & owner_sidewall_nodes
                    )
                    for curve_id in expected_terminal_curve_ids
                    if curve_id in curve_tags
                }
                exact_terminal_coverage = set(credited_terminal_curve_ids) == set(
                    expected_terminal_curve_ids
                )
                status = (
                    "pass"
                    if (
                        shared_nodes
                        and exact_terminal_coverage
                        and len(terminal_curve_node_counts)
                        == len(expected_terminal_curve_ids)
                        and all(terminal_curve_node_counts.values())
                    )
                    else "fail"
                )
                if status == "fail":
                    failures.append(
                        _failure(
                            "lumped_port_missing_shared_mesh_nodes",
                            [port_surface.surface_id, owner_id],
                            f"Route {plan.route} lumped-port sheet must share exact "
                            "terminal curves and mesh nodes with each owner PEC "
                            "surface.",
                        )
                    )
                records.append(
                    {
                        "source_record_kind": (
                            f"route_{plan.route.lower()}_lumped_port_mesh"
                        ),
                        "source_record_id": port_surface.surface_id,
                        "owner_semantic_id": owner_id,
                        "overlap_id": overlap_id,
                        "backend_port_tag": port_tag,
                        "backend_owner_pec_surface_tags": sorted(sidewall_tags),
                        "bound_owner_pec_surface_ids": sorted(owner_surface_ids),
                        "expected_terminal_curve_ids": sorted(
                            expected_terminal_curve_ids
                        ),
                        "credited_terminal_curve_ids": sorted(
                            credited_terminal_curve_ids
                        ),
                        "per_terminal_curve_shared_node_counts": (
                            terminal_curve_node_counts
                        ),
                        "planned_shared_curve_ids": sorted(credited_terminal_curve_ids),
                        "backend_shared_curve_tags": sorted(
                            curve_tags[curve_id]
                            for curve_id in credited_terminal_curve_ids
                            if curve_id in curve_tags
                        ),
                        "shared_mesh_node_count": len(shared_nodes),
                        "status": status,
                    }
                )
    except Exception as exc:  # noqa: BLE001 - report a failed disposable mesh probe
        for port_surface in port_surfaces:
            failures.append(
                _failure(
                    "lumped_port_mesh_generation_failed",
                    [port_surface.surface_id],
                    f"Route {plan.route} lumped-port mesh probe failed: {exc}",
                )
            )
    finally:
        gmsh.model.mesh.clear()


def _surface_mesh_node_tags(gmsh: Any, surface_tag: int) -> set[int]:
    node_tags, _, _ = gmsh.model.mesh.getNodes(
        2,
        surface_tag,
        includeBoundary=True,
    )
    return {int(node_tag) for node_tag in node_tags}


def _curve_mesh_node_tags(gmsh: Any, curve_tag: int) -> set[int]:
    node_tags, _, _ = gmsh.model.mesh.getNodes(
        1,
        curve_tag,
        includeBoundary=True,
    )
    return {int(node_tag) for node_tag in node_tags}


def _route_b_port_overlap(port_surface: Any, owner_id: str) -> Mapping[str, Any] | None:
    source_provenance = port_surface.metadata.get("source_provenance", {})
    if not isinstance(source_provenance, Mapping):
        return None
    for overlap in source_provenance.get("overlaps", ()):
        if isinstance(overlap, Mapping) and overlap.get("host_semantic_id") == owner_id:
            return overlap
    return None


def _route_b_expected_terminal_curve_ids(
    plan: ConstructionPlanRecord,
    *,
    port_curve_ids: set[str],
    overlap: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    if overlap is None:
        return ()
    overlap_loop = overlap.get("overlap_loop")
    if not isinstance(overlap_loop, Sequence):
        return ()
    points_by_id = {point.point_id: point.coordinate for point in plan.points}
    curves_by_id = {curve.curve_id: curve for curve in plan.curves}
    return tuple(
        sorted(
            curve_id
            for curve_id in port_curve_ids
            if curve_id in curves_by_id
            and _curve_matches_overlap(
                curves_by_id[curve_id],
                points_by_id=points_by_id,
                overlap_loop=overlap_loop,
            )
        )
    )


def _curve_matches_overlap(
    curve: Any,
    *,
    points_by_id: Mapping[str, tuple[float, float, float]],
    overlap_loop: Sequence[Any],
) -> bool:
    start = points_by_id.get(curve.start_point_id)
    end = points_by_id.get(curve.end_point_id)
    if start is None or end is None:
        return False
    return any(
        _segments_overlap(start[:2], end[:2], overlap_start, overlap_end)
        for overlap_start, overlap_end in _ring_edges(overlap_loop)
    )


def _ring_edges(
    loop: Sequence[Any],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    points = tuple((float(point[0]), float(point[1])) for point in loop)
    return tuple(
        (points[index], points[(index + 1) % len(points)])
        for index in range(len(points))
    )


def _segments_overlap(
    start: tuple[float, float],
    end: tuple[float, float],
    other_start: tuple[float, float],
    other_end: tuple[float, float],
) -> bool:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-18:
        return False
    cross_start = (other_start[0] - start[0]) * dy - (other_start[1] - start[1]) * dx
    cross_end = (other_end[0] - start[0]) * dy - (other_end[1] - start[1]) * dx
    if abs(cross_start) > 1e-9 or abs(cross_end) > 1e-9:
        return False
    first = (
        (other_start[0] - start[0]) * dx + (other_start[1] - start[1]) * dy
    ) / length_sq
    second = (
        (other_end[0] - start[0]) * dx + (other_end[1] - start[1]) * dy
    ) / length_sq
    return min(1.0, max(first, second)) - max(0.0, min(first, second)) > 1e-9


def _route_b_port_sheet_binding_matches(
    surface: Any,
    *,
    port_surface_id: str,
    overlap_id: str | None,
    host_semantic_id: str,
) -> bool:
    return any(
        isinstance(binding, Mapping)
        and binding.get("port_surface_id") == port_surface_id
        and binding.get("overlap_id") == overlap_id
        and binding.get("host_semantic_id") == host_semantic_id
        for binding in surface.metadata.get("route_b_port_sheet_bindings", ())
    )


def engine_gate_gmsh_brep_conformality(
    plan: ConstructionPlanRecord,
    *,
    gmsh: Any,
    source_tags: Mapping[tuple[str, str], Sequence[tuple[int, int]]],
    curve_tags: Mapping[str, int],
) -> dict[str, Any]:
    """Prove lowered Gmsh BRep topology still matches the compiler plan.

    This gate runs after `gmsh.model.occ.synchronize()`, when live backend tags
    exist. It checks that each planned surface maps to exactly one Gmsh surface,
    each planned volume maps to exactly one Gmsh volume, surface boundaries use
    the planned curve tags, and volume boundaries use the planned surface tags.
    Coordinate coincidence is not enough: this gate is about live BRep tags.
    """
    failures: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    loops_by_id = {loop.loop_id: loop for loop in plan.surface_loops}
    surface_tag_by_id = _single_tag_map(source_tags, "surface", failures)
    volume_tag_by_id = _single_tag_map(source_tags, "volume", failures)

    for surface in plan.surfaces:
        if surface.construction_only:
            continue
        surface_tag = surface_tag_by_id.get(surface.surface_id)
        if surface_tag is None:
            failures.append(
                _failure(
                    "missing_backend_surface_tag",
                    [surface.surface_id],
                    "Live surface has no lowered Gmsh surface tag.",
                )
            )
            continue
        expected_curves = _planned_surface_curve_tags(
            surface,
            loops_by_id,
            curve_tags,
        )
        actual_curves = {
            tag
            for dim, tag in gmsh.model.getBoundary(
                [(2, surface_tag)],
                oriented=False,
                recursive=False,
            )
            if dim == 1
        }
        status = "pass" if actual_curves == expected_curves else "fail"
        if status == "fail":
            failures.append(
                _failure(
                    "surface_boundary_curve_mismatch",
                    [surface.surface_id],
                    "Gmsh surface boundary curves do not match planned curves.",
                )
            )
        records.append(
            {
                "source_record_kind": "surface",
                "source_record_id": surface.surface_id,
                "backend_tag": surface_tag,
                "expected_curve_tags": sorted(expected_curves),
                "actual_curve_tags": sorted(actual_curves),
                "status": status,
            }
        )

    for volume in plan.volumes:
        if volume.construction_only:
            continue
        volume_tag = volume_tag_by_id.get(volume.volume_id)
        if volume_tag is None:
            failures.append(
                _failure(
                    "missing_backend_volume_tag",
                    [volume.volume_id],
                    "Live volume has no lowered Gmsh volume tag.",
                )
            )
            continue
        expected_surfaces = {
            surface_tag_by_id[surface_ref.surface_id]
            for surface_ref in volume.surface_refs
            if surface_ref.surface_id in surface_tag_by_id
        }
        actual_surfaces = {
            tag
            for dim, tag in gmsh.model.getBoundary(
                [(3, volume_tag)],
                oriented=False,
                recursive=False,
            )
            if dim == 2
        }
        status = "pass" if actual_surfaces == expected_surfaces else "fail"
        if status == "fail":
            failures.append(
                _failure(
                    "volume_boundary_surface_mismatch",
                    [volume.volume_id],
                    "Gmsh volume boundary surfaces do not match planned surfaces.",
                )
            )
        records.append(
            {
                "source_record_kind": "volume",
                "source_record_id": volume.volume_id,
                "backend_tag": volume_tag,
                "expected_surface_tags": sorted(expected_surfaces),
                "actual_surface_tags": sorted(actual_surfaces),
                "status": status,
            }
        )

    _append_lumped_port_mesh_conformality(
        plan,
        gmsh=gmsh,
        loops_by_id=loops_by_id,
        curve_tags=curve_tags,
        surface_tag_by_id=surface_tag_by_id,
        failures=failures,
        records=records,
    )

    return _report(
        plan,
        "gmsh_brep_conformality",
        "after_occ_synchronize",
        failures,
        records,
        counts={
            "surfaces": len(surface_tag_by_id),
            "volumes": len(volume_tag_by_id),
        },
    )


def _single_tag_map(
    source_tags: Mapping[tuple[str, str], Sequence[tuple[int, int]]],
    source_kind: str,
    failures: list[dict[str, Any]],
) -> dict[str, int]:
    records: dict[str, int] = {}
    dimtags_seen: dict[tuple[int, int], str] = {}
    for (kind, source_id), dimtags in source_tags.items():
        if kind != source_kind:
            continue
        if len(dimtags) != 1:
            failures.append(
                _failure(
                    "source_maps_to_multiple_backend_tags",
                    [source_id],
                    "A live source id must map to exactly one backend dim-tag.",
                )
            )
            continue
        dimtag = tuple(dimtags[0])
        existing = dimtags_seen.get(dimtag)
        if existing is not None:
            failures.append(
                _failure(
                    "backend_tag_shared_by_sources",
                    [existing, source_id],
                    "A backend dim-tag is shared by multiple source ids.",
                )
            )
        dimtags_seen[dimtag] = source_id
        records[source_id] = int(dimtag[1])
    return records


def _planned_surface_curve_tags(
    surface: Any,
    loops_by_id: Mapping[str, Any],
    curve_tags: Mapping[str, int],
) -> set[int]:
    curve_ids: set[str] = set()
    loop_ids = (
        *((surface.outer_loop_ref,) if surface.outer_loop_ref is not None else ()),
        *surface.hole_loop_refs,
    )
    for loop_id in loop_ids:
        loop = loops_by_id.get(loop_id)
        if loop is None:
            continue
        curve_ids.update(curve_ref.curve_id for curve_ref in loop.curve_refs)
    return {curve_tags[curve_id] for curve_id in curve_ids if curve_id in curve_tags}


def _report(
    plan: ConstructionPlanRecord,
    engine_gate: str,
    stage: str,
    failures: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema": "sgb.engine_gate.v1",
        "engine_gate": engine_gate,
        "route": plan.route,
        "status": "fail" if failures else "pass",
        "stage": stage,
        "checked_record_ids": {
            "interfaces": [record.interface_id for record in plan.interfaces],
            "surface_partitions": [
                record.partition_id for record in plan.surface_partitions
            ],
            "points": [record.point_id for record in plan.points],
            "curves": [record.curve_id for record in plan.curves],
            "surface_loops": [record.loop_id for record in plan.surface_loops],
            "surfaces": [record.surface_id for record in plan.surfaces],
            "volumes": [record.volume_id for record in plan.volumes],
            "backend_entity_tags": [
                record.source_record_id for record in plan.backend_entity_tags
            ],
        },
        "counts": dict(counts),
        "tolerances": {
            "coordinate_um": _COORD_TOL_UM,
        },
        "failures": failures,
        "records": records,
    }


def _failure(
    code: str,
    record_ids: Sequence[str],
    message: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "record_ids": list(record_ids),
        "message": message,
    }
