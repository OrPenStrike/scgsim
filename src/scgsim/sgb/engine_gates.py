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
