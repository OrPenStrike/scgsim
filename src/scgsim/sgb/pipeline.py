"""Public route-aware semantic geometry pipeline facade.

The heavy semantic work lives in `planning.py`, `engine_gates.py`, and
`validation.py`. This module keeps the public import path stable and owns only
run orchestration: validate, plan, write sidecars, run Engine Gates, hand the
plan to the bottom-up OCC backend, then export physical groups.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from scgsim.sgb.backends.gmsh_occ_backend import (
    write_occ_geometry_from_plan,
)
from scgsim.sgb.engine_gates import (
    assert_engine_gate_pass,
    engine_gate_volume_adjacency_conformality,
)
from scgsim.sgb.export import export_physical_group_records
from scgsim.sgb.models import (
    RUN_METADATA_DIR,
    SEMANTIC_GEOMETRY_METADATA_DIR,
    FinalPhysicalGroupRecord,
    GeometryBuildInput,
    PathInput,
    RouteLiteral,
)
from scgsim.sgb.planning import (
    build_route_construction_plan,
    plan_conductor_contact_patches,
    plan_cut_host_operations,
    plan_mm_contact_records,
    plan_route_construction_bodies,
    plan_route_surfaces,
    plan_route_tags,
    plan_route_volumes,
    plan_surface_partitions,
    recognize_route_interfaces,
)
from scgsim.sgb.validation import (
    validate_backend_tag_ledger,
    validate_curve_plan_coverage,
    validate_geometry_input,
    validate_interface_surface_source_of_truth,
    validate_no_surface_overlap,
    validate_route_operation_coverage,
    validate_route_volume_surface_refs,
    validate_selected_route,
    validate_surface_deduplication,
    validate_surface_partition_coverage,
    validate_surface_sheet_interface_coverage,
    validate_surface_use_counts,
    validate_tag_plan_coverage,
    validate_volume_surface_closure,
)

__all__ = [
    "SemanticGeometryBuilder",
    "build_route_construction_plan",
    "export_physical_group_records",
    "plan_conductor_contact_patches",
    "plan_cut_host_operations",
    "plan_mm_contact_records",
    "plan_route_construction_bodies",
    "plan_route_surfaces",
    "plan_route_tags",
    "plan_route_volumes",
    "plan_surface_partitions",
    "recognize_route_interfaces",
    "validate_backend_tag_ledger",
    "validate_curve_plan_coverage",
    "validate_geometry_input",
    "validate_interface_surface_source_of_truth",
    "validate_no_surface_overlap",
    "validate_route_operation_coverage",
    "validate_route_volume_surface_refs",
    "validate_selected_route",
    "validate_surface_deduplication",
    "validate_surface_partition_coverage",
    "validate_surface_sheet_interface_coverage",
    "validate_surface_use_counts",
    "validate_tag_plan_coverage",
    "validate_volume_surface_closure",
]


class SemanticGeometryBuilder:
    """Facade for one route-aware semantic geometry construction run.

    The builder must not construct arbitrary volumes and then ask the backend
    to fragment them to discover topology. Route selection is an early input:
    Route A plans sheets/shells, Route B plans cutout shells, and Route C plans
    retained material volumes with explicit shared surfaces where required.

    `build()` is the reviewable top-level pipeline:

    1. validate the adapter-owned `GeometryBuildInput`;
    2. build a route-specific `ConstructionPlanRecord`;
    3. write the pre-lowering volume-adjacency Engine Gate artifact;
    4. call the bottom-up Gmsh/OCC backend with that plan;
    5. write the post-lowering Gmsh BRep Engine Gate artifact;
    6. export `FinalPhysicalGroupRecord`s from backend dim-tags.

    One call writes one route XAO file. Mesh generation is downstream.
    """

    def build(
        self,
        build_input: GeometryBuildInput,
        *,
        route: RouteLiteral,
        run_folder: PathInput,
    ) -> tuple[FinalPhysicalGroupRecord, ...]:
        """Validate input, plan geometry, build OCC, then export groups."""
        run_path = Path(run_folder)
        metadata_dir = run_path / RUN_METADATA_DIR / SEMANTIC_GEOMETRY_METADATA_DIR
        geometry_dir = run_path / "geometry"
        geometry_dir.mkdir(parents=True, exist_ok=True)
        timings: list[dict[str, Any]] = []
        plan = None
        built_plan = None
        started = time.perf_counter()

        try:
            validated_input = _timed(
                timings,
                "validate_geometry_input",
                lambda: validate_geometry_input(build_input),
            )
            _timed(
                timings,
                "validate_selected_route",
                lambda: validate_selected_route(validated_input, route),
            )
            _timed(
                timings,
                "write_01_validate_geometry_input",
                lambda: _write_stage_sidecar(
                    metadata_dir,
                    "01_validate_geometry_input",
                    validated_input,
                ),
            )

            plan = _timed(
                timings,
                "build_route_construction_plan",
                lambda: build_route_construction_plan(validated_input, route=route),
            )
            _timed(
                timings,
                "write_02_build_route_construction_plan",
                lambda: _write_stage_sidecar(
                    metadata_dir,
                    "02_build_route_construction_plan",
                    plan,
                ),
            )
            volume_adjacency_gate = _timed(
                timings,
                "engine_gate_volume_adjacency_conformality",
                lambda: engine_gate_volume_adjacency_conformality(plan),
            )
            _timed(
                timings,
                "write_engine_gate_volume_adjacency_conformality",
                lambda: _write_stage_sidecar(
                    metadata_dir,
                    "engine_gate_volume_adjacency_conformality",
                    volume_adjacency_gate,
                ),
            )
            _timed(
                timings,
                "assert_engine_gate_volume_adjacency_conformality",
                lambda: assert_engine_gate_pass(volume_adjacency_gate),
            )

            built_plan = _timed(
                timings,
                "write_occ_geometry_from_plan",
                lambda: write_occ_geometry_from_plan(
                    plan,
                    xao_path=(
                        geometry_dir / f"semantic_geometry_route_{route.lower()}.xao"
                    ),
                ),
            )
            _timed(
                timings,
                "write_03_build_occ_geometry",
                lambda: _write_stage_sidecar(
                    metadata_dir,
                    "03_build_occ_geometry",
                    built_plan,
                ),
            )
            gmsh_brep_gate = built_plan.metadata.get(
                "engine_gate_gmsh_brep_conformality"
            )
            _timed(
                timings,
                "write_engine_gate_gmsh_brep_conformality",
                lambda: _write_stage_sidecar(
                    metadata_dir,
                    "engine_gate_gmsh_brep_conformality",
                    gmsh_brep_gate,
                ),
            )
            _timed(
                timings,
                "assert_engine_gate_gmsh_brep_conformality",
                lambda: assert_engine_gate_pass(gmsh_brep_gate),
            )

            _timed(
                timings,
                "validate_backend_tag_ledger",
                lambda: validate_backend_tag_ledger(
                    backend_tags=built_plan.backend_entity_tags
                ),
            )
            physical_groups = _timed(
                timings,
                "export_physical_group_records",
                lambda: export_physical_group_records(built_plan),
            )
            _timed(
                timings,
                "write_04_export_physical_groups",
                lambda: _write_stage_sidecar(
                    metadata_dir,
                    "04_export_physical_groups",
                    physical_groups,
                ),
            )
            return physical_groups
        finally:
            _write_stage_sidecar(
                metadata_dir,
                "00_timing",
                _timing_payload(route, timings, plan, built_plan, started),
            )


def _timed(
    timings: list[dict[str, Any]],
    stage: str,
    fn: Any,
) -> Any:
    started = time.perf_counter()
    try:
        result = fn()
    except Exception:
        timings.append(_timing_record(stage, started, "failed"))
        raise
    timings.append(_timing_record(stage, started, "done"))
    return result


def _timing_record(stage: str, started: float, status: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "seconds": round(time.perf_counter() - started, 6),
        "status": status,
    }


def _timing_payload(
    route: RouteLiteral,
    timings: list[dict[str, Any]],
    plan: Any,
    built_plan: Any,
    started: float,
) -> dict[str, Any]:
    return {
        "route": route,
        "total_seconds": round(time.perf_counter() - started, 6),
        "counts": _timing_counts(plan, built_plan),
        "stages": timings,
        "planning_stages": list(getattr(plan, "metadata", {}).get("timings", ())),
        "backend_stages": list(
            getattr(built_plan, "metadata", {}).get("backend_timings", ())
        ),
    }


def _timing_counts(plan: Any, built_plan: Any) -> dict[str, Any]:
    if plan is None:
        return {}
    counts = {
        "interfaces": len(getattr(plan, "interfaces", ())),
        "surface_partitions": len(getattr(plan, "surface_partitions", ())),
        "points": len(getattr(plan, "points", ())),
        "curves": len(getattr(plan, "curves", ())),
        "surface_loops": len(getattr(plan, "surface_loops", ())),
        "surfaces": len(getattr(plan, "surfaces", ())),
        "volumes": len(getattr(plan, "volumes", ())),
        "construction_bodies": len(getattr(plan, "construction_bodies", ())),
        "cut_operations": len(getattr(plan, "cut_operations", ())),
        "tags": len(getattr(plan, "tags", ())),
    }
    xao_path = (
        getattr(built_plan, "metadata", {}).get("xao_path") if built_plan else None
    )
    if xao_path and Path(xao_path).is_file():
        counts["xao_bytes"] = Path(xao_path).stat().st_size
    return counts


def _write_stage_sidecar(path: Path, stage_name: str, payload: Any) -> None:
    path.mkdir(parents=True, exist_ok=True)
    # Sidecars are retained with handoff artifacts.  They may describe files
    # the builder used locally, but must not publish a workstation path.  Keep
    # paths inside this run addressable from the run root and reduce any
    # external input to a non-location-bearing label.
    public_payload = _redact_local_paths(payload, run_root=path.parents[1])
    path.joinpath(f"{stage_name}.json").write_text(
        json.dumps(public_payload, indent=2, sort_keys=True, default=asdict),
        encoding="utf-8",
    )


def _redact_local_paths(payload: Any, *, run_root: Path) -> Any:
    """Return sidecar-safe data without machine-local absolute path strings."""
    if is_dataclass(payload) and not isinstance(payload, type):
        return _redact_local_paths(asdict(payload), run_root=run_root)
    if isinstance(payload, str):
        candidate = Path(payload)
        if candidate.is_absolute():
            return _redact_one_path(candidate, run_root=run_root)
        return re.sub(
            r"(?<!\S)(/[^\s]+)",
            lambda match: _redact_one_path(Path(match.group(1)), run_root=run_root),
            payload,
        )
    if isinstance(payload, Mapping):
        return {
            key: _redact_local_paths(value, run_root=run_root)
            for key, value in payload.items()
        }
    if isinstance(payload, tuple):
        return tuple(_redact_local_paths(value, run_root=run_root) for value in payload)
    if isinstance(payload, list):
        return [_redact_local_paths(value, run_root=run_root) for value in payload]
    return payload


def _redact_one_path(candidate: Path, *, run_root: Path) -> str:
    try:
        return str(candidate.relative_to(run_root))
    except ValueError:
        return f"<external>/{candidate.name}"
