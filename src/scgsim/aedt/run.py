"""Explicit local execution for one prepared AEDT handoff.

This module solely owns CLI parsing, the AEDT Desktop transaction, receipt
state, family dispatch, release, and failure precedence.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Any

from ._epr_eigenmode import (
    prepare_epr_hfss,
    prepared_epr_result,
    solve_and_export_epr,
)
from ._hfss_runtime import run_hfss
from ._epr_results import seal_saved_solution
from ._native_common import owned_application_constructor, pyaedt_version
from ._q2d_runtime import _export_q2d, run_q2d
from ._q3d_runtime import run_q3d
from ._runtime_provenance import (
    RECEIPT_V1,
    RECEIPT_V2,
    RECEIPT_V3,
    runtime_source_identity,
    validate_runtime_source,
)
from .spec import (
    LOCKED_PYAEDT,
    REQUIRED_AEDT_VERSION,
    AedtSpec,
    HfssEprSpec,
    HfssEprAnalysisSpec,
    Q2dSpec,
    Q3dSpec,
    parse_aedt_spec,
)
from .util import file_sha256, read_json, write_json

# Thin private compatibility aliases; implementation remains in family modules.
_solve = run_hfss
_solve_q3d = run_q3d
_solve_q2d = run_q2d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one prepared SCGSim AEDT handoff")
    parser.add_argument("--handoff", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Open local AEDT and solve the prepared handoff",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create and verify an EPR native model without solving it",
    )
    parser.add_argument(
        "--analyze-epr",
        action="store_true",
        help="Analyze the sealed saved-copy EPR request without solving",
    )
    args = parser.parse_args(argv)
    metadata_path = Path(args.handoff).resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"handoff metadata is missing: {metadata_path}")
    if not args.execute:
        raise RuntimeError(
            "prepared handoff is not executed; use run_aedt.sh or pass --execute explicitly"
        )
    return _execute(
        metadata_path,
        prepare_only=args.prepare_only,
        analyze_epr=args.analyze_epr,
    )


def _execute(
    metadata_path: Path,
    *,
    prepare_only: bool = False,
    analyze_epr: bool = False,
) -> int:
    run_dir = metadata_path.parent.parent
    os.chdir(run_dir)
    metadata = _object(read_json(metadata_path), "handoff metadata")
    files = _canonical_metadata_files(metadata, metadata_path, run_dir)
    receipt_path = run_dir / files["receipt"]
    receipt = _object(read_json(receipt_path), "receipt")
    if receipt.get("schema_version") not in {RECEIPT_V1, RECEIPT_V2, RECEIPT_V3}:
        raise RuntimeError("handoff receipt schema is invalid")
    if receipt.get("status") != "not_run":
        raise RuntimeError("one-shot handoff is not in not_run state")
    spec_path = run_dir / files["spec"]
    spec = parse_aedt_spec(_object(read_json(spec_path), "spec"), base_dir=run_dir)
    if (
        not isinstance(spec, (HfssEprAnalysisSpec, HfssEprSpec, Q2dSpec))
        and spec.gds_path.resolve() != (run_dir / "geometry/design.gds").resolve()
    ):
        raise RuntimeError("prepared spec must use geometry/design.gds")
    valid_flags = (
        (isinstance(spec, HfssEprSpec) and not analyze_epr)
        or (
            isinstance(spec, HfssEprAnalysisSpec)
            and analyze_epr
            and not prepare_only
        )
        or (
            not isinstance(spec, (HfssEprAnalysisSpec, HfssEprSpec))
            and not prepare_only
            and not analyze_epr
        )
    )
    if not valid_flags:
        raise RuntimeError(
            "EPR execution flags do not match the prepared handoff workflow"
        )
    _require_pristine_run(run_dir, spec)
    cohort = _verify_prepared_cohort(run_dir, metadata, receipt, spec)
    started = _utc_now()
    execution_started = time.perf_counter()
    receipt.update(
        {
            "schema_version": RECEIPT_V3,
            "expected_receipt_schema": RECEIPT_V3,
            "preparation_cohort": cohort["preparation_cohort"],
            "prepared_runtime_source": cohort["prepared_runtime_source"],
            "prepared_receipt_sha256": cohort["prepared_receipt_sha256"],
            "verified_prepared_manifest_sha256": cohort[
                "verified_prepared_manifest_sha256"
            ],
            "status": "running",
            "started_at_utc": started,
            "save": {"ok": False},
            "release": {"ok": False},
        }
    )
    write_json(receipt_path, receipt)

    desktop: Any | None = None
    status = "failed"
    failure: str | None = None
    result: dict[str, Any] | None = None
    epr_solver_attempted = False
    try:
        receipt["mode"] = spec.mode
        _verify_prepared_hashes(metadata, spec_path, spec)
        receipt["runtime_source"] = _runtime_source_identity()
        if _pyaedt_version() != LOCKED_PYAEDT:
            raise RuntimeError("PyAEDT lock mismatch")
        from ansys.aedt.core import Desktop, Hfss, Q2d, Q3d

        desktop = Desktop(
            version=REQUIRED_AEDT_VERSION,
            non_graphical=True,
            new_desktop=True,
            close_on_exit=False,
        )
        if desktop.aedt_version_id != REQUIRED_AEDT_VERSION:
            raise RuntimeError(f"AEDT version mismatch: {desktop.aedt_version_id!r}")
        if isinstance(spec, HfssEprAnalysisSpec):
            from ._epr_eigenmode import analyze_saved_epr

            result = analyze_saved_epr(
                owned_application_constructor(Hfss, desktop), run_dir, spec
            )
            status = "epr_analysis_completed"
        elif isinstance(spec, HfssEprSpec):
            prepared = prepare_epr_hfss(
                owned_application_constructor(Hfss, desktop), run_dir, spec
            )
            if prepare_only:
                result = prepared_epr_result(prepared)
                status = "native_preparation_only"
            else:
                epr_solver_attempted = True
                result = solve_and_export_epr(prepared)
                status = "completed"
        elif isinstance(spec, Q3dSpec):
            result = _solve_q3d(
                owned_application_constructor(Q3d, desktop), run_dir, spec
            )
        elif isinstance(spec, Q2dSpec):
            result = _solve_q2d(
                owned_application_constructor(Q2d, desktop), run_dir, spec
            )
        else:
            result = _solve(
                owned_application_constructor(Hfss, desktop), run_dir, spec
            )
        if not isinstance(spec, (HfssEprAnalysisSpec, HfssEprSpec)):
            status = "completed"
    except Exception as exc:  # noqa: BLE001 -- receipt must record any solver failure.
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if result is not None and isinstance(spec, (HfssEprAnalysisSpec, HfssEprSpec)):
            receipt["outputs"] = result["outputs"]
            receipt["connected"] = result["connected"]
            receipt["project"] = result["project"]
            receipt["geometry"] = result["geometry"]
            receipt["setup"] = result["setup"]
            receipt["expressions"] = result.get("expressions", [])
            receipt["cache"] = result.get("cache", {"status": "not_applicable"})
            receipt["timings"] = result.get("timings", {})
            receipt["solver_invoked"] = result["solver_invoked"]
            receipt["workflow_status"] = result["workflow_status"]
            if "convergence" in result:
                receipt["convergence"] = result["convergence"]
            if "result_readback" in result:
                receipt["result_readback"] = result["result_readback"]
            if "result" in result:
                receipt["epr_result"] = result["result"]
            if "saved_field_evidence" in result:
                receipt["saved_field_evidence"] = result["saved_field_evidence"]
        elif result is not None:
            receipt["save"] = result["save"]
        if desktop is not None:
            release_started = time.perf_counter()
            try:
                released = bool(
                    desktop.release_desktop(close_projects=True, close_on_exit=True)
                )
                receipt["release"] = {"ok": released}
                if not released:
                    raise RuntimeError("owned AEDT Desktop release returned false")
            except Exception as exc:  # noqa: BLE001 -- receipt records release failure.
                receipt["release"] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if failure is None:
                    failure = receipt["release"]["error"]
            receipt.setdefault("timings", {})["release_seconds"] = round(
                time.perf_counter() - release_started, 6
            )
        if failure is not None:
            status = "failed"
            receipt["error"] = failure
        if isinstance(spec, HfssEprSpec):
            receipt["solver_invoked"] = epr_solver_attempted
        if result is not None and isinstance(spec, (HfssEprAnalysisSpec, HfssEprSpec)):
            project_relative = result["project"]
            project_path = _contained(run_dir, project_relative)
            if not project_path.is_file():
                status = "failed"
                if failure is None:
                    failure = "EPR project is missing after owned Desktop release"
                    receipt["error"] = failure
            else:
                project_sha256 = file_sha256(project_path)
                result["outputs"] = {project_relative: project_sha256, **{
                    key: value
                    for key, value in result["outputs"].items()
                    if key != project_relative
                }}
                result["save"] = {
                    "ok": True,
                    "project_sha256": project_sha256,
                    "identity_stage": "after_owned_desktop_release",
                }
                receipt["outputs"] = result["outputs"]
                receipt["save"] = result["save"]
                if (
                    isinstance(spec, HfssEprSpec)
                    and spec.epr_request is not None
                    and status == "completed"
                    and failure is None
                    and receipt.get("release") == {"ok": True}
                ):
                    try:
                        sealed = seal_saved_solution(
                            run_dir,
                            project_name=spec.project_name,
                            design_name=spec.design_name,
                            setup_name=spec.run_control.setup_name,
                            model_source_sha256=spec.geometry.model_sha256,
                            evidence=result["saved_field_evidence"],
                        )
                        result["outputs"].update(
                            {
                                sealed["manifest"]: sealed["manifest_sha256"],
                                sealed["receipt"]: sealed["receipt_sha256"],
                            }
                        )
                        receipt["outputs"] = result["outputs"]
                        receipt["saved_solution"] = sealed
                    except Exception as exc:  # noqa: BLE001 -- sealing is required.
                        seal_error = f"{type(exc).__name__}: {exc}"
                        receipt["saved_solution"] = {
                            "status": "unavailable",
                            "error": seal_error,
                        }
                        status = "failed"
                        if failure is None:
                            failure = seal_error
                            receipt["error"] = failure
        elif result is not None:
            receipt["outputs"] = result["outputs"]
            receipt["connected"] = result["connected"]
            receipt["project"] = result["project"]
            receipt["ports"] = result.get("ports", [])
            receipt["nets"] = result.get("nets", [])
            receipt["conductors"] = result.get("conductors", [])
            receipt["mesh"] = result.get("mesh", {})
            receipt["materials"] = result["materials"]
            receipt["region"] = result["region"]
            receipt["result_readback"] = result["result_readback"]
            receipt["setup"] = result.get("setup")
            if "convergence" in result:
                receipt["convergence"] = result["convergence"]
        receipt["diagnostics"] = _read_physics_warnings(run_dir)
        receipt["status"] = status
        receipt["finished_at_utc"] = _utc_now()
        receipt["execution_seconds"] = round(time.perf_counter() - execution_started, 6)
        write_json(receipt_path, receipt)
    return 0 if status in {
        "completed",
        "epr_analysis_completed",
        "native_preparation_only",
    } else 1


def _canonical_metadata_files(
    metadata: dict[str, Any], metadata_path: Path, run_dir: Path
) -> dict[str, str]:
    expected_path = run_dir / "metadata/aedt_handoff_metadata.json"
    if metadata_path != expected_path:
        raise RuntimeError("handoff metadata path is not canonical")
    schema = metadata.get("schema_version")
    if schema not in {"scgsim.aedt.handoff.v1", "scgsim.aedt.handoff.v2"} or metadata.get(
        "status"
    ) != "prepared":
        raise RuntimeError("handoff metadata schema or status is invalid")
    files = _object(metadata.get("files"), "files")
    expected = {"spec": "aedt_spec.json", "receipt": "metadata/aedt_run_receipt.json"}
    workflow = metadata.get("workflow")
    if schema == "scgsim.aedt.handoff.v1" and metadata.get("mode") != "q2d":
        expected["gds"] = "geometry/design.gds"
    elif schema == "scgsim.aedt.handoff.v2" and workflow == "epr_analysis":
        for key in ("saved_project", "saved_results"):
            value = files.get(key)
            if (
                not isinstance(value, str)
                or not value.startswith("saved/")
                or _contained(run_dir, value) != run_dir / value
            ):
                raise RuntimeError("analysis handoff saved paths are not canonical")
            expected[key] = value
    if (schema == "scgsim.aedt.handoff.v2") != (
        workflow in {"body_first_eigenmode", "epr", "epr_analysis"}
    ):
        raise RuntimeError("handoff metadata workflow is inconsistent")
    if files != expected:
        raise RuntimeError("handoff metadata file map is not canonical")
    return expected


def _verify_prepared_hashes(
    metadata: dict[str, Any], spec_path: Path, spec: AedtSpec
) -> None:
    receipt = _object(
        read_json(spec_path.parent / "metadata/aedt_run_receipt.json"), "receipt"
    )
    source = _object(receipt.get("source"), "receipt.source")
    if source.get("spec") != "aedt_spec.json":
        raise RuntimeError("receipt source paths are not canonical")
    if file_sha256(spec_path) != _text(source.get("spec_sha256"), "source.spec_sha256"):
        raise RuntimeError("prepared spec hash mismatch")
    if isinstance(spec, (HfssEprAnalysisSpec, HfssEprSpec)):
        expected_source_keys = {
            "spec",
            "spec_sha256",
            "planar_source_sha256",
            "planar_model_sha256",
        }
        if isinstance(spec, HfssEprAnalysisSpec):
            expected_source_keys.add("saved_solution_sha256")
        if set(source) != expected_source_keys:
            raise RuntimeError("EPR handoff source members are not canonical")
        if source.get("planar_source_sha256") != spec.geometry.source_sha256:
            raise RuntimeError("prepared planar source hash mismatch")
        if source.get("planar_model_sha256") != spec.geometry.model_sha256:
            raise RuntimeError("prepared planar model hash mismatch")
        if "gds_sha256" in metadata:
            raise RuntimeError("EPR handoff must not contain a GDS source")
        if isinstance(spec, HfssEprAnalysisSpec) and source.get(
            "saved_solution_sha256"
        ) != spec.saved_solution.content_sha256:
            raise RuntimeError("prepared saved solution hash mismatch")
        return
    if isinstance(spec, Q2dSpec):
        if set(source) != {"spec", "spec_sha256"} or "gds_sha256" in metadata:
            raise RuntimeError("Q2D handoff must not contain a GDS source")
        return
    gds_path = spec.gds_path
    if source.get("gds") != "geometry/design.gds":
        raise RuntimeError("receipt GDS path is not canonical")
    if file_sha256(gds_path) != _text(metadata.get("gds_sha256"), "gds_sha256"):
        raise RuntimeError("copied GDS hash mismatch")
    if file_sha256(gds_path) != _text(source.get("gds_sha256"), "source.gds_sha256"):
        raise RuntimeError("receipt GDS hash mismatch")


def _verify_prepared_cohort(
    run_dir: Path, metadata: dict[str, Any], receipt: dict[str, Any], spec: AedtSpec
) -> dict[str, Any]:
    manifest_path = run_dir / "metadata/aedt_handoff_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"handoff manifest is missing: {manifest_path}")
    manifest = _object(read_json(manifest_path), "handoff manifest")
    expected_manifest_schema = (
        "scgsim.aedt.handoff-manifest.v2"
        if isinstance(spec, (HfssEprAnalysisSpec, HfssEprSpec))
        else "scgsim.aedt.handoff-manifest.v1"
    )
    if manifest.get("schema_version") != expected_manifest_schema:
        raise RuntimeError("handoff manifest schema is invalid")
    metadata_has = "expected_receipt_schema" in metadata
    manifest_has = "expected_receipt_schema" in manifest
    markers = (
        metadata.get("expected_receipt_schema"),
        manifest.get("expected_receipt_schema"),
    )
    schema = receipt.get("schema_version")
    if receipt.get("mode") != spec.mode:
        raise RuntimeError("prepared receipt mode does not match the spec")
    initial_keys = {
        "schema_version",
        "status",
        "mode",
        "requested",
        "pdk_materials",
        "vacuum_material_id",
        "source",
        "outputs",
        "prepared_at_utc",
    }
    if schema in {RECEIPT_V2, RECEIPT_V3}:
        initial_keys.add("prepared_runtime_source")
        if markers != (schema, schema):
            raise RuntimeError("prepared receipt cohort markers are inconsistent")
        validate_runtime_source(
            receipt.get("prepared_runtime_source"), stage="prepared"
        )
        preparation_cohort = "prepared_v3" if schema == RECEIPT_V3 else "prepared_v2"
        prepared_source = receipt["prepared_runtime_source"]
    elif schema == RECEIPT_V1:
        if metadata_has or manifest_has:
            if RECEIPT_V2 in markers:
                raise RuntimeError(
                    "legacy receipt conflicts with v2 expectation markers"
                )
            raise RuntimeError(
                "legacy receipt conflicts with later expectation markers"
            )
        if "prepared_runtime_source" in receipt or "expected_receipt_schema" in receipt:
            raise RuntimeError("legacy receipt contains mixed preparation provenance")
        preparation_cohort = "verified_legacy_v1"
        prepared_source = {"status": "legacy_v1_not_recorded"}
    else:
        raise RuntimeError("handoff receipt schema is invalid")
    if set(receipt) != initial_keys or receipt.get("outputs") != {}:
        raise RuntimeError("prepared receipt members are not canonical")

    expected_paths = ["run_aedt.sh", "aedt_spec.json"]
    if isinstance(spec, HfssEprAnalysisSpec):
        expected_paths.extend(
            f"saved/{item['path']}" for item in spec.saved_solution.members
        )
    elif not isinstance(spec, (HfssEprSpec, Q2dSpec)):
        expected_paths.append("geometry/design.gds")
    expected_paths += [
        "metadata/aedt_handoff_metadata.json",
        "metadata/aedt_run_receipt.json",
        "metadata/aedt_handoff_manifest.json",
    ]
    if manifest.get("allowed_paths") != expected_paths:
        raise RuntimeError("handoff manifest allowed paths are not canonical")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != len(expected_paths) - 1:
        raise RuntimeError("handoff manifest members are not canonical")
    if [
        item.get("path") if isinstance(item, dict) else None for item in members
    ] != expected_paths[:-1]:
        raise RuntimeError("handoff manifest members are not canonical")
    hashes: dict[str, str] = {}
    for item in members:
        if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
            raise RuntimeError("handoff manifest member is invalid")
        relative = item["path"]
        path = _contained(run_dir, relative)
        if (
            not isinstance(item["bytes"], int)
            or isinstance(item["bytes"], bool)
            or item["bytes"] < 0
            or not isinstance(item["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
            or not path.is_file()
            or path.stat().st_size != item["bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise RuntimeError(f"handoff manifest member mismatch: {relative}")
        hashes[relative] = item["sha256"]
    return {
        "preparation_cohort": preparation_cohort,
        "prepared_runtime_source": prepared_source,
        "prepared_receipt_sha256": hashes["metadata/aedt_run_receipt.json"],
        "verified_prepared_manifest_sha256": file_sha256(manifest_path),
    }


def _contained(root: Path, relative: str) -> Path:
    requested = Path(relative)
    if requested.is_absolute():
        raise RuntimeError(f"handoff path escapes run directory: {relative!r}")
    path = (root / requested).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError(f"handoff path escapes run directory: {relative!r}")
    return path


def _require_pristine_run(run_dir: Path, spec: AedtSpec) -> None:
    if isinstance(spec, HfssEprAnalysisSpec):
        if (run_dir / "analysis-work").exists():
            raise RuntimeError("saved-field analysis workcopy already exists")
        return
    if (run_dir / f"{spec.project_name}.aedt").exists() or (
        run_dir / "results"
    ).exists():
        raise RuntimeError("one-shot handoff already owns project or results artifacts")


def _pyaedt_version() -> str:
    return pyaedt_version()


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _read_physics_warnings(run_dir: Path) -> dict[str, Any]:
    """Preserve HFSS terminal-mode diagnostics without treating them as gates."""
    path = run_dir / "batch.log"
    if not path.is_file():
        return {"batch_log": "batch.log", "present": False, "physics_warnings": []}
    warnings: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        conductor = re.search(
            r"Port '([^']+)' has (\d+) signal conductors with (\d+) terminals", raw_line
        )
        slow_mode = re.search(
            r"Port ([^ ]+) supports an additional propagating and/or slowly decaying mode",
            raw_line,
        )
        if conductor:
            port, signals, terminals = conductor.groups()
            detail = {
                "kind": "signal_conductors_per_terminal",
                "port": port,
                "signal_conductors": int(signals),
                "terminals": int(terminals),
            }
        elif slow_mode:
            detail = {
                "kind": "additional_propagating_or_slow_mode",
                "port": slow_mode.group(1),
            }
        else:
            continue
        key = (detail["kind"], detail["port"])
        record = warnings.setdefault(key, {**detail, "occurrences": 0})
        record["occurrences"] += 1
    return {
        "batch_log": "batch.log",
        "sha256": file_sha256(path),
        "present": True,
        "physics_warnings": [warnings[key] for key in sorted(warnings)],
    }


def _runtime_source_identity() -> dict[str, Any]:
    return runtime_source_identity()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
