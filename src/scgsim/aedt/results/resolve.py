"""Offline strict AEDT result and receipt resolution."""

from __future__ import annotations


import csv

import math

import re

from dataclasses import dataclass
from decimal import Decimal

from pathlib import Path

from typing import Any, Literal

from scgsim.aedt._io import file_sha256, read_json

from scgsim.aedt.epr.models import EprResult

from scgsim.aedt.preparation.cohort import (
    canonical_handoff_paths,
    canonical_member_paths,
    validate_geometry_source,
    validate_hfss_import_source,
)

from scgsim.aedt.results.benchmark import read_simulation_benchmark

from scgsim.aedt.results.convergence.hfss import read_hfss_convergence

from scgsim.aedt.results.convergence.q2d import read_q2d_convergence

from scgsim.aedt.results.convergence.q3d import read_q3d_convergence

from scgsim.aedt.results.matrices import parse_matrix_export, read_q2d_rlgc_matrix

from scgsim.aedt.results.provenance import (
    RECEIPT_V1,
    RECEIPT_V2,
    RECEIPT_V3,
    initial_receipt_payload,
    initial_receipt_sha256,
    validate_runtime_source,
)

from scgsim.aedt.runtime.benchmark import BENCHMARK_RELATIVE

from scgsim.aedt.specs.common import ModalPort

from scgsim.aedt.specs.hfss import HfssDrivenGeometrySpec, HfssDrivenSpec, HfssEigenmodeSpec, HfssEprSpec

from scgsim.aedt.specs.parse import parse_aedt_spec

from scgsim.aedt.specs.q2d import Q2dSpec

from scgsim.aedt.specs.q3d import Q3dSpec


_Q3D_REGION_DIRECTIONS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")

_Q3D_REGION_SHEET_NAMES = {
    "+X": ("SCGSimRegionGroundPX", "SCGSimRegionGroundPXThinConductor"),
    "-X": ("SCGSimRegionGroundNX", "SCGSimRegionGroundNXThinConductor"),
    "+Y": ("SCGSimRegionGroundPY", "SCGSimRegionGroundPYThinConductor"),
    "-Y": ("SCGSimRegionGroundNY", "SCGSimRegionGroundNYThinConductor"),
    "+Z": ("SCGSimRegionGroundPZ", "SCGSimRegionGroundPZThinConductor"),
    "-Z": ("SCGSimRegionGroundNZ", "SCGSimRegionGroundNZThinConductor"),
}


@dataclass(frozen=True)
class ResolvedRun:
    """Canonical result paths verified against the completed run receipt."""

    mode: Literal["terminal", "modal", "eigenmode", "q3d", "q2d"]
    project_path: Path
    primary_csv: Path
    touchstone_path: Path | None
    provenance_path: Path | None
    receipt_path: Path
    convergence: dict[str, Any] | None = None
    benchmark_path: Path | None = None
    _output_hashes: tuple[tuple[str, str], ...] = ()
    _spec_sha256: str | None = None
    _execution_seconds: float | None = None
    _setup_name: str | None = None
    _epr_requested: bool = False
    _epr_result: EprResult | None = None
    _lumped_boundaries: tuple[dict[str, Any], ...] = ()

    def _verified_output(self, path: Path) -> Path:
        root = self.receipt_path.parent.parent
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError("resolved output escapes the run directory") from exc
        return _verified(root, relative, dict(self._output_hashes))

    def _verified_spec(self) -> Path:
        root = self.receipt_path.parent.parent
        return _verified(
            root, "aedt_spec.json", {"spec_sha256": self._spec_sha256}, "spec_sha256"
        )

    def epr_result(self) -> EprResult | None:
        """Return the immutable, receipt-bound adaptive EPR record, if requested."""

        if not self._epr_requested:
            return None
        if self._epr_result is None:
            raise RuntimeError("requested EPR result was not resolved")
        return self._epr_result

    def show_all_results(
        self, *, show_details: bool = False, theme: str = "light"
    ) -> None:
        """Display verified primary, benchmark, and requested EPR evidence."""

        if type(show_details) is not bool:
            raise TypeError("show_details must be a bool")
        from scgsim.presentation.notebook import checked_theme

        checked_theme(theme)
        from scgsim.aedt.presentation.runs import display_resolved_run

        display_resolved_run(self, show_details=show_details, theme=theme)

    def physics_results(self) -> tuple[dict[str, str], ...]:
        """Return the verified primary result as string-valued rows."""

        self._verified_output(self.primary_csv)

        if self.mode == "q2d":
            root = self.receipt_path.parent.parent
            spec = parse_aedt_spec(read_json(self._verified_spec()), base_dir=root, allow_historical_modeling=True)
            if not isinstance(spec, Q2dSpec):
                raise RuntimeError("resolved Q2D result has a non-Q2D spec")
            rows, _ = read_q2d_rlgc_matrix(self.primary_csv, spec)
            return tuple(
                {key: str(value) for key, value in row.items()} for row in rows
            )

        if self.mode == "q3d":
            root = self.receipt_path.parent.parent
            spec = parse_aedt_spec(read_json(self._verified_spec()), base_dir=root, allow_historical_modeling=True)
            if not isinstance(spec, Q3dSpec):
                raise RuntimeError("resolved Q3D result has a non-Q3D spec")
            if not spec.solve_ac_rl:
                rows, _ = parse_matrix_export(
                    self.primary_csv,
                    "Q3D",
                    "C",
                    spec.run_control.frequency_ghz,
                    {"Capacitance Matrix": "C", "Conductance Matrix": "G"},
                )
                return tuple(
                    {key: str(value) for key, value in row.items()} for row in rows
                )

        with self.primary_csv.open(newline="", encoding="utf-8-sig") as stream:
            return tuple(dict(row) for row in csv.DictReader(stream))

    def simulation_benchmark(self) -> dict[str, Any]:
        """Return the receipt-bound execution summary for notebook display."""

        self._verified_output(self.project_path)
        self._verified_output(self.primary_csv)
        if self._execution_seconds is None:
            raise RuntimeError("resolved execution duration is unavailable")
        return {
            "mode": self.mode,
            "execution_seconds": self._execution_seconds,
            "project_bytes": self.project_path.stat().st_size,
            "primary_csv_bytes": self.primary_csv.stat().st_size,
            "benchmark": (
                {"status": "not_recorded"}
                if self.benchmark_path is None
                else read_simulation_benchmark(
                    self._verified_output(self.benchmark_path),
                    self._setup_name,
                )
            ),
        }

    def show_simulation_benchmark(
        self, *, show_details: bool = False, theme: str = "light"
    ) -> AedtBenchmarkReport:
        """Return a notebook-displayable, offline view of native profile evidence."""
        from scgsim.presentation.notebook import checked_theme

        checked_theme(theme)
        return AedtBenchmarkReport(self.simulation_benchmark(), show_details, theme)


@dataclass(frozen=True)
class AedtBenchmarkReport:
    """Offline native benchmark view; ``data`` retains the complete profile tree."""

    data: dict[str, Any]
    show_details: bool = False
    theme: str = "light"

    def _ipython_display_(self) -> None:
        from scgsim.aedt.presentation.benchmark import display_benchmark

        display_benchmark(self.data, show_details=self.show_details, theme=self.theme)


def _captured_fields(
    receipt: dict[str, Any],
    outputs: dict[str, Any],
    spec: Any,
    *,
    epr_result: EprResult | None = None,
) -> dict[str, Any]:
    """Keep the identities needed by later report reads without replaying a run."""

    return {
        "_output_hashes": tuple(sorted(outputs.items())),
        "_spec_sha256": receipt["source"]["spec_sha256"],
        "_execution_seconds": receipt.get("execution_seconds"),
        "_setup_name": spec.run_control.setup_name,
        "_epr_requested": isinstance(spec, HfssEprSpec)
        and spec.epr_request is not None,
        "_epr_result": epr_result,
        "_lumped_boundaries": tuple(receipt.get("geometry", {}).get("lumped_rlcs", ()))
        + tuple(item for item in receipt.get("ports", ()) if item.get("kind") == "lumped"),
    }


def _resolved_adaptive_epr(
    path: Path, spec: HfssEprSpec, convergence: dict[str, Any]
) -> EprResult:
    from scgsim.aedt.results.epr import resolve_epr_result

    result = resolve_epr_result(path)
    requested_modes = (
        tuple(range(1, spec.run_control.num_modes + 1))
        if spec.epr_request is None or spec.epr_request.mode_indices is None
        else spec.epr_request.mode_indices
    )
    if (
        result.result_kind != "adaptive_history"
        or result.setup_name != spec.run_control.setup_name
        or result.provenance.get("model_source_sha256") != spec.geometry.model_sha256
        or result.provenance.get("analysis_source_sha256")
        != spec.geometry.source_sha256
        or tuple(result.provenance.get("requested_modes", ())) != requested_modes
        or result.provenance.get("solver_last_completed_pass")
        != convergence["final_pass"]
    ):
        raise RuntimeError(
            "adaptive EPR result differs from the completed source and setup"
        )
    if any(
        row["mode"] not in requested_modes
        or row["native_pass"] > convergence["final_pass"]
        for row in result.rows
    ):
        raise RuntimeError(
            "adaptive EPR rows exceed the requested modes or observed passes"
        )
    return result


def resolve_results(run_dir: str | Path) -> ResolvedRun:
    """Return results only when a complete exact receipt proves every artifact."""
    root = Path(run_dir).resolve()
    receipt_path = _contained(root, "metadata/aedt_run_receipt.json")
    if not receipt_path.is_file():
        raise FileNotFoundError(f"run receipt is missing: {receipt_path}")
    receipt = read_json(receipt_path)
    if not isinstance(receipt, dict) or receipt.get("schema_version") not in {
        RECEIPT_V1,
        RECEIPT_V2,
        RECEIPT_V3,
    }:
        raise RuntimeError("handoff receipt schema is invalid")
    if receipt.get("status") != "completed":
        raise RuntimeError("handoff has not completed successfully")
    receipt_schema = receipt["schema_version"]
    if receipt_schema == RECEIPT_V1:
        _reject_legacy_v2_markers(root, receipt)
    save = receipt.get("save")
    if (
        not isinstance(save, dict)
        or save.get("ok") is not True
        or not isinstance(save.get("project_sha256"), str)
    ):
        raise RuntimeError("completed receipt lacks a successful exact project save")
    if receipt.get("release") != {"ok": True}:
        raise RuntimeError("completed receipt lacks a successful owned Desktop release")
    source = receipt.get("source")
    mode = receipt.get("mode")
    if not isinstance(source, dict) or source.get("spec") != "aedt_spec.json":
        raise RuntimeError("receipt source paths are not canonical")
    spec_path = _verified(root, "aedt_spec.json", source, "spec_sha256")
    spec = parse_aedt_spec(read_json(spec_path), base_dir=root, allow_historical_modeling=True)
    if receipt_schema in {RECEIPT_V2, RECEIPT_V3}:
        _validate_completion_cohort(root, receipt, spec)
    if isinstance(spec, (HfssEprSpec, HfssDrivenGeometrySpec)):
        expected_source = {
            "spec",
            "spec_sha256",
            "planar_source_sha256",
            "planar_model_sha256",
        }
        if set(source) != expected_source:
            raise RuntimeError("body-first Eigenmode source members are not canonical")
        if (
            source["planar_source_sha256"] != spec.geometry.source_sha256
            or source["planar_model_sha256"] != spec.geometry.model_sha256
        ):
            raise RuntimeError("body-first Eigenmode source identity differs")
    elif isinstance(spec, (Q2dSpec, Q3dSpec)):
        if set(source) != {"spec", "spec_sha256"}:
            raise RuntimeError(
                "Q2D receipt must not contain a GDS source"
                if isinstance(spec, Q2dSpec)
                else "Q3D body receipt must not contain an imported GDS source"
            )
        if isinstance(spec, Q3dSpec):
            validate_geometry_source(root, source, spec)
    else:
        if source.get("gds") != "geometry/design.gds":
            raise RuntimeError("receipt GDS path is not canonical")
        _verified(root, "geometry/design.gds", source, "gds_sha256")
    if (
        mode not in {"terminal", "modal", "eigenmode", "q3d", "q2d"}
        or spec.mode != mode
    ):
        raise RuntimeError("AEDT receipt mode is invalid")
    if isinstance(spec, HfssEprSpec):
        _validate_lumped_readback(receipt, spec.geometry)
        if spec.lumped_rlcs:
            _validate_eigenmode_readback(receipt, spec)
        if receipt.get("workflow_status") != "completed":
            raise RuntimeError("body-first Eigenmode receipt is not a completed solve")
        if not isinstance(receipt.get("geometry"), dict) or not isinstance(
            receipt.get("setup"), dict
        ):
            raise RuntimeError("body-first Eigenmode native readback is incomplete")
    else:
        _validate_readback(root, receipt, spec)
    if receipt_schema == RECEIPT_V3 and isinstance(
        spec, (HfssDrivenSpec, HfssEigenmodeSpec)
    ):
        _validate_hfss_pec_bindings(receipt, spec)
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise TypeError("completed receipt has no output hash manifest")
    benchmark_path = _resolved_benchmark(
        root, receipt, outputs, spec.run_control.setup_name
    )
    project_relative = f"{spec.project_name}.aedt"
    if receipt.get("project") != project_relative:
        raise RuntimeError("completed receipt project path is not canonical")
    project = _verified(root, project_relative, outputs, project_relative)
    if save["project_sha256"] != outputs.get(project_relative):
        raise RuntimeError("saved project hash does not match output manifest")
    if mode == "q2d":
        superseded = (
            "results/q2d/cg_matrix.csv",
            "results/q2d/rl_matrix.csv",
            "results/q2d/matrices.csv",
        )
        if any(_contained(root, relative).exists() for relative in superseded):
            raise RuntimeError("Q2D run contains superseded matrix artifacts")
        expected = {
            project_relative,
            "results/q2d/rlgc_matrix.csv",
        }
        if benchmark_path is not None:
            expected.add(BENCHMARK_RELATIVE)
        if set(outputs) != expected:
            raise RuntimeError("Q2D output manifest is not canonical")
        return ResolvedRun(
            "q2d",
            project,
            _verified(root, "results/q2d/rlgc_matrix.csv", outputs),
            None,
            None,
            receipt_path,
            convergence=receipt["convergence"],
            benchmark_path=benchmark_path,
            **_captured_fields(receipt, outputs, spec),
        )
    if mode == "q3d":
        expected = {project_relative, "results/q3d/c_matrix.csv"}
        if benchmark_path is not None:
            expected.add(BENCHMARK_RELATIVE)
        primary = "results/q3d/c_matrix.csv"
        if spec.solve_ac_rl:
            expected.update(
                {"results/q3d/ac_rl_matrix.csv", "results/q3d/matrices.csv"}
            )
            primary = "results/q3d/matrices.csv"
        elif any(
            _contained(root, relative).exists()
            for relative in ("results/q3d/ac_rl_matrix.csv", "results/q3d/matrices.csv")
        ):
            raise RuntimeError("Q3D capacitance-only run contains AC-RL artifacts")
        if set(outputs) != expected:
            raise RuntimeError("Q3D output manifest is not canonical")
        return ResolvedRun(
            "q3d",
            project,
            _verified(root, primary, outputs),
            None,
            None,
            receipt_path,
            convergence=receipt["convergence"],
            benchmark_path=benchmark_path,
            **_captured_fields(receipt, outputs, spec),
        )
    if mode == "eigenmode":
        if isinstance(spec, HfssEprSpec):
            saved_manifest = "saved-solution-manifest.json"
            saved_receipt = "metadata/saved-solution-receipt.json"
            expected = {
                project_relative,
                "results/epr/eigenmodes.csv",
                "results/epr/eigenmodes.eig",
                "results/epr/adaptive-convergence.prop",
                "results/epr/adaptive-mode-history.json",
            }
            if benchmark_path is not None:
                expected.add(BENCHMARK_RELATIVE)
            if spec.epr_request is not None:
                expected.add("results/epr/adaptive-epr-result.json")
                saved_summary = receipt.get("saved_solution")
                if (
                    isinstance(saved_summary, dict)
                    and saved_summary.get("status") == "complete"
                ):
                    from scgsim.aedt.results.epr import resolve_saved_solution

                    saved = resolve_saved_solution(root / saved_manifest)
                    expected_saved_summary = {
                        "status": "complete",
                        "manifest": saved_manifest,
                        "manifest_sha256": file_sha256(root / saved_manifest),
                        "receipt": saved_receipt,
                        "receipt_sha256": file_sha256(root / saved_receipt),
                        "content_sha256": saved.content_sha256,
                        "identity": dict(saved.identity),
                        "member_count": len(saved.members),
                    }
                    if saved_summary != expected_saved_summary:
                        raise RuntimeError(
                            "completed EPR saved-solution binding is invalid"
                        )
                    expected.update({saved_manifest, saved_receipt})
                elif not (
                    isinstance(saved_summary, dict)
                    and saved_summary.get("status") == "unavailable"
                    and isinstance(saved_summary.get("error"), str)
                    and saved_summary["error"]
                ):
                    raise RuntimeError(
                        "completed EPR run lacks an explicit saved-solution status"
                    )
            if set(outputs) != expected:
                raise RuntimeError(
                    "body-first Eigenmode output manifest is not canonical"
                )
            _verified(root, "results/epr/adaptive-convergence.prop", outputs)
            _verified(root, "results/epr/adaptive-mode-history.json", outputs)
            if receipt.get("convergence") != read_hfss_convergence(
                root,
                spec,
                historical_profile=_eigenmode_convergence_source(receipt) == "profile",
            ):
                raise RuntimeError(
                    "body-first Eigenmode convergence evidence is invalid"
                )
            adaptive_result = (
                None
                if spec.epr_request is None
                else _resolved_adaptive_epr(
                    _verified(root, "results/epr/adaptive-epr-result.json", outputs),
                    spec,
                    receipt["convergence"],
                )
            )
            return ResolvedRun(
                "eigenmode",
                project,
                _verified(root, "results/epr/eigenmodes.csv", outputs),
                None,
                _verified(root, "results/epr/eigenmodes.eig", outputs),
                receipt_path,
                convergence=receipt["convergence"],
                benchmark_path=benchmark_path,
                **_captured_fields(receipt, outputs, spec, epr_result=adaptive_result),
            )
        expected = {
            project_relative,
            "results/eigenmode/eigenmodes.csv",
            "results/eigenmode/eigenmodes.eig",
        }
        if benchmark_path is not None:
            expected.add(BENCHMARK_RELATIVE)
        if _eigenmode_convergence_source(receipt) == "export_convergence":
            expected.add("results/eigenmode/adaptive-convergence.prop")
        if set(outputs) != expected:
            raise RuntimeError("Eigenmode output manifest is not canonical")
        if "results/eigenmode/adaptive-convergence.prop" in expected:
            _verified(root, "results/eigenmode/adaptive-convergence.prop", outputs)
        return ResolvedRun(
            "eigenmode",
            project,
            _verified(root, "results/eigenmode/eigenmodes.csv", outputs),
            None,
            _verified(root, "results/eigenmode/eigenmodes.eig", outputs),
            receipt_path,
            convergence=receipt["convergence"],
            benchmark_path=benchmark_path,
            **_captured_fields(receipt, outputs, spec),
        )
    result_stem = "terminal_st" if mode == "terminal" else "modal_s"
    expected = {
        project_relative,
        f"results/{mode}/{result_stem}.csv",
        f"results/{mode}/{mode}.s2p",
    }
    if benchmark_path is not None:
        expected.add(BENCHMARK_RELATIVE)
    if set(outputs) != expected:
        raise RuntimeError("terminal output manifest is not canonical")
    return ResolvedRun(
        mode,
        project,
        _verified(root, f"results/{mode}/{result_stem}.csv", outputs),
        _verified(root, f"results/{mode}/{mode}.s2p", outputs),
        None,
        receipt_path,
        convergence=receipt["convergence"],
        benchmark_path=benchmark_path,
        **_captured_fields(receipt, outputs, spec),
    )


def _resolved_benchmark(
    root: Path, receipt: dict[str, Any], outputs: dict[str, Any], setup_name: str
) -> Path | None:
    marker = receipt.get("benchmark")
    if marker is None:
        if BENCHMARK_RELATIVE in outputs:
            raise RuntimeError("benchmark output lacks a receipt binding")
        return None
    if (
        not isinstance(marker, dict)
        or set(marker) != {"status", "path"}
        or marker["path"] != BENCHMARK_RELATIVE
    ):
        raise RuntimeError("benchmark receipt binding is invalid")
    path = _verified(root, BENCHMARK_RELATIVE, outputs)
    payload = read_simulation_benchmark(path, setup_name)
    if marker["status"] != payload["status"]:
        raise RuntimeError("benchmark receipt status differs from its artifact")
    return path


def _contained(root: Path, relative: str) -> Path:
    requested = Path(relative)
    if requested.is_absolute():
        raise RuntimeError(f"receipt path escapes handoff root: {relative!r}")
    resolved = (root / requested).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"receipt path escapes handoff root: {relative!r}")
    return resolved


def _reject_legacy_v2_markers(root: Path, receipt: dict[str, Any]) -> None:
    """Reject partial receipt downgrades without burdening historical v1 runs."""
    if "expected_receipt_schema" in receipt or "prepared_runtime_source" in receipt:
        raise RuntimeError("legacy receipt conflicts with v2 expectation markers")
    for relative in (
        "metadata/aedt_handoff_metadata.json",
        "metadata/aedt_handoff_manifest.json",
    ):
        path = _contained(root, relative)
        if not path.is_file():
            continue
        try:
            candidate = read_json(path)
        except (OSError, TypeError, ValueError):
            # Historical v1 resolution never required either preparation file.
            continue
        if isinstance(candidate, dict) and "expected_receipt_schema" in candidate:
            raise RuntimeError("legacy receipt conflicts with v2 expectation markers")


def _validate_completion_cohort(root: Path, receipt: dict[str, Any], spec: Any) -> None:
    """Verify the immutable preparation cohort and complete execution provenance."""
    receipt_schema = receipt.get("schema_version")
    if receipt_schema not in {RECEIPT_V2, RECEIPT_V3}:
        raise RuntimeError("completed receipt cohort schema is invalid")
    version = receipt_schema.rsplit(".", 1)[-1]
    if receipt.get("expected_receipt_schema") != receipt_schema:
        raise RuntimeError(f"completed {version} receipt expectation marker is invalid")
    mode = receipt.get("mode")
    if mode not in {"terminal", "modal", "eigenmode", "q3d", "q2d"}:
        raise RuntimeError("AEDT receipt mode is invalid")

    metadata_path = _contained(root, "metadata/aedt_handoff_metadata.json")
    manifest_path = _contained(root, "metadata/aedt_handoff_manifest.json")
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"completed {version} receipt lacks preparation cohort files"
        )
    metadata = read_json(metadata_path)
    manifest = read_json(manifest_path)
    epr = isinstance(spec, (HfssEprSpec, HfssDrivenGeometrySpec))
    expected_manifest_schema = (
        "scgsim.aedt.handoff-manifest.v2" if epr else "scgsim.aedt.handoff-manifest.v1"
    )
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version")
        != ("scgsim.aedt.handoff.v2" if epr else "scgsim.aedt.handoff.v1")
        or metadata.get("status") != "prepared"
        or metadata.get("mode") != mode
        or not isinstance(manifest, dict)
        or manifest.get("schema_version") != expected_manifest_schema
    ):
        raise RuntimeError(f"completed {version} preparation cohort is invalid")

    expected_files = {
        "spec": "aedt_spec.json",
        "receipt": "metadata/aedt_run_receipt.json",
    }
    if mode not in {"q2d", "q3d"} and not epr:
        expected_files["gds"] = "geometry/design.gds"
    if isinstance(spec, HfssDrivenGeometrySpec) and metadata.get("workflow") != "body_first_driven_terminal":
        raise RuntimeError("body-first Driven Terminal preparation workflow is invalid")
    if isinstance(spec, HfssEprSpec) and metadata.get("workflow") not in {"body_first_eigenmode", "epr"}:
        raise RuntimeError("body-first Eigenmode preparation workflow is invalid")
    if metadata.get("files") != expected_files:
        raise RuntimeError(f"completed {version} preparation file map is not canonical")

    metadata_has = "expected_receipt_schema" in metadata
    manifest_has = "expected_receipt_schema" in manifest
    preparation_cohort = receipt.get("preparation_cohort")
    prepared_source = receipt.get("prepared_runtime_source")
    if preparation_cohort == "prepared_v3" and receipt_schema == RECEIPT_V3:
        if (
            metadata.get("expected_receipt_schema") != RECEIPT_V3
            or manifest.get("expected_receipt_schema") != RECEIPT_V3
        ):
            raise RuntimeError("completed v3 preparation markers are inconsistent")
        validate_runtime_source(prepared_source, stage="prepared")
    elif preparation_cohort == "prepared_v2":
        if (
            metadata.get("expected_receipt_schema") != RECEIPT_V2
            or manifest.get("expected_receipt_schema") != RECEIPT_V2
        ):
            raise RuntimeError("completed v2 preparation markers are inconsistent")
        validate_runtime_source(prepared_source, stage="prepared")
    elif preparation_cohort == "verified_legacy_v1":
        if metadata_has or manifest_has:
            raise RuntimeError(
                f"verified legacy cohort contains {version} preparation markers"
            )
        if prepared_source != {"status": "legacy_v1_not_recorded"}:
            raise RuntimeError("verified legacy preparation provenance is invalid")
    else:
        raise RuntimeError(f"completed {version} preparation cohort is invalid")

    validate_runtime_source(receipt.get("runtime_source"), stage="actual")
    prepared_receipt_sha = receipt.get("prepared_receipt_sha256")
    verified_manifest_sha = receipt.get("verified_prepared_manifest_sha256")
    if (
        not isinstance(prepared_receipt_sha, str)
        or not _sha256_text(prepared_receipt_sha)
        or not isinstance(verified_manifest_sha, str)
        or not _sha256_text(verified_manifest_sha)
        or file_sha256(manifest_path) != verified_manifest_sha
        or _initial_receipt_sha256(receipt, preparation_cohort) != prepared_receipt_sha
    ):
        raise RuntimeError(f"completed {version} preparation hash bindings are invalid")

    expected_paths = canonical_handoff_paths(spec)
    if manifest.get("allowed_paths") != expected_paths:
        raise RuntimeError(
            f"completed {version} manifest allowed paths are not canonical"
        )
    members = manifest.get("members")
    if (
        not isinstance(members, list)
        or len(members) != len(expected_paths) - 1
        or [item.get("path") if isinstance(item, dict) else None for item in members]
        != canonical_member_paths(spec)
    ):
        raise RuntimeError(f"completed {version} manifest members are not canonical")
    for member in members:
        if (
            not isinstance(member, dict)
            or set(member) != {"path", "bytes", "sha256"}
            or not isinstance(member.get("bytes"), int)
            or isinstance(member.get("bytes"), bool)
            or member["bytes"] < 0
            or not isinstance(member.get("sha256"), str)
            or not _sha256_text(member["sha256"])
        ):
            raise RuntimeError(f"completed {version} manifest member is invalid")
        relative = member["path"]
        if relative == "metadata/aedt_run_receipt.json":
            if member["sha256"] != prepared_receipt_sha:
                raise RuntimeError(
                    f"completed {version} initial receipt binding is invalid"
                )
            continue
        path = _contained(root, relative)
        if (
            not path.is_file()
            or path.stat().st_size != member["bytes"]
            or file_sha256(path) != member["sha256"]
        ):
            raise RuntimeError(
                f"completed {version} manifest member mismatch: {relative}"
            )

    validate_hfss_import_source(root, read_json(root / "aedt_spec.json"), spec)


def _sha256_text(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _initial_receipt_sha256(receipt: dict[str, Any], preparation_cohort: Any) -> str:
    schema_by_cohort = {
        "prepared_v3": RECEIPT_V3,
        "prepared_v2": RECEIPT_V2,
        "verified_legacy_v1": RECEIPT_V1,
    }
    initial = initial_receipt_payload(
        schema_version=schema_by_cohort.get(preparation_cohort, ""),
        mode=receipt.get("mode"),
        requested=receipt.get("requested"),
        pdk_materials=receipt.get("pdk_materials"),
        vacuum_material_id=receipt.get("vacuum_material_id"),
        source=receipt.get("source"),
        prepared_runtime_source_value=(
            receipt.get("prepared_runtime_source")
            if preparation_cohort in {"prepared_v2", "prepared_v3"}
            else None
        ),
        outputs={},
        prepared_at_utc=receipt.get("prepared_at_utc"),
    )
    return initial_receipt_sha256(initial)


def _validate_readback(root: Path, receipt: dict[str, Any], spec: Any) -> None:
    connected = receipt.get("connected")
    if not isinstance(connected, dict) or connected != {
        "aedt_version": spec.aedt_version,
        "pyaedt_version": spec.pyaedt_version,
    }:
        raise RuntimeError("completed receipt has invalid connected version identity")
    readback = receipt.get("result_readback")
    if isinstance(spec, Q2dSpec):
        _validate_q2d_readback(root, receipt, spec)
        _validate_diagnostics(root, receipt.get("diagnostics"))
        return
    if isinstance(spec, Q3dSpec):
        _validate_q3d_readback(root, receipt, spec)
        _validate_diagnostics(root, receipt.get("diagnostics"))
        return
    if isinstance(spec, HfssEigenmodeSpec):
        _validate_eigenmode_readback(receipt, spec)
        _validate_hfss_setup_and_convergence(root, receipt, spec)
        _validate_diagnostics(root, receipt.get("diagnostics"))
        return
    key = "terminal_st" if spec.mode == "terminal" else "modal_s"
    if not isinstance(readback, dict) or not isinstance(readback.get(key), dict):
        raise TypeError("completed receipt has no canonical result readback")
    frequency = readback[key]
    if frequency != {
        "records": int(spec.run_control.sweep.points),
        "frequency_unit": "GHz",
        "first_frequency_ghz": spec.run_control.sweep.start_ghz,
        "last_frequency_ghz": spec.run_control.sweep.stop_ghz,
        "strictly_increasing": True,
    }:
        raise RuntimeError("completed receipt result frequency readback is invalid")
    touchstone = readback.get("touchstone")
    ports = receipt.get("ports")
    port_key = "terminal_excitation" if spec.mode == "terminal" else "modal_excitation"
    port_order = (
        [record.get(port_key) for record in ports]
        if isinstance(ports, list)
        and len(ports) == 2
        and all(isinstance(record, dict) for record in ports)
        and [record.get("index") for record in ports] == [1, 2]
        else None
    )
    if (
        not isinstance(touchstone, dict)
        or touchstone.get("path") != f"{spec.mode}.s2p"
        or touchstone.get("ports") != 2
        or touchstone.get("records") != int(spec.run_control.sweep.points)
        or touchstone.get("frequency_unit") != "GHz"
        or touchstone.get("first_frequency_ghz") != spec.run_control.sweep.start_ghz
        or touchstone.get("last_frequency_ghz") != spec.run_control.sweep.stop_ghz
        or touchstone.get("strictly_increasing") is not True
        or touchstone.get("port_order") != port_order
        or not all(isinstance(name, str) and name for name in port_order or [])
        or len(set(port_order or [])) != 2
        or not isinstance(touchstone.get("bytes"), int)
        or touchstone["bytes"] <= 0
    ):
        raise RuntimeError(
            "completed receipt is missing a two-port Touchstone readback"
        )
    if spec.mode == "terminal":
        if isinstance(spec, HfssDrivenGeometrySpec):
            _validate_body_terminal_readback(receipt, spec)
        else:
            _validate_terminal_native_evidence(ports, spec)
    else:
        _validate_modal_native_evidence(ports, spec)
    _validate_hfss_setup_and_convergence(root, receipt, spec)
    _validate_diagnostics(root, receipt.get("diagnostics"))


def _validate_hfss_pec_bindings(
    receipt: dict[str, Any], spec: HfssDrivenSpec | HfssEigenmodeSpec
) -> None:
    """Verify V3 PEC evidence against source material and object bindings."""
    records = receipt.get("materials")
    if not isinstance(records, list) or len(records) != len(spec.object_bindings):
        raise RuntimeError("HFSS V3 material binding evidence is invalid")
    materials = dict(spec.materials)
    layers = {layer.layer: layer for layer in spec.layer_imports}
    effective_layers = (
        {}
        if spec.modeling is None
        else {
            (record["layer"], record["datatype"]): record
            for record in spec.effective_layer_imports
        }
    )
    pec_records: list[dict[str, Any]] = []
    sheet_records: list[dict[str, Any]] = []
    for record, object_binding in zip(records, spec.object_bindings, strict=True):
        material = materials[object_binding.material_id]
        layer = layers[object_binding.layer]
        if effective_layers:
            candidates = [
                item
                for item in spec.layer_imports
                if object_binding.object_name.startswith(f"{item.layer_name}_")
            ]
            if len(candidates) != 1:
                raise RuntimeError("HFSS current source destination is ambiguous")
            layer = candidates[0]
            effective = effective_layers[(layer.layer, layer.datatype)]
            expected_fields = {
                "datatype": layer.datatype,
                "native_import_layer": effective["native_import_layer"],
                "physical_layer_id": effective["physical_layer_id"],
                "representation": effective["representation"],
                "source_z_min_um": effective["z_min_um"],
                "source_z_max_um": effective["z_max_um"],
                "physical_thickness_um": effective["physical_thickness_um"],
                "effective_z_min_um": effective["effective_z_min_um"],
                "effective_z_max_um": effective["effective_z_max_um"],
                "native_object_type": "Sheet"
                if effective["representation"] == "sheet"
                else "Solid",
            }
            if not isinstance(record, dict) or any(
                record.get(key) != value for key, value in expected_fields.items()
            ):
                raise RuntimeError(
                    "HFSS effective material geometry differs from source mapping"
                )
        if (
            not isinstance(record, dict)
            or record.get("object_name") != object_binding.object_name
            or record.get("layer") != object_binding.layer
            or record.get("layer_name") != layer.layer_name
            or record.get("role") != object_binding.role
            or record.get("material_id") != material.material_id
            or record.get("kind") != material.kind
            or record.get("is_superconducting") is not material.is_superconducting
            or record.get("requested_library_name") != material.library_name
            or record.get("native_destination_layer_prefix") != layer.layer_name
        ):
            raise RuntimeError("HFSS V3 material binding does not match the spec")
        binding = record.get("hfss_pec_binding")
        if not material.is_superconducting:
            if binding is not None:
                raise RuntimeError("non-PEC HFSS material has PEC binding evidence")
            if effective_layers:
                observed = record.get("observed")
                if (
                    not isinstance(observed, dict)
                    or not isinstance(observed.get("native_material_name"), str)
                    or observed["native_material_name"].casefold()
                    != material.library_name.casefold()
                    or not _positive_unique_ids([record.get("native_object_id")], 1)
                ):
                    raise RuntimeError(
                        "HFSS dielectric material native binding differs"
                    )
            continue
        face_ids = binding.get("native_face_ids") if isinstance(binding, dict) else None
        if not isinstance(binding, dict) or (
            binding.get("source_object") != object_binding.object_name
            or binding.get("source_material_id") != material.material_id
            or binding.get("source_material_kind") != material.kind
            or binding.get("source_library_name") != material.library_name
            or not _positive_unique_ids([binding.get("native_object_id")], 1)
            or not isinstance(face_ids, list)
            or not face_ids
            or not _positive_unique_ids(face_ids, len(face_ids))
        ):
            raise RuntimeError("HFSS V3 PEC source binding evidence is invalid")
        if effective_layers and (
            binding.get("native_object_id") != record.get("native_object_id")
            or binding.get("native_object_type") != record.get("native_object_type")
            or binding.get("native_face_ids") != record.get("native_face_ids")
        ):
            raise RuntimeError("HFSS effective object and PEC identities differ")
        evidence = binding.get("verified_evidence")
        if binding.get("native_object_type") == "Solid":
            if (
                binding.get("implementation") != "pec_material_solve_inside_false"
                or set(binding)
                != {
                    "source_object",
                    "source_material_id",
                    "source_material_kind",
                    "source_library_name",
                    "native_object_id",
                    "native_object_type",
                    "native_face_ids",
                    "implementation",
                    "verified_evidence",
                }
                or not isinstance(evidence, dict)
                or set(evidence) != {"native_material_name", "native_solve_inside"}
                or not isinstance(evidence.get("native_material_name"), str)
                or evidence["native_material_name"].casefold() != "pec"
                or evidence.get("native_solve_inside") is not False
                or record.get("observed") != evidence
                or "requested_pec_boundary" in record
            ):
                raise RuntimeError("HFSS V3 solid PEC evidence is invalid")
        elif binding.get("native_object_type") == "Sheet":
            if (
                binding.get("implementation") != "perfect_e_sheet"
                or set(binding)
                != {
                    "source_object",
                    "source_material_id",
                    "source_material_kind",
                    "source_library_name",
                    "native_object_id",
                    "native_object_type",
                    "native_face_ids",
                    "implementation",
                    "verified_evidence",
                }
                or record.get("requested_pec_boundary") != "SCGSimPEC"
            ):
                raise RuntimeError("HFSS V3 sheet PEC evidence is invalid")
            sheet_records.append(record)
        else:
            raise RuntimeError("HFSS V3 PEC native object type is invalid")
        pec_records.append(record)

    object_by_native_id: dict[int, str] = {}
    object_by_face_id: dict[int, str] = {}
    for record in pec_records:
        binding = record["hfss_pec_binding"]
        faces = set(binding["native_face_ids"])
        if set(object_by_face_id) & faces:
            raise RuntimeError("HFSS V3 PEC native faces are ambiguous")
        object_name = binding["source_object"]
        object_id = binding["native_object_id"]
        if object_id in object_by_native_id:
            raise RuntimeError("HFSS V3 PEC native object IDs are ambiguous")
        object_by_native_id[object_id] = object_name
        object_by_face_id.update({face_id: object_name for face_id in faces})
    if set(object_by_native_id) & set(object_by_face_id):
        raise RuntimeError("HFSS V3 PEC native assignment IDs are ambiguous")
    if not sheet_records:
        return

    all_target_faces: set[int] = set()
    target_faces_by_object: dict[str, set[int]] = {}
    target_object_id_by_object: dict[str, int] = {}
    for record in sheet_records:
        binding = record["hfss_pec_binding"]
        object_name = binding["source_object"]
        faces = set(binding["native_face_ids"])
        target_faces_by_object[object_name] = faces
        target_object_id_by_object[object_name] = binding["native_object_id"]
        all_target_faces.update(faces)

    reference = sheet_records[0]["hfss_pec_binding"].get("verified_evidence")
    if not isinstance(reference, dict):
        raise RuntimeError("HFSS V3 sheet PEC evidence is invalid")
    raw_ids = reference.get("raw_assignment_ids")
    typed = reference.get("typed_assignment")
    if (
        reference.get("native_boundary_name") != "SCGSimPEC"
        or reference.get("native_boundary_type") != "Perfect E"
        or not isinstance(raw_ids, list)
        or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in raw_ids
        )
        or not isinstance(typed, list)
        or len(typed) != len(raw_ids)
    ):
        raise RuntimeError("HFSS V3 sheet PEC boundary evidence is invalid")
    covered: set[int] = set()
    covered_objects: set[str] = set()
    for raw_id, item in zip(raw_ids, typed, strict=True):
        if (
            not isinstance(item, dict)
            or set(item) != {"raw_id", "native_kind", "object_name", "face_ids"}
            or item.get("raw_id") != raw_id
        ):
            raise RuntimeError("HFSS V3 PEC typed assignment is invalid")
        object_name = item.get("object_name")
        faces = item.get("face_ids")
        if object_name not in target_faces_by_object or not isinstance(faces, list):
            raise RuntimeError("HFSS V3 PEC assignment targets a non-sheet object")
        object_owner = object_by_native_id.get(raw_id)
        face_owner = object_by_face_id.get(raw_id)
        if object_owner is not None and face_owner is not None:
            raise RuntimeError("HFSS V3 PEC assignment ID is ambiguous")
        if object_owner is None and face_owner is None:
            raise RuntimeError("HFSS V3 PEC assignment ID is unknown")
        if item.get("native_kind") == "object":
            if (
                object_owner != object_name
                or raw_id != target_object_id_by_object[object_name]
                or set(faces) != target_faces_by_object[object_name]
            ):
                raise RuntimeError("HFSS V3 PEC object assignment is incomplete")
        elif item.get("native_kind") == "face":
            if face_owner != object_name or faces != [raw_id]:
                raise RuntimeError("HFSS V3 PEC face assignment is invalid")
        else:
            raise RuntimeError("HFSS V3 PEC assignment kind is invalid")
        covered.update(faces)
        covered_objects.add(object_name)
    if covered != all_target_faces or covered_objects != set(target_faces_by_object):
        raise RuntimeError("HFSS V3 PEC assignment coverage is incomplete")
    for record in sheet_records:
        binding = record["hfss_pec_binding"]
        evidence = binding.get("verified_evidence")
        expected_covered = sorted(target_faces_by_object[binding["source_object"]])
        if (
            not isinstance(evidence, dict)
            or set(evidence)
            != {
                "native_boundary_name",
                "native_boundary_type",
                "raw_assignment_ids",
                "typed_assignment",
                "covered_face_ids",
            }
            or evidence.get("native_boundary_name") != "SCGSimPEC"
            or evidence.get("native_boundary_type") != "Perfect E"
            or evidence.get("raw_assignment_ids") != raw_ids
            or evidence.get("typed_assignment") != typed
            or evidence.get("covered_face_ids") != expected_covered
            or record.get("observed")
            != {
                "native_pec_boundary": "SCGSimPEC",
                "native_pec_boundary_type": "Perfect E",
                "native_pec_face_ids": expected_covered,
                "native_pec_objects": sorted(target_faces_by_object),
            }
        ):
            raise RuntimeError("HFSS V3 per-sheet PEC evidence is inconsistent")


def _validate_hfss_setup_and_convergence(
    root: Path, receipt: dict[str, Any], spec: HfssDrivenSpec | HfssEigenmodeSpec
) -> None:
    if isinstance(spec, HfssEigenmodeSpec):
        native = {
            "minimum_frequency": f"{spec.run_control.minimum_frequency_ghz:g}GHz",
            "num_modes": spec.run_control.num_modes,
            "maximum_delta_frequency_percent": (
                spec.run_control.maximum_delta_frequency_percent
            ),
            "maximum_passes": spec.run_control.maximum_passes,
            "minimum_passes": spec.run_control.minimum_passes,
            "minimum_converged_passes": spec.run_control.minimum_converged_passes,
            "percent_refinement": spec.run_control.percent_refinement,
        }
    else:
        native = {
            "solve_type": "Broadband",
            "low_frequency": f"{spec.run_control.sweep.start_ghz:g}GHz",
            "high_frequency": f"{spec.run_control.sweep.stop_ghz:g}GHz",
            "maximum_delta_s": spec.run_control.maximum_delta_s,
            "maximum_passes": spec.run_control.maximum_passes,
            "minimum_passes": spec.run_control.minimum_passes,
            "minimum_converged_passes": spec.run_control.minimum_converged_passes,
            "percent_refinement": spec.run_control.percent_refinement,
        }
    if receipt.get("setup") != {"name": spec.run_control.setup_name, "native": native}:
        raise RuntimeError("HFSS native setup evidence is invalid")
    historical_profile = (
        _eigenmode_convergence_source(receipt) == "profile"
        if isinstance(spec, HfssEigenmodeSpec)
        else False
    )
    if receipt.get("convergence") != read_hfss_convergence(
        root, spec, historical_profile=historical_profile
    ):
        raise RuntimeError("HFSS native convergence evidence is invalid")


def _eigenmode_convergence_source(receipt: dict[str, Any]) -> str:
    convergence = receipt.get("convergence")
    sources = convergence.get("sources") if isinstance(convergence, dict) else None
    if not isinstance(sources, dict):
        raise RuntimeError("Eigenmode convergence sources are invalid")
    if set(sources) == {"export_convergence"}:
        return "export_convergence"
    if set(sources) == {"asol", "profile"}:
        return "profile"
    raise RuntimeError("Eigenmode convergence sources are not canonical")


def _validate_q2d_readback(root: Path, receipt: dict[str, Any], spec: Q2dSpec) -> None:
    if receipt.get("ports") != [] or receipt.get("nets") != []:
        raise RuntimeError("Q2D receipt must not contain HFSS ports or Q3D nets")
    conductors = receipt.get("conductors")
    if not isinstance(conductors, list) or len(conductors) != len(spec.conductors):
        raise RuntimeError("Q2D receipt has invalid conductor evidence")
    for record, expected in zip(conductors, spec.conductors, strict=True):
        if (
            not isinstance(record, dict)
            or record.get("name") != expected.name
            or record.get("conductor_type") != expected.conductor_type
            or record.get("object_names") != list(expected.object_names)
            or record.get("thickness_um") != expected.thickness_um
            or record.get("solve_option") != "SolveOnBoundary"
            or not isinstance(record.get("native_object_ids"), list)
            or len(record["native_object_ids"]) != len(expected.object_names)
            or not all(isinstance(value, int) for value in record["native_object_ids"])
        ):
            raise RuntimeError("Q2D native conductor evidence does not match the spec")
    expected_setup = {
        "name": spec.run_control.setup_name,
        "native": {
            "adaptive_frequency": f"{spec.run_control.frequency_ghz:g}GHz",
            "cg_maximum_passes": spec.run_control.maximum_passes,
            "cg_minimum_passes": spec.run_control.minimum_passes,
            "cg_minimum_converged_passes": spec.run_control.minimum_converged_passes,
            "cg_convergence_percent": spec.run_control.convergence_percent,
            "cg_percent_refinement": spec.run_control.percent_refinement,
            "rl_maximum_passes": spec.run_control.maximum_passes,
            "rl_minimum_passes": spec.run_control.minimum_passes,
            "rl_minimum_converged_passes": spec.run_control.minimum_converged_passes,
            "rl_convergence_percent": spec.run_control.convergence_percent,
            "rl_percent_refinement": spec.run_control.percent_refinement,
        },
    }
    if receipt.get("setup") != expected_setup:
        raise RuntimeError("Q2D native setup evidence is invalid")
    if receipt.get("convergence") != read_q2d_convergence(root, spec):
        raise RuntimeError("Q2D native convergence evidence is invalid")
    readback = receipt.get("result_readback")
    matrices = readback.get("matrices") if isinstance(readback, dict) else None
    rows, summary = read_q2d_rlgc_matrix(
        _contained(root, "results/q2d/rlgc_matrix.csv"), spec
    )
    expected_matrices = {
        "path": "results/q2d/rlgc_matrix.csv",
        "frequency_ghz": spec.run_control.frequency_ghz,
        "length_setting": "Distributed",
        "length": "1meter",
        "matrix_type": "Maxwell, Spice, Couple",
        "native": summary,
        "primary_rows": len(rows),
    }
    if matrices != expected_matrices:
        raise RuntimeError("Q2D matrix readback is invalid")


def _validate_q3d_readback(root: Path, receipt: dict[str, Any], spec: Q3dSpec) -> None:
    if receipt.get("ports") != []:
        raise RuntimeError("Q3D receipt must not contain HFSS ports")
    nets = receipt.get("nets")
    if not isinstance(nets, list) or len(nets) != len(spec.nets):
        raise RuntimeError("Q3D receipt has invalid net evidence")
    for record, expected in zip(nets, spec.nets, strict=True):
        if (
            not isinstance(record, dict)
            or record.get("name") != expected.name
            or record.get("net_type") != expected.net_type
            or record.get("object_names") != list(expected.object_names)
            or not isinstance(record.get("native_object_ids"), list)
            or not _positive_unique_ids(
                record["native_object_ids"], len(expected.object_names)
            )
        ):
            raise RuntimeError("Q3D native net evidence does not match the spec")
        if expected.net_type == "Signal" and expected.source_object is not None:
            for kind, object_name, side in (
                ("source", expected.source_object, expected.source_side),
                ("sink", expected.sink_object, expected.sink_side),
            ):
                terminal = record.get(kind)
                if (
                    not isinstance(terminal, dict)
                    or terminal.get("name") != f"{expected.name}{kind.title()}"
                    or terminal.get("object_name") != object_name
                    or terminal.get("side") != side
                ):
                    raise RuntimeError("Q3D native terminal evidence is invalid")
                effective = (
                    {}
                    if spec.modeling is None
                    else {body["body_id"]: body for body in spec.effective_bodies}
                )
                if effective and effective[object_name]["representation"] == "sheet":
                    from scgsim.aedt.runtime.native.q3d_bodies import (
                        q3d_terminal_segments,
                    )

                    body = next(
                        body for body in spec.bodies if body.body_id == object_name
                    )
                    indices = q3d_terminal_segments(body, side)
                    edges = terminal.get("native_edge_ids")
                    if (
                        not isinstance(edges, list)
                        or not _positive_unique_ids(edges, len(indices))
                        or terminal.get("source_edges")
                        != [
                            {"source_segment_index": index, "native_edge_id": edge}
                            for index, edge in zip(indices, edges, strict=True)
                        ]
                        or terminal.get("terminal_type") != "ConstantVoltage"
                        or "native_face_ids" in terminal
                    ):
                        raise RuntimeError(
                            "Q3D Sheet terminal evidence differs from source exterior"
                        )
                else:
                    faces = terminal.get("native_face_ids")
                    if not isinstance(faces, list) or not _positive_unique_ids(
                        faces, 1
                    ):
                        raise RuntimeError(
                            "Q3D Solid terminal face evidence is invalid"
                        )
        elif "source" in record or "sink" in record:
            raise RuntimeError("Q3D net receipt contains undeclared terminals")
    net_object_ids = [
        object_id for record in nets for object_id in record["native_object_ids"]
    ]
    if len(set(net_object_ids)) != len(net_object_ids):
        raise RuntimeError("Q3D native object IDs must be unique across nets")
    _validate_q3d_body_bindings(receipt, spec, nets, root)
    _validate_q3d_region_ground(receipt, spec, nets)
    expected_setup = {
        "name": spec.run_control.setup_name,
        "native": {
            "adaptive_frequency": f"{spec.run_control.frequency_ghz:g}GHz",
            "capacitance_maximum_passes": spec.run_control.maximum_passes,
            "capacitance_minimum_passes": spec.run_control.minimum_passes,
            "capacitance_minimum_converged_passes": (
                spec.run_control.minimum_converged_passes
            ),
            "capacitance_convergence_percent": spec.run_control.convergence_percent,
            "capacitance_percent_refinement": spec.run_control.percent_refinement,
            "dc_enabled": False,
        },
    }
    if spec.solve_ac_rl:
        expected_setup["native"].update(
            {
                "ac_rl_maximum_passes": spec.run_control.maximum_passes,
                "ac_rl_minimum_passes": spec.run_control.minimum_passes,
                "ac_rl_minimum_converged_passes": (
                    spec.run_control.minimum_converged_passes
                ),
                "ac_rl_convergence_percent": spec.run_control.convergence_percent,
                "ac_rl_percent_refinement": spec.run_control.percent_refinement,
            }
        )
    if receipt.get("setup") != expected_setup:
        raise RuntimeError("Q3D native setup evidence is invalid")
    if receipt.get("convergence") != read_q3d_convergence(root, spec):
        raise RuntimeError("Q3D native convergence evidence is invalid")
    readback = receipt.get("result_readback")
    matrices = readback.get("matrices") if isinstance(readback, dict) else None
    if spec.solve_ac_rl:
        if (
            not isinstance(matrices, dict)
            or matrices.get("frequency_ghz") != spec.run_control.frequency_ghz
            or set(matrices.get("native", {})) != {"c", "ac_rl"}
            or not isinstance(matrices.get("normalized_rows"), int)
            or matrices["normalized_rows"] <= 0
        ):
            raise RuntimeError("Q3D matrix readback is invalid")
        _validate_normalized_matrices(
            _contained(root, "results/q3d/matrices.csv"),
            matrices["normalized_rows"],
            {"C", "AC RL"},
        )
        _read_q3d_original_cg(root, spec)
        return
    rows, summary = _read_q3d_original_cg(root, spec)
    expected_matrices = {
        "path": "results/q3d/c_matrix.csv",
        "frequency_ghz": spec.run_control.frequency_ghz,
        "native": summary,
        "primary_rows": len(rows),
    }
    if matrices != expected_matrices:
        raise RuntimeError("Q3D capacitance-only matrix readback is invalid")


def _read_q3d_original_cg(
    root: Path, spec: Q3dSpec
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    options = {
        "expected_label_set": {
            net.name for net in spec.nets if net.net_type == "Signal"
        }
    }
    return parse_matrix_export(
        _contained(root, "results/q3d/c_matrix.csv"),
        "Q3D",
        "C",
        spec.run_control.frequency_ghz,
        {"Capacitance Matrix": "C", "Conductance Matrix": "G"},
        **options,
    )


def _validate_q3d_body_bindings(
    receipt: dict[str, Any], spec: Q3dSpec, nets: list[dict[str, Any]], root: Path
) -> None:
    """Bind every created body to its declared geometry and final native identity.

    Native incidence may be unavailable; source declarations never substitute
    for native topology. The receipt must state which readback actually exists.
    """
    materials = receipt.get("materials")
    if not isinstance(materials, list) or len(materials) != len(spec.bodies):
        raise RuntimeError("Q3D body material readback is invalid")
    net_object_ids = {
        name: native_id
        for net_record, net_spec in zip(nets, spec.nets, strict=True)
        for name, native_id in zip(
            net_spec.object_names, net_record["native_object_ids"], strict=True
        )
    }
    records = {
        record.get("object_name"): record
        for record in materials
        if isinstance(record, dict)
    }
    if len(records) != len(materials) or set(records) != {
        body.body_id for body in spec.bodies
    }:
        raise RuntimeError("Q3D native body inventory differs from the spec")
    historical = spec.modeling is None
    raw_bodies = {
        body["body_id"]: body for body in read_json(root / "aedt_spec.json")["bodies"]
    }
    effective_bodies = (
        {} if historical else {body["body_id"]: body for body in spec.effective_bodies}
    )
    seen_native_ids = set()
    for body in spec.bodies:
        record = records[body.body_id]
        material = spec.materials[body.material_id]
        observed = record.get("observed")
        expected_native_material = (
            "pec" if material.is_superconducting else material.library_name
        )
        sheet = (
            not historical
            and effective_bodies[body.body_id]["representation"] == "sheet"
        )
        observed_key = (
            "native_boundary_material_name" if sheet else "native_material_name"
        )
        if (
            record.get("material_id") != body.material_id
            or record.get("role") != body.physical_role
            or record.get("kind") != material.kind
            or record.get("is_superconducting") is not material.is_superconducting
            or record.get("requested_library_name") != material.library_name
            or not isinstance(observed, dict)
            or not isinstance(observed.get(observed_key), str)
            or observed[observed_key].casefold() != expected_native_material.casefold()
        ):
            raise RuntimeError("Q3D body material binding differs from the spec")
        if sheet and (
            observed.get("native_material_property") != "not_applicable_sheet"
            or "native_material_name" in observed
        ):
            raise RuntimeError(
                "Q3D Sheet evidence must identify boundary-only material"
            )
        evidence = record.get("body_binding")
        if not isinstance(evidence, dict):
            raise RuntimeError("Q3D native body binding is absent")
        native_id = evidence.get("native_object_id")
        faces = evidence.get("native_face_ids")
        if (
            evidence.get("body")
            != (raw_bodies[body.body_id] if historical else body.to_payload())
            or evidence.get("object_name") != body.body_id
            or evidence.get("native_object_type")
            != (
                "Sheet"
                if not historical
                and effective_bodies[body.body_id]["representation"] == "sheet"
                else "Solid"
            )
            or not _positive_unique_ids([native_id], 1)
            or native_id in seen_native_ids
            or not isinstance(faces, list)
            or not faces
            or not _positive_unique_ids(faces, len(faces))
            or (
                body.net_id is not None
                and net_object_ids.get(body.body_id) != native_id
            )
        ):
            raise RuntimeError(
                "Q3D body binding differs from the native identity/source declaration"
            )
        if not historical:
            effective = effective_bodies[body.body_id]
            if evidence.get("effective_geometry") != effective:
                raise RuntimeError(
                    "Q3D effective geometry binding differs from the spec"
                )
            thin = evidence.get("thin_conductor")
            if effective["representation"] == "sheet":
                if not isinstance(thin, dict):
                    raise RuntimeError("Q3D ThinConductor evidence is not recorded")
                raw_material = thin.get("material_raw")
                if (
                    isinstance(raw_material, list)
                    and len(raw_material) == 2
                    and raw_material[0] == "Material:="
                ):
                    raw_material = raw_material[1]
                if not isinstance(raw_material, str):
                    raise RuntimeError(
                        "Q3D ThinConductor Material has unknown native shape"
                    )
                material_name = raw_material.strip('"')
                # Offline replay of pinned PyAEDT 1.3 Length conversion, without AEDT.
                raw_thickness = thin.get("thickness")
                length = re.fullmatch(
                    r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)([A-Za-z]+)",
                    raw_thickness if isinstance(raw_thickness, str) else "",
                )
                scales = {
                    "fm": 1e-15,
                    "pm": 1e-12,
                    "nm": 1e-9,
                    "um": 1e-6,
                    "mm": 1e-3,
                    "cm": 1e-2,
                    "dm": 1e-1,
                    "meter": 1.0,
                    "km": 1e3,
                    "uin": 0.0254 * 1e-6,
                    "mil": 0.0254 * 1e-3,
                    "in": 0.0254,
                    "ft": 0.0254 * 12,
                    "yd": 0.0254 * 36,
                    "mile": 0.0254 * 63360,
                }
                if length is None or length[2].lower() not in scales:
                    raise RuntimeError(
                        "Q3D ThinConductor Thickness is not a native length"
                    )
                thickness_um = float(length[1]) * scales[length[2].lower()] / 1e-6
                raw_ids = thin.get("native_assignment_ids_raw")
                if not isinstance(raw_ids, list) or any(
                    not (
                        type(value) is int
                        or (isinstance(value, str) and re.fullmatch(r"[0-9]+", value))
                    )
                    for value in raw_ids
                ):
                    raise RuntimeError(
                        "Q3D ThinConductor assignment readback is malformed"
                    )
                assigned_ids = [int(value) for value in raw_ids]
                expected_thin = {
                    "name": f"{body.body_id}ThinConductor",
                    "bound_type": "ThinConductor",
                    "material": material_name,
                    "material_raw": thin.get("material_raw"),
                    "thickness": raw_thickness,
                    "thickness_um": thickness_um,
                    "native_assignment_ids_raw": raw_ids,
                    "native_object_ids": assigned_ids,
                }
                if (
                    thin != expected_thin
                    or material_name.casefold() != "pec"
                    or material_name != observed[observed_key]
                    or thickness_um != effective["physical_thickness_um"]
                    or assigned_ids != [native_id]
                ):
                    raise RuntimeError(
                        "Q3D ThinConductor binding differs from physical source"
                    )
            elif thin is not None:
                raise RuntimeError("Q3D Solid body contains Sheet boundary evidence")
        seen_native_ids.add(native_id)
        incidence = evidence.get("incidence")
        if not isinstance(incidence, dict):
            raise RuntimeError("Q3D body incidence availability is not recorded")
        if incidence.get("status") == "unavailable":
            if (
                not isinstance(incidence.get("unavailable_reason"), str)
                or not incidence["unavailable_reason"]
            ):
                raise RuntimeError("Q3D unavailable incidence lacks a reason")
        elif incidence.get("status") == "observed":
            if (
                incidence.get("face_ids") != faces
                or not isinstance(incidence.get("face_edge_ids"), dict)
                or not isinstance(incidence.get("edge_vertex_ids"), dict)
            ):
                raise RuntimeError("Q3D observed native incidence is malformed")
        else:
            raise RuntimeError("Q3D native incidence status is invalid")


def _validate_q3d_region_ground(
    receipt: dict[str, Any], spec: Q3dSpec, nets: list[Any]
) -> None:
    region = receipt.get("region")
    if not isinstance(region, dict):
        raise TypeError("Q3D receipt has invalid Region evidence")
    requested_padding = list(spec.region_padding_um)
    if (
        region.get("padding_um") != requested_padding
        or region.get("requested_padding_um") != requested_padding
    ):
        raise RuntimeError("Q3D receipt Region padding does not match the spec")
    evidence = region.get("grounded_region")
    if spec.grounded_region_net is None:
        if evidence is not None:
            raise RuntimeError("open Q3D Region receipt contains grounding evidence")
        return
    if not isinstance(evidence, dict):
        raise TypeError("grounded Q3D Region receipt lacks evidence")
    source_id, source_bounds, source_faces = _q3d_region_evidence(
        evidence.get("source_region")
    )
    final_id, final_bounds, _final_faces = _q3d_region_evidence(
        evidence.get("final_region")
    )
    sheets = evidence.get("sheets")
    if (
        region.get("native_region_object_id") != final_id
        or region.get("native_bounding_box_um") != list(final_bounds)
        or evidence.get("target_net") != spec.grounded_region_net
        or evidence.get("target_net_origin") != "generated_region_enclosure"
        or evidence.get("native_final_region_padding_um") != [0.0] * 6
        or evidence.get("native_region_object_id") != final_id
        or evidence.get("native_bounding_box_um") != list(final_bounds)
        or not _q3d_values_close(list(source_bounds), list(final_bounds))
        or not isinstance(sheets, list)
        or len(sheets) != len(_Q3D_REGION_DIRECTIONS)
    ):
        raise RuntimeError("grounded Q3D Region receipt is invalid")
    sheet_ids: list[int] = []
    thin_boundaries: list[dict[str, Any]] = []
    source_face_ids: set[int] = set()
    for direction, record in zip(_Q3D_REGION_DIRECTIONS, sheets, strict=True):
        if not isinstance(record, dict):
            raise TypeError("grounded Q3D Region sheet receipt is invalid")
        sheet_name, boundary_name = _Q3D_REGION_SHEET_NAMES[direction]
        face_id = record.get("source_face_id")
        sheet_id = record.get("sheet_object_id")
        source = source_faces[direction]
        expected_bounds = _q3d_sheet_bounds(direction, final_bounds)
        if (
            record.get("direction") != direction
            or record.get("source_region_object_id") != source_id
            or face_id != source["native_face_id"]
            or face_id in source_face_ids
            or record.get("source_face_center_um") != source["native_face_center_um"]
            or record.get("source_face_normal") != source["native_face_normal"]
            or record.get("sheet_name") != sheet_name
            or not _positive_unique_ids([sheet_id], 1)
            or not _q3d_values_close(
                record.get("sheet_bounding_box_um"), expected_bounds
            )
            or record.get("boundary_name") != boundary_name
            or record.get("native_thin_conductor_object_ids") != [sheet_id]
        ):
            raise RuntimeError("grounded Q3D Region sheet receipt is invalid")
        source_face_ids.add(face_id)
        sheet_ids.append(sheet_id)
        thin_boundaries.append(
            {
                "name": boundary_name,
                "bound_type": "ThinConductor",
                "object_ids": [sheet_id],
                "material": "pec",
                "thickness": "1um",
            }
        )
    declared_net_ids = {record["name"]: record["native_object_ids"] for record in nets}
    all_net_ids = {
        object_id for record in nets for object_id in record["native_object_ids"]
    }
    if (
        spec.grounded_region_net in declared_net_ids
        or evidence.get("native_declared_net_object_ids") != declared_net_ids
        or len(sheet_ids) != 6
        or len(set(sheet_ids)) != 6
        or set(sheet_ids) & all_net_ids
        or final_id in all_net_ids
        or final_id in sheet_ids
    ):
        raise RuntimeError("grounded Q3D Region object IDs are invalid")
    expected_ids = sheet_ids
    saved = evidence.get("native_saved_boundaries")
    if (
        evidence.get("native_target_net_object_ids") != expected_ids
        or evidence.get("native_design_validation")
        != {"method": "ValidateDesign", "ok": True}
        or not isinstance(saved, dict)
        or saved.get("target")
        != {
            "name": spec.grounded_region_net,
            "bound_type": "GroundNet",
            "origin": "generated_region_enclosure",
            "object_ids": expected_ids,
        }
        or saved.get("thin_conductors") != thin_boundaries
    ):
        raise RuntimeError("grounded Q3D Region native assignment receipt is invalid")


def _positive_unique_ids(value: Any, count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == count
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in value
        )
        and len(set(value)) == count
    )


def _q3d_region_evidence(
    value: Any,
) -> tuple[
    int, tuple[float, float, float, float, float, float], dict[str, dict[str, Any]]
]:
    if not isinstance(value, dict):
        raise TypeError("Q3D Region evidence is invalid")
    region_id = value.get("native_object_id")
    bounds = _q3d_bounds(value.get("native_bounding_box_um"))
    faces = value.get("faces")
    if (
        not isinstance(region_id, int)
        or isinstance(region_id, bool)
        or region_id <= 0
        or not isinstance(faces, list)
        or len(faces) != len(_Q3D_REGION_DIRECTIONS)
    ):
        raise RuntimeError("Q3D Region evidence is invalid")
    by_direction: dict[str, dict[str, Any]] = {}
    face_ids: set[int] = set()
    for direction, record in zip(_Q3D_REGION_DIRECTIONS, faces, strict=True):
        if (
            not isinstance(record, dict)
            or record.get("direction") != direction
            or not _positive_unique_ids([record.get("native_face_id")], 1)
            or record["native_face_id"] in face_ids
            or not _q3d_face_center_matches(
                direction, record.get("native_face_center_um"), bounds
            )
            or not _q3d_face_normal_matches(direction, record.get("native_face_normal"))
        ):
            raise RuntimeError("Q3D Region face evidence is invalid")
        face_ids.add(record["native_face_id"])
        by_direction[direction] = record
    return region_id, bounds, by_direction


def _q3d_bounds(value: Any) -> tuple[float, float, float, float, float, float]:
    if not isinstance(value, list) or len(value) != 6:
        raise RuntimeError("grounded Q3D Region bounds are invalid")
    try:
        bounds = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("grounded Q3D Region bounds are invalid") from exc
    if not all(math.isfinite(item) for item in bounds) or any(
        bounds[index] >= bounds[index + 3] for index in range(3)
    ):
        raise RuntimeError("grounded Q3D Region bounds are invalid")
    return bounds


def _q3d_face_center_matches(
    direction: str, value: Any, bounds: tuple[float, float, float, float, float, float]
) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    try:
        center = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(item) for item in center):
        return False
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    expected = {
        "+X": [x_max, (y_min + y_max) / 2, (z_min + z_max) / 2],
        "-X": [x_min, (y_min + y_max) / 2, (z_min + z_max) / 2],
        "+Y": [(x_min + x_max) / 2, y_max, (z_min + z_max) / 2],
        "-Y": [(x_min + x_max) / 2, y_min, (z_min + z_max) / 2],
        "+Z": [(x_min + x_max) / 2, (y_min + y_max) / 2, z_max],
        "-Z": [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min],
    }[direction]
    return all(
        math.isclose(actual, required, rel_tol=0.0, abs_tol=1e-9)
        for actual, required in zip(center, expected, strict=True)
    )


def _q3d_sheet_bounds(
    direction: str, bounds: tuple[float, float, float, float, float, float]
) -> list[float]:
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    return {
        "+X": [x_max, y_min, z_min, x_max, y_max, z_max],
        "-X": [x_min, y_min, z_min, x_min, y_max, z_max],
        "+Y": [x_min, y_max, z_min, x_max, y_max, z_max],
        "-Y": [x_min, y_min, z_min, x_max, y_min, z_max],
        "+Z": [x_min, y_min, z_max, x_max, y_max, z_max],
        "-Z": [x_min, y_min, z_min, x_max, y_max, z_min],
    }[direction]


def _q3d_values_close(value: Any, expected: list[float]) -> bool:
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    try:
        observed = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(
        math.isclose(actual, required, rel_tol=0.0, abs_tol=1e-9)
        for actual, required in zip(observed, expected, strict=True)
    )


def _q3d_face_normal_matches(direction: str, value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    try:
        normal = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(item) for item in normal):
        return False
    axis, sign = {
        "+X": (0, 1.0),
        "-X": (0, -1.0),
        "+Y": (1, 1.0),
        "-Y": (1, -1.0),
        "+Z": (2, 1.0),
        "-Z": (2, -1.0),
    }[direction]
    return math.isclose(normal[axis], sign, rel_tol=0.0, abs_tol=1e-9) and all(
        math.isclose(item, 0.0, rel_tol=0.0, abs_tol=1e-9)
        for index, item in enumerate(normal)
        if index != axis
    )


def _validate_normalized_matrices(
    path: Path, expected_rows: int, problem_types: set[str]
) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_rows or not rows:
        raise RuntimeError("normalized matrix row count is invalid")
    expected_quantities = {"C", "G", "L", "R"}
    if {row.get("quantity") for row in rows} != expected_quantities:
        raise RuntimeError("normalized matrix quantities are incomplete")
    for row in rows:
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("normalized matrix value is invalid") from exc
        if (
            not math.isfinite(value)
            or row.get("problem_type") not in problem_types
            or not row.get("row")
            or not row.get("column")
            or not row.get("unit")
        ):
            raise RuntimeError("normalized matrix row is invalid")


def _validate_eigenmode_readback(
    receipt: dict[str, Any], spec: HfssEigenmodeSpec | HfssEprSpec
) -> None:
    readback = receipt.get("result_readback")
    if isinstance(spec, HfssEprSpec):
        # Body-first loads belong to geometry; this producer has no ports field.
        if "ports" in receipt:
            raise RuntimeError("body-first Eigenmode receipt must not contain ports")
        readback = readback.get("final_modes") if isinstance(readback, dict) else None
    elif receipt.get("ports") != []:
        raise RuntimeError("HFSS Eigenmode receipt must not contain ports")
    values = readback.get("eigenmodes") if isinstance(readback, dict) else None
    if not isinstance(values, dict):
        raise TypeError("completed receipt has no Eigenmode result readback")
    modes = spec.run_control.num_modes
    frequencies = values.get("frequencies_ghz")
    q_factors = values.get("q_factors")
    if (
        values.get("modes") != modes
        or values.get("frequency_unit") != "GHz"
        or values.get("mode_indices") != list(range(1, modes + 1))
        or not isinstance(frequencies, list)
        or len(frequencies) != modes
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in frequencies
        )
        or not isinstance(q_factors, list)
        or len(q_factors) != modes
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in q_factors
        )
        or values.get("native_export") != "eigenmodes.eig"
        or not isinstance(values.get("native_export_bytes"), int)
        or values["native_export_bytes"] <= 0
    ):
        raise RuntimeError("completed receipt Eigenmode result readback is invalid")


def _validate_modal_native_evidence(ports: Any, spec: HfssDrivenSpec) -> None:
    if not isinstance(ports, list) or len(ports) != 2:
        raise RuntimeError("completed receipt has invalid modal ports")
    native_ports = spec.ports if spec.modeling is None else spec.effective_ports
    for record, port, native_port in zip(ports, spec.ports, native_ports, strict=True):
        if not isinstance(port, ModalPort) or not isinstance(record, dict):
            raise TypeError("completed receipt has invalid modal port identity")
        if (
            record.get("index") != port.index
            or record.get("boundary") != port.name
            or record.get("modal_excitation") != port.name
            or not isinstance(record.get("face_id"), int)
            or not isinstance(record.get("face_center_um"), list)
            or len(record["face_center_um"]) != 3
            or record.get("requested")
            != {
                "integration_line_um": [
                    list(point) for point in port.integration_line_um
                ],
                "modes": 1,
                "renormalize": False,
                "deembed_um": 0.0,
                "characteristic_impedance": "Zpi",
            }
        ):
            raise RuntimeError("completed receipt modal request does not match spec")
        if spec.modeling is not None and record.get("effective") != {
            "integration_line_um": [
                list(point) for point in native_port.integration_line_um
            ]
        }:
            raise RuntimeError(
                "completed receipt modal effective endpoints differ from source map"
            )
        native = record.get("native")
        if not isinstance(native, dict):
            raise TypeError("completed receipt has no native modal evidence")
        expected_names = [item.name for item in spec.ports[: port.index]]
        expected_boundary = {
            "bound_type": "Wave Port",
            "wave_port_type": "Modal",
            "faces": [record["face_id"]],
            "num_modes": 1,
            "deembed": False,
            "mode_number": 1,
            "use_integration_line": True,
            "characteristic_impedance": "Zpi",
        }
        saved_boundary = native.get("saved_boundary")
        integration_line = (
            saved_boundary.get("integration_line_um")
            if isinstance(saved_boundary, dict)
            else None
        )
        if (
            native.get("excitation_names") != expected_names
            or native.get("boundary_properties")
            != {
                "Deembed": False,
                "Name": port.name,
                "Num Modes": "1",
                "Renorm All Modes": False,
                "Type": "Wave Port",
            }
            or not isinstance(saved_boundary, dict)
            or {
                key: value
                for key, value in saved_boundary.items()
                if key != "integration_line_um"
            }
            != expected_boundary
            or not isinstance(integration_line, list)
            or len(integration_line) != 2
            or any(
                not math.isclose(float(actual), wanted, abs_tol=1e-9)
                for actual_point, expected_point in zip(
                    integration_line, native_port.integration_line_um, strict=True
                )
                for actual, wanted in zip(actual_point, expected_point, strict=True)
            )
        ):
            raise RuntimeError("completed receipt modal native evidence mismatch")


def _validate_lumped_readback(receipt: dict[str, Any], prepared: Any) -> None:
    """Bind current generic loads/supports; historical empty treatment is unchanged."""
    from scgsim.aedt.epr.models import detached
    from scgsim.aedt.runtime.native.hfss_lumped import rlc_scalar_si

    supports = prepared.source.get("lumped_supports", ())
    if not supports and not prepared.lumped_rlcs:
        return
    geometry = receipt.get("geometry")
    if (
        not isinstance(geometry, dict)
        or geometry.get("source_sha256") != prepared.source_sha256
    ):
        raise RuntimeError("native support source identity differs")
    native_supports = geometry.get("lumped_supports")
    if not isinstance(native_supports, list) or len(native_supports) != len(supports):
        raise RuntimeError("native support inventory differs")
    by_id = {}
    for requested, native in zip(supports, native_supports, strict=True):
        requested = detached(requested)
        if (
            any(
                native.get(key) != requested[key]
                for key in ("support_id", "source", "effective")
            )
            or native.get("native_object_type") != "Sheet"
            or not isinstance(native.get("native_object_id"), int)
            or len(native.get("native_face_ids", ())) != 1
        ):
            raise RuntimeError("native neutral support attribution differs")
        by_id[native["support_id"]] = native
    records = geometry.get("lumped_rlcs", [])
    if len(records) != len(prepared.lumped_rlcs):
        raise RuntimeError("generic RLC inventory differs")
    for treatment, record in zip(prepared.lumped_rlcs, records, strict=True):
        support = by_id[treatment.support_id]
        expected = {
            "topology": treatment.topology,
            "native_topology": {"series": "Serial", "parallel": "Parallel"}[
                treatment.topology
            ],
            "contact_points_um": support["effective"]["contact_points_um"],
            "resistance_ohm": treatment.resistance_ohm,
            "inductance_h": treatment.inductance_h,
            "capacitance_f": treatment.capacitance_f,
        }
        if (
            record.get("support_id") != treatment.support_id
            or record.get("source") != treatment.to_payload()
            or record.get("requested") != expected
            or record.get("object_name") != support["object_name"]
            or record.get("native_object_id") != support["native_object_id"]
            or record.get("native_face_ids") != support["native_face_ids"]
        ):
            raise RuntimeError("generic RLC source/native identity differs")
        assignment = record.get("assignment", {})
        if assignment.get("covered_objects") != [
            support["object_name"]
        ] or assignment.get("covered_faces") != sorted(support["native_face_ids"]):
            raise RuntimeError("generic RLC assignment differs")
        saved = record.get("saved_boundary", {})
        positions = saved.get("CurrentLine", {}).get("GeometryPosition", ())
        if (
            saved.get("BoundType") != "Lumped RLC"
            or record.get("saved_model_units") != "um"
            or len(positions) != 2
            or not _sha256_text(record.get("saved_project_sha256", ""))
            or record.get("saved_contact_points_um") != expected["contact_points_um"]
        ):
            raise RuntimeError("saved generic RLC directed line evidence differs")
        for point, position in zip(
            record["saved_contact_points_um"], positions, strict=True
        ):
            if (
                position.get("IsAttachedToEntity") is False
                and position.get("PositionType") == "AbsolutePosition"
            ):
                if [float(position[axis + "Position"]) for axis in "XYZ"] != point:
                    raise RuntimeError("saved generic RLC absolute position differs")
            elif (
                position.get("IsAttachedToEntity") is True
                and position.get("PositionType") == "EdgeCenter"
            ):
                contacts = [
                    item
                    for item in record.get("attached_contact_edges", ())
                    if item.get("edge_id") == position.get("EntityID")
                ]
                if (
                    len(contacts) != 1
                    or contacts[0].get("midpoint_um") != point
                    or contacts[0].get("object_name") != support["object_name"]
                    or contacts[0].get("native_object_id")
                    != support["native_object_id"]
                ):
                    raise RuntimeError(
                        "saved generic RLC attached edge identity unavailable"
                    )
            else:
                raise RuntimeError("saved generic RLC line position kind unsupported")
        properties = record.get("native_properties", {})
        if properties.get("RLC Type") != expected["native_topology"]:
            raise RuntimeError("generic RLC native topology differs")
        for key, enable, prop in (
            ("resistance_ohm", "Use Resist", "Resistance"),
            ("inductance_h", "Use Induct", "Inductance"),
            ("capacitance_f", "Use Cap", "Capacitance"),
        ):
            raw = properties.get(enable)
            if raw in (True, "true", "True", 1):
                enabled = True
            elif raw in (False, "false", "False", 0):
                enabled = False
            else:
                raise RuntimeError("generic RLC enable readback unavailable")
            value = expected[key]
            if enabled != (value is not None):
                raise RuntimeError("generic RLC enable differs")
            if enabled:
                observed = rlc_scalar_si(properties.get(prop), key)
                if observed != Decimal(str(value)) or observed != Decimal(
                    record["normalized_values_si"][key]
                ):
                    raise RuntimeError("generic RLC native component value differs")


def _validate_body_terminal_readback(
    receipt: dict[str, Any], spec: HfssDrivenGeometrySpec
) -> None:
    from scgsim.aedt.specs.common import LumpedTerminalPort

    geometry = receipt.get("geometry", {})
    if geometry.get("source_sha256") != spec.geometry.source_sha256:
        raise RuntimeError("body-first Terminal geometry source differs")
    _validate_lumped_readback(receipt, spec.geometry)
    conductors = spec.geometry.source["conductors"]
    objects = geometry.get("objects", [])

    def names(entity_ids: Any) -> list[str]:
        result = []
        for entity_id in entity_ids:
            ids = {
                item["semantic_id"]
                for item in conductors
                if entity_id in {item["semantic_id"], item["source_semantic_id"]}
            }
            matches = [
                item["object_name"]
                for item in objects
                if item["kind"] == "conductor" and item["semantic_id"] in ids
            ]
            if not matches:
                raise RuntimeError(
                    "body-first Terminal source Entity has no native bodies"
                )
            result.extend(name for name in matches if name not in result)
        return result

    global_references = {
        name
        for port in spec.ports
        for name in names(
            port.reference_entity_ids
            if isinstance(port, LumpedTerminalPort)
            else port.reference_objects
        )
    }
    for record, port in zip(receipt["ports"], spec.ports, strict=True):
        reference_ids = (
            port.reference_entity_ids
            if isinstance(port, LumpedTerminalPort)
            else port.reference_objects
        )
        expected = {
            "signal_entity_ids": list(port.signal_entity_ids),
            "reference_entity_ids": list(reference_ids),
            "signal_objects": names(port.signal_entity_ids),
            "reference_objects": names(reference_ids),
        }
        native = record.get("native", {})
        requested = (
            port.to_payload()
            if isinstance(port, LumpedTerminalPort)
            else {
                "index": port.index,
                "name": port.name,
                "side": port.side,
                "reference_objects": list(port.reference_objects),
                "signal_entity_ids": list(port.signal_entity_ids),
                "renormalize": False,
                "deembed_um": port.deembed_um,
            }
        )
        if record.get("requested") != requested:
            raise RuntimeError("body-first Terminal authored treatment differs")
        if isinstance(port, LumpedTerminalPort):
            support = next(
                item
                for item in geometry["lumped_supports"]
                if item["support_id"] == port.support_id
            )
            if (
                record.get("support") != support
                or record.get("object_name") != support["object_name"]
            ):
                raise RuntimeError("body-first Lumped support identity differs")
        if (
            record.get("index") != port.index
            or record.get("boundary") != port.name
            or record.get("source_binding") != expected
            or not native.get("signal_objects")
            or not set(native["signal_objects"]).issubset(expected["signal_objects"])
            or set(native.get("reference_conductors", ())) != global_references
        ):
            raise RuntimeError("body-first Terminal native ownership differs")
        terminal = native.get("saved_terminal", {})
        boundary = native.get("saved_boundary", {})
        expected_type = (
            "Lumped Port" if isinstance(port, LumpedTerminalPort) else "Wave Port"
        )
        if (
            boundary.get("BoundType") != expected_type
            or native.get("terminal_properties", {}).get("Port Name") != port.name
        ):
            raise RuntimeError("body-first Terminal native family differs")
        if [int(value) for value in boundary.get("Faces", ())] != [
            record.get("face_id")
        ]:
            raise RuntimeError("body-first Terminal native support face differs")
        edges = native.get("terminal_edges", [])
        if not edges or [item["edge_id"] for item in edges] != [
            int(value) for value in terminal.get("Edges", ())
        ]:
            raise RuntimeError("body-first Terminal native edge assignment differs")
        if set(item["object_name"] for item in edges) != set(native["signal_objects"]):
            raise RuntimeError("body-first Terminal contact ownership differs")
        properties = native.get("boundary_properties", {})
        terminal_properties = native.get("terminal_properties", {})
        renorm = port.renormalize if isinstance(port, LumpedTerminalPort) else False
        impedance = port.impedance_ohm if isinstance(port, LumpedTerminalPort) else 50
        if (
            properties.get("Type") != expected_type
            or properties.get("Renorm All Terminals") != renorm
            or properties.get("Deembed") != (not isinstance(port, LumpedTerminalPort))
            or terminal_properties.get("Terminal Renormalizing Impedance")
            != f"{impedance:g}ohm"
        ):
            raise RuntimeError(
                "body-first Terminal native electrical treatment differs"
            )


def _validate_terminal_native_evidence(ports: Any, spec: HfssDrivenSpec) -> None:
    if not isinstance(ports, list) or len(ports) != 2:
        raise RuntimeError("completed receipt has invalid terminal ports")
    for record, port in zip(ports, spec.ports, strict=True):
        if not isinstance(record, dict) or "unresolved_native" in record:
            raise RuntimeError(
                "completed receipt has unresolved terminal native evidence"
            )
        terminal_name = record.get("terminal_excitation")
        if (
            record.get("index") != port.index
            or record.get("boundary") != port.name
            or not isinstance(terminal_name, str)
            or not terminal_name
            or not isinstance(record.get("face_id"), int)
            or not isinstance(record.get("face_center_um"), list)
            or len(record["face_center_um"]) != 3
        ):
            raise RuntimeError("completed receipt terminal identity is invalid")
        requested = record.get("requested")
        if requested != {
            "reference_objects": list(port.reference_objects),
            "renormalize": False,
            "deembed_um": port.deembed_um,
        }:
            raise RuntimeError("completed receipt terminal request does not match spec")
        native = record.get("native")
        if not isinstance(native, dict):
            raise TypeError("completed receipt has no native terminal evidence")
        if (
            native.get("excitation_names")
            != [item.name for item in spec.ports[: port.index]]
            or native.get("terminal_names") != [terminal_name]
            or native.get("reference_conductors") != list(port.reference_objects)
            or not isinstance(native.get("reference_conductor_ids"), list)
            or not native["reference_conductor_ids"]
            or not all(
                isinstance(value, int) for value in native["reference_conductor_ids"]
            )
        ):
            raise RuntimeError("completed receipt terminal native identity mismatch")
        boundary = native.get("boundary_properties")
        terminal = native.get("terminal_properties")
        if boundary != {
            "Deembed": True,
            "Deembed Dist": f"{port.deembed_um:g}um",
            "Name": port.name,
            "Num Terminals": "1",
            "Renorm All Terminals": False,
            "Type": "Wave Port",
            "Wave Port Type": "Terminal",
        } or terminal != {
            "Name": terminal_name,
            "Port Name": port.name,
            "Terminal Renormalizing Impedance": "50ohm",
            "Type": "Terminal",
        }:
            raise RuntimeError("completed receipt terminal native property mismatch")


def _validate_diagnostics(root: Path, diagnostics: Any) -> None:
    if not isinstance(diagnostics, dict) or diagnostics.get("batch_log") != "batch.log":
        raise RuntimeError("completed receipt diagnostics are invalid")
    batch_log = _contained(root, "batch.log")
    if (
        diagnostics.get("present") is not True
        or not batch_log.is_file()
        or diagnostics.get("sha256") != file_sha256(batch_log)
        or not isinstance(diagnostics.get("physics_warnings"), list)
    ):
        raise RuntimeError("completed receipt diagnostics hash is invalid")


def _verified(
    root: Path, relative: Any, hashes: dict[str, Any], hash_key: str | None = None
) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError("receipt has an invalid relative output path")
    path = _contained(root, relative)
    expected = hashes.get(hash_key or relative)
    if (
        not isinstance(expected, str)
        or not path.is_file()
        or file_sha256(path) != expected
    ):
        raise RuntimeError(f"receipt output hash mismatch: {relative}")
    return path
