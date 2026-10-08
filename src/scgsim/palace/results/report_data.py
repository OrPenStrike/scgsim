"""Detached Palace report records and result-file readers."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .resolve import (
    _compute_hash_entry,
    _confined_path,
    _extract_hash_map,
    _read_csv_table,
    _read_json,
    _validate_index_entries,
    _validate_surface_mask_table,
    _validate_surface_table,
)

_ITERATION_DIR = re.compile(r"^iteration(\d+)$")
_FREQ_HEADER = "Re{f} (GHz)"
_CAP_HEADER_PREFIX = "C[i]["
_INDEX_COLUMNS = {"m", "i"}
_SKIP_EIG_COLUMNS = {"Error (Bkwd.)", "Error (Abs.)"}
_ERROR_INDICATOR_TRACES = ("Norm", "Maximum", "Mean")
_SURFACE_TYPES = ("MA", "MS", "SA")


def _parent_has_physics(results: Path, problem: str) -> bool:
    if problem == "Eigenmode":
        return (results / "eig.csv").is_file()
    return (results / "terminal-C.csv").is_file()


def _same_primary_physics(left: AmrPassSnapshot, right: AmrPassSnapshot) -> bool:
    if left.frequencies_ghz is not None and right.frequencies_ghz is not None:
        return left.frequencies_ghz == right.frequencies_ghz
    if left.capacitance_matrix_f is not None and right.capacitance_matrix_f is not None:
        return left.capacitance_matrix_f == right.capacitance_matrix_f
    return False


def _same_masked_physics(
    left: tuple[SurfaceMaskEprSeriesSnapshot, ...] | None,
    right: tuple[SurfaceMaskEprSeriesSnapshot, ...] | None,
) -> bool:
    if left is None or right is None:
        return left is right
    return tuple(
        (snapshot.series_index, snapshot.series_kind, snapshot.records)
        for snapshot in left
    ) == tuple(
        (snapshot.series_index, snapshot.series_kind, snapshot.records)
        for snapshot in right
    )


def _problem_size_conflicts(left: AmrPassSnapshot, right: AmrPassSnapshot) -> bool:
    return any(
        left_value is not None and right_value is not None and left_value != right_value
        for left_value, right_value in (
            (left.degrees_of_freedom, right.degrees_of_freedom),
            (left.mesh_elements, right.mesh_elements),
        )
    )


def _surface_inverse_loss(records: Sequence[SurfaceEprRecord]) -> float | None:
    if any(record.loss_tangent is None for record in records):
        return None
    inverse_loss = sum(
        record.participation * float(record.loss_tangent) for record in records
    )
    return inverse_loss if math.isfinite(inverse_loss) and inverse_loss >= 0 else None


def _mapping_max(payload: Any) -> float | None:
    if isinstance(payload, dict):
        value = payload.get("Max")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return float(payload)
    return None


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _read_json(path)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return int(value)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("optional semantic identity must be a non-empty string.")
    return value


@dataclass(frozen=True)
class SurfaceEprRecord:
    """One solver-native surface result bound to structured SGB provenance."""

    index: int
    interface_type: str
    surface_id: str
    face_kind: str
    owner_semantic_ids: tuple[str, ...]
    net_id: str | None
    equipotential_id: str | None
    source_provenance: dict[str, Any]
    participation: float
    quality_factor: float
    loss_tangent: float | None


@dataclass(frozen=True)
class SurfaceEprSeriesSnapshot:
    """Surface-EPR values for one mode or excitation at one solver snapshot."""

    pass_index: int
    source: str
    series_index: int
    series_kind: Literal["mode", "excitation"]
    records: tuple[SurfaceEprRecord, ...]
    quality_factor_total: float | None
    t1_seconds: float | None
    loss_status: Literal["available", "unavailable_missing", "unavailable_nonfinite"]


@dataclass(frozen=True)
class SurfaceMaskEprRecord:
    """One native Inset-mask result, explicitly bound to its unmasked surface."""

    index: int
    baseline_index: int
    margin_index: int
    margin_um: float
    native_margin: float
    model_l0_m: float
    interface_type: str
    surface_id: str
    face_kind: str
    owner_semantic_ids: tuple[str, ...]
    net_id: str | None
    equipotential_id: str | None
    source_provenance: dict[str, Any]
    participation: float
    quality_factor: float
    energy_j: float
    loss_tangent: float | None
    contribution_status: Literal["available", "zero_contribution"]
    retained_area_status: Literal["unavailable"] = "unavailable"


@dataclass(frozen=True)
class SurfaceMaskEprSeriesSnapshot:
    """Native Inset-mask values for one mode or excitation at one source pass."""

    pass_index: int
    source: str
    series_index: int
    series_kind: Literal["mode", "excitation"]
    records: tuple[SurfaceMaskEprRecord, ...]


@dataclass(frozen=True)
class AmrPassSnapshot:
    """One solver snapshot from an AMR pass or the parent results folder."""

    pass_index: int
    source: str
    path: Path
    frequencies_ghz: tuple[float, ...] | None
    eig_columns: dict[str, tuple[float, ...]] | None
    port_epr: dict[str, tuple[float, ...]] | None
    capacitance_matrix_f: tuple[tuple[float, ...], ...] | None
    surface_epr: tuple[SurfaceEprSeriesSnapshot, ...] | None
    error_indicators: dict[str, float] | None
    error_norm: float | None
    degrees_of_freedom: int | None
    mesh_elements: int | None
    elapsed_total_s: float | None
    peak_node_memory_mb: float | None
    surface_mask_epr: tuple[SurfaceMaskEprSeriesSnapshot, ...] | None = None


@dataclass(frozen=True)
class PassCostRecord:
    """One AMR pass worth of solver cost, for this run and later accumulation."""

    pass_index: int
    source: str
    degrees_of_freedom: int | None
    mesh_elements: int | None
    elapsed_cumulative_s: float | None
    elapsed_pass_s: float | None
    seconds_per_million_dof: float | None
    peak_node_memory_mb: float | None


@dataclass(frozen=True)
class PalaceResultSelection:
    """The readable snapshot selected independently from the run outcome.

    ``selected_path`` is always relative to the run directory.
    """

    final_snapshot_status: Literal["readable", "missing", "unreadable"]
    selected_source: str | None
    selected_path: Path | None
    selected_pass_index: int | None
    reason: Literal[
        "final_snapshot",
        "latest_complete_iteration_after_failed_attempt",
        "no_complete_snapshot",
    ]
    integrity: Literal["receipt_bound", "observed_unsealed", "unavailable"]


@dataclass(frozen=True)
class PalaceFailureDiagnosis:
    """Evidence-based diagnosis that never changes the returned run status."""

    category: Literal[
        "config_compatibility",
        "out_of_memory",
        "signal_killed",
        "solver_error",
        "output_capture_error",
        "unknown",
    ]
    exit_code: int | None
    execution_stage: Literal["config_compatibility", "solver"] | None
    solver_invoked: bool | None
    preflight_exit_code: int | None
    solver_exit_code: int | None
    tee_exit_code: int | None
    summary: str
    evidence: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _CollectedSnapshots:
    passes: tuple[AmrPassSnapshot, ...]
    final_snapshot_status: Literal["readable", "missing", "unreadable"]


def _declares_surface_masks(
    config: dict[str, Any] | None, index_map: dict[str, Any] | None
) -> bool:
    boundaries = config.get("Boundaries") if isinstance(config, dict) else None
    postprocessing = (
        boundaries.get("Postprocessing") if isinstance(boundaries, dict) else None
    )
    dielectric = (
        postprocessing.get("Dielectric") if isinstance(postprocessing, dict) else None
    )
    if isinstance(dielectric, list) and any(
        isinstance(row, dict) and "Mask" in row for row in dielectric
    ):
        return True
    entries = index_map.get("entries") if isinstance(index_map, dict) else None
    return isinstance(entries, list) and any(
        isinstance(entry, dict) and "mask" in entry for entry in entries
    )


def _validate_inspection_receipt(
    root: Path,
    handoff: dict[str, Any],
    receipt: dict[str, Any] | None,
) -> frozenset[str]:
    if receipt is None:
        return frozenset()
    for field in ("handoff_id", "route", "problem"):
        value = receipt.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"returned receipt {field} must be a non-empty string.")
        if field in handoff and handoff.get(field) != value:
            raise ValueError(
                f"returned receipt {field} does not match handoff metadata."
            )
    if receipt.get("route_a_thin_film") != handoff.get("route_a_thin_film"):
        raise ValueError("returned receipt Route-A thin-film identity mismatch.")
    if not isinstance(receipt.get("status"), str) or not receipt["status"]:
        raise ValueError("returned receipt status must be a non-empty string.")
    _receipt_execution_state(receipt)

    input_entries = receipt.get("input_hashes")
    handoff_entries = handoff.get("hashes")
    if not isinstance(input_entries, list) or not isinstance(handoff_entries, list):
        raise TypeError("returned receipt and handoff input hashes must be lists.")
    receipt_inputs = _extract_hash_map(input_entries)
    handoff_inputs = _extract_hash_map(handoff_entries)
    if receipt_inputs != handoff_inputs:
        raise ValueError("returned receipt input hashes do not match handoff metadata.")
    for relative, expected in receipt_inputs.items():
        observed = _compute_hash_entry(_confined_path(root, relative))
        if observed != expected:
            raise ValueError(f"returned receipt input hash mismatch for {relative}.")

    output_entries = receipt.get("output_files")
    if not isinstance(output_entries, list):
        raise TypeError("returned receipt output_files must be a list.")
    verified: set[str] = set()
    seen: set[str] = set()
    for index, entry in enumerate(output_entries):
        relative = _verify_receipt_file_record(root, entry, f"output_files[{index}]")
        if relative in seen:
            raise ValueError(f"returned receipt contains duplicate path {relative!r}.")
        seen.add(relative)
        if entry["present"] is True:
            verified.add(relative)

    log = receipt.get("log")
    if not isinstance(log, dict):
        raise TypeError("returned receipt log must be a mapping.")
    log_relative = _verify_receipt_file_record(root, log, "log")
    if log["present"] is True:
        verified.add(log_relative)
    return frozenset(verified)


def _verify_receipt_file_record(root: Path, entry: Any, label: str) -> str:
    if not isinstance(entry, dict):
        raise TypeError(f"returned receipt {label} must be a mapping.")
    relative = entry.get("path")
    present = entry.get("present")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"returned receipt {label} path must be non-empty.")
    if not isinstance(present, bool):
        raise TypeError(f"returned receipt {label} present must be bool.")
    observed = _compute_hash_entry(_confined_path(root, relative))
    if present:
        size = entry.get("bytes")
        sha256 = entry.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"returned receipt {label} bytes must be non-negative.")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"returned receipt {label} sha256 must have 64 digits.")
        if observed != (size, sha256):
            raise ValueError(f"returned receipt output hash mismatch for {relative}.")
    elif entry.get("bytes") is not None or entry.get("sha256") is not None:
        raise ValueError(f"returned receipt missing {label} must not carry a hash.")
    elif observed != (None, None):
        raise ValueError(f"returned receipt missing output now exists: {relative}.")
    return relative


def _receipt_exit_code(receipt: dict[str, Any], field: str) -> int:
    value = receipt.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"returned receipt {field} must be an integer.")
    return value


def _receipt_execution_state(
    receipt: dict[str, Any],
) -> tuple[
    Literal["config_compatibility", "solver"] | None,
    bool | None,
    int | None,
    int | None,
    int,
    int,
]:
    exit_code = _receipt_exit_code(receipt, "exit_code")
    tee_exit_code = _receipt_exit_code(receipt, "tee_exit_code")
    stage = receipt.get("execution_stage")
    if stage is None:
        return (
            None,
            None,
            None,
            _receipt_exit_code(receipt, "solver_exit_code"),
            exit_code,
            tee_exit_code,
        )
    if stage not in {"config_compatibility", "solver"}:
        raise ValueError("returned receipt execution_stage is invalid.")
    solver_invoked = receipt.get("solver_invoked")
    if not isinstance(solver_invoked, bool):
        raise TypeError("returned receipt solver_invoked must be boolean.")
    preflight_exit_code = _receipt_exit_code(receipt, "preflight_exit_code")
    solver_exit_code = receipt.get("solver_exit_code")
    if stage == "config_compatibility":
        if solver_invoked or solver_exit_code is not None:
            raise ValueError(
                "config compatibility failure must record solver not invoked."
            )
    else:
        if not solver_invoked or preflight_exit_code != 0:
            raise ValueError(
                "solver execution requires a successful config compatibility preflight."
            )
        solver_exit_code = _receipt_exit_code(receipt, "solver_exit_code")
    return (
        stage,
        solver_invoked,
        preflight_exit_code,
        solver_exit_code,
        exit_code,
        tee_exit_code,
    )


def _result_selection(
    *,
    root: Path,
    problem: str,
    collected: _CollectedSnapshots,
    receipt_paths: frozenset[str],
) -> PalaceResultSelection:
    if not collected.passes:
        return PalaceResultSelection(
            final_snapshot_status=collected.final_snapshot_status,
            selected_source=None,
            selected_path=None,
            selected_pass_index=None,
            reason="no_complete_snapshot",
            integrity="unavailable",
        )
    selected = collected.passes[-1]
    reason = (
        "final_snapshot"
        if selected.source == "final"
        else "latest_complete_iteration_after_failed_attempt"
    )
    artifact_paths = _snapshot_artifact_paths(selected.path, problem)
    relative_paths = {path.relative_to(root).as_posix() for path in artifact_paths}
    integrity: Literal["receipt_bound", "observed_unsealed"] = (
        "receipt_bound"
        if relative_paths and relative_paths.issubset(receipt_paths)
        else "observed_unsealed"
    )
    return PalaceResultSelection(
        final_snapshot_status=collected.final_snapshot_status,
        selected_source=selected.source,
        selected_path=selected.path.relative_to(root),
        selected_pass_index=selected.pass_index,
        reason=reason,
        integrity=integrity,
    )


def _snapshot_artifact_paths(path: Path, problem: str) -> tuple[Path, ...]:
    names = (
        (
            "eig.csv",
            "port-EPR.csv",
            "surface-Q.csv",
            "error-indicators.csv",
            "palace.json",
        )
        if problem == "Eigenmode"
        else ("terminal-C.csv", "surface-Q.csv", "error-indicators.csv", "palace.json")
    )
    return tuple(
        candidate
        for name in (*names, "surface-mask-Q.csv", "surface-mask-energy.csv")
        if (candidate := path / name).is_file()
    )


def _failure_diagnosis(
    root: Path,
    receipt: dict[str, Any] | None,
    receipt_paths: frozenset[str],
) -> PalaceFailureDiagnosis | None:
    if receipt is None or receipt.get("status") != "failed":
        return None
    (
        execution_stage,
        solver_invoked,
        preflight_exit,
        solver_exit,
        exit_code,
        tee_exit,
    ) = _receipt_execution_state(receipt)
    evidence: list[dict[str, Any]] = [
        {
            "source": "returned_receipt",
            "exit_code": exit_code,
            "execution_stage": execution_stage,
            "solver_invoked": solver_invoked,
            "preflight_exit_code": preflight_exit,
            "solver_exit_code": solver_exit,
            "tee_exit_code": tee_exit,
        }
    ]
    oom_evidence = (
        None
        if execution_stage == "config_compatibility"
        else _slurm_oom_evidence(root, receipt, receipt_paths)
    )
    if execution_stage == "config_compatibility" and preflight_exit != 0:
        category = "config_compatibility"
        summary = (
            "Palace config compatibility preflight exited with status "
            f"{preflight_exit}; the solver was not invoked."
        )
    elif execution_stage == "config_compatibility" and tee_exit != 0:
        category = "output_capture_error"
        summary = (
            "Config compatibility output capture exited with status "
            f"{tee_exit}; the solver was not invoked."
        )
    elif execution_stage == "config_compatibility":
        category = "unknown"
        summary = (
            "The run failed during config compatibility preflight; the solver "
            "was not invoked."
        )
    elif oom_evidence is not None:
        evidence.append(oom_evidence)
        category = "out_of_memory"
        summary = "Slurm reported an out-of-memory event."
    elif solver_exit == 137 or exit_code == 137:
        category = "signal_killed"
        summary = (
            "The solver was killed with exit 137; out of memory is possible but "
            "not confirmed by scheduler evidence."
        )
    elif solver_exit != 0:
        category = "solver_error"
        summary = f"The Palace solver exited with status {solver_exit}."
    elif tee_exit != 0:
        category = "output_capture_error"
        summary = f"Solver output capture exited with status {tee_exit}."
    else:
        category = "unknown"
        summary = "The returned receipt reports failure without a classified cause."
    return PalaceFailureDiagnosis(
        category=category,
        exit_code=exit_code,
        execution_stage=execution_stage,
        solver_invoked=solver_invoked,
        preflight_exit_code=preflight_exit,
        solver_exit_code=solver_exit,
        tee_exit_code=tee_exit,
        summary=summary,
        evidence=tuple(evidence),
    )


def _slurm_oom_evidence(
    root: Path,
    receipt: dict[str, Any],
    receipt_paths: frozenset[str],
) -> dict[str, Any] | None:
    log = receipt.get("log")
    if not isinstance(log, dict):
        return None
    relative = log.get("path")
    if not isinstance(relative, str) or relative not in receipt_paths:
        return None
    path = _confined_path(root, relative)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lowered = line.lower()
            if "slurm" not in lowered:
                continue
            if (
                "oom_kill event" in lowered
                or "oom-kill event" in lowered
                or "out_of_memory" in lowered
            ):
                return {
                    "source": "hash_verified_log",
                    "path": relative,
                    "marker": "slurm_out_of_memory",
                    "sha256": log.get("sha256"),
                }
    return None


def _selection_payload(selection: PalaceResultSelection) -> dict[str, Any]:
    return {
        "final_snapshot_status": selection.final_snapshot_status,
        "selected_source": selection.selected_source,
        "selected_path": (
            None if selection.selected_path is None else str(selection.selected_path)
        ),
        "selected_pass_index": selection.selected_pass_index,
        "reason": selection.reason,
        "integrity": selection.integrity,
    }


def _failure_payload(
    failure: PalaceFailureDiagnosis | None,
) -> dict[str, Any] | None:
    if failure is None:
        return None
    return {
        "category": failure.category,
        "exit_code": failure.exit_code,
        "execution_stage": failure.execution_stage,
        "solver_invoked": failure.solver_invoked,
        "preflight_exit_code": failure.preflight_exit_code,
        "solver_exit_code": failure.solver_exit_code,
        "tee_exit_code": failure.tee_exit_code,
        "summary": failure.summary,
        "evidence": failure.evidence,
    }


def _collect_amr_passes(
    root: Path,
    problem: str,
    surface_bindings: tuple[dict[str, Any], ...] | None,
    surface_mask_bindings: tuple[dict[str, Any], ...] | None,
    *,
    failed_attempt: bool,
) -> _CollectedSnapshots:
    results = root / "results" / "palace"
    if not results.is_dir():
        return _CollectedSnapshots((), "missing")
    iteration_dirs = []
    for child in results.iterdir():
        match = _ITERATION_DIR.match(child.name)
        if match and child.is_dir():
            iteration_dirs.append((int(match.group(1)), child))
    iteration_dirs.sort()
    snapshots: list[AmrPassSnapshot] = []
    for index, path in iteration_dirs:
        snapshot = _load_optional_snapshot(
            pass_index=index - 1,
            source=path.name,
            path=path,
            problem=problem,
            surface_bindings=surface_bindings,
            surface_mask_bindings=surface_mask_bindings,
        )
        if snapshot is not None:
            snapshots.append(snapshot)
    if not _parent_has_physics(results, problem):
        return _CollectedSnapshots(tuple(snapshots), "missing")
    parent = _load_optional_snapshot(
        pass_index=max((index for index, _path in iteration_dirs), default=0),
        source="final",
        path=results,
        problem=problem,
        surface_bindings=surface_bindings,
        surface_mask_bindings=surface_mask_bindings,
    )
    if parent is None:
        return _CollectedSnapshots(tuple(snapshots), "unreadable")
    if snapshots and _same_primary_physics(snapshots[-1], parent):
        if failed_attempt and _problem_size_conflicts(snapshots[-1], parent):
            return _CollectedSnapshots(tuple(snapshots), "unreadable")
        parent = replace(parent, pass_index=snapshots[-1].pass_index)
        if _same_masked_physics(
            snapshots[-1].surface_mask_epr, parent.surface_mask_epr
        ):
            snapshots[-1] = replace(parent, pass_index=snapshots[-1].pass_index)
            return _CollectedSnapshots(tuple(snapshots), "readable")
    return _CollectedSnapshots((*snapshots, parent), "readable")


def _load_optional_snapshot(
    *,
    pass_index: int,
    source: str,
    path: Path,
    problem: str,
    surface_bindings: tuple[dict[str, Any], ...] | None,
    surface_mask_bindings: tuple[dict[str, Any], ...] | None,
) -> AmrPassSnapshot | None:
    try:
        return _load_snapshot(
            pass_index=pass_index,
            source=source,
            path=path,
            problem=problem,
            surface_bindings=surface_bindings,
            surface_mask_bindings=surface_mask_bindings,
        )
    except (OSError, TypeError, ValueError):
        return None


def _load_snapshot(
    *,
    pass_index: int,
    source: str,
    path: Path,
    problem: str,
    surface_bindings: tuple[dict[str, Any], ...] | None,
    surface_mask_bindings: tuple[dict[str, Any], ...] | None,
) -> AmrPassSnapshot:
    eig_columns = (
        _read_numeric_table_columns(path / "eig.csv")
        if problem == "Eigenmode"
        else None
    )
    port_epr = (
        _read_numeric_table_columns(path / "port-EPR.csv")
        if problem == "Eigenmode"
        else None
    )
    capacitance = (
        _read_capacitance(path / "terminal-C.csv")
        if problem == "Electrostatic"
        else None
    )
    error_indicators = _read_error_indicators(path / "error-indicators.csv")
    frequencies = None if eig_columns is None else eig_columns.get(_FREQ_HEADER)
    expected_rows = (
        len(frequencies)
        if frequencies is not None
        else len(capacitance)
        if capacitance is not None
        else 0
    )
    surface_epr = _read_surface_epr(
        path / "surface-Q.csv",
        problem=problem,
        pass_index=pass_index,
        source=source,
        frequencies_ghz=frequencies,
        expected_rows=expected_rows,
        bindings=surface_bindings,
        surface_count=len(surface_bindings or ()) + len(surface_mask_bindings or ()),
    )
    surface_mask_epr = _read_surface_mask_epr(
        path / "surface-mask-Q.csv",
        path / "surface-mask-energy.csv",
        problem=problem,
        pass_index=pass_index,
        source=source,
        expected_rows=expected_rows,
        bindings=surface_mask_bindings,
    )
    error_norm = None if error_indicators is None else error_indicators.get("Norm")
    palace_payload = _read_optional_json(path / "palace.json")
    problem_block = palace_payload.get("Problem") if palace_payload else None
    elapsed = None
    if palace_payload is not None:
        durations = palace_payload.get("ElapsedTime", {})
        if isinstance(durations, dict):
            duration_map = durations.get("Durations")
            if isinstance(duration_map, dict) and isinstance(
                duration_map.get("Total"), (int, float)
            ):
                elapsed = float(duration_map["Total"])
    return AmrPassSnapshot(
        pass_index=pass_index,
        source=source,
        path=path,
        frequencies_ghz=frequencies,
        eig_columns=eig_columns,
        port_epr=port_epr,
        capacitance_matrix_f=capacitance,
        surface_epr=surface_epr,
        error_indicators=error_indicators,
        error_norm=error_norm,
        degrees_of_freedom=_as_int(
            problem_block.get("DegreesOfFreedom")
            if isinstance(problem_block, dict)
            else None
        ),
        mesh_elements=_as_int(
            problem_block.get("MeshElements")
            if isinstance(problem_block, dict)
            else None
        ),
        elapsed_total_s=elapsed,
        peak_node_memory_mb=_mapping_max(
            palace_payload.get("PeakNodeMemoryMegabytes") if palace_payload else None
        ),
        surface_mask_epr=surface_mask_epr,
    )


def _read_numeric_table_columns(path: Path) -> dict[str, tuple[float, ...]] | None:
    if not path.is_file():
        return None
    table = _read_csv_table(path)
    columns: dict[str, tuple[float, ...]] = {}
    for header in table.headers:
        if header in _INDEX_COLUMNS:
            continue
        values: list[float] = []
        usable = True
        for row in table.rows:
            value = row.get(header)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                usable = False
                break
            values.append(float(value))
        if usable and values:
            columns[header] = tuple(values)
    return columns or None


def _read_capacitance(path: Path) -> tuple[tuple[float, ...], ...] | None:
    if not path.is_file():
        return None
    table = _read_csv_table(path)
    matrix: list[tuple[float, ...]] = []
    value_headers = [
        header for header in table.headers if header.startswith(_CAP_HEADER_PREFIX)
    ]
    if not value_headers:
        return None
    for row in table.rows:
        values: list[float] = []
        for header in value_headers:
            value = row.get(header)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None
            values.append(float(value))
        matrix.append(tuple(values))
    return tuple(matrix) if matrix else None


def _read_surface_bindings(
    index_map: dict[str, Any] | None,
) -> tuple[dict[str, Any], ...] | None:
    if index_map is None:
        return None
    try:
        _validate_index_entries(index_map)
        entries = [
            entry
            for entry in index_map["entries"]
            if entry["section"] == "Boundaries.Postprocessing.Dielectric"
        ]
        entries.sort(key=lambda entry: entry["index"])
        bindings: list[dict[str, Any]] = []
        for entry in entries:
            if "mask" in entry:
                continue
            metadata = entry["metadata"]
            interface_type = metadata.get("interface_type")
            surface_id = metadata.get("surface_id")
            face_kind = metadata.get("face_kind")
            owners = metadata.get("owner_semantic_ids")
            provenance = metadata.get("source_provenance")
            if interface_type not in _SURFACE_TYPES:
                raise ValueError("surface interface_type must be MA, MS, or SA.")
            if not isinstance(surface_id, str) or not surface_id:
                raise ValueError("surface_id must be a non-empty string.")
            if not isinstance(face_kind, str) or not face_kind:
                raise ValueError("face_kind must be a non-empty string.")
            if (
                not isinstance(owners, list)
                or not owners
                or not all(isinstance(owner, str) and owner for owner in owners)
            ):
                raise ValueError("owner_semantic_ids must be non-empty strings.")
            if not isinstance(provenance, dict):
                raise TypeError("source_provenance must be a mapping.")
            epr_spec = entry.get("epr_spec")
            loss_tangent = (
                epr_spec.get("loss_tangent") if isinstance(epr_spec, dict) else None
            )
            if (
                isinstance(epr_spec, dict)
                and epr_spec.get("schema_version") != "scgsim.palace.surface-film.v2"
                and loss_tangent == 0
            ):
                loss_tangent = None
            if loss_tangent is not None and (
                isinstance(loss_tangent, bool)
                or not isinstance(loss_tangent, (int, float))
                or not math.isfinite(float(loss_tangent))
                or loss_tangent < 0
            ):
                raise ValueError(
                    "surface loss_tangent must be finite and non-negative."
                )
            bindings.append(
                {
                    "index": entry["index"],
                    "interface_type": interface_type,
                    "surface_id": surface_id,
                    "face_kind": face_kind,
                    "owner_semantic_ids": tuple(owners),
                    "net_id": _optional_string(metadata.get("net_id")),
                    "equipotential_id": _optional_string(
                        metadata.get("equipotential_id")
                    ),
                    "source_provenance": provenance,
                    "loss_tangent": (
                        None if loss_tangent is None else float(loss_tangent)
                    ),
                }
            )
        return tuple(bindings) or None
    except (KeyError, TypeError, ValueError):
        return None


def _read_surface_mask_bindings(
    index_map: dict[str, Any] | None,
) -> tuple[dict[str, Any], ...] | None:
    if index_map is None:
        return None
    try:
        _validate_index_entries(index_map)
        bindings: list[dict[str, Any]] = []
        for entry in sorted(index_map["entries"], key=lambda item: item["index"]):
            if (
                entry.get("section") != "Boundaries.Postprocessing.Dielectric"
                or "mask" not in entry
            ):
                continue
            metadata = entry["metadata"]
            mask = entry["mask"]
            owners = metadata.get("owner_semantic_ids")
            provenance = metadata.get("source_provenance")
            if (
                metadata.get("interface_type") not in _SURFACE_TYPES
                or not isinstance(metadata.get("surface_id"), str)
                or not metadata["surface_id"]
                or not isinstance(metadata.get("face_kind"), str)
                or not metadata["face_kind"]
                or not isinstance(owners, list)
                or not owners
                or not all(isinstance(owner, str) and owner for owner in owners)
                or not isinstance(provenance, dict)
            ):
                raise ValueError("surface mask provenance is incomplete.")
            epr_spec = entry.get("epr_spec")
            loss_tangent = (
                epr_spec.get("loss_tangent") if isinstance(epr_spec, dict) else None
            )
            if (
                isinstance(epr_spec, dict)
                and epr_spec.get("schema_version") != "scgsim.palace.surface-film.v2"
                and loss_tangent == 0
            ):
                loss_tangent = None
            if loss_tangent is not None and (
                isinstance(loss_tangent, bool)
                or not isinstance(loss_tangent, (int, float))
                or not math.isfinite(float(loss_tangent))
                or loss_tangent < 0
            ):
                raise ValueError(
                    "surface mask loss_tangent must be finite and non-negative."
                )
            bindings.append(
                {
                    "index": int(entry["index"]),
                    "baseline_index": int(entry["baseline_index"]),
                    "margin_index": int(mask["margin_index"]),
                    "margin_um": float(mask["margin_um"]),
                    "native_margin": float(mask["native_margin"]),
                    "model_l0_m": float(mask["model_l0_m"]),
                    "interface_type": metadata["interface_type"],
                    "surface_id": metadata["surface_id"],
                    "face_kind": metadata["face_kind"],
                    "owner_semantic_ids": tuple(owners),
                    "net_id": _optional_string(metadata.get("net_id")),
                    "equipotential_id": _optional_string(
                        metadata.get("equipotential_id")
                    ),
                    "source_provenance": provenance,
                    "loss_tangent": (
                        None if loss_tangent is None else float(loss_tangent)
                    ),
                }
            )
        return tuple(bindings) or None
    except (KeyError, TypeError, ValueError):
        return None


def _read_surface_epr(
    path: Path,
    *,
    problem: str,
    pass_index: int,
    source: str,
    frequencies_ghz: tuple[float, ...] | None,
    expected_rows: int,
    bindings: tuple[dict[str, Any], ...] | None,
    surface_count: int | None = None,
) -> tuple[SurfaceEprSeriesSnapshot, ...] | None:
    if not path.is_file() or not bindings or expected_rows <= 0:
        return None
    try:
        table = _read_csv_table(path)
        _validate_surface_table(
            table,
            len(bindings) if surface_count is None else surface_count,
            expected_rows,
        )
        snapshots: list[SurfaceEprSeriesSnapshot] = []
        index_name = "m" if problem == "Eigenmode" else "i"
        for row in table.rows:
            series_index = int(row[index_name])
            records = tuple(
                SurfaceEprRecord(
                    **binding,
                    participation=float(row[f"p_surf[{binding['index']}]"]),
                    quality_factor=float(row[f"Q_surf[{binding['index']}]"]),
                )
                for binding in bindings
            )
            inverse_loss = _surface_inverse_loss(records)
            quality_factor = (
                1.0 / inverse_loss
                if inverse_loss is not None and inverse_loss > 0
                else None
            )
            frequency_hz = None
            if frequencies_ghz is not None and 0 < series_index <= len(frequencies_ghz):
                frequency_hz = frequencies_ghz[series_index - 1] * 1e9
            t1_seconds = (
                quality_factor / (2.0 * math.pi * frequency_hz)
                if quality_factor is not None
                and frequency_hz is not None
                and frequency_hz > 0
                else None
            )
            snapshots.append(
                SurfaceEprSeriesSnapshot(
                    pass_index=pass_index,
                    source=source,
                    series_index=series_index,
                    series_kind="mode" if problem == "Eigenmode" else "excitation",
                    records=records,
                    quality_factor_total=quality_factor,
                    t1_seconds=t1_seconds,
                    loss_status=(
                        "unavailable_missing"
                        if inverse_loss is None
                        else "unavailable_nonfinite"
                        if quality_factor is None
                        else "available"
                    ),
                )
            )
        return tuple(snapshots)
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _read_surface_mask_epr(
    q_path: Path,
    energy_path: Path,
    *,
    problem: str,
    pass_index: int,
    source: str,
    expected_rows: int,
    bindings: tuple[dict[str, Any], ...] | None,
) -> tuple[SurfaceMaskEprSeriesSnapshot, ...] | None:
    """Load both native mask tables as one source-local, all-or-nothing snapshot."""
    if (
        not q_path.is_file()
        or not energy_path.is_file()
        or not bindings
        or expected_rows <= 0
    ):
        return None
    try:
        q_table = _read_csv_table(q_path)
        energy_table = _read_csv_table(energy_path)
        _validate_surface_mask_table(
            q_table,
            problem=problem,
            expected_rows=expected_rows,
            entries=bindings,
        )
        _validate_surface_mask_table(
            energy_table,
            problem=problem,
            expected_rows=expected_rows,
            entries=bindings,
        )
        index_name = "m" if problem == "Eigenmode" else "i"
        snapshots: list[SurfaceMaskEprSeriesSnapshot] = []
        for q_row, energy_row in zip(q_table.rows, energy_table.rows, strict=True):
            series_index = int(q_row[index_name])
            if energy_row[index_name] != q_row[index_name]:
                raise ValueError("native mask Q and energy rows do not align.")
            records = tuple(
                SurfaceMaskEprRecord(
                    **binding,
                    participation=float(q_row[f"p_surf_mask[{binding['index']}]"]),
                    quality_factor=float(q_row[f"Q_surf_mask[{binding['index']}]"]),
                    energy_j=float(energy_row[f"E_surf_mask[{binding['index']}] (J)"]),
                    contribution_status=(
                        "zero_contribution"
                        if q_row[f"p_surf_mask[{binding['index']}]"] == 0
                        or energy_row[f"E_surf_mask[{binding['index']}] (J)"] == 0
                        else "available"
                    ),
                )
                for binding in bindings
            )
            snapshots.append(
                SurfaceMaskEprSeriesSnapshot(
                    pass_index=pass_index,
                    source=source,
                    series_index=series_index,
                    series_kind="mode" if problem == "Eigenmode" else "excitation",
                    records=records,
                )
            )
        return tuple(snapshots)
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _surface_snapshots(
    passes: Sequence[AmrPassSnapshot],
) -> tuple[SurfaceEprSeriesSnapshot, ...]:
    return tuple(snapshot for pass_ in passes for snapshot in (pass_.surface_epr or ()))


def _surface_mask_snapshots(
    passes: Sequence[AmrPassSnapshot],
) -> tuple[SurfaceMaskEprSeriesSnapshot, ...]:
    return tuple(
        snapshot for pass_ in passes for snapshot in (pass_.surface_mask_epr or ())
    )


def _read_error_indicators(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    table = _read_csv_table(path)
    if not table.rows:
        return None
    payload: dict[str, float] = {}
    for header in table.headers:
        value = table.rows[0].get(header)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            payload[header] = float(value)
    return payload or None


__all__ = [
    "AmrPassSnapshot",
    "PalaceFailureDiagnosis",
    "PalaceResultSelection",
    "PassCostRecord",
    "SurfaceEprRecord",
    "SurfaceEprSeriesSnapshot",
    "SurfaceMaskEprRecord",
    "SurfaceMaskEprSeriesSnapshot",
]
