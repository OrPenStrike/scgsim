"""Strict HFSS adaptive-pass and native profile reader."""

from __future__ import annotations


import math

import re

from pathlib import Path

from typing import Any

from scgsim.aedt._io import file_sha256

from scgsim.aedt.specs.hfss import HfssEigenmodeSpec, HfssEprSpec, HfssSpec


def read_hfss_convergence(
    run_dir: Path, spec: HfssSpec, *, historical_profile: bool = False
) -> dict[str, Any]:
    """Parse and bind HFSS Driven or Eigenmode adaptive convergence."""
    root = run_dir.resolve()
    if isinstance(spec, (HfssEigenmodeSpec, HfssEprSpec)) and not historical_profile:
        return _read_eigenmode_convergence(root, spec)
    solver = f"HFSS {spec.mode}"
    results_dir = _contained(root, f"{spec.project_name}.aedtresults", solver)
    asol_path = _contained(results_dir, f"{spec.design_name}.asol", solver)
    asol = _read(asol_path, solver)
    if re.findall(r"(?m)^\s*SimSetupName='([^']+)'\s*$", asol) != [
        spec.run_control.setup_name
    ]:
        raise RuntimeError(f"{solver} native solution setup identity is invalid")
    profile_names = re.findall(r"\bP\([^\n]*\bFile='([^']+\.profile)'", asol)
    if len(profile_names) != 1:
        raise RuntimeError(
            f"{solver} native solution does not identify exactly one profile"
        )
    profile_path = _contained(
        results_dir / f"{spec.design_name}.results", profile_names[0], solver
    )
    profile = _read(profile_path, solver)

    passes = [
        int(value)
        for value in re.findall(r"(?m)^\s*Name='Adaptive Pass (\d+)'\s*$", profile)
    ]
    if not passes or passes != list(range(1, len(passes) + 1)):
        raise RuntimeError(f"{solver} native adaptive-pass sequence is invalid")
    metric, unit, target = (
        (
            "maximum_delta_frequency",
            "percent",
            spec.run_control.maximum_delta_frequency_percent,
        )
        if isinstance(spec, (HfssEigenmodeSpec, HfssEprSpec))
        else ("maximum_magnitude_delta_s", "ratio", spec.run_control.maximum_delta_s)
    )
    label = "Max Delta Freq. %" if unit == "percent" else "Max Mag. Delta S"
    deltas = [
        _float(value, solver)
        for value in re.findall(rf"\\'{re.escape(label)}\\',\s*([^,\s]+),", profile)
    ]
    if not deltas or deltas[-1] < 0:
        raise RuntimeError(f"{solver} native final convergence delta is unavailable")
    tetrahedra = re.findall(r"\\'Max solved tets\\',\s*(\d+),", profile)
    if len(tetrahedra) != 1 or int(tetrahedra[0]) <= 0:
        raise RuntimeError(f"{solver} native final tetrahedron count is invalid")

    converged_text = "Adaptive Passes converged"
    not_converged_text = "Adaptive Passes did not converge"
    status = (profile.count(converged_text), profile.count(not_converged_text))
    final_pass = passes[-1]
    final_delta = deltas[-1]
    if status == (1, 0):
        converged = True
        stop_reason = converged_text
    elif status == (0, 1):
        converged = False
        stop_reason = not_converged_text
    else:
        raise RuntimeError(f"{solver} native convergence status is ambiguous")

    return {
        "sources": {
            "asol": _source(asol_path, root),
            "profile": _source(profile_path, root),
        },
        "quantity": metric,
        "unit": unit,
        "target": target,
        "converged": converged,
        "stop_reason": stop_reason,
        "final_pass": final_pass,
        "final_delta": final_delta,
        "final_tetrahedron_count": int(tetrahedra[0]),
    }


def _read_eigenmode_convergence(
    root: Path, spec: HfssEigenmodeSpec | HfssEprSpec
) -> dict[str, Any]:
    solver = "HFSS Eigenmode"
    folder = "epr" if isinstance(spec, HfssEprSpec) else "eigenmode"
    path = _contained(root, f"results/{folder}/adaptive-convergence.prop", solver)
    native = _read(path, solver)

    def field(label: str) -> str:
        values = re.findall(
            rf"(?m)^{re.escape(label)}[ \t]*:[ \t]*([^\r\n]+)[ \t]*$", native
        )
        if len(values) != 1:
            raise RuntimeError(
                f"{solver} native convergence {label!r} is missing or ambiguous"
            )
        return values[0].strip()

    if field("Setup") != spec.run_control.setup_name:
        raise RuntimeError(f"{solver} native convergence setup identity is invalid")
    completed = _positive_int(field("Completed"), solver)
    if (
        _positive_int(field("Maximum"), solver) != spec.run_control.maximum_passes
        or _positive_int(field("Minimum"), solver) != spec.run_control.minimum_passes
        or _positive_int(field("Target Consecutive Passes"), solver)
        != spec.run_control.minimum_converged_passes
        or field("Criterion") != "Max Delta Freq. %"
    ):
        raise RuntimeError(f"{solver} native convergence setup controls do not match")
    if completed > spec.run_control.maximum_passes:
        raise RuntimeError(f"{solver} native convergence pass count is invalid")
    target = spec.run_control.maximum_delta_frequency_percent
    if not math.isclose(
        _float(field("Target"), solver), target, rel_tol=0, abs_tol=1e-12
    ):
        raise RuntimeError(f"{solver} native convergence target does not match")
    current = _float(field("Current"), solver)
    consecutive_text = field("Current Consecutive Passes")
    if not re.fullmatch(r"\d+", consecutive_text):
        raise RuntimeError(f"{solver} native convergence count is invalid")
    consecutive = int(consecutive_text)
    if consecutive > completed:
        raise RuntimeError(f"{solver} native convergence count is invalid")
    status = field("Converged")
    if status not in {"Yes", "No"}:
        raise RuntimeError(f"{solver} native convergence status is invalid")

    header = "Pass Number|Solved Elements|Max Delta Freq. %|"
    if native.count(header) != 1:
        raise RuntimeError(
            f"{solver} native convergence pass table is missing or ambiguous"
        )
    rows = native.split(header, 1)[1].strip().splitlines()
    parsed = []
    for row in rows:
        match = re.fullmatch(r"\s*(\d+)\|\s*(\d+)\|\s*([^|]+)\|\s*", row)
        if match is None:
            raise RuntimeError(f"{solver} native convergence pass row is malformed")
        parsed.append((int(match[1]), int(match[2]), match[3].strip()))
    if (
        len(parsed) != completed
        or [row[0] for row in parsed] != list(range(1, completed + 1))
        or any(row[1] <= 0 for row in parsed)
        or parsed[0][2] != "N/A"
    ):
        raise RuntimeError(f"{solver} native convergence pass sequence is invalid")
    deltas = [_float(row[2], solver) for row in parsed[1:]]
    if any(delta < 0 for delta in deltas) or current < 0:
        raise RuntimeError(f"{solver} native convergence delta is invalid")
    if completed > 1 and not math.isclose(
        current, deltas[-1], rel_tol=0, abs_tol=1e-12
    ):
        raise RuntimeError(f"{solver} native final convergence delta does not match")
    converged = status == "Yes"
    return {
        "sources": {"export_convergence": _source(path, root)},
        "quantity": "maximum_delta_frequency",
        "unit": "percent",
        "target": target,
        "converged": converged,
        "stop_reason": "Converged : Yes" if converged else "Converged : No",
        "final_pass": completed,
        "final_delta": current,
        "final_solved_element_count": parsed[-1][1],
    }


def _positive_int(value: str, solver: str) -> int:
    if not re.fullmatch(r"\d+", value) or int(value) <= 0:
        raise RuntimeError(f"{solver} native convergence count is invalid")
    return int(value)


def _source(path: Path, root: Path) -> dict[str, str | int]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _contained(root: Path, relative: str, solver: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise RuntimeError(f"{solver} native evidence path is invalid")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError(f"{solver} native evidence escapes the run directory")
    return path


def _read(path: Path, solver: str) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{solver} native convergence evidence is missing: {path}")
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except UnicodeError as exc:
        raise RuntimeError(
            f"{solver} native convergence evidence is not UTF-8: {path}"
        ) from exc


def _float(value: str, solver: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{solver} native convergence value is not numeric") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"{solver} native convergence value is not finite")
    return result
