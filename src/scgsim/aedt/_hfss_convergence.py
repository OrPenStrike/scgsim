"""Strict extraction of native HFSS adaptive-pass convergence evidence."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .spec import HfssEigenmodeSpec, HfssSpec
from .util import file_sha256


def read_hfss_convergence(run_dir: Path, spec: HfssSpec) -> dict[str, Any]:
    """Parse and bind HFSS Driven or Eigenmode adaptive convergence."""
    root = run_dir.resolve()
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
        if isinstance(spec, HfssEigenmodeSpec)
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
        if final_delta > target and not math.isclose(
            final_delta, target, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"{solver} reports convergence above its native target")
    elif status == (0, 1):
        converged = False
        stop_reason = not_converged_text
        if final_pass != spec.run_control.maximum_passes:
            raise RuntimeError(f"{solver} stopped before its configured maximum passes")
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


__all__ = ["read_hfss_convergence"]
