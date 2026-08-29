"""Strict extraction of native AEDT matrix-solver adaptive-pass evidence."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .spec import Q2dSpec, Q3dSpec
from .util import file_sha256


def read_q2d_convergence(run_dir: Path, spec: Q2dSpec) -> dict[str, Any]:
    """Parse and bind CG/RL convergence from AEDT 2024.2 result files."""
    return _read_convergence(
        run_dir,
        spec,
        solver="Q2D",
        problems=(
            ("cg", "CGConv", "de", "ee"),
            ("rl", "RLConv", "de", "ee"),
        ),
    )


def read_q3d_convergence(run_dir: Path, spec: Q3dSpec) -> dict[str, Any]:
    """Parse and bind capacitance/AC-RL convergence from AEDT 2024.2 files."""
    problems = [("capacitance", "CapConv", "delta", None)]
    if spec.solve_ac_rl:
        problems.append(("ac_rl", "ACRLConv", "delta", None))
    return _read_convergence(
        run_dir,
        spec,
        solver="Q3D",
        problems=tuple(problems),
        forbidden_block_names=() if spec.solve_ac_rl else ("ACRLConv",),
    )


def _read_convergence(
    run_dir: Path,
    spec: Q2dSpec | Q3dSpec,
    *,
    solver: str,
    problems: tuple[tuple[str, str, str, str | None], ...],
    forbidden_block_names: tuple[str, ...] = (),
) -> dict[str, Any]:
    root = run_dir.resolve()
    results_dir = _contained(root, f"{spec.project_name}.aedtresults", solver)
    asol_path = _contained(results_dir, f"{spec.design_name}.asol", solver)
    asol = _read(asol_path, solver)
    profile_names = set(re.findall(r"\bPRF\([^\n]*\bFile='([^']+\.profile)'", asol))
    if len(profile_names) != 1:
        raise RuntimeError(
            f"{solver} native solution does not identify exactly one profile"
        )
    profile_path = _contained(
        results_dir / f"{spec.design_name}.results", profile_names.pop(), solver
    )
    profile = _read(profile_path, solver)

    block_names = {name for _, name, _, _ in problems}
    blocks: dict[str, tuple[str, str]] = {}
    found_names: set[str] = set()
    for match in re.finditer(
        r"(?ms)^\s*\$begin '(?P<id>\d+)'\s*$"
        r"(?P<body>.*?)^\s*\$end '(?P=id)'\s*$",
        asol,
    ):
        body = match.group("body")
        name = _optional(body, r"(?m)^\s*ConvSetupName='([^']+)'\s*$")
        if name is not None:
            found_names.add(name)
        if name in block_names:
            if name in blocks:
                raise RuntimeError(f"{solver} native solution repeats {name}")
            blocks[name] = (match.group("id"), body)
    if set(blocks) != block_names:
        raise RuntimeError(
            f"{solver} native solution lacks exact matrix convergence blocks"
        )
    if set(forbidden_block_names) & found_names:
        raise RuntimeError(
            f"{solver} native solution contains a disabled convergence block"
        )

    evidence = {
        problem: _problem_evidence(
            solver=solver,
            block_id=blocks[name][0],
            body=blocks[name][1],
            profile=profile,
            expected_target=spec.run_control.convergence_percent,
            expected_maximum_passes=spec.run_control.maximum_passes,
            delta_field=delta_field,
            error_field=error_field,
        )
        for problem, name, delta_field, error_field in problems
    }
    return {
        "sources": {
            "asol": _source(asol_path, root),
            "profile": _source(profile_path, root),
        },
        **evidence,
    }


def _problem_evidence(
    *,
    solver: str,
    block_id: str,
    body: str,
    profile: str,
    expected_target: float,
    expected_maximum_passes: int,
    delta_field: str,
    error_field: str | None,
) -> dict[str, Any]:
    target = _float(_required(body, r"(?m)^\s*ConvTarget='([^']+)'\s*$"))
    maximum_passes = _int(_required(body, r"(?m)^\s*MaxPasses='([^']+)'\s*$"))
    if not math.isclose(target, expected_target, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"{solver} native convergence target does not match the spec"
        )
    if maximum_passes != expected_maximum_passes:
        raise RuntimeError(
            f"{solver} native convergence maximum passes do not match the spec"
        )

    pass_rows = re.findall(r"(?m)^\s*c\((.*)\)\s*$", body)
    if not pass_rows:
        raise RuntimeError(f"{solver} native convergence block has no adaptive passes")
    final = pass_rows[-1]
    final_pass = _int(_field(final, "p"))
    final_triangles = _int(_field(final, "tri"))
    final_delta = _float(_field(final, delta_field))
    final_error = _float(_field(final, error_field)) if error_field else None
    if (
        final_pass <= 0
        or final_pass > maximum_passes
        or final_triangles <= 0
        or final_delta < 0.0
        or (final_error is not None and final_error < 0.0)
    ):
        raise RuntimeError(f"{solver} native final convergence values are invalid")

    profile_block = _block(profile, block_id, solver)
    converged_text = "Adaptive Passes converged"
    not_converged_text = "Adaptive Passes did not converge"
    converged_count = profile_block.count(converged_text)
    not_converged_count = profile_block.count(not_converged_text)
    if (converged_count, not_converged_count) == (1, 0):
        converged = True
        stop_reason = converged_text
        if final_delta > target and not math.isclose(
            final_delta, target, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError(
                f"{solver} profile reports convergence above its native target"
            )
    elif (converged_count, not_converged_count) == (0, 1):
        converged = False
        stop_reason = not_converged_text
        if final_pass != maximum_passes:
            raise RuntimeError(
                f"{solver} non-converged profile stopped before maximum passes"
            )
    else:
        raise RuntimeError(f"{solver} native profile has ambiguous convergence status")

    result = {
        "target_percent": target,
        "converged": converged,
        "stop_reason": stop_reason,
        "final_pass": final_pass,
        "final_matrix_delta_percent": final_delta,
        "final_triangle_count": final_triangles,
    }
    if final_error is not None:
        result["final_error_percent"] = final_error
    return result


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


def _block(text: str, name: str, solver: str) -> str:
    begin = f"$begin '{name}'"
    end = f"$end '{name}'"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise RuntimeError(f"{solver} native profile lacks one exact {name} block")
    return text.split(begin, 1)[1].split(end, 1)[0]


def _required(text: str, pattern: str) -> str:
    values = re.findall(pattern, text)
    if len(values) != 1:
        raise RuntimeError(
            "AEDT native matrix convergence field is missing or repeated"
        )
    return values[0]


def _optional(text: str, pattern: str) -> str | None:
    values = re.findall(pattern, text)
    if len(values) > 1:
        raise RuntimeError("AEDT native matrix convergence field is repeated")
    return values[0] if values else None


def _field(row: str, name: str) -> str:
    return _required(row, rf"(?:^|,\s*){re.escape(name)}=([^,\s)]+)")


def _float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise RuntimeError(
            "AEDT native matrix convergence value is not numeric"
        ) from exc
    if not math.isfinite(result):
        raise RuntimeError("AEDT native matrix convergence value is not finite")
    return result


def _int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise RuntimeError(
            "AEDT native matrix convergence value is not an integer"
        ) from exc
    return result


__all__ = ["read_q2d_convergence", "read_q3d_convergence"]
