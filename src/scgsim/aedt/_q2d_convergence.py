"""Strict extraction of native AEDT Q2D adaptive-pass evidence."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .spec import Q2dSpec
from .util import file_sha256


def read_q2d_convergence(run_dir: Path, spec: Q2dSpec) -> dict[str, Any]:
    """Parse and bind CG/RL convergence from AEDT 2024.2 result files."""
    root = run_dir.resolve()
    results_dir = _contained(root, f"{spec.project_name}.aedtresults")
    asol_path = _contained(results_dir, f"{spec.design_name}.asol")
    asol = _read(asol_path)
    profile_names = set(re.findall(r"\bPRF\([^\n]*\bFile='([^']+\.profile)'", asol))
    if len(profile_names) != 1:
        raise RuntimeError("Q2D native solution does not identify exactly one profile")
    profile_path = _contained(
        results_dir / f"{spec.design_name}.results", profile_names.pop()
    )
    profile = _read(profile_path)

    blocks: dict[str, tuple[str, str]] = {}
    for match in re.finditer(
        r"(?ms)^\s*\$begin '(?P<id>\d+)'\s*$"
        r"(?P<body>.*?)^\s*\$end '(?P=id)'\s*$",
        asol,
    ):
        body = match.group("body")
        name = _optional(body, r"(?m)^\s*ConvSetupName='([^']+)'\s*$")
        if name in {"CGConv", "RLConv"}:
            if name in blocks:
                raise RuntimeError(f"Q2D native solution repeats {name}")
            blocks[name] = (match.group("id"), body)
    if set(blocks) != {"CGConv", "RLConv"}:
        raise RuntimeError("Q2D native solution lacks exact CG/RL convergence blocks")

    evidence = {
        problem: _problem_evidence(
            block_id=blocks[name][0],
            body=blocks[name][1],
            profile=profile,
            expected_target=spec.run_control.convergence_percent,
            expected_maximum_passes=spec.run_control.maximum_passes,
        )
        for problem, name in (("cg", "CGConv"), ("rl", "RLConv"))
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
    block_id: str,
    body: str,
    profile: str,
    expected_target: float,
    expected_maximum_passes: int,
) -> dict[str, Any]:
    target = _float(_required(body, r"(?m)^\s*ConvTarget='([^']+)'\s*$"))
    maximum_passes = _int(_required(body, r"(?m)^\s*MaxPasses='([^']+)'\s*$"))
    if not math.isclose(target, expected_target, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Q2D native convergence target does not match the spec")
    if maximum_passes != expected_maximum_passes:
        raise RuntimeError(
            "Q2D native convergence maximum passes do not match the spec"
        )

    pass_rows = re.findall(r"(?m)^\s*c\((.*)\)\s*$", body)
    if not pass_rows:
        raise RuntimeError("Q2D native convergence block has no adaptive passes")
    final = pass_rows[-1]
    final_pass = _int(_field(final, "p"))
    final_triangles = _int(_field(final, "tri"))
    final_delta = _float(_field(final, "de"))
    final_error = _float(_field(final, "ee"))
    if (
        final_pass <= 0
        or final_pass > maximum_passes
        or final_triangles <= 0
        or final_delta < 0.0
        or final_error < 0.0
    ):
        raise RuntimeError("Q2D native final convergence values are invalid")

    profile_block = _block(profile, block_id)
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
                "Q2D profile reports convergence above its native target"
            )
    elif (converged_count, not_converged_count) == (0, 1):
        converged = False
        stop_reason = not_converged_text
        if final_pass != maximum_passes:
            raise RuntimeError(
                "Q2D non-converged profile stopped before maximum passes"
            )
    else:
        raise RuntimeError("Q2D native profile has ambiguous convergence status")

    return {
        "target_percent": target,
        "converged": converged,
        "stop_reason": stop_reason,
        "final_pass": final_pass,
        "final_matrix_delta_percent": final_delta,
        "final_error_percent": final_error,
        "final_triangle_count": final_triangles,
    }


def _source(path: Path, root: Path) -> dict[str, str | int]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _contained(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise RuntimeError("Q2D native evidence path is invalid")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError("Q2D native evidence escapes the run directory")
    return path


def _read(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Q2D native convergence evidence is missing: {path}")
    try:
        return path.read_text(encoding="utf-8", errors="strict")
    except UnicodeError as exc:
        raise RuntimeError(
            f"Q2D native convergence evidence is not UTF-8: {path}"
        ) from exc


def _block(text: str, name: str) -> str:
    begin = f"$begin '{name}'"
    end = f"$end '{name}'"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise RuntimeError(f"Q2D native profile lacks one exact {name} block")
    return text.split(begin, 1)[1].split(end, 1)[0]


def _required(text: str, pattern: str) -> str:
    values = re.findall(pattern, text)
    if len(values) != 1:
        raise RuntimeError("Q2D native convergence field is missing or repeated")
    return values[0]


def _optional(text: str, pattern: str) -> str | None:
    values = re.findall(pattern, text)
    if len(values) > 1:
        raise RuntimeError("Q2D native convergence field is repeated")
    return values[0] if values else None


def _field(row: str, name: str) -> str:
    return _required(row, rf"(?:^|,\s*){re.escape(name)}=([^,\s)]+)")


def _float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise RuntimeError("Q2D native convergence value is not numeric") from exc
    if not math.isfinite(result):
        raise RuntimeError("Q2D native convergence value is not finite")
    return result


def _int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise RuntimeError("Q2D native convergence value is not an integer") from exc
    return result


__all__ = ["read_q2d_convergence"]
