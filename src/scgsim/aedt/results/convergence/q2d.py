"""Q2D convergence result reader."""

from __future__ import annotations


from pathlib import Path

from typing import Any

from scgsim.aedt.results.convergence.common import _read_convergence

from scgsim.aedt.specs.q2d import Q2dSpec


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
