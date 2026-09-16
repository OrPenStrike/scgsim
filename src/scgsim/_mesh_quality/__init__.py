"""Private implementation of the public Palace mesh-quality diagnostic."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

from .numerics import measure
from .reader import read_msh22
from .report import MeshQualityReport


def check_mesh_quality(
    mesh_path: str | Path, *, length_scale_m: float | None = None
) -> MeshQualityReport:
    """Read and measure an ASCII MSH 2.2 mesh without starting a backend."""

    if length_scale_m is not None:
        if (
            isinstance(length_scale_m, bool)
            or not isinstance(length_scale_m, Real)
            or not math.isfinite(float(length_scale_m))
            or float(length_scale_m) <= 0.0
        ):
            raise ValueError("length_scale_m must be a finite positive real number or None")
        length_scale_m = float(length_scale_m)
    mesh = read_msh22(mesh_path)
    metrics = measure(mesh, length_scale_m)
    return MeshQualityReport.create(mesh, metrics, length_scale_m)


__all__ = ["MeshQualityReport", "check_mesh_quality"]
