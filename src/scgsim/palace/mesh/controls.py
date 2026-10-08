"""Geometry-mesh controls, independent of Palace's finite-element order."""

from __future__ import annotations

from typing import Any

MESH_CONTROL_KEYS = (
    "algorithm_3d",
    "threads",
    "surface_threads",
    "geometry_order",
    "high_order_optimize",
)


def mesh_controls(
    *,
    algorithm_3d: str = "Delaunay",
    threads: int = 1,
    surface_threads: int = 1,
    geometry_order: int = 1,
    high_order_optimize: bool = True,
) -> dict[str, Any]:
    """Normalize supported native controls without imposing a research envelope."""
    if algorithm_3d not in {"Delaunay", "HXT"}:
        raise ValueError("algorithm_3d must be 'Delaunay' or 'HXT'.")
    for name, value in (("threads", threads), ("surface_threads", surface_threads)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer native Gmsh thread setting.")
    if (
        isinstance(geometry_order, bool)
        or not isinstance(geometry_order, int)
        or geometry_order < 1
    ):
        raise ValueError("geometry_order must be a positive integer.")
    if not isinstance(high_order_optimize, bool):
        raise TypeError("high_order_optimize must be a bool.")
    return dict(
        algorithm_3d=algorithm_3d,
        threads=threads,
        surface_threads=surface_threads,
        geometry_order=geometry_order,
        high_order_optimize=high_order_optimize,
    )
