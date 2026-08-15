"""Shared scalar aliases and constants for SGB model records.

This module is the small common vocabulary used by stage-specific record
modules. It must stay free of dataclasses so those modules can depend on it
without creating model import cycles.
"""

from __future__ import annotations

from os import PathLike
from typing import Literal

PathInput = str | PathLike[str]
RUN_METADATA_DIR = "metadata"
SEMANTIC_GEOMETRY_METADATA_DIR = "semantic_geometry"

RouteLiteral = Literal["A", "B", "C"]
InterfaceKindLiteral = Literal["MS", "MA", "SA", "AA", "MM", "SS"]
DimensionLiteral = Literal[1, 2, 3]
SurfaceOrientationLiteral = Literal["forward", "reversed"]
Coordinate = tuple[float, float]
PolygonRing = tuple[Coordinate, ...]
Vector3D = tuple[float, float, float]
GmshDimTag = tuple[int, int]

ConductorPartRoleLiteral = Literal[
    "face_metal",
    "contact_pad",
    "bump_body",
    "airbridge_post",
    "airbridge_deck",
]
HIGH_COUNT_LOCAL_CONDUCTOR_PART_ROLES = frozenset(
    ("bump_body", "airbridge_post", "airbridge_deck")
)
ConductorRepresentationLiteral = Literal[
    "material_volume",
    "cutout_boundary_shell",
    "surface_sheet",
]
ROUTE_ALLOWED_REPRESENTATIONS: dict[
    RouteLiteral,
    frozenset[ConductorRepresentationLiteral],
] = {
    "A": frozenset(("surface_sheet", "cutout_boundary_shell")),
    "B": frozenset(("cutout_boundary_shell",)),
    "C": frozenset(("material_volume",)),
}
CurveKindLiteral = Literal["line_segment"]
CurveOrientationLiteral = Literal[1, -1]
SurfaceLoopRoleLiteral = Literal["outer", "hole"]
TagSourceKindLiteral = Literal["surface", "volume"]
SolverUseLiteral = Literal["solver_active"]
DEFAULT_INTERFACE_SOLVER_USE: dict[InterfaceKindLiteral, SolverUseLiteral] = {
    "MS": "solver_active",
    "MA": "solver_active",
    "SA": "solver_active",
    "AA": "solver_active",
    "MM": "solver_active",
    "SS": "solver_active",
}
