"""Solver-neutral semantic scene records and deterministic palette selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

_MODES = ("materials", "boundaries", "surface_epr", "mesh")

_MODE_TITLES = {
    "materials": "Materials",
    "boundaries": "Boundaries",
    "surface_epr": "Surface EPR",
    "mesh": "Mesh",
}

_VIEWS = (
    ("plus-x", "+X", (1.0, 0.0, 0.0), None),
    ("minus-x", "-X", (-1.0, 0.0, 0.0), None),
    ("plus-y", "+Y", (0.0, 1.0, 0.0), None),
    ("minus-y", "-Y", (0.0, -1.0, 0.0), None),
    ("plus-z", "+Z", (0.0, 0.0, 1.0), None),
    ("minus-z", "-Z", (0.0, 0.0, -1.0), None),
    ("above-ne", "Above +X +Y", (1.0, 1.0, 1.0), None),
    ("above-sw", "Above -X -Y", (-1.0, -1.0, 1.0), None),
    ("below-se", "Below +X -Y", (1.0, -1.0, -1.0), None),
    ("below-nw", "Below -X +Y", (-1.0, 1.0, -1.0), None),
    (
        "x-center-clip",
        "X-center visualization clip — not solver boundary",
        (1.0, 1.0, 1.0),
        ("x", 1.0),
    ),
    (
        "y-center-clip",
        "Y-center visualization clip — not solver boundary",
        (1.0, 1.0, 1.0),
        ("y", 1.0),
    ),
)

_FIXED_COLORS = {
    "Ground": "#3A3A3A",
    "PEC": "#3A3A3A",
    "LumpedPort": "#CC3377",
    "MA": "#E69F00",
    "MS": "#009E73",
    "SA": "#0072B2",
    "MS_MA": "#7B2CBF",
    "unassigned": "#D3D3D3",
}

_OTHER_COLORS = (
    "#1F77B4",
    "#AEC7E8",
    "#FF7F0E",
    "#FFBB78",
    "#2CA02C",
    "#98DF8A",
    "#D62728",
    "#FF9896",
    "#9467BD",
    "#C5B0D5",
    "#8C564B",
    "#C49C94",
    "#E377C2",
    "#F7B6D2",
    "#7F7F7F",
    "#C7C7C7",
    "#BCBD22",
    "#DBDB8D",
    "#17BECF",
    "#9EDAE5",
)


@dataclass(frozen=True)
class _Part:
    dataset: Any
    semantic_id: str
    label: str
    role: str
    color: str
    opacity: float
    count: int
    show_edges: bool = False


@dataclass(frozen=True)
class _Mode:
    parts: tuple[_Part, ...] = ()
    unavailable_reason: str | None = None


def _with_scene_palette(mode: _Mode) -> _Mode:
    if mode.unavailable_reason is not None:
        return mode
    identities = sorted(
        {part.semantic_id for part in mode.parts if part.role not in _FIXED_COLORS}
    )
    assigned = {
        identity: _OTHER_COLORS[index % len(_OTHER_COLORS)]
        for index, identity in enumerate(identities)
    }
    return replace(
        mode,
        parts=tuple(
            replace(
                part,
                color=(
                    _FIXED_COLORS[part.role]
                    if part.role in _FIXED_COLORS
                    else assigned[part.semantic_id]
                ),
            )
            for part in mode.parts
        ),
    )


def _color(identity: str) -> str:
    if identity in _FIXED_COLORS:
        return _FIXED_COLORS[identity]
    return _OTHER_COLORS[_palette_index(identity)]


def _palette_index(identity: str) -> int:
    digest = hashlib.sha256(identity.encode()).digest()
    return int.from_bytes(digest[:2], "big") % len(_OTHER_COLORS)
