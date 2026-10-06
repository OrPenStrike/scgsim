"""Shared surface-first native construction preserving canonical shell faces.

Raw CAD orientation is retained. Detached outward directions are compiler
observations, not native face repair or a positive-mass eligibility policy.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def add_surface_first_volume(gmsh: Any, shell_surfaces: Sequence[Sequence[int]]) -> int:
    """Sew canonical shells and construct their solid without replacing faces.

    Native auto-fixing can replace shared faces, so it is disabled for this
    construction and restored even when a native operation raises. Native
    errors propagate unchanged; no solid repair or alternate geometry follows.
    """
    previous = gmsh.option.getNumber("Geometry.OCCAutoFix")
    try:
        gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
        shell_tags = [gmsh.model.occ.addSurfaceLoop(tags, sewing=True)
                      for tags in shell_surfaces]
        return gmsh.model.occ.addVolume(shell_tags)
    finally:
        gmsh.option.setNumber("Geometry.OCCAutoFix", previous)
