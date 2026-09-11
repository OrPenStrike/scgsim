"""Shared value preparation and ordered persistence for Palace workflows.

Simulation classes retain lifecycle and mutable-state ownership.  This module
only composes the existing preparation transforms and writes caller-provided
payloads in their declared order.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scgsim.sgb import VacuumRegionSpec
from scgsim.sgb.ground_bumps import _prepare_indium_ground_bump_fill

from ._staged import (
    RouteAThinFilm,
    apply_route_a_thin_film_to_stack,
    apply_vacuum_region_to_stack,
)


@dataclass(frozen=True)
class PreparedMeshInput:
    """Values prepared before a simulation applies its airbox and builds a mesh."""

    component: Any
    stack: Mapping[str, Any]
    indium_ground_bump_fill: Mapping[str, Any] | None
    materials: dict[str, Mapping[str, Any]] | None


def prepare_mesh_input(
    *,
    component: Any,
    stack: Mapping[str, Any],
    route: str,
    route_a_thin_film: RouteAThinFilm | None,
    vacuum_region: VacuumRegionSpec | None,
    indium_ground_bumps: Mapping[str, Any] | None,
) -> PreparedMeshInput:
    """Apply the common transforms while preserving their established order."""
    prepared_stack = stack
    if vacuum_region is not None:
        prepared_stack = apply_vacuum_region_to_stack(stack, vacuum_region)
    if route == "A":
        prepared_stack = apply_route_a_thin_film_to_stack(
            prepared_stack,
            source_stack=stack,
            variant=route_a_thin_film,
        )
    indium_fill = None
    prepared_component = component
    if indium_ground_bumps is not None:
        indium_fill = _prepare_indium_ground_bump_fill(
            component=component,
            stack=prepared_stack,
            **indium_ground_bumps,
        )
        prepared_component = indium_fill["component"]
        prepared_stack = indium_fill["stack"]
    prepared_materials = prepared_stack.get("materials")
    materials = None
    if isinstance(prepared_materials, Mapping):
        materials = {
            str(material_id): dict(material)
            for material_id, material in prepared_materials.items()
            if isinstance(material, Mapping)
        }
    return PreparedMeshInput(
        component=prepared_component,
        stack=prepared_stack,
        indium_ground_bump_fill=indium_fill,
        materials=materials,
    )


def persist_problem_files(
    *,
    metadata_files: Sequence[tuple[Path, Mapping[str, Any]]],
    config_path: Path,
    config: Mapping[str, Any],
) -> Path:
    """Replace ordered metadata files independently, then replace config last."""
    for path, payload in metadata_files:
        _atomic_json(path, payload)
    _atomic_json(config_path, config)
    return config_path


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = ["PreparedMeshInput", "persist_problem_files", "prepare_mesh_input"]
