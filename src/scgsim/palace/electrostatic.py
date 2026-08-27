"""Thin staged Route-A/Route-B electrostatic handoff API; it never runs Palace."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from scgsim.sgb import VacuumRegionSpec
from scgsim.sgb.ground_bumps import _prepare_indium_ground_bump_fill

from ._config import (
    TerminalBinding,
    build_electrostatic_config,
    configure_numerical_controls,
)
from ._epr import normalize_surface_epr_specs
from ._mesh import MeshBuildResult, build_route_mesh
from ._staged import (
    RouteAThinFilm,
    apply_airbox_to_stack,
    apply_route_a_thin_film_to_stack,
    apply_vacuum_region_to_stack,
    normalize_route_a_thin_film,
    validate_non_negative_int,
    validate_nonempty_string,
    validate_positive_number,
)
from .handoff import HandoffPlan, prepare_handoff


@dataclass
class ElectrostaticSim:
    """Prepare explicit SGB geometry and manual Palace electrostatic handoff files."""

    component: Any | None = None
    stack: Mapping[str, Any] | None = None
    output_dir: Path | None = None
    airbox: dict[str, float] = field(default_factory=dict)
    route: Literal["A", "B"] = "B"
    route_a_thin_film: RouteAThinFilm | None = None
    terminals: list[TerminalBinding] = field(default_factory=list)
    surface_epr_specs: dict[str, dict[str, Any]] | None = None
    save_fields: int = 0
    unassigned_conductor_policy: Literal["ground", "error"] = "ground"
    exterior_boundary_policy: Literal["none", "ground"] = "none"
    numerical: dict[str, Any] = field(default_factory=configure_numerical_controls)
    _materials: dict[str, Mapping[str, Any]] | None = field(default=None, init=False)
    vacuum_region: VacuumRegionSpec | None = None
    indium_ground_bumps: dict[str, Any] | None = None
    _mesh_result: MeshBuildResult | None = field(default=None, init=False)
    config_path: Path | None = field(default=None, init=False)
    handoff_plan: HandoffPlan | None = field(default=None, init=False)

    def _invalidate_mesh(self) -> None:
        self._mesh_result = None
        self.config_path = None
        self.handoff_plan = None

    def _invalidate_config(self) -> None:
        self.config_path = None
        self.handoff_plan = None

    def set_geometry(self, component: Any) -> None:
        if component is None or not callable(getattr(component, "write_gds", None)):
            raise TypeError("component must provide write_gds(path).")
        self.component = component
        self._invalidate_mesh()

    def set_stack(self, stack: Mapping[str, Any] | str | Path) -> None:
        payload = _load_stack(stack)
        materials = payload.get("materials")
        if not isinstance(materials, Mapping) or not materials:
            raise ValueError(
                "stack must define a non-empty explicit materials mapping."
            )
        resolved = {
            str(key): value
            for key, value in materials.items()
            if isinstance(value, Mapping)
        }
        if len(resolved) != len(materials):
            raise TypeError("every explicit stack material must be a mapping.")
        _validate_stack_material_kinds(payload, resolved)
        self.stack = payload
        self._materials = resolved
        self._invalidate_mesh()

    def set_output_dir(self, path: str | Path) -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError("output path must be a path string or Path.")
        self.output_dir = Path(path).expanduser().resolve()
        self._invalidate_mesh()

    def set_airbox(
        self,
        *,
        margin_x: float,
        margin_y: float,
        z_above: float | None = None,
        z_below: float | None = None,
    ) -> None:
        if self.vacuum_region is not None:
            raise ValueError(
                "set_airbox is mutually exclusive with set_vacuum_region()."
            )
        self.airbox = {
            "margin_x": validate_positive_number(margin_x, "margin_x"),
            "margin_y": validate_positive_number(margin_y, "margin_y"),
        }
        if z_above is not None:
            self.airbox["z_above"] = validate_positive_number(z_above, "z_above")
        if z_below is not None:
            self.airbox["z_below"] = validate_positive_number(z_below, "z_below")
        self._invalidate_mesh()

    def set_vacuum_region(
        self,
        padding: float | list[float] | tuple[float, ...] | Mapping[str, Any] = 0.0,
    ) -> None:
        """Set an auto-generated vacuum envelope around non-vacuum solution regions."""
        if self.airbox:
            raise ValueError(
                "set_vacuum_region is mutually exclusive with set_airbox()."
            )
        self.vacuum_region = VacuumRegionSpec.from_padding(padding)
        self._invalidate_mesh()

    def set_indium_ground_bumps(
        self, *, fill: bool, fill_pitch_um: float, fill_clearance_um: float
    ) -> None:
        """Request public-PDK authored-plus-ground-fill bumps for the next mesh."""
        if not isinstance(fill, bool):
            raise TypeError("fill must be a bool.")
        self.indium_ground_bumps = {
            "fill": fill,
            "fill_pitch_um": validate_positive_number(fill_pitch_um, "fill_pitch_um"),
            "fill_clearance_um": _non_negative_number(
                fill_clearance_um, "fill_clearance_um"
            ),
        }
        self._invalidate_mesh()

    def add_terminal(self, name: str, *, net_id: str) -> None:
        binding = TerminalBinding(
            name=validate_nonempty_string(name, "terminal name"),
            net_id=validate_nonempty_string(net_id, "net_id"),
        )
        self.terminals = [item for item in self.terminals if item.name != binding.name]
        self.terminals.append(binding)
        self._invalidate_config()

    def set_surface_epr(
        self,
        *,
        representation: str,
        specs: Mapping[str, Mapping[str, Any]],
        route_a_thin_film: RouteAThinFilm | None = None,
    ) -> None:
        """Configure Surface EPR and the required Route-A thin-film lowering."""
        route = (
            representation.strip().upper() if isinstance(representation, str) else ""
        )
        if route not in {"A", "B"}:
            raise ValueError("Surface EPR representation must be 'A' or 'B'.")
        if self._materials is None:
            raise ValueError("set_stack() must run before set_surface_epr().")
        normalized_thin_film = normalize_route_a_thin_film(route, route_a_thin_film)
        self.route = route  # type: ignore[assignment]
        self.route_a_thin_film = normalized_thin_film
        self.surface_epr_specs = normalize_surface_epr_specs(
            specs, materials=self._materials
        )
        self._invalidate_mesh()

    def set_electrostatic(
        self,
        *,
        save_fields: int = 0,
        unassigned_conductor_policy: Literal["ground", "error"] = "ground",
        exterior_boundary_policy: Literal["none", "ground"] = "none",
    ) -> None:
        if unassigned_conductor_policy not in {"ground", "error"}:
            raise ValueError("unassigned_conductor_policy must be 'ground' or 'error'.")
        if exterior_boundary_policy not in {"none", "ground"}:
            raise ValueError("exterior_boundary_policy must be 'none' or 'ground'.")
        if (
            exterior_boundary_policy == "ground"
            and unassigned_conductor_policy != "error"
        ):
            raise ValueError(
                "exterior ground requires unassigned_conductor_policy='error'."
            )
        self.save_fields = validate_non_negative_int(save_fields, "save_fields")
        self.unassigned_conductor_policy = unassigned_conductor_policy
        self.exterior_boundary_policy = exterior_boundary_policy
        self._invalidate_config()

    def set_mesh(self, *, refined_mesh_size: float, max_mesh_size: float) -> None:
        """Set the mesh-only numerical controls for the next mesh build."""
        numerical = configure_numerical_controls(
            **{
                **self.numerical,
                "refined_mesh_size": refined_mesh_size,
                "max_mesh_size": max_mesh_size,
            }
        )
        if (
            numerical["refined_mesh_size"] != self.numerical["refined_mesh_size"]
            or numerical["max_mesh_size"] != self.numerical["max_mesh_size"]
        ):
            self.numerical = numerical
            self._invalidate_mesh()

    def set_numerical(
        self,
        *,
        order: int = 1,
        tolerance: float = 1e-6,
        max_iterations: int = 400,
        solver_type: str = "Default",
        preconditioner: str = "Default",
        device: str = "CPU",
        refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None,
        amr_max_passes: int = 0,
        amr_nonconformal: bool = False,
        amr_tolerance: float = 1e-2,
        amr_update_fraction: float | None = None,
        save_adapt_iterations: bool | None = None,
        estimator_mg: bool | None = None,
        output_paraview: bool | None = None,
        output_grid_function: bool | None = None,
    ) -> None:
        """Set solver, FEM, AMR, and output controls for the next config write.

        ``amr_max_passes`` is Palace ``MaxIts``. ``amr_nonconformal`` is Palace
        ``Model.Refinement.Nonconformal`` and defaults to ``False``. Set
        ``True`` only to opt into nonconformal AMR.
        """
        if (refined_mesh_size is None) != (max_mesh_size is None):
            raise ValueError(
                "refined_mesh_size and max_mesh_size must be provided together."
            )
        previous_mesh_sizes = {
            "refined_mesh_size": self.numerical["refined_mesh_size"],
            "max_mesh_size": self.numerical["max_mesh_size"],
        }
        mesh_sizes = previous_mesh_sizes
        if refined_mesh_size is not None:
            mesh_sizes = {
                "refined_mesh_size": refined_mesh_size,
                "max_mesh_size": max_mesh_size,
            }
        self.numerical = configure_numerical_controls(
            order=order,
            tolerance=tolerance,
            max_iterations=max_iterations,
            solver_type=solver_type,
            preconditioner=preconditioner,
            device=device,
            **mesh_sizes,
            amr_max_passes=amr_max_passes,
            amr_nonconformal=amr_nonconformal,
            amr_tolerance=amr_tolerance,
            amr_update_fraction=amr_update_fraction,
            save_adapt_iterations=save_adapt_iterations,
            estimator_mg=estimator_mg,
            output_paraview=output_paraview,
            output_grid_function=output_grid_function,
        )
        if (
            self.numerical["refined_mesh_size"]
            != previous_mesh_sizes["refined_mesh_size"]
            or self.numerical["max_mesh_size"] != previous_mesh_sizes["max_mesh_size"]
        ):
            self._invalidate_mesh()
        else:
            self._invalidate_config()

    def mesh(self) -> Path:
        self._invalidate_mesh()
        if self.component is None or self.stack is None or self.output_dir is None:
            raise ValueError(
                "set_geometry(), set_stack(), and set_output_dir() must run before mesh()."
            )
        if self.surface_epr_specs is None:
            raise ValueError("set_surface_epr() must run before mesh().")
        prepared_stack = self.stack
        if self.vacuum_region is not None:
            prepared_stack = apply_vacuum_region_to_stack(
                self.stack,
                self.vacuum_region,
            )
        if self.route == "A":
            prepared_stack = apply_route_a_thin_film_to_stack(
                prepared_stack,
                source_stack=self.stack,
                variant=self.route_a_thin_film,
            )
        indium_fill = None
        mesh_component = self.component
        if self.indium_ground_bumps is not None:
            indium_fill = _prepare_indium_ground_bump_fill(
                component=self.component,
                stack=prepared_stack,
                **self.indium_ground_bumps,
            )
            mesh_component = indium_fill["component"]
            prepared_stack = indium_fill["stack"]
        prepared_materials = prepared_stack.get("materials")
        if isinstance(prepared_materials, Mapping):
            self._materials = {
                str(material_id): dict(material)
                for material_id, material in prepared_materials.items()
                if isinstance(material, Mapping)
            }
        self._mesh_result = build_route_mesh(
            component=mesh_component,
            stack=apply_airbox_to_stack(prepared_stack, self.airbox),
            route=self.route,
            output_dir=self.output_dir,
            refined_mesh_size=self.numerical["refined_mesh_size"],
            max_mesh_size=self.numerical["max_mesh_size"],
            indium_ground_bump_fill=indium_fill,
        )
        return self._mesh_result.mesh_path

    def write_config(self) -> Path:
        self._invalidate_config()
        if (
            self._mesh_result is None
            or self._materials is None
            or self.surface_epr_specs is None
        ):
            raise ValueError(
                "set_stack(), set_surface_epr(), and mesh() must run before write_config()."
            )
        if not self._mesh_result.mesh_path.is_file():
            raise FileNotFoundError(
                "current palace.msh is missing; mesh() must be rerun."
            )
        result = build_electrostatic_config(
            groups=self._mesh_result.groups,
            terminals=self.terminals,
            materials=self._materials,
            save_fields=self.save_fields,
            unassigned_conductor_policy=self.unassigned_conductor_policy,
            exterior_boundary_policy=self.exterior_boundary_policy,
            mesh_path=self._mesh_result.mesh_path,
            surface_epr_specs=self.surface_epr_specs,
            numerical=self.numerical,
        )
        metadata_dir = self._mesh_result.output_dir / "metadata"
        index_payload = {
            "schema_version": 1,
            "entries": [
                *result.terminal_index_map,
                *result.domain_energy_index_map,
                *result.surface_epr_index_map,
            ],
        }
        material_payload = {
            "schema_version": 1,
            "solution_volumes": result.material_resolution,
        }
        config_path = self._mesh_result.output_dir / "config.json"
        _atomic_json(metadata_dir / "palace_index_map.json", index_payload)
        _atomic_json(metadata_dir / "palace_material_resolution.json", material_payload)
        _atomic_json(metadata_dir / "palace_numerical_controls.json", self.numerical)
        _atomic_json(config_path, result.config)
        self.config_path = config_path
        return config_path

    def prepare_handoff(
        self,
        *,
        profile: str,
        executable: str,
        resources: Mapping[str, Any] | None = None,
        setup_commands: tuple[str, ...] = (),
        petsc_options: tuple[str, ...] = (),
    ) -> HandoffPlan:
        if self._mesh_result is None or self.config_path is None:
            raise ValueError(
                "mesh() and write_config() must run before prepare_handoff()."
            )
        self.handoff_plan = prepare_handoff(
            profile=profile,
            mesh_result=self._mesh_result,
            config_path=self.config_path,
            executable=executable,
            resources=resources,
            setup_commands=setup_commands,
            petsc_options=petsc_options,
        )
        return self.handoff_plan


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _non_negative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number.")
    return result


def _load_stack(stack: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(stack, Mapping):
        return dict(stack)
    if isinstance(stack, Path):
        if not stack.is_file():
            raise FileNotFoundError(f"stack JSON path does not exist: {stack}")
        payload = json.loads(stack.read_text(encoding="utf-8"))
    elif isinstance(stack, str):
        stripped = stack.strip()
        if stripped.startswith(("{", "[")):
            payload = json.loads(stripped)
        else:
            path = Path(stack)
            try:
                exists = path.is_file()
            except OSError as exc:
                raise ValueError(
                    "stack string must be JSON text or an existing JSON path."
                ) from exc
            if not exists:
                raise FileNotFoundError(
                    "stack string must be JSON text or an existing JSON path."
                )
            payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise TypeError("stack must be a mapping, JSON string, or JSON path.")
    if not isinstance(payload, Mapping):
        raise TypeError("stack payload must be a mapping.")
    return dict(payload)


def _validate_stack_material_kinds(
    stack: Mapping[str, Any], materials: Mapping[str, Mapping[str, Any]]
) -> None:
    def material_kind(material_id: Any, owner: str) -> str:
        if not isinstance(material_id, str) or material_id not in materials:
            raise ValueError(f"{owner} must reference an explicit stack material_id.")
        kind = materials[material_id].get("kind")
        if kind not in {"vacuum", "dielectric", "conductor"}:
            raise ValueError(
                f"material {material_id!r} requires kind vacuum, dielectric, or conductor."
            )
        return str(kind)

    regions = stack.get("solution_regions")
    if not isinstance(regions, Mapping) or not regions:
        raise ValueError("stack must define explicit solution_regions.")
    for name, region in regions.items():
        if not isinstance(region, Mapping) or material_kind(
            region.get("material_id"), f"solution region {name!r}"
        ) not in {"vacuum", "dielectric"}:
            raise ValueError(
                f"solution region {name!r} must use vacuum or dielectric material kind."
            )
    layers = stack.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("stack must define explicit layers.")
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise TypeError("stack layers must be mappings.")
        if (
            layer.get("role") == "metal"
            and material_kind(layer.get("material_id"), "metal layer") != "conductor"
        ):
            raise ValueError("metal layer material must have conductor kind.")


__all__ = ["ElectrostaticSim"]
