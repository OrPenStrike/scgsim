"""Thin staged Route-A/Route-B electrostatic handoff API; it never runs Palace."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from scgsim.sgb import VacuumRegionSpec
from ._config import (
    TerminalBinding,
    build_electrostatic_config,
    configure_numerical_controls,
)
from ._epr import normalize_surface_epr_specs
from ._inputs import (
    _load_stack,
    _non_negative_number,
    _validate_stack_material_kinds,
)
from ._mesh import MeshBuildResult, build_route_mesh
from ._staged import (
    RouteAThinFilm,
    apply_airbox_to_stack,
    normalize_route_a_thin_film,
    validate_non_negative_int,
    validate_nonempty_string,
    validate_positive_number,
)
from ._workflow import persist_problem_files, prepare_mesh_input
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
    ground_net_ids: list[str] = field(default_factory=list)
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

    def add_ground(self, *, net_id: str) -> None:
        """Bind exact SGB conductor-net identity to physical Palace Ground."""
        ground_net_id = validate_nonempty_string(net_id, "net_id")
        if ground_net_id not in self.ground_net_ids:
            self.ground_net_ids.append(ground_net_id)
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
        prepared = prepare_mesh_input(
            component=self.component,
            stack=self.stack,
            route=self.route,
            route_a_thin_film=self.route_a_thin_film,
            vacuum_region=self.vacuum_region,
            indium_ground_bumps=self.indium_ground_bumps,
        )
        if prepared.materials is not None:
            self._materials = prepared.materials
        self._mesh_result = build_route_mesh(
            component=prepared.component,
            stack=apply_airbox_to_stack(prepared.stack, self.airbox),
            route=self.route,
            output_dir=self.output_dir,
            refined_mesh_size=self.numerical["refined_mesh_size"],
            max_mesh_size=self.numerical["max_mesh_size"],
            indium_ground_bump_fill=prepared.indium_ground_bump_fill,
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
            ground_net_ids=self.ground_net_ids,
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
            "schema_version": 2,
            "entries": [
                *result.terminal_index_map,
                *result.domain_energy_index_map,
                *result.surface_epr_index_map,
            ],
            "ground_boundary": result.ground_boundary_resolution,
        }
        material_payload = {
            "schema_version": 1,
            "solution_volumes": result.material_resolution,
        }
        config_path = self._mesh_result.output_dir / "config.json"
        self.config_path = persist_problem_files(
            metadata_files=(
                (metadata_dir / "palace_index_map.json", index_payload),
                (metadata_dir / "palace_material_resolution.json", material_payload),
                (metadata_dir / "palace_numerical_controls.json", self.numerical),
            ),
            config_path=config_path,
            config=result.config,
        )
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


__all__ = ["ElectrostaticSim"]
