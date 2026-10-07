"""SGB-authored Route-A/Route-B Eigenmode preparation; it never runs Palace."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Literal

from scgsim._mesh_quality import MeshQualityReport, check_mesh_quality
from scgsim.sgb import (
    GeometryPlanSnapshot, InputSummary, VacuumRegionSpec,
    summarize_geometry_input,
)
from ._config import (
    _MODEL_L0_M,
    LayoutPortBinding,
    build_eigenmode_config,
    configure_numerical_controls,
)
from ._epr import normalize_surface_epr_specs
from ._mesh import MeshBuildResult, build_route_mesh
from ._mesh_controls import MESH_CONTROL_KEYS
from ._staged import (
    RouteAThinFilm,
    apply_airbox_to_stack,
    normalize_route_a_thin_film,
    validate_non_negative_int,
    validate_nonempty_string,
    validate_positive_number,
)
from ._inputs import (
    _load_stack,
    _non_negative_number,
    _validate_stack_material_kinds,
)
from ._workflow import persist_problem_files, prepare_mesh_input
from .handoff import HandoffPlan, prepare_handoff
from ._epr_results import PalaceEprResult, reanalyze_epr, show_epr


@dataclass(frozen=True)
class _RequestedPort:
    name: str
    layer: str
    inductance: float


@dataclass
class EigenmodeSim:
    """Prepare an SGB-authored Eigenmode mesh, config, and manual handoff."""

    component: Any | None = None
    stack: Mapping[str, Any] | None = None
    output_dir: Path | None = None
    airbox: dict[str, float] = field(default_factory=dict)
    route: Literal["A", "B"] = "B"
    route_a_thin_film: RouteAThinFilm | None = None
    ports: list[_RequestedPort] = field(default_factory=list)
    surface_epr_specs: dict[str, dict[str, Any]] | None = None
    epr_request: dict[str, tuple[str, ...] | None] | None = None
    num_modes: int = 10
    target_hz: float | None = None
    eigenmode_tolerance: float = 1e-6
    save_fields: int = 0
    numerical: dict[str, Any] = field(default_factory=configure_numerical_controls)
    _materials: dict[str, Mapping[str, Any]] | None = field(default=None, init=False)
    _resolved_ports: list[LayoutPortBinding] = field(default_factory=list, init=False)
    vacuum_region: VacuumRegionSpec | None = None
    indium_ground_bumps: dict[str, Any] | None = None
    _mesh_result: MeshBuildResult | None = field(default=None, init=False)
    config_path: Path | None = field(default=None, init=False)
    handoff_plan: HandoffPlan | None = field(default=None, init=False)
    _plan_snapshot: GeometryPlanSnapshot | None = field(default=None, init=False, repr=False)

    @staticmethod
    def reanalyze_epr(
        result: PalaceEprResult,
        *,
        surface_defaults: Mapping[str, Mapping[str, Any]] | None = None,
        group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> PalaceEprResult:
        return reanalyze_epr(result, surface_defaults=surface_defaults, group_overrides=group_overrides)

    @staticmethod
    def show_epr(result: PalaceEprResult, *, mode: int | None = None) -> Any:
        return show_epr(result, mode=mode)

    def _invalidate_mesh(self) -> None:
        self._mesh_result = None
        self._resolved_ports = []
        self.config_path = None
        self.handoff_plan = None

    def _invalidate_config(self) -> None:
        self.config_path = None
        self.handoff_plan = None

    def set_geometry(self, component: Any) -> None:
        if self._plan_snapshot is not None:
            raise ValueError("GeometryPlan owns the paired source; use set_plan(new_snapshot) or a new Sim")
        if component is None or not callable(getattr(component, "write_gds", None)):
            raise TypeError("component must provide write_gds(path).")
        self.component = component
        self._plan_snapshot = None
        self._invalidate_mesh()

    def set_plan(self, snapshot: GeometryPlanSnapshot) -> None:
        """Consume one detached source/stack pair from a notebook GeometryPlan."""
        if not isinstance(snapshot, GeometryPlanSnapshot):
            raise TypeError("set_plan requires a GeometryPlanSnapshot")
        if "VACUUM_REGION" in snapshot.stack["solution_regions"] and (
            self.vacuum_region is not None or self.airbox
        ):
            raise ValueError("GeometryPlan owns the outer vacuum envelope")
        trial = EigenmodeSim()
        trial.set_stack(snapshot.stack)
        self.component = None
        self.stack = trial.stack
        self._materials = trial._materials
        self._plan_snapshot = snapshot
        self._invalidate_mesh()

    def input_summary(self) -> InputSummary:
        """Show the bound Plan source and Palace settings before mesh work."""

        if self._plan_snapshot is None:
            raise ValueError(
                "input_summary requires a bound GeometryPlan snapshot; normalize "
                "a raw Component explicitly first"
            )
        data = summarize_geometry_input(
            self._plan_snapshot.geometry_input,
            prepared_stack=self._plan_snapshot.stack,
        ).data
        data["backend"] = {
            "name": "palace",
            "route": self.route,
            "route_a_thin_film": self.route_a_thin_film,
            "airbox": self.airbox,
            "ports": [
                {"name": item.name, "layer": item.layer,
                 "inductance_h": item.inductance}
                for item in self.ports
            ],
            "surface_epr_specs": self.surface_epr_specs,
            "epr_request": self.epr_request,
            "num_modes": self.num_modes,
            "target_hz": self.target_hz,
            "eigenmode_tolerance": self.eigenmode_tolerance,
            "save_fields": self.save_fields,
            "numerical": self.numerical,
            "indium_ground_bumps": self.indium_ground_bumps,
        }
        return InputSummary(data)

    def set_stack(self, stack: Mapping[str, Any] | str | Path) -> None:
        if self._plan_snapshot is not None:
            raise ValueError("GeometryPlan owns the paired source; use set_plan(new_snapshot) or a new Sim")
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
        self._plan_snapshot = None
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
        if self._plan_snapshot is not None and "VACUUM_REGION" in self._plan_snapshot.stack["solution_regions"]:
            raise ValueError("GeometryPlan owns the outer vacuum envelope")
        if self.vacuum_region is not None:
            raise ValueError(
                "set_airbox is mutually exclusive with set_vacuum_region()."
            )
        airbox = {
            "margin_x": validate_positive_number(margin_x, "margin_x"),
            "margin_y": validate_positive_number(margin_y, "margin_y"),
        }
        if z_above is not None:
            airbox["z_above"] = validate_positive_number(z_above, "z_above")
        if z_below is not None:
            airbox["z_below"] = validate_positive_number(z_below, "z_below")
        self.airbox = airbox
        self._invalidate_mesh()

    def set_vacuum_region(
        self,
        padding: float | list[float] | tuple[float, ...] | Mapping[str, Any] = 0.0,
    ) -> None:
        """Set six-direction padding for a route-aware generated vacuum region."""
        if self._plan_snapshot is not None and "VACUUM_REGION" in self._plan_snapshot.stack["solution_regions"]:
            raise ValueError("GeometryPlan owns the outer vacuum envelope")
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
        normalized_specs = normalize_surface_epr_specs(specs, materials=self._materials)
        self.route = route  # type: ignore[assignment]
        self.route_a_thin_film = normalized_thin_film
        self.surface_epr_specs = normalized_specs
        self._invalidate_mesh()

    def add_port(
        self,
        name: str,
        *,
        layer: str,
        layout_sheet: bool = False,
        inductance: float,
    ) -> None:
        """Bind one authored GDSFactory port to its target logical metal."""
        if layout_sheet is not True:
            raise ValueError("Eigenmode V1 supports only layout_sheet=True ports.")
        name = validate_nonempty_string(name, "port name")
        layer = validate_nonempty_string(layer, "port target layer")
        if any(port.name == name for port in self.ports):
            raise ValueError(f"layout_sheet port {name!r} is already configured.")
        self.ports.append(
            _RequestedPort(
                name=name,
                layer=layer,
                inductance=validate_positive_number(inductance, "inductance"),
            )
        )
        self._invalidate_mesh()

    def set_epr_request(
        self,
        *,
        surface_interfaces: Sequence[str] | None = None,
        bulk_domain_ids: Sequence[str] | None = None,
        port_names: Sequence[str] | None = None,
    ) -> None:
        """Select reported EPR quantities without changing the physical solve."""

        def selected(values: Sequence[str] | None, name: str) -> tuple[str, ...] | None:
            if values is None:
                return None
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise TypeError(f"{name} must be a sequence of IDs or None")
            normalized = tuple(values)
            if any(not isinstance(value, str) or not value for value in normalized):
                raise ValueError(f"{name} must contain non-empty text IDs")
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{name} must not repeat IDs")
            return normalized

        surfaces = selected(surface_interfaces, "surface_interfaces")
        if surfaces is not None and set(surfaces) - {"MA", "MS", "SA"}:
            raise ValueError("surface_interfaces supports only MA, MS, and SA")
        request = {
            "surface_interfaces": surfaces,
            "bulk_domain_ids": selected(bulk_domain_ids, "bulk_domain_ids"),
            "port_names": selected(port_names, "port_names"),
        }
        self.epr_request = request
        self._invalidate_config()

    def set_eigenmode(
        self,
        *,
        num_modes: int = 10,
        target: float | None = None,
        tolerance: float = 1e-6,
        save: int = 0,
    ) -> None:
        if (
            not isinstance(num_modes, int)
            or isinstance(num_modes, bool)
            or num_modes < 1
        ):
            raise ValueError("num_modes must be a positive integer.")
        if target is not None:
            target = validate_positive_number(target, "target")
        normalized_tolerance = validate_positive_number(tolerance, "tolerance")
        normalized_save = validate_non_negative_int(save, "save")
        self.num_modes = num_modes
        self.target_hz = target
        self.eigenmode_tolerance = normalized_tolerance
        self.save_fields = normalized_save
        self._invalidate_config()

    def set_mesh(
        self, *, refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None, algorithm_3d: str | None = None,
        threads: int | None = None, surface_threads: int | None = None,
        geometry_order: int | None = None, high_order_optimize: bool | None = None,
    ) -> None:
        """Set geometry-mesh controls; every omitted value retains its setting.

        Geometry order is independent of ``set_numerical(order=...)``. Elevated
        geometry is optimized with Gmsh HighOrder once when enabled.
        """
        proposed = dict(self.numerical)
        proposed.update({key: value for key, value in {
            "refined_mesh_size": refined_mesh_size, "max_mesh_size": max_mesh_size,
            "algorithm_3d": algorithm_3d, "threads": threads,
            "surface_threads": surface_threads, "geometry_order": geometry_order,
            "high_order_optimize": high_order_optimize,
        }.items() if value is not None})
        numerical = configure_numerical_controls(**proposed)
        if numerical != self.numerical:
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
        save_adapt_mesh: bool | None = None,
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
            **{key: self.numerical[key] for key in MESH_CONTROL_KEYS},
            amr_max_passes=amr_max_passes,
            amr_nonconformal=amr_nonconformal,
            amr_tolerance=amr_tolerance,
            amr_update_fraction=amr_update_fraction,
            save_adapt_iterations=save_adapt_iterations,
            save_adapt_mesh=save_adapt_mesh,
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
        if (self.component is None and self._plan_snapshot is None) or self.stack is None or self.output_dir is None:
            raise ValueError(
                "set_geometry(), set_stack(), and set_output_dir() must run before mesh()."
            )
        if self._plan_snapshot is not None:
            if self.indium_ground_bumps is not None and self.indium_ground_bumps["fill"]:
                raise ValueError("GeometryPlan source already owns its geometry; add authored bumps before prepare()")
            available_records = self._plan_snapshot.stack["metadata"].get(
                "port_sheet_source_layers", ()
            )
            available = {record["name"]: record for record in available_records}
            if len(available) != len(available_records):
                raise ValueError("GeometryPlan port names must be unique")
            resolved, source_records = [], []
            for requested in self.ports:
                record = available.get(requested.name)
                if record is None or record["target_layer"] != requested.layer:
                    raise ValueError(f"GeometryPlan has no matching authored port {requested.name!r}")
                resolved.append(LayoutPortBinding(
                    index=len(resolved) + 1, name=requested.name,
                    target_layer=requested.layer,
                    source_layer=f"{record['layer']}/{record['datatype']}",
                    direction=tuple(record["direction"]),
                    inductance=requested.inductance,
                ))
                source_records.append({**record, "port_index": len(resolved)})
            source_stack = self._plan_snapshot.stack
            source_stack["metadata"].pop("port_sheet_source_layers", None)
            prepared = prepare_mesh_input(
                component=None,
                stack=source_stack,
                route=self.route,
                route_a_thin_film=self.route_a_thin_film,
                vacuum_region=(
                    None if "VACUUM_REGION" in source_stack["solution_regions"]
                    else self.vacuum_region
                ),
                indium_ground_bumps=None,
                build_input=self._plan_snapshot.geometry_input,
            )
            if prepared.materials is not None:
                self._materials = prepared.materials
            self._mesh_result = build_route_mesh(
                component=None,
                stack=apply_airbox_to_stack(prepared.stack, self.airbox),
                route=self.route,
                output_dir=self.output_dir,
                refined_mesh_size=self.numerical["refined_mesh_size"],
                max_mesh_size=self.numerical["max_mesh_size"],
                **{key: self.numerical[key] for key in MESH_CONTROL_KEYS},
                port_sheet_source_layers=source_records,
                source_gds_bytes=self._plan_snapshot.gds_bytes,
                source_geometry_input=self._plan_snapshot.geometry_input,
            )
            self._resolved_ports = resolved
            return self._mesh_result.mesh_path
        resolved, source_records = (
            _resolve_layout_ports(self.component, self.ports)
            if self.ports else ([], [])
        )
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
            **{key: self.numerical[key] for key in MESH_CONTROL_KEYS},
            port_sheet_source_layers=source_records,
            indium_ground_bump_fill=prepared.indium_ground_bump_fill,
        )
        self._resolved_ports = resolved
        return self._mesh_result.mesh_path

    def mesh_summary(self) -> dict[str, Any]:
        """Detach actual mesh observations and the current FEM topology estimate."""
        from ._mesh_summary import summary_from_files

        if self._mesh_result is None:
            raise RuntimeError("mesh_summary requires a generated mesh.")
        return summary_from_files(
            self._mesh_result.mesh_path, self._mesh_result.mesh_manifest_path,
            problem="Eigenmode", fem_order=self.numerical["order"],
        )

    def check_mesh_quality(self) -> MeshQualityReport:
        """Inspect the current mesh without changing simulation state."""

        if self._mesh_result is None:
            raise ValueError("mesh() must run before check_mesh_quality().")
        return check_mesh_quality(
            self._mesh_result.mesh_path, length_scale_m=_MODEL_L0_M
        )

    def write_config(self) -> Path:
        self._invalidate_config()
        if (
            self._mesh_result is None
            or self._materials is None
        ):
            raise ValueError("set_stack() and mesh() are required.")
        if not self._mesh_result.mesh_path.is_file():
            raise FileNotFoundError(
                "current palace.msh is missing; mesh() must be rerun."
            )
        result = build_eigenmode_config(
            groups=self._mesh_result.groups,
            ports=self._resolved_ports,
            materials=self._materials,
            surface_epr_specs=self.surface_epr_specs or {},
            numerical=self.numerical,
            num_modes=self.num_modes,
            target_hz=self.target_hz,
            eigenmode_tolerance=self.eigenmode_tolerance,
            save_fields=self.save_fields,
            mesh_path=self._mesh_result.mesh_path,
        )
        request = self.epr_request
        if request is not None:
            available = {
                "surface_interfaces": {
                    item["metadata"]["interface_type"] for item in result.index_entries
                    if item["section"] == "Boundaries.Postprocessing.Dielectric"
                },
                "bulk_domain_ids": {
                    item["entry_name"] for item in result.index_entries
                    if item["section"] == "Domains.Postprocessing.Energy"
                },
                "port_names": {
                    item["port_name"] for item in result.index_entries
                    if item["section"] == "Boundaries.LumpedPort"
                },
            }
            for name, configured in available.items():
                chosen = request[name]
                if chosen is not None and set(chosen) - configured:
                    raise ValueError(f"EPR request selects an unknown {name} binding")
        index_payload = {"schema_version": 1, "entries": result.index_entries}
        if request is not None:
            index_payload["epr_request"] = {
                "schema_version": "scgsim.palace.epr-selection.v1",
                **{
                    name: None if values is None else list(values)
                    for name, values in request.items()
                },
            }
        metadata = self._mesh_result.output_dir / "metadata"
        config_path = self._mesh_result.output_dir / "config.json"
        self.config_path = persist_problem_files(
            metadata_files=(
                (
                    metadata / "palace_index_map.json",
                    index_payload,
                ),
                (
                    metadata / "palace_material_resolution.json",
                    {
                        "schema_version": 1,
                        "solution_volumes": result.material_resolution,
                    },
                ),
                (
                    metadata / "port_information.json",
                    {"schema_version": 1, "ports": result.port_information},
                ),
                (metadata / "palace_numerical_controls.json", self.numerical),
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
            problem="Eigenmode",
        )
        return self.handoff_plan


def _resolve_layout_ports(
    component: Any, ports: list[_RequestedPort]
) -> tuple[list[LayoutPortBinding], list[dict[str, Any]]]:
    resolved: list[LayoutPortBinding] = []
    records: list[dict[str, Any]] = []
    used_layers: set[tuple[int, int]] = set()
    component_ports = getattr(component, "ports", None)
    if component_ports is None:
        raise TypeError("component must expose authored ports.")
    for index, requested in enumerate(ports, start=1):
        try:
            port = component_ports[requested.name]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"component has no authored port {requested.name!r}."
            ) from exc
        source = _port_layer(port)
        if source in used_layers:
            raise ValueError(
                "layout_sheet ports must use distinct authored GDS layers."
            )
        used_layers.add(source)
        orientation = getattr(port, "orientation", None)
        if not isinstance(orientation, Real) or isinstance(orientation, bool):
            raise TypeError(
                f"authored port {requested.name!r} needs numeric orientation."
            )
        angle = math.radians(float(orientation))
        direction = (round(math.cos(angle), 15), round(math.sin(angle), 15), 0.0)
        length = math.hypot(direction[0], direction[1])
        if not math.isfinite(length) or length == 0.0:
            raise ValueError(f"authored port {requested.name!r} has invalid direction.")
        direction = (direction[0] / length, direction[1] / length, 0.0)
        source_layer = f"{source[0]}/{source[1]}"
        resolved.append(
            LayoutPortBinding(
                index=index,
                name=requested.name,
                target_layer=requested.layer,
                source_layer=source_layer,
                direction=direction,
                inductance=requested.inductance,
            )
        )
        records.append(
            {
                "layer": source[0],
                "datatype": source[1],
                "name": requested.name,
                "source": "palace_lumped_port_sheet",
                "port_index": index,
                "target_layer": requested.layer,
                "direction": list(direction),
                "direction_sign_convention": "gdsfactory_port_orientation_outward",
            }
        )
    return resolved, records


def _port_layer(port: Any) -> tuple[int, int]:
    raw = getattr(port, "layer", None)
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        return (int(raw[0]), int(raw[1]))
    if hasattr(raw, "layer") and hasattr(raw, "datatype"):
        return (int(raw.layer), int(raw.datatype))
    try:
        info = port.kcl.layout.get_info(int(str(raw)))
        return (int(info.layer), int(info.datatype))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("layout_sheet port requires an exact GDS layer.") from exc


__all__ = ["EigenmodeSim"]
