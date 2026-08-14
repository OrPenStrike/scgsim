"""SGB-authored Route-A/Route-B Eigenmode preparation; it never runs Palace."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Literal

from scgsim.sgb import VacuumRegionSpec

from ._config import LayoutPortBinding, build_eigenmode_config
from ._epr import normalize_surface_epr_specs
from ._mesh import MeshBuildResult, build_route_mesh
from ._staged import (
    apply_airbox_to_stack,
    apply_vacuum_region_to_stack,
    validate_non_negative_int,
    validate_nonempty_string,
    validate_positive_number,
)
from .electrostatic import (
    _DEVICES,
    _PRECONDITIONERS,
    _SOLVER_TYPES,
    _atomic_json,
    _load_stack,
    _validate_stack_material_kinds,
)
from .handoff import HandoffPlan, prepare_handoff


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
    ports: list[_RequestedPort] = field(default_factory=list)
    surface_epr_specs: dict[str, dict[str, Any]] | None = None
    num_modes: int = 10
    target_hz: float | None = None
    eigenmode_tolerance: float = 1e-6
    save_fields: int = 0
    numerical: dict[str, Any] = field(
        default_factory=lambda: {
            "order": 1,
            "tolerance": 1e-6,
            "max_iterations": 400,
            "solver_type": "Default",
            "preconditioner": "Default",
            "device": "CPU",
            "refined_mesh_size": 5.0,
            "max_mesh_size": 300.0,
        }
    )
    _materials: dict[str, Mapping[str, Any]] | None = field(default=None, init=False)
    _resolved_ports: list[LayoutPortBinding] = field(default_factory=list, init=False)
    vacuum_region: VacuumRegionSpec | None = None
    _mesh_result: MeshBuildResult | None = field(default=None, init=False)
    config_path: Path | None = field(default=None, init=False)
    handoff_plan: HandoffPlan | None = field(default=None, init=False)

    def _invalidate_mesh(self) -> None:
        self._mesh_result = None
        self._resolved_ports = []
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

    def set_surface_epr(
        self, *, representation: str, specs: Mapping[str, Mapping[str, Any]]
    ) -> None:
        route = (
            representation.strip().upper() if isinstance(representation, str) else ""
        )
        if route not in {"A", "B"}:
            raise ValueError("Surface EPR representation must be 'A' or 'B'.")
        if self._materials is None:
            raise ValueError("set_stack() must run before set_surface_epr().")
        self.route = route  # type: ignore[assignment]
        self.surface_epr_specs = normalize_surface_epr_specs(
            specs, materials=self._materials
        )
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
        self.num_modes = num_modes
        self.target_hz = target
        self.eigenmode_tolerance = validate_positive_number(tolerance, "tolerance")
        self.save_fields = validate_non_negative_int(save, "save")
        self._invalidate_config()

    def set_numerical(
        self,
        *,
        order: int = 1,
        tolerance: float = 1e-6,
        max_iterations: int = 400,
        solver_type: str = "Default",
        preconditioner: str = "Default",
        device: str = "CPU",
        refined_mesh_size: float = 5.0,
        max_mesh_size: float = 300.0,
    ) -> None:
        if not isinstance(order, int) or isinstance(order, bool) or not 1 <= order <= 4:
            raise ValueError("order must be an integer from 1 through 4.")
        if (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
            or max_iterations <= 0
        ):
            raise ValueError("max_iterations must be a positive integer.")
        if (
            solver_type not in _SOLVER_TYPES
            or preconditioner not in _PRECONDITIONERS
            or device not in _DEVICES
        ):
            raise ValueError("unsupported solver_type, preconditioner, or device.")
        refined = validate_positive_number(refined_mesh_size, "refined_mesh_size")
        maximum = validate_positive_number(max_mesh_size, "max_mesh_size")
        if maximum < refined:
            raise ValueError("max_mesh_size must be >= refined_mesh_size.")
        self.numerical = {
            "order": order,
            "tolerance": validate_positive_number(tolerance, "tolerance"),
            "max_iterations": max_iterations,
            "solver_type": solver_type,
            "preconditioner": preconditioner,
            "device": device,
            "refined_mesh_size": refined,
            "max_mesh_size": maximum,
        }
        self._invalidate_mesh()

    def mesh(self) -> Path:
        self._invalidate_mesh()
        if self.component is None or self.stack is None or self.output_dir is None:
            raise ValueError(
                "set_geometry(), set_stack(), and set_output_dir() must run before mesh()."
            )
        if self.surface_epr_specs is None:
            raise ValueError("set_surface_epr() must run before mesh().")
        if not self.ports:
            raise ValueError("add_port(..., layout_sheet=True) must run before mesh().")
        resolved, source_records = _resolve_layout_ports(self.component, self.ports)
        prepared_stack = self.stack
        if self.vacuum_region is not None:
            prepared_stack = apply_vacuum_region_to_stack(
                self.stack,
                self.vacuum_region,
            )
        prepared_materials = prepared_stack.get("materials")
        if isinstance(prepared_materials, Mapping):
            self._materials = {
                str(material_id): dict(material)
                for material_id, material in prepared_materials.items()
                if isinstance(material, Mapping)
            }
        self._mesh_result = build_route_mesh(
            component=self.component,
            stack=apply_airbox_to_stack(prepared_stack, self.airbox),
            route=self.route,
            output_dir=self.output_dir,
            refined_mesh_size=self.numerical["refined_mesh_size"],
            max_mesh_size=self.numerical["max_mesh_size"],
            port_sheet_source_layers=source_records,
        )
        self._resolved_ports = resolved
        return self._mesh_result.mesh_path

    def write_config(self) -> Path:
        self._invalidate_config()
        if (
            self._mesh_result is None
            or self._materials is None
            or self.surface_epr_specs is None
            or not self._resolved_ports
        ):
            raise ValueError(
                "set_stack(), set_surface_epr(), add_port(), and mesh() are required."
            )
        if not self._mesh_result.mesh_path.is_file():
            raise FileNotFoundError(
                "current palace.msh is missing; mesh() must be rerun."
            )
        result = build_eigenmode_config(
            groups=self._mesh_result.groups,
            ports=self._resolved_ports,
            materials=self._materials,
            surface_epr_specs=self.surface_epr_specs,
            numerical=self.numerical,
            num_modes=self.num_modes,
            target_hz=self.target_hz,
            eigenmode_tolerance=self.eigenmode_tolerance,
            save_fields=self.save_fields,
            mesh_path=self._mesh_result.mesh_path,
        )
        metadata = self._mesh_result.output_dir / "metadata"
        config_path = self._mesh_result.output_dir / "config.json"
        _atomic_json(
            metadata / "palace_index_map.json",
            {"schema_version": 1, "entries": result.index_entries},
        )
        _atomic_json(
            metadata / "palace_material_resolution.json",
            {"schema_version": 1, "solution_volumes": result.material_resolution},
        )
        _atomic_json(
            metadata / "port_information.json",
            {"schema_version": 1, "ports": result.port_information},
        )
        _atomic_json(metadata / "palace_numerical_controls.json", self.numerical)
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
