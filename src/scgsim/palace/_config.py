"""Palace config construction and terminal/field validation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Literal

from ._epr import build_surface_epr_postprocessing


@dataclass(frozen=True)
class TerminalBinding:
    """Exact net mapping record for a conductor terminal."""

    name: str
    net_id: str


@dataclass(frozen=True)
class ConfigBuildResult:
    config: dict[str, Any]
    domain_names: list[str]
    terminal_index_map: list[dict[str, Any]]
    surface_epr_index_map: list[dict[str, Any]]
    domain_energy_index_map: list[dict[str, Any]]
    material_resolution: list[dict[str, Any]]
    ground_boundary_resolution: dict[str, Any]


@dataclass(frozen=True)
class LayoutPortBinding:
    """One authored GDSFactory junction port resolved for SGB lowering."""

    index: int
    name: str
    target_layer: str
    source_layer: str
    direction: tuple[float, float, float]
    inductance: float


@dataclass(frozen=True)
class EigenmodeConfigBuildResult:
    config: dict[str, Any]
    port_information: list[dict[str, Any]]
    index_entries: list[dict[str, Any]]
    material_resolution: list[dict[str, Any]]


def build_electrostatic_config(
    *,
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    terminals: Sequence[TerminalBinding],
    ground_net_ids: Sequence[str] = (),
    materials: Mapping[str, Mapping[str, Any]],
    save_fields: int,
    unassigned_conductor_policy: Literal["ground", "error"],
    exterior_boundary_policy: Literal["none", "ground"],
    mesh_path: Path,
    surface_epr_specs: Mapping[str, Mapping[str, Any]],
    numerical: Mapping[str, Any],
) -> ConfigBuildResult:
    """Build a minimal Palace electrostatic config payload."""
    if (
        not isinstance(save_fields, int)
        or isinstance(save_fields, bool)
        or save_fields < 0
    ):
        raise ValueError("save_fields must be a non-bool int >= 0.")
    if exterior_boundary_policy == "ground" and unassigned_conductor_policy != "error":
        raise ValueError(
            "exterior ground requires unassigned_conductor_policy='error'."
        )

    boundary_map, terminal_index_map, ground_boundary_resolution = (
        _electrostatic_boundaries(
            groups=groups,
            terminals=terminals,
            ground_net_ids=ground_net_ids,
            unassigned_conductor_policy=unassigned_conductor_policy,
            exterior_boundary_policy=exterior_boundary_policy,
        )
    )
    domain_volumes = _domains_from_solution_volumes(groups.get("volumes", {}))
    materials_payload = _materials_from_solution_volumes(
        domain_volumes=domain_volumes,
        materials=materials,
    )
    epr_rows, epr_index_map = build_surface_epr_postprocessing(
        groups, surface_epr_specs
    )
    energy_rows = [
        {
            "Index": index,
            "Attributes": sorted(_physical_group_values(volume.get("phys_group"))),
        }
        for index, (_, volume) in enumerate(domain_volumes, start=1)
    ]
    energy_index_map = [
        {
            "section": "Domains.Postprocessing.Energy",
            "index": index,
            "entry_name": name,
            "role": "domain_energy",
            "physical_name": name,
            "physical_names": [name],
            "attributes": row["Attributes"],
            "metadata": {
                "physical_attribute": volume.get("physical_attribute", {}),
                "material": _material_identity(volume, materials),
                "source_provenance": volume.get("source_provenance", {}),
                "representation": volume.get("representation"),
                "route": volume.get("route"),
            },
        }
        for index, ((name, volume), row) in enumerate(
            zip(domain_volumes, energy_rows, strict=True), start=1
        )
    ]

    linear = _linear_solver(numerical)
    problem_block = _build_output_formats(numerical)
    problem = {
        "Type": "Electrostatic",
        "Verbose": 3,
        "Output": "results/palace",
    }
    problem.update(problem_block)
    config: dict[str, Any] = {
        "Problem": problem,
        "Model": {
            "Mesh": str(mesh_path.name),
            "L0": 1e-6,
            "Refinement": _build_refinement(numerical),
        },
        "Solver": {
            "Linear": linear,
            "Order": int(numerical["order"]),
            "Device": numerical["device"],
            "Electrostatic": {"Save": save_fields},
        },
        "Boundaries": boundary_map,
        "Domains": {
            "Materials": materials_payload,
            "Postprocessing": {"Energy": energy_rows, "Probe": []},
        },
    }
    if epr_rows:
        config["Boundaries"]["Postprocessing"] = {"Dielectric": epr_rows}
    return ConfigBuildResult(
        config=config,
        domain_names=[name for name, _ in domain_volumes],
        terminal_index_map=terminal_index_map,
        surface_epr_index_map=epr_index_map,
        domain_energy_index_map=energy_index_map,
        material_resolution=[
            {
                "solution_volume": name,
                "attributes": sorted(_physical_group_values(volume.get("phys_group"))),
                "physical_attribute": volume.get("physical_attribute", {}),
                "material": _material_identity(volume, materials),
                "source_provenance": volume.get("source_provenance", {}),
            }
            for name, volume in domain_volumes
        ],
        ground_boundary_resolution=ground_boundary_resolution,
    )


def build_eigenmode_config(
    *,
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ports: Sequence[LayoutPortBinding],
    materials: Mapping[str, Mapping[str, Any]],
    surface_epr_specs: Mapping[str, Mapping[str, Any]],
    numerical: Mapping[str, Any],
    num_modes: int,
    target_hz: float | None,
    eigenmode_tolerance: float,
    save_fields: int,
    mesh_path: Path,
) -> EigenmodeConfigBuildResult:
    """Build one non-Floquet Eigenmode config from SGB-authored groups."""
    if not ports:
        raise ValueError("Eigenmode layout-sheet candidate requires at least one port.")
    if not isinstance(num_modes, int) or isinstance(num_modes, bool) or num_modes < 1:
        raise ValueError("num_modes must be a positive integer.")
    if not _positive_finite(eigenmode_tolerance):
        raise ValueError("Eigenmode tolerance must be finite and positive.")
    if target_hz is not None and not _positive_finite(target_hz):
        raise ValueError("Eigenmode target must be finite positive Hz.")
    if (
        not isinstance(save_fields, int)
        or isinstance(save_fields, bool)
        or save_fields < 0
    ):
        raise ValueError("save_fields must be a non-negative integer.")

    domain_volumes = _domains_from_solution_volumes(groups.get("volumes", {}))
    material_rows = _materials_from_solution_volumes(
        domain_volumes=domain_volumes, materials=materials
    )
    energy_rows, energy_index = _domain_energy_rows(domain_volumes, materials)
    epr_rows, epr_index = build_surface_epr_postprocessing(groups, surface_epr_specs)
    lumped_ports, port_info, port_index = _eigenmode_lumped_ports(groups, ports)
    epr_attributes = {
        int(value) for row in epr_rows for value in row.get("Attributes", ())
    }
    port_attributes = {
        int(value) for row in lumped_ports for value in row.get("Attributes", ())
    }
    if epr_attributes.intersection(port_attributes):
        raise ValueError(
            "LumpedPort attributes must not also be Surface-EPR loss groups."
        )

    pec_attributes = sorted(
        {
            int(value)
            for info in groups.get("pec_surfaces", {}).values()
            for value in _physical_group_values(info.get("phys_group"))
        }
    )
    if not pec_attributes:
        raise ValueError("Eigenmode config requires SGB-authored PEC surfaces.")
    eigenmode: dict[str, Any] = {
        "N": num_modes,
        "Tol": float(eigenmode_tolerance),
        "Save": save_fields,
    }
    if target_hz is not None:
        eigenmode["Target"] = float(target_hz) / 1e9
    boundaries: dict[str, Any] = {
        "PEC": {"Attributes": pec_attributes},
        "LumpedPort": lumped_ports,
    }
    if epr_rows:
        boundaries["Postprocessing"] = {"Dielectric": epr_rows}
    problem_block = _build_output_formats(numerical)
    problem = {
        "Type": "Eigenmode",
        "Verbose": 3,
        "Output": "results/palace",
    }
    problem.update(problem_block)
    config = {
        "Problem": problem,
        "Model": {
            "Mesh": mesh_path.name,
            "L0": 1e-6,
            "Refinement": _build_refinement(numerical),
        },
        "Solver": {
            "Linear": _linear_solver(numerical),
            "Order": int(numerical["order"]),
            "Device": numerical["device"],
            "Eigenmode": eigenmode,
        },
        "Boundaries": boundaries,
        "Domains": {
            "Materials": material_rows,
            "Postprocessing": {"Energy": energy_rows, "Probe": []},
        },
    }
    return EigenmodeConfigBuildResult(
        config=config,
        port_information=port_info,
        index_entries=[*port_index, *energy_index, *epr_index],
        material_resolution=[
            {
                "solution_volume": name,
                "attributes": sorted(_physical_group_values(volume.get("phys_group"))),
                "physical_attribute": volume.get("physical_attribute", {}),
                "material": _material_identity(volume, materials),
                "source_provenance": volume.get("source_provenance", {}),
            }
            for name, volume in domain_volumes
        ],
    )


def _electrostatic_boundaries(
    *,
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    terminals: Sequence[TerminalBinding],
    ground_net_ids: Sequence[str],
    unassigned_conductor_policy: str,
    exterior_boundary_policy: str,
) -> tuple[dict[str, object], list[dict[str, Any]], dict[str, Any]]:
    """Map exact conductor_component_id/net_id pairs to terminal and ground boundaries."""
    component_attrs: dict[str, set[int]] = {}
    component_nets: dict[str, str] = {}
    component_physical_names: dict[str, set[str]] = {}
    attribute_components: dict[int, str] = {}

    for physical_name, info in groups.get("boundary_surfaces", {}).items():
        if not isinstance(info, Mapping):
            continue
        if (
            info.get("sgb_record") != "final_physical_group"
            or info.get("source") != "volume_interface"
            or info.get("solver_use") != "solver_active"
            or str(info.get("interface_type")).upper() not in {"MA", "MS"}
        ):
            continue

        component_id = _optional_string(info.get("conductor_component_id"))
        net_id = _optional_string(info.get("net_id"))
        if component_id is None:
            raise ValueError(
                "SGB solver-active conductor surface missing conductor_component_id."
            )
        if net_id is None:
            raise ValueError(
                f"SGB conductor component {component_id!r} missing exact net_id."
            )

        attrs = _physical_group_values(info.get("phys_group"))
        if not attrs:
            raise ValueError(
                f"SGB conductor component {component_id!r} has no physical attributes."
            )

        component_attrs.setdefault(component_id, set()).update(attrs)
        if component_id in component_nets and component_nets[component_id] != net_id:
            raise ValueError(
                f"SGB conductor component {component_id!r} has conflicting net_id values."
            )
        component_nets[component_id] = net_id
        component_physical_names.setdefault(component_id, set()).add(str(physical_name))

        for attr in attrs:
            existing = attribute_components.get(int(attr))
            if existing is None:
                attribute_components[int(attr)] = component_id
            elif existing != component_id:
                raise ValueError(
                    f"SGB physical attribute {int(attr)} belongs to multiple conductor components."
                )

    if not component_attrs:
        raise ValueError(
            "SGB config needs at least one solver-active conductor component."
        )

    terminal_entries: list[dict[str, object]] = []
    terminal_index_map: list[dict[str, Any]] = []
    assigned: set[str] = set()
    seen_nets: set[str] = set()

    for index, terminal in enumerate(terminals, start=1):
        net_id = _optional_string(terminal.net_id)
        name = _optional_string(terminal.name)
        if net_id is None or name is None:
            raise ValueError("terminal binding names and net ids must be non-empty.")
        if net_id in seen_nets:
            raise ValueError(f"SGB terminal net {net_id!r} is assigned more than once.")
        seen_nets.add(net_id)

        matching = sorted(
            component_id
            for component_id, value in component_nets.items()
            if value == net_id
        )
        if not matching:
            raise ValueError(f"SGB terminal {name!r} has no exact net match.")
        overlap = assigned.intersection(matching)
        if overlap:
            raise ValueError(
                f"SGB terminal {name!r} splits an already-assigned conductor component."
            )

        attrs = sorted(
            int(attr)
            for component_id in matching
            for attr in component_attrs[component_id]
        )
        terminal_entries.append({"Index": index, "Attributes": attrs})
        terminal_index_map.append(
            {
                "section": "Boundaries.Terminal",
                "index": index,
                "entry_name": name,
                "role": "terminal",
                "attributes": attrs,
                "physical_names": sorted(
                    name
                    for name, info in groups.get("boundary_surfaces", {}).items()
                    if isinstance(info, Mapping)
                    and _optional_string(info.get("conductor_component_id")) in matching
                ),
                "metadata": {
                    "net_id": net_id,
                    "conductor_component_ids": matching,
                },
                "terminal_name": name,
                "net_id": net_id,
                "conductor_component_ids": matching,
                "terminal_attributes": attrs,
            }
        )
        assigned.update(matching)

    requested_ground_nets: list[str] = []
    physical_ground_components: set[str] = set()
    for value in ground_net_ids:
        net_id = _optional_string(value)
        if net_id is None:
            raise ValueError("physical Ground net ids must be non-empty.")
        if net_id in requested_ground_nets:
            raise ValueError(f"SGB physical Ground net {net_id!r} is repeated.")
        matching = {
            component_id
            for component_id, candidate in component_nets.items()
            if candidate == net_id
        }
        if not matching:
            raise ValueError(f"SGB physical Ground net {net_id!r} has no exact match.")
        overlap = assigned.intersection(matching)
        if overlap:
            raise ValueError(
                f"SGB physical Ground net {net_id!r} overlaps a Terminal binding."
            )
        requested_ground_nets.append(net_id)
        physical_ground_components.update(matching)
        assigned.update(matching)

    unassigned = set(component_attrs).difference(assigned)
    if unassigned and unassigned_conductor_policy == "error":
        raise ValueError(
            "Unassigned conductor components: " + ", ".join(sorted(unassigned))
        )

    if unassigned_conductor_policy == "ground":
        physical_ground_components.update(unassigned)

    boundary_map: dict[str, object] = {"Terminal": terminal_entries}
    physical_ground_attrs: set[int] = {
        int(attr)
        for component_id in physical_ground_components
        for attr in component_attrs[component_id]
    }
    exterior_ground_attrs: set[int] = set()
    exterior_physical_names: set[str] = set()

    if exterior_boundary_policy == "ground":
        for physical_name, info in groups.get("boundary_surfaces", {}).items():
            if (
                isinstance(info, Mapping)
                and info.get("sgb_record") == "final_physical_group"
                and info.get("source") == "domain_boundary"
                and info.get("solver_use") == "solver_active"
            ):
                exterior_ground_attrs.update(
                    _physical_group_values(info.get("phys_group"))
                )
                exterior_physical_names.add(str(physical_name))
        if not exterior_ground_attrs:
            raise ValueError(
                "exterior_boundary_policy='ground' requires at least one exterior boundary."
            )

    if set(attribute_components).intersection(exterior_ground_attrs):
        raise ValueError("conductor and exterior boundary attributes must be disjoint.")
    ground_attrs = physical_ground_attrs | exterior_ground_attrs
    if ground_attrs:
        boundary_map["Ground"] = {"Attributes": sorted(ground_attrs)}

    if sorted(
        (entry["index"], list(entry["attributes"])) for entry in terminal_index_map
    ) != sorted(
        (entry["Index"], list(entry["Attributes"])) for entry in terminal_entries
    ):
        raise ValueError("Terminal-to-boundary mapping mismatch after synthesis.")

    ground_components = sorted(physical_ground_components)
    return (
        boundary_map,
        terminal_index_map,
        {
            "requested_net_ids": sorted(requested_ground_nets),
            "physical_net_ids": sorted(
                {component_nets[component_id] for component_id in ground_components}
            ),
            "conductor_component_ids": ground_components,
            "physical_names": sorted(
                {
                    name
                    for component_id in ground_components
                    for name in component_physical_names[component_id]
                }
            ),
            "physical_attributes": sorted(physical_ground_attrs),
            "exterior_physical_names": sorted(exterior_physical_names),
            "exterior_attributes": sorted(exterior_ground_attrs),
            "union_attributes": sorted(ground_attrs),
        },
    )


def _linear_solver(numerical: Mapping[str, Any]) -> dict[str, Any]:
    linear = {
        "Type": numerical["solver_type"],
        "KSPType": "GMRES",
        "Tol": float(numerical["tolerance"]),
        "MaxIts": int(numerical["max_iterations"]),
    }
    if numerical["solver_type"] == "MUMPS":
        linear.update(
            {
                "MaxIts": 1,
                "MGMaxLevels": 1,
                "PCMatReal": False,
                "ComplexCoarseSolve": True,
            }
        )
    if (
        numerical["solver_type"] == "Default"
        and numerical["preconditioner"] != "Default"
    ):
        linear["Type"] = numerical["preconditioner"]

    if numerical.get("estimator_mg") is not None:
        linear["EstimatorMG"] = bool(numerical["estimator_mg"])
    return linear


def _build_refinement(numerical: Mapping[str, Any]) -> dict[str, Any]:
    # Palace defaults Model.Refinement.Nonconformal to true if the key is omitted.
    # SCGSim always writes the key. The default is false (conformal tet AMR);
    # notebooks may set amr_nonconformal=True to opt into NC AMR.
    refinement = {
        "UniformLevels": 0,
        "Tol": float(numerical["amr_tolerance"]),
        "MaxIts": int(numerical["amr_max_passes"]),
        "Nonconformal": bool(numerical.get("amr_nonconformal", False)),
    }
    if numerical.get("save_adapt_iterations") is not None:
        refinement["SaveAdaptIterations"] = bool(numerical["save_adapt_iterations"])
    if numerical.get("amr_update_fraction") is not None:
        refinement["UpdateFraction"] = float(numerical["amr_update_fraction"])
    return refinement


def _build_output_formats(numerical: Mapping[str, Any]) -> dict[str, Any]:
    output_formats = {
        "Paraview": numerical.get("output_paraview"),
        "GridFunction": numerical.get("output_grid_function"),
    }
    outputs = {
        key: bool(value) for key, value in output_formats.items() if value is not None
    }
    if not outputs:
        return {}
    return {"OutputFormats": outputs}


def configure_numerical_controls(
    *,
    order: int = 1,
    tolerance: float = 1e-6,
    max_iterations: int = 400,
    solver_type: str = "Default",
    preconditioner: str = "Default",
    device: str = "CPU",
    refined_mesh_size: float = 5.0,
    max_mesh_size: float = 300.0,
    amr_max_passes: int = 0,
    amr_nonconformal: bool = False,
    amr_tolerance: float = 1e-2,
    amr_update_fraction: float | None = None,
    save_adapt_iterations: bool | None = None,
    estimator_mg: bool | None = None,
    output_paraview: bool | None = None,
    output_grid_function: bool | None = None,
) -> dict[str, Any]:
    """Validate notebook numerical controls for Palace config synthesis.

    ``amr_max_passes`` becomes ``Model.Refinement.MaxIts``.
    ``amr_nonconformal`` becomes ``Model.Refinement.Nonconformal`` and defaults
    to ``False`` so generated configs do not inherit Palace's ``true`` default.
    """
    if not isinstance(order, int) or isinstance(order, bool) or not 1 <= order <= 4:
        raise ValueError("order must be an integer from 1 through 4.")
    if not _positive_finite(tolerance):
        raise ValueError("tolerance must be finite and positive.")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer.")
    if solver_type not in {"Default", "SuperLU", "STRUMPACK", "MUMPS"}:
        raise ValueError("unsupported solver_type.")
    if preconditioner not in {"Default", "AMS", "BoomerAMG"}:
        raise ValueError("unsupported preconditioner.")
    if device not in {"CPU", "GPU"}:
        raise ValueError("unsupported device.")
    if solver_type != "Default" and preconditioner != "Default":
        raise ValueError("non-default preconditioner requires Default solver.")
    if solver_type == "MUMPS":
        max_iterations = 1

    if not _positive_finite(refined_mesh_size):
        raise ValueError("refined_mesh_size must be finite and > 0.")
    if not _positive_finite(max_mesh_size):
        raise ValueError("max_mesh_size must be finite and > 0.")
    refined = float(refined_mesh_size)
    maximum = float(max_mesh_size)
    if maximum < refined:
        raise ValueError("max_mesh_size must be >= refined_mesh_size.")

    if not isinstance(amr_max_passes, int) or isinstance(amr_max_passes, bool):
        raise TypeError("amr_max_passes must be an integer.")
    if amr_max_passes < 0:
        raise ValueError("amr_max_passes must be >= 0.")
    if not isinstance(amr_nonconformal, bool):
        raise TypeError("amr_nonconformal must be a bool.")
    if not _positive_finite(amr_tolerance):
        raise ValueError("amr_tolerance must be finite and positive.")
    if amr_update_fraction is not None:
        if not isinstance(amr_update_fraction, Real) or isinstance(
            amr_update_fraction, bool
        ):
            raise TypeError("amr_update_fraction must be a real number.")
        amr_update = float(amr_update_fraction)
        if not math.isfinite(amr_update) or not 0.0 < amr_update < 1.0:
            raise ValueError(
                "amr_update_fraction must be finite and strictly between 0 and 1."
            )
    if save_adapt_iterations is not None and not isinstance(
        save_adapt_iterations, bool
    ):
        raise TypeError("save_adapt_iterations must be bool or None.")
    if estimator_mg is not None and not isinstance(estimator_mg, bool):
        raise TypeError("estimator_mg must be bool or None.")
    if output_paraview is not None and not isinstance(output_paraview, bool):
        raise TypeError("output_paraview must be bool or None.")
    if output_grid_function is not None and not isinstance(output_grid_function, bool):
        raise TypeError("output_grid_function must be bool or None.")

    return {
        "order": int(order),
        "tolerance": float(tolerance),
        "max_iterations": int(max_iterations),
        "solver_type": str(solver_type),
        "preconditioner": str(preconditioner),
        "device": str(device),
        "refined_mesh_size": refined,
        "max_mesh_size": maximum,
        "amr_max_passes": int(amr_max_passes),
        "amr_nonconformal": bool(amr_nonconformal),
        "amr_tolerance": float(amr_tolerance),
        "amr_update_fraction": (
            float(amr_update_fraction) if amr_update_fraction is not None else None
        ),
        "save_adapt_iterations": save_adapt_iterations,
        "estimator_mg": estimator_mg,
        "output_paraview": output_paraview,
        "output_grid_function": output_grid_function,
    }


def _domain_energy_rows(
    domain_volumes: list[tuple[str, Mapping[str, Any]]],
    materials: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        {
            "Index": index,
            "Attributes": sorted(_physical_group_values(info.get("phys_group"))),
        }
        for index, (_, info) in enumerate(domain_volumes, start=1)
    ]
    index_rows = [
        {
            "section": "Domains.Postprocessing.Energy",
            "index": index,
            "entry_name": name,
            "role": "domain_energy",
            "physical_name": name,
            "physical_names": [name],
            "attributes": row["Attributes"],
            "metadata": {
                "physical_attribute": info.get("physical_attribute", {}),
                "material": _material_identity(info, materials),
                "source_provenance": info.get("source_provenance", {}),
                "representation": info.get("representation"),
                "route": info.get("route"),
            },
        }
        for index, ((name, info), row) in enumerate(
            zip(domain_volumes, rows, strict=True), start=1
        )
    ]
    return rows, index_rows


def _eigenmode_lumped_ports(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ports: Sequence[LayoutPortBinding],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    information: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    for port in ports:
        key = f"P{port.index}"
        group = groups.get("port_surfaces", {}).get(key)
        if group is None:
            raise ValueError(f"layout_sheet port {port.name!r} has no SGB group {key}.")
        if not isinstance(group, Mapping):
            raise TypeError(f"SGB layout_sheet group {key} must be a mapping.")
        attribute = group.get("physical_attribute")
        if not isinstance(attribute, Mapping):
            raise TypeError(
                f"SGB layout_sheet port {port.name!r} lacks physical_attribute."
            )
        direction = _direction_tuple(group.get("direction"), port.name)
        if (
            attribute.get("port_index") != port.index
            or attribute.get("port_name") != port.name
            or attribute.get("source_layer") != port.source_layer
            or attribute.get("target_layer") != port.target_layer
            or not all(
                math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                for a, b in zip(direction, port.direction, strict=True)
            )
        ):
            raise ValueError(
                f"layout_sheet port {port.name!r} does not match SGB-authored identity."
            )
        attributes = sorted(_physical_group_values(group.get("phys_group")))
        rows.append(
            {
                "Index": port.index,
                "Direction": list(direction),
                "Excitation": False,
                "Attributes": attributes,
                "L": port.inductance,
            }
        )
        info = {
            "portnumber": port.index,
            "name": port.name,
            "type": "lumped_sheet",
            "direction": list(direction),
            "physical_name": group.get("physical_name"),
            "attributes": attributes,
            "source_layer": port.source_layer,
            "target_layer": port.target_layer,
        }
        information.append(info)
        index_rows.append(
            {
                "section": "Boundaries.LumpedPort",
                "index": port.index,
                "entry_name": key,
                "role": "port_surface",
                "attributes": attributes,
                "physical_names": [group.get("physical_name")],
                "port_name": port.name,
                "metadata": {
                    field: group[field]
                    for field in (
                        "surface_id",
                        "owner_semantic_ids",
                        "adjacent_solution_volume_ids",
                        "embedded_volume_id",
                        "interface_of",
                        "source_provenance",
                        "physical_attribute",
                        "representation",
                        "route",
                    )
                    if field in group
                },
            }
        )
    if len(groups.get("port_surfaces", {})) != len(ports):
        raise ValueError(
            "SGB produced unconfigured or missing layout-sheet port groups."
        )
    return rows, information, index_rows


def _direction_tuple(value: Any, name: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (list, tuple))
        or len(value) != 3
    ):
        raise ValueError(f"layout_sheet port {name!r} requires a 3D direction.")
    direction = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in direction):
        raise ValueError(f"layout_sheet port {name!r} direction must be finite.")
    return direction


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _domains_from_solution_volumes(
    groups: Mapping[str, Mapping[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Sort solver volumes as deterministic domains."""
    return sorted(
        ((name, dict(info)) for name, info in groups.items()), key=lambda item: item[0]
    )


def _materials_from_solution_volumes(
    *,
    domain_volumes: list[tuple[str, Mapping[str, Any]]],
    materials: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build Palace-domain material rows from explicit material records."""
    if not isinstance(materials, Mapping) or not materials:
        raise ValueError(
            "stack must include top-level materials mapping for config generation."
        )

    material_rows: list[dict[str, Any]] = []
    for name, volume in domain_volumes:
        attrs = _physical_group_values(volume.get("phys_group"))
        physical_attribute = volume.get("physical_attribute")
        if not isinstance(physical_attribute, Mapping):
            raise TypeError(f"SGB solution volume {name!r} missing physical_attribute.")
        material_ids = physical_attribute.get("material_ids")
        if not isinstance(material_ids, (list, tuple)) or len(material_ids) != 1:
            raise ValueError(
                f"SGB solution volume {name!r} must define a single material_ids entry."
            )
        material_id = _optional_string(material_ids[0])
        if not material_id:
            raise ValueError(
                f"SGB solution volume {name!r} must define a non-empty material id."
            )
        material = materials.get(material_id)
        if material is None:
            raise ValueError(
                f"SGB solution volume {name!r} references unknown material id {material_id!r}."
            )
        if not isinstance(material, Mapping):
            raise TypeError(f"SGB material record {material_id!r} must be a mapping.")
        if material.get("kind") not in {"vacuum", "dielectric"}:
            raise ValueError(
                f"SGB solution material {material_id!r} must have vacuum or dielectric kind."
            )
        permittivity = material.get("permittivity")
        loss_tangent = material.get("loss_tangent")
        if (
            not isinstance(permittivity, Real)
            or not math.isfinite(float(permittivity))
            or float(permittivity) <= 0.0
        ):
            raise ValueError(
                f"Material {material_id!r} requires finite positive permittivity."
            )
        if (
            not isinstance(loss_tangent, Real)
            or not math.isfinite(float(loss_tangent))
            or float(loss_tangent) < 0.0
        ):
            raise ValueError(
                f"Material {material_id!r} requires finite non-negative loss_tangent."
            )
        if material_rows and [
            x for x in material_rows if list(x["Attributes"]) == list(attrs)
        ]:
            # allow only deterministic duplicate attributes if metadata diverges.
            raise ValueError(
                f"SGB solution volume {name!r} duplicates physical attributes {list(attrs)}."
            )
        material_rows.append(
            {
                "Attributes": sorted(int(item) for item in attrs),
                "Permittivity": float(permittivity),
                "LossTan": float(loss_tangent),
            }
        )

    return material_rows


def _material_identity(
    volume: Mapping[str, Any], materials: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    ids = volume.get("physical_attribute", {}).get("material_ids")
    if (
        not isinstance(ids, (list, tuple))
        or len(ids) != 1
        or not isinstance(ids[0], str)
    ):
        raise ValueError("solution volume must carry one explicit material id.")
    material_id = ids[0]
    material = materials.get(material_id)
    if not isinstance(material, Mapping) or material.get("kind") not in {
        "vacuum",
        "dielectric",
    }:
        raise ValueError(
            f"solution material {material_id!r} must carry vacuum or dielectric kind."
        )
    return {"material_id": material_id, "kind": str(material["kind"])}


def _physical_group_values(value: Any) -> tuple[int, ...]:
    raw = value if isinstance(value, (list, tuple, set)) else (value,)
    values = [
        int(item)
        for item in raw
        if isinstance(item, int) and not isinstance(item, bool)
    ]
    if not values:
        raise ValueError("Physical group must have at least one integer attribute.")
    return tuple(values)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
