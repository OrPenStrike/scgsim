"""Adapted Mesh workflow for Route-A/Route-B structured SGB lowering."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal

GSIM_SHA = "8f5dc6c05255d003a9c6d8959537bcf8068379d3"
SGB_RUNTIME_AUTHORITY = "scgsim.sgb"
SGB_DERIVATION_BASE_SHA = "e74a343154c6b19b6ba32d6fb297e700cfe08ff2"
SGB_DERIVATION_IMPORTED_SHA = "f3fd898d6e4eaf31595c9aaca6a0658f0cb7f3b1"

_STRUCTURED_INTERFACE_TYPES = {"MA", "MS", "SA", "MS_MA", "AA", "MM", "SS"}
_FACE_KIND_SEGMENTS = {"TOP": "top", "BOTTOM": "bottom", "SIDEWALL": "sidewall"}
_STRUCTURED_FACE_KINDS = {
    **_FACE_KIND_SEGMENTS,
    "INTERFACE": "interface",
    "SHEET_CONTACT_CAP": "sheet_contact_cap",
}
_STRUCTURED_SURFACE_FIELDS = frozenset(
    (
        "representation",
        "surface_id",
        "interface_type",
        "contact_kind",
        "face_kind",
        "owner_semantic_ids",
        "adjacent_solution_volume_ids",
        "conductor_component_id",
        "net_id",
        "equipotential_id",
        "source_provenance",
        "physical_attribute",
    )
)


def _scgsim_version() -> str:
    """Return installed scgsim version from package metadata."""
    try:
        return package_version("scgsim")
    except PackageNotFoundError as exc:
        raise RuntimeError("Unable to determine installed scgsim version.") from exc


@dataclass(frozen=True)
class MeshBuildResult:
    """Result of SGB-backed mesh generation."""

    output_dir: Path
    mesh_path: Path
    groups: dict[str, dict[str, Any]]
    mesh_manifest_path: Path
    provenance_path: Path
    stack_path: Path
    xao_path: Path
    sidecar_path: Path


def build_route_mesh(
    *,
    component: Any,
    stack: Mapping[str, Any],
    route: str,
    output_dir: str | Path,
    refined_mesh_size: float = 5.0,
    max_mesh_size: float = 300.0,
    port_sheet_source_layers: Sequence[Mapping[str, Any]] = (),
) -> MeshBuildResult:
    """Lower Route-A/Route-B SGB geometry, mesh it, and emit artifacts."""

    if not isinstance(stack, Mapping):
        raise TypeError("stack must be a mapping after set_stack processing.")
    if route not in {"A", "B"}:
        raise ValueError("route must be either 'A' or 'B'.")
    if (
        refined_mesh_size <= 0
        or max_mesh_size <= 0
        or max_mesh_size < refined_mesh_size
    ):
        raise ValueError(
            "mesh sizes must be positive and max_mesh_size >= refined_mesh_size"
        )

    stack = _with_port_sheet_metadata(stack, port_sheet_source_layers)
    run_dir = Path(output_dir).expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError(f"output_dir must be empty before mesh: {run_dir}")
    geometry_dir = run_dir / "geometry"
    semantic_dir = run_dir / "metadata" / "semantic_geometry"

    for path in (
        run_dir,
        geometry_dir,
        semantic_dir,
        run_dir / "logs",
        run_dir / "results" / "palace",
    ):
        path.mkdir(parents=True, exist_ok=True)

    gds_path = geometry_dir / "design.gds"
    stack_path = geometry_dir / "design.stack.json"
    xao_path = geometry_dir / f"semantic_geometry_route_{route.lower()}.xao"
    sidecar_path = semantic_dir / "04_export_physical_groups.json"
    mesh_path = run_dir / "palace.msh"
    manifest_path = run_dir / "metadata" / "mesh_manifest.json"
    provenance_path = run_dir / "metadata" / f"route_{route.lower()}_provenance.json"

    _write_component_gds(component=component, path=gds_path)
    top_cell = _component_gds_top_cell_name(component=component, gds_path=gds_path)
    stack_path.write_text(json.dumps(dict(stack), indent=2) + "\n", encoding="utf-8")

    from scgsim.sgb import SemanticGeometryBuilder, build_gds_stack_geometry_input

    build_input = build_gds_stack_geometry_input(
        gds_file=gds_path,
        stack_file=stack_path,
        top_cell_name=top_cell,
        metadata={
            "scgsim": {"version": _scgsim_version(), "route": route},
            "source": "scgsim.palace._mesh.build_route_mesh",
        },
    )
    SemanticGeometryBuilder().build(
        build_input,
        route=route,
        run_folder=run_dir,
    )

    if not xao_path.is_file():
        raise FileNotFoundError(f"SGB Route-{route} XAO not generated: {xao_path}")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"SGB Route-{route} sidecar not generated: {sidecar_path}"
        )

    records = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError("04_export_physical_groups.json must contain a list")

    mesh_path, groups = _mesh_from_route_xao(
        xao_path=xao_path,
        route=route,
        records=records,
        stack=stack,
        mesh_path=mesh_path,
        refined_mesh_size=refined_mesh_size,
        max_mesh_size=max_mesh_size,
    )

    manifest = _build_mesh_manifest(
        groups=groups,
        mesh_name=mesh_path.name,
        route=route,
    )
    _write_json(manifest_path, manifest)
    _write_json(
        provenance_path,
        {
            "scgsim_version": _scgsim_version(),
            "route": route,
            "source_sha": {
                "gsim": GSIM_SHA,
                "scgsim_sgb": SGB_RUNTIME_AUTHORITY,
                "sgb_derivation": {
                    "base": SGB_DERIVATION_BASE_SHA,
                    "imported_development": SGB_DERIVATION_IMPORTED_SHA,
                },
            },
            "component_top_cell": top_cell,
            "run_dir": ".",
            "stack_path": str(stack_path.relative_to(run_dir)),
            "xao_path": str(xao_path.relative_to(run_dir)),
            "sidecar_path": str(sidecar_path.relative_to(run_dir)),
        },
    )

    return MeshBuildResult(
        output_dir=run_dir,
        mesh_path=mesh_path,
        groups=groups,
        mesh_manifest_path=manifest_path,
        provenance_path=provenance_path,
        stack_path=stack_path,
        xao_path=xao_path,
        sidecar_path=sidecar_path,
    )


def _mesh_from_route_xao(
    *,
    xao_path: Path,
    route: str,
    records: Sequence[Any],
    stack: Mapping[str, Any],
    mesh_path: Path,
    refined_mesh_size: float,
    max_mesh_size: float,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Open SGB XAO and map live physical groups with strict structured checks."""

    route = route.upper()
    _sgb_records_route_context(records, route)
    groups = _build_groups_from_sgb_records()
    solution_material_kinds = _solution_material_kinds(records, stack)

    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    import gmsh

    gmsh.initialize()
    try:
        gmsh.open(str(xao_path))
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", refined_mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_mesh_size)

        route = _sgb_records_route_context(records, route)
        live = _live_physical_groups()
        _augment_groups_from_live_records(
            records=records,
            groups=groups,
            live=live,
            route=route,
            solution_material_kinds=solution_material_kinds,
        )
        if not groups["volumes"]:
            raise ValueError(
                "SGB XAO did not expose any solver-active solution-volume groups. "
                "Check that 04_export_physical_groups.json matches the XAO."
            )

        _embed_route_b_port_surfaces(groups)
        _setup_xao_refinement(groups, refined_mesh_size, max_mesh_size)
        if groups["port_surfaces"]:
            gmsh.option.setNumber("Mesh.Algorithm3D", 4)
        gmsh.model.mesh.generate(3)
        _validate_tetrahedral_solution_groups(groups=groups, live=live)
        _validate_port_pec_node_intersections(groups)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(mesh_path))
        return mesh_path, groups
    finally:
        gmsh.clear()
        gmsh.finalize()


def _build_groups_from_sgb_records() -> dict[str, dict[str, Any]]:
    """Pre-allocate group dicts before physical mapping enrichment."""

    return {
        "volumes": {},
        "boundary_surfaces": {},
        "surface_groups": {},
        "pec_surfaces": {},
        "port_surfaces": {},
    }


def _augment_groups_from_live_records(
    *,
    records: Sequence[Any],
    groups: dict[str, dict[str, Any]],
    live: Mapping[tuple[int, str], tuple[int, tuple[int, ...]]],
    route: str,
    solution_material_kinds: Mapping[str, str],
) -> None:
    """Populate mesh groups by exact (dimension, physical_name) mapping."""
    route = route.upper()
    missing: list[str] = []

    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("SGB physical group records must be JSON objects.")
        if _optional_string(record.get("route", "")) != route:
            raise ValueError(f"SGB physical-group record route must be {route}.")
        if (
            record.get("sgb_record")
            not in {None, "final_physical_group", "preliminary_physical_group"}
            and str(record.get("solver_use")) == "solver_active"
        ):
            missing.append("unknown_sgb_record")

    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("SGB physical group records must be JSON objects.")
        dim = int(record.get("dimension", 0) or 0)
        name = _optional_string(record.get("physical_name"))
        if not name:
            if record.get("solver_use") == "solver_active":
                missing.append(f"{dim}:<missing physical_name>")
            continue

        if _optional_string(record.get("route", "")) != route:
            raise ValueError(
                f"SGB physical-group route must be {route} for this mesh mode."
            )
        if record.get("solver_use") == "not_solver_active":
            continue

        if (dim, name) not in live:
            if str(record.get("solver_use", "")).lower() == "solver_active":
                missing.append(f"{dim}:{name}")
            continue

        if dim == 3 and not _is_solver_active_volume(record):
            raise ValueError(
                f"SGB physical volume {name!r} is not an explicit solver-active "
                "material_volume and cannot enter the Palace solution."
            )
        if dim == 2 and record.get("role") == "material_volume":
            raise ValueError(f"SGB surface {name!r} cannot claim material_volume role.")

        phys_group, entity_tags = live[(dim, name)]
        if dim == 3 and _is_solver_active_volume(record):
            _add_volume_group(
                groups=groups,
                name=name,
                phys_group=phys_group,
                entity_tags=entity_tags,
                record=record,
            )
            continue

        if dim == 2:
            _add_surface_group_records(
                groups=groups,
                name=name,
                phys_group=phys_group,
                entity_tags=entity_tags,
                record=record,
                route=route,
                solution_material_kinds=solution_material_kinds,
            )

    if missing:
        raise ValueError(
            "SGB XAO is missing solver-active physical groups from sidecar: "
            + ", ".join(sorted(missing))
        )


def _add_volume_group(
    *,
    groups: dict[str, dict[str, Any]],
    name: str,
    phys_group: int,
    entity_tags: tuple[int, ...],
    record: Mapping[str, Any],
) -> None:
    """Add one solver-active solution volume."""
    _validate_structured_volume_record(name, record)
    role = str(record.get("role", "")).lower()
    if "construction" in role or role == "mm_contact":
        raise ValueError(
            f"SGB volume {name!r} has construction-only/MM-contact role {role!r} "
            "and must not become a solver volume."
        )

    info = {
        "phys_group": phys_group,
        "tags": list(entity_tags),
        "dim": 3,
        "source": SGB_RUNTIME_AUTHORITY,
        "physical_name": name,
        "stack_layer": record.get("source_provenance", {}).get("stack_layer"),
        "source_layer": record.get("source_provenance", {}).get(
            "conductor_source_layer_name"
        ),
        "representation": record.get("representation"),
        "route": record.get("route"),
        "source_provenance": dict(record.get("source_provenance", {})),
        "physical_attribute": dict(record.get("physical_attribute", {})),
        "sgb_record": "final_physical_group",
        "solver_use": record.get("solver_use", "solver_active"),
        "role": record.get("role"),
        "owner_semantic_ids": tuple(record.get("owner_semantic_ids", ())),
        "adjacent_solution_volume_ids": tuple(
            record.get("adjacent_solution_volume_ids", ())
        ),
        "metadata": dict(record.get("metadata", {})),
    }

    groups["volumes"][name] = info


def _add_surface_group_records(
    *,
    groups: dict[str, dict[str, Any]],
    name: str,
    phys_group: int,
    entity_tags: tuple[int, ...],
    record: Mapping[str, Any],
    route: str,
    solution_material_kinds: Mapping[str, str],
) -> None:
    """Classify structured surface records into boundary + interface mappings."""
    if record.get("solver_use") == "not_solver_active":
        return

    role = str(record.get("role", "")).lower()
    metadata = record.get("metadata")
    source_record_ids = (
        tuple(
            value
            for value in (
                metadata.get("source_record_ids", ())
                if isinstance(metadata, Mapping)
                else ()
            )
            if isinstance(value, str) and value
        )
        if isinstance(metadata, Mapping)
        else ()
    )

    if role == "domain_boundary":
        _validate_structured_domain_boundary_record(name, record)
        info = {
            "phys_group": phys_group,
            "tags": list(entity_tags),
            "dim": 2,
            "source": "domain_boundary",
            "surface_epr": False,
            "representation": str(record.get("representation", "")),
            "route": record.get("route"),
            "interface_type": record.get("interface_type"),
            "contact_kind": record.get("contact_kind"),
            "face_kind": _optional_string(record.get("face_kind")),
            "source_id": _optional_string(record.get("surface_id")) or name,
            "owner_semantic_ids": tuple(record.get("owner_semantic_ids", ())),
            "adjacent_solution_volume_ids": tuple(
                record.get("adjacent_solution_volume_ids", ())
            ),
            "surface_id": _optional_string(record.get("surface_id")),
            "physical_attribute": dict(record.get("physical_attribute", {})),
            "source_provenance": dict(record.get("source_provenance", {})),
            "conductor_component_id": _optional_string(
                record.get("conductor_component_id")
            ),
            "net_id": _optional_string(record.get("net_id")),
            "equipotential_id": _optional_string(record.get("equipotential_id")),
            "sgb_record": "final_physical_group",
            "solver_use": record.get("solver_use", "solver_active"),
            "sgb_metadata": dict(record.get("metadata", {})),
            "source_record_ids": source_record_ids,
        }
        groups["boundary_surfaces"][name] = info
        return

    if not _has_structured_record_marker(record):
        if str(record.get("solver_use", "")).lower() == "solver_active":
            raise ValueError(f"SGB surface {name!r} is missing structured metadata.")
        groups["surface_groups"][name] = {"physical_name": name, "dim": 2}
        return

    if str(record.get("solver_use")) == "not_solver_active":
        return

    interface_type = str(record.get("interface_type", "")).upper()
    if interface_type == "LUMPED_PORT" or role in {"lumped_port", "port_surface"}:
        _add_port_surface_group(
            groups=groups,
            name=name,
            phys_group=phys_group,
            entity_tags=entity_tags,
            record=record,
            route=route,
        )
        return
    _validate_structured_surface_record(name=name, record=record)
    _validate_route_representation(name=name, record=record, route=route)
    if interface_type not in _STRUCTURED_INTERFACE_TYPES:
        raise ValueError(
            f"SGB surface {name!r} has unsupported interface_type {interface_type!r}."
        )

    component_id = _optional_string(record.get("conductor_component_id"))
    net_id = _optional_string(record.get("net_id"))
    if interface_type in {"MA", "MS", "MS_MA"} and (
        component_id is None or net_id is None
    ):
        raise ValueError(
            f"SGB conductor surface {name!r} must include conductor_component_id and net_id."
        )
    if interface_type == "SA" and net_id is not None:
        raise ValueError(
            f"SGB SA surface {name!r} must not carry conductor/net identity."
        )

    if interface_type in {"MM", "SS", "AA"}:
        groups["boundary_surfaces"][name] = _structured_surface_info(
            name=name,
            phys_group=phys_group,
            entity_tags=entity_tags,
            record=record,
            interface_type=interface_type,
            surface_epr=False,
        )
        return
    _validate_surface_adjacency(name, record, interface_type, solution_material_kinds)
    surface_id = str(record["surface_id"])
    if interface_type == "MS_MA":
        source_id = _structured_conductor_source_layer(name=name, record=record)
        # One SGB physical attribute carries both semantic aliases.  Do not use
        # geometric orientation or material names to invent top/bottom meaning.
        parsed_records = (
            ("MS", source_id, "interface", f"{surface_id}__MS"),
            ("MA", source_id, "interface", f"{surface_id}__MA"),
        )
    else:
        source_id = (
            surface_id
            if interface_type == "SA"
            else _structured_conductor_source_layer(name=name, record=record)
        )
        parsed_records = (
            (interface_type, source_id, str(record["face_kind"]), surface_id),
        )

    for parsed in parsed_records:
        interface_type, source_id, face_kind, alias_name = parsed
        if interface_type not in _STRUCTURED_INTERFACE_TYPES:
            continue
        alias_info = _structured_surface_info(
            name=alias_name,
            phys_group=phys_group,
            entity_tags=entity_tags,
            record=record,
            interface_type=interface_type,
            surface_epr=interface_type in {"MA", "MS", "SA"},
            source_id=source_id,
            face_kind=face_kind,
        )
        groups["boundary_surfaces"][alias_name] = alias_info
        if interface_type in {"MA", "MS"} and component_id is not None:
            groups["pec_surfaces"][alias_name] = {
                "phys_group": phys_group,
                "tags": list(entity_tags),
                "dim": 2,
                "source": SGB_RUNTIME_AUTHORITY,
                "source_id": source_id,
                "layer": source_id,
                "representation": str(record.get("representation", "")).upper(),
                "owner_semantic_ids": tuple(record.get("owner_semantic_ids", ())),
            }


def _add_port_surface_group(
    *,
    groups: dict[str, dict[str, Any]],
    name: str,
    phys_group: int,
    entity_tags: tuple[int, ...],
    record: Mapping[str, Any],
    route: str,
) -> None:
    """Preserve one SGB-authored layout sheet without reconstructing its meaning."""
    if record.get("solver_use") != "solver_active":
        raise ValueError(f"SGB lumped-port sheet {name!r} must be solver_active.")
    if record.get("representation") != "lumped_port_sheet":
        raise ValueError(
            f"SGB lumped-port sheet {name!r} requires lumped_port_sheet representation."
        )
    surface_id = _required_string(record.get("surface_id"), f"{name} surface_id")
    owners = _required_distinct_strings(
        record.get("owner_semantic_ids"), f"{name} owner_semantic_ids", count=2
    )
    adjacent = _required_distinct_strings(
        record.get("adjacent_solution_volume_ids"),
        f"{name} adjacent_solution_volume_ids",
        count=2 if route == "A" else 1,
    )
    attribute = record.get("physical_attribute")
    provenance = record.get("source_provenance")
    if not isinstance(attribute, Mapping) or not isinstance(provenance, Mapping):
        raise TypeError(
            f"SGB lumped-port sheet {name!r} requires physical_attribute and source_provenance."
        )
    port_index = attribute.get("port_index")
    if (
        isinstance(port_index, bool)
        or not isinstance(port_index, int)
        or port_index < 1
    ):
        raise ValueError(f"SGB lumped-port sheet {name!r} requires 1-based port_index.")
    port_name = _required_string(attribute.get("port_name"), f"{name} port_name")
    source_layer = _required_string(
        attribute.get("source_layer"), f"{name} source_layer"
    )
    target_layer = _required_string(
        attribute.get("target_layer"), f"{name} target_layer"
    )
    embedded_volume_id = _required_string(
        attribute.get("embedded_volume_id"), f"{name} embedded_volume_id"
    )
    direction = _normalized_direction(attribute.get("direction"), name)
    attribute_owners = _required_distinct_strings(
        attribute.get("owner_semantic_ids"),
        f"{name} physical_attribute.owner_semantic_ids",
        count=2,
    )
    if attribute_owners != owners:
        raise ValueError(f"SGB lumped-port sheet {name!r} owner identities disagree.")
    owner_provenance = attribute.get("owner_provenance")
    if not isinstance(owner_provenance, (list, tuple)) or len(owner_provenance) != 2:
        raise ValueError(
            f"SGB lumped-port sheet {name!r} requires two owner provenance rows."
        )
    for item in owner_provenance:
        if not isinstance(item, Mapping):
            raise TypeError(
                f"SGB lumped-port sheet {name!r} owner provenance must be mappings."
            )
        owner_id = _required_string(
            item.get("semantic_id"), f"{name} owner semantic_id"
        )
        if owner_id not in owners:
            raise ValueError(
                f"SGB lumped-port sheet {name!r} has foreign owner provenance."
            )
        for field in ("conductor_component_id", "net_id", "equipotential_id"):
            value = item.get(field)
            if value is not None:
                _required_string(value, f"{name} owner {field}")
    if route == "A":
        if embedded_volume_id not in adjacent:
            raise ValueError(
                f"SGB Route-A lumped-port sheet {name!r} embedded volume must be in actual adjacency."
            )
    elif adjacent != (embedded_volume_id,):
        raise ValueError(
            f"SGB Route-B lumped-port sheet {name!r} requires one matching embedded volume."
        )
    if (
        provenance.get("source_name") != port_name
        or provenance.get("port_index") != port_index
        or provenance.get("source_layer") != source_layer
        or provenance.get("target_layer") != target_layer
        or _normalized_direction(provenance.get("direction"), name) != direction
        or not _optional_string(provenance.get("source_polygon_id"))
        or not _optional_string(provenance.get("direction_sign_convention"))
    ):
        raise ValueError(
            f"SGB lumped-port sheet {name!r} source provenance is inconsistent."
        )
    key = f"P{port_index}"
    if key in groups["port_surfaces"]:
        raise ValueError(f"Duplicate SGB lumped-port sheet index {port_index}.")
    groups["port_surfaces"][key] = {
        "phys_group": phys_group,
        "tags": list(entity_tags),
        "dim": 2,
        "type": "lumped_sheet",
        "source": SGB_RUNTIME_AUTHORITY,
        "surface_epr": False,
        "sgb_record": "final_physical_group",
        "solver_use": record.get("solver_use"),
        "role": record.get("role"),
        "route": route,
        "representation": record.get("representation"),
        "surface_id": surface_id,
        "interface_type": "lumped_port",
        "face_kind": record.get("face_kind"),
        "owner_semantic_ids": owners,
        "adjacent_solution_volume_ids": adjacent,
        "interface_of": adjacent if route == "A" else (),
        "embedded_volume_id": embedded_volume_id,
        "direction": direction,
        "physical_name": name,
        "source_provenance": dict(provenance),
        "physical_attribute": dict(attribute),
        "conductor_component_id": record.get("conductor_component_id"),
        "net_id": record.get("net_id"),
        "equipotential_id": record.get("equipotential_id"),
    }


def _setup_xao_refinement(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    refined_cellsize: float,
    max_cellsize: float,
) -> None:
    """Apply line-based mesh refinement from Route-B structured boundary lines."""
    import gmsh

    lines: set[int] = set()
    for surface_group in ("boundary_surfaces", "pec_surfaces", "port_surfaces"):
        for info in groups.get(surface_group, {}).values():
            for tag in info.get("tags", ()):  # pragma: no branch
                try:
                    boundary = gmsh.model.getBoundary(
                        [(2, int(tag))],
                        combined=False,
                        oriented=False,
                        recursive=False,
                    )
                except RuntimeError:
                    continue
                lines.update(int(btag) for bdim, btag in boundary if bdim == 1)
    if not lines:
        return

    field_id = _setup_mesh_refinement(
        sorted(lines),
        refined_cellsize,
        max_cellsize,
    )
    _finalize_mesh_fields([field_id])


def _embed_route_b_port_surfaces(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """Embed Route-B sheets in the exact SGB-authored solution volume."""
    import gmsh

    for key, port in groups.get("port_surfaces", {}).items():
        if port.get("route") != "B":
            continue
        volume_id = _required_string(
            port.get("embedded_volume_id"), f"{key} embedded_volume_id"
        )
        volume = _solution_volume_by_stable_id(groups.get("volumes", {}), volume_id)
        surface_tags = [int(tag) for tag in port.get("tags", ())]
        volume_tags = [int(tag) for tag in volume.get("tags", ())]
        if not surface_tags or not volume_tags:
            raise ValueError(f"SGB Route-B lumped-port sheet {key!r} has no live tags.")
        for volume_tag in volume_tags:
            gmsh.model.mesh.embed(2, surface_tags, 3, volume_tag)


def _solution_volume_by_stable_id(
    volumes: Mapping[str, Mapping[str, Any]], stable_id: str
) -> Mapping[str, Any]:
    matches: list[Mapping[str, Any]] = []
    for volume in volumes.values():
        provenance = volume.get("source_provenance")
        explicit_ids = set(volume.get("owner_semantic_ids", ()))
        if isinstance(provenance, Mapping):
            explicit_ids.update(provenance.get("volume_ids", ()))
        if stable_id in explicit_ids:
            matches.append(volume)
    if len(matches) != 1:
        raise ValueError(
            f"SGB embedded solution id {stable_id!r} must resolve exactly one structured volume."
        )
    return matches[0]


def _setup_mesh_refinement(
    boundary_line_tags: list[int],
    refined_cellsize: float,
    max_cellsize: float,
    *,
    sampling: int = 200,
    dist_min: float = 0.0,
    dist_max: float | None = None,
) -> int:
    """Set up gmsh size-transition field around boundary lines."""
    import gmsh

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", boundary_line_tags)
    gmsh.model.mesh.field.setNumber(1, "Sampling", int(sampling))
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", refined_cellsize)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", max_cellsize)
    gmsh.model.mesh.field.setNumber(2, "DistMin", dist_min)
    gmsh.model.mesh.field.setNumber(
        2,
        "DistMax",
        max_cellsize if dist_max is None else float(dist_max),
    )
    return 2


def _finalize_mesh_fields(field_ids: list[int]) -> None:
    """Finalize gmsh background field setup from one or more fields."""
    import gmsh

    min_field_id = max(field_ids) + 1
    gmsh.model.mesh.field.add("Min", min_field_id)
    gmsh.model.mesh.field.setNumbers(min_field_id, "FieldsList", field_ids)
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field_id)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)


def _solution_material_kinds(
    records: Sequence[Any], stack: Mapping[str, Any]
) -> dict[str, str]:
    """Map SGB solution-volume stable IDs to explicit stack material kinds."""
    materials = stack.get("materials")
    if not isinstance(materials, Mapping):
        raise TypeError("stack must carry explicit materials while meshing.")
    result: dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping) or not _is_solver_active_volume(record):
            continue
        physical = record.get("physical_attribute")
        provenance = record.get("source_provenance")
        ids = physical.get("material_ids") if isinstance(physical, Mapping) else None
        kinds = (
            physical.get("material_kinds") if isinstance(physical, Mapping) else None
        )
        volume_ids = (
            provenance.get("volume_ids") if isinstance(provenance, Mapping) else None
        )
        owner_ids = record.get("owner_semantic_ids")
        if not all(
            isinstance(value, (list, tuple)) and value
            for value in (ids, kinds, volume_ids, owner_ids)
        ):
            raise ValueError(
                "SGB solution volume needs explicit material_ids/material_kinds, provenance.volume_ids, and owner_semantic_ids."
            )
        if not (len(ids) == len(kinds) == len(volume_ids) == len(owner_ids)):
            raise ValueError(
                "SGB solution volume material kind and stable-id cardinalities must agree."
            )
        for material_id, material_kind, volume_id, owner_id in zip(
            ids, kinds, volume_ids, owner_ids, strict=True
        ):
            if not all(
                isinstance(value, str) and value
                for value in (material_id, material_kind, volume_id, owner_id)
            ):
                raise ValueError(
                    "SGB solution volume stable ids, material ids, and material kinds must be non-empty strings."
                )
            material = materials.get(material_id)
            stack_kind = material.get("kind") if isinstance(material, Mapping) else None
            if stack_kind not in {"vacuum", "dielectric"}:
                raise ValueError(
                    "SGB solution volume material must resolve vacuum or dielectric kind."
                )
            if material_kind not in {"vacuum", "dielectric"}:
                raise ValueError(
                    "SGB solution volume material_kinds must be vacuum or dielectric."
                )
            if material_kind != stack_kind:
                raise ValueError(
                    "SGB solution volume material_kind disagrees with explicit stack material kind."
                )
            for stable_id in (volume_id, owner_id):
                previous = result.setdefault(stable_id, material_kind)
                if previous != material_kind:
                    raise ValueError(
                        f"Structured solution id {stable_id!r} has conflicting material kinds."
                    )
    if not result:
        raise ValueError(
            "SGB sidecar has no explicit solution-volume material records."
        )
    return result


def _validate_route_representation(
    *, name: str, record: Mapping[str, Any], route: str
) -> None:
    representation = _optional_string(record.get("representation"))
    if str(record.get("interface_type", "")).upper() == "SA":
        if representation != "solution_surface":
            raise ValueError(
                f"SGB solution interface {name!r} must retain solution_surface representation."
            )
        return
    allowed = (
        {"surface_sheet", "cutout_boundary_shell"}
        if route == "A"
        else {"cutout_boundary_shell"}
    )
    if representation not in allowed:
        raise ValueError(
            f"SGB Route-{route} surface {name!r} representation {representation!r} is incompatible."
        )


def _validate_surface_adjacency(
    name: str,
    record: Mapping[str, Any],
    interface_type: str,
    material_kinds: Mapping[str, str],
) -> None:
    adjacent = record.get("adjacent_solution_volume_ids")
    if not isinstance(adjacent, (list, tuple)) or not adjacent:
        raise ValueError(
            f"SGB surface {name!r} has no explicit adjacent solution volumes."
        )
    kinds: list[str] = []
    for identifier in adjacent:
        if not isinstance(identifier, str) or identifier not in material_kinds:
            raise ValueError(
                f"SGB surface {name!r} adjacent solution id {identifier!r} cannot resolve explicit material kind."
            )
        kinds.append(material_kinds[identifier])
    if interface_type == "MS_MA":
        if str(record.get("face_kind", "")).lower() != "interface" or sorted(kinds) != [
            "dielectric",
            "vacuum",
        ]:
            raise ValueError(
                f"SGB MS_MA surface {name!r} requires face_kind=interface and exactly dielectric/vacuum adjacency."
            )
    elif interface_type in {"MA", "MS", "SA"}:
        # The required material pairing is an explicit semantic compatibility
        # check, never a geometric orientation or material-name inference.
        expected = {"MA": "vacuum", "MS": "dielectric", "SA": "vacuum"}[interface_type]
        if interface_type == "SA":
            if sorted(kinds) != ["dielectric", "vacuum"]:
                raise ValueError(
                    f"SGB SA surface {name!r} requires exactly dielectric/vacuum adjacency."
                )
        elif expected not in kinds:
            raise ValueError(
                f"SGB {interface_type} surface {name!r} lacks explicit adjacent {expected} solution material."
            )


def _structured_surface_info(
    *,
    name: str,
    phys_group: int,
    entity_tags: tuple[int, ...],
    record: Mapping[str, Any],
    interface_type: str,
    surface_epr: bool,
    source_id: str | None = None,
    face_kind: str | None = None,
) -> dict[str, Any]:
    """Keep every structured identity field when lowering a surface group."""
    return {
        "phys_group": phys_group,
        "tags": list(entity_tags),
        "dim": 2,
        "source": "volume_interface",
        "surface_epr": surface_epr,
        "interface_id": name,
        "interface_type": interface_type,
        "sgb_record": "final_physical_group",
        "solver_use": record.get("solver_use"),
        "conductor_component_id": _optional_string(
            record.get("conductor_component_id")
        ),
        "net_id": _optional_string(record.get("net_id")),
        "equipotential_id": _optional_string(record.get("equipotential_id")),
        "physical_attribute": dict(record.get("physical_attribute", {})),
        "source_id": source_id or _optional_string(record.get("surface_id")) or name,
        "face_kind": face_kind or _optional_string(record.get("face_kind")),
        "contact_kind": _optional_string(record.get("contact_kind")),
        "representation": _optional_string(record.get("representation")),
        "route": _optional_string(record.get("route")),
        "source_provenance": dict(record.get("source_provenance", {})),
        "source_record_ids": tuple(
            record.get("metadata", {}).get("source_record_ids", ())
        )
        if isinstance(record.get("metadata"), Mapping)
        else (),
        "surface_id": _optional_string(record.get("surface_id")) or name,
        "owner_semantic_ids": tuple(record.get("owner_semantic_ids", ())),
        "adjacent_solution_volume_ids": tuple(
            record.get("adjacent_solution_volume_ids", ())
        ),
    }


def _validate_tetrahedral_solution_groups(
    *,
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    live: Mapping[tuple[int, str], tuple[int, tuple[int, ...]]],
) -> None:
    """Require exact live solver volume ownership and at least one 3D element each."""
    import gmsh

    expected = set(groups.get("volumes", {}))
    live_volume_names = {name for (dim, name) in live if dim == 3}
    if live_volume_names != expected:
        raise ValueError(
            "XAO live 3D physical groups differ from structured solution volumes: "
            f"live={sorted(live_volume_names)}, solution={sorted(expected)}."
        )
    for name, info in groups.get("volumes", {}).items():
        has_elements = False
        for tag in info["tags"]:
            element_types, element_tags, _ = gmsh.model.mesh.getElements(3, int(tag))
            for element_type in element_types:
                element_name, dimension, *_ = gmsh.model.mesh.getElementProperties(
                    element_type
                )
                if dimension != 3 or "tetrahedron" not in element_name.lower():
                    raise ValueError(
                        f"Solver solution volume {name!r} has non-tetrahedral 3D element {element_name!r}."
                    )
            has_elements = has_elements or any(len(items) for items in element_tags)
        if not has_elements:
            raise ValueError(
                f"Solver solution volume {name!r} has no 3D mesh elements."
            )


def _validate_port_pec_node_intersections(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    """Require each SGB port terminal edge to share actual nodes with its PEC owner."""
    for key, port in groups.get("port_surfaces", {}).items():
        port_nodes = _surface_nodes(port.get("tags", ()))
        if not port_nodes:
            raise ValueError(f"SGB lumped-port sheet {key!r} has no mesh nodes.")
        for owner_id in port.get("owner_semantic_ids", ()):
            owner_tags = {
                int(tag)
                for info in groups.get("pec_surfaces", {}).values()
                if owner_id in info.get("owner_semantic_ids", ())
                for tag in info.get("tags", ())
            }
            if not owner_tags:
                raise ValueError(
                    f"SGB lumped-port sheet {key!r} owner {owner_id!r} has no PEC surface."
                )
            if len(port_nodes.intersection(_surface_nodes(owner_tags))) < 2:
                raise ValueError(
                    f"SGB lumped-port sheet {key!r} owner {owner_id!r} does not share a mesh edge."
                )


def _surface_nodes(tags: Sequence[Any]) -> set[int]:
    import gmsh

    nodes: set[int] = set()
    for tag in tags:
        node_tags, _, _ = gmsh.model.mesh.getNodes(
            2, int(tag), includeBoundary=True, returnParametricCoord=False
        )
        nodes.update(int(node) for node in node_tags)
    return nodes


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sgb_records_route_context(
    records: Sequence[Any], expected_route: str | None = None
) -> Literal["A", "B"]:
    """Require one explicit structured SGB Route A/B context."""
    routes: set[str] = set()
    if expected_route is not None:
        expected_route = expected_route.upper()
        if expected_route not in {"A", "B"}:
            raise ValueError("expected_route must be 'A' or 'B'.")
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("SGB physical group records must be JSON objects.")
        route = _optional_string(record.get("route"))
        if route not in {"A", "B"}:
            raise ValueError(
                "SGB physical-group records require explicit route A or B."
            )
        routes.add(route)
    if expected_route is not None and routes != {expected_route}:
        raise ValueError(
            f"SGB physical-group records must all use route {expected_route}, got {sorted(routes)}."
        )
    if len(routes) != 1:
        raise ValueError("SGB physical-group records must all use a single route.")
    return routes.pop()


def _is_solver_active_volume(record: Mapping[str, Any]) -> bool:
    return (
        int(record.get("dimension", 0) or 0) == 3
        and record.get("role") == "material_volume"
        and record.get("interface_type") == "material_volume"
        and record.get("solver_use", "solver_active") == "solver_active"
    )


def _has_structured_record_marker(record: Mapping[str, Any]) -> bool:
    return any(field in record for field in _STRUCTURED_SURFACE_FIELDS)


def _live_physical_groups() -> dict[tuple[int, str], tuple[int, tuple[int, ...]]]:
    """Map live gmsh (dimension, physical name) -> (physical-group, tags)."""
    import gmsh

    live: dict[tuple[int, str], tuple[int, tuple[int, ...]]] = {}
    for dim, phys_group in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, phys_group)
        if not name:
            continue
        live[(int(dim), str(name))] = (
            int(phys_group),
            tuple(
                int(tag)
                for tag in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_group)
            ),
        )
    return live


def _build_mesh_manifest(
    *,
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    mesh_name: str,
    route: str,
) -> dict[str, Any]:
    """Build a compact manifest from structured groups only."""
    entries: list[dict[str, Any]] = []

    for section in (
        "volumes",
        "boundary_surfaces",
        "surface_groups",
        "pec_surfaces",
        "port_surfaces",
    ):
        for name, info in groups.get(section, {}).items():
            if not isinstance(info, Mapping):
                continue
            entry = {
                "name": name,
                "section": section,
                "dimension": int(info.get("dim", 0) or 0),
                "tags": list(info.get("tags", ())),
                "attributes": [
                    int(item) for item in _physical_group_values(info.get("phys_group"))
                ],
                "structured": bool(info.get("sgb_record") == "final_physical_group"),
                "source": str(info.get("source", "")),
            }
            for key in (
                "route",
                "role",
                "solver_use",
                "representation",
                "surface_id",
                "interface_type",
                "face_kind",
                "owner_semantic_ids",
                "adjacent_solution_volume_ids",
                "conductor_component_id",
                "net_id",
                "equipotential_id",
                "source_provenance",
                "physical_attribute",
                "contact_kind",
                "source_id",
                "interface_of",
                "embedded_volume_id",
                "direction",
            ):
                if key in info:
                    entry[key] = info[key]
            entries.append(entry)

    return {
        "schema_version": 1,
        "route": route,
        "mesh_file": mesh_name,
        "source": {
            "gsim": GSIM_SHA,
            "scgsim_sgb": SGB_RUNTIME_AUTHORITY,
            "sgb_derivation": {
                "base": SGB_DERIVATION_BASE_SHA,
                "imported_development": SGB_DERIVATION_IMPORTED_SHA,
            },
        },
        "groups": entries,
    }


def _write_component_gds(component: Any, path: Path) -> None:
    """Write the component's GDS, supporting both string and pathlib styles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_gds = getattr(component, "write_gds", None)
    if write_gds is None or not callable(write_gds):
        raise TypeError("component must provide write_gds(path)")

    try:
        write_gds(path)
    except TypeError:
        write_gds(str(path))


def _with_port_sheet_metadata(
    stack: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    """Attach authored ports after marking their exact face-metal owners splittable."""
    payload = copy.deepcopy(dict(stack))
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("stack metadata must be a mapping.")
    metadata = dict(metadata)
    existing = metadata.get("port_sheet_source_layers")
    if existing not in (None, [], ()) and records:
        raise ValueError("stack already defines port_sheet_source_layers.")
    if records:
        layers = payload.get("layers")
        if not isinstance(layers, list):
            raise TypeError("stack layers must be a list for authored port sheets.")
        rewritten_layers = list(layers)
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("port-sheet source records must be mappings.")
            target_layer = record.get("target_layer")
            if not isinstance(target_layer, str) or not target_layer:
                raise ValueError("port-sheet source records require target_layer.")
            matches = [
                index
                for index, layer in enumerate(rewritten_layers)
                if isinstance(layer, Mapping)
                and (
                    layer.get("semantic_id") == target_layer
                    or (
                        isinstance(layer.get("metadata"), Mapping)
                        and layer["metadata"].get("logical_layer_id") == target_layer
                    )
                )
            ]
            if not matches:
                raise ValueError(
                    f"authored port target_layer {target_layer!r} must match a logical stack layer."
                )
            for index in matches:
                layer = rewritten_layers[index]
                if not isinstance(layer, Mapping):
                    raise TypeError("matched logical stack layer must be a mapping.")
                if (
                    layer.get("role") != "metal"
                    or layer.get("part_role") != "face_metal"
                ):
                    raise ValueError(
                        f"authored port target_layer {target_layer!r} must have face-metal semantics."
                    )
                geometry = layer.get("geometry")
                if not isinstance(geometry, Mapping):
                    raise TypeError(
                        f"authored port target_layer {target_layer!r} must define geometry mapping."
                    )
                rewritten = dict(layer)
                rewritten_geometry = dict(geometry)
                rewritten_geometry["split_polygons_as_entities"] = True
                rewritten["geometry"] = rewritten_geometry
                rewritten_layers[index] = rewritten
        payload["layers"] = rewritten_layers
        metadata["port_sheet_source_layers"] = copy.deepcopy(list(records))
    payload["metadata"] = metadata
    return payload


def _component_gds_top_cell_name(*, component: Any, gds_path: Path) -> str:
    """Select a deterministic top cell name for SGB input."""
    try:
        import gdstk
    except ImportError as exc:  # pragma: no cover - optional
        raise ImportError("gdstk is required for SGB input validation") from exc

    library = gdstk.read_gds(str(gds_path))
    cells_by_name = {cell.name: cell for cell in library.cells}

    component_name = str(getattr(component, "name", "component"))
    candidates = [
        component_name,
        component_name.split("$", 1)[0],
        "component",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in cells_by_name:
            return candidate

    top_cells = sorted(
        (cell for cell in library.top_level() if not str(cell.name).startswith("$$$")),
        key=lambda item: str(item.name),
    )
    if len(top_cells) != 1:
        names = ", ".join(sorted(cell.name for cell in top_cells))
        raise ValueError(
            "SGB GDS export needs deterministic top-level cell selection. "
            f"Candidates: {names}"
        )
    return top_cells[0].name


def _optional_string(value: Any) -> str | None:
    """Return trimmed text when present, else None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_string(value: Any, field: str) -> str:
    text = _optional_string(value)
    if text is None:
        raise ValueError(f"{field} must be a non-empty string.")
    return text


def _required_distinct_strings(
    value: Any, field: str, *, count: int
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"{field} must contain exactly {count} strings.")
    result = tuple(_required_string(item, field) for item in value)
    if len(set(result)) != count:
        raise ValueError(f"{field} values must be distinct.")
    return result


def _normalized_direction(value: Any, name: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, (list, tuple))
        or len(value) != 3
    ):
        raise ValueError(f"SGB lumped-port sheet {name!r} requires a 3D direction.")
    direction = tuple(float(item) for item in value)
    length = math.hypot(direction[0], direction[1])
    if (
        not all(math.isfinite(item) for item in direction)
        or not math.isclose(direction[2], 0.0, abs_tol=1e-12)
        or not math.isclose(length, 1.0, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError(
            f"SGB lumped-port sheet {name!r} direction must be normalized finite XY."
        )
    return (direction[0], direction[1], 0.0)


def _is_nonempty_string_sequence(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _validate_structured_domain_boundary_record(
    name: str, record: Mapping[str, Any]
) -> None:
    required = (
        "surface_id",
        "representation",
        "interface_type",
        "contact_kind",
        "face_kind",
        "owner_semantic_ids",
        "adjacent_solution_volume_ids",
        "conductor_component_id",
        "net_id",
        "equipotential_id",
        "source_provenance",
        "physical_attribute",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(
            f"SGB domain boundary {name!r} misses fields: {', '.join(missing)}."
        )
    if not _optional_string(record.get("route")):
        raise ValueError(f"SGB domain boundary {name!r} has empty route.")
    if _optional_string(record.get("representation")) is None:
        raise ValueError(f"SGB domain boundary {name!r} has empty representation.")
    if not _is_nonempty_string_sequence(record.get("owner_semantic_ids")):
        raise ValueError(f"SGB domain boundary {name!r} needs owner_semantic_ids.")
    if not _is_nonempty_string_sequence(record.get("adjacent_solution_volume_ids")):
        raise ValueError(
            f"SGB domain boundary {name!r} needs adjacent_solution_volume_ids."
        )
    for field in ("source_provenance", "physical_attribute"):
        if not isinstance(record.get(field), Mapping):
            raise TypeError(f"SGB domain boundary {name!r} needs mapping {field}.")


def _validate_structured_surface_record(name: str, record: Mapping[str, Any]) -> None:
    required_strings = ("surface_id", "interface_type", "face_kind", "representation")
    missing = [
        field
        for field in required_strings
        if _optional_string(record.get(field)) is None
    ]
    missing.extend(
        field
        for field in (
            "contact_kind",
            "conductor_component_id",
            "net_id",
            "equipotential_id",
        )
        if field not in record
    )
    if missing:
        raise ValueError(
            f"SGB surface {name!r} misses structured fields: {', '.join(missing)}."
        )
    representation = _optional_string(record.get("representation"))
    route = _optional_string(record.get("route"))
    if representation is None:
        raise ValueError(f"SGB surface {name!r} has empty representation.")
    if route not in {"A", "B"}:
        raise ValueError(f"SGB surface {name!r} has invalid route {route!r}.")
    interface_type = str(record.get("interface_type")).upper()
    if interface_type not in _STRUCTURED_INTERFACE_TYPES:
        raise ValueError(
            f"SGB surface {name!r} has unsupported interface_type {interface_type!r}."
        )
    if not _is_nonempty_string_sequence(record.get("owner_semantic_ids")):
        raise ValueError(f"SGB surface {name!r} needs exact owner_semantic_ids.")
    if not _is_nonempty_string_sequence(record.get("adjacent_solution_volume_ids")):
        raise ValueError(
            f"SGB surface {name!r} needs exact adjacent_solution_volume_ids."
        )
    for field in ("source_provenance", "physical_attribute"):
        if not isinstance(record.get(field), Mapping):
            raise TypeError(f"SGB surface {name!r} needs mapping {field}.")
    face_kind = str(record.get("face_kind")).upper()
    if face_kind not in _STRUCTURED_FACE_KINDS:
        raise ValueError(
            f"SGB surface {name!r} has invalid face_kind {record.get('face_kind')!r}."
        )
    if interface_type == "SA":
        component_id = _optional_string(record.get("conductor_component_id"))
        if component_id is not None:
            raise ValueError(
                f"SGB SA surface {name!r} must not carry conductor_component_id."
            )


def _validate_structured_volume_record(name: str, record: Mapping[str, Any]) -> None:
    for field in ("representation", "source_provenance", "physical_attribute"):
        if field not in record:
            raise ValueError(f"SGB volume {name!r} misses {field}.")
    if not _optional_string(record.get("representation")):
        raise ValueError(f"SGB volume {name!r} has empty representation.")
    if not isinstance(record.get("source_provenance"), Mapping):
        raise TypeError(f"SGB volume {name!r} needs mapping source_provenance.")
    if not isinstance(record.get("physical_attribute"), Mapping):
        raise TypeError(f"SGB volume {name!r} needs mapping physical_attribute.")


def _structured_conductor_source_layer(name: str, record: Mapping[str, Any]) -> str:
    provenance = record.get("source_provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError(f"SGB surface {name!r} needs mapping source_provenance.")
    direct = _optional_string(provenance.get("conductor_source_layer_name"))
    if direct is not None:
        return direct
    sources = provenance.get("sources")
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError(
            f"SGB surface {name!r} needs exact conductor source-layer provenance."
        )

    source_layers: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise TypeError(f"SGB surface {name!r} has invalid provenance source.")
        source_layer = _optional_string(source.get("conductor_source_layer_name"))
        if source_layer is None:
            raise ValueError(
                f"SGB surface {name!r} lacks conductor_source_layer_name in source provenance."
            )
        source_layers.add(source_layer)

    if len(source_layers) != 1:
        raise ValueError(
            f"SGB surface {name!r} has ambiguous conductor source-layer provenance."
        )
    return source_layers.pop()


def _physical_group_values(value: Any) -> tuple[int, ...]:
    raw = value if isinstance(value, (list, tuple, set)) else (value,)
    return tuple(
        int(item)
        for item in raw
        if isinstance(item, int) and not isinstance(item, bool)
    )
