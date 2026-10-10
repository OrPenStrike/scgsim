"""Own detached source composition, occurrence transforms and caller final Net membership. Component local facts remain source authority."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scgsim.geometry.models.input import (
    GeometryBuildInput,
    SemanticEntitySpec,
    VacuumRegionSpec,
)
from scgsim.geometry.models.lumped import LumpedSupport
from scgsim.geometry.source._normalization import (
    _coupon_domain_bounds,
    _finite_number,
    _gds_layer,
    _materials,
    _semantic_layer_records,
    _solution_regions,
)
from scgsim.geometry.source._occurrence import (
    _entity_from_layer_record,
    _occurrence_ground_polygons_for_entity,
    _occurrence_include_polygon,
    _occurrence_polygons_for_entity,
    _occurrence_port_polygon,
    _occurrence_lumped_polygon,
)
from scgsim.geometry.source.adapter import build_gds_stack_geometry_input
from scgsim.geometry.source.curves import boundary_binding, transform_boundary
from scgsim.geometry.source.vacuum import apply_vacuum_region_to_stack

_PATH_SEGMENT = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*\Z")


_FORBIDDEN_AUTHOR_KEYS = {
    "net_id",
    "signal_group",
    "semantic_group_id",
    "physical_group_id",
    "physical_group_name",
}


def _path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(_PATH_SEGMENT.fullmatch(part) is None for part in value.split("/"))
    ):
        raise ValueError(
            "instance and entity paths need non-empty slash-separated names"
        )
    return value


def _affine(reference: Any) -> tuple[float, float, float, float, float, float]:
    from klayout import db as kdb

    transform = reference.dcplx_trans
    origin = transform * kdb.DPoint(0.0, 0.0)
    x_axis = transform * kdb.DPoint(1.0, 0.0)
    y_axis = transform * kdb.DPoint(0.0, 1.0)
    values = (
        float(x_axis.x - origin.x),
        float(y_axis.x - origin.x),
        float(x_axis.y - origin.y),
        float(y_axis.y - origin.y),
        float(origin.x),
        float(origin.y),
    )
    if not all(math.isfinite(item) for item in values):
        raise ValueError("instance transform must be finite")
    a, b, c, d, _, _ = values
    if not math.isclose(abs(a * d - b * c), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("GeometryPlan supports rigid rotation and mirror only")
    return values


def _compose(
    parent: tuple[float, ...], child: tuple[float, ...]
) -> tuple[float, float, float, float, float, float]:
    a, b, c, d, x, y = parent
    e, f, g, h, u, v = child
    return (
        a * e + b * g,
        a * f + b * h,
        c * e + d * g,
        c * f + d * h,
        a * u + b * v + x,
        c * u + d * v + y,
    )


def _point(transform: tuple[float, ...], value: Sequence[float]) -> list[float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("selector and port point must be a two-item coordinate")
    x, y = float(value[0]), float(value[1])
    a, b, c, d, u, v = transform
    result = [a * x + b * y + u, c * x + d * y + v]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("transformed point is not finite")
    return result


def _direction(transform: tuple[float, ...], angle_degrees: float) -> list[float]:
    angle = math.radians(float(angle_degrees))
    x, y = math.cos(angle), math.sin(angle)
    a, b, c, d, _, _ = transform
    result = [a * x + b * y, c * x + d * y]
    magnitude = math.hypot(*result)
    if not math.isfinite(magnitude) or magnitude == 0:
        raise ValueError("transformed port direction is invalid")
    return [result[0] / magnitude, result[1] / magnitude, 0.0]


def _local_semantics(component: Any, path: str) -> Mapping[str, Any]:
    try:
        raw = component.info["component_semantics"]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"{path} has no component_semantics v2 declaration") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 2:
        raise ValueError(f"{path} requires component_semantics schema_version=2")
    if set(raw) - {
        "schema_version",
        "conductor_regions",
        "ports",
        "ground_plane_contributions",
        "boundary_curves",
        "boundary_reconstruction",
        "metadata",
    }:
        raise ValueError(f"{path} has unknown component_semantics v2 fields")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{path} component metadata must be a mapping")
    if "signal_group" in metadata:
        raise ValueError(f"{path} cannot author a public Group ID")
    records = raw.get("conductor_regions", ())
    ports = raw.get("ports", ())
    contributions = raw.get("ground_plane_contributions", ())
    if any(
        isinstance(value, (str, bytes)) or not isinstance(value, Sequence)
        for value in (records, ports, contributions)
    ):
        raise TypeError(
            f"{path} conductor_regions, ports, and ground_plane_contributions must be sequences"
        )
    return raw


@dataclass(frozen=True)
class GeometryPlanSnapshot:
    """One detached GDS, stack, normalized input, and source-identity pair."""

    _geometry_input: GeometryBuildInput
    _stack_json: str
    gds_bytes: bytes
    gds_sha256: str
    _source_occurrences: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self._geometry_input, GeometryBuildInput):
            raise TypeError("snapshot requires GeometryBuildInput")
        if hashlib.sha256(self.gds_bytes).hexdigest() != self.gds_sha256:
            raise ValueError("snapshot GDS digest differs")
        object.__setattr__(self, "_geometry_input", copy.deepcopy(self._geometry_input))
        object.__setattr__(
            self, "_source_occurrences", copy.deepcopy(self._source_occurrences)
        )
        stack = json.loads(self._stack_json)
        metadata = stack.get("metadata", {})
        if metadata.get("gds_sha256") != self.gds_sha256:
            raise ValueError("snapshot stack GDS identity differs")
        if metadata.get("source_occurrences") != list(self._source_occurrences):
            raise ValueError("snapshot occurrence identity differs")
        if self._geometry_input.metadata.get("gds_sha256") != self.gds_sha256:
            raise ValueError("snapshot normalized-input GDS identity differs")

    @property
    def geometry_input(self) -> GeometryBuildInput:
        return copy.deepcopy(self._geometry_input)

    @property
    def stack(self) -> dict[str, Any]:
        return json.loads(self._stack_json)

    @property
    def source_occurrences(self) -> tuple[Mapping[str, Any], ...]:
        return copy.deepcopy(self._source_occurrences)

    @property
    def lumped_supports(self) -> tuple[LumpedSupport, ...]:
        return copy.deepcopy(self._geometry_input.lumped_supports)


class GeometryPlan:
    """Bind named source occurrences and final Nets before backend preparation."""

    def __init__(
        self,
        component: Any,
        *,
        root_id: str,
        layer_stack: Any,
        material_records: Mapping[str, Mapping[str, Any]],
        coupon_padding_um: float,
    ) -> None:
        if not callable(getattr(component, "write_gds", None)):
            raise TypeError("component must be an assembled GDSFactory Component")
        self.root_id = _path(root_id)
        if "/" in self.root_id:
            raise ValueError("root_id must be one path segment")
        _local_semantics(component, self.root_id)
        self._component = component
        self._layer_stack = layer_stack
        self._material_records = material_records
        self._coupon_padding_um = _finite_number(coupon_padding_um, "coupon_padding_um")
        if self._coupon_padding_um < 0:
            raise ValueError("coupon_padding_um must be nonnegative")
        self._instances: dict[str, tuple[Any, Any, tuple[float, ...]]] = {
            self.root_id: (component, None, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        }
        self._nets: dict[str, tuple[str, ...]] = {}
        self._vacuum_region: VacuumRegionSpec | None = None
        self._boundary_curves = ()
        self._boundary_reconstruction = ()
        self._lumped_supports = ()

    def set_lumped_supports(self, supports: Sequence[LumpedSupport]) -> None:
        """Bind caller-declared local nonmetal supports; no excitation is inferred."""
        records = tuple(supports)
        if any(not isinstance(record, LumpedSupport) for record in records):
            raise TypeError("lumped supports must be LumpedSupport records")
        if len({record.support_id for record in records}) != len(records):
            raise ValueError("lumped support IDs must be unique")
        self._lumped_supports = records

    def set_boundary_curves(self, boundaries) -> None:
        """Bind explicitly authored curves in assembled source coordinates."""
        prepared = tuple(boundary_binding(value) for value in boundaries)
        self._boundary_curves = prepared

    def set_boundary_reconstruction(self, boundaries) -> None:
        """Select GDS boundary reconstruction explicitly; omitted inputs stay polygonal."""
        prepared = tuple(
            boundary_binding(value, reconstruction=True) for value in boundaries
        )
        self._boundary_reconstruction = prepared

    def set_vacuum_region(
        self, padding: float | Sequence[float] | Mapping[str, Any] = 0.0
    ) -> None:
        """Set one explicit six-face outer envelope for the next snapshot."""
        prepared = VacuumRegionSpec.from_padding(padding)
        self._vacuum_region = prepared

    def register_instance(self, path: str, reference: Any) -> None:
        path = _path(path)
        parent_path, _, _ = path.rpartition("/")
        if (
            not parent_path
            or parent_path not in self._instances
            or path in self._instances
        ):
            raise ValueError(
                "instance path needs a registered parent and must be unique"
            )
        parent_component, _, parent_transform = self._instances[parent_path]
        if getattr(reference, "parent_cell", None) != parent_component:
            raise ValueError("reference is not an occurrence in its declared parent")
        matches = [
            item
            for item in parent_component.insts
            if item.instance == reference.instance
        ]
        if len(matches) != 1:
            raise ValueError("reference must resolve to exactly one parent occurrence")
        if any(
            other_parent == parent_path
            and other_ref is not None
            and other_ref.instance == reference.instance
            for key, (_, other_ref, _) in self._instances.items()
            if (other_parent := key.rpartition("/")[0])
        ):
            raise ValueError("the same source occurrence is already registered")
        child = reference.cell
        _local_semantics(child, path)
        transform = _compose(parent_transform, _affine(reference))
        self._instances[path] = (child, reference, transform)

    @property
    def entities(self) -> tuple[SemanticEntitySpec, ...]:
        stack = self._stack()
        return tuple(
            _entity_from_layer_record(record, materials=stack["materials"])
            for record in stack["layers"]
        )

    def set_nets(self, nets: Mapping[str, Sequence[str]]) -> None:
        if not isinstance(nets, Mapping):
            raise TypeError("nets must map final Net IDs to qualified Entity IDs")
        available = {entity.semantic_id for entity in self.entities}
        owners: dict[str, str] = {}
        prepared: dict[str, tuple[str, ...]] = {}
        for net, members in nets.items():
            if not isinstance(net, str) or not net:
                raise ValueError("Net ID must be nonempty text")
            if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
                raise TypeError(f"Net {net!r} members must be Entity IDs")
            unique = tuple(dict.fromkeys(members))
            if not unique:
                raise ValueError(f"Net {net!r} needs at least one Entity ID")
            for entity_id in unique:
                if not isinstance(entity_id, str) or entity_id not in available:
                    raise ValueError(f"Net {net!r} has unknown Entity ID {entity_id!r}")
                prior = owners.setdefault(entity_id, net)
                if prior != net:
                    raise ValueError(f"Entity {entity_id!r} belongs to two Nets")
            prepared[net] = unique
        self._nets = prepared

    def _validate_registered_sources(self) -> None:
        """Reject authored descendants that would disappear during GDS flattening."""

        for path, (cell, reference, transform) in self._instances.items():
            if reference is None:
                continue
            parent_path = path.rpartition("/")[0]
            parent, _, parent_transform = self._instances[parent_path]
            if (
                reference.parent_cell != parent
                or reference.cell != cell
                or sum(item.instance == reference.instance for item in parent.insts)
                != 1
                or _compose(parent_transform, _affine(reference)) != transform
            ):
                raise ValueError(
                    f"registered occurrence {path!r} changed before prepare"
                )

        def visit(cell: Any, path: str) -> None:
            registered = (
                (reference.instance, child_path)
                for child_path, (_, reference, _) in self._instances.items()
                if reference is not None and child_path.rpartition("/")[0] == path
            )
            expected = tuple(registered)
            for reference in cell.insts:
                child = reference.cell
                child_path = next(
                    (
                        name
                        for instance, name in expected
                        if instance == reference.instance
                    ),
                    None,
                )
                if child_path is None and "component_semantics" in child.info:
                    raise ValueError(
                        f"authored source occurrence in {path!r} is not registered: "
                        f"{child.name!r}"
                    )
                visit(child, child_path or f"{path}/<unregistered>")

        visit(self._component, self.root_id)

    def _stack(self) -> dict[str, Any]:
        levels = getattr(self._layer_stack, "layers", None)
        if not isinstance(levels, Mapping):
            raise TypeError("layer_stack must expose layers")
        bounds = _coupon_domain_bounds(self._component, self._coupon_padding_um)
        materials = _materials(self._material_records)
        regions = _solution_regions(levels, materials=materials, bounds=bounds)
        if not regions:
            raise ValueError("plan needs at least one PDK solution region")
        owners = {
            entity_id: net
            for net, members in self._nets.items()
            for entity_id in members
        }
        declarations: list[dict[str, Any]] = []
        ports: list[dict[str, Any]] = []
        contributions: list[tuple[str, dict[str, Any]]] = []
        contribution_ids: set[tuple[str, str]] = set()
        for path, (cell, _, transform) in self._instances.items():
            semantics = _local_semantics(cell, path)
            for raw in semantics.get("conductor_regions", ()):
                if not isinstance(raw, Mapping):
                    raise TypeError(f"{path} conductor records must be mappings")
                if set(raw) & _FORBIDDEN_AUTHOR_KEYS:
                    raise ValueError(f"{path} v2 entity cannot author Net or Group")
                metadata = raw.get("metadata", {})
                if (
                    not isinstance(metadata, Mapping)
                    or set(metadata) & _FORBIDDEN_AUTHOR_KEYS
                ):
                    raise ValueError(
                        f"{path} v2 entity metadata cannot author Net or Group"
                    )
                local_id = _path(raw.get("semantic_id"))
                if "/" in local_id:
                    raise ValueError("component-local entity ID must be one segment")
                entity_id = f"{path}/{local_id}"
                geometry = dict(raw.get("geometry", {}))
                for key in ("selector_point_um",):
                    if key in geometry:
                        geometry[key] = _point(transform, geometry[key])
                for key in ("include_selector_points_um", "exclude_selector_points_um"):
                    if key in geometry:
                        geometry[key] = [
                            _point(transform, point) for point in geometry[key]
                        ]
                declarations.append(
                    {
                        **raw,
                        "semantic_id": entity_id,
                        "net_id": owners.get(entity_id),
                        "geometry": geometry,
                        "metadata": {
                            **metadata,
                            "source_occurrence_path": path,
                            "source_local_entity_id": local_id,
                            "source_semantic_id": entity_id,
                        },
                    }
                )
            for raw in semantics.get("ground_plane_contributions", ()):
                if not isinstance(raw, Mapping) or set(raw) != {
                    "local_id",
                    "plane_id",
                    "level",
                    "layer",
                    "selector_point_um",
                }:
                    raise ValueError(
                        f"{path} ground contribution needs local_id, plane_id, level, layer, selector_point_um"
                    )
                local_id, plane_id, level = (
                    _path(raw[key]) for key in ("local_id", "plane_id", "level")
                )
                if any("/" in value for value in (local_id, plane_id, level)):
                    raise ValueError(
                        "ground contribution IDs must be component-local names"
                    )
                key = (path, local_id)
                if key in contribution_ids:
                    raise ValueError(f"duplicate ground contribution {path}/{local_id}")
                contribution_ids.add(key)
                layer = _gds_layer(
                    raw["layer"], f"{path}/{local_id} ground contribution"
                )
                contributions.append(
                    (
                        plane_id,
                        {
                            "source_occurrence_path": path,
                            "source_local_id": local_id,
                            "level": level,
                            "layer": list(layer),
                            "selector_point_um": _point(
                                transform, raw["selector_point_um"]
                            ),
                        },
                    )
                )
            for raw in semantics.get("ports", ()):
                if not isinstance(raw, Mapping) or set(raw) != {
                    "name",
                    "layer",
                    "target_layer",
                }:
                    raise ValueError(f"{path} port requires name, layer, target_layer")
                name = _path(raw["name"])
                if "/" in name:
                    raise ValueError("component-local port name must be one segment")
                port = cell.ports[name]
                layer = _gds_layer(raw["layer"], f"{path}/{name} port")
                raw_port_layer = port.layer
                if hasattr(raw_port_layer, "layer") and hasattr(
                    raw_port_layer, "datatype"
                ):
                    port_layer = raw_port_layer
                else:
                    port_layer = port.kcl.layout.get_info(int(str(raw_port_layer)))
                if (int(port_layer.layer), int(port_layer.datatype)) != layer:
                    raise ValueError(f"{path}/{name} authored port layer differs")
                ports.append(
                    {
                        "layer": layer[0],
                        "datatype": layer[1],
                        "name": f"{path}/{name}",
                        "source": "palace_lumped_port_sheet",
                        "port_index": len(ports) + 1,
                        "target_layer": raw["target_layer"],
                        "direction": _direction(transform, float(port.orientation)),
                        "direction_sign_convention": "gdsfactory_port_orientation_outward",
                        "selector_point_um": _point(transform, port.center),
                        "source_occurrence_path": path,
                    }
                )
        layers = _semantic_layer_records(
            declarations,
            levels=levels,
            materials=materials,
            solution_region_ids=regions,
            require_net=False,
        )
        by_id = {record["semantic_id"]: record for record in layers}
        if len(by_id) != len(layers):
            raise ValueError("GeometryPlan Entity IDs must be unique")
        physical_planes: set[str] = set()
        for record in layers:
            if (
                record["geometry"].get("geometry_source")
                != "die_face_minus_ground_mask"
            ):
                continue
            if record["metadata"]["source_occurrence_path"] != self.root_id:
                from gdsfactory.pdk import get_layer_tuple
                from gdsfactory.technology.layer_stack import LogicalLayer

                bounds_ref = record["geometry"].get("plane_bounds_ref")
                if bounds_ref not in regions or bounds_ref not in levels:
                    raise ValueError("occurrence Ground requires its declared plane_bounds_ref")
                source_layer = levels[bounds_ref].layer
                if not isinstance(source_layer, LogicalLayer):
                    raise ValueError("occurrence Ground bounds require a source logical layer")
                regions[bounds_ref]["metadata"]["source_layer_datatype"] = list(
                    get_layer_tuple(source_layer.layer)
                )
            level = record["metadata"]["logical_layer_id"]
            if level in physical_planes:
                raise ValueError(
                    f"multiple ground planes claim physical level {level!r}"
                )
            physical_planes.add(level)
        for plane_id, contribution in contributions:
            target = by_id.get(f"{self.root_id}/{plane_id}")
            if (
                target is None
                or target["geometry"].get("geometry_source")
                != "die_face_minus_ground_mask"
            ):
                raise ValueError(
                    f"ground contribution targets no root derived plane {plane_id!r}"
                )
            if target["metadata"]["logical_layer_id"] != contribution["level"]:
                raise ValueError(
                    f"ground contribution {contribution['source_local_id']!r} level differs from target plane"
                )
            geometry = target["geometry"]
            if "include_layer" in geometry or "include_selector_points_um" in geometry:
                raise ValueError(
                    "root ground plane cannot mix global includes with source contributions"
                )
            geometry.setdefault("source_occurrence_includes", []).append(contribution)
            target["metadata"].setdefault(
                "ground_plane_contribution_sources", []
            ).append(
                {
                    "source_occurrence_path": contribution["source_occurrence_path"],
                    "source_local_id": contribution["source_local_id"],
                    "level": contribution["level"],
                    "layer": contribution["layer"],
                }
            )
        return {
            "solution_regions": regions,
            "materials": materials,
            "layers": layers,
            "metadata": {
                "adapter": "scgsim.sgb.GeometryPlan",
                "component_semantics_schema_version": 2,
                "coupon_domain_bounds_um": dict(bounds),
                "coupon_padding_um": self._coupon_padding_um,
                "top_cell_name": self._component.name,
                "port_sheet_source_layers": ports,
            },
        }

    def prepare(self) -> GeometryPlanSnapshot:
        self._validate_registered_sources()
        stack = self._stack()
        available = {record["semantic_id"] for record in stack["layers"]}
        for net, members in self._nets.items():
            if not members or any(member not in available for member in members):
                raise ValueError(
                    f"Net {net!r} changed after set_nets; call set_nets again"
                )
        if self._vacuum_region is not None:
            stack = dict(apply_vacuum_region_to_stack(stack, self._vacuum_region))
        occurrences = tuple(
            {
                "path": path,
                "parent_path": path.rpartition("/")[0] or None,
                "cell_name": cell.name,
                "transform": list(transform),
                "reference_index": (
                    None
                    if reference is None
                    else next(
                        index
                        for index, item in enumerate(
                            self._instances[path.rpartition("/")[0]][0].insts
                        )
                        if item.instance == reference.instance
                    )
                ),
            }
            for path, (cell, reference, transform) in sorted(self._instances.items())
        )
        with TemporaryDirectory(prefix="scgsim-geometry-plan-") as temporary:
            root = Path(temporary)
            gds_file, stack_file = root / "source.gds", root / "source.stack.json"
            self._component.write_gds(gds_file)
            gds_bytes = gds_file.read_bytes()
            gds_sha256 = hashlib.sha256(gds_bytes).hexdigest()
            stack["metadata"]["gds_sha256"] = gds_sha256
            stack["metadata"]["source_occurrences"] = occurrences
            import gdstk

            source_cells = {
                cell.name: cell for cell in gdstk.read_gds(str(gds_file)).cells
            }
            child_reference_indexes = {
                path: tuple(
                    row["reference_index"]
                    for row in occurrences
                    if row["parent_path"] == path
                )
                for path in self._instances
            }
            for record in stack["layers"]:
                geometry_source = record["geometry"].get("geometry_source", "gds_polygon")
                path = record["metadata"]["source_occurrence_path"]
                local_ground = geometry_source == "die_face_minus_ground_mask" and path != self.root_id
                if geometry_source != "gds_polygon" and not local_ground:
                    continue
                cell, _, transform = self._instances[path]
                if cell.name not in source_cells:
                    raise ValueError(
                        f"source occurrence {path!r} is absent from written GDS"
                    )
                selector = (_occurrence_ground_polygons_for_entity if local_ground
                            else _occurrence_polygons_for_entity)
                record["geometry"]["source_occurrence_polygons_um"] = (
                    selector(
                        _entity_from_layer_record(record, materials=stack["materials"]),
                        cell=source_cells[cell.name],
                        transform=transform,
                        excluded_reference_indexes=child_reference_indexes[path],
                        **({"plane_layer": stack["solution_regions"][record["geometry"]["plane_bounds_ref"]]
                            ["metadata"]["source_layer_datatype"]} if local_ground else {}),
                    )
                )
            for record in stack["layers"]:
                for contribution in record["geometry"].get(
                    "source_occurrence_includes", ()
                ):
                    path = contribution["source_occurrence_path"]
                    cell, _, transform = self._instances[path]
                    if cell.name not in source_cells:
                        raise ValueError(
                            f"ground contribution source {path!r} is absent from GDS"
                        )
                    contribution["polygon"] = _occurrence_include_polygon(
                        contribution,
                        cell=source_cells[cell.name],
                        transform=transform,
                        excluded_reference_indexes=child_reference_indexes[path],
                    )
            for record in stack["metadata"]["port_sheet_source_layers"]:
                path = record["source_occurrence_path"]
                cell, _, transform = self._instances[path]
                if cell.name not in source_cells:
                    raise ValueError(
                        f"source port occurrence {path!r} is absent from written GDS"
                    )
                record["source_occurrence_polygon_um"] = _occurrence_port_polygon(
                    record,
                    cell=source_cells[cell.name],
                    transform=transform,
                    excluded_reference_indexes=child_reference_indexes[path],
                )
            stack_file.write_text(json.dumps(stack, sort_keys=True), encoding="utf-8")
            build_input = build_gds_stack_geometry_input(
                gds_file=gds_file,
                stack_file=stack_file,
                top_cell_name=self._component.name,
            )
            lumped_supports = []
            entities = {entity.semantic_id: entity for entity in build_input.entities}
            for support in self._lumped_supports:
                path = support.source_occurrence_path
                if path not in self._instances:
                    raise ValueError(f"lumped support occurrence {path!r} is not registered")
                cell, _, transform = self._instances[path]
                if support.source_level not in self._layer_stack.layers:
                    raise ValueError(f"lumped support level {support.source_level!r} is not declared")
                level = self._layer_stack.layers[support.source_level]
                from gdsfactory.pdk import get_layer_tuple
                if get_layer_tuple(level.layer.layer) != support.source_layer:
                    raise ValueError("lumped support source level/layer binding differs")
                source_z_um = float(level.zmin)
                if support.source_z_um is not None and support.source_z_um != source_z_um:
                    raise ValueError("lumped support source plane differs from locator Level")
                if any(point[2] != source_z_um for point in support.contact_points_um):
                    raise ValueError("lumped support contacts differ from locator plane")
                for entity_id in (*support.terminal_a_entity_ids, *support.terminal_b_entity_ids):
                    if entity_id not in entities or entities[entity_id].material_kind != "conductor":
                        raise ValueError(f"lumped support terminal Entity {entity_id!r} is not a conductor")
                    if entities[entity_id].net_id is None:
                        raise ValueError(f"lumped support terminal Entity {entity_id!r} lacks final Net ownership")
                polygon = _occurrence_lumped_polygon(
                    support, cell=source_cells[cell.name],
                    excluded_reference_indexes=child_reference_indexes[path],
                )
                lumped_supports.append(replace(
                    support, exterior=polygon.exterior, holes=polygon.holes,
                    source_transform=transform, source_z_um=source_z_um,
                ))
        # The source is self-contained after temporary input files disappear.
        metadata = dict(build_input.metadata)
        metadata.pop("gds_file", None)
        metadata.pop("stack_file", None)
        metadata["gds_sha256"] = gds_sha256
        metadata["geometry_plan_root_id"] = self.root_id
        polygon_owners = {
            polygon_id: entity
            for entity in build_input.entities
            for polygon_id in entity.polygon_ids
        }
        polygons = tuple(
            replace(
                polygon,
                metadata={
                    **polygon.metadata,
                    **(
                        {
                            "source_entity_id": polygon_owners[
                                polygon.polygon_id
                            ].semantic_id,
                            "source_occurrence_path": polygon_owners[
                                polygon.polygon_id
                            ].metadata["source_occurrence_path"],
                        }
                        if polygon.polygon_id in polygon_owners
                        else {}
                    ),
                },
            )
            for polygon in build_input.polygons
        )
        boundary_curves = list(self._boundary_curves)
        reconstruction = list(self._boundary_reconstruction)
        for path, (cell, _, transform) in self._instances.items():
            semantics = _local_semantics(cell, path)
            for key, target, reconstruct in (
                ("boundary_curves", boundary_curves, False),
                ("boundary_reconstruction", reconstruction, True),
            ):
                target.extend(
                    transform_boundary(
                        value,
                        path=path,
                        transform=transform,
                        point=_point,
                        reconstruction=reconstruct,
                    )
                    for value in semantics.get(key, ())
                )
        build_input = replace(
            build_input,
            polygons=polygons,
            metadata=metadata,
            boundary_curves=tuple(boundary_curves),
            boundary_reconstruction=tuple(reconstruction),
            lumped_supports=tuple(lumped_supports),
        )
        return GeometryPlanSnapshot(
            build_input,
            json.dumps(stack, sort_keys=True),
            gds_bytes,
            gds_sha256,
            occurrences,
        )
