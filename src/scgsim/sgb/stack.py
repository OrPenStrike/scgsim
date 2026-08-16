"""Declarative semantic-stack construction from explicit technology facts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any


def build_component_stack(
    *,
    component: Any,
    layer_stack: Any,
    material_records: Mapping[str, Mapping[str, Any]],
    coupon_padding_um: float,
) -> dict[str, Any]:
    """Compile PDK process facts and component topology into one SGB stack.

    The PDK layer stack owns material, fabrication, host-volume, and 3D
    integration semantics. The component owns conductor identities, nets, and
    topology selectors. Geometry bounds size the coupon only.
    """
    padding = _positive_number(coupon_padding_um, "coupon_padding_um")
    bounds = _coupon_domain_bounds(component, padding)
    try:
        spec = _mapping(
            component.info["component_semantics"], "component component_semantics"
        )
    except (AttributeError, KeyError) as exc:
        raise ValueError(
            "component must provide an authored info['component_semantics'] annotation."
        ) from exc
    if spec.get("schema_version") != 1:
        raise ValueError("component_semantics schema_version must be 1.")
    unknown = set(spec) - {"schema_version", "conductor_regions", "metadata"}
    if unknown:
        raise ValueError(
            f"component_semantics has unsupported fields {sorted(unknown)!r}."
        )
    conductor_regions = spec.get("conductor_regions")
    metadata = spec.get("metadata")
    levels = getattr(layer_stack, "layers", None)
    if not isinstance(levels, Mapping):
        raise TypeError("layer_stack must expose a mapping 'layers'.")
    if isinstance(conductor_regions, str | bytes) or not isinstance(
        conductor_regions, Sequence
    ):
        raise TypeError("component conductor_regions must be a sequence of mappings.")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping when provided.")

    materials = _materials(material_records)
    regions = _solution_regions(levels, materials=materials, bounds=bounds)
    if not regions:
        raise ValueError(
            "PDK layer_stack must declare at least one included solution region."
        )
    layers = _semantic_layer_records(
        conductor_regions,
        levels=levels,
        materials=materials,
        solution_region_ids=regions,
    )
    return {
        "solution_regions": regions,
        "materials": materials,
        "layers": layers,
        "metadata": {
            **dict(metadata or {}),
            "adapter": "scgsim.sgb.stack.build_component_stack",
            "coupon_domain_bounds_um": dict(bounds),
            "coupon_padding_um": padding,
        },
    }


def _solution_regions(
    levels: Mapping[str, Any],
    *,
    materials: Mapping[str, Mapping[str, Any]],
    bounds: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for level_name, level in levels.items():
        info = _level_info(level, level_name)
        if info.get("simulation_role") != "solution_region":
            continue
        include = info.get("include_in_component_simulation")
        if not isinstance(include, bool):
            raise TypeError(
                f"PDK solution level {level_name!r} must set boolean "
                "include_in_component_simulation."
            )
        if not include:
            continue
        semantic_id = _identifier(level_name, "PDK solution level id")
        material_id, z_min_um, thickness_um, _ = _level_facts(level, level_name)
        _require_material(material_id, materials, f"solution region {semantic_id!r}")
        result[semantic_id] = {
            "role": "solution_region",
            "material_id": material_id,
            "geometry_kind": "domain",
            "geometry": {
                "z_min_um": z_min_um,
                "z_max_um": z_min_um + thickness_um,
                "domain_bounds_um": dict(bounds),
            },
            "metadata": {
                "pdk_level_id": semantic_id,
                "pdk_semantic_authority": "layer_stack",
            },
        }
    return result


def _semantic_layer_records(
    declarations: Sequence[Mapping[str, Any]],
    *,
    levels: Mapping[str, Any],
    materials: Mapping[str, Mapping[str, Any]],
    solution_region_ids: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    semantic_ids: set[str] = set()
    for declaration in declarations:
        if not isinstance(declaration, Mapping):
            raise TypeError("component conductor_regions items must be mappings.")
        unknown = set(declaration) - {
            "semantic_id",
            "level",
            "gds_layer",
            "net_id",
            "geometry",
            "metadata",
        }
        if unknown:
            raise ValueError(
                "component conductor region contains PDK-owned or unsupported fields "
                f"{sorted(unknown)!r}."
            )
        required = ("semantic_id", "level", "gds_layer", "net_id")
        missing = [field for field in required if field not in declaration]
        if missing:
            raise ValueError(f"component conductor region lacks fields {missing!r}.")
        semantic_id = _identifier(declaration["semantic_id"], "semantic_id")
        if semantic_id in semantic_ids:
            raise ValueError(f"duplicate semantic layer {semantic_id!r}.")
        semantic_ids.add(semantic_id)
        level_name = _identifier(declaration["level"], f"{semantic_id!r} level")
        level = _level(levels, level_name)
        level_info = _level_info(level, level_name)
        if level_info.get("simulation_role") != "conductor":
            raise ValueError(
                f"PDK level {level_name!r} is not declared as a conductor."
            )
        material_id, z_um, thickness_um, priority = _level_facts(level, level_name)
        _require_material(material_id, materials, f"semantic layer {semantic_id!r}")
        host = _identifier(
            level_info.get("host_void_semantic_id"),
            f"{semantic_id!r} host_void_semantic_id",
        )
        if host not in solution_region_ids:
            raise ValueError(
                f"semantic layer {semantic_id!r} host {host!r} is not a solution region."
            )
        layer, datatype = _gds_layer(declaration["gds_layer"], semantic_id)
        pdk_geometry = _mapping(
            level_info.get("geometry", {}), f"PDK level {level_name!r} geometry"
        )
        component_geometry = _mapping(
            declaration.get("geometry", {}), f"{semantic_id!r} geometry"
        )
        unknown_geometry = set(component_geometry) - {
            "geometry_source",
            "selector_point_um",
            "mask_layer",
            "include_layer",
            "include_selector_points_um",
        }
        if unknown_geometry:
            raise ValueError(
                f"component conductor region {semantic_id!r} contains PDK-owned or "
                f"unsupported geometry fields {sorted(unknown_geometry)!r}."
            )
        if "geometry_source" in component_geometry and component_geometry[
            "geometry_source"
        ] != pdk_geometry.get("geometry_source"):
            pdk_geometry = {}
        component_metadata = _mapping(
            declaration.get("metadata", {}), f"{semantic_id!r} metadata"
        )
        part_role = _identifier(
            level_info.get("part_role"), f"PDK level {level_name!r} part_role"
        )
        record: dict[str, Any] = {
            "layer": layer,
            "datatype": datatype,
            "semantic_id": semantic_id,
            "role": "metal",
            "material_id": material_id,
            "priority": priority,
            "part_role": part_role,
            "host_void_semantic_id": host,
            "geometry_kind": "layout_extrusion",
            "geometry": {
                "z_um": z_um,
                "thickness_um": thickness_um,
                "route_ab_fused_selector_mode": True,
                **pdk_geometry,
                **component_geometry,
            },
            "route_representations": _route_representations(part_role),
            "metadata": {
                "logical_layer_id": level_name,
                "source_layer_name": level_name,
                "pdk_semantic_authority": "layer_stack",
                "component_semantic_authority": "component.info",
                **component_metadata,
            },
            "net_id": _identifier(declaration["net_id"], f"{semantic_id!r} net_id"),
        }
        if "ground_bump_fill_spec" in level_info:
            record["metadata"]["ground_bump_fill_spec"] = _mapping(
                level_info["ground_bump_fill_spec"],
                f"PDK level {level_name!r} ground_bump_fill_spec",
            )
        result.append(record)
    return result


def _coupon_domain_bounds(component: Any, padding_um: float) -> dict[str, float]:
    try:
        bbox = component.dbbox()
    except AttributeError as exc:
        raise TypeError("component must expose dbbox().") from exc
    if bbox is None:
        raise ValueError("component dbbox() must return physical-unit bounds.")
    try:
        x_min, y_min, x_max, y_max = (
            float(bbox.left),
            float(bbox.bottom),
            float(bbox.right),
            float(bbox.top),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "component dbbox() must expose left, bottom, right, and top."
        ) from exc
    if not all(isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        raise ValueError("component dbbox() bounds must be finite.")
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("component dbbox() must have positive X and Y extent.")
    return {
        "x_min_um": x_min - padding_um,
        "y_min_um": y_min - padding_um,
        "x_max_um": x_max + padding_um,
        "y_max_um": y_max + padding_um,
    }


def _level(levels: Mapping[str, Any], name: Any) -> Any:
    name = _identifier(name, "layer-stack level")
    try:
        return levels[name]
    except KeyError as exc:
        raise ValueError(f"layer_stack lacks explicit level {name!r}.") from exc


def _level_info(level: Any, name: Any) -> dict[str, Any]:
    return _mapping(getattr(level, "info", None), f"PDK level {name!r} info")


def _level_facts(level: Any, name: Any) -> tuple[str, float, float, int]:
    material_id = _identifier(
        getattr(level, "material", None), f"level {name!r} material"
    )
    z_um = _finite_number(getattr(level, "zmin", None), f"level {name!r} zmin")
    thickness_um = _positive_number(
        getattr(level, "thickness", None), f"level {name!r} thickness"
    )
    try:
        priority = int(getattr(level, "mesh_order", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"level {name!r} mesh_order must be an integer.") from exc
    return material_id, z_um, thickness_um, priority


def _require_material(
    material_id: str, materials: Mapping[str, Mapping[str, Any]], context: str
) -> None:
    if material_id not in materials:
        raise ValueError(
            f"{context} material {material_id!r} lacks a supported record."
        )


def _gds_layer(value: Any, semantic_id: str) -> tuple[int, int]:
    if (
        isinstance(value, str | bytes)
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise TypeError(
            f"semantic layer {semantic_id!r} gds_layer must be a two-item sequence."
        )
    layer, datatype = value
    if any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in (layer, datatype)
    ):
        raise TypeError(
            f"semantic layer {semantic_id!r} gds_layer must contain integers."
        )
    return layer, datatype


def _route_representations(part_role: str) -> dict[str, str]:
    if part_role == "face_metal":
        return {"A": "surface_sheet", "B": "cutout_boundary_shell"}
    if part_role == "bump_body":
        return {"A": "cutout_boundary_shell", "B": "cutout_boundary_shell"}
    raise ValueError(f"unsupported PDK conductor part_role {part_role!r}.")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number.")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _positive_number(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _materials(records: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Normalize material records with the established stack-material mapping."""
    if not isinstance(records, Mapping):
        raise TypeError("material_records must be a mapping.")
    kinds = {
        "superconductor": "conductor",
        "conductive": "conductor",
        "conductor": "conductor",
        "vacuum": "vacuum",
        "dielectric": "dielectric",
    }
    result: dict[str, dict[str, Any]] = {}
    for name, record in records.items():
        if not isinstance(record, Mapping):
            raise TypeError("material_records values must be mappings.")
        raw_kind = record.get("material_kind", record.get("kind"))
        if raw_kind not in kinds:
            continue
        normalized = kinds[raw_kind]
        item: dict[str, Any] = {"kind": normalized}
        if normalized != "conductor":
            permittivity = record.get(
                "relative_permittivity", record.get("permittivity")
            )
            if isinstance(permittivity, (int, float)):
                item["permittivity"] = float(permittivity)
        loss_tangent = record.get("loss_tangent")
        if isinstance(loss_tangent, (int, float)):
            item["loss_tangent"] = float(loss_tangent)
        result[str(name)] = item
    if not result:
        raise ValueError("material_records contain no supported material kinds.")
    return result
