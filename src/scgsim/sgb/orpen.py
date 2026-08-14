"""Public OrPen SC PDK semantic-stack adapter for the Xmon example."""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, isfinite, radians, sin
from typing import Any

_LEVELS = (
    "D0_SUBSTRATE",
    "D0_TO_D1_GAP",
    "D1_SUBSTRATE",
    "OUTER_VACUUM",
    "D0_TOP_M1",
    "D1_BOTTOM_M1",
    "D0_D1_INDIUM_BUMP",
)


def build_kosen2024_flip_chip_xmon_stack(
    *,
    component: Any,
    layer_stack: Any,
    material_records: Mapping[str, Mapping[str, Any]],
    d0_top_ground_mask_layer: tuple[int, int],
    indium_bump_layer: tuple[int, int],
    coupon_padding_um: float,
    air_below_thickness_um: float,
    air_above_thickness_um: float,
) -> dict[str, Any]:
    """Project public PDK facts into the one SCGSim Route-A/B stack contract.

    PDK records remain the authority for material, z, thickness and layers.
    Signal identities come only from the documented topology anchor and authored
    component ports; no layout-name, bbox, or residual-policy inference occurs.
    """
    levels = getattr(layer_stack, "layers", None)
    if not isinstance(levels, Mapping):
        raise TypeError("layer_stack must expose public LayerStack.layers.")
    missing = [name for name in _LEVELS if name not in levels]
    if missing:
        raise ValueError(f"public OrPen LayerStack lacks {missing!r}.")
    d0_substrate, gap, d1_substrate, outer = (levels[name] for name in _LEVELS[:4])
    d0_top, d1_bottom, bump = (levels[name] for name in _LEVELS[4:])
    coupon_bounds_um = _coupon_domain_bounds(component, coupon_padding_um)
    d0_z_min = _zmin(d0_substrate)
    d0_z_max = d0_z_min + _thickness(d0_substrate)
    gap_z_min = _zmin(gap)
    gap_z_max = gap_z_min + _thickness(gap)
    d1_z_min = _zmin(d1_substrate)
    d1_z_max = d1_z_min + _thickness(d1_substrate)
    if (abs(d0_z_max - gap_z_min) > 1e-9) or (abs(gap_z_max - d1_z_min) > 1e-9):
        raise ValueError("public OrPen D0/gap/D1 solution regions must be contiguous.")
    air_below_thickness = _positive_number(
        air_below_thickness_um, "air_below_thickness_um"
    )
    air_above_thickness = _positive_number(
        air_above_thickness_um, "air_above_thickness_um"
    )
    info = getattr(component, "info", {})
    try:
        layers = info["layers"]
    except (KeyError, TypeError):
        layers = {}
    d1_draw = _layer_tuple(layers.get("q_chip_draw"))
    d1_mask = _layer_tuple(layers.get("q_chip_ground_mask"))
    d0_mask = _layer_tuple(d0_top_ground_mask_layer)
    bump_layer = _layer_tuple(indium_bump_layer)
    if not all((d1_draw, d1_mask, d0_mask, bump_layer)):
        raise ValueError("public PDK lacks required D0/D1/bump GDS-layer facts.")

    materials = _materials(material_records)
    signal_group = "D1_BOTTOM_SIGNAL_GROUP"
    common = _metal_record(
        layer=d1_draw,
        level=d1_bottom,
        host="D0_TO_D1_GAP",
        part_role="face_metal",
        representations={"A": "surface_sheet", "B": "cutout_boundary_shell"},
        source_layer_name="D1_BOTTOM_M1_DRAW",
    )
    selectors = _signal_selectors(component)
    signal_layers = [
        {
            **common,
            "semantic_id": semantic_id,
            "net_id": net_id,
            "geometry": {**common["geometry"], "selector_point_um": point},
            "metadata": {
                **common["metadata"],
                "semantic_group_id": signal_group,
                "logical_layer_id": "D1_BOTTOM_M1",
                "source_selector": source,
            },
        }
        for semantic_id, net_id, point, source in selectors
    ]
    d0_ground = _metal_record(
        layer=d0_mask,
        level=d0_top,
        host="D0_TO_D1_GAP",
        part_role="face_metal",
        representations={"A": "surface_sheet", "B": "cutout_boundary_shell"},
        source_layer_name="D0_TOP_GROUND_MASK",
    )
    d1_ground = _metal_record(
        layer=d1_mask,
        level=d1_bottom,
        host="D0_TO_D1_GAP",
        part_role="face_metal",
        representations={"A": "surface_sheet", "B": "cutout_boundary_shell"},
        source_layer_name="D1_BOTTOM_GROUND_MASK",
    )
    d0_ground.update(
        {
            "semantic_id": "D0_TOP_GROUND_PLANE",
            "net_id": "Ground",
            "geometry": {
                **d0_ground["geometry"],
                "geometry_source": "die_face_minus_ground_mask",
                "plane_bounds_ref": "D0_SUBSTRATE",
            },
        }
    )
    d1_ground.update(
        {
            "semantic_id": "D1_BOTTOM_GROUND_PLANE",
            "net_id": "Ground",
            "geometry": {
                **d1_ground["geometry"],
                "geometry_source": "die_face_minus_ground_mask",
                "mask_layer": d1_mask,
                "include_layer": d1_draw,
                "include_selector_points_um": [(-62.0, -37.0)],
                "plane_bounds_ref": "D1_SUBSTRATE",
            },
            "metadata": {
                **d1_ground["metadata"],
                "logical_layer_id": "D1_BOTTOM_M1",
                "source_selector": "public lower-left junction ground-arm anchor",
            },
        }
    )
    bumps = _metal_record(
        layer=bump_layer,
        level=bump,
        host="D0_TO_D1_GAP",
        part_role="bump_body",
        representations={"A": "cutout_boundary_shell", "B": "cutout_boundary_shell"},
        source_layer_name="D0_D1_INDIUM_BUMP",
    )
    bumps.update(
        {
            "semantic_id": "D0_D1_INDIUM_BUMP",
            "net_id": "Ground",
            "geometry": {**bumps["geometry"], "split_polygons_as_entities": True},
        }
    )
    return {
        "solution_regions": {
            "AIR_BELOW": _solution(
                material_id=str(outer.material),
                z_min_um=d0_z_min - air_below_thickness,
                z_max_um=d0_z_min,
                domain_bounds_um=coupon_bounds_um,
                is_airbox=True,
                airbox_side="below",
            ),
            "D0_SUBSTRATE": _solution(
                material_id=str(d0_substrate.material),
                z_min_um=d0_z_min,
                z_max_um=d0_z_max,
                domain_bounds_um=coupon_bounds_um,
            ),
            "D0_TO_D1_GAP": _solution(
                material_id=str(gap.material),
                z_min_um=gap_z_min,
                z_max_um=gap_z_max,
                domain_bounds_um=coupon_bounds_um,
            ),
            "D1_SUBSTRATE": _solution(
                material_id=str(d1_substrate.material),
                z_min_um=d1_z_min,
                z_max_um=d1_z_max,
                domain_bounds_um=coupon_bounds_um,
            ),
            "AIR_ABOVE": _solution(
                material_id=str(outer.material),
                z_min_um=d1_z_max,
                z_max_um=d1_z_max + air_above_thickness,
                domain_bounds_um=coupon_bounds_um,
                is_airbox=True,
                airbox_side="above",
            ),
        },
        "materials": materials,
        "layers": [d0_ground, d1_ground, *signal_layers, bumps],
        "metadata": {
            "adapter": "scgsim.sgb.orpen.build_kosen2024_flip_chip_xmon_stack",
            "component_contract": "kosen2024_flip_chip_xmon_qubit public zero-argument cell",
            "excluded_layers": ("D1_BOTTOM_JJ_DRAW", "D0_D1_UNDER_BUMP"),
            "signal_group": signal_group,
            "coupon_domain_bounds_um": coupon_bounds_um,
            "coupon_padding_um": float(coupon_padding_um),
            "air_below_thickness_um": air_below_thickness,
            "air_above_thickness_um": air_above_thickness,
        },
    }


def _materials(records: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    kinds = {
        "superconductor": "conductor",
        "conductive": "conductor",
        "conductor": "conductor",
        "vacuum": "vacuum",
        "dielectric": "dielectric",
    }
    result: dict[str, dict[str, Any]] = {}
    for name, record in records.items():
        raw_kind = record.get("material_kind")
        if raw_kind not in kinds:
            continue
        normalized = kinds[raw_kind]
        item: dict[str, Any] = {"kind": normalized}
        if normalized != "conductor":
            permittivity = record.get("relative_permittivity")
            if isinstance(permittivity, (int, float)):
                item["permittivity"] = float(permittivity)
        loss_tangent = record.get("loss_tangent")
        if isinstance(loss_tangent, (int, float)):
            item["loss_tangent"] = float(loss_tangent)
        result[str(name)] = item
    if not result:
        raise ValueError(
            "public PDK material records contain no supported material kinds."
        )
    return result


def _solution(
    *,
    material_id: str,
    z_min_um: float,
    z_max_um: float,
    domain_bounds_um: Mapping[str, float],
    is_airbox: bool = False,
    airbox_side: str | None = None,
) -> dict[str, Any]:
    geometry: dict[str, Any] = {
        "z_min_um": z_min_um,
        "z_max_um": z_max_um,
        "domain_bounds_um": dict(domain_bounds_um),
    }
    record: dict[str, Any] = {
        "role": "solution_region",
        "is_airbox": is_airbox,
        "material_id": material_id,
        "geometry_kind": "domain",
        "geometry": geometry,
    }
    if airbox_side is not None:
        record["airbox_side"] = airbox_side
    return record


def _coupon_domain_bounds(component: Any, padding_um: float) -> dict[str, float]:
    padding = _positive_number(padding_um, "coupon_padding_um")
    bbox = component.dbbox()
    if bbox is None:
        raise ValueError("public Xmon component must have a physical-unit dbbox.")
    return {
        "x_min_um": float(bbox.left) - padding,
        "y_min_um": float(bbox.bottom) - padding,
        "x_max_um": float(bbox.right) + padding,
        "y_max_um": float(bbox.top) + padding,
    }


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number.")
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _metal_record(
    *,
    layer: tuple[int, int],
    level: Any,
    host: str,
    part_role: str,
    representations: Mapping[str, str],
    source_layer_name: str,
) -> dict[str, Any]:
    return {
        "layer": layer[0],
        "datatype": layer[1],
        "role": "metal",
        "material_id": str(level.material),
        "priority": int(getattr(level, "mesh_order", 0) or 0),
        "part_role": part_role,
        "host_void_semantic_id": host,
        "geometry_kind": "layout_extrusion",
        "geometry": {
            "z_um": _zmin(level),
            "thickness_um": _thickness(level),
            "geometry_source": "gds_polygon",
            "route_ab_fused_selector_mode": True,
        },
        "route_representations": dict(representations),
        "metadata": {
            "source_layer_name": source_layer_name,
            "source_authority": "orpen-sc-pdk public LayerStack",
        },
    }


def _signal_selectors(
    component: Any,
) -> tuple[tuple[str, str, tuple[float, float], str], ...]:
    ports = getattr(component, "ports", None)
    if ports is None:
        raise TypeError("public Xmon component must expose authored ports.")
    result = [("D1_XMON_PAD", "xmon_pad", (0.0, 0.0), "public topology anchor (0, 0)")]
    for index, name in enumerate(("o1", "o2", "o3", "o4"), 1):
        port = ports[name]
        center = tuple(float(value) for value in port.center)
        distance = float(port.width) / 2
        angle = radians(float(port.orientation))
        result.append(
            (
                f"D1_COUPLER_{index}",
                f"coupler_{index}",
                (center[0] - distance * cos(angle), center[1] - distance * sin(angle)),
                f"authored component port {name}",
            )
        )
    return tuple(result)


def _layer_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if hasattr(value, "layer") and hasattr(value, "datatype"):
        return int(value.layer), int(value.datatype)
    candidates = (value, getattr(value, "layer", None))
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            layout = candidate.layout
            raw = int(getattr(candidate, "layer", candidate))
            info = layout.get_info(raw)
            return int(info.layer), int(info.datatype)
        except (AttributeError, TypeError, ValueError):
            continue
    return None


def _zmin(level: Any) -> float:
    return float(level.zmin)


def _thickness(level: Any) -> float:
    value = float(level.thickness)
    if value <= 0:
        raise ValueError("public PDK LayerLevel thickness must be positive.")
    return value


__all__ = ["build_kosen2024_flip_chip_xmon_stack"]
