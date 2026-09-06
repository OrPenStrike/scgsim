"""Explicit Surface-EPR lowering for structured SGB interface records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


def normalize_surface_epr_specs(
    specs: Mapping[str, Mapping[str, Any]],
    *,
    materials: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Normalize configured MA/MS/SA layers from numeric or explicit material data."""
    if not isinstance(specs, Mapping):
        raise TypeError("surface_epr specs must be a mapping.")
    unknown_types = set(specs).difference({"MA", "MS", "SA"})
    if unknown_types:
        raise ValueError(
            f"surface_epr has unsupported interface types: {sorted(unknown_types)}."
        )
    result: dict[str, dict[str, Any]] = {}
    for interface_type, raw in sorted(specs.items()):
        if not isinstance(raw, Mapping):
            raise TypeError(f"Surface EPR {interface_type} spec must be a mapping.")
        unknown = set(raw).difference(
            {
                "thickness",
                "permittivity",
                "material_id",
                "loss_tangent",
                "face_kinds",
                "required",
                "inset_margins_um",
            }
        )
        if unknown:
            raise ValueError(
                f"Surface EPR {interface_type} has unknown fields: {sorted(unknown)}."
            )
        thickness = raw.get("thickness")
        if not _positive(thickness):
            raise ValueError(
                f"Surface EPR {interface_type} requires positive thickness."
            )
        numeric = raw.get("permittivity")
        material_id = raw.get("material_id")
        if (numeric is None) == (material_id is None):
            raise ValueError(
                f"Surface EPR {interface_type} requires exactly one of permittivity or material_id."
            )
        resolved: dict[str, Any]
        if material_id is not None:
            if not isinstance(material_id, str) or not material_id.strip():
                raise ValueError(
                    f"Surface EPR {interface_type} material_id must be non-empty."
                )
            material = materials.get(material_id)
            if (
                not isinstance(material, Mapping)
                or material.get("kind") != "dielectric"
            ):
                raise ValueError(
                    f"Surface EPR {interface_type} material_id must resolve an explicit dielectric material."
                )
            numeric = material.get("permittivity")
            default_loss = material.get("loss_tangent", 0.0)
            resolved = {"material_id": material_id}
        else:
            default_loss = 0.0
            resolved = {}
        if not _positive(numeric):
            raise ValueError(
                f"Surface EPR {interface_type} requires positive resolved permittivity."
            )
        loss_tangent = raw.get("loss_tangent", default_loss)
        if not _nonnegative(loss_tangent):
            raise ValueError(
                f"Surface EPR {interface_type} requires non-negative loss_tangent."
            )
        face_kinds = raw.get("face_kinds")
        if face_kinds is not None:
            if not isinstance(face_kinds, (list, tuple)) or not face_kinds:
                raise ValueError(
                    f"Surface EPR {interface_type} face_kinds must be a non-empty sequence."
                )
            normalized_faces = sorted(
                {str(value).strip().lower() for value in face_kinds}
            )
            if not all(value for value in normalized_faces):
                raise ValueError(
                    f"Surface EPR {interface_type} face_kinds must be non-empty."
                )
            resolved["face_kinds"] = normalized_faces
        required = raw.get("required", True)
        if not isinstance(required, bool):
            raise TypeError(f"Surface EPR {interface_type} required must be a bool.")
        if "inset_margins_um" in raw:
            inset_margins_um = raw["inset_margins_um"]
            if isinstance(inset_margins_um, (str, bytes)) or not isinstance(
                inset_margins_um, (list, tuple)
            ):
                raise TypeError(
                    f"Surface EPR {interface_type} inset_margins_um must be an ordered non-empty sequence."
                )
            if not inset_margins_um:
                raise ValueError(
                    f"Surface EPR {interface_type} inset_margins_um must be non-empty."
                )
            normalized_margins = []
            for margin in inset_margins_um:
                if not _nonnegative(margin):
                    raise ValueError(
                        f"Surface EPR {interface_type} inset_margins_um values must be finite non-negative reals."
                    )
                normalized_margins.append(float(margin))
            resolved["inset_margins_um"] = normalized_margins
        result[interface_type] = {
            "thickness": float(thickness),
            "permittivity": float(numeric),
            "loss_tangent": float(loss_tangent),
            "required": required,
            **resolved,
        }
    return result


def build_surface_epr_postprocessing(
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    specs: Mapping[str, Mapping[str, Any]],
    *,
    model_l0_m: float = 1e-6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Emit Palace rows using only exact structured surface identity fields."""
    if not _positive(model_l0_m):
        raise ValueError("Model.L0 must be a finite positive length in metres.")
    rows: list[dict[str, Any]] = []
    index_map: list[dict[str, Any]] = []
    mask_requests: list[
        tuple[int, str, Mapping[str, Any], Mapping[str, Any], list[int], int, float]
    ]
    mask_requests = []
    masks_requested = any("inset_margins_um" in preset for preset in specs.values())
    for interface_type, preset in sorted(specs.items()):
        matches = []
        for name, info in sorted(groups.get("boundary_surfaces", {}).items()):
            if not isinstance(info, Mapping) or not info.get("surface_epr"):
                continue
            if info.get("interface_type") != interface_type:
                continue
            face_filter = preset.get("face_kinds")
            if face_filter is not None and info.get("face_kind") not in face_filter:
                continue
            matches.append((name, info))
        if not matches and preset["required"]:
            raise ValueError(
                f"Required Surface EPR {interface_type} has no matching structured group."
            )
        for name, info in matches:
            index = len(rows) + 1
            attrs = _attributes(info.get("phys_group"))
            rows.append(
                {
                    "Index": index,
                    "Attributes": attrs,
                    "Type": interface_type,
                    "Thickness": preset["thickness"],
                    "Permittivity": preset["permittivity"],
                    "LossTan": preset["loss_tangent"],
                }
            )
            baseline_entry = {
                "section": "Boundaries.Postprocessing.Dielectric",
                "index": index,
                "entry_name": name,
                "role": "surface_epr",
                "attributes": attrs,
                "physical_names": [name],
                "metadata": _structured_metadata(info),
                "epr_spec": dict(preset),
            }
            if masks_requested:
                baseline_entry["baseline_index"] = index
            index_map.append(baseline_entry)
            for margin_index, margin_um in enumerate(
                preset.get("inset_margins_um", ())
            ):
                mask_requests.append(
                    (index, name, info, preset, attrs, margin_index, float(margin_um))
                )

    for (
        baseline_index,
        name,
        info,
        preset,
        attrs,
        margin_index,
        margin_um,
    ) in mask_requests:
        # The input file coordinates are Model.L0 units, despite Palace's
        # historical MaskData::margin comment claiming metres.
        native_margin = margin_um * 1e-6 / float(model_l0_m)
        index = len(rows) + 1
        rows.append(
            {
                "Index": index,
                "Attributes": attrs,
                "Type": info["interface_type"],
                "Thickness": preset["thickness"],
                "Permittivity": preset["permittivity"],
                "LossTan": preset["loss_tangent"],
                "Mask": {"Type": "Inset", "Margin": native_margin},
            }
        )
        index_map.append(
            {
                "section": "Boundaries.Postprocessing.Dielectric",
                "index": index,
                "entry_name": f"{name}__inset_{margin_index}",
                "role": "surface_epr",
                "attributes": attrs,
                "physical_names": [name],
                "metadata": _structured_metadata(info),
                "epr_spec": dict(preset),
                "baseline_index": baseline_index,
                "mask": {
                    "type": "Inset",
                    "margin_um": margin_um,
                    "margin_index": margin_index,
                    "native_margin": native_margin,
                    "model_l0_m": float(model_l0_m),
                },
            }
        )
    return rows, index_map


def _structured_metadata(info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: info[key]
        for key in (
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
            "representation",
            "route",
        )
        if key in info
    }


def _attributes(raw: Any) -> list[int]:
    values = raw if isinstance(raw, (list, tuple, set)) else (raw,)
    result = [
        int(value)
        for value in values
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    if not result:
        raise ValueError("Surface EPR group requires physical attributes.")
    return sorted(set(result))


def _positive(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _nonnegative(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )
