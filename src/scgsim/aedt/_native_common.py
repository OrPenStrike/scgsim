"""Narrow native helpers shared by AEDT solver-family runtimes.

This module never imports a family or owns a Desktop transaction.
"""

from __future__ import annotations

import math
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .spec import LOCKED_PYAEDT, HfssSpec, Q3dSpec


def import_and_bind(hfss: Any, spec: HfssSpec | Q3dSpec) -> list[dict[str, Any]]:
    mapping = {
        item.layer: [
            (item.z_min_um, item.z_max_um - item.z_min_um),
            item.layer_name,
        ]
        for item in spec.layer_imports
    }
    if not hfss.import_gds_3d(str(spec.gds_path), mapping, units="um", import_method=1):
        raise RuntimeError("HFSS import_gds_3d failed")
    hfss.modeler.refresh_all_ids()
    actual = set(hfss.modeler.object_names)
    expected = {item.object_name for item in spec.object_bindings}
    if actual != expected:
        raise RuntimeError(
            f"import object readback mismatch: expected {sorted(expected)!r}, got {sorted(actual)!r}"
        )
    layers = {item.layer: item for item in spec.layer_imports}
    materials = dict(spec.materials)
    pec: list[str] = []
    observed: list[dict[str, Any]] = []
    for binding in spec.object_bindings:
        obj = hfss.modeler.get_object_from_name(binding.object_name)
        if obj is None:
            raise RuntimeError(f"missing declared object {binding.object_name!r}")
        layer = layers[binding.layer]
        # AEDT's Geometry3D ``Group`` remains ``Model`` after GDS import. PyAEDT
        # 1.3.0 exposes the explicit destination layer through the imported
        # object-name prefix, so bind every declared object to that exact prefix.
        matches = [
            candidate
            for candidate in spec.layer_imports
            if obj.name.startswith(f"{candidate.layer_name}_")
        ]
        if matches != [layer]:
            raise RuntimeError(
                f"import destination-layer mismatch for {binding.object_name!r}: "
                f"expected exactly {layer.layer_name!r}, got "
                f"{[candidate.layer_name for candidate in matches]!r}"
            )
        material = materials[binding.material_id]
        record: dict[str, Any] = {
            "object_name": binding.object_name,
            "layer": binding.layer,
            "layer_name": layer.layer_name,
            "role": binding.role,
            "material_id": material.material_id,
            "kind": material.kind,
            "is_superconducting": material.is_superconducting,
            "requested_library_name": material.library_name,
            "native_destination_layer_prefix": layer.layer_name,
        }
        if material.is_superconducting:
            if isinstance(spec, Q3dSpec):
                obj.material_name = "pec"
                observed_material = native_object_property(obj, "Material").strip('"')
                if observed_material.casefold() != "pec":
                    raise RuntimeError(
                        f"Q3D PEC material readback mismatch for {binding.object_name!r}"
                    )
                record["observed"] = {
                    "native_material_name": observed_material,
                }
            else:
                pec.append(binding.object_name)
                record["requested_pec_boundary"] = "SCGSimPEC"
        else:
            existing = hfss.materials.exists_material(material.library_name)
            if not existing:
                raise RuntimeError(
                    f"AEDT library material is unavailable: {material.library_name!r}"
                )
            obj.material_name = material.library_name
            observed_material = native_object_property(obj, "Material").strip('"')
            if observed_material.casefold() != material.library_name.casefold():
                raise RuntimeError(
                    f"material readback mismatch for {binding.object_name!r}"
                )
            record["observed"] = {"native_material_name": observed_material}
        observed.append(record)
    if pec:
        boundary = hfss.assign_perfect_e(pec, name="SCGSimPEC")
        if boundary is None or "SCGSimPEC" not in native_boundary_names(hfss):
            raise RuntimeError("PEC assignment readback failed")
        assigned_face_ids = [
            int(object_id)
            for object_id in hfss.oboundary.GetBoundaryAssignment("SCGSimPEC")
        ]
        face_objects = {
            int(face.id): object_name
            for object_name in hfss.modeler.object_names
            for face in hfss.modeler.get_object_from_name(object_name).faces
        }
        assigned_objects = [face_objects[face_id] for face_id in assigned_face_ids]
        if assigned_objects != pec:
            raise RuntimeError("PEC native assignment mismatch")
        for record in observed:
            if record["is_superconducting"]:
                record["observed"] = {
                    "native_pec_boundary": "SCGSimPEC",
                    "native_pec_face_ids": assigned_face_ids,
                    "native_pec_objects": assigned_objects,
                }
    return observed


def create_region(hfss: Any, spec: HfssSpec | Q3dSpec) -> dict[str, Any]:
    if hfss.modeler.get_object_from_name("Region") is not None:
        raise RuntimeError("new V1 design unexpectedly already has Region")
    region = hfss.modeler.create_region(
        pad_value=list(spec.region_padding_um),
        pad_type="Absolute Offset",
        name="Region",
    )
    vacuum = spec.materials[spec.vacuum_material_id]
    if not hfss.materials.exists_material(vacuum.library_name):
        raise RuntimeError(
            f"AEDT vacuum material is unavailable: {vacuum.library_name!r}"
        )
    region.material_name = vacuum.library_name
    observed_material = native_object_property(region, "Material").strip('"')
    if (
        region.name != "Region"
        or observed_material.casefold() != vacuum.library_name.casefold()
    ):
        raise RuntimeError("vacuum region readback failed")
    result = {
        "material_id": vacuum.material_id,
        "requested_library_name": vacuum.library_name,
        "observed_material_name": observed_material,
        "padding_um": list(spec.region_padding_um),
    }
    if isinstance(spec, Q3dSpec):
        result["native_region_object_id"] = int(region.id)
        result["native_bounding_box_um"] = list(q3d_region_bounds(region))
        result["requested_padding_um"] = list(spec.region_padding_um)
    return result


def q3d_region_bounds(region: Any) -> tuple[float, float, float, float, float, float]:
    try:
        values = tuple(float(value) for value in region.bounding_box)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Q3D Region native bounds are unavailable") from exc
    if (
        len(values) != 6
        or not all(math.isfinite(value) for value in values)
        or any(values[index] >= values[index + 3] for index in range(3))
    ):
        raise RuntimeError("Q3D Region native bounds are invalid")
    return values  # type: ignore[return-value]


def saved_setup_properties(app: Any, setup_name: str) -> dict[str, Any]:
    analysis = app.design_properties.get("AnalysisSetup")
    setups = analysis.get("SolveSetups") if isinstance(analysis, dict) else None
    setup_names = (
        [name for name, value in setups.items() if isinstance(value, dict)]
        if isinstance(setups, dict)
        else []
    )
    if setup_names != [setup_name]:
        raise RuntimeError("saved AEDT project does not contain one exact setup")
    setup = setups[setup_name]
    if not isinstance(setup, dict):
        raise TypeError("saved AEDT setup properties are invalid")
    return setup


def pyaedt_version() -> str:
    try:
        installed = version("pyaedt")
    except PackageNotFoundError as exc:
        raise RuntimeError("PyAEDT is not installed") from exc
    if installed != LOCKED_PYAEDT:
        raise RuntimeError(
            f"PyAEDT version mismatch: expected {LOCKED_PYAEDT}, got {installed}"
        )
    return installed


def native_object_property(obj: Any, property_name: str) -> str:
    # Direct AEDT property query avoids the mutable PyAEDT object-property cache.
    value = obj._oeditor.GetPropertyValue(
        "Geometry3DAttributeTab", obj.name, property_name
    )
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"AEDT native object property {property_name!r} is unavailable for {obj.name!r}"
        )
    return value


def native_boundary_names(hfss: Any) -> list[str]:
    if "GetBoundaries" not in hfss.oboundary.__dir__():
        raise RuntimeError("AEDT native boundary collection is unavailable")
    return list(hfss.oboundary.GetBoundaries())


__all__ = [
    "create_region",
    "import_and_bind",
    "native_boundary_names",
    "native_object_property",
    "pyaedt_version",
    "q3d_region_bounds",
    "saved_setup_properties",
]
