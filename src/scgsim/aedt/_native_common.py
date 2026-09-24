"""Narrow native helpers shared by AEDT solver-family runtimes.

This module never imports a family or owns a Desktop transaction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .spec import AedtSpec, LOCKED_PYAEDT, HfssSpec, Q3dSpec, parse_aedt_spec


def _positive_identity(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} is invalid")
    return value


def _native_desktop_process_id(desktop: Any, label: str) -> int:
    try:
        value = desktop.odesktop.GetProcessID()
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(f"{label} is unavailable") from exc
    return _positive_identity(value, label)


def owned_application_constructor(
    factory: Callable[..., Any], desktop: Any
) -> Callable[..., Any]:
    """Pin one family application constructor to the transaction-owned Desktop."""

    expected_pid = _positive_identity(
        getattr(desktop, "aedt_process_id", None),
        "owned AEDT Desktop process identity",
    )
    expected_port = _positive_identity(
        getattr(desktop, "port", None), "owned AEDT Desktop endpoint"
    )
    native_pid = _native_desktop_process_id(
        desktop, "owned AEDT Desktop native process identity"
    )
    if native_pid != expected_pid:
        raise RuntimeError("owned AEDT Desktop native process identity is inconsistent")

    def construct(*args: Any, **kwargs: Any) -> Any:
        if "aedt_process_id" in kwargs or "port" in kwargs:
            raise TypeError("family application cannot override the owned AEDT identity")
        app = factory(
            *args,
            aedt_process_id=expected_pid,
            port=expected_port,
            **kwargs,
        )
        app_desktop = getattr(app, "desktop_class", None)
        actual_pid = _positive_identity(
            getattr(app_desktop, "aedt_process_id", None),
            "family application Desktop process identity",
        )
        actual_port = _positive_identity(
            getattr(app_desktop, "port", None),
            "family application Desktop endpoint",
        )
        if actual_pid != expected_pid or actual_port != expected_port:
            raise RuntimeError("family application did not bind the owned AEDT Desktop")
        actual_native_pid = _native_desktop_process_id(
            app_desktop, "family application native process identity"
        )
        if actual_native_pid != expected_pid:
            raise RuntimeError("family application native process identity changed")
        return app

    return construct


def _freeze_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("bound AEDT payload keys must be strings")
        return MappingProxyType(
            {key: _freeze_payload(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_payload(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"bound AEDT payload contains unsupported {type(value).__name__}")


def _thaw_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_payload(item) for item in value]
    return value


@dataclass(frozen=True)
class BoundAedtRequest:
    """One detached spec payload bound to one absolute execution workspace."""

    workspace: Path
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        workspace = Path(self.workspace).expanduser().resolve()
        payload = _freeze_payload(self.payload)
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "payload", payload)
        parse_aedt_spec(_thaw_payload(payload), base_dir=workspace)

    @classmethod
    def bind(cls, run_dir: str | Path, spec: AedtSpec) -> BoundAedtRequest:
        """Detach an already validated DTO without retaining caller containers."""
        return cls(Path(run_dir), spec.to_payload())

    def parse(self) -> AedtSpec:
        """Parse a fresh DTO using only the retained payload and bound root."""
        return parse_aedt_spec(_thaw_payload(self.payload), base_dir=self.workspace)

    def payload_copy(self) -> dict[str, Any]:
        """Return detached plain data for diagnostics and hashing."""
        return _thaw_payload(self.payload)


def detached_data(value: Any) -> Any:
    """Copy nested plain readback data without traversing native app handles."""
    if isinstance(value, Mapping):
        return {key: detached_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detached_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detached_data(item) for item in value)
    return value


def _native_object_evidence(hfss: Any, object_name: str) -> dict[str, Any]:
    """Read object identity, shape kind, and faces directly from the editor."""
    editor = hfss.modeler.oeditor
    object_id = int(editor.GetObjectIDByName(object_name))
    memberships = [
        object_type
        for object_type, group in (
            ("Solid", "Solids"),
            ("Sheet", "Sheets"),
            ("Line", "Lines"),
            ("Unclassified", "Unclassified"),
        )
        if object_name in {str(name) for name in editor.GetObjectsInGroup(group)}
    ]
    if len(memberships) != 1:
        raise RuntimeError(
            f"native object type is ambiguous for {object_name!r}: {memberships!r}"
        )
    face_ids = [int(face_id) for face_id in editor.GetFaceIDs(object_name)]
    if len(face_ids) != len(set(face_ids)) or not face_ids:
        raise RuntimeError(f"native object faces are invalid for {object_name!r}")
    return {
        "native_object_id": object_id,
        "native_object_type": memberships[0],
        "native_face_ids": face_ids,
    }


def _native_boundary_type(hfss: Any, name: str) -> str:
    raw = [str(value) for value in hfss.oboundary.GetBoundaries()]
    if len(raw) % 2:
        raise RuntimeError("native HFSS boundary list is invalid")
    matches = [raw[index + 1] for index in range(0, len(raw), 2) if raw[index] == name]
    if len(matches) != 1:
        raise RuntimeError(f"native boundary type is unavailable for {name!r}")
    return matches[0]


def _resolve_native_assignment(
    hfss: Any, raw_ids: list[int]
) -> tuple[list[dict[str, Any]], set[int], set[str]]:
    """Resolve AEDT's face-or-object boundary IDs without guessing their kind."""
    by_object_id: dict[int, tuple[str, list[int]]] = {}
    by_face_id: dict[int, str] = {}
    for object_name in [str(name) for name in hfss.modeler.object_names]:
        native = _native_object_evidence(hfss, object_name)
        object_id = native["native_object_id"]
        face_ids = native["native_face_ids"]
        if object_id in by_object_id:
            raise RuntimeError(f"duplicate native object ID: {object_id}")
        by_object_id[object_id] = (object_name, face_ids)
        for face_id in face_ids:
            if face_id in by_face_id:
                raise RuntimeError(f"ambiguous native face ID: {face_id}")
            by_face_id[face_id] = object_name

    resolved: list[dict[str, Any]] = []
    covered_faces: set[int] = set()
    covered_objects: set[str] = set()
    for raw_id in raw_ids:
        object_match = by_object_id.get(raw_id)
        face_match = by_face_id.get(raw_id)
        if object_match is not None and face_match is not None:
            raise RuntimeError(f"ambiguous PEC native assignment ID: {raw_id}")
        if object_match is not None:
            object_name, face_ids = object_match
            resolved.append(
                {
                    "raw_id": raw_id,
                    "native_kind": "object",
                    "object_name": object_name,
                    "face_ids": list(face_ids),
                }
            )
            covered_faces.update(face_ids)
            covered_objects.add(object_name)
        elif face_match is not None:
            resolved.append(
                {
                    "raw_id": raw_id,
                    "native_kind": "face",
                    "object_name": face_match,
                    "face_ids": [raw_id],
                }
            )
            covered_faces.add(raw_id)
            covered_objects.add(face_match)
        else:
            raise RuntimeError(f"unknown PEC native assignment ID: {raw_id}")
    return resolved, covered_faces, covered_objects


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
    pec_sheets: list[str] = []
    pec_solids: list[str] = []
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
                native = _native_object_evidence(hfss, binding.object_name)
                if native["native_object_type"] == "Solid":
                    pec_solids.append(binding.object_name)
                    obj.material_name = "pec"
                    obj.solve_inside = False
                    fresh = _native_object_evidence(hfss, binding.object_name)
                    if (
                        fresh["native_object_id"] != native["native_object_id"]
                        or fresh["native_object_type"] != native["native_object_type"]
                        or set(fresh["native_face_ids"])
                        != set(native["native_face_ids"])
                    ):
                        raise RuntimeError(
                            f"HFSS PEC native object identity changed for {binding.object_name!r}"
                        )
                    observed_material = native_object_property(obj, "Material").strip(
                        '"'
                    )
                    observed_solve_inside = _native_object_boolean_property(
                        obj, "Solve Inside"
                    )
                    if observed_material.casefold() != "pec" or observed_solve_inside:
                        raise RuntimeError(
                            f"HFSS solid PEC readback mismatch for {binding.object_name!r}"
                        )
                    record["hfss_pec_binding"] = {
                        "source_object": binding.object_name,
                        "source_material_id": material.material_id,
                        "source_material_kind": material.kind,
                        "source_library_name": material.library_name,
                        **fresh,
                        "implementation": "pec_material_solve_inside_false",
                        "verified_evidence": {
                            "native_material_name": observed_material,
                            "native_solve_inside": False,
                        },
                    }
                    record["observed"] = {
                        "native_material_name": observed_material,
                        "native_solve_inside": False,
                    }
                elif native["native_object_type"] == "Sheet":
                    pec_sheets.append(binding.object_name)
                    record["requested_pec_boundary"] = "SCGSimPEC"
                    record["hfss_pec_binding"] = {
                        "source_object": binding.object_name,
                        "source_material_id": material.material_id,
                        "source_material_kind": material.kind,
                        "source_library_name": material.library_name,
                        **native,
                        "implementation": "perfect_e_sheet",
                    }
                else:
                    raise RuntimeError(
                        f"unsupported HFSS PEC native object type for "
                        f"{binding.object_name!r}: {native['native_object_type']!r}"
                    )
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
    if pec_sheets:
        boundary = hfss.assign_perfect_e(pec_sheets, name="SCGSimPEC")
        if boundary is None or "SCGSimPEC" not in native_boundary_names(hfss):
            raise RuntimeError("PEC assignment readback failed")
        boundary_type = _native_boundary_type(hfss, "SCGSimPEC")
        if boundary_type != "Perfect E":
            raise RuntimeError("PEC native boundary type mismatch")
        raw_assignment_ids = [
            int(native_id)
            for native_id in hfss.oboundary.GetBoundaryAssignment("SCGSimPEC")
        ]
        resolved, covered_faces, covered_objects = _resolve_native_assignment(
            hfss, raw_assignment_ids
        )
        target_faces = {
            int(face_id)
            for record in observed
            if record.get("hfss_pec_binding", {}).get("implementation")
            == "perfect_e_sheet"
            for face_id in record["hfss_pec_binding"]["native_face_ids"]
        }
        if covered_faces != target_faces or covered_objects != set(pec_sheets):
            raise RuntimeError("PEC native assignment mismatch")
        for record in observed:
            binding = record.get("hfss_pec_binding")
            if (
                isinstance(binding, dict)
                and binding.get("implementation") == "perfect_e_sheet"
            ):
                target = set(binding["native_face_ids"])
                binding["verified_evidence"] = {
                    "native_boundary_name": "SCGSimPEC",
                    "native_boundary_type": boundary_type,
                    "raw_assignment_ids": raw_assignment_ids,
                    "typed_assignment": resolved,
                    "covered_face_ids": sorted(covered_faces & target),
                }
                record["observed"] = {
                    "native_pec_boundary": "SCGSimPEC",
                    "native_pec_boundary_type": boundary_type,
                    "native_pec_face_ids": sorted(covered_faces & target),
                    "native_pec_objects": sorted(covered_objects),
                }
    elif pec_solids and "SCGSimPEC" in native_boundary_names(hfss):
        raise RuntimeError("solid PEC binding unexpectedly has SCGSimPEC boundary")
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


def _native_object_boolean_property(obj: Any, property_name: str) -> bool:
    # Keep boolean readback direct too; AEDT transports it as either bool or text.
    value = obj._oeditor.GetPropertyValue(
        "Geometry3DAttributeTab", obj.name, property_name
    )
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    raise RuntimeError(
        f"AEDT native boolean property {property_name!r} is unavailable for {obj.name!r}"
    )


def native_boundary_names(hfss: Any) -> list[str]:
    if "GetBoundaries" not in hfss.oboundary.__dir__():
        raise RuntimeError("AEDT native boundary collection is unavailable")
    return list(hfss.oboundary.GetBoundaries())


__all__ = [
    "create_region",
    "import_and_bind",
    "native_boundary_names",
    "native_object_property",
    "owned_application_constructor",
    "pyaedt_version",
    "q3d_region_bounds",
    "saved_setup_properties",
]
