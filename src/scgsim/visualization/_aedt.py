"""AEDT 2024.2 non-graphical geometry-preview loader."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any

from scgsim.aedt.spec import LOCKED_PYAEDT, REQUIRED_AEDT_VERSION

from ._preview import (
    _FIXED_COLORS,
    GeometryPreview,
    _color,
    _confined,
    _Mode,
    _Part,
    _receipt_authority_sha256,
    _sha256,
)


def inspect_aedt_geometry(
    source: str | Path,
    *,
    design: str | None = None,
) -> GeometryPreview:
    """Open one AEDT project headlessly and bind native geometry readback."""
    source_path = Path(source).expanduser().resolve()
    if source_path.is_file():
        if source_path.suffix.lower() != ".aedt":
            raise ValueError("raw AEDT geometry source must be an .aedt project")
        if not design or not design.strip():
            raise ValueError("raw .aedt preview requires an explicit design")
        root = source_path.parent
        project_path = source_path
        receipt: dict[str, Any] | None = None
        receipt_path: Path | None = None
    elif source_path.is_dir():
        root = source_path
        spec_path = root / "aedt_spec.json"
        if not spec_path.is_file():
            raise FileNotFoundError("AEDT run folder has no aedt_spec.json")
        spec = _json(spec_path)
        project = _mapping(spec.get("project"), "spec.project")
        recorded_design = _text(project.get("design"), "spec.project.design")
        if design is not None and design != recorded_design:
            raise ValueError("explicit design disagrees with the AEDT run spec")
        design = recorded_design
        project_name = _text(project.get("name"), "spec.project.name")
        project_path = _confined(root, f"{project_name}.aedt")
        receipt_path = root / "metadata/aedt_run_receipt.json"
        receipt = _json(receipt_path) if receipt_path.is_file() else None
        if receipt is not None:
            if receipt.get("project") != project_path.relative_to(root).as_posix():
                raise ValueError(
                    "AEDT receipt project identity disagrees with the spec"
                )
            outputs = _mapping(receipt.get("outputs", {}), "receipt.outputs")
            relative = project_path.relative_to(root).as_posix()
            expected = outputs.get(relative)
            if not isinstance(expected, str) or _sha256(project_path) != expected:
                raise ValueError("AEDT project hash disagrees with the run receipt")
    else:
        raise FileNotFoundError(f"AEDT geometry source does not exist: {source_path}")
    if not project_path.is_file():
        raise FileNotFoundError(f"AEDT project is missing: {project_path.name}")

    try:
        import pyvista as pv
        from ansys.aedt.core import Desktop
    except ImportError as exc:
        raise RuntimeError(
            "AEDT geometry preview requires scgsim[aedt,visualization]"
        ) from exc
    if version("pyaedt") != LOCKED_PYAEDT:
        raise RuntimeError(
            f"AEDT preview requires PyAEDT {LOCKED_PYAEDT}, got {version('pyaedt')}"
        )

    desktop = None
    app = None
    try:
        desktop = Desktop(
            version=REQUIRED_AEDT_VERSION,
            non_graphical=True,
            new_desktop=True,
            close_on_exit=False,
        )
        if desktop.aedt_version_id != REQUIRED_AEDT_VERSION:
            raise RuntimeError("AEDT Desktop release does not match the locked runtime")
        app = desktop.load_project(str(project_path), design_name=design)
        if app is False or app is None:
            raise RuntimeError("PyAEDT could not open the requested project/design")
        if str(app.design_name) != design:
            raise ValueError("native AEDT design readback disagrees with the request")
        app.modeler.refresh_all_ids()
        object_names = tuple(str(name) for name in app.modeler.object_names)
        if not object_names:
            raise ValueError("AEDT design contains no model objects")
        with tempfile.TemporaryDirectory(prefix="scgsim-aedt-preview-") as raw:
            exported = app.post.export_model_obj(
                assignment=list(object_names),
                export_path=raw,
                export_as_multiple_objects=True,
                air_objects=True,
            )
            datasets: dict[str, Any] = {}
            for item in exported:
                path = Path(item[0])
                name = path.stem
                if name in datasets:
                    raise ValueError(f"AEDT exported duplicate object {name}")
                datasets[name] = pv.read(path).copy(deep=True)
        missing = sorted(set(object_names) - set(datasets))
        if missing:
            raise ValueError(f"AEDT OBJ export omitted native objects: {missing!r}")

        if receipt is not None:
            _validate_native_receipt(app, receipt)
        materials = _materials(app, datasets, receipt)
        boundaries = _boundaries(app, datasets, receipt, pv)
    finally:
        if desktop is not None:
            released = desktop.release_desktop(close_projects=True, close_on_exit=True)
            if released is False:
                raise RuntimeError("AEDT Desktop did not confirm release after preview")

    source_hashes = {project_path.relative_to(root).as_posix(): _sha256(project_path)}
    if root != project_path.parent or (root / "aedt_spec.json").is_file():
        for relative in ("aedt_spec.json",):
            path = root / relative
            if path.is_file():
                source_hashes[relative] = _sha256(path)
    source_identity: dict[str, Any] = {
        "design": design,
        "aedt_version": REQUIRED_AEDT_VERSION,
        "pyaedt_version": LOCKED_PYAEDT,
    }
    if receipt_path is not None and receipt is not None:
        source_identity["receipt_authority_sha256"] = _receipt_authority_sha256(
            receipt_path
        )
    return GeometryPreview(
        root=root,
        backend="aedt",
        source_hashes=source_hashes,
        source_identity=source_identity,
        bind_aedt_receipt=receipt is not None,
        modes={
            "materials": _Mode(tuple(materials)),
            "boundaries": _Mode(tuple(boundaries)),
            "surface_epr": _Mode(
                unavailable_reason="AEDT V1 has no structured Surface-EPR assignment authority."
            ),
            "mesh": _Mode(
                unavailable_reason="AEDT V1 has no readable native mesh artifact."
            ),
        },
    )


def _validate_native_receipt(app: Any, receipt: Mapping[str, Any]) -> None:
    mode = receipt.get("mode")
    object_ids = {name: int(app.modeler[name].id) for name in app.modeler.object_names}
    if mode in {"terminal", "modal", "eigenmode"}:
        native_boundaries = {
            str(boundary.name): dict(boundary.props) for boundary in app.boundaries
        }
        pec_records = {
            (
                str(item.get("observed", {}).get("native_pec_boundary")),
                tuple(item.get("observed", {}).get("native_pec_objects", ())),
            )
            for item in receipt.get("materials", ())
            if item.get("observed", {}).get("native_pec_boundary")
        }
        for boundary_name, object_names in pec_records:
            props = native_boundaries.get(boundary_name)
            if props is None or props.get("BoundType") != "Perfect E":
                raise ValueError("native AEDT PEC boundary disagrees with the receipt")
            expected_ids = sorted(object_ids[name] for name in object_names)
            if sorted(int(value) for value in props.get("Objects", ())) != expected_ids:
                raise ValueError("native AEDT PEC object assignment changed")
        for port in receipt.get("ports", ()):
            boundary_name = _text(port.get("boundary"), "receipt port boundary")
            props = native_boundaries.get(boundary_name)
            if props is None or props.get("BoundType") not in {
                "Wave Port",
                "Lumped Port",
            }:
                raise ValueError("native AEDT port boundary disagrees with the receipt")
            if int(port["face_id"]) not in {
                int(value) for value in props.get("Faces", ())
            }:
                raise ValueError("native AEDT port face assignment changed")
        if mode == "eigenmode" and receipt.get("ports"):
            raise ValueError("AEDT Eigenmode receipt must not contain ports")
    elif mode == "q3d":
        if list(app.net_names) != [
            str(item.get("name")) for item in receipt.get("nets", ())
        ]:
            raise ValueError("native AEDT Q3D net order changed")
        for net in receipt.get("nets", ()):
            names = tuple(net.get("object_names", ()))
            expected = sorted(object_ids[name] for name in names)
            native = sorted(
                int(value)
                for value in app.oboundary.GetExcitationAssignment(str(net.get("name")))
            )
            if native != expected or native != sorted(
                int(value) for value in net.get("native_object_ids", ())
            ):
                raise ValueError("native AEDT Q3D net object assignment changed")
            for role in ("source", "sink"):
                record = net.get(role)
                if record is None:
                    continue
                record = _mapping(record, f"receipt Q3D {role}")
                native_faces = sorted(
                    int(value)
                    for value in app.oboundary.GetExcitationAssignment(
                        _text(record.get("name"), f"receipt Q3D {role} name")
                    )
                )
                if native_faces != sorted(
                    int(value) for value in record.get("native_face_ids", ())
                ):
                    raise ValueError(f"native AEDT Q3D {role} assignment changed")
    elif mode == "q2d":
        for conductor in receipt.get("conductors", ()):
            names = tuple(conductor.get("object_names", ()))
            expected = sorted(object_ids[name] for name in names)
            native = sorted(
                int(value)
                for value in app.oboundary.GetExcitationAssignment(
                    str(conductor.get("name"))
                )
            )
            if native != expected or native != sorted(
                int(value) for value in conductor.get("native_object_ids", ())
            ):
                raise ValueError("native AEDT Q2D conductor assignment changed")
    else:
        raise ValueError(f"unsupported AEDT receipt mode: {mode!r}")


def _materials(
    app: Any, datasets: Mapping[str, Any], receipt: dict[str, Any] | None
) -> list[_Part]:
    receipt_items: dict[str, Mapping[str, Any]] = {}
    pdk_materials: Mapping[str, Any] = {}
    region: Mapping[str, Any] = {}
    if receipt is not None:
        receipt_items = {
            _text(item.get("object_name"), "receipt.materials.object_name"): item
            for item in receipt.get("materials", ())
        }
        pdk_materials = _mapping(
            receipt.get("pdk_materials", {}), "receipt.pdk_materials"
        )
        region = _mapping(receipt.get("region", {}), "receipt.region")
    parts: list[_Part] = []
    for object_name, dataset in sorted(datasets.items()):
        native_material = str(app.modeler[object_name].material_name).lower()
        item = receipt_items.get(object_name)
        if item is not None:
            material_id = _text(item.get("material_id"), "receipt material_id")
            material = _mapping(pdk_materials.get(material_id), "receipt PDK material")
            kind = _text(material.get("kind"), "receipt material kind")
            observed = _mapping(item.get("observed", {}), "receipt material observed")
            observed_name = observed.get("native_material_name")
            if (
                observed_name is not None
                and str(observed_name).lower() != native_material
            ):
                raise ValueError(f"native AEDT material changed for {object_name}")
        elif object_name == "Region" and region:
            material_id = _text(region.get("material_id"), "receipt region material_id")
            material = _mapping(pdk_materials.get(material_id), "receipt PDK material")
            kind = _text(material.get("kind"), "receipt material kind")
            if str(region.get("observed_material_name", "")).lower() != native_material:
                raise ValueError("native AEDT Region material changed")
        elif receipt is not None:
            raise ValueError(f"AEDT receipt has no material identity for {object_name}")
        else:
            material_id = native_material
            kind = "native AEDT material"
        parts.append(
            _Part(
                dataset,
                f"material:{material_id}",
                material_id,
                kind,
                _color(material_id),
                0.08 if kind == "vacuum" else 0.42,
                1,
            )
        )
    return parts


def _boundaries(
    app: Any,
    datasets: Mapping[str, Any],
    receipt: dict[str, Any] | None,
    pv: Any,
) -> list[_Part]:
    assigned: set[str] = set()
    parts: list[_Part] = []
    if receipt is not None:
        mode = receipt.get("mode")
        if mode in {"terminal", "modal", "eigenmode"}:
            for item in receipt.get("materials", ()):
                observed = _mapping(item.get("observed", {}), "receipt observed")
                native_objects = tuple(observed.get("native_pec_objects", ()))
                if native_objects:
                    object_name = _text(item.get("object_name"), "receipt object_name")
                    if object_name not in native_objects:
                        raise ValueError(
                            "AEDT PEC receipt object readback is inconsistent"
                        )
                    _append_object(parts, datasets, assigned, object_name, "PEC", "PEC")
            for port in receipt.get("ports", ()):
                face_id = int(port["face_id"])
                face = app.modeler.get_face_by_id(face_id)
                if face is None:
                    raise ValueError(f"AEDT native port face {face_id} is unavailable")
                points = [
                    tuple(float(value) for value in vertex.position)
                    for vertex in face.vertices
                ]
                if len(points) < 3:
                    raise ValueError(f"AEDT native port face {face_id} is degenerate")
                dataset = pv.PolyData(points, faces=[len(points), *range(len(points))])
                label = _text(port.get("boundary"), "receipt port boundary")
                parts.append(
                    _Part(
                        dataset,
                        f"port:{label}",
                        label,
                        "LumpedPort",
                        _FIXED_COLORS["LumpedPort"],
                        1.0,
                        1,
                    )
                )
        elif mode == "q3d":
            for net in receipt.get("nets", ()):
                role = _text(net.get("net_type"), "receipt net_type")
                name = _text(net.get("name"), "receipt net name")
                for object_name in net.get("object_names", ()):
                    object_name = _text(object_name, "receipt net object")
                    _append_object(
                        parts,
                        datasets,
                        assigned,
                        object_name,
                        name,
                        "Ground" if role == "Ground" else "Terminal",
                    )
        elif mode == "q2d":
            for conductor in receipt.get("conductors", ()):
                role = _text(conductor.get("conductor_type"), "receipt conductor_type")
                name = _text(conductor.get("name"), "receipt conductor name")
                for object_name in conductor.get("object_names", ()):
                    object_name = _text(object_name, "receipt conductor object")
                    _append_object(
                        parts,
                        datasets,
                        assigned,
                        object_name,
                        name,
                        "Ground" if role == "ReferenceGround" else "Terminal",
                    )
        else:
            raise ValueError(f"unsupported AEDT receipt mode: {mode!r}")
    else:
        for boundary in app.boundaries:
            props = dict(boundary.props)
            boundary_type = str(props.get("BoundType") or getattr(boundary, "type", ""))
            name = str(getattr(boundary, "name", props.get("Name", boundary_type)))
            if boundary_type in {"Terminal"}:
                continue
            if boundary_type not in {"Perfect E", "Wave Port", "Lumped Port"}:
                raise ValueError(
                    f"unsupported native AEDT boundary type: {boundary_type!r}"
                )
            role = "PEC" if boundary_type == "Perfect E" else "LumpedPort"
            for object_name in _native_object_names(app, props.get("Objects", ())):
                _append_object(parts, datasets, assigned, object_name, name, role)
            for face_id in props.get("Faces", ()):
                face = app.modeler.get_face_by_id(int(face_id))
                if face is None:
                    raise ValueError(
                        f"native AEDT boundary face {face_id} is unavailable"
                    )
                points = [
                    tuple(float(value) for value in vertex.position)
                    for vertex in face.vertices
                ]
                dataset = pv.PolyData(points, faces=[len(points), *range(len(points))])
                parts.append(
                    _Part(dataset, f"boundary:{name}", name, role, _color(role), 1.0, 1)
                )
    for object_name, dataset in sorted(datasets.items()):
        if object_name not in assigned:
            parts.append(
                _Part(
                    dataset,
                    f"unassigned:{object_name}",
                    f"{object_name}: no explicit boundary assignment",
                    "unassigned",
                    _FIXED_COLORS["unassigned"],
                    0.35,
                    1,
                )
            )
    return parts


def _append_object(
    parts: list[_Part],
    datasets: Mapping[str, Any],
    assigned: set[str],
    object_name: str,
    label: str,
    role: str,
) -> None:
    if object_name not in datasets:
        raise ValueError(f"AEDT boundary object was not exported: {object_name}")
    if object_name in assigned:
        return
    assigned.add(object_name)
    parts.append(
        _Part(
            datasets[object_name],
            f"boundary:{label}:{object_name}",
            label,
            role,
            _color(role if role in _FIXED_COLORS else label),
            0.95,
            1,
        )
    )


def _native_object_names(app: Any, values: Any) -> tuple[str, ...]:
    by_id = {int(app.modeler[name].id): str(name) for name in app.modeler.object_names}
    names: list[str] = []
    for value in values:
        if str(value) in app.modeler.object_names:
            names.append(str(value))
            continue
        try:
            names.append(by_id[int(value)])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"native AEDT boundary object is unavailable: {value!r}"
            ) from exc
    return tuple(names)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain a JSON object")
    return payload


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be non-empty text")
    return value.strip()
