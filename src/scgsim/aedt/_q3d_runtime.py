"""Q3D native preparation, solve, and export stages.

PreparedQ3d is a frozen carrier for one mutable native application handle; it
is not a deeply immutable semantic snapshot and does not own the Desktop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._matrix_export import parse_matrix_export
from ._native_common import (
    BoundAedtRequest,
    create_region as _create_region,
    detached_data,
    import_and_bind as _import_and_bind,
    native_object_property as _native_object_property,
    pyaedt_version as _pyaedt_version,
    q3d_region_bounds as _q3d_region_bounds,
    saved_setup_properties as _saved_setup_properties,
)
from ._q2d_convergence import read_q3d_convergence
from .spec import REQUIRED_AEDT_VERSION, Q3dSpec
from .util import file_sha256, write_csv

_Q3D_REGION_DIRECTIONS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
_Q3D_REGION_SHEET_NAMES = {
    "+X": ("SCGSimRegionGroundPX", "SCGSimRegionGroundPXThinConductor"),
    "-X": ("SCGSimRegionGroundNX", "SCGSimRegionGroundNXThinConductor"),
    "+Y": ("SCGSimRegionGroundPY", "SCGSimRegionGroundPYThinConductor"),
    "-Y": ("SCGSimRegionGroundNY", "SCGSimRegionGroundNYThinConductor"),
    "+Z": ("SCGSimRegionGroundPZ", "SCGSimRegionGroundPZThinConductor"),
    "-Z": ("SCGSimRegionGroundNZ", "SCGSimRegionGroundNZThinConductor"),
}


@dataclass(frozen=True)
class PreparedQ3d:
    """Carrier for one prepared Q3D app and its native readback records."""

    app: Any
    request: BoundAedtRequest
    project_path: Path
    materials: list[dict[str, Any]]
    region: dict[str, Any]
    nets: list[dict[str, Any]]
    setup: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.request, BoundAedtRequest):
            raise TypeError("PreparedQ3d requires a bound AEDT request")
        for name in ("materials", "region", "nets", "setup"):
            object.__setattr__(self, name, detached_data(getattr(self, name)))


def run_q3d(Q3d: Any, run_dir: Path, spec: Q3dSpec) -> dict[str, Any]:
    """Use the same preparation stage as diagnostics, then solve and export."""
    prepared = prepare_q3d(Q3d, run_dir, spec)
    solve_q3d(prepared)
    return export_q3d(prepared)


def prepare_q3d(Q3d: Any, run_dir: Path, spec: Q3dSpec) -> PreparedQ3d:
    """Construct and read back the Q3D model before the explicit solve stage."""
    request = BoundAedtRequest.bind(run_dir, spec)
    bound_spec = request.parse()
    if not isinstance(bound_spec, Q3dSpec):
        raise TypeError("bound Q3D request did not retain a Q3D spec")
    run_dir = request.workspace
    spec = bound_spec
    project_path = run_dir / f"{spec.project_name}.aedt"
    app = Q3d(
        project=str(project_path),
        design=spec.design_name,
        new_desktop=False,
        close_on_exit=False,
    )
    if app.desktop_class.aedt_version_id != REQUIRED_AEDT_VERSION:
        raise RuntimeError("Q3D did not bind the owned AEDT 2024.2 desktop")
    app.modeler.model_units = "um"
    materials = _import_and_bind(app, spec)
    region = _create_region(app, spec)
    nets = _assign_q3d_nets(app, spec)
    if spec.grounded_region_net is not None:
        grounded_region = _seal_q3d_region(app, spec)
        region["native_region_object_id"] = grounded_region["native_region_object_id"]
        region["native_bounding_box_um"] = grounded_region["native_bounding_box_um"]
        region["grounded_region"] = grounded_region
    _setup_q3d(app, spec)
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("Q3D project was not saved before solve")
    setup = _read_q3d_setup(app, spec)
    if spec.grounded_region_net is not None:
        region["grounded_region"] = _read_q3d_region_ground(app, spec, region)
        region["native_region_object_id"] = region["grounded_region"][
            "native_region_object_id"
        ]
        region["native_bounding_box_um"] = region["grounded_region"][
            "native_bounding_box_um"
        ]
        if app.validate_simple() != 1:
            raise RuntimeError("Q3D grounded Region failed native ValidateDesign")
        region["grounded_region"]["native_design_validation"] = {
            "method": "ValidateDesign",
            "ok": True,
        }
    return PreparedQ3d(
        app,
        request,
        project_path,
        materials,
        region,
        nets,
        setup,
    )


def solve_q3d(prepared: PreparedQ3d) -> None:
    """Run the one explicit Q3D setup solve."""
    spec = prepared.request.parse()
    if not isinstance(spec, Q3dSpec):
        raise TypeError("bound Q3D request did not retain a Q3D spec")
    if not prepared.app.analyze_setup(name=spec.run_control.setup_name, blocking=True):
        raise RuntimeError(
            f"Q3D failed to analyze setup {spec.run_control.setup_name!r}"
        )


def export_q3d(prepared: PreparedQ3d) -> dict[str, Any]:
    """Export, perform the final save, and bind convergence readback."""
    run_dir = prepared.request.workspace
    spec = prepared.request.parse()
    if not isinstance(spec, Q3dSpec):
        raise TypeError("bound Q3D request did not retain a Q3D spec")
    outputs, result_readback = _export_q3d(prepared.app, run_dir, spec)
    outputs = detached_data(outputs)
    result_readback = detached_data(result_readback)
    if not prepared.app.save_project() or not prepared.project_path.is_file():
        raise RuntimeError("Q3D project was not saved")
    region = detached_data(prepared.region)
    if spec.grounded_region_net is not None:
        region["grounded_region"] = _read_q3d_region_ground(prepared.app, spec, region)
        region["native_region_object_id"] = region["grounded_region"][
            "native_region_object_id"
        ]
        region["native_bounding_box_um"] = region["grounded_region"][
            "native_bounding_box_um"
        ]
    convergence = read_q3d_convergence(run_dir, spec)
    relative = prepared.project_path.relative_to(run_dir).as_posix()
    outputs[relative] = file_sha256(prepared.project_path)
    return {
        "outputs": outputs,
        "connected": {
            "aedt_version": prepared.app.desktop_class.aedt_version_id,
            "pyaedt_version": _pyaedt_version(),
        },
        "project": relative,
        "nets": detached_data(prepared.nets),
        "materials": detached_data(prepared.materials),
        "region": region,
        "setup": detached_data(prepared.setup),
        "convergence": convergence,
        "result_readback": result_readback,
        "save": {"ok": True, "project_sha256": outputs[relative]},
    }


def _q3d_region_faces(
    region: Any, bounds: tuple[float, float, float, float, float, float]
) -> dict[str, tuple[Any, tuple[float, float, float], tuple[float, float, float]]]:
    faces: dict[
        str, tuple[Any, tuple[float, float, float], tuple[float, float, float]]
    ] = {}
    for face in region.faces:
        try:
            center = tuple(float(value) for value in face.center)
            normal = tuple(float(value) for value in face.normal)
            face_id = int(face.id)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Q3D Region native face readback is invalid") from exc
        if (
            face_id <= 0
            or len(center) != 3
            or len(normal) != 3
            or not all(math.isfinite(value) for value in [*center, *normal])
        ):
            raise RuntimeError("Q3D Region native face readback is invalid")
        matches = [
            direction
            for direction, axis, boundary in (
                ("+X", 0, bounds[3]),
                ("-X", 0, bounds[0]),
                ("+Y", 1, bounds[4]),
                ("-Y", 1, bounds[1]),
                ("+Z", 2, bounds[5]),
                ("-Z", 2, bounds[2]),
            )
            if math.isclose(center[axis], boundary, rel_tol=0.0, abs_tol=1e-9)
        ]
        if len(matches) != 1 or matches[0] in faces:
            raise RuntimeError("Q3D Region native faces do not identify six directions")
        direction = matches[0]
        expected_center = _q3d_face_center(direction, bounds)
        if not all(
            math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-9)
            for value, expected in zip(center, expected_center, strict=True)
        ):
            raise RuntimeError("Q3D Region native face center is invalid")
        axis, sign = {
            "+X": (0, 1.0),
            "-X": (0, -1.0),
            "+Y": (1, 1.0),
            "-Y": (1, -1.0),
            "+Z": (2, 1.0),
            "-Z": (2, -1.0),
        }[direction]
        if not math.isclose(normal[axis], sign, rel_tol=0.0, abs_tol=1e-9) or any(
            not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-9)
            for index, value in enumerate(normal)
            if index != axis
        ):
            raise RuntimeError("Q3D Region native face normal is invalid")
        faces[direction] = (face, center, normal)  # type: ignore[assignment]
    if set(faces) != set(_Q3D_REGION_DIRECTIONS):
        raise RuntimeError("Q3D Region must expose exactly one face in each direction")
    if len({int(face.id) for face, _, _ in faces.values()}) != len(faces):
        raise RuntimeError("Q3D Region native face IDs are not unique")
    return faces


def _q3d_face_center(
    direction: str, bounds: tuple[float, float, float, float, float, float]
) -> list[float]:
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2
    return {
        "+X": [x_max, y_mid, z_mid],
        "-X": [x_min, y_mid, z_mid],
        "+Y": [x_mid, y_max, z_mid],
        "-Y": [x_mid, y_min, z_mid],
        "+Z": [x_mid, y_mid, z_max],
        "-Z": [x_mid, y_mid, z_min],
    }[direction]


def _q3d_region_evidence(
    region: Any,
    bounds: tuple[float, float, float, float, float, float],
    faces: dict[
        str, tuple[Any, tuple[float, float, float], tuple[float, float, float]]
    ],
) -> dict[str, Any]:
    return {
        "native_object_id": int(region.id),
        "native_bounding_box_um": list(bounds),
        "faces": [
            {
                "direction": direction,
                "native_face_id": int(faces[direction][0].id),
                "native_face_center_um": list(faces[direction][1]),
                "native_face_normal": list(faces[direction][2]),
            }
            for direction in _Q3D_REGION_DIRECTIONS
        ],
    }


def _q3d_region_sheet_geometry(
    direction: str, bounds: tuple[float, float, float, float, float, float]
) -> tuple[str, list[float], list[float], list[float]]:
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    if direction == "+X":
        return (
            "YZ",
            [x_max, y_min, z_min],
            [y_max - y_min, z_max - z_min],
            [x_max, y_min, z_min, x_max, y_max, z_max],
        )
    if direction == "-X":
        return (
            "YZ",
            [x_min, y_min, z_min],
            [y_max - y_min, z_max - z_min],
            [x_min, y_min, z_min, x_min, y_max, z_max],
        )
    if direction == "+Y":
        return (
            "ZX",
            [x_min, y_max, z_min],
            [z_max - z_min, x_max - x_min],
            [x_min, y_max, z_min, x_max, y_max, z_max],
        )
    if direction == "-Y":
        return (
            "ZX",
            [x_min, y_min, z_min],
            [z_max - z_min, x_max - x_min],
            [x_min, y_min, z_min, x_max, y_min, z_max],
        )
    if direction == "+Z":
        return (
            "XY",
            [x_min, y_min, z_max],
            [x_max - x_min, y_max - y_min],
            [x_min, y_min, z_max, x_max, y_max, z_max],
        )
    if direction == "-Z":
        return (
            "XY",
            [x_min, y_min, z_min],
            [x_max - x_min, y_max - y_min],
            [x_min, y_min, z_min, x_max, y_max, z_min],
        )
    raise RuntimeError("Q3D Region direction is invalid")


def _q3d_assignment(app: Any, name: str) -> list[int]:
    try:
        assignment = [
            int(value) for value in app.oboundary.GetExcitationAssignment(name)
        ]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Q3D native assignment is unavailable for {name!r}"
        ) from exc
    if not assignment:
        raise RuntimeError(f"Q3D native assignment is empty for {name!r}")
    return assignment


def _same_q3d_bounds(observed: Any, expected: list[float]) -> bool:
    try:
        values = [float(value) for value in observed]
    except (TypeError, ValueError):
        return False
    return len(values) == len(expected) and all(
        math.isclose(actual, required, rel_tol=0.0, abs_tol=1e-9)
        for actual, required in zip(values, expected, strict=True)
    )


def _seal_q3d_region(app: Any, spec: Q3dSpec) -> dict[str, Any]:
    initial_region = app.modeler.get_object_from_name("Region")
    if initial_region is None or initial_region.name != "Region":
        raise RuntimeError("Q3D Region is unavailable for grounding")
    intended_bounds = _q3d_region_bounds(initial_region)
    source_faces = _q3d_region_faces(initial_region, intended_bounds)
    source_region = _q3d_region_evidence(initial_region, intended_bounds, source_faces)
    declared_net_ids = {net.name: _q3d_assignment(app, net.name) for net in spec.nets}
    if spec.grounded_region_net in app.net_names:
        raise RuntimeError("Q3D grounded Region net already exists")
    sheet_names = [name for name, _ in _Q3D_REGION_SHEET_NAMES.values()]
    if set(sheet_names) & set(app.modeler.sheet_names):
        raise RuntimeError("Q3D Region grounding sheet name already exists")
    initial_sheets = set(app.modeler.sheet_names)
    if not app.modeler.delete("Region"):
        raise RuntimeError("Q3D initial Region could not be frozen for grounding")
    created: dict[str, tuple[Any, list[float]]] = {}
    for direction in _Q3D_REGION_DIRECTIONS:
        sheet_name, _ = _Q3D_REGION_SHEET_NAMES[direction]
        orientation, origin, sizes, expected_bounds = _q3d_region_sheet_geometry(
            direction, intended_bounds
        )
        sheet = app.modeler.create_rectangle(
            orientation=orientation,
            origin=origin,
            sizes=sizes,
            name=sheet_name,
            material="pec",
        )
        if (
            sheet is None
            or sheet.name != sheet_name
            or sheet_name not in app.modeler.sheet_names
            or not _same_q3d_bounds(sheet.bounding_box, expected_bounds)
        ):
            raise RuntimeError(
                f"Q3D Region grounding sheet creation failed for {direction}"
            )
        created[direction] = (sheet, expected_bounds)
    region = app.modeler.create_region(
        pad_value=[0.0] * 6,
        pad_type="Absolute Offset",
        name="Region",
    )
    vacuum = spec.materials[spec.vacuum_material_id]
    region.material_name = vacuum.library_name
    if (
        region.name != "Region"
        or _native_object_property(region, "Material").strip('"').casefold()
        != vacuum.library_name.casefold()
        or not _same_q3d_bounds(region.bounding_box, list(intended_bounds))
    ):
        raise RuntimeError("Q3D final grounded Region readback failed")
    bounds = _q3d_region_bounds(region)
    faces = _q3d_region_faces(region, bounds)
    records: list[dict[str, Any]] = []
    for direction in _Q3D_REGION_DIRECTIONS:
        source_face, source_center, source_normal = source_faces[direction]
        sheet_name, boundary_name = _Q3D_REGION_SHEET_NAMES[direction]
        sheet, expected_bounds = created[direction]
        boundary = app.assign_thin_conductor(
            assignment=sheet,
            material="pec",
            thickness=1,
            name=boundary_name,
        )
        if not boundary or boundary.name != boundary_name:
            raise RuntimeError(
                f"Q3D Region thin conductor assignment failed for {direction}"
            )
        sheet_id = int(sheet.id)
        if _q3d_assignment(app, boundary_name) != [sheet_id]:
            raise RuntimeError(
                f"Q3D Region thin conductor readback failed for {direction}"
            )
        records.append(
            {
                "direction": direction,
                "source_region_object_id": int(initial_region.id),
                "source_face_id": int(source_face.id),
                "source_face_center_um": list(source_center),
                "source_face_normal": list(source_normal),
                "sheet_name": sheet_name,
                "sheet_object_id": sheet_id,
                "sheet_bounding_box_um": [float(value) for value in sheet.bounding_box],
                "boundary_name": boundary_name,
                "native_thin_conductor_object_ids": [sheet_id],
            }
        )
    if set(app.modeler.sheet_names) - initial_sheets != set(sheet_names):
        raise RuntimeError("Q3D Region grounding did not create exactly six sheets")
    target = app.assign_net(sheet_names, spec.grounded_region_net, "Ground")
    if (
        target is None
        or target.name != spec.grounded_region_net
        or target.type != "GroundNet"
    ):
        raise RuntimeError("Q3D Region enclosure Ground net creation failed")
    native_target_ids = _q3d_assignment(app, spec.grounded_region_net)
    if native_target_ids != [record["sheet_object_id"] for record in records]:
        raise RuntimeError("Q3D Region grounding target net readback failed")
    if any(
        _q3d_assignment(app, name) != object_ids
        for name, object_ids in declared_net_ids.items()
    ):
        raise RuntimeError(
            "Q3D declared net assignment changed during Region grounding"
        )
    if not _same_q3d_bounds(region.bounding_box, list(bounds)):
        raise RuntimeError("Q3D Region native geometry changed during grounding")
    return {
        "target_net": spec.grounded_region_net,
        "target_net_origin": "generated_region_enclosure",
        "native_region_object_id": int(region.id),
        "native_bounding_box_um": list(bounds),
        "native_final_region_padding_um": [0.0] * 6,
        "source_region": source_region,
        "final_region": _q3d_region_evidence(region, bounds, faces),
        "sheets": records,
        "native_declared_net_object_ids": declared_net_ids,
        "native_target_net_object_ids": native_target_ids,
    }


def _read_q3d_region_ground(
    app: Any, spec: Q3dSpec, region_record: dict[str, Any]
) -> dict[str, Any]:
    evidence = region_record.get("grounded_region")
    region = app.modeler.get_object_from_name("Region")
    if not isinstance(evidence, dict) or region is None:
        raise RuntimeError("Q3D grounded Region evidence is unavailable after save")
    bounds = _q3d_region_bounds(region)
    faces = _q3d_region_faces(region, bounds)
    if (
        evidence.get("target_net") != spec.grounded_region_net
        or evidence.get("target_net_origin") != "generated_region_enclosure"
        or not _same_q3d_bounds(evidence.get("native_bounding_box_um"), list(bounds))
        or evidence.get("native_final_region_padding_um") != [0.0] * 6
        or _native_object_property(region, "Material").strip('"').casefold()
        != spec.materials[spec.vacuum_material_id].library_name.casefold()
    ):
        raise RuntimeError("Q3D grounded Region geometry changed after save")
    boundary_setup = app.design_properties.get("BoundarySetup")
    boundaries = (
        boundary_setup.get("Boundaries") if isinstance(boundary_setup, dict) else None
    )
    if not isinstance(boundaries, dict):
        raise TypeError("Q3D saved boundary records are unavailable")
    sheets = evidence.get("sheets")
    source_region = evidence.get("source_region")
    source_faces = (
        source_region.get("faces") if isinstance(source_region, dict) else None
    )
    if (
        not isinstance(sheets, list)
        or len(sheets) != len(_Q3D_REGION_DIRECTIONS)
        or not isinstance(source_faces, list)
        or len(source_faces) != len(_Q3D_REGION_DIRECTIONS)
    ):
        raise RuntimeError("Q3D grounded Region sheet evidence is invalid")
    native_thin_conductors: list[dict[str, Any]] = []
    sheet_ids: list[int] = []
    for direction, record, source in zip(
        _Q3D_REGION_DIRECTIONS, sheets, source_faces, strict=True
    ):
        if not isinstance(record, dict) or not isinstance(source, dict):
            raise TypeError("Q3D grounded Region sheet evidence is invalid")
        sheet_name, boundary_name = _Q3D_REGION_SHEET_NAMES[direction]
        sheet_id = record.get("sheet_object_id")
        _, _, _, expected_bounds = _q3d_region_sheet_geometry(direction, bounds)
        sheet = app.modeler.get_object_from_name(sheet_name)
        saved = boundaries.get(boundary_name)
        if (
            record.get("direction") != direction
            or record.get("source_region_object_id")
            != source_region.get("native_object_id")
            or record.get("source_face_id") != source.get("native_face_id")
            or record.get("source_face_center_um")
            != source.get("native_face_center_um")
            or record.get("source_face_normal") != source.get("native_face_normal")
            or record.get("sheet_name") != sheet_name
            or not isinstance(sheet_id, int)
            or not _same_q3d_bounds(
                record.get("sheet_bounding_box_um"), expected_bounds
            )
            or record.get("boundary_name") != boundary_name
            or record.get("native_thin_conductor_object_ids") != [sheet_id]
            or sheet is None
            or int(sheet.id) != sheet_id
            or not _same_q3d_bounds(sheet.bounding_box, expected_bounds)
            or not isinstance(saved, dict)
            or saved.get("BoundType") != "ThinConductor"
            or saved.get("Objects") != [sheet_id]
            or saved.get("Material") != "pec"
            or saved.get("Thickness") != "1um"
            or _q3d_assignment(app, boundary_name) != [sheet_id]
        ):
            raise RuntimeError("Q3D saved thin conductor readback is invalid")
        native_thin_conductors.append(
            {
                "name": boundary_name,
                "bound_type": "ThinConductor",
                "object_ids": [sheet_id],
                "material": "pec",
                "thickness": "1um",
            }
        )
        sheet_ids.append(sheet_id)
    target = boundaries.get(spec.grounded_region_net)
    expected_target = evidence.get("native_target_net_object_ids")
    declared_net_ids = {
        net.name: [
            int(app.modeler.get_object_from_name(name).id) for name in net.object_names
        ]
        for net in spec.nets
    }
    if (
        not isinstance(target, dict)
        or target.get("BoundType") != "GroundNet"
        or target.get("Objects") != expected_target
        or expected_target != sheet_ids
        or _q3d_assignment(app, spec.grounded_region_net) != expected_target
        or evidence.get("native_declared_net_object_ids") != declared_net_ids
        or any(
            _q3d_assignment(app, name) != object_ids
            for name, object_ids in declared_net_ids.items()
        )
    ):
        raise RuntimeError("Q3D saved grounded Region net readback is invalid")
    return {
        **evidence,
        "native_region_object_id": int(region.id),
        "native_bounding_box_um": list(bounds),
        "final_region": _q3d_region_evidence(region, bounds, faces),
        "native_saved_boundaries": {
            "target": {
                "name": spec.grounded_region_net,
                "bound_type": "GroundNet",
                "origin": "generated_region_enclosure",
                "object_ids": expected_target,
            },
            "thin_conductors": native_thin_conductors,
        },
    }


def _assign_q3d_nets(app: Any, spec: Q3dSpec) -> list[dict[str, Any]]:
    directions = {"-X": 0, "-Y": 1, "-Z": 2, "+X": 3, "+Y": 4, "+Z": 5}
    records: list[dict[str, Any]] = []
    for net in spec.nets:
        boundary = app.assign_net(list(net.object_names), net.name, net.net_type)
        if boundary is None or net.name not in app.net_names:
            raise RuntimeError(f"Q3D net assignment failed for {net.name!r}")
        expected_ids = [
            int(app.modeler.get_object_from_name(name).id) for name in net.object_names
        ]
        observed_ids = [
            int(value) for value in app.oboundary.GetExcitationAssignment(net.name)
        ]
        if observed_ids != expected_ids:
            raise RuntimeError(f"Q3D native net assignment mismatch for {net.name!r}")
        record: dict[str, Any] = {
            "name": net.name,
            "net_type": net.net_type,
            "object_names": list(net.object_names),
            "native_object_ids": observed_ids,
        }
        if net.net_type == "Signal":
            source_name = f"{net.name}Source"
            sink_name = f"{net.name}Sink"
            source_direction = directions[net.source_side]
            sink_direction = directions[net.sink_side]
            source = app.source(
                net.source_object,
                direction=source_direction,
                name=source_name,
                net_name=net.name,
            )
            sink = app.sink(
                net.sink_object,
                direction=sink_direction,
                name=sink_name,
                net_name=net.name,
            )
            if source is None or sink is None:
                raise RuntimeError(
                    f"Q3D source/sink assignment failed for {net.name!r}"
                )
            source_faces = [
                int(value)
                for value in app.oboundary.GetExcitationAssignment(source_name)
            ]
            sink_faces = [
                int(value) for value in app.oboundary.GetExcitationAssignment(sink_name)
            ]
            expected_source = int(
                app.modeler._get_faceid_on_axis(net.source_object, source_direction)
            )
            expected_sink = int(
                app.modeler._get_faceid_on_axis(net.sink_object, sink_direction)
            )
            if source_faces != [expected_source] or sink_faces != [expected_sink]:
                raise RuntimeError(
                    f"Q3D native source/sink assignment mismatch for {net.name!r}"
                )
            record["source"] = {
                "name": source_name,
                "object_name": net.source_object,
                "side": net.source_side,
                "native_face_ids": source_faces,
            }
            record["sink"] = {
                "name": sink_name,
                "object_name": net.sink_object,
                "side": net.sink_side,
                "native_face_ids": sink_faces,
            }
        records.append(record)
    if app.net_names != [net.name for net in spec.nets]:
        raise RuntimeError("Q3D native net order does not match the structured spec")
    return records


def _setup_q3d(app: Any, spec: Q3dSpec) -> None:
    if app.setup_names:
        raise RuntimeError("new Q3D design must not inherit a setup")
    setup = app.create_setup(spec.run_control.setup_name)
    setup.dc_enabled = False
    setup.ac_rl_enabled = spec.solve_ac_rl
    setup.capacitance_enabled = True
    setup.props["AdaptiveFreq"] = f"{spec.run_control.frequency_ghz:g}GHz"
    setup.props["Cap"]["MaxPass"] = spec.run_control.maximum_passes
    setup.props["Cap"]["MinPass"] = spec.run_control.minimum_passes
    setup.props["Cap"]["MinConvPass"] = spec.run_control.minimum_converged_passes
    setup.props["Cap"]["PerError"] = spec.run_control.convergence_percent
    setup.props["Cap"]["PerRefine"] = spec.run_control.percent_refinement
    if spec.solve_ac_rl:
        setup.props["AC"]["MaxPass"] = spec.run_control.maximum_passes
        setup.props["AC"]["MinPass"] = spec.run_control.minimum_passes
        setup.props["AC"]["MinConvPass"] = spec.run_control.minimum_converged_passes
        setup.props["AC"]["PerError"] = spec.run_control.convergence_percent
        setup.props["AC"]["PerRefine"] = spec.run_control.percent_refinement
    if not setup.update():
        raise RuntimeError("Q3D setup update failed")


def _read_q3d_setup(app: Any, spec: Q3dSpec) -> dict[str, Any]:
    raw = _saved_setup_properties(app, spec.run_control.setup_name)
    cap = raw.get("Cap")
    ac = raw.get("AC")
    if not isinstance(cap, dict) or (spec.solve_ac_rl != isinstance(ac, dict)):
        raise TypeError("Q3D saved setup does not match its capacitance/AC-RL mode")
    if not spec.solve_ac_rl and ("AC" in raw or "DC" in raw):
        raise RuntimeError("Q3D capacitance-only setup saved AC-RL or DC controls")
    native = {
        "adaptive_frequency": raw.get("AdaptiveFreq"),
        "capacitance_maximum_passes": cap.get("MaxPass"),
        "capacitance_minimum_passes": cap.get("MinPass"),
        "capacitance_minimum_converged_passes": cap.get("MinConvPass"),
        "capacitance_convergence_percent": cap.get("PerError"),
        "capacitance_percent_refinement": cap.get("PerRefine"),
        "dc_enabled": "DC" in raw,
    }
    if spec.solve_ac_rl:
        native.update(
            {
                "ac_rl_maximum_passes": ac.get("MaxPass"),
                "ac_rl_minimum_passes": ac.get("MinPass"),
                "ac_rl_minimum_converged_passes": ac.get("MinConvPass"),
                "ac_rl_convergence_percent": ac.get("PerError"),
                "ac_rl_percent_refinement": ac.get("PerRefine"),
            }
        )
    expected = {
        "adaptive_frequency": f"{spec.run_control.frequency_ghz:g}GHz",
        "capacitance_maximum_passes": spec.run_control.maximum_passes,
        "capacitance_minimum_passes": spec.run_control.minimum_passes,
        "capacitance_minimum_converged_passes": (
            spec.run_control.minimum_converged_passes
        ),
        "capacitance_convergence_percent": spec.run_control.convergence_percent,
        "capacitance_percent_refinement": spec.run_control.percent_refinement,
        "dc_enabled": False,
    }
    if spec.solve_ac_rl:
        expected.update(
            {
                "ac_rl_maximum_passes": spec.run_control.maximum_passes,
                "ac_rl_minimum_passes": spec.run_control.minimum_passes,
                "ac_rl_minimum_converged_passes": (
                    spec.run_control.minimum_converged_passes
                ),
                "ac_rl_convergence_percent": spec.run_control.convergence_percent,
                "ac_rl_percent_refinement": spec.run_control.percent_refinement,
            }
        )
    if native != expected:
        raise RuntimeError(f"Q3D saved setup readback mismatch: {native!r}")
    return {"name": spec.run_control.setup_name, "native": native}


def _export_q3d(
    app: Any, run_dir: Path, spec: Q3dSpec
) -> tuple[dict[str, str], dict[str, Any]]:
    output_dir = run_dir / "results" / "q3d"
    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    summaries: dict[str, Any] = {}
    frequency_hz = spec.run_control.frequency_ghz * 1e9
    exports = (("C", "c"), ("AC RL", "ac_rl")) if spec.solve_ac_rl else (("C", "c"),)
    normalized: list[dict[str, Any]] = []
    for problem, stem in exports:
        path = output_dir / f"{stem}_matrix.csv"
        app.odesign.ExportMatrixData(
            str(path),
            problem,
            "",
            f"{spec.run_control.setup_name} : LastAdaptive",
            "Original",
            "ohm",
            "nH",
            "pF",
            "mho",
            frequency_hz,
            "Maxwell",
            0,
            False,
            15,
            20,
            1,
        )
        titles = {
            "C": {"Capacitance Matrix": "C", "Conductance Matrix": "G"},
            "AC RL": {"AC Inductance Matrix": "L", "AC Resistance Matrix": "R"},
        }[problem]
        rows, summary = parse_matrix_export(
            path, "Q3D", problem, spec.run_control.frequency_ghz, titles
        )
        if spec.solve_ac_rl:
            normalized.extend(rows)
        summaries[stem] = summary
        hashes[path.relative_to(run_dir).as_posix()] = file_sha256(path)
    if not spec.solve_ac_rl:
        path = output_dir / "c_matrix.csv"
        return hashes, {
            "matrices": {
                "path": path.relative_to(run_dir).as_posix(),
                "frequency_ghz": spec.run_control.frequency_ghz,
                "native": summaries["c"],
                "primary_rows": len(rows),
            }
        }
    normalized_path = output_dir / "matrices.csv"
    write_csv(
        normalized_path,
        normalized,
        fieldnames=["problem_type", "quantity", "row", "column", "value", "unit"],
    )
    hashes[normalized_path.relative_to(run_dir).as_posix()] = file_sha256(
        normalized_path
    )
    return hashes, {
        "matrices": {
            "frequency_ghz": spec.run_control.frequency_ghz,
            "native": summaries,
            "normalized_rows": len(normalized),
        }
    }


__all__ = ["PreparedQ3d", "export_q3d", "prepare_q3d", "run_q3d", "solve_q3d"]
