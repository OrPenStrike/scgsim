"""Explicit local HFSS execution for one prepared two-port CPW handoff."""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from itertools import pairwise
from pathlib import Path
from typing import Any

from .spec import (
    LOCKED_PYAEDT,
    POINT_COUNT,
    REQUIRED_AEDT_VERSION,
    SURFACE_APPROXIMATION_LEVEL,
    HfssDrivenSpec,
    HfssEigenmodeSpec,
    HfssSpec,
    ModalPort,
    TerminalPort,
    parse_hfss_spec,
)
from .util import file_sha256, read_json, write_csv, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one prepared SCGSim HFSS CPW handoff"
    )
    parser.add_argument("--handoff", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Open local AEDT and solve the prepared handoff",
    )
    args = parser.parse_args(argv)
    metadata_path = Path(args.handoff).resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"handoff metadata is missing: {metadata_path}")
    if not args.execute:
        raise RuntimeError(
            "prepared handoff is not executed; use run_hfss.sh or pass --execute explicitly"
        )
    return _execute(metadata_path)


def _execute(metadata_path: Path) -> int:
    run_dir = metadata_path.parent.parent
    os.chdir(run_dir)
    metadata = _object(read_json(metadata_path), "handoff metadata")
    files = _canonical_metadata_files(metadata, metadata_path, run_dir)
    receipt_path = run_dir / files["receipt"]
    receipt = _object(read_json(receipt_path), "receipt")
    if receipt.get("schema_version") != "scgsim.aedt.receipt.v1":
        raise RuntimeError("handoff receipt schema is invalid")
    if receipt.get("status") != "not_run":
        raise RuntimeError("one-shot handoff is not in not_run state")
    spec_path = run_dir / files["spec"]
    spec = parse_hfss_spec(_object(read_json(spec_path), "spec"), base_dir=run_dir)
    if spec.gds_path.resolve() != (run_dir / "geometry" / "design.gds").resolve():
        raise RuntimeError("prepared spec must use geometry/design.gds")
    _require_pristine_run(run_dir, spec)
    started = _utc_now()
    execution_started = time.perf_counter()
    receipt.update(
        {
            "status": "running",
            "started_at_utc": started,
            "save": {"ok": False},
            "release": {"ok": False},
        }
    )
    write_json(receipt_path, receipt)

    desktop: Any | None = None
    status = "failed"
    failure: str | None = None
    result: dict[str, Any] | None = None
    try:
        receipt["mode"] = spec.mode
        _verify_prepared_hashes(metadata, spec_path, spec.gds_path)
        if _pyaedt_version() != LOCKED_PYAEDT:
            raise RuntimeError("PyAEDT lock mismatch")
        from ansys.aedt.core import Desktop, Hfss

        desktop = Desktop(
            version=REQUIRED_AEDT_VERSION,
            non_graphical=True,
            new_desktop=True,
            close_on_exit=False,
        )
        if desktop.aedt_version_id != REQUIRED_AEDT_VERSION:
            raise RuntimeError(f"AEDT version mismatch: {desktop.aedt_version_id!r}")
        receipt["runtime_source"] = _runtime_source_identity()
        result = _solve(Hfss, run_dir, spec)
        status = "completed"
    except Exception as exc:  # noqa: BLE001 -- receipt must record any solver failure.
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if result is not None:
            receipt["save"] = result["save"]
        if desktop is not None:
            try:
                released = bool(
                    desktop.release_desktop(close_projects=True, close_on_exit=True)
                )
                receipt["release"] = {"ok": released}
                if not released:
                    raise RuntimeError("owned AEDT Desktop release returned false")
            except Exception as exc:  # noqa: BLE001 -- release failure is receipt-authoritative.
                receipt["release"] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if failure is None:
                    failure = receipt["release"]["error"]
        if failure is not None:
            status = "failed"
            receipt["error"] = failure
        if result is not None:
            receipt["outputs"] = result["outputs"]
            receipt["connected"] = result["connected"]
            receipt["project"] = result["project"]
            receipt["ports"] = result["ports"]
            receipt["mesh"] = result["mesh"]
            receipt["materials"] = result["materials"]
            receipt["region"] = result["region"]
            receipt["result_readback"] = result["result_readback"]
        receipt["diagnostics"] = _read_physics_warnings(run_dir)
        receipt["status"] = status
        receipt["finished_at_utc"] = _utc_now()
        receipt["execution_seconds"] = round(time.perf_counter() - execution_started, 6)
        write_json(receipt_path, receipt)
    return 0 if status == "completed" else 1


def _solve(Hfss: Any, run_dir: Path, spec: HfssSpec) -> dict[str, Any]:
    project_path = run_dir / f"{spec.project_name}.aedt"
    hfss = Hfss(
        project=str(project_path),
        design=spec.design_name,
        solution_type={
            "terminal": "DrivenTerminal",
            "modal": "DrivenModal",
            "eigenmode": "Eigenmode",
        }[spec.mode],
        new_desktop=False,
        close_on_exit=False,
    )
    if hfss.desktop_class.aedt_version_id != REQUIRED_AEDT_VERSION:
        raise RuntimeError("HFSS did not bind the owned AEDT 2024.2 desktop")
    hfss.modeler.model_units = "um"
    materials = _import_and_bind(hfss, spec)
    region = _create_region(hfss, spec)
    mesh = _assign_mesh(hfss, spec)
    ports = _assign_ports(hfss, spec)
    _setup(hfss, spec)
    if not hfss.save_project() or not project_path.is_file():
        raise RuntimeError("HFSS project was not saved before native port readback")
    ports = _bind_port_evidence(hfss, spec, ports)
    if not hfss.analyze_setup(name=spec.run_control.setup_name, blocking=True):
        raise RuntimeError(
            f"HFSS failed to analyze setup {spec.run_control.setup_name!r}"
        )
    outputs, result_readback = _export(hfss, run_dir, spec, ports)
    saved = bool(hfss.save_project())
    if not saved or not project_path.is_file():
        raise RuntimeError("HFSS project was not saved")
    outputs[project_path.relative_to(run_dir).as_posix()] = file_sha256(project_path)
    return {
        "outputs": outputs,
        "connected": {
            "aedt_version": hfss.desktop_class.aedt_version_id,
            "pyaedt_version": _pyaedt_version(),
        },
        "project": project_path.relative_to(run_dir).as_posix(),
        "ports": ports,
        "mesh": mesh,
        "materials": materials,
        "region": region,
        "result_readback": result_readback,
        "save": {
            "ok": True,
            "project_sha256": outputs[project_path.relative_to(run_dir).as_posix()],
        },
    }


def _import_and_bind(hfss: Any, spec: HfssSpec) -> list[dict[str, Any]]:
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
            pec.append(binding.object_name)
            record["requested_pec_boundary"] = "SCGSimPEC"
        else:
            existing = hfss.materials.exists_material(material.library_name)
            if not existing:
                raise RuntimeError(
                    f"AEDT library material is unavailable: {material.library_name!r}"
                )
            obj.material_name = material.library_name
            observed_material = _native_object_property(obj, "Material").strip('"')
            if observed_material.casefold() != material.library_name.casefold():
                raise RuntimeError(
                    f"material readback mismatch for {binding.object_name!r}"
                )
            record["observed"] = {"native_material_name": observed_material}
        observed.append(record)
    if pec:
        boundary = hfss.assign_perfect_e(pec, name="SCGSimPEC")
        if boundary is None or "SCGSimPEC" not in _native_boundary_names(hfss):
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


def _create_region(hfss: Any, spec: HfssSpec) -> dict[str, Any]:
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
    observed_material = _native_object_property(region, "Material").strip('"')
    if (
        region.name != "Region"
        or observed_material.casefold() != vacuum.library_name.casefold()
    ):
        raise RuntimeError("vacuum region readback failed")
    return {
        "material_id": vacuum.material_id,
        "requested_library_name": vacuum.library_name,
        "observed_material_name": observed_material,
        "padding_um": list(spec.region_padding_um),
    }


def _assign_mesh(hfss: Any, spec: HfssSpec) -> dict[str, Any]:
    grounds = [
        item.object_name for item in spec.object_bindings if item.role == "ground"
    ]
    signals = [
        item.object_name for item in spec.object_bindings if item.role == "signal"
    ]
    result: dict[str, Any] = {"surface": {}, "length": {}}
    for role, objects, name in (
        ("signal", signals, "SignalSurfaceApprox9"),
        ("ground", grounds, "GroundSurfaceApprox9"),
    ):
        operation = hfss.mesh.assign_surface_mesh(
            objects, SURFACE_APPROXIMATION_LEVEL, name
        )
        if operation is None or name not in _native_mesh_operation_names(hfss):
            raise RuntimeError(f"{role} surface mesh readback failed")
        native = _native_mesh_readback(hfss, name, objects)
        _require_native_mesh_properties(
            native,
            {
                "Name": name,
                "Type": "Surface Approximation Based",
                "Region": "On Selection",
                "Curved Mesh Approximation Type": "Use Slider",
                "Curved Surface Mesh Resolution": SURFACE_APPROXIMATION_LEVEL,
            },
            f"{role} surface mesh",
        )
        result["surface"][role] = {
            "requested": {
                "name": name,
                "objects": objects,
                "level": SURFACE_APPROXIMATION_LEVEL,
            },
            "native": native,
        }
    if spec.length_mesh is not None:
        for role, objects, name in (
            ("signal", list(spec.length_mesh.signal_objects), "UniformCpwSignalLength"),
            ("ground", list(spec.length_mesh.ground_objects), "UniformCpwGroundLength"),
        ):
            operation = hfss.mesh.assign_length_mesh(
                objects,
                inside_selection=False,
                maximum_length=f"{spec.length_mesh.maximum_length_um:g}um",
                maximum_elements=1_000_000,
                name=name,
            )
            requested = {
                "Objects": objects,
                "MaxLength": f"{spec.length_mesh.maximum_length_um:g}um",
                "NumMaxElem": "1000000",
                "RestrictElem": True,
                "RestrictLength": True,
                "RefineInside": False,
                "Enabled": True,
            }
            if operation is None or name not in _native_mesh_operation_names(hfss):
                raise RuntimeError(f"uniform CPW {role} length mesh readback failed")
            native = _native_mesh_readback(hfss, name, objects)
            _require_native_mesh_properties(
                native,
                {
                    "Name": name,
                    "Type": "Length Based",
                    "Region": "On Selection",
                    "Enabled": True,
                    "Restrict Length": True,
                    "Max Length": f"{spec.length_mesh.maximum_length_um:g}um",
                    "Restrict Max Elems": True,
                    "Max Elems": 1_000_000,
                },
                f"uniform CPW {role} length mesh",
            )
            result["length"][role] = {
                "requested": {"name": name, "properties": requested},
                "native": native,
            }
    return result


def _assign_ports(hfss: Any, spec: HfssSpec) -> list[dict[str, Any]]:
    if isinstance(spec, HfssEigenmodeSpec):
        if hfss.get_oo_name(hfss.odesign, "Excitations"):
            raise RuntimeError("HFSS Eigenmode design must not contain excitations")
        return []
    region = hfss.modeler.get_object_from_name("Region")
    if region is None:
        raise RuntimeError("Region is missing")
    faces = [
        (int(face.id), tuple(float(value) for value in face.center))
        for face in region.faces
    ]
    centers = {face_id: center for face_id, center in faces}
    records: list[dict[str, Any]] = []
    for port in spec.ports:
        face_id = _face_for_side(faces, port.side)
        if isinstance(port, TerminalPort):
            before = set(hfss.oboundary.GetExcitationsOfType("Terminal"))
            boundary = hfss.wave_port(
                face_id,
                reference=list(port.reference_objects),
                name=port.name,
                renormalize=False,
                deembed=f"{port.deembed_um:g}um",
                terminals_rename=False,
            )
            after = set(hfss.oboundary.GetExcitationsOfType("Terminal"))
            names = sorted(after - before)
            if boundary is None or len(names) != 1:
                raise RuntimeError(
                    f"terminal excitation readback failed for {port.name!r}: {names!r}"
                )
            boundary_name = _text(getattr(boundary, "name", port.name), "boundary.name")
            native = _native_terminal_readback(hfss, boundary_name, names[0], port)
            records.append(
                {
                    "index": port.index,
                    "boundary": boundary_name,
                    "terminal_excitation": names[0],
                    "face_id": face_id,
                    "face_center_um": list(centers[face_id]),
                    "native_terminal_names": names,
                    "requested": {
                        "reference_objects": list(port.reference_objects),
                        "renormalize": False,
                        "deembed_um": port.deembed_um,
                    },
                    "native": native,
                }
            )
        elif isinstance(port, ModalPort):
            before = set(hfss.get_oo_name(hfss.odesign, "Excitations"))
            boundary = hfss.wave_port(
                face_id,
                integration_line=[list(point) for point in port.integration_line_um],
                modes=1,
                impedance=50,
                name=port.name,
                renormalize=False,
                deembed=0,
                characteristic_impedance="Zpi",
            )
            after = set(hfss.get_oo_name(hfss.odesign, "Excitations"))
            names = sorted(after - before)
            if boundary is None or names != [port.name]:
                raise RuntimeError(
                    f"modal excitation readback failed for {port.name!r}: {names!r}"
                )
            native = _native_modal_oo_readback(hfss, port.name)
            records.append(
                {
                    "index": port.index,
                    "boundary": port.name,
                    "modal_excitation": port.name,
                    "face_id": face_id,
                    "face_center_um": list(centers[face_id]),
                    "requested": {
                        "integration_line_um": [
                            list(point) for point in port.integration_line_um
                        ],
                        "modes": 1,
                        "renormalize": False,
                        "deembed_um": 0.0,
                        "characteristic_impedance": "Zpi",
                    },
                    "native": native,
                }
            )
        else:  # pragma: no cover - HfssDrivenSpec already closes this boundary.
            raise TypeError("unsupported HFSS driven port type")
    return records


def _face_for_side(
    faces: list[tuple[int, tuple[float, float, float]]], side: str
) -> int:
    axis, sign = {
        "-X": (0, -1),
        "+X": (0, 1),
        "-Y": (1, -1),
        "+Y": (1, 1),
        "-Z": (2, -1),
        "+Z": (2, 1),
    }[side]
    edge = (
        max(center[axis] for _, center in faces)
        if sign > 0
        else min(center[axis] for _, center in faces)
    )
    matches = [face_id for face_id, center in faces if abs(center[axis] - edge) <= 1e-9]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one Region face on {side}, got {matches!r}"
        )
    return matches[0]


def _setup(hfss: Any, spec: HfssSpec) -> None:
    if hfss.setup_names:
        raise RuntimeError("new V1 design must not inherit a setup")
    setup = hfss.create_setup(spec.run_control.setup_name)
    if isinstance(spec, HfssEigenmodeSpec):
        setup.props["MinimumFrequency"] = (
            f"{spec.run_control.minimum_frequency_ghz:g}GHz"
        )
        setup.props["NumModes"] = spec.run_control.num_modes
        setup.props["MaxDeltaFreq"] = spec.run_control.maximum_delta_frequency_percent
        setup.props["MaximumPasses"] = spec.run_control.maximum_passes
        if not setup.update():
            raise RuntimeError("HFSS Eigenmode setup update failed")
        observed = {
            key: setup.props.get(key)
            for key in (
                "MinimumFrequency",
                "NumModes",
                "MaxDeltaFreq",
                "MaximumPasses",
            )
        }
        expected = {
            "MinimumFrequency": f"{spec.run_control.minimum_frequency_ghz:g}GHz",
            "NumModes": spec.run_control.num_modes,
            "MaxDeltaFreq": spec.run_control.maximum_delta_frequency_percent,
            "MaximumPasses": spec.run_control.maximum_passes,
        }
        if observed != expected:
            raise RuntimeError(f"HFSS Eigenmode setup readback mismatch: {observed!r}")
        return
    setup.props["SolveType"] = (
        "DrivenTerminal" if spec.mode == "terminal" else "DrivenModal"
    )
    if not setup.enable_adaptive_setup_broadband(
        f"{spec.run_control.sweep.start_ghz}GHz",
        f"{spec.run_control.sweep.stop_ghz}GHz",
        max_passes=99,
        max_delta_s=0.02,
    ):
        raise RuntimeError("adaptive setup initialization failed")
    sweep = hfss.create_linear_count_sweep(
        spec.run_control.setup_name,
        "GHz",
        spec.run_control.sweep.start_ghz,
        spec.run_control.sweep.stop_ghz,
        num_of_freq_points=POINT_COUNT,
        name=spec.run_control.sweep_name,
        save_fields=False,
        sweep_type="Fast",
    )
    if sweep is None:
        raise RuntimeError("Fast sweep creation failed")
    actual = (
        str(sweep.props.get("Type")),
        str(sweep.props.get("RangeStart")),
        str(sweep.props.get("RangeEnd")),
        int(sweep.props.get("RangeCount")),
    )
    expected = (
        "Fast",
        f"{spec.run_control.sweep.start_ghz}GHz",
        f"{spec.run_control.sweep.stop_ghz}GHz",
        POINT_COUNT,
    )
    if actual != expected:
        raise RuntimeError(f"Fast sweep readback mismatch: {actual!r} != {expected!r}")


def _export(
    hfss: Any, run_dir: Path, spec: HfssSpec, ports: list[dict[str, Any]]
) -> tuple[dict[str, str], dict[str, Any]]:
    output_dir = run_dir / "results" / spec.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(spec, HfssEigenmodeSpec):
        return _export_eigenmode(hfss, run_dir, output_dir, spec)
    setup_sweep = f"{spec.run_control.setup_name} : {spec.run_control.sweep_name}"
    terminal = spec.mode == "terminal"
    names = [
        record["terminal_excitation" if terminal else "modal_excitation"]
        for record in ports
    ]
    hashes: dict[str, str] = {}
    readback: dict[str, Any] = {}
    touchstone = output_dir / f"{spec.mode}.s2p"
    if (
        not hfss.export_touchstone(
            setup=spec.run_control.setup_name,
            sweep=spec.run_control.sweep_name,
            output_file=str(touchstone),
        )
        or not touchstone.is_file()
    ):
        raise RuntimeError(f"{spec.mode} Touchstone export failed")
    readback["touchstone"] = _verify_touchstone(touchstone, spec, names)
    prefix = "St" if terminal else "S"
    expressions = [f"{prefix}({left},{right})" for left in names for right in names]
    data = hfss.post.get_solution_data(
        expressions=expressions,
        setup_sweep_name=setup_sweep,
        report_category=(
            "Terminal Solution Data" if terminal else "Modal Solution Data"
        ),
    )
    csv_path = output_dir / f"{spec.mode}_{prefix.casefold()}.csv"
    readback[f"{spec.mode}_{prefix.casefold()}"] = _write_complex_csv(
        data, expressions, csv_path, suffix="", spec=spec
    )
    hashes[touchstone.relative_to(run_dir).as_posix()] = file_sha256(touchstone)
    hashes[csv_path.relative_to(run_dir).as_posix()] = file_sha256(csv_path)
    return hashes, readback


def _export_eigenmode(
    hfss: Any,
    run_dir: Path,
    output_dir: Path,
    spec: HfssEigenmodeSpec,
) -> tuple[dict[str, str], dict[str, Any]]:
    setup = f"{spec.run_control.setup_name} : LastAdaptive"
    raw = output_dir / "eigenmodes.eig"
    hfss.osolution.ExportEigenmodes(setup, "", str(raw))
    if not raw.is_file() or raw.stat().st_size == 0:
        raise RuntimeError("HFSS Eigenmode native export is missing or empty")
    rows = _parse_eigenmode_export(raw, spec)
    path = output_dir / "eigenmodes.csv"
    write_csv(path, rows, fieldnames=["mode", "frequency_ghz", "q_factor"])
    raw_relative = raw.relative_to(run_dir).as_posix()
    path_relative = path.relative_to(run_dir).as_posix()
    return {
        raw_relative: file_sha256(raw),
        path_relative: file_sha256(path),
    }, {
        "eigenmodes": {
            "modes": len(rows),
            "frequency_unit": "GHz",
            "mode_indices": [row["mode"] for row in rows],
            "frequencies_ghz": [row["frequency_ghz"] for row in rows],
            "q_factors": [row["q_factor"] for row in rows],
            "native_export": raw.name,
            "native_export_bytes": raw.stat().st_size,
        }
    }


def _parse_eigenmode_export(
    path: Path, spec: HfssEigenmodeSpec
) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if (
        not lines
        or lines[0] != "# Ansys eigenmode data file.  Version 2.0"
        or f"# Design:     {spec.design_name}" not in lines
        or f"# Solution:   {spec.run_control.setup_name} : LastAdaptive" not in lines
        or not any(
            "Mode" in line and "Frequency (GHz)" in line and "Q" in line
            for line in lines
        )
    ):
        raise RuntimeError("HFSS Eigenmode native export header is invalid")
    rows: list[dict[str, Any]] = []
    for line in lines:
        tokens = line.split()
        if len(tokens) != 3 or not tokens[0].isdigit():
            continue
        mode = int(tokens[0])
        frequency = float(tokens[1])
        q_factor = float(tokens[2])
        if (
            not math.isfinite(frequency)
            or frequency <= 0
            or not math.isfinite(q_factor)
            or q_factor < 0
        ):
            raise RuntimeError("HFSS Eigenmode native result is invalid")
        rows.append({"mode": mode, "frequency_ghz": frequency, "q_factor": q_factor})
    if [row["mode"] for row in rows] != list(range(1, spec.run_control.num_modes + 1)):
        raise RuntimeError("HFSS Eigenmode native mode count/order is invalid")
    return rows


def _write_complex_csv(
    data: Any, expressions: list[str], path: Path, *, suffix: str, spec: HfssDrivenSpec
) -> dict[str, Any]:
    if not data:
        raise RuntimeError("solution-data extraction failed")
    frequency, _ = data.get_expression_data(expressions[0], "real")
    if len(frequency) != POINT_COUNT:
        raise RuntimeError(
            f"extracted point count must be {POINT_COUNT}, got {len(frequency)}"
        )
    unit = data.units_sweeps.get(data.primary_sweep)
    values = [float(value) for value in frequency]
    if (
        data.primary_sweep != "Freq"
        or unit != "GHz"
        or not math.isclose(values[0], spec.run_control.sweep.start_ghz, abs_tol=1e-9)
        or not math.isclose(values[-1], spec.run_control.sweep.stop_ghz, abs_tol=1e-9)
        or any(right <= left for left, right in pairwise(values))
    ):
        raise RuntimeError(
            "solution frequency records have invalid units, endpoints, or ordering"
        )
    columns: dict[str, tuple[Any, Any]] = {
        expression: (
            data.get_expression_data(expression, "real")[1],
            data.get_expression_data(expression, "imag")[1],
        )
        for expression in expressions
    }
    if any(
        len(real) != POINT_COUNT or len(imag) != POINT_COUNT
        for real, imag in columns.values()
    ):
        raise RuntimeError("solution-data expression length mismatch")
    names = [
        "frequency_ghz",
        *[
            name
            for expression in expressions
            for name in (f"Re({expression}){suffix}", f"Im({expression}){suffix}")
        ],
    ]
    rows = []
    for index in range(POINT_COUNT):
        row: dict[str, Any] = {"frequency_ghz": frequency[index]}
        for expression, (real, imag) in columns.items():
            row[f"Re({expression}){suffix}"] = real[index]
            row[f"Im({expression}){suffix}"] = imag[index]
        rows.append(row)
    write_csv(path, rows, fieldnames=names)
    return {
        "records": len(values),
        "frequency_unit": unit,
        "first_frequency_ghz": values[0],
        "last_frequency_ghz": values[-1],
        "strictly_increasing": True,
    }


def _verify_touchstone(
    path: Path, spec: HfssDrivenSpec, native_port_names: list[str]
) -> dict[str, Any]:
    if path.suffix.lower() != ".s2p" or path.stat().st_size == 0:
        raise RuntimeError("Touchstone must be a nonempty .s2p file")
    if len(native_port_names) != 2 or len(set(native_port_names)) != 2:
        raise RuntimeError("Touchstone requires two ordered native port names")
    unit: str | None = None
    frequencies: list[float] = []
    header_ports: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("!"):
            match = re.fullmatch(r"!\s*Port\[(1|2)]\s*=\s*(.+?)\s*", stripped)
            if match:
                index, name = int(match.group(1)), match.group(2)
                if index in header_ports:
                    raise RuntimeError("Touchstone repeats an indexed port-name header")
                header_ports[index] = name
            continue
        if stripped.startswith("#"):
            tokens = stripped[1:].split()
            if not tokens or tokens[0].casefold() != "ghz":
                raise RuntimeError("Touchstone frequency unit must be GHz")
            unit = "GHz"
            continue
        if stripped.startswith("["):
            continue
        tokens = stripped.split()
        try:
            frequency = float(tokens[0])
        except (IndexError, ValueError):
            continue
        if len(tokens) != 9:
            raise RuntimeError("Touchstone is not a two-port data record")
        frequencies.append(frequency)
    if (
        unit != "GHz"
        or len(frequencies) != POINT_COUNT
        or not math.isclose(
            frequencies[0], spec.run_control.sweep.start_ghz, abs_tol=1e-9
        )
        or not math.isclose(
            frequencies[-1], spec.run_control.sweep.stop_ghz, abs_tol=1e-9
        )
        or any(right <= left for left, right in pairwise(frequencies))
    ):
        raise RuntimeError(
            "Touchstone records have invalid count, units, endpoints, or ordering"
        )
    if [header_ports.get(1), header_ports.get(2)] != native_port_names:
        raise RuntimeError(
            "Touchstone indexed port-name order does not match native ports"
        )
    return {
        "path": path.name,
        "ports": 2,
        "records": len(frequencies),
        "frequency_unit": unit,
        "first_frequency_ghz": frequencies[0],
        "last_frequency_ghz": frequencies[-1],
        "strictly_increasing": True,
        "port_order": native_port_names,
        "bytes": path.stat().st_size,
    }


def _canonical_metadata_files(
    metadata: dict[str, Any], metadata_path: Path, run_dir: Path
) -> dict[str, str]:
    expected_path = run_dir / "metadata" / "hfss_handoff_metadata.json"
    if metadata_path != expected_path:
        raise RuntimeError("handoff metadata path is not canonical")
    if (
        metadata.get("schema_version") != "scgsim.aedt.handoff.v1"
        or metadata.get("status") != "prepared"
    ):
        raise RuntimeError("handoff metadata schema or status is invalid")
    files = _object(metadata.get("files"), "files")
    expected = {
        "spec": "hfss_driven_spec.json",
        "gds": "geometry/design.gds",
        "receipt": "metadata/hfss_run_receipt.json",
    }
    if files != expected:
        raise RuntimeError("handoff metadata file map is not canonical")
    return expected


def _verify_prepared_hashes(
    metadata: dict[str, Any], spec_path: Path, gds_path: Path
) -> None:
    if file_sha256(gds_path) != _text(metadata.get("gds_sha256"), "gds_sha256"):
        raise RuntimeError("copied GDS hash mismatch")
    receipt = _object(
        read_json(spec_path.parent / "metadata" / "hfss_run_receipt.json"), "receipt"
    )
    source = _object(receipt.get("source"), "receipt.source")
    if (
        source.get("spec") != "hfss_driven_spec.json"
        or source.get("gds") != "geometry/design.gds"
    ):
        raise RuntimeError("receipt source paths are not canonical")
    if file_sha256(spec_path) != _text(source.get("spec_sha256"), "source.spec_sha256"):
        raise RuntimeError("prepared spec hash mismatch")
    if file_sha256(gds_path) != _text(source.get("gds_sha256"), "source.gds_sha256"):
        raise RuntimeError("receipt GDS hash mismatch")


def _require_pristine_run(run_dir: Path, spec: HfssDrivenSpec) -> None:
    project_path = run_dir / f"{spec.project_name}.aedt"
    if project_path.exists() or (run_dir / "results").exists():
        raise RuntimeError("one-shot handoff already owns project or results artifacts")


def _pyaedt_version() -> str:
    try:
        installed = version("pyaedt")
    except PackageNotFoundError as exc:
        raise RuntimeError("PyAEDT is not installed") from exc
    if installed != LOCKED_PYAEDT:
        raise RuntimeError(
            f"PyAEDT version mismatch: expected {LOCKED_PYAEDT}, got {installed}"
        )
    return installed


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be a JSON object")
    return value


def _native_object_property(obj: Any, property_name: str) -> str:
    # Direct AEDT property query avoids the mutable PyAEDT object-property cache.
    value = obj._oeditor.GetPropertyValue(
        "Geometry3DAttributeTab", obj.name, property_name
    )
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"AEDT native object property {property_name!r} is unavailable for {obj.name!r}"
        )
    return value


def _native_boundary_names(hfss: Any) -> list[str]:
    if "GetBoundaries" not in hfss.oboundary.__dir__():
        raise RuntimeError("AEDT native boundary collection is unavailable")
    return list(hfss.oboundary.GetBoundaries())


def _native_mesh_operation_names(hfss: Any) -> list[str]:
    names = hfss.get_oo_name(hfss.odesign, "Mesh")
    if not isinstance(names, list):
        raise TypeError("AEDT native mesh operation collection is unavailable")
    return names


def _native_mesh_readback(
    hfss: Any, expected_name: str, expected_objects: list[str]
) -> dict[str, Any]:
    """Read AEDT OOP mesh properties and MeshSetup assignments directly."""
    operation_names = _native_mesh_operation_names(hfss)
    if expected_name not in operation_names:
        raise RuntimeError(f"mesh operation is missing natively: {expected_name}")
    property_names = hfss.get_oo_properties(hfss.odesign, f"Mesh/{expected_name}")
    if not property_names:
        raise RuntimeError(
            f"mesh operation native properties are unavailable: {expected_name}"
        )
    properties = {
        property_name: hfss.get_oo_property_value(
            hfss.odesign, f"Mesh/{expected_name}", property_name
        )
        for property_name in property_names
    }
    assigned_ids = [
        int(object_id)
        for object_id in hfss.mesh.omeshmodule.GetMeshOpAssignment(expected_name)
    ]
    objects = [hfss.oeditor.GetObjectNameByID(object_id) for object_id in assigned_ids]
    if objects != expected_objects:
        raise RuntimeError(
            f"mesh operation native assignment mismatch: {expected_name}"
        )
    return {
        "operation_names": operation_names,
        "properties": properties,
        "object_ids": assigned_ids,
        "objects": objects,
    }


def _require_native_mesh_properties(
    native: dict[str, Any], expected: dict[str, Any], context: str
) -> None:
    """Compare required settings while retaining AEDT's raw native properties."""
    observed = native["properties"]
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if key not in observed or not _native_value_matches(observed[key], value)
    }
    if mismatches:
        raise RuntimeError(f"{context} native property mismatch: {mismatches!r}")


def _native_value_matches(observed: Any, expected: Any) -> bool:
    """Accept AEDT native scalar spelling without converting its recorded value."""
    if isinstance(expected, bool):
        return observed is expected or observed == str(expected).lower()
    if isinstance(expected, int):
        return observed == expected or observed == str(expected)
    return observed == expected


def _native_terminal_readback(
    hfss: Any, boundary_name: str, terminal_name: str, port: TerminalPort
) -> dict[str, Any]:
    """Read port and terminal state from AEDT's Excitations object tree."""
    excitation_names = hfss.get_oo_name(hfss.odesign, "Excitations")
    if boundary_name not in excitation_names:
        raise RuntimeError(f"AEDT terminal boundary is missing: {boundary_name!r}")
    terminal_names = hfss.get_oo_name(hfss.odesign, f"Excitations\\{boundary_name}")
    if terminal_names != [terminal_name]:
        raise RuntimeError(
            f"AEDT terminal child mismatch for {boundary_name!r}: {terminal_names!r}"
        )
    boundary_properties = _native_oo_properties(hfss, f"Excitations\\{boundary_name}")
    terminal_properties = _native_oo_properties(
        hfss, f"Excitations\\{boundary_name}\\{terminal_name}"
    )
    _require_native_properties(
        boundary_properties,
        {
            "Name": boundary_name,
            "Type": "Wave Port",
            "Wave Port Type": "Terminal",
            "Num Terminals": 1,
            "Deembed": True,
            "Deembed Dist": f"{port.deembed_um:g}um",
            "Renorm All Terminals": False,
        },
        f"terminal boundary {boundary_name!r}",
    )
    _require_native_properties(
        terminal_properties,
        {"Name": terminal_name, "Port Name": boundary_name, "Type": "Terminal"},
        f"terminal child {terminal_name!r}",
    )
    return {
        "excitation_names": excitation_names,
        "terminal_names": terminal_names,
        "boundary_properties": boundary_properties,
        "terminal_properties": terminal_properties,
    }


def _native_modal_oo_readback(hfss: Any, boundary_name: str) -> dict[str, Any]:
    """Read the direct AEDT object-tree state of one modal wave port."""
    excitation_names = hfss.get_oo_name(hfss.odesign, "Excitations")
    if boundary_name not in excitation_names:
        raise RuntimeError(f"AEDT modal boundary is missing: {boundary_name!r}")
    properties = _native_oo_properties(hfss, f"Excitations\\{boundary_name}")
    _require_native_properties(
        properties,
        {
            "Name": boundary_name,
            "Type": "Wave Port",
            "Num Modes": 1,
            "Deembed": False,
            "Renorm All Modes": False,
        },
        f"modal boundary {boundary_name!r}",
    )
    return {
        "excitation_names": excitation_names,
        "boundary_properties": properties,
    }


def _bind_port_evidence(
    hfss: Any, spec: HfssSpec, ports: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if isinstance(spec, HfssEigenmodeSpec):
        if ports:
            raise RuntimeError("HFSS Eigenmode must not bind port evidence")
        return ports
    if spec.mode == "terminal":
        return _bind_terminal_reference_evidence(hfss, spec, ports)
    return _bind_modal_evidence(hfss, spec, ports)


def _bind_modal_evidence(
    hfss: Any, spec: HfssDrivenSpec, ports: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Bind modal integration lines from AEDT's saved native design state."""
    try:
        boundaries = hfss.design_properties["BoundarySetup"]["Boundaries"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("AEDT native modal boundaries are unavailable") from exc
    if not isinstance(boundaries, dict):
        raise TypeError("AEDT native modal boundary data is invalid")
    for record, port in zip(ports, spec.ports, strict=True):
        if not isinstance(port, ModalPort):
            raise TypeError("modal evidence requires ModalPort entries")
        boundary = boundaries.get(port.name)
        try:
            mode = boundary["Modes"]["Mode1"]
            positions = mode["IntLine"]["GeometryPosition"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"AEDT native modal integration line is unavailable: {port.name!r}"
            ) from exc
        observed = [
            [float(item[f"{axis}Position"]) for axis in "XYZ"] for item in positions
        ]
        expected = [list(point) for point in port.integration_line_um]
        if (
            boundary.get("BoundType") != "Wave Port"
            or boundary.get("WavePortType") != "Modal"
            or boundary.get("NumModes") != 1
            or boundary.get("Faces") != [record["face_id"]]
            or boundary.get("DoDeembed") is not False
            or mode.get("ModeNum") != 1
            or mode.get("UseIntLine") is not True
            or mode.get("CharImp") != "Zpi"
            or len(observed) != 2
            or any(
                not math.isclose(actual, wanted, abs_tol=1e-9)
                for actual_point, expected_point in zip(observed, expected, strict=True)
                for actual, wanted in zip(actual_point, expected_point, strict=True)
            )
        ):
            raise RuntimeError(
                f"AEDT native modal port readback mismatch: {port.name!r}"
            )
        native = record.get("native")
        if not isinstance(native, dict):
            raise TypeError("modal native OO evidence is unavailable")
        native["saved_boundary"] = {
            "bound_type": boundary["BoundType"],
            "wave_port_type": boundary["WavePortType"],
            "faces": boundary["Faces"],
            "num_modes": boundary["NumModes"],
            "deembed": boundary["DoDeembed"],
            "mode_number": mode["ModeNum"],
            "use_integration_line": mode["UseIntLine"],
            "integration_line_um": observed,
            "characteristic_impedance": mode["CharImp"],
        }
    return ports


def _bind_terminal_reference_evidence(
    hfss: Any, spec: HfssDrivenSpec, ports: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Bind global AEDT terminal reference IDs after saving the owned project."""
    if len(ports) != len(spec.ports):
        raise RuntimeError("terminal port evidence count is invalid")
    reference_ids = _native_terminal_reference_ids(hfss)
    reference_objects = [
        hfss.oeditor.GetObjectNameByID(object_id) for object_id in reference_ids
    ]
    expected = list(spec.ports[0].reference_objects)
    if reference_objects != expected:
        raise RuntimeError("AEDT native terminal reference conductor mismatch")
    for record, port in zip(ports, spec.ports, strict=True):
        if (
            not isinstance(port, TerminalPort)
            or list(port.reference_objects) != expected
        ):
            raise RuntimeError("terminal reference contract is not globally consistent")
        native = record.get("native")
        if not isinstance(native, dict):
            raise TypeError("terminal native evidence is unavailable")
        native["reference_conductor_ids"] = reference_ids
        native["reference_conductors"] = reference_objects
    return ports


def _native_terminal_reference_ids(hfss: Any) -> list[int]:
    """Read terminal reference conductors from HFSS native design properties."""
    try:
        values = hfss.design_properties["BoundarySetup"]["ProductSpecificData"][
            "TerminalReferenceConductors"
        ]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "AEDT native terminal reference conductors are unavailable"
        ) from exc
    if not isinstance(values, list) or not values:
        raise RuntimeError("AEDT native terminal reference conductors are invalid")
    try:
        return [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("AEDT native terminal reference IDs are invalid") from exc


def _native_oo_properties(hfss: Any, path: str) -> dict[str, Any]:
    """Return properties directly from an AEDT object-oriented child object."""
    child = hfss.get_oo_object(hfss.odesign, path)
    if not child:
        raise RuntimeError(f"AEDT native child object is unavailable: {path!r}")
    names = child.GetPropNames()
    if not names:
        raise RuntimeError(f"AEDT native child has no properties: {path!r}")
    return {name: child.GetPropValue(name) for name in names}


def _require_native_properties(
    observed: dict[str, Any], expected: dict[str, Any], context: str
) -> None:
    """Require semantic settings from their direct AEDT native representation."""
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if key not in observed or not _native_value_matches(observed[key], value)
    }
    if mismatches:
        raise RuntimeError(f"{context} native property mismatch: {mismatches!r}")


def _read_physics_warnings(run_dir: Path) -> dict[str, Any]:
    """Preserve HFSS terminal-mode diagnostics without treating them as gates."""
    path = run_dir / "batch.log"
    if not path.is_file():
        return {"batch_log": "batch.log", "present": False, "physics_warnings": []}
    warnings: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        conductor = re.search(
            r"Port '([^']+)' has (\d+) signal conductors with (\d+) terminals",
            raw_line,
        )
        slow_mode = re.search(
            r"Port ([^ ]+) supports an additional propagating and/or slowly decaying mode",
            raw_line,
        )
        if conductor:
            port, signals, terminals = conductor.groups()
            kind = "signal_conductors_per_terminal"
            detail = {
                "kind": kind,
                "port": port,
                "signal_conductors": int(signals),
                "terminals": int(terminals),
            }
        elif slow_mode:
            kind = "additional_propagating_or_slow_mode"
            detail = {"kind": kind, "port": slow_mode.group(1)}
        else:
            continue
        key = (kind, detail["port"])
        record = warnings.setdefault(key, {**detail, "occurrences": 0})
        record["occurrences"] += 1
    return {
        "batch_log": "batch.log",
        "sha256": file_sha256(path),
        "present": True,
        "physics_warnings": [warnings[key] for key in sorted(warnings)],
    }


def _runtime_source_identity() -> dict[str, str]:
    """Bind this receipt to the exact SCGSim source bytes that launched AEDT."""
    source_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("SCGSim runtime source revision is invalid")
    return {
        "revision": revision,
        "run_py_sha256": file_sha256(Path(__file__).resolve()),
        "spec_py_sha256": file_sha256(Path(__file__).with_name("spec.py")),
    }


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - exercised by the handoff launcher
    raise SystemExit(main())
