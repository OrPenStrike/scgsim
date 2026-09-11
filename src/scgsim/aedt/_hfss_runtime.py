"""HFSS native preparation, solve, and export stages.

PreparedHfss is a frozen carrier for one mutable native application handle; it
is not a deeply immutable semantic snapshot and does not own the Desktop.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from ._hfss_convergence import read_hfss_convergence
from ._native_common import (
    create_region as _create_region,
    import_and_bind as _import_and_bind,
    native_boundary_names as _native_boundary_names,
    pyaedt_version as _pyaedt_version,
    saved_setup_properties as _saved_setup_properties,
)
from .spec import (
    POINT_COUNT,
    REQUIRED_AEDT_VERSION,
    SURFACE_APPROXIMATION_LEVEL,
    HfssEigenmodeSpec,
    HfssSpec,
    ModalPort,
    TerminalPort,
)
from .util import file_sha256, write_csv


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
        setup.props["MinimumPasses"] = spec.run_control.minimum_passes
        setup.props["MinimumConvergedPasses"] = (
            spec.run_control.minimum_converged_passes
        )
        setup.props["PercentRefinement"] = spec.run_control.percent_refinement
        if not setup.update():
            raise RuntimeError("HFSS Eigenmode setup update failed")
        return
    setup.props["SolveType"] = (
        "DrivenTerminal" if spec.mode == "terminal" else "DrivenModal"
    )
    if not setup.enable_adaptive_setup_broadband(
        f"{spec.run_control.sweep.start_ghz}GHz",
        f"{spec.run_control.sweep.stop_ghz}GHz",
        max_passes=spec.run_control.maximum_passes,
        max_delta_s=spec.run_control.maximum_delta_s,
    ):
        raise RuntimeError("adaptive setup initialization failed")
    setup.props["MinimumPasses"] = spec.run_control.minimum_passes
    setup.props["MinimumConvergedPasses"] = spec.run_control.minimum_converged_passes
    setup.props["PercentRefinement"] = spec.run_control.percent_refinement
    if not setup.update():
        raise RuntimeError("HFSS Driven adaptive setup update failed")
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


def _read_hfss_setup(hfss: Any, spec: HfssSpec) -> dict[str, Any]:
    raw = _saved_setup_properties(hfss, spec.run_control.setup_name)
    if isinstance(spec, HfssEigenmodeSpec):
        native = {
            "minimum_frequency": raw.get("MinimumFrequency"),
            "num_modes": raw.get("NumModes"),
            "maximum_delta_frequency_percent": raw.get("MaxDeltaFreq"),
            "maximum_passes": raw.get("MaximumPasses"),
            "minimum_passes": raw.get("MinimumPasses"),
            "minimum_converged_passes": raw.get("MinimumConvergedPasses"),
            "percent_refinement": raw.get("PercentRefinement"),
        }
        expected = {
            "minimum_frequency": f"{spec.run_control.minimum_frequency_ghz:g}GHz",
            "num_modes": spec.run_control.num_modes,
            "maximum_delta_frequency_percent": (
                spec.run_control.maximum_delta_frequency_percent
            ),
            "maximum_passes": spec.run_control.maximum_passes,
            "minimum_passes": spec.run_control.minimum_passes,
            "minimum_converged_passes": spec.run_control.minimum_converged_passes,
            "percent_refinement": spec.run_control.percent_refinement,
        }
    else:
        frequencies = raw.get("MultipleAdaptiveFreqsSetup")
        if not isinstance(frequencies, dict):
            raise TypeError("HFSS Driven saved setup lacks broadband frequencies")
        native = {
            "solve_type": raw.get("SolveType"),
            "low_frequency": frequencies.get("Low"),
            "high_frequency": frequencies.get("High"),
            "maximum_delta_s": raw.get("MaxDeltaS"),
            "maximum_passes": raw.get("MaximumPasses"),
            "minimum_passes": raw.get("MinimumPasses"),
            "minimum_converged_passes": raw.get("MinimumConvergedPasses"),
            "percent_refinement": raw.get("PercentRefinement"),
        }
        expected = {
            "solve_type": "Broadband",
            "low_frequency": f"{spec.run_control.sweep.start_ghz:g}GHz",
            "high_frequency": f"{spec.run_control.sweep.stop_ghz:g}GHz",
            "maximum_delta_s": spec.run_control.maximum_delta_s,
            "maximum_passes": spec.run_control.maximum_passes,
            "minimum_passes": spec.run_control.minimum_passes,
            "minimum_converged_passes": spec.run_control.minimum_converged_passes,
            "percent_refinement": spec.run_control.percent_refinement,
        }
    if native != expected:
        raise RuntimeError(f"HFSS saved setup readback mismatch: {native!r}")
    return {"name": spec.run_control.setup_name, "native": native}


def _saved_setup_properties(app: Any, setup_name: str) -> dict[str, Any]:
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


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text")
    return value


@dataclass(frozen=True)
class PreparedHfss:
    """Carrier for one prepared HFSS app and its native readback records."""

    app: Any
    project_path: Path
    materials: list[dict[str, Any]]
    region: dict[str, Any]
    mesh: dict[str, Any]
    ports: list[dict[str, Any]]
    setup: dict[str, Any]


def prepare_hfss(Hfss: Any, run_dir: Path, spec: HfssSpec) -> PreparedHfss:
    """Construct and read back the HFSS model before the explicit solve stage."""
    project_path = run_dir / f"{spec.project_name}.aedt"
    app = Hfss(
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
    if app.desktop_class.aedt_version_id != REQUIRED_AEDT_VERSION:
        raise RuntimeError("HFSS did not bind the owned AEDT 2024.2 desktop")
    app.modeler.model_units = "um"
    materials = _import_and_bind(app, spec)
    region = _create_region(app, spec)
    mesh = _assign_mesh(app, spec)
    ports = _assign_ports(app, spec)
    _setup(app, spec)
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("HFSS project was not saved before native port readback")
    setup = _read_hfss_setup(app, spec)
    ports = _bind_port_evidence(app, spec, ports)
    return PreparedHfss(app, project_path, materials, region, mesh, ports, setup)


def solve_hfss(prepared: PreparedHfss, spec: HfssSpec) -> None:
    """Run the one explicit HFSS setup solve."""
    if not prepared.app.analyze_setup(name=spec.run_control.setup_name, blocking=True):
        raise RuntimeError(
            f"HFSS failed to analyze setup {spec.run_control.setup_name!r}"
        )


def export_hfss(
    prepared: PreparedHfss, run_dir: Path, spec: HfssSpec
) -> dict[str, Any]:
    """Export, perform the final save, and bind convergence readback."""
    outputs, result_readback = _export(prepared.app, run_dir, spec, prepared.ports)
    saved = bool(prepared.app.save_project())
    if not saved or not prepared.project_path.is_file():
        raise RuntimeError("HFSS project was not saved")
    convergence = read_hfss_convergence(run_dir, spec)
    relative = prepared.project_path.relative_to(run_dir).as_posix()
    outputs[relative] = file_sha256(prepared.project_path)
    return {
        "outputs": outputs,
        "connected": {
            "aedt_version": prepared.app.desktop_class.aedt_version_id,
            "pyaedt_version": _pyaedt_version(),
        },
        "project": relative,
        "ports": prepared.ports,
        "mesh": prepared.mesh,
        "materials": prepared.materials,
        "region": prepared.region,
        "setup": prepared.setup,
        "convergence": convergence,
        "result_readback": result_readback,
        "save": {"ok": True, "project_sha256": outputs[relative]},
    }


def run_hfss(Hfss: Any, run_dir: Path, spec: HfssSpec) -> dict[str, Any]:
    """Use the same preparation stage as diagnostics, then solve and export."""
    prepared = prepare_hfss(Hfss, run_dir, spec)
    solve_hfss(prepared, spec)
    return export_hfss(prepared, run_dir, spec)


__all__ = ["PreparedHfss", "export_hfss", "prepare_hfss", "run_hfss", "solve_hfss"]
