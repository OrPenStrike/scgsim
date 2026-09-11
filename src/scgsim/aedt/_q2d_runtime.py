"""Q2D native preparation, solve, and export stages.

PreparedQ2d is a frozen carrier for one mutable native application handle; it
is not a deeply immutable semantic snapshot and does not own the Desktop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._matrix_export import read_q2d_rlgc_matrix
from ._native_common import (
    native_object_property as _native_object_property,
    pyaedt_version as _pyaedt_version,
    saved_setup_properties as _saved_setup_properties,
)
from ._q2d_convergence import read_q2d_convergence
from .spec import REQUIRED_AEDT_VERSION, Q2dSpec
from .util import file_sha256


def _create_q2d_geometry(
    app: Any, spec: Q2dSpec
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if app.modeler.object_names:
        raise RuntimeError("new Q2D design must not inherit geometry")
    objects: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for rectangle in spec.rectangles:
        material = spec.materials[rectangle.material_id]
        library_name = "pec" if material.is_superconducting else material.library_name
        if not material.is_superconducting and not app.materials.exists_material(
            library_name
        ):
            raise RuntimeError(
                f"AEDT library material is unavailable: {library_name!r}"
            )
        obj = app.create_rectangle(
            list(rectangle.origin_um),
            list(rectangle.size_um),
            name=rectangle.name,
            material=library_name,
        )
        if obj is None or obj.name != rectangle.name:
            raise RuntimeError(f"Q2D rectangle creation failed for {rectangle.name!r}")
        observed = _native_object_property(obj, "Material").strip('"')
        if observed.casefold() != library_name.casefold():
            raise RuntimeError(f"Q2D material readback mismatch for {rectangle.name!r}")
        objects[rectangle.name] = obj
        records.append(
            {
                "object_name": rectangle.name,
                "origin_um": list(rectangle.origin_um),
                "size_um": list(rectangle.size_um),
                "material_id": material.material_id,
                "kind": material.kind,
                "is_superconducting": material.is_superconducting,
                "requested_library_name": material.library_name,
                "observed": {"native_material_name": observed},
            }
        )
    if set(app.modeler.object_names) != set(objects):
        raise RuntimeError("Q2D native object inventory does not match the spec")
    return records, objects


def _create_q2d_region(app: Any, spec: Q2dSpec) -> dict[str, Any]:
    plus_x, minus_x, plus_y, minus_y = spec.region_padding_um
    region = app.modeler.create_region(
        pad_value=[plus_x, plus_y, minus_x, minus_y],
        pad_type="Absolute Offset",
        name="Region",
    )
    vacuum = spec.materials[spec.vacuum_material_id]
    if not app.materials.exists_material(vacuum.library_name):
        raise RuntimeError(
            f"AEDT vacuum material is unavailable: {vacuum.library_name!r}"
        )
    region.material_name = vacuum.library_name
    observed_material = _native_object_property(region, "Material").strip('"')
    if region.name != "Region" or observed_material.casefold() != "vacuum":
        raise RuntimeError("Q2D vacuum region readback failed")
    return {
        "material_id": vacuum.material_id,
        "requested_library_name": vacuum.library_name,
        "observed_material_name": observed_material,
        "padding_um": list(spec.region_padding_um),
        "native_bounding_box_um": [float(value) for value in region.bounding_box],
    }


def _assign_q2d_conductors(
    app: Any, spec: Q2dSpec, objects: dict[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for conductor in spec.conductors:
        selected = [objects[name] for name in conductor.object_names]
        boundary = app.assign_single_conductor(
            selected,
            name=conductor.name,
            conductor_type=conductor.conductor_type,
            solve_option="SolveOnBoundary",
            thickness=conductor.thickness_um,
            units="um",
        )
        if boundary is None:
            raise RuntimeError(
                f"Q2D conductor assignment failed for {conductor.name!r}"
            )
        expected_ids = [int(objects[name].id) for name in conductor.object_names]
        native_ids = [
            int(value)
            for value in app.oboundary.GetExcitationAssignment(conductor.name)
        ]
        if native_ids != expected_ids:
            raise RuntimeError(
                f"Q2D native conductor assignment mismatch for {conductor.name!r}"
            )
        records.append(
            {
                "name": conductor.name,
                "conductor_type": conductor.conductor_type,
                "object_names": list(conductor.object_names),
                "thickness_um": conductor.thickness_um,
                "solve_option": "SolveOnBoundary",
                "native_object_ids": native_ids,
            }
        )
    return records


def _setup_q2d(app: Any, spec: Q2dSpec) -> None:
    if app.setup_names:
        raise RuntimeError("new Q2D design must not inherit a setup")
    setup = app.create_setup(spec.run_control.setup_name)
    setup.props["AdaptiveFreq"] = f"{spec.run_control.frequency_ghz:g}GHz"
    setup.props["CGDataBlock"]["MaxPass"] = spec.run_control.maximum_passes
    setup.props["CGDataBlock"]["MinPass"] = spec.run_control.minimum_passes
    setup.props["CGDataBlock"][
        "MinConvPass"
    ] = spec.run_control.minimum_converged_passes
    setup.props["CGDataBlock"]["PerError"] = spec.run_control.convergence_percent
    setup.props["CGDataBlock"]["PerRefine"] = spec.run_control.percent_refinement
    setup.props["RLDataBlock"]["MaxPass"] = spec.run_control.maximum_passes
    setup.props["RLDataBlock"]["MinPass"] = spec.run_control.minimum_passes
    setup.props["RLDataBlock"][
        "MinConvPass"
    ] = spec.run_control.minimum_converged_passes
    setup.props["RLDataBlock"]["PerError"] = spec.run_control.convergence_percent
    setup.props["RLDataBlock"]["PerRefine"] = spec.run_control.percent_refinement
    if not setup.update():
        raise RuntimeError("Q2D setup update failed")


def _read_q2d_setup(app: Any, spec: Q2dSpec) -> dict[str, Any]:
    raw = _saved_setup_properties(app, spec.run_control.setup_name)
    cg = raw.get("CGDataBlock")
    rl = raw.get("RLDataBlock")
    if not isinstance(cg, dict) or not isinstance(rl, dict):
        raise TypeError("Q2D saved setup lacks CG or RL controls")
    native = {
        "adaptive_frequency": raw.get("AdaptiveFreq"),
        "cg_maximum_passes": cg.get("MaxPass"),
        "cg_minimum_passes": cg.get("MinPass"),
        "cg_minimum_converged_passes": cg.get("MinConvPass"),
        "cg_convergence_percent": cg.get("PerError"),
        "cg_percent_refinement": cg.get("PerRefine"),
        "rl_maximum_passes": rl.get("MaxPass"),
        "rl_minimum_passes": rl.get("MinPass"),
        "rl_minimum_converged_passes": rl.get("MinConvPass"),
        "rl_convergence_percent": rl.get("PerError"),
        "rl_percent_refinement": rl.get("PerRefine"),
    }
    expected = {
        "adaptive_frequency": f"{spec.run_control.frequency_ghz:g}GHz",
        "cg_maximum_passes": spec.run_control.maximum_passes,
        "cg_minimum_passes": spec.run_control.minimum_passes,
        "cg_minimum_converged_passes": spec.run_control.minimum_converged_passes,
        "cg_convergence_percent": spec.run_control.convergence_percent,
        "cg_percent_refinement": spec.run_control.percent_refinement,
        "rl_maximum_passes": spec.run_control.maximum_passes,
        "rl_minimum_passes": spec.run_control.minimum_passes,
        "rl_minimum_converged_passes": spec.run_control.minimum_converged_passes,
        "rl_convergence_percent": spec.run_control.convergence_percent,
        "rl_percent_refinement": spec.run_control.percent_refinement,
    }
    if native != expected:
        raise RuntimeError(f"Q2D saved setup readback mismatch: {native!r}")
    return {"name": spec.run_control.setup_name, "native": native}


def _export_q2d(
    app: Any, run_dir: Path, spec: Q2dSpec
) -> tuple[dict[str, str], dict[str, Any]]:
    output_dir = run_dir / "results" / "q2d"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rlgc_matrix.csv"
    frequency_hz = spec.run_control.frequency_ghz * 1e9
    exported = app.export_matrix_data(
        file_name=str(path),
        problem_type="CG, RL",
        variations="",
        setup=spec.run_control.setup_name,
        sweep="LastAdaptive",
        reduce_matrix="Original",
        r_unit="ohm",
        l_unit="nH",
        c_unit="pF",
        g_unit="mho",
        freq=f"{frequency_hz:g}",
        matrix_type="Maxwell, Spice, Couple",
        export_ac_dc_res=False,
        precision=15,
        field_width=20,
        use_sci_notation=True,
        length_setting="Distributed",
        length="1meter",
    )
    if exported is not True:
        raise RuntimeError("Q2D combined CG/RL matrix export failed")
    rows, summary = read_q2d_rlgc_matrix(path, spec)
    relative = path.relative_to(run_dir).as_posix()
    return {relative: file_sha256(path)}, {
        "matrices": {
            "path": relative,
            "frequency_ghz": spec.run_control.frequency_ghz,
            "length_setting": "Distributed",
            "length": "1meter",
            "matrix_type": "Maxwell, Spice, Couple",
            "native": summary,
            "primary_rows": len(rows),
        }
    }


@dataclass(frozen=True)
class PreparedQ2d:
    """Carrier for one prepared Q2D app and its native readback records."""

    app: Any
    project_path: Path
    materials: list[dict[str, Any]]
    region: dict[str, Any]
    conductors: list[dict[str, Any]]
    setup: dict[str, Any]


def prepare_q2d(Q2d: Any, run_dir: Path, spec: Q2dSpec) -> PreparedQ2d:
    """Construct and read back the Q2D model before the explicit solve stage."""
    project_path = run_dir / f"{spec.project_name}.aedt"
    app = Q2d(
        project=str(project_path),
        design=spec.design_name,
        new_desktop=False,
        close_on_exit=False,
    )
    if app.desktop_class.aedt_version_id != REQUIRED_AEDT_VERSION:
        raise RuntimeError("Q2D did not bind the owned AEDT 2024.2 desktop")
    app.modeler.model_units = "um"
    materials, objects = _create_q2d_geometry(app, spec)
    region = _create_q2d_region(app, spec)
    conductors = _assign_q2d_conductors(app, spec, objects)
    _setup_q2d(app, spec)
    if not app.save_project() or not project_path.is_file():
        raise RuntimeError("Q2D project was not saved before solve")
    setup = _read_q2d_setup(app, spec)
    return PreparedQ2d(app, project_path, materials, region, conductors, setup)


def solve_q2d(prepared: PreparedQ2d, spec: Q2dSpec) -> None:
    """Run the one explicit Q2D setup solve."""
    if not prepared.app.analyze_setup(name=spec.run_control.setup_name, blocking=True):
        raise RuntimeError(
            f"Q2D failed to analyze setup {spec.run_control.setup_name!r}"
        )


def export_q2d(prepared: PreparedQ2d, run_dir: Path, spec: Q2dSpec) -> dict[str, Any]:
    """Export, perform the final save, and bind convergence readback."""
    outputs, result_readback = _export_q2d(prepared.app, run_dir, spec)
    if not prepared.app.save_project() or not prepared.project_path.is_file():
        raise RuntimeError("Q2D project was not saved")
    convergence = read_q2d_convergence(run_dir, spec)
    relative = prepared.project_path.relative_to(run_dir).as_posix()
    outputs[relative] = file_sha256(prepared.project_path)
    return {
        "outputs": outputs,
        "connected": {
            "aedt_version": prepared.app.desktop_class.aedt_version_id,
            "pyaedt_version": _pyaedt_version(),
        },
        "project": relative,
        "conductors": prepared.conductors,
        "materials": prepared.materials,
        "region": prepared.region,
        "setup": prepared.setup,
        "convergence": convergence,
        "result_readback": result_readback,
        "save": {"ok": True, "project_sha256": outputs[relative]},
    }


def run_q2d(Q2d: Any, run_dir: Path, spec: Q2dSpec) -> dict[str, Any]:
    """Use the same preparation stage as diagnostics, then solve and export."""
    prepared = prepare_q2d(Q2d, run_dir, spec)
    solve_q2d(prepared, spec)
    return export_q2d(prepared, run_dir, spec)


__all__ = ["PreparedQ2d", "export_q2d", "prepare_q2d", "run_q2d", "solve_q2d"]
