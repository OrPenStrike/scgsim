"""Receipt-bound resolution of one completed AEDT handoff."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ._hfss_convergence import read_hfss_convergence
from ._matrix_export import parse_matrix_export, read_q2d_rlgc_matrix
from ._q2d_convergence import read_q2d_convergence, read_q3d_convergence
from .spec import (
    HfssDrivenSpec,
    HfssEigenmodeSpec,
    ModalPort,
    Q2dSpec,
    Q3dSpec,
    parse_aedt_spec,
)
from .util import file_sha256, read_json

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
class ResolvedRun:
    """Canonical result paths verified against the completed run receipt."""

    mode: Literal["terminal", "modal", "eigenmode", "q3d", "q2d"]
    project_path: Path
    primary_csv: Path
    touchstone_path: Path | None
    provenance_path: Path | None
    receipt_path: Path
    convergence: dict[str, Any] | None = None

    def physics_results(self) -> tuple[dict[str, str], ...]:
        """Return the verified primary result as string-valued rows."""

        if self.mode == "q2d":
            root = self.receipt_path.parent.parent
            spec = parse_aedt_spec(read_json(root / "aedt_spec.json"), base_dir=root)
            if not isinstance(spec, Q2dSpec):
                raise RuntimeError("resolved Q2D result has a non-Q2D spec")
            rows, _ = read_q2d_rlgc_matrix(self.primary_csv, spec)
            return tuple(
                {key: str(value) for key, value in row.items()} for row in rows
            )

        if self.mode == "q3d":
            root = self.receipt_path.parent.parent
            spec = parse_aedt_spec(read_json(root / "aedt_spec.json"), base_dir=root)
            if not isinstance(spec, Q3dSpec):
                raise RuntimeError("resolved Q3D result has a non-Q3D spec")
            if not spec.solve_ac_rl:
                rows, _ = parse_matrix_export(
                    self.primary_csv,
                    "Q3D",
                    "C",
                    spec.run_control.frequency_ghz,
                    {"Capacitance Matrix": "C", "Conductance Matrix": "G"},
                )
                return tuple(
                    {key: str(value) for key, value in row.items()} for row in rows
                )

        with self.primary_csv.open(newline="", encoding="utf-8-sig") as stream:
            return tuple(dict(row) for row in csv.DictReader(stream))

    def simulation_benchmark(self) -> dict[str, Any]:
        """Return the receipt-bound execution summary for notebook display."""

        receipt = read_json(self.receipt_path)
        return {
            "mode": self.mode,
            "execution_seconds": receipt["execution_seconds"],
            "project_bytes": self.project_path.stat().st_size,
            "primary_csv_bytes": self.primary_csv.stat().st_size,
        }


def resolve_results(run_dir: str | Path) -> ResolvedRun:
    """Return results only when a complete exact receipt proves every artifact."""
    root = Path(run_dir).resolve()
    receipt_path = _contained(root, "metadata/aedt_run_receipt.json")
    if not receipt_path.is_file():
        raise FileNotFoundError(f"run receipt is missing: {receipt_path}")
    receipt = read_json(receipt_path)
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != "scgsim.aedt.receipt.v1"
    ):
        raise RuntimeError("handoff receipt schema is invalid")
    if receipt.get("status") != "completed":
        raise RuntimeError("handoff has not completed successfully")
    save = receipt.get("save")
    if (
        not isinstance(save, dict)
        or save.get("ok") is not True
        or not isinstance(save.get("project_sha256"), str)
    ):
        raise RuntimeError("completed receipt lacks a successful exact project save")
    if receipt.get("release") != {"ok": True}:
        raise RuntimeError("completed receipt lacks a successful owned Desktop release")
    source = receipt.get("source")
    mode = receipt.get("mode")
    if not isinstance(source, dict) or source.get("spec") != "aedt_spec.json":
        raise RuntimeError("receipt source paths are not canonical")
    spec_path = _verified(root, "aedt_spec.json", source, "spec_sha256")
    if mode == "q2d":
        if set(source) != {"spec", "spec_sha256"}:
            raise RuntimeError("Q2D receipt must not contain a GDS source")
    else:
        if source.get("gds") != "geometry/design.gds":
            raise RuntimeError("receipt GDS path is not canonical")
        _verified(root, "geometry/design.gds", source, "gds_sha256")
    spec = parse_aedt_spec(read_json(spec_path), base_dir=root)
    if (
        mode not in {"terminal", "modal", "eigenmode", "q3d", "q2d"}
        or spec.mode != mode
    ):
        raise RuntimeError("AEDT receipt mode is invalid")
    _validate_readback(root, receipt, spec)
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise TypeError("completed receipt has no output hash manifest")
    project_relative = f"{spec.project_name}.aedt"
    if receipt.get("project") != project_relative:
        raise RuntimeError("completed receipt project path is not canonical")
    project = _verified(root, project_relative, outputs, project_relative)
    if save["project_sha256"] != outputs.get(project_relative):
        raise RuntimeError("saved project hash does not match output manifest")
    if mode == "q2d":
        superseded = (
            "results/q2d/cg_matrix.csv",
            "results/q2d/rl_matrix.csv",
            "results/q2d/matrices.csv",
        )
        if any(_contained(root, relative).exists() for relative in superseded):
            raise RuntimeError("Q2D run contains superseded matrix artifacts")
        expected = {
            project_relative,
            "results/q2d/rlgc_matrix.csv",
        }
        if set(outputs) != expected:
            raise RuntimeError("Q2D output manifest is not canonical")
        return ResolvedRun(
            "q2d",
            project,
            _verified(root, "results/q2d/rlgc_matrix.csv", outputs),
            None,
            None,
            receipt_path,
            convergence=receipt["convergence"],
        )
    if mode == "q3d":
        expected = {project_relative, "results/q3d/c_matrix.csv"}
        primary = "results/q3d/c_matrix.csv"
        if spec.solve_ac_rl:
            expected.update(
                {"results/q3d/ac_rl_matrix.csv", "results/q3d/matrices.csv"}
            )
            primary = "results/q3d/matrices.csv"
        elif any(
            _contained(root, relative).exists()
            for relative in ("results/q3d/ac_rl_matrix.csv", "results/q3d/matrices.csv")
        ):
            raise RuntimeError("Q3D capacitance-only run contains AC-RL artifacts")
        if set(outputs) != expected:
            raise RuntimeError("Q3D output manifest is not canonical")
        return ResolvedRun(
            "q3d",
            project,
            _verified(root, primary, outputs),
            None,
            None,
            receipt_path,
            convergence=receipt["convergence"],
        )
    if mode == "eigenmode":
        expected = {
            project_relative,
            "results/eigenmode/eigenmodes.csv",
            "results/eigenmode/eigenmodes.eig",
        }
        if set(outputs) != expected:
            raise RuntimeError("Eigenmode output manifest is not canonical")
        return ResolvedRun(
            "eigenmode",
            project,
            _verified(root, "results/eigenmode/eigenmodes.csv", outputs),
            None,
            _verified(root, "results/eigenmode/eigenmodes.eig", outputs),
            receipt_path,
            convergence=receipt["convergence"],
        )
    result_stem = "terminal_st" if mode == "terminal" else "modal_s"
    expected = {
        project_relative,
        f"results/{mode}/{result_stem}.csv",
        f"results/{mode}/{mode}.s2p",
    }
    if set(outputs) != expected:
        raise RuntimeError("terminal output manifest is not canonical")
    return ResolvedRun(
        mode,
        project,
        _verified(root, f"results/{mode}/{result_stem}.csv", outputs),
        _verified(root, f"results/{mode}/{mode}.s2p", outputs),
        None,
        receipt_path,
        convergence=receipt["convergence"],
    )


def _contained(root: Path, relative: str) -> Path:
    requested = Path(relative)
    if requested.is_absolute():
        raise RuntimeError(f"receipt path escapes handoff root: {relative!r}")
    resolved = (root / requested).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"receipt path escapes handoff root: {relative!r}")
    return resolved


def _validate_readback(root: Path, receipt: dict[str, Any], spec: Any) -> None:
    connected = receipt.get("connected")
    if not isinstance(connected, dict) or connected != {
        "aedt_version": spec.aedt_version,
        "pyaedt_version": spec.pyaedt_version,
    }:
        raise RuntimeError("completed receipt has invalid connected version identity")
    readback = receipt.get("result_readback")
    if isinstance(spec, Q2dSpec):
        _validate_q2d_readback(root, receipt, spec)
        _validate_diagnostics(root, receipt.get("diagnostics"))
        return
    if isinstance(spec, Q3dSpec):
        _validate_q3d_readback(root, receipt, spec)
        _validate_diagnostics(root, receipt.get("diagnostics"))
        return
    if isinstance(spec, HfssEigenmodeSpec):
        _validate_eigenmode_readback(receipt, spec)
        _validate_hfss_setup_and_convergence(root, receipt, spec)
        _validate_diagnostics(root, receipt.get("diagnostics"))
        return
    key = "terminal_st" if spec.mode == "terminal" else "modal_s"
    if not isinstance(readback, dict) or not isinstance(readback.get(key), dict):
        raise TypeError("completed receipt has no canonical result readback")
    frequency = readback[key]
    if frequency != {
        "records": 20_000,
        "frequency_unit": "GHz",
        "first_frequency_ghz": spec.run_control.sweep.start_ghz,
        "last_frequency_ghz": spec.run_control.sweep.stop_ghz,
        "strictly_increasing": True,
    }:
        raise RuntimeError("completed receipt result frequency readback is invalid")
    touchstone = readback.get("touchstone")
    ports = receipt.get("ports")
    port_key = "terminal_excitation" if spec.mode == "terminal" else "modal_excitation"
    port_order = (
        [record.get(port_key) for record in ports]
        if isinstance(ports, list)
        and len(ports) == 2
        and all(isinstance(record, dict) for record in ports)
        and [record.get("index") for record in ports] == [1, 2]
        else None
    )
    if (
        not isinstance(touchstone, dict)
        or touchstone.get("path") != f"{spec.mode}.s2p"
        or touchstone.get("ports") != 2
        or touchstone.get("records") != 20_000
        or touchstone.get("frequency_unit") != "GHz"
        or touchstone.get("first_frequency_ghz") != spec.run_control.sweep.start_ghz
        or touchstone.get("last_frequency_ghz") != spec.run_control.sweep.stop_ghz
        or touchstone.get("strictly_increasing") is not True
        or touchstone.get("port_order") != port_order
        or not all(isinstance(name, str) and name for name in port_order or [])
        or len(set(port_order or [])) != 2
        or not isinstance(touchstone.get("bytes"), int)
        or touchstone["bytes"] <= 0
    ):
        raise RuntimeError(
            "completed receipt is missing a two-port Touchstone readback"
        )
    if spec.mode == "terminal":
        _validate_terminal_native_evidence(ports, spec)
    else:
        _validate_modal_native_evidence(ports, spec)
    _validate_hfss_setup_and_convergence(root, receipt, spec)
    _validate_diagnostics(root, receipt.get("diagnostics"))


def _validate_hfss_setup_and_convergence(
    root: Path, receipt: dict[str, Any], spec: HfssDrivenSpec | HfssEigenmodeSpec
) -> None:
    if isinstance(spec, HfssEigenmodeSpec):
        native = {
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
        native = {
            "solve_type": "Broadband",
            "low_frequency": f"{spec.run_control.sweep.start_ghz:g}GHz",
            "high_frequency": f"{spec.run_control.sweep.stop_ghz:g}GHz",
            "maximum_delta_s": spec.run_control.maximum_delta_s,
            "maximum_passes": spec.run_control.maximum_passes,
            "minimum_passes": spec.run_control.minimum_passes,
            "minimum_converged_passes": spec.run_control.minimum_converged_passes,
            "percent_refinement": spec.run_control.percent_refinement,
        }
    if receipt.get("setup") != {"name": spec.run_control.setup_name, "native": native}:
        raise RuntimeError("HFSS native setup evidence is invalid")
    if receipt.get("convergence") != read_hfss_convergence(root, spec):
        raise RuntimeError("HFSS native convergence evidence is invalid")


def _validate_q2d_readback(root: Path, receipt: dict[str, Any], spec: Q2dSpec) -> None:
    if receipt.get("ports") != [] or receipt.get("nets") != []:
        raise RuntimeError("Q2D receipt must not contain HFSS ports or Q3D nets")
    conductors = receipt.get("conductors")
    if not isinstance(conductors, list) or len(conductors) != len(spec.conductors):
        raise RuntimeError("Q2D receipt has invalid conductor evidence")
    for record, expected in zip(conductors, spec.conductors, strict=True):
        if (
            not isinstance(record, dict)
            or record.get("name") != expected.name
            or record.get("conductor_type") != expected.conductor_type
            or record.get("object_names") != list(expected.object_names)
            or record.get("thickness_um") != expected.thickness_um
            or record.get("solve_option") != "SolveOnBoundary"
            or not isinstance(record.get("native_object_ids"), list)
            or len(record["native_object_ids"]) != len(expected.object_names)
            or not all(isinstance(value, int) for value in record["native_object_ids"])
        ):
            raise RuntimeError("Q2D native conductor evidence does not match the spec")
    expected_setup = {
        "name": spec.run_control.setup_name,
        "native": {
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
        },
    }
    if receipt.get("setup") != expected_setup:
        raise RuntimeError("Q2D native setup evidence is invalid")
    if receipt.get("convergence") != read_q2d_convergence(root, spec):
        raise RuntimeError("Q2D native convergence evidence is invalid")
    readback = receipt.get("result_readback")
    matrices = readback.get("matrices") if isinstance(readback, dict) else None
    rows, summary = read_q2d_rlgc_matrix(
        _contained(root, "results/q2d/rlgc_matrix.csv"), spec
    )
    expected_matrices = {
        "path": "results/q2d/rlgc_matrix.csv",
        "frequency_ghz": spec.run_control.frequency_ghz,
        "length_setting": "Distributed",
        "length": "1meter",
        "matrix_type": "Maxwell, Spice, Couple",
        "native": summary,
        "primary_rows": len(rows),
    }
    if matrices != expected_matrices:
        raise RuntimeError("Q2D matrix readback is invalid")


def _validate_q3d_readback(root: Path, receipt: dict[str, Any], spec: Q3dSpec) -> None:
    if receipt.get("ports") != []:
        raise RuntimeError("Q3D receipt must not contain HFSS ports")
    nets = receipt.get("nets")
    if not isinstance(nets, list) or len(nets) != len(spec.nets):
        raise RuntimeError("Q3D receipt has invalid net evidence")
    for record, expected in zip(nets, spec.nets, strict=True):
        if (
            not isinstance(record, dict)
            or record.get("name") != expected.name
            or record.get("net_type") != expected.net_type
            or record.get("object_names") != list(expected.object_names)
            or not isinstance(record.get("native_object_ids"), list)
            or not _positive_unique_ids(
                record["native_object_ids"], len(expected.object_names)
            )
        ):
            raise RuntimeError("Q3D native net evidence does not match the spec")
        if expected.net_type == "Signal":
            for kind, object_name, side in (
                ("source", expected.source_object, expected.source_side),
                ("sink", expected.sink_object, expected.sink_side),
            ):
                terminal = record.get(kind)
                if (
                    not isinstance(terminal, dict)
                    or terminal.get("name") != f"{expected.name}{kind.title()}"
                    or terminal.get("object_name") != object_name
                    or terminal.get("side") != side
                    or not isinstance(terminal.get("native_face_ids"), list)
                    or len(terminal["native_face_ids"]) != 1
                    or not isinstance(terminal["native_face_ids"][0], int)
                ):
                    raise RuntimeError("Q3D native terminal evidence is invalid")
        elif "source" in record or "sink" in record:
            raise RuntimeError("Q3D Ground net receipt must not contain terminals")
    net_object_ids = [
        object_id for record in nets for object_id in record["native_object_ids"]
    ]
    if len(set(net_object_ids)) != len(net_object_ids):
        raise RuntimeError("Q3D native object IDs must be unique across nets")
    _validate_q3d_region_ground(receipt, spec, nets)
    expected_setup = {
        "name": spec.run_control.setup_name,
        "native": {
            "adaptive_frequency": f"{spec.run_control.frequency_ghz:g}GHz",
            "capacitance_maximum_passes": spec.run_control.maximum_passes,
            "capacitance_minimum_passes": spec.run_control.minimum_passes,
            "capacitance_minimum_converged_passes": (
                spec.run_control.minimum_converged_passes
            ),
            "capacitance_convergence_percent": spec.run_control.convergence_percent,
            "capacitance_percent_refinement": spec.run_control.percent_refinement,
            "dc_enabled": False,
        },
    }
    if spec.solve_ac_rl:
        expected_setup["native"].update(
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
    if receipt.get("setup") != expected_setup:
        raise RuntimeError("Q3D native setup evidence is invalid")
    if receipt.get("convergence") != read_q3d_convergence(root, spec):
        raise RuntimeError("Q3D native convergence evidence is invalid")
    readback = receipt.get("result_readback")
    matrices = readback.get("matrices") if isinstance(readback, dict) else None
    if spec.solve_ac_rl:
        if (
            not isinstance(matrices, dict)
            or matrices.get("frequency_ghz") != spec.run_control.frequency_ghz
            or set(matrices.get("native", {})) != {"c", "ac_rl"}
            or not isinstance(matrices.get("normalized_rows"), int)
            or matrices["normalized_rows"] <= 0
        ):
            raise RuntimeError("Q3D matrix readback is invalid")
        _validate_normalized_matrices(
            _contained(root, "results/q3d/matrices.csv"),
            matrices["normalized_rows"],
            {"C", "AC RL"},
        )
        return
    rows, summary = parse_matrix_export(
        _contained(root, "results/q3d/c_matrix.csv"),
        "Q3D",
        "C",
        spec.run_control.frequency_ghz,
        {"Capacitance Matrix": "C", "Conductance Matrix": "G"},
    )
    expected_matrices = {
        "path": "results/q3d/c_matrix.csv",
        "frequency_ghz": spec.run_control.frequency_ghz,
        "native": summary,
        "primary_rows": len(rows),
    }
    if matrices != expected_matrices:
        raise RuntimeError("Q3D capacitance-only matrix readback is invalid")


def _validate_q3d_region_ground(
    receipt: dict[str, Any], spec: Q3dSpec, nets: list[Any]
) -> None:
    region = receipt.get("region")
    if not isinstance(region, dict):
        raise TypeError("Q3D receipt has invalid Region evidence")
    requested_padding = list(spec.region_padding_um)
    if (
        region.get("padding_um") != requested_padding
        or region.get("requested_padding_um") != requested_padding
    ):
        raise RuntimeError("Q3D receipt Region padding does not match the spec")
    evidence = region.get("grounded_region")
    if spec.grounded_region_net is None:
        if evidence is not None:
            raise RuntimeError("open Q3D Region receipt contains grounding evidence")
        return
    if not isinstance(evidence, dict):
        raise TypeError("grounded Q3D Region receipt lacks evidence")
    source_id, source_bounds, source_faces = _q3d_region_evidence(
        evidence.get("source_region")
    )
    final_id, final_bounds, _final_faces = _q3d_region_evidence(
        evidence.get("final_region")
    )
    sheets = evidence.get("sheets")
    if (
        region.get("native_region_object_id") != final_id
        or region.get("native_bounding_box_um") != list(final_bounds)
        or evidence.get("target_net") != spec.grounded_region_net
        or evidence.get("target_net_origin") != "generated_region_enclosure"
        or evidence.get("native_final_region_padding_um") != [0.0] * 6
        or evidence.get("native_region_object_id") != final_id
        or evidence.get("native_bounding_box_um") != list(final_bounds)
        or not _q3d_values_close(list(source_bounds), list(final_bounds))
        or not isinstance(sheets, list)
        or len(sheets) != len(_Q3D_REGION_DIRECTIONS)
    ):
        raise RuntimeError("grounded Q3D Region receipt is invalid")
    sheet_ids: list[int] = []
    thin_boundaries: list[dict[str, Any]] = []
    source_face_ids: set[int] = set()
    for direction, record in zip(_Q3D_REGION_DIRECTIONS, sheets, strict=True):
        if not isinstance(record, dict):
            raise TypeError("grounded Q3D Region sheet receipt is invalid")
        sheet_name, boundary_name = _Q3D_REGION_SHEET_NAMES[direction]
        face_id = record.get("source_face_id")
        sheet_id = record.get("sheet_object_id")
        source = source_faces[direction]
        expected_bounds = _q3d_sheet_bounds(direction, final_bounds)
        if (
            record.get("direction") != direction
            or record.get("source_region_object_id") != source_id
            or face_id != source["native_face_id"]
            or face_id in source_face_ids
            or record.get("source_face_center_um") != source["native_face_center_um"]
            or record.get("source_face_normal") != source["native_face_normal"]
            or record.get("sheet_name") != sheet_name
            or not _positive_unique_ids([sheet_id], 1)
            or not _q3d_values_close(
                record.get("sheet_bounding_box_um"), expected_bounds
            )
            or record.get("boundary_name") != boundary_name
            or record.get("native_thin_conductor_object_ids") != [sheet_id]
        ):
            raise RuntimeError("grounded Q3D Region sheet receipt is invalid")
        source_face_ids.add(face_id)
        sheet_ids.append(sheet_id)
        thin_boundaries.append(
            {
                "name": boundary_name,
                "bound_type": "ThinConductor",
                "object_ids": [sheet_id],
                "material": "pec",
                "thickness": "1um",
            }
        )
    declared_net_ids = {record["name"]: record["native_object_ids"] for record in nets}
    all_net_ids = {
        object_id for record in nets for object_id in record["native_object_ids"]
    }
    if (
        spec.grounded_region_net in declared_net_ids
        or evidence.get("native_declared_net_object_ids") != declared_net_ids
        or len(sheet_ids) != 6
        or len(set(sheet_ids)) != 6
        or set(sheet_ids) & all_net_ids
        or final_id in all_net_ids
        or final_id in sheet_ids
    ):
        raise RuntimeError("grounded Q3D Region object IDs are invalid")
    expected_ids = sheet_ids
    saved = evidence.get("native_saved_boundaries")
    if (
        evidence.get("native_target_net_object_ids") != expected_ids
        or evidence.get("native_design_validation")
        != {"method": "ValidateDesign", "ok": True}
        or not isinstance(saved, dict)
        or saved.get("target")
        != {
            "name": spec.grounded_region_net,
            "bound_type": "GroundNet",
            "origin": "generated_region_enclosure",
            "object_ids": expected_ids,
        }
        or saved.get("thin_conductors") != thin_boundaries
    ):
        raise RuntimeError("grounded Q3D Region native assignment receipt is invalid")


def _positive_unique_ids(value: Any, count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == count
        and all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in value
        )
        and len(set(value)) == count
    )


def _q3d_region_evidence(
    value: Any,
) -> tuple[
    int, tuple[float, float, float, float, float, float], dict[str, dict[str, Any]]
]:
    if not isinstance(value, dict):
        raise TypeError("Q3D Region evidence is invalid")
    region_id = value.get("native_object_id")
    bounds = _q3d_bounds(value.get("native_bounding_box_um"))
    faces = value.get("faces")
    if (
        not isinstance(region_id, int)
        or isinstance(region_id, bool)
        or region_id <= 0
        or not isinstance(faces, list)
        or len(faces) != len(_Q3D_REGION_DIRECTIONS)
    ):
        raise RuntimeError("Q3D Region evidence is invalid")
    by_direction: dict[str, dict[str, Any]] = {}
    face_ids: set[int] = set()
    for direction, record in zip(_Q3D_REGION_DIRECTIONS, faces, strict=True):
        if (
            not isinstance(record, dict)
            or record.get("direction") != direction
            or not _positive_unique_ids([record.get("native_face_id")], 1)
            or record["native_face_id"] in face_ids
            or not _q3d_face_center_matches(
                direction, record.get("native_face_center_um"), bounds
            )
            or not _q3d_face_normal_matches(direction, record.get("native_face_normal"))
        ):
            raise RuntimeError("Q3D Region face evidence is invalid")
        face_ids.add(record["native_face_id"])
        by_direction[direction] = record
    return region_id, bounds, by_direction


def _q3d_bounds(value: Any) -> tuple[float, float, float, float, float, float]:
    if not isinstance(value, list) or len(value) != 6:
        raise RuntimeError("grounded Q3D Region bounds are invalid")
    try:
        bounds = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("grounded Q3D Region bounds are invalid") from exc
    if not all(math.isfinite(item) for item in bounds) or any(
        bounds[index] >= bounds[index + 3] for index in range(3)
    ):
        raise RuntimeError("grounded Q3D Region bounds are invalid")
    return bounds  # type: ignore[return-value]


def _q3d_face_center_matches(
    direction: str, value: Any, bounds: tuple[float, float, float, float, float, float]
) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    try:
        center = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(item) for item in center):
        return False
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    expected = {
        "+X": [x_max, (y_min + y_max) / 2, (z_min + z_max) / 2],
        "-X": [x_min, (y_min + y_max) / 2, (z_min + z_max) / 2],
        "+Y": [(x_min + x_max) / 2, y_max, (z_min + z_max) / 2],
        "-Y": [(x_min + x_max) / 2, y_min, (z_min + z_max) / 2],
        "+Z": [(x_min + x_max) / 2, (y_min + y_max) / 2, z_max],
        "-Z": [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min],
    }[direction]
    return all(
        math.isclose(actual, required, rel_tol=0.0, abs_tol=1e-9)
        for actual, required in zip(center, expected, strict=True)
    )


def _q3d_sheet_bounds(
    direction: str, bounds: tuple[float, float, float, float, float, float]
) -> list[float]:
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    return {
        "+X": [x_max, y_min, z_min, x_max, y_max, z_max],
        "-X": [x_min, y_min, z_min, x_min, y_max, z_max],
        "+Y": [x_min, y_max, z_min, x_max, y_max, z_max],
        "-Y": [x_min, y_min, z_min, x_max, y_min, z_max],
        "+Z": [x_min, y_min, z_max, x_max, y_max, z_max],
        "-Z": [x_min, y_min, z_min, x_max, y_max, z_min],
    }[direction]


def _q3d_values_close(value: Any, expected: list[float]) -> bool:
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    try:
        observed = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return all(
        math.isclose(actual, required, rel_tol=0.0, abs_tol=1e-9)
        for actual, required in zip(observed, expected, strict=True)
    )


def _q3d_face_normal_matches(direction: str, value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    try:
        normal = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(item) for item in normal):
        return False
    axis, sign = {
        "+X": (0, 1.0),
        "-X": (0, -1.0),
        "+Y": (1, 1.0),
        "-Y": (1, -1.0),
        "+Z": (2, 1.0),
        "-Z": (2, -1.0),
    }[direction]
    return math.isclose(normal[axis], sign, rel_tol=0.0, abs_tol=1e-9) and all(
        math.isclose(item, 0.0, rel_tol=0.0, abs_tol=1e-9)
        for index, item in enumerate(normal)
        if index != axis
    )


def _validate_normalized_matrices(
    path: Path, expected_rows: int, problem_types: set[str]
) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_rows or not rows:
        raise RuntimeError("normalized matrix row count is invalid")
    expected_quantities = {"C", "G", "L", "R"}
    if {row.get("quantity") for row in rows} != expected_quantities:
        raise RuntimeError("normalized matrix quantities are incomplete")
    for row in rows:
        try:
            value = float(row["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("normalized matrix value is invalid") from exc
        if (
            not math.isfinite(value)
            or row.get("problem_type") not in problem_types
            or not row.get("row")
            or not row.get("column")
            or not row.get("unit")
        ):
            raise RuntimeError("normalized matrix row is invalid")


def _validate_eigenmode_readback(
    receipt: dict[str, Any], spec: HfssEigenmodeSpec
) -> None:
    if receipt.get("ports") != []:
        raise RuntimeError("HFSS Eigenmode receipt must not contain ports")
    readback = receipt.get("result_readback")
    values = readback.get("eigenmodes") if isinstance(readback, dict) else None
    if not isinstance(values, dict):
        raise TypeError("completed receipt has no Eigenmode result readback")
    modes = spec.run_control.num_modes
    frequencies = values.get("frequencies_ghz")
    q_factors = values.get("q_factors")
    if (
        values.get("modes") != modes
        or values.get("frequency_unit") != "GHz"
        or values.get("mode_indices") != list(range(1, modes + 1))
        or not isinstance(frequencies, list)
        or len(frequencies) != modes
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in frequencies
        )
        or not isinstance(q_factors, list)
        or len(q_factors) != modes
        or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in q_factors
        )
        or values.get("native_export") != "eigenmodes.eig"
        or not isinstance(values.get("native_export_bytes"), int)
        or values["native_export_bytes"] <= 0
    ):
        raise RuntimeError("completed receipt Eigenmode result readback is invalid")


def _validate_modal_native_evidence(ports: Any, spec: HfssDrivenSpec) -> None:
    if not isinstance(ports, list) or len(ports) != 2:
        raise RuntimeError("completed receipt has invalid modal ports")
    for record, port in zip(ports, spec.ports, strict=True):
        if not isinstance(port, ModalPort) or not isinstance(record, dict):
            raise TypeError("completed receipt has invalid modal port identity")
        if (
            record.get("index") != port.index
            or record.get("boundary") != port.name
            or record.get("modal_excitation") != port.name
            or not isinstance(record.get("face_id"), int)
            or not isinstance(record.get("face_center_um"), list)
            or len(record["face_center_um"]) != 3
            or record.get("requested")
            != {
                "integration_line_um": [
                    list(point) for point in port.integration_line_um
                ],
                "modes": 1,
                "renormalize": False,
                "deembed_um": 0.0,
                "characteristic_impedance": "Zpi",
            }
        ):
            raise RuntimeError("completed receipt modal request does not match spec")
        native = record.get("native")
        if not isinstance(native, dict):
            raise TypeError("completed receipt has no native modal evidence")
        expected_names = [item.name for item in spec.ports[: port.index]]
        expected_boundary = {
            "bound_type": "Wave Port",
            "wave_port_type": "Modal",
            "faces": [record["face_id"]],
            "num_modes": 1,
            "deembed": False,
            "mode_number": 1,
            "use_integration_line": True,
            "characteristic_impedance": "Zpi",
        }
        saved_boundary = native.get("saved_boundary")
        integration_line = (
            saved_boundary.get("integration_line_um")
            if isinstance(saved_boundary, dict)
            else None
        )
        if (
            native.get("excitation_names") != expected_names
            or native.get("boundary_properties")
            != {
                "Deembed": False,
                "Name": port.name,
                "Num Modes": "1",
                "Renorm All Modes": False,
                "Type": "Wave Port",
            }
            or not isinstance(saved_boundary, dict)
            or {
                key: value
                for key, value in saved_boundary.items()
                if key != "integration_line_um"
            }
            != expected_boundary
            or not isinstance(integration_line, list)
            or len(integration_line) != 2
            or any(
                not math.isclose(float(actual), wanted, abs_tol=1e-9)
                for actual_point, expected_point in zip(
                    integration_line, port.integration_line_um, strict=True
                )
                for actual, wanted in zip(actual_point, expected_point, strict=True)
            )
        ):
            raise RuntimeError("completed receipt modal native evidence mismatch")


def _validate_terminal_native_evidence(ports: Any, spec: HfssDrivenSpec) -> None:
    if not isinstance(ports, list) or len(ports) != 2:
        raise RuntimeError("completed receipt has invalid terminal ports")
    for record, port in zip(ports, spec.ports, strict=True):
        if not isinstance(record, dict) or "unresolved_native" in record:
            raise RuntimeError(
                "completed receipt has unresolved terminal native evidence"
            )
        terminal_name = record.get("terminal_excitation")
        if (
            record.get("index") != port.index
            or record.get("boundary") != port.name
            or not isinstance(terminal_name, str)
            or not terminal_name
            or not isinstance(record.get("face_id"), int)
            or not isinstance(record.get("face_center_um"), list)
            or len(record["face_center_um"]) != 3
        ):
            raise RuntimeError("completed receipt terminal identity is invalid")
        requested = record.get("requested")
        if requested != {
            "reference_objects": list(port.reference_objects),
            "renormalize": False,
            "deembed_um": port.deembed_um,
        }:
            raise RuntimeError("completed receipt terminal request does not match spec")
        native = record.get("native")
        if not isinstance(native, dict):
            raise TypeError("completed receipt has no native terminal evidence")
        if (
            native.get("excitation_names")
            != [item.name for item in spec.ports[: port.index]]
            or native.get("terminal_names") != [terminal_name]
            or native.get("reference_conductors") != list(port.reference_objects)
            or not isinstance(native.get("reference_conductor_ids"), list)
            or not native["reference_conductor_ids"]
            or not all(
                isinstance(value, int) for value in native["reference_conductor_ids"]
            )
        ):
            raise RuntimeError("completed receipt terminal native identity mismatch")
        boundary = native.get("boundary_properties")
        terminal = native.get("terminal_properties")
        if boundary != {
            "Deembed": True,
            "Deembed Dist": f"{port.deembed_um:g}um",
            "Name": port.name,
            "Num Terminals": "1",
            "Renorm All Terminals": False,
            "Type": "Wave Port",
            "Wave Port Type": "Terminal",
        } or terminal != {
            "Name": terminal_name,
            "Port Name": port.name,
            "Terminal Renormalizing Impedance": "50ohm",
            "Type": "Terminal",
        }:
            raise RuntimeError("completed receipt terminal native property mismatch")


def _validate_diagnostics(root: Path, diagnostics: Any) -> None:
    if not isinstance(diagnostics, dict) or diagnostics.get("batch_log") != "batch.log":
        raise RuntimeError("completed receipt diagnostics are invalid")
    batch_log = _contained(root, "batch.log")
    if (
        diagnostics.get("present") is not True
        or not batch_log.is_file()
        or diagnostics.get("sha256") != file_sha256(batch_log)
        or not isinstance(diagnostics.get("physics_warnings"), list)
    ):
        raise RuntimeError("completed receipt diagnostics hash is invalid")


def _verified(
    root: Path, relative: Any, hashes: dict[str, Any], hash_key: str | None = None
) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError("receipt has an invalid relative output path")
    path = _contained(root, relative)
    expected = hashes.get(hash_key or relative)
    if (
        not isinstance(expected, str)
        or not path.is_file()
        or file_sha256(path) != expected
    ):
        raise RuntimeError(f"receipt output hash mismatch: {relative}")
    return path


__all__ = ["ResolvedRun", "resolve_results"]
