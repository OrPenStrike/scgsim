"""Receipt-bound resolution of one completed HFSS CPW handoff."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .spec import HfssDrivenSpec, ModalPort
from .util import file_sha256, read_json


@dataclass(frozen=True)
class ResolvedRun:
    """Canonical result paths verified against the completed run receipt."""

    mode: Literal["terminal", "modal"]
    project_path: Path
    primary_csv: Path
    touchstone_path: Path | None
    provenance_path: Path | None
    receipt_path: Path


def resolve_results(run_dir: str | Path) -> ResolvedRun:
    """Return results only when a complete exact receipt proves every artifact."""
    root = Path(run_dir).resolve()
    receipt_path = _contained(root, "metadata/hfss_run_receipt.json")
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
    if (
        not isinstance(source, dict)
        or source.get("spec") != "hfss_driven_spec.json"
        or source.get("gds") != "geometry/design.gds"
    ):
        raise RuntimeError("receipt source paths are not canonical")
    spec_path = _verified(root, "hfss_driven_spec.json", source, "spec_sha256")
    _verified(root, "geometry/design.gds", source, "gds_sha256")
    spec = HfssDrivenSpec.from_payload(read_json(spec_path), base_dir=root)
    mode = receipt.get("mode")
    if mode not in {"terminal", "modal"} or spec.mode != mode:
        raise RuntimeError("AEDT driven receipt mode is invalid")
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
    )


def _contained(root: Path, relative: str) -> Path:
    requested = Path(relative)
    if requested.is_absolute():
        raise RuntimeError(f"receipt path escapes handoff root: {relative!r}")
    resolved = (root / requested).resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"receipt path escapes handoff root: {relative!r}")
    return resolved


def _validate_readback(
    root: Path, receipt: dict[str, Any], spec: HfssDrivenSpec
) -> None:
    connected = receipt.get("connected")
    if not isinstance(connected, dict) or connected != {
        "aedt_version": spec.aedt_version,
        "pyaedt_version": spec.pyaedt_version,
    }:
        raise RuntimeError("completed receipt has invalid connected version identity")
    readback = receipt.get("result_readback")
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
    _validate_diagnostics(root, receipt.get("diagnostics"))


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
