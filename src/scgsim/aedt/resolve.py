"""Receipt-bound resolution of one completed HFSS CPW handoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .spec import HfssDrivenSpec
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
    if mode != spec.mode or mode not in {"terminal", "modal"}:
        raise RuntimeError("completed receipt has an invalid mode")
    _validate_readback(receipt, spec)
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise TypeError("completed receipt has no output hash manifest")
    project_relative = f"{spec.project_name}.aedt"
    if receipt.get("project") != project_relative:
        raise RuntimeError("completed receipt project path is not canonical")
    project = _verified(root, project_relative, outputs, project_relative)
    if save["project_sha256"] != outputs.get(project_relative):
        raise RuntimeError("saved project hash does not match output manifest")
    if mode == "terminal":
        expected = {
            project_relative,
            "results/terminal/terminal_st.csv",
            "results/terminal/terminal.s2p",
        }
        if set(outputs) != expected:
            raise RuntimeError("terminal output manifest is not canonical")
        return ResolvedRun(
            "terminal",
            project,
            _verified(root, "results/terminal/terminal_st.csv", outputs),
            _verified(root, "results/terminal/terminal.s2p", outputs),
            None,
            receipt_path,
        )
    expected = {
        project_relative,
        "results/modal/port_zo.csv",
        "results/modal/port_zo_provenance.json",
    }
    if set(outputs) != expected:
        raise RuntimeError("modal output manifest is not canonical")
    return ResolvedRun(
        "modal",
        project,
        _verified(root, "results/modal/port_zo.csv", outputs),
        None,
        _verified(root, "results/modal/port_zo_provenance.json", outputs),
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


def _validate_readback(receipt: dict[str, Any], spec: HfssDrivenSpec) -> None:
    connected = receipt.get("connected")
    if not isinstance(connected, dict) or connected != {
        "aedt_version": spec.aedt_version,
        "pyaedt_version": spec.pyaedt_version,
    }:
        raise RuntimeError("completed receipt has invalid connected version identity")
    readback = receipt.get("result_readback")
    key = "terminal_st" if spec.mode == "terminal" else "port_zo"
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
    terminal_order = (
        [record.get("terminal_excitation") for record in ports]
        if isinstance(ports, list)
        and len(ports) == 2
        and all(isinstance(record, dict) for record in ports)
        and [record.get("index") for record in ports] == [1, 2]
        else None
    )
    if spec.mode == "terminal" and (
        not isinstance(touchstone, dict)
        or touchstone.get("path") != "terminal.s2p"
        or touchstone.get("ports") != 2
        or touchstone.get("records") != 20_000
        or touchstone.get("frequency_unit") != "GHz"
        or touchstone.get("first_frequency_ghz") != spec.run_control.sweep.start_ghz
        or touchstone.get("last_frequency_ghz") != spec.run_control.sweep.stop_ghz
        or touchstone.get("strictly_increasing") is not True
        or touchstone.get("port_order") != terminal_order
        or not all(isinstance(name, str) and name for name in terminal_order or [])
        or len(set(terminal_order or [])) != 2
        or not isinstance(touchstone.get("bytes"), int)
        or touchstone["bytes"] <= 0
    ):
        raise RuntimeError(
            "completed receipt is missing a two-port Touchstone readback"
        )


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
