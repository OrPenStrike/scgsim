"""Receipt-bound native AEDT simulation profile, separate from physics results."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from .util import file_sha256, read_json, write_json

BENCHMARK_RELATIVE = "results/simulation-benchmark.v1.json"
BENCHMARK_SCHEMA = "scgsim.aedt.simulation-benchmark.v1"

_METRIC_LABELS = {
    "Real time": "real_seconds",
    "Elapsed Time": "elapsed_seconds",
    "Elapsed time": "elapsed_seconds",
    "Cpu time": "cpu_seconds",
}
_COUNT_LABELS = {
    "Tetrahedra": "tetrahedra",
    "Solved Elements": "solved_elements",
    "Elements": "elements",
    "Matrix size": "linear_matrix_size",
}


def _seconds(value: Any, label: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"native profile {label} is not formatted time")
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})(?:\.(\d+))?", value)
    if match is None or int(match[2]) > 59 or int(match[3]) > 59:
        raise ValueError(f"native profile {label} has invalid time {value!r}")
    fractional = float(f"0.{match[4]}") if match[4] else 0.0
    return int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3]) + fractional


def _metrics(properties: Any) -> dict[str, Any]:
    if not isinstance(properties, dict):
        raise TypeError("native profile node has no properties")
    result: dict[str, Any] = {}
    for label, key in _METRIC_LABELS.items():
        if label in properties:
            value = _seconds(properties[label], label)
            if key in result and result[key] != value:
                raise ValueError(f"native profile has conflicting {key} values")
            result[key] = value
            result[f"{key}_native"] = properties[label]
    if "Memory" in properties:
        from ansys.aedt.core.modules.profile import MemoryGB

        native = properties["Memory"]
        if not isinstance(native, str):
            raise ValueError("native profile Memory is not a formatted quantity")
        sdk_input = native[:-2] + "G" if native.endswith(" GB") else native
        value = MemoryGB(sdk_input).value
        if not math.isfinite(value) or value < 0:
            raise ValueError("native profile Memory is invalid")
        result["memory_native"] = native
        result["memory_gb_sdk"] = value
        if sdk_input != native:
            result["memory_sdk_input"] = sdk_input
            result["memory_unit_alias"] = "native GB is canonicalized to SDK G (decimal gigabytes)"
        result["memory_scope"] = "individual task executable peak, not whole host"
    for label, key in _COUNT_LABELS.items():
        if label in properties:
            native = properties[label]
            value = int(native) if isinstance(native, str) and native.isdecimal() else native
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"native profile {label} is invalid")
            result[key] = value
            result[f"{key}_native_label"] = label
            result[f"{key}_native"] = native
    return result


def _node(name: str, native: Any) -> dict[str, Any]:
    properties = {
        str(label): native.GetPropValue(label)
        for label in native.GetPropNames()
    }
    result = {"name": str(name), "metrics": _metrics(properties)}
    match = re.fullmatch(r"Adaptive Pass (\d+)(?: Group)?", str(name))
    if match:
        result["adaptive_pass"] = int(match[1])
    identity = {
        label: properties[label]
        for label in ("Info", "Type", "Frequency", "Status", "Start Time", "Stop Time")
        if label in properties and isinstance(properties[label], (str, int, float, bool))
    }
    if identity:
        result["native_identity"] = identity
    result["native_properties"] = {
        str(label): value
        for label, value in properties.items()
        if isinstance(value, (str, int, float, bool))
    }
    children = [
        _node(str(child_name), native.GetChildObject(child_name))
        for child_name in native.GetChildNames()
    ]
    if children:
        result["children"] = children
    if children:
        stages = []
        pending = list(children)
        while pending:
            child = pending.pop()
            if "memory_gb_sdk" in child["metrics"]:
                stages.append(child)
            pending.extend(child.get("children", []))
        if stages:
            peak = max(stages, key=lambda child: child["metrics"]["memory_gb_sdk"])
            result["stage_memory_peak"] = {
                "stage": peak["name"],
                "memory_native": peak["metrics"]["memory_native"],
                "memory_gb_sdk": peak["metrics"]["memory_gb_sdk"],
                "scope": "maximum available descendant stage task memory, not host peak",
            }
    return result


def _native_profile(app: Any, setup_name: str) -> dict[str, Any]:
    try:
        root = app.odesign.GetChildObject("results/profile")
    except Exception:
        # Native children distinguish an absent profile from a failed read.
        root_names = [str(name).lower() for name in app.odesign.GetChildNames()]
        if "results/profile" not in root_names and "results" not in root_names:
            return {"status": "unavailable", "reason": "native results/profile tree is absent", "profiles": []}
        if "results" in root_names:
            results = app.odesign.GetChildObject("results")
            if "profile" not in [str(name).lower() for name in results.GetChildNames()]:
                return {"status": "unavailable", "reason": "native results/profile tree is absent", "profiles": []}
        raise
    names = [str(item) for item in root.GetChildNames()]
    selected = [name for name in names if name == setup_name or name.startswith(f"{setup_name} ")]
    if not selected:
        return {"status": "unavailable", "reason": "native setup profile is absent", "profiles": []}
    profiles = []
    for name in selected:
        setup_object = root.GetChildObject(name)
        setup_properties = {
            str(label): setup_object.GetPropValue(label)
            for label in setup_object.GetPropNames()
        }
        groups = [
            _node(str(group_name), setup_object.GetChildObject(group_name))
            for group_name in setup_object.GetChildNames()
        ]
        if not groups:
            raise RuntimeError(f"native setup profile has no process groups: {name!r}")
        profiles.append({
            "native_setup_profile": name,
            "native_setup_properties": {
                str(label): value for label, value in setup_properties.items()
                if isinstance(value, (str, int, float, bool))
            },
            "process_groups": groups,
        })
    return {"status": "complete", "profiles": profiles}


def export_simulation_benchmark(app: Any, run_dir: Path, setup_name: str) -> tuple[str, str, dict[str, str]]:
    """Capture one native profile; profile failure is data, artifact failure is fatal."""
    payload: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA,
        "setup_name": setup_name,
        "native_source": "AEDT results/profile native OO tree GetChildNames/GetPropNames/GetPropValue",
        "memory_conversion": "PyAEDT 1.3.0 MemoryGB.value; native GB spelling normalized to SDK G, raw retained",
    }
    try:
        payload["design_name"] = app.design_name
        payload.update(_native_profile(app, setup_name))
    except Exception as exc:  # Profile failure must not change the physics solve result.
        payload.update(status="error", reason=f"{type(exc).__name__}: {exc}", profiles=[])
    path = run_dir / BENCHMARK_RELATIVE
    write_json(path, payload)
    return BENCHMARK_RELATIVE, file_sha256(path), {
        "status": payload["status"], "path": BENCHMARK_RELATIVE
    }


def attach_simulation_benchmark(
    result: dict[str, Any], app: Any, run_dir: Path, setup_name: str
) -> dict[str, Any]:
    relative, digest, summary = export_simulation_benchmark(app, run_dir, setup_name)
    result["outputs"][relative] = digest
    result["benchmark"] = summary
    return result


def read_simulation_benchmark(path: Path, setup_name: str) -> dict[str, Any]:
    """Read a previously hash-verified benchmark, never opening AEDT."""
    payload = read_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != BENCHMARK_SCHEMA
        or payload.get("setup_name") != setup_name
        or payload.get("status") not in {"complete", "unavailable", "error"}
        or not isinstance(payload.get("profiles"), list)
    ):
        raise RuntimeError("claimed AEDT benchmark artifact is invalid")
    if payload["status"] == "complete":
        if not payload["profiles"] or not all(
            isinstance(profile, dict)
            and isinstance(profile.get("native_setup_profile"), str)
            and isinstance(profile.get("process_groups"), list)
            and profile["process_groups"]
            for profile in payload["profiles"]
        ):
            raise RuntimeError("claimed AEDT benchmark profile is incomplete")
        def check_node(node: Any) -> None:
            if (
                not isinstance(node, dict)
                or not isinstance(node.get("name"), str)
                or not isinstance(node.get("metrics"), dict)
                or not isinstance(node.get("native_properties"), dict)
            ):
                raise RuntimeError("claimed AEDT benchmark node is invalid")
            for key, value in node["metrics"].items():
                if key.endswith("_seconds") or key == "memory_gb_sdk":
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise RuntimeError("claimed AEDT benchmark metric is invalid")
            if "children" in node:
                if not isinstance(node["children"], list):
                    raise RuntimeError("claimed AEDT benchmark children are invalid")
                for child in node["children"]:
                    check_node(child)
        for profile in payload["profiles"]:
            for group in profile["process_groups"]:
                check_node(group)
    elif not isinstance(payload.get("reason"), str) or not payload["reason"] or payload["profiles"]:
        raise RuntimeError("claimed AEDT benchmark unavailability is invalid")
    return payload
