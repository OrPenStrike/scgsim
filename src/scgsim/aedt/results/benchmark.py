"""Offline benchmark artifact reader."""

from __future__ import annotations


import math

from pathlib import Path

from typing import Any

from scgsim.aedt._io import read_json

from scgsim.aedt.runtime.benchmark import BENCHMARK_SCHEMA


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
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                    ):
                        raise RuntimeError("claimed AEDT benchmark metric is invalid")
            if "children" in node:
                if not isinstance(node["children"], list):
                    raise RuntimeError("claimed AEDT benchmark children are invalid")
                for child in node["children"]:
                    check_node(child)

        for profile in payload["profiles"]:
            for group in profile["process_groups"]:
                check_node(group)
    elif (
        not isinstance(payload.get("reason"), str)
        or not payload["reason"]
        or payload["profiles"]
    ):
        raise RuntimeError("claimed AEDT benchmark unavailability is invalid")
    return payload
