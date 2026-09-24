"""Deterministic SCGSim source provenance for AEDT preparation and execution."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal

from .util import file_sha256

RECEIPT_V1 = "scgsim.aedt.receipt.v1"
RECEIPT_V2 = "scgsim.aedt.receipt.v2"
RECEIPT_V3 = "scgsim.aedt.receipt.v3"
SOURCE_SCHEMA_V1 = "scgsim.aedt.runtime-source.v1"
SOURCE_SCHEMA_V2 = "scgsim.aedt.runtime-source.v2"
SOURCE_SCHEMA = SOURCE_SCHEMA_V2

# Frozen membership of the public runtime-source.v1 evidence format. Changing
# producer structure must not silently change what historical readers mean.
_RUNTIME_SOURCE_V1_MODULES = (
    "_hfss_convergence.py",
    "_hfss_runtime.py",
    "_matrix_export.py",
    "_native_common.py",
    "_q2d_convergence.py",
    "_q2d_runtime.py",
    "_q3d_runtime.py",
    "_runtime_provenance.py",
    "handoff.py",
    "resolve.py",
    "run.py",
    "spec.py",
    "util.py",
)
_RUNTIME_SOURCE_V1_PATHS = tuple(
    f"scgsim/aedt/{name}" for name in sorted(_RUNTIME_SOURCE_V1_MODULES)
)

_RUNTIME_SOURCE_V2_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V1_PATHS,
            "scgsim/aedt/_epr_eigenmode.py",
            "scgsim/aedt/_epr_fields.py",
            "scgsim/aedt/_epr_geometry.py",
            "scgsim/aedt/_epr_models.py",
            "scgsim/aedt/_epr_results.py",
            "scgsim/semantics/route_a.py",
            "scgsim/sgb/planning.py",
        )
    )
)

def _module_manifest() -> list[dict[str, str]]:
    package_root = Path(__file__).resolve().parents[1]
    return [
        {
            "module": path.removesuffix(".py").replace("/", "."),
            "path": path,
            "sha256": file_sha256(package_root.parent / path),
        }
        for path in _RUNTIME_SOURCE_V2_PATHS
    ]


def _content_digest(modules: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        modules, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observed_revision() -> str | None:
    try:
        direct_url = distribution("scgsim").read_text("direct_url.json")
    except PackageNotFoundError:
        direct_url = None
    revision = ""
    if direct_url is not None:
        try:
            revision = str(
                json.loads(direct_url).get("vcs_info", {}).get("commit_id", "")
            )
        except (TypeError, ValueError):
            revision = ""
    if not revision:
        source_root = next(
            (
                parent
                for parent in Path(__file__).resolve().parents
                if (parent / ".git").exists() and (parent / "pyproject.toml").is_file()
            ),
            None,
        )
        if source_root is not None:
            try:
                completed = subprocess.run(
                    ["git", "-C", str(source_root), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError):
                revision = ""
            else:
                revision = completed.stdout.strip()
    return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else None


def prepared_runtime_source() -> dict[str, Any]:
    """Return complete module hashes plus an optional truthful Git observation."""
    modules = _module_manifest()
    revision = _observed_revision()
    return {
        "schema_version": SOURCE_SCHEMA,
        "stage": "prepared",
        "modules": modules,
        "content_sha256": _content_digest(modules),
        "revision_observation": (
            {"status": "available", "revision": revision}
            if revision is not None
            else {"status": "unavailable"}
        ),
    }


def runtime_source_identity() -> dict[str, Any]:
    """Bind execution to complete module bytes and the legacy required revision."""
    modules = _module_manifest()
    revision = _observed_revision()
    if revision is None:
        raise RuntimeError("SCGSim runtime source revision is unavailable")
    by_path = {item["path"]: item["sha256"] for item in modules}
    return {
        "revision": revision,
        "run_py_sha256": by_path["scgsim/aedt/run.py"],
        "spec_py_sha256": by_path["scgsim/aedt/spec.py"],
        "hfss_convergence_py_sha256": by_path["scgsim/aedt/_hfss_convergence.py"],
        "q2d_convergence_py_sha256": by_path["scgsim/aedt/_q2d_convergence.py"],
        "q3d_convergence_py_sha256": by_path["scgsim/aedt/_q2d_convergence.py"],
        "matrix_export_py_sha256": by_path["scgsim/aedt/_matrix_export.py"],
        "schema_version": SOURCE_SCHEMA,
        "stage": "actual",
        "modules": modules,
        "content_sha256": _content_digest(modules),
    }


def validate_runtime_source(
    value: Any, *, stage: Literal["prepared", "actual"]
) -> None:
    """Validate detached provenance shape without comparing it to installed bytes."""
    if not isinstance(value, dict):
        raise RuntimeError(f"{stage} runtime source provenance is invalid")
    modules = value.get("modules")
    schema = value.get("schema_version")
    expected_paths = {
        SOURCE_SCHEMA_V1: _RUNTIME_SOURCE_V1_PATHS,
        SOURCE_SCHEMA_V2: _RUNTIME_SOURCE_V2_PATHS,
    }.get(schema)
    if (
        expected_paths is None
        or value.get("stage") != stage
        or not isinstance(modules, list)
        or len(modules) != len(expected_paths)
    ):
        raise RuntimeError(f"{stage} runtime source provenance is invalid")
    paths: list[str] = []
    for item in modules:
        expected_module = (
            item["path"].removesuffix(".py").replace("/", ".")
            if isinstance(item, dict) and isinstance(item.get("path"), str)
            else None
        )
        if (
            not isinstance(item, dict)
            or set(item) != {"module", "path", "sha256"}
            or item["module"] != expected_module
            or not isinstance(item["path"], str)
            or not item["path"].startswith("scgsim/")
            or Path(item["path"]).is_absolute()
            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
        ):
            raise RuntimeError(f"{stage} runtime source module manifest is invalid")
        paths.append(item["path"])
    if tuple(paths) != expected_paths:
        raise RuntimeError(f"{stage} runtime source module manifest is invalid")
    if value.get("content_sha256") != _content_digest(modules):
        raise RuntimeError(f"{stage} runtime source content digest is invalid")
    if stage == "prepared":
        if set(value) != {
            "schema_version",
            "stage",
            "modules",
            "content_sha256",
            "revision_observation",
        }:
            raise RuntimeError("prepared runtime source provenance is invalid")
        observation = value.get("revision_observation")
        if not isinstance(observation, dict):
            raise RuntimeError(
                "prepared runtime source revision observation is invalid"
            )
        if observation == {"status": "unavailable"}:
            return
        if (
            set(observation) != {"status", "revision"}
            or observation.get("status") != "available"
            or not re.fullmatch(r"[0-9a-f]{40}", str(observation.get("revision", "")))
        ):
            raise RuntimeError(
                "prepared runtime source revision observation is invalid"
            )
    else:
        required = {
            "revision",
            "run_py_sha256",
            "spec_py_sha256",
            "hfss_convergence_py_sha256",
            "q2d_convergence_py_sha256",
            "q3d_convergence_py_sha256",
            "matrix_export_py_sha256",
        }
        if set(value) != {
            "schema_version",
            "stage",
            "modules",
            "content_sha256",
            *required,
        } or any(
            not re.fullmatch(
                r"[0-9a-f]{40}" if key == "revision" else r"[0-9a-f]{64}",
                str(value.get(key, "")),
            )
            for key in required
        ):
            raise RuntimeError("actual runtime source legacy identity is invalid")
        by_path = {item["path"]: item["sha256"] for item in modules}
        expected_legacy = {
            "run_py_sha256": by_path["scgsim/aedt/run.py"],
            "spec_py_sha256": by_path["scgsim/aedt/spec.py"],
            "hfss_convergence_py_sha256": by_path["scgsim/aedt/_hfss_convergence.py"],
            "q2d_convergence_py_sha256": by_path["scgsim/aedt/_q2d_convergence.py"],
            # Preserve the existing legacy key's historical source-file binding.
            "q3d_convergence_py_sha256": by_path["scgsim/aedt/_q2d_convergence.py"],
            "matrix_export_py_sha256": by_path["scgsim/aedt/_matrix_export.py"],
        }
        if any(value[key] != digest for key, digest in expected_legacy.items()):
            raise RuntimeError("actual runtime source legacy identity is invalid")


def initial_receipt_payload(
    *,
    schema_version: str,
    mode: Any,
    requested: Any,
    pdk_materials: Any,
    vacuum_material_id: Any,
    source: Any,
    outputs: Any,
    prepared_at_utc: Any,
    prepared_runtime_source_value: Any = None,
) -> dict[str, Any]:
    """Build the canonical field order for an initial v1, v2, or v3 receipt."""
    if schema_version not in {RECEIPT_V1, RECEIPT_V2, RECEIPT_V3}:
        raise ValueError("initial receipt schema is unsupported")
    if outputs != {}:
        raise ValueError("initial receipt outputs must be empty")
    result = {
        "schema_version": schema_version,
        "status": "not_run",
        "mode": mode,
        "requested": requested,
        "pdk_materials": pdk_materials,
        "vacuum_material_id": vacuum_material_id,
        "source": source,
    }
    if schema_version in {RECEIPT_V2, RECEIPT_V3}:
        if prepared_runtime_source_value is None:
            version = schema_version.rsplit(".", 1)[-1]
            raise ValueError(
                f"{version} initial receipt requires prepared runtime source"
            )
        result["prepared_runtime_source"] = prepared_runtime_source_value
    elif prepared_runtime_source_value is not None:
        raise ValueError("v1 initial receipt excludes prepared runtime source")
    result["outputs"] = outputs
    result["prepared_at_utc"] = prepared_at_utc
    return result


def encode_initial_receipt(value: Mapping[str, Any]) -> bytes:
    """Encode exact historical initial-receipt bytes with fixed indentation."""
    canonical = initial_receipt_payload(
        schema_version=value.get("schema_version"),
        mode=value.get("mode"),
        requested=value.get("requested"),
        pdk_materials=value.get("pdk_materials"),
        vacuum_material_id=value.get("vacuum_material_id"),
        source=value.get("source"),
        prepared_runtime_source_value=value.get("prepared_runtime_source"),
        outputs=value.get("outputs"),
        prepared_at_utc=value.get("prepared_at_utc"),
    )
    if dict(value) != canonical:
        raise ValueError("initial receipt members are not canonical")
    return (json.dumps(canonical, indent=2) + "\n").encode("utf-8")


def initial_receipt_sha256(value: Mapping[str, Any]) -> str:
    """Hash the exact versioned initial-receipt codec bytes."""
    return hashlib.sha256(encode_initial_receipt(value)).hexdigest()


__all__ = [
    "RECEIPT_V1",
    "RECEIPT_V2",
    "prepared_runtime_source",
    "runtime_source_identity",
    "validate_runtime_source",
]
