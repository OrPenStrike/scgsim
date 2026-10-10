"""Versioned AEDT runtime source provenance and receipt codecs."""

from __future__ import annotations


import hashlib

import json

import re

import subprocess

from collections.abc import Mapping

from importlib.metadata import PackageNotFoundError, distribution

from pathlib import Path

from typing import Any, Literal

from scgsim.aedt._io import file_sha256


RECEIPT_V1 = "scgsim.aedt.receipt.v1"

RECEIPT_V2 = "scgsim.aedt.receipt.v2"

RECEIPT_V3 = "scgsim.aedt.receipt.v3"

SOURCE_SCHEMA_V1 = "scgsim.aedt.runtime-source.v1"

SOURCE_SCHEMA_V2 = "scgsim.aedt.runtime-source.v2"

SOURCE_SCHEMA_V3 = "scgsim.aedt.runtime-source.v3"

SOURCE_SCHEMA_V4 = "scgsim.aedt.runtime-source.v4"

SOURCE_SCHEMA_V5 = "scgsim.aedt.runtime-source.v5"

SOURCE_SCHEMA_V6 = "scgsim.aedt.runtime-source.v6"

SOURCE_SCHEMA_V7 = "scgsim.aedt.runtime-source.v7"

SOURCE_SCHEMA_V8 = "scgsim.aedt.runtime-source.v8"

SOURCE_SCHEMA_V9 = "scgsim.aedt.runtime-source.v9"

SOURCE_SCHEMA_V10 = "scgsim.aedt.runtime-source.v10"

SOURCE_SCHEMA_V11 = "scgsim.aedt.runtime-source.v11"

SOURCE_SCHEMA_V12 = "scgsim.aedt.runtime-source.v12"

SOURCE_SCHEMA_V13 = "scgsim.aedt.runtime-source.v13"

SOURCE_SCHEMA_V15 = "scgsim.aedt.runtime-source.v15"

SOURCE_SCHEMA_V16 = "scgsim.aedt.runtime-source.v16"

SOURCE_SCHEMA_V17 = "scgsim.aedt.runtime-source.v17"

SOURCE_SCHEMA = SOURCE_SCHEMA_V17

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

_RUNTIME_SOURCE_V3_PATHS = tuple(
    sorted((*_RUNTIME_SOURCE_V2_PATHS, "scgsim/aedt/_benchmark.py"))
)

_RUNTIME_SOURCE_V4_PATHS = tuple(
    sorted((*_RUNTIME_SOURCE_V3_PATHS, "scgsim/aedt/_presentation.py"))
)

_RUNTIME_SOURCE_V5_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V4_PATHS,
            "scgsim/sgb/geometry_plan.py",
            "scgsim/sgb/__init__.py",
        )
    )
)

_RUNTIME_SOURCE_V6_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V5_PATHS,
            "scgsim/sgb/summary.py",
        )
    )
)

_RUNTIME_SOURCE_V7_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V6_PATHS,
            "scgsim/_notebook_presentation.py",
        )
    )
)

_RUNTIME_SOURCE_V8_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V7_PATHS,
            "scgsim/aedt/_junction_partition.py",
        )
    )
)

_RUNTIME_SOURCE_V9_PATHS = tuple(
    sorted(
        (
            *_RUNTIME_SOURCE_V8_PATHS,
            "scgsim/aedt/__init__.py",
            "scgsim/aedt/_q3d_geometry.py",
            "scgsim/sgb/adapter.py",
            "scgsim/sgb/stack.py",
        )
    )
)

_RUNTIME_SOURCE_V10_PATHS = tuple(
    sorted((*_RUNTIME_SOURCE_V9_PATHS, "scgsim/aedt/_handoff_cohort.py"))
)

_RUNTIME_SOURCE_V11_PATHS = tuple(
    sorted((*_RUNTIME_SOURCE_V10_PATHS, "scgsim/aedt/_q3d_bodies.py"))
)

_RUNTIME_SOURCE_V12_PATHS = _RUNTIME_SOURCE_V11_PATHS

_RUNTIME_SOURCE_V13_PATHS = (
    "src/scgsim/aedt/__init__.py",
    "src/scgsim/aedt/_io.py",
    "src/scgsim/aedt/eigenmode.py",
    "src/scgsim/aedt/epr/analysis.py",
    "src/scgsim/aedt/epr/cache.py",
    "src/scgsim/aedt/epr/fields.py",
    "src/scgsim/aedt/epr/geometry.py",
    "src/scgsim/aedt/epr/junction_partition.py",
    "src/scgsim/aedt/epr/models.py",
    "src/scgsim/aedt/epr/native.py",
    "src/scgsim/aedt/epr/selection.py",
    "src/scgsim/aedt/epr/workflow.py",
    "src/scgsim/aedt/preparation/cohort.py",
    "src/scgsim/aedt/preparation/geometry.py",
    "src/scgsim/aedt/preparation/handoff.py",
    "src/scgsim/aedt/preparation/q3d_geometry.py",
    "src/scgsim/aedt/presentation/benchmark.py",
    "src/scgsim/aedt/presentation/epr.py",
    "src/scgsim/aedt/presentation/runs.py",
    "src/scgsim/aedt/results/benchmark.py",
    "src/scgsim/aedt/results/convergence/common.py",
    "src/scgsim/aedt/results/convergence/hfss.py",
    "src/scgsim/aedt/results/convergence/q2d.py",
    "src/scgsim/aedt/results/convergence/q3d.py",
    "src/scgsim/aedt/results/epr.py",
    "src/scgsim/aedt/results/matrices.py",
    "src/scgsim/aedt/results/provenance.py",
    "src/scgsim/aedt/results/resolve.py",
    "src/scgsim/aedt/run.py",
    "src/scgsim/aedt/runtime/benchmark.py",
    "src/scgsim/aedt/runtime/families/hfss.py",
    "src/scgsim/aedt/runtime/families/q2d.py",
    "src/scgsim/aedt/runtime/families/q3d.py",
    "src/scgsim/aedt/runtime/native/common.py",
    "src/scgsim/aedt/runtime/native/q3d_bodies.py",
    "src/scgsim/aedt/runtime/transaction.py",
    "src/scgsim/aedt/specs/common.py",
    "src/scgsim/aedt/specs/hfss.py",
    "src/scgsim/aedt/specs/parse.py",
    "src/scgsim/aedt/specs/q2d.py",
    "src/scgsim/aedt/specs/q3d.py",
    "src/scgsim/geometry/__init__.py",
    "src/scgsim/geometry/_primitives/entities.py",
    "src/scgsim/geometry/_primitives/loops.py",
    "src/scgsim/geometry/_primitives/spatial.py",
    "src/scgsim/geometry/compiler/dispatch.py",
    "src/scgsim/geometry/compiler/validation.py",
    "src/scgsim/geometry/planning/domain.py",
    "src/scgsim/geometry/planning/interfaces.py",
    "src/scgsim/geometry/planning/ports.py",
    "src/scgsim/geometry/planning/surfaces.py",
    "src/scgsim/geometry/planning/tags.py",
    "src/scgsim/geometry/planning/topology.py",
    "src/scgsim/geometry/planning/volumes.py",
    "src/scgsim/geometry/source/adapter.py",
    "src/scgsim/geometry/source/plan.py",
    "src/scgsim/geometry/source/stack.py",
    "src/scgsim/geometry/source/summary.py",
    "src/scgsim/geometry/source/validation.py",
    "src/scgsim/presentation/notebook.py",
    "src/scgsim/semantics/route_a.py",
)


_RUNTIME_SOURCE_V15_PATHS = tuple(sorted((*_RUNTIME_SOURCE_V13_PATHS,
    "src/scgsim/aedt/specs/modeling.py",
    "src/scgsim/geometry/_primitives/geometry_refs.py",
    "src/scgsim/geometry/_primitives/surface_records.py",
    "src/scgsim/geometry/planning/evidence.py",
    "src/scgsim/geometry/source/intents.py",
    "src/scgsim/geometry/source/_normalization.py",
)))

_RUNTIME_SOURCE_V16_PATHS = tuple(sorted((
    *_RUNTIME_SOURCE_V15_PATHS,
    "src/scgsim/aedt/runtime/native/hfss_eigenmode.py",
)))

_RUNTIME_SOURCE_V17_PATHS = tuple(sorted((
    *_RUNTIME_SOURCE_V16_PATHS,
    "src/scgsim/geometry/models/lumped.py",
    "src/scgsim/aedt/specs/__init__.py",
    "src/scgsim/aedt/runtime/native/hfss_lumped.py",
)))

def _module_manifest() -> list[dict[str, str]]:
    source_root = Path(__file__).resolve().parents[3]
    return [
        {
            "module": path.removeprefix("src/").removesuffix(".py").replace("/", "."),
            "path": path,
            "sha256": file_sha256(source_root / path.removeprefix("src/")),
        }
        for path in _RUNTIME_SOURCE_V17_PATHS
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
    """Bind execution to measured module bytes and a truthful Git observation."""
    modules = _module_manifest()
    revision = _observed_revision()
    return {
        "schema_version": SOURCE_SCHEMA,
        "stage": "actual",
        "modules": modules,
        "content_sha256": _content_digest(modules),
        "revision_observation": (
            {"status": "available", "revision": revision}
            if revision is not None
            else {"status": "unavailable"}
        ),
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
        SOURCE_SCHEMA_V3: _RUNTIME_SOURCE_V3_PATHS,
        SOURCE_SCHEMA_V4: _RUNTIME_SOURCE_V4_PATHS,
        SOURCE_SCHEMA_V5: _RUNTIME_SOURCE_V5_PATHS,
        SOURCE_SCHEMA_V6: _RUNTIME_SOURCE_V6_PATHS,
        SOURCE_SCHEMA_V7: _RUNTIME_SOURCE_V7_PATHS,
        SOURCE_SCHEMA_V8: _RUNTIME_SOURCE_V8_PATHS,
        SOURCE_SCHEMA_V9: _RUNTIME_SOURCE_V9_PATHS,
        SOURCE_SCHEMA_V10: _RUNTIME_SOURCE_V10_PATHS,
        SOURCE_SCHEMA_V11: _RUNTIME_SOURCE_V11_PATHS,
        SOURCE_SCHEMA_V12: _RUNTIME_SOURCE_V12_PATHS,
        SOURCE_SCHEMA_V13: _RUNTIME_SOURCE_V13_PATHS,
        SOURCE_SCHEMA_V15: _RUNTIME_SOURCE_V15_PATHS,
        SOURCE_SCHEMA_V16: _RUNTIME_SOURCE_V16_PATHS,
        SOURCE_SCHEMA_V17: _RUNTIME_SOURCE_V17_PATHS,
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
            item["path"].removeprefix("src/").removesuffix(".py").replace("/", ".")
            if isinstance(item, dict) and isinstance(item.get("path"), str)
            else None
        )
        if (
            not isinstance(item, dict)
            or set(item) != {"module", "path", "sha256"}
            or item["module"] != expected_module
            or not isinstance(item["path"], str)
            or not item["path"].startswith(
                "src/scgsim/" if schema in {SOURCE_SCHEMA_V13, SOURCE_SCHEMA_V15, SOURCE_SCHEMA_V16, SOURCE_SCHEMA_V17} else "scgsim/"
            )
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
        _validate_revision_observation(value.get("revision_observation"), stage=stage)
    elif schema in {SOURCE_SCHEMA_V13, SOURCE_SCHEMA_V15, SOURCE_SCHEMA_V16, SOURCE_SCHEMA_V17}:
        if set(value) != {
            "schema_version",
            "stage",
            "modules",
            "content_sha256",
            "revision_observation",
        }:
            raise RuntimeError("actual runtime source provenance is invalid")
        _validate_revision_observation(value.get("revision_observation"), stage=stage)
    else:
        required_hashes = {
            "run_py_sha256",
            "spec_py_sha256",
            "hfss_convergence_py_sha256",
            "q2d_convergence_py_sha256",
            "q3d_convergence_py_sha256",
            "matrix_export_py_sha256",
        }
        current = schema == SOURCE_SCHEMA_V12
        revision_field = "revision_observation" if current else "revision"
        if set(value) != {
            "schema_version",
            "stage",
            "modules",
            "content_sha256",
            revision_field,
            *required_hashes,
        } or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value.get(key, "")))
            for key in required_hashes
        ):
            raise RuntimeError("actual runtime source legacy identity is invalid")
        if current:
            _validate_revision_observation(
                value.get("revision_observation"), stage=stage
            )
        elif not re.fullmatch(r"[0-9a-f]{40}", str(value.get("revision", ""))):
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


def _validate_revision_observation(value: Any, *, stage: str) -> None:
    """Unavailable VCS provenance is distinct from missing measured code bytes."""
    if not isinstance(value, dict):
        raise RuntimeError(f"{stage} runtime source revision observation is invalid")
    if value == {"status": "unavailable"}:
        return
    if (
        set(value) != {"status", "revision"}
        or value.get("status") != "available"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("revision", "")))
    ):
        raise RuntimeError(f"{stage} runtime source revision observation is invalid")


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
