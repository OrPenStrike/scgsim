"""Saved-solution and EPR artifact verification/readback."""

from __future__ import annotations


import hashlib

import json

from collections.abc import Mapping

from typing import Any

from pathlib import Path

from scgsim.aedt._io import file_sha256, read_json, write_json

from scgsim.aedt.epr.models import EprResult, SavedSolution, detached

from scgsim.aedt.epr.selection import _exact_mapping


def resolve_saved_solution(manifest_path: str | Path) -> SavedSolution:
    """Verify one explicitly sealed saved-field cohort without opening AEDT."""

    requested_manifest = Path(manifest_path).expanduser().absolute()
    manifest_source = requested_manifest.resolve()
    if (
        requested_manifest != manifest_source
        or not manifest_source.is_file()
        or manifest_source.is_symlink()
    ):
        raise FileNotFoundError(
            f"saved-solution manifest is missing: {manifest_source}"
        )
    base = manifest_source.parent
    manifest = _exact_mapping(read_json(manifest_source), "saved-solution manifest")
    expected = {
        "schema_version",
        "project",
        "results",
        "members",
        "content_sha256",
        "identity",
        "receipt",
        "receipt_sha256",
    }
    if (
        set(manifest) != expected
        or manifest.get("schema_version") != "scgsim.aedt.saved-solution-manifest.v1"
    ):
        raise ValueError("saved-solution manifest members are not canonical")
    payload = {
        "schema_version": "scgsim.aedt.saved-solution.v1",
        "project": manifest["project"],
        "results": manifest["results"],
        "members": manifest["members"],
        "content_sha256": manifest["content_sha256"],
        "identity": manifest["identity"],
    }
    saved = SavedSolution.from_payload(base, payload)
    if saved.project_path.is_symlink() or saved.result_path.is_symlink():
        raise RuntimeError("saved-solution project/results cannot be symlinks")
    if (
        saved.project_path.resolve() != saved.project_path
        or saved.result_path.resolve() != saved.result_path
    ):
        raise RuntimeError("saved-solution project/results traverse a symlink")
    if not saved.project_path.is_file() or not saved.result_path.is_dir():
        raise FileNotFoundError("saved-solution project or result directory is missing")
    receipt_relative = _safe_relative(manifest["receipt"], "receipt")
    receipt = base / receipt_relative
    if (
        not receipt.is_file()
        or receipt.is_symlink()
        or file_sha256(receipt) != manifest["receipt_sha256"]
    ):
        raise RuntimeError("saved-solution receipt hash mismatch")
    receipt_payload = _exact_mapping(read_json(receipt), "saved-solution receipt")
    if (
        receipt_payload.get("status") != "completed"
        or receipt_payload.get("saved_fields") is not True
    ):
        raise RuntimeError("saved-solution receipt does not attest completed fields")
    if receipt_payload.get("identity") != detached(saved.identity):
        raise RuntimeError("saved-solution receipt identity mismatch")
    observed_members: list[dict[str, Any]] = []
    declared_paths: set[str] = set()
    for member in saved.members:
        item = _exact_mapping(member, "saved-solution member")
        if set(item) != {"path", "bytes", "sha256"}:
            raise ValueError("saved-solution member is not canonical")
        relative = _safe_relative(item["path"], "saved-solution member.path")
        path = base / relative
        if relative.as_posix() in declared_paths:
            raise ValueError("saved-solution member paths must be unique")
        declared_paths.add(relative.as_posix())
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve() != path
            or path.stat().st_size != item["bytes"]
            or file_sha256(path) != item["sha256"]
        ):
            raise RuntimeError(f"saved-solution member mismatch: {relative}")
        observed_members.append(dict(item))
    actual_paths = {saved.project_path.relative_to(base).as_posix()}
    for path in saved.result_path.rglob("*"):
        if path.is_symlink() or path.resolve() != path:
            raise RuntimeError(f"saved solution contains a symlink: {path}")
        if path.is_file():
            actual_paths.add(path.relative_to(base).as_posix())
    if declared_paths != actual_paths:
        raise RuntimeError(
            "saved-solution manifest does not cover the exact result inventory"
        )
    digest = hashlib.sha256(
        json.dumps(observed_members, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if digest != saved.content_sha256:
        raise RuntimeError("saved-solution content digest mismatch")
    return saved


def seal_saved_solution(
    run_dir: str | Path,
    *,
    project_name: str,
    design_name: str,
    setup_name: str,
    model_source_sha256: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a released project and its complete saved-result inventory."""

    root = Path(run_dir).resolve()
    project = root / f"{project_name}.aedt"
    results = root / f"{project_name}.aedtresults"
    if project.is_symlink() or project.resolve() != project or not project.is_file():
        raise RuntimeError("released saved-solution project is unavailable")
    if results.is_symlink() or results.resolve() != results or not results.is_dir():
        raise RuntimeError("released saved-solution result directory is unavailable")
    evidence = _exact_mapping(evidence, "saved-field evidence")
    expected_evidence = {
        "saved_fields",
        "solver_last_completed_pass",
        "saved_fields_pass",
        "fields_solution",
        "physical_variation",
    }
    if set(evidence) != expected_evidence or evidence.get("saved_fields") is not True:
        raise ValueError("saved-field evidence is not canonical")
    for name in ("solver_last_completed_pass", "saved_fields_pass"):
        if type(evidence.get(name)) is not int or evidence[name] <= 0:
            raise ValueError(f"saved-field evidence {name} is invalid")
    if (
        not isinstance(evidence.get("fields_solution"), str)
        or not evidence["fields_solution"]
    ):
        raise ValueError("saved-field evidence solution identity is invalid")
    if not isinstance(evidence.get("physical_variation"), str):
        raise TypeError("saved-field physical variation must be text")

    inventory = [project]
    for path in sorted(results.rglob("*")):
        if path.is_symlink() or path.resolve() != path:
            raise RuntimeError(f"saved solution contains a symlink: {path}")
        if path.is_file():
            inventory.append(path)
    if len(inventory) == 1:
        raise RuntimeError("saved-solution result inventory is empty")
    members = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in inventory
    ]
    content_sha256 = hashlib.sha256(
        json.dumps(members, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = {
        "model_source_sha256": model_source_sha256,
        "project_name": project_name,
        "design_name": design_name,
        "setup_name": setup_name,
        "physical_variation": evidence["physical_variation"],
        "solver_last_completed_pass": evidence["solver_last_completed_pass"],
        "saved_fields_pass": evidence["saved_fields_pass"],
        "saved_fields": True,
    }
    receipt_path = root / "metadata/saved-solution-receipt.json"
    manifest_path = root / "saved-solution-manifest.json"
    receipt_payload = {
        "schema_version": "scgsim.aedt.saved-solution-receipt.v1",
        "status": "completed",
        "saved_fields": True,
        "identity": identity,
        "fields_solution": evidence["fields_solution"],
        "content_sha256": content_sha256,
    }
    write_json(receipt_path, receipt_payload)
    manifest_payload = {
        "schema_version": "scgsim.aedt.saved-solution-manifest.v1",
        "project": project.relative_to(root).as_posix(),
        "results": results.relative_to(root).as_posix(),
        "members": members,
        "content_sha256": content_sha256,
        "identity": identity,
        "receipt": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": file_sha256(receipt_path),
    }
    write_json(manifest_path, manifest_payload)
    return {
        "status": "complete",
        "manifest": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": file_sha256(manifest_path),
        "receipt": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": file_sha256(receipt_path),
        "content_sha256": content_sha256,
        "identity": identity,
        "member_count": len(members),
    }


def _safe_relative(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise ValueError(f"{name} must be a contained relative path")
    return relative


def resolve_epr_result(path: str | Path) -> EprResult:
    """Load one strict offline EPR result without filling incomplete rows."""

    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"EPR result is missing: {source}")
    return EprResult.from_payload(read_json(source))
