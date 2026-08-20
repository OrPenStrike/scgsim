"""Resolver for completed manual Palace runs.

Ownership: SCGSim Development.
Failure intent: fail closed for malformed or inconsistent run artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._archive_layout import logical_tar_member_name, resolve_run_archive_path

_PROBLEM_FAMILIES = {
    "Electrostatic": (
        "terminal-C",
        "terminal-Cm",
        "terminal-Cinv",
        "terminal-V",
        "domain-E",
        "surface-Q",
        "error-indicators",
    ),
    "Eigenmode": (
        "eig",
        "port-EPR",
        "port-I",
        "port-V",
        "domain-E",
        "surface-Q",
        "error-indicators",
    ),
}

_REQUIRED_FILES = (
    "metadata/palace_handoff_metadata.json",
    "metadata/palace_run_metadata.json",
    "metadata/palace_resource_record.json",
    "metadata/palace_handoff_archive_manifest.json",
    "metadata/palace_index_map.json",
    "metadata/mesh_manifest.json",
    "metadata/palace_returned_run_receipt.json",
    "config.json",
    "palace.msh",
    "results/palace/palace.json",
)

_REQUIRED_HASH_ENTRIES = (
    "palace.msh",
    "config.json",
    "metadata/mesh_manifest.json",
    "metadata/palace_index_map.json",
)

_RETURNED_RECEIPT_SCHEMA = "palace-returned-run-receipt.v1"
_RETURNED_RECEIPT_FILE = "metadata/palace_returned_run_receipt.json"


@dataclass(frozen=True)
class ParsedTable:
    """Single parsed CSV artifact with typed row payload."""

    name: str
    path: Path
    headers: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PalacePerformance:
    """Canonical elapsed count/duration timing payload."""

    counts: dict[str, int | float]
    durations: dict[str, float]


@dataclass(frozen=True)
class PalaceCost:
    """Conventional solver resource/cost metadata."""

    problem_degrees_of_freedom: int | None
    mesh_elements: int | None
    mpi_size: int | None
    openmp_threads: int | None
    peak_memory_megabytes: dict[str, float | None]
    peak_node_memory_megabytes: dict[str, float | None]
    linear_solver: dict[str, float | int]
    git_tag: str | None


@dataclass(frozen=True)
class PalaceReturnedReceipt:
    """Normalized payload from returned receipt writer."""

    schema: str
    schema_version: int
    handoff_id: str
    status: str
    exit_code: int
    solver_exit_code: int
    tee_exit_code: int
    route: str
    problem: str
    identity_verified: bool
    timestamp_utc: str
    job_identity: dict[str, Any]
    input_hashes: tuple[dict[str, Any], ...]
    output_files: tuple[dict[str, Any], ...]
    log: dict[str, Any] | None
    solver_identity: dict[str, Any]


@dataclass(frozen=True)
class PalaceProvenance:
    """Resolved provenance payloads used to claim run identity."""

    handoff_metadata: dict[str, Any]
    run_metadata: dict[str, Any]
    resource_record: dict[str, Any]
    handoff_archive_manifest: dict[str, Any]
    index_map: dict[str, Any]
    mesh_manifest: dict[str, Any]
    config: dict[str, Any]
    palace_json: dict[str, Any]
    returned_receipt: dict[str, Any]


@dataclass(frozen=True)
class ResolvedPalaceResult:
    """Complete resolved view of one Palace run directory."""

    run_dir: Path
    problem: str
    route: str
    has_returned_outputs: bool
    status: str
    performance: PalacePerformance
    cost: PalaceCost
    tables: dict[str, ParsedTable]
    returned_receipt: PalaceReturnedReceipt
    provenance: PalaceProvenance

    def show_run_trustworthiness(self, *, theme: str = "light"):
        from .report import _show_run_trustworthiness

        return _show_run_trustworthiness(self, theme=theme)

    def show_all_results(
        self, *, theme: str = "light", ranking_limit: int | None = 20
    ) -> None:
        from .report import _show_all_results

        _show_all_results(self, theme=theme, ranking_limit=ranking_limit)

    def show_physics_quantities(
        self, *, theme: str = "light", ranking_limit: int | None = 20
    ):
        from .report import _show_physics_quantities

        return _show_physics_quantities(self, theme=theme, ranking_limit=ranking_limit)

    def show_simulation_benchmark(self) -> dict[str, dict[str, Any]]:
        from .report import _show_simulation_benchmark

        return _show_simulation_benchmark(self)


def resolve_palace_result(
    run_dir: str | Path, *, expected_handoff_id: str
) -> ResolvedPalaceResult:
    """Validate and resolve a completed Palace manual-handoff run directory."""

    if not isinstance(expected_handoff_id, str) or not expected_handoff_id:
        raise ValueError("expected_handoff_id must be a non-empty string.")

    root = Path(run_dir).expanduser().resolve()
    for relative in _REQUIRED_FILES:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"required artifact missing: {path}")

    handoff_metadata = _read_json(root / "metadata" / "palace_handoff_metadata.json")
    run_metadata = _read_json(root / "metadata" / "palace_run_metadata.json")
    resource_record = _read_json(root / "metadata" / "palace_resource_record.json")
    archive_manifest = _read_json(
        root / "metadata" / "palace_handoff_archive_manifest.json"
    )
    index_map = _read_json(root / "metadata/palace_index_map.json")
    mesh_manifest = _read_json(root / "metadata/mesh_manifest.json")
    config = _read_json(root / "config.json")
    palace_payload = _read_json(root / "results/palace/palace.json")
    receipt_payload = _read_json(root / _RETURNED_RECEIPT_FILE)

    route = _expect_scalar(handoff_metadata, "route", str)
    problem = _expect_scalar(handoff_metadata, "problem", str)

    if route not in {"A", "B"}:
        raise ValueError(f"invalid route {route!r}; expected 'A' or 'B'.")
    if problem not in _PROBLEM_FAMILIES:
        raise ValueError(
            f"unsupported problem {problem!r}; expected one of {tuple(_PROBLEM_FAMILIES)}."
        )

    if run_metadata.get("kind") != "manual_handoff":
        raise ValueError("run metadata does not identify manual_handoff kind.")
    if handoff_metadata.get("kind") != "manual_handoff":
        raise ValueError("handoff metadata does not identify manual_handoff kind.")
    _validate_contract_schema(handoff_metadata, "handoff metadata")
    _validate_contract_schema(run_metadata, "run metadata")
    _validate_contract_schema(resource_record, "resource record")

    if _expect_scalar(run_metadata, "status", str, fallback="") == "":
        raise ValueError("run metadata status must be a non-empty string.")
    if _expect_scalar(resource_record, "status", str, fallback="") == "":
        raise ValueError("resource record status must be a non-empty string.")

    _validate_scalar_identity(handoff_metadata, run_metadata)
    _validate_execution_identity(root, handoff_metadata)
    _validate_handoff_id(handoff_metadata, expected_handoff_id)
    _validate_resource_identity(resource_record, handoff_metadata)

    _validate_receipt_payload(
        receipt_payload=receipt_payload,
        expected_handoff_id=expected_handoff_id,
        handoff_metadata=handoff_metadata,
        run_metadata=run_metadata,
        problem=problem,
        route=route,
        root=root,
    )
    _validate_returned_state_agreement(
        receipt_payload=receipt_payload,
        run_metadata=run_metadata,
        resource_record=resource_record,
    )
    _validate_hashes(root, handoff_metadata, "handoff")
    _validate_hashes(root, run_metadata, "run")
    _validate_archive_record(root, resource_record)
    _validate_archive_manifest(
        root, archive_manifest, resource_record, handoff_metadata
    )
    index_counts = _index_counts(index_map)
    _validate_route_consistency(
        route,
        run_metadata,
        handoff_metadata,
        mesh_manifest,
        index_map,
        index_counts,
        problem,
    )
    _validate_config_problem(config, problem)
    _validate_config_index_correspondence(config, index_map, index_counts, problem)

    tables = _discover_results(root, problem, index_counts, config)

    returned_receipt = _build_returned_receipt(receipt_payload)
    cost = _build_cost(palace_payload)
    performance = _build_performance(palace_payload)

    return ResolvedPalaceResult(
        run_dir=root,
        problem=problem,
        route=route,
        has_returned_outputs=True,
        status=_expect_scalar(run_metadata, "status", str),
        performance=performance,
        cost=cost,
        tables=tables,
        returned_receipt=returned_receipt,
        provenance=PalaceProvenance(
            handoff_metadata=handoff_metadata,
            run_metadata=run_metadata,
            resource_record=resource_record,
            handoff_archive_manifest=archive_manifest,
            index_map=index_map,
            mesh_manifest=mesh_manifest,
            config=config,
            palace_json=palace_payload,
            returned_receipt=receipt_payload,
        ),
    )


def _discover_results(
    root: Path,
    problem: str,
    index_counts: dict[str, int],
    config: dict[str, Any],
) -> dict[str, ParsedTable]:
    results_root = root / "results" / "palace"
    discovered: dict[str, ParsedTable] = {}
    for family in _PROBLEM_FAMILIES[problem]:
        path = results_root / f"{family}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"required result family is missing: {path}")
        table = _read_csv_table(path)
        _validate_table(problem, table, index_counts, config)
        discovered[family] = table
    return discovered


def _build_performance(palace_payload: dict[str, Any]) -> PalacePerformance:
    elapsed = palace_payload.get("ElapsedTime")
    if not isinstance(elapsed, dict):
        raise TypeError("palace.json missing ElapsedTime block.")

    counts_raw = elapsed.get("Counts")
    durations_raw = elapsed.get("Durations")
    if not isinstance(counts_raw, dict):
        raise TypeError("palace.json ElapsedTime.Counts must be a mapping.")
    if not isinstance(durations_raw, dict):
        raise TypeError("palace.json ElapsedTime.Durations must be a mapping.")

    counts: dict[str, int | float] = {
        str(key): float(value)
        for key, value in counts_raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    durations: dict[str, float] = {
        str(key): float(value)
        for key, value in durations_raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not counts or not durations:
        raise ValueError(
            "palace.json performance payload is missing numeric timing data."
        )
    if any(not math.isfinite(value) or value < 0 for value in counts.values()) or any(
        not math.isfinite(value) or value < 0 for value in durations.values()
    ):
        raise ValueError("palace.json timing payload contains invalid numeric data.")
    if (
        "Total" not in counts
        or "Total" not in durations
        or counts["Total"] < 1
        or durations["Total"] < 0
    ):
        raise ValueError("palace.json timing payload must include non-negative Total.")
    return PalacePerformance(counts=counts, durations=durations)


def _build_cost(palace_payload: dict[str, Any]) -> PalaceCost:
    problem_block = palace_payload.get("Problem")
    if not isinstance(problem_block, dict):
        raise TypeError("palace.json missing Problem block.")

    linear_solver = palace_payload.get("LinearSolver")
    if not isinstance(linear_solver, dict):
        linear_solver = {}

    cost = PalaceCost(
        problem_degrees_of_freedom=_expect_int(problem_block.get("DegreesOfFreedom")),
        mesh_elements=_expect_int(problem_block.get("MeshElements")),
        mpi_size=_expect_int(problem_block.get("MPISize")),
        openmp_threads=_expect_int(problem_block.get("OpenMPThreads")),
        peak_memory_megabytes=_as_float_map(
            palace_payload.get("PeakMemoryMegabytes"),
            ("Average", "Max", "Min", "Total"),
        ),
        peak_node_memory_megabytes=_as_float_map(
            palace_payload.get("PeakNodeMemoryMegabytes"),
            ("Average", "Max", "Min", "Total"),
        ),
        linear_solver={
            key: _to_float(value)
            for key, value in linear_solver.items()
            if isinstance(linear_solver, dict)
            and isinstance(key, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        },
        git_tag=(
            palace_payload["GitTag"]
            if isinstance(palace_payload.get("GitTag"), str)
            else None
        ),
    )
    if any(
        value is None
        for value in (
            cost.problem_degrees_of_freedom,
            cost.mesh_elements,
            cost.mpi_size,
            cost.openmp_threads,
            cost.git_tag,
        )
    ):
        raise ValueError("palace.json is missing required solver cost metadata.")
    if any(
        value is None or value <= 0
        for value in (
            cost.problem_degrees_of_freedom,
            cost.mesh_elements,
            cost.mpi_size,
            cost.openmp_threads,
        )
    ):
        raise ValueError("palace.json has invalid solver cost metadata.")
    if (
        not cost.linear_solver
        or any(
            value is None or not math.isfinite(value) or value < 0
            for value in cost.peak_memory_megabytes.values()
        )
        or any(
            value is None or not math.isfinite(value) or value < 0
            for value in cost.peak_node_memory_megabytes.values()
        )
        or any(
            not math.isfinite(value) or value < 0
            for value in cost.linear_solver.values()
        )
    ):
        raise ValueError("palace.json is missing required resource cost metadata.")
    return cost


def _validate_scalar_identity(
    handoff_metadata: dict[str, Any], run_metadata: dict[str, Any]
) -> None:
    handoff_identity = handoff_metadata.get("source_revisions")
    run_identity = run_metadata.get("source_revisions")
    if not isinstance(handoff_identity, dict) or not handoff_identity:
        raise ValueError("handoff source_revisions must be a non-empty mapping.")
    if not isinstance(run_identity, dict) or not run_identity:
        raise ValueError("run source_revisions must be a non-empty mapping.")

    _validate_palace_identity(
        _expect_mapping(
            handoff_metadata.get("palace_identity"), "handoff_metadata.palace_identity"
        )
    )
    _validate_palace_identity(
        _expect_mapping(
            run_metadata.get("palace_identity"), "run_metadata.palace_identity"
        )
    )

    if handoff_identity != run_identity:
        raise ValueError(
            "manual handoff identity mismatch between handoff and run metadata."
        )

    handoff_palace_identity = handoff_metadata.get("palace_identity")
    run_palace_identity = run_metadata.get("palace_identity")
    if handoff_palace_identity != run_palace_identity:
        raise ValueError("palace identity mismatch between handoff and run metadata.")

    handoff_id = handoff_metadata.get("handoff_id")
    run_id = run_metadata.get("handoff_id")
    if not isinstance(handoff_id, str) or not handoff_id:
        raise ValueError("handoff metadata must include handoff_id.")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run metadata must include handoff_id.")
    if handoff_id != run_id:
        raise ValueError("handoff id mismatch between handoff and run metadata.")


def _validate_handoff_id(
    handoff_metadata: dict[str, Any], expected_handoff_id: str
) -> None:
    """Recompute the identity; colocated copied IDs are not evidence."""

    computed = _compute_handoff_id(
        profile=_expect_scalar(handoff_metadata, "profile", str),
        route=_expect_scalar(handoff_metadata, "route", str),
        problem=_expect_scalar(handoff_metadata, "problem", str),
        source_revisions=_expect_mapping(
            handoff_metadata.get("source_revisions"), "handoff source_revisions"
        ),
        palace_identity=_expect_mapping(
            handoff_metadata.get("palace_identity"), "handoff palace_identity"
        ),
        hashes=_expect_list(handoff_metadata.get("hashes"), "handoff metadata hashes"),
        execution_identity=_expect_mapping(
            handoff_metadata.get("execution_identity"), "handoff execution_identity"
        ),
    )
    recorded = _expect_scalar(handoff_metadata, "handoff_id", str)
    if computed != recorded or computed != expected_handoff_id:
        raise ValueError(
            "recomputed handoff identity does not match expected handoff_id."
        )


def _validate_execution_identity(root: Path, handoff_metadata: dict[str, Any]) -> None:
    execution = _expect_mapping(
        handoff_metadata.get("execution_identity"), "handoff execution_identity"
    )
    if execution != {
        "profile": _expect_scalar(handoff_metadata, "profile", str),
        "resources": _expect_mapping(
            handoff_metadata.get("requested_resources"), "handoff requested_resources"
        ),
        "setup_commands": _expect_list(
            handoff_metadata.get("setup_commands"),
            "handoff setup_commands",
            allow_empty=True,
        ),
        "petsc_options": _expect_list(
            handoff_metadata.get("petsc_options"),
            "handoff petsc_options",
            allow_empty=True,
        ),
        "command": _expect_mapping(handoff_metadata.get("command"), "handoff command"),
        "script": execution.get("script"),
        "recorder": execution.get("recorder"),
    }:
        raise ValueError("handoff execution identity does not match handoff metadata.")
    for field, relative in (
        ("script", _expect_scalar(handoff_metadata, "script", str)),
        (
            "recorder",
            _expect_scalar(handoff_metadata, "returned_receipt_recorder", str),
        ),
    ):
        item = _expect_mapping(execution.get(field), f"execution identity {field}")
        if _expect_scalar(item, "path", str) != relative:
            raise ValueError(
                f"handoff execution identity {field} path is inconsistent."
            )
        path = _confined_path(root, relative)
        if not path.is_file() or _expect_scalar(item, "sha256", str) != _sha256(path):
            raise ValueError(
                f"handoff execution identity {field} hash is inconsistent."
            )


def _validate_resource_identity(
    resource_record: dict[str, Any], handoff_metadata: dict[str, Any]
) -> None:
    for field in (
        "profile",
        "route",
        "problem",
        "handoff_id",
        "source_revisions",
        "palace_identity",
        "execution_identity",
        "requested_resources",
        "resolved_resources",
    ):
        if resource_record.get(field) != handoff_metadata.get(field):
            raise ValueError(
                f"resource record {field} does not match handoff metadata."
            )
    if _expect_scalar(resource_record, "status", str) != "completed":
        raise ValueError("resource record status must be completed for a returned run.")
    if _expect_int(resource_record.get("exit_code")) != 0:
        raise ValueError("resource record exit_code must be zero for a returned run.")


def _compute_handoff_id(
    *,
    profile: str,
    route: str,
    problem: str,
    source_revisions: Mapping[str, Any],
    palace_identity: Mapping[str, Any],
    hashes: list[Any],
    execution_identity: Mapping[str, Any],
) -> str:
    payload = {
        "route": route,
        "problem": problem,
        "profile": profile,
        "source_revisions": source_revisions,
        "palace_identity": palace_identity,
        "hashes": hashes,
        "execution_identity": execution_identity,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_hashes(root: Path, metadata: dict[str, Any], label: str) -> None:
    hashes = _expect_list(metadata.get("hashes"), f"{label} metadata hashes")
    if not hashes:
        raise ValueError(f"{label} metadata has no hash entries.")

    observed: dict[str, tuple[int, str]] = {}
    for entry in hashes:
        if not isinstance(entry, dict):
            raise TypeError(f"{label} metadata hashes must be mapping entries.")
        path_name = _expect_scalar(entry, "path", str)
        bytes_value = _expect_int(entry.get("bytes"))
        sha_value = _expect_scalar(entry, "sha256", str)
        file_path = _confined_path(root, path_name)
        if not file_path.is_file():
            raise FileNotFoundError(
                f"{label} metadata references missing artifact {path_name}."
            )

        observed[path_name] = (
            _expect_int(file_path.stat().st_size),
            _sha256(file_path),
        )
        if bytes_value != observed[path_name][0]:
            raise ValueError(
                f"{label} hash byte size mismatch for {path_name}: "
                f"declared {bytes_value}, observed {observed[path_name][0]}"
            )
        if sha_value != observed[path_name][1]:
            raise ValueError(f"{label} hash mismatch for {path_name}.")

    missing = [name for name in _REQUIRED_HASH_ENTRIES if name not in observed]
    if missing:
        raise ValueError(
            f"{label} metadata missing hash entries for: {', '.join(missing)}"
        )


def _validate_archive_record(root: Path, resource_record: dict[str, Any]) -> None:
    if resource_record.get("kind") != "manual_handoff":
        raise ValueError("resource record kind must be manual_handoff.")

    if _expect_scalar(resource_record, "status", str, fallback="") == "":
        raise ValueError("resource record status must be a non-empty string.")
    requested_resources = _optional_dict(resource_record.get("requested_resources"))
    resolved_resources = _optional_dict(resource_record.get("resolved_resources"))
    if not requested_resources:
        raise ValueError("resource record requested_resources must be a mapping.")
    if not resolved_resources:
        raise ValueError("resource record resolved_resources must be a mapping.")

    archive = _optional_dict(resource_record.get("archive"))
    if not archive:
        raise ValueError("resource record archive must be a non-empty mapping.")
    path_name = _expect_scalar(archive, "path", str)
    expected_sha = _expect_scalar(archive, "sha256", str)
    archive_path = resolve_run_archive_path(root, path_name)
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"resource record references missing archive payload {archive_path}"
        )
    if _sha256(archive_path) != expected_sha:
        raise ValueError("resource record archive hash mismatch.")


def _validate_archive_manifest(
    root: Path,
    manifest: dict[str, Any],
    resource_record: dict[str, Any],
    handoff_metadata: dict[str, Any],
) -> None:
    if _expect_scalar(manifest, "schema_version", int, fallback=0) != 1:
        raise ValueError("handoff archive manifest schema_version must be integer 1.")
    archive = _expect_mapping(resource_record.get("archive"), "resource_record.archive")
    if _extract_hash_map(
        _expect_list(manifest.get("input_hashes"), "archive manifest input_hashes")
    ) != _extract_hash_map(
        _expect_list(handoff_metadata.get("hashes"), "handoff metadata hashes")
    ):
        raise ValueError("archive manifest input hashes do not match handoff metadata.")
    if _expect_scalar(manifest, "archive", str) != _expect_scalar(archive, "path", str):
        raise ValueError(
            "handoff archive manifest archive path does not match receipt."
        )
    members = _expect_list(manifest.get("members"), "handoff archive manifest members")
    member_paths: set[str] = set()
    member_types: dict[str, str] = {}
    file_members: dict[str, tuple[int, str]] = {}
    for entry in members:
        data = _expect_mapping(entry, "handoff archive manifest member")
        path = _expect_scalar(data, "path", str)
        if not path or path in member_paths:
            raise ValueError("handoff archive manifest members must have unique paths.")
        member_paths.add(path)
        member_type = _expect_scalar(data, "type", str)
        if member_type not in {"file", "directory"}:
            raise ValueError("handoff archive manifest member type is invalid.")
        member_types[path] = member_type
        if member_type == "file":
            bytes_value = _expect_scalar(data, "bytes", int)
            sha_value = _expect_scalar(data, "sha256", str)
            file_members[path] = (bytes_value, sha_value)
    required = {
        "config.json",
        "palace.msh",
        "metadata/mesh_manifest.json",
        "metadata/palace_index_map.json",
        "metadata/palace_handoff_metadata.json",
        "metadata/palace_run_metadata.json",
    }
    if missing := required - member_paths:
        raise ValueError(
            "handoff archive manifest lacks required members: "
            + ", ".join(sorted(missing))
        )
    if non_files := required - set(file_members):
        raise ValueError(
            "handoff archive manifest requires regular-file members: "
            + ", ".join(sorted(non_files))
        )
    manifest_input_hashes = _extract_hash_map(
        _expect_list(manifest.get("input_hashes"), "archive manifest input_hashes")
    )
    handoff_input_hashes = _extract_hash_map(
        _expect_list(handoff_metadata.get("hashes"), "handoff metadata hashes")
    )
    for path, expected in handoff_input_hashes.items():
        if manifest_input_hashes.get(path) != expected:
            raise ValueError(f"archive manifest input hash mismatch for {path}.")
        if file_members.get(path) != expected:
            raise ValueError(
                f"archive manifest must declare input {path} as an exact regular file."
            )
    archive_path = resolve_run_archive_path(root, _expect_scalar(archive, "path", str))
    with tarfile.open(archive_path, "r:gz") as bundle:
        bundled: dict[str, tarfile.TarInfo] = {}
        for member in bundle.getmembers():
            logical = logical_tar_member_name(member.name, root.name)
            if logical is None:
                continue
            bundled[logical.rstrip("/")] = member
        allowed_extra = {
            path.rstrip("/")
            for path in _expect_list(
                manifest.get("archive_members_excluded_from_self_listing"),
                "archive manifest excluded self members",
            )
        }
        manifest_member_paths = {path.rstrip("/") for path in member_paths}
        if set(bundled) != manifest_member_paths | allowed_extra:
            raise ValueError("archive members do not match the archive manifest.")
        for path, member_type in member_types.items():
            member = bundled.get(path.rstrip("/"))
            if member is None:
                raise ValueError(f"archive member is missing: {path}.")
            if member_type == "file" and not member.isfile():
                raise ValueError(f"archive member must be a regular file: {path}.")
            if member_type == "directory" and not member.isdir():
                raise ValueError(f"archive member must be a directory: {path}.")
        for path, (expected_bytes, expected_sha) in file_members.items():
            member = bundled.get(path)
            if member is None or not member.isfile() or member.size != expected_bytes:
                raise ValueError(f"archive member size mismatch for {path}.")
            handle = bundle.extractfile(member)
            if handle is None:
                raise ValueError(f"archive member {path} cannot be read.")
            digest = hashlib.sha256(handle.read()).hexdigest()
            if digest != expected_sha:
                raise ValueError(f"archive member hash mismatch for {path}.")


def _validate_route_consistency(
    route: str,
    run_metadata: dict[str, Any],
    handoff_metadata: dict[str, Any],
    mesh_manifest: dict[str, Any],
    index_map: dict[str, Any],
    index_counts: dict[str, int],
    problem: str,
) -> None:
    if _expect_scalar(mesh_manifest, "schema_version", int, fallback=0) != 1:
        raise ValueError("mesh manifest schema_version must be integer 1.")
    if _expect_scalar(mesh_manifest, "route", str, fallback="") != route:
        raise ValueError("mesh manifest route does not match resolved route.")

    run_route = _expect_scalar(run_metadata, "route", str)
    handoff_route = _expect_scalar(handoff_metadata, "route", str)
    if run_route != route or handoff_route != route:
        raise ValueError("metadata route does not match resolved route.")

    if _expect_scalar(index_map, "schema_version", int, fallback=0) != 1:
        raise ValueError("palace index map schema_version must be integer 1.")
    _validate_index_entries(index_map)
    entries = _expect_list(index_map.get("entries"), "palace_index_map.entries")
    for entry in entries:
        candidate = _expect_scalar(
            _optional_dict(entry).get("metadata", {}),
            "route",
            str,
            fallback="",
        )
        if not candidate:
            candidate = _expect_scalar(entry, "route", str, fallback="")
        if candidate in {"A", "B"} and candidate != route:
            raise ValueError("index map route is inconsistent with run route.")

    run_problem = _expect_scalar(run_metadata, "problem", str)
    handoff_problem = _expect_scalar(handoff_metadata, "problem", str)
    if run_problem != problem or handoff_problem != problem:
        raise ValueError("metadata problem does not match resolved problem.")


def _validate_config_index_correspondence(
    config: dict[str, Any],
    index_map: dict[str, Any],
    index_counts: dict[str, int],
    problem: str,
) -> None:
    boundaries = _optional_dict(config.get("Boundaries")) or {}
    if problem == "Electrostatic":
        terminals = _expect_list(
            boundaries.get("Terminal"), "config.Boundaries.Terminal"
        )
        if index_counts["terminal"] != len(terminals):
            raise ValueError("terminal count mismatch between config and index map.")
    else:
        ports = _expect_list(
            boundaries.get("LumpedPort"), "config.Boundaries.LumpedPort"
        )
        if index_counts["lumped_port"] != len(ports):
            raise ValueError("lumped port count mismatch between config and index map.")

    postprocessing = _optional_dict(boundaries.get("Postprocessing")) or {}
    post_diel = _expect_list(
        postprocessing.get("Dielectric"),
        "config.Boundaries.Postprocessing.Dielectric",
        allow_empty=True,
    )
    if post_diel and index_counts["surface"] < len(post_diel):
        raise ValueError(
            "surface count in index map does not cover postprocessing list."
        )
    expected_sections = {
        "Boundaries.Terminal": terminals if problem == "Electrostatic" else [],
        "Boundaries.LumpedPort": ports if problem == "Eigenmode" else [],
        "Domains.Postprocessing.Energy": _expect_list(
            _expect_mapping(
                _expect_mapping(config.get("Domains"), "config.Domains").get(
                    "Postprocessing"
                ),
                "config.Domains.Postprocessing",
            ).get("Energy"),
            "config.Domains.Postprocessing.Energy",
        ),
        "Boundaries.Postprocessing.Dielectric": post_diel,
    }
    entries = _expect_list(index_map.get("entries"), "palace_index_map.entries")
    for section, configured in expected_sections.items():
        indexed = sorted(
            (
                _expect_mapping(entry, "palace index entry")
                for entry in entries
                if _expect_scalar(entry, "section", str) == section
            ),
            key=lambda entry: _expect_scalar(entry, "index", int),
        )
        if len(indexed) != len(configured):
            raise ValueError(f"config/index entry count mismatch for {section}.")
        for position, (entry, configured_item) in enumerate(
            zip(indexed, configured, strict=True), start=1
        ):
            item = _expect_mapping(configured_item, f"config {section} item")
            if _expect_scalar(entry, "index", int) != _expect_scalar(
                item, "Index", int
            ):
                raise ValueError(
                    f"config/index Index mismatch for {section} {position}."
                )
            if _expect_list(
                entry.get("attributes"), "index attributes"
            ) != _expect_list(item.get("Attributes"), "config Attributes"):
                raise ValueError(
                    f"config/index Attributes mismatch for {section} {position}."
                )


def _validate_config_problem(config: dict[str, Any], problem: str) -> None:
    problem_block = _optional_dict(config.get("Problem"))
    if not problem_block:
        raise ValueError("config.json missing Problem block.")
    config_problem = _expect_scalar(problem_block, "Type", str, fallback="")
    if config_problem != problem:
        raise ValueError(
            f"config Problem.Type {config_problem!r} does not match metadata problem {problem!r}."
        )
    model_block = _optional_dict(config.get("Model"))
    mesh_name = (
        _expect_scalar(model_block, "Mesh", str, fallback="") if model_block else ""
    )
    if mesh_name != "palace.msh":
        raise ValueError("config Model.Mesh must reference palace.msh.")

    if problem_block.get("Output") not in {"results/palace", "results/palace/", None}:
        raise ValueError("config Problem.Output must be results/palace.")


def _validate_index_entries(index_map: dict[str, Any]) -> None:
    entries = index_map.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("palace_index_map.json must include non-empty entries list.")
    expected_roles = {
        "Boundaries.Terminal": "terminal",
        "Boundaries.LumpedPort": "port_surface",
        "Domains.Postprocessing.Energy": "domain_energy",
        "Boundaries.Postprocessing.Dielectric": "surface_epr",
    }
    indices: dict[str, list[int]] = {section: [] for section in expected_roles}
    names: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not entry:
            raise ValueError("index map entries must be non-empty mappings.")
        section = _expect_scalar(entry, "section", str)
        if section not in expected_roles:
            raise ValueError(f"index map has unsupported section {section!r}.")
        if _expect_scalar(entry, "role", str) != expected_roles[section]:
            raise ValueError("index map section and role are inconsistent.")
        index = _expect_scalar(entry, "index", int)
        if index < 1:
            raise ValueError("index map indices must be positive.")
        indices[section].append(index)
        entry_name = _expect_scalar(entry, "entry_name", str)
        if not entry_name or (section, entry_name) in names:
            raise ValueError(
                "index entry names must be non-empty and unique per section."
            )
        names.add((section, entry_name))
        physical_names = _expect_list(
            entry.get("physical_names"), "index physical_names"
        )
        if not all(isinstance(name, str) and name for name in physical_names):
            raise ValueError("index physical_names must be non-empty strings.")
        entry_attributes = _expect_list(entry.get("attributes"), "index attributes")
        for attribute in entry_attributes:
            if (
                not isinstance(attribute, int)
                or isinstance(attribute, bool)
                or attribute < 1
            ):
                raise ValueError("index attributes must be positive integers.")
        metadata = _expect_mapping(entry.get("metadata"), "index metadata")
        if section == "Boundaries.Terminal":
            _expect_scalar(entry, "terminal_name", str)
            _expect_scalar(entry, "net_id", str)
            _expect_list(
                entry.get("conductor_component_ids"), "terminal conductor components"
            )
        elif section == "Boundaries.LumpedPort":
            _expect_scalar(entry, "port_name", str)
            owners = _expect_list(
                metadata.get("owner_semantic_ids"), "port owner semantic ids"
            )
            if len(owners) != 2 or not all(
                isinstance(owner, str) and owner for owner in owners
            ):
                raise ValueError(
                    "lumped port index entry must have exactly two owners."
                )
        elif section == "Domains.Postprocessing.Energy":
            _expect_mapping(metadata.get("material"), "domain index material")
        elif section == "Boundaries.Postprocessing.Dielectric":
            _expect_mapping(
                metadata.get("physical_attribute"), "surface index attributes"
            )
    for section, values in indices.items():
        if values and sorted(values) != list(range(1, len(values) + 1)):
            raise ValueError(f"index map {section} entries must be ordered 1..N.")


def _validate_table(
    problem: str,
    table: ParsedTable,
    index_counts: dict[str, int],
    config: dict[str, Any],
) -> None:
    if not table.rows:
        raise ValueError(f"{table.path} has no result rows.")

    if problem == "Electrostatic":
        expected_rows = index_counts["terminal"]
        if table.name == "terminal-C":
            _validate_matrix_table(table, "C[i][", index_counts["terminal"])
        elif table.name == "terminal-Cm":
            _validate_matrix_table(table, "C_m[i][", index_counts["terminal"])
        elif table.name == "terminal-Cinv":
            _validate_matrix_table(table, "C⁻¹[i][", index_counts["terminal"])
        elif table.name == "terminal-V":
            _validate_terminal_voltage_table(table, index_counts["terminal"])
        elif table.name == "domain-E":
            _validate_domain_table(table, index_counts["domain"], expected_rows)
        elif table.name == "surface-Q":
            _validate_surface_table(table, index_counts["surface"], expected_rows)
        elif table.name == "error-indicators":
            _validate_error_indicators(table)

    elif problem == "Eigenmode":
        expected_rows = _eigenmode_count(config)
        if table.name == "eig":
            _validate_eig_table(table, expected_rows)
        elif table.name == "port-EPR":
            _validate_port_table(
                table, index_counts["lumped_port"], expected_rows, kind="EPR"
            )
        elif table.name == "port-I":
            _validate_port_table(
                table, index_counts["lumped_port"], expected_rows, kind="I"
            )
        elif table.name == "port-V":
            _validate_port_table(
                table, index_counts["lumped_port"], expected_rows, kind="V"
            )
        elif table.name == "domain-E":
            _validate_domain_table(table, index_counts["domain"], expected_rows)
        elif table.name == "surface-Q":
            _validate_surface_table(table, index_counts["surface"], expected_rows)
        elif table.name == "error-indicators":
            _validate_error_indicators(table)


def _validate_matrix_table(
    table: ParsedTable,
    prefix: str,
    expected_count: int,
    *,
    include_index: bool = True,
) -> None:
    if len(table.rows) != expected_count:
        raise ValueError(
            f"{table.path} must contain {expected_count} data rows; found {len(table.rows)}."
        )
    expected_cols = expected_count + 1
    if len(table.headers) != expected_cols:
        raise ValueError(
            f"{table.path} has {len(table.headers)} columns; expected {expected_cols}."
        )

    if include_index and table.headers[0] != "i":
        raise ValueError(f"{table.path} row index column must be named 'i'.")

    suffixes = {"C[i][": " (F)", "C_m[i][": " (F)", "C⁻¹[i][": " (1/F)"}
    suffix = suffixes.get(prefix)
    if suffix is None:
        raise ValueError(f"unsupported native matrix prefix {prefix!r}.")
    expected_headers = (
        "i",
        *(f"{prefix}{column}]{suffix}" for column in range(1, expected_count + 1)),
    )
    if table.headers != expected_headers:
        raise ValueError(f"{table.path} native matrix headers are malformed.")

    for row in table.rows:
        _require_finite_data(row)
        if include_index:
            _require_positive_index(row.get("i"))
    _require_ordered_index(table, "i", expected_count)


def _validate_terminal_voltage_table(table: ParsedTable, terminal_count: int) -> None:
    if len(table.rows) != terminal_count:
        raise ValueError(
            f"{table.path} must contain {terminal_count} terminal voltage rows; "
            f"found {len(table.rows)}."
        )
    if len(table.headers) != 2 or table.headers[0] != "i":
        raise ValueError(f"{table.path} terminal voltage headers are malformed.")
    if table.headers[1] != "V_inc[i] (V)":
        raise ValueError(f"{table.path} terminal voltage header must be V_inc[i].")
    for row in table.rows:
        _require_finite_data(row)
        _require_positive_index(row.get("i"))
    _require_ordered_index(table, "i", terminal_count)


def _validate_eig_table(table: ParsedTable, expected_modes: int) -> None:
    if tuple(table.headers) != (
        "m",
        "Re{f} (GHz)",
        "Im{f} (GHz)",
        "Q",
        "Error (Bkwd.)",
        "Error (Abs.)",
    ):
        raise ValueError(
            f"{table.path} eig header does not match current Palace native schema."
        )
    for row in table.rows:
        _require_finite_data(row)
    if len(table.rows) != expected_modes:
        raise ValueError(
            f"{table.path} must contain {expected_modes} eig rows; found {len(table.rows)}."
        )
    _require_ordered_index(table, "m", expected_modes)


def _validate_port_table(
    table: ParsedTable, port_count: int, expected_modes: int, *, kind: str
) -> None:
    if kind == "EPR":
        if port_count == 0:
            raise ValueError(f"{table.path} requires at least one lumped port.")
        expected = 1 + port_count
        if len(table.headers) != expected:
            raise ValueError(
                f"{table.path} has {len(table.headers)} columns; expected {expected}."
            )
        expected_headers = ("m", *(f"p[{idx}]" for idx in range(1, port_count + 1)))
        if table.headers != expected_headers:
            raise ValueError(f"{table.path} port-EPR headers are malformed.")
        for idx, row in enumerate(table.rows, start=1):
            _require_finite_data(row)
            if len(row) != expected:
                raise ValueError(f"{table.path} row {idx} has malformed data width.")
    elif kind in {"I", "V"}:
        expected = 1 + 2 * port_count
        if len(table.headers) != expected:
            raise ValueError(
                f"{table.path} has {len(table.headers)} columns; expected {expected}."
            )
        unit = "A" if kind == "I" else "V"
        expected_headers = ["m"]
        for port in range(1, port_count + 1):
            expected_headers.extend(
                (f"Re{{{kind}[{port}]}} ({unit})", f"Im{{{kind}[{port}]}} ({unit})")
            )
        if table.headers != tuple(expected_headers):
            raise ValueError(f"{table.path} port-{kind} headers are malformed.")
        for idx, row in enumerate(table.rows, start=1):
            _require_finite_data(row)
            if len(row) != expected:
                raise ValueError(f"{table.path} row {idx} has malformed data width.")
    else:
        raise ValueError(f"unsupported port table kind {kind!r}.")
    if len(table.rows) != expected_modes:
        raise ValueError(
            f"{table.path} must contain {expected_modes} mode rows; found {len(table.rows)}."
        )
    _require_ordered_index(table, "m", expected_modes)


def _validate_domain_table(
    table: ParsedTable, domain_count: int, expected_rows: int
) -> None:
    if domain_count <= 0:
        raise ValueError("domain count must be positive when domain table is present.")
    expected = 1 + 4 + 4 * domain_count
    if len(table.headers) != expected:
        raise ValueError(
            f"{table.path} must contain {expected} columns for {domain_count} domains;"
            f" found {len(table.headers)}."
        )
    index_name = table.headers[0]
    if index_name not in {"i", "m"}:
        raise ValueError(f"{table.path} has an invalid domain index column.")
    expected_headers = [index_name, "E_elec (J)", "E_mag (J)", "E_cap (J)", "E_ind (J)"]
    for index in range(1, domain_count + 1):
        expected_headers.extend(
            (
                f"E_elec[{index}] (J)",
                f"p_elec[{index}]",
                f"E_mag[{index}] (J)",
                f"p_mag[{index}]",
            )
        )
    if table.headers != tuple(expected_headers):
        raise ValueError(f"{table.path} domain energy headers are malformed.")
    if len(table.rows) != expected_rows:
        raise ValueError(
            f"{table.path} must contain {expected_rows} rows; found {len(table.rows)}."
        )
    for row in table.rows:
        _require_finite_data(row)
    _require_ordered_index(table, index_name, expected_rows)


def _validate_surface_table(
    table: ParsedTable, surface_count: int, expected_rows: int
) -> None:
    if surface_count <= 0:
        raise ValueError(
            "surface count must be positive when surface table is present."
        )
    expected = 1 + 2 * surface_count
    if len(table.headers) != expected:
        raise ValueError(
            f"{table.path} must contain {expected} columns for {surface_count} surfaces;"
            f" found {len(table.headers)}."
        )
    index_name = table.headers[0]
    if index_name not in {"i", "m"}:
        raise ValueError(f"{table.path} has an invalid surface index column.")
    expected_headers = [index_name]
    for index in range(1, surface_count + 1):
        expected_headers.extend((f"p_surf[{index}]", f"Q_surf[{index}]"))
    if table.headers != tuple(expected_headers):
        raise ValueError(f"{table.path} surface Q headers are malformed.")
    if len(table.rows) != expected_rows:
        raise ValueError(
            f"{table.path} must contain {expected_rows} rows; found {len(table.rows)}."
        )
    for row in table.rows:
        _require_surface_q_data(row)
    _require_ordered_index(table, index_name, expected_rows)


def _validate_error_indicators(table: ParsedTable) -> None:
    if tuple(table.headers) != ("Norm", "Minimum", "Maximum", "Mean"):
        raise ValueError(f"{table.path} error-indicators header is malformed.")
    for row in table.rows:
        _require_finite_data(row)
    if len(table.rows) != 1:
        raise ValueError(f"{table.path} must contain exactly one error indicator row.")


def _read_csv_table(path: Path) -> ParsedTable:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, skipinitialspace=True)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} has no header row.")

    raw_headers = tuple(cell.strip() for cell in rows[0])
    if not raw_headers or any(cell == "" for cell in raw_headers):
        raise ValueError(f"{path} must declare non-empty headers.")
    if len(set(raw_headers)) != len(raw_headers):
        raise ValueError(f"{path} has duplicate headers.")

    parsed_rows: list[dict[str, Any]] = []
    for row in rows[1:]:
        values = tuple(cell.strip() for cell in row)
        if not any(values):
            continue
        if len(values) != len(raw_headers):
            raise ValueError(
                f"{path} contains malformed row with {len(values)} cells; expected {len(raw_headers)}."
            )
        parsed: dict[str, Any] = {}
        for name, value in zip(raw_headers, values):
            parsed[name] = _parse_cell(value)
        parsed_rows.append(parsed)

    if not parsed_rows:
        raise ValueError(f"{path} has no data rows.")
    return ParsedTable(
        name=path.stem, path=path, headers=raw_headers, rows=tuple(parsed_rows)
    )


def _validate_receipt_payload(
    *,
    receipt_payload: dict[str, Any],
    expected_handoff_id: str,
    handoff_metadata: dict[str, Any],
    run_metadata: dict[str, Any],
    problem: str,
    route: str,
    root: Path,
) -> None:
    if _expect_scalar(receipt_payload, "schema", str) != _RETURNED_RECEIPT_SCHEMA:
        raise ValueError("returned run receipt schema is invalid.")
    if _expect_int(receipt_payload.get("schema_version"), optional=False) != 1:
        raise ValueError("returned run receipt schema_version must be 1.")

    status = _expect_scalar(receipt_payload, "status", str)
    if status != "completed":
        raise ValueError("returned run receipt status is not completed.")

    if _expect_scalar(receipt_payload, "route", str) != route:
        raise ValueError("returned run receipt route mismatch.")
    if _expect_scalar(receipt_payload, "problem", str) != problem:
        raise ValueError("returned run receipt problem mismatch.")

    if (
        _expect_scalar(run_metadata, "handoff_id", str, fallback="")
        != expected_handoff_id
    ):
        raise ValueError("run metadata handoff_id does not match expected_handoff_id.")
    if (
        _expect_scalar(handoff_metadata, "handoff_id", str, fallback="")
        != expected_handoff_id
    ):
        raise ValueError(
            "handoff metadata handoff_id does not match expected_handoff_id."
        )
    if handoff_metadata.get("returned_receipt_path") != _RETURNED_RECEIPT_FILE:
        raise ValueError("handoff metadata returned_receipt_path is incorrect.")
    if run_metadata.get("returned_receipt_path") != _RETURNED_RECEIPT_FILE:
        raise ValueError("run metadata returned_receipt_path is incorrect.")

    handoff_id = _expect_scalar(receipt_payload, "handoff_id", str)
    if handoff_id != expected_handoff_id:
        raise ValueError(
            "returned run receipt handoff_id does not match expected_handoff_id."
        )
    if _expect_scalar(receipt_payload, "identity_verified", bool) is not True:
        raise ValueError("returned run receipt identity is not verified.")

    exit_code = _expect_int(receipt_payload.get("exit_code"))
    if exit_code != 0:
        raise ValueError(
            "returned run receipt exit_code must be zero for completed runs."
        )
    if _expect_int(receipt_payload.get("solver_exit_code")) != 0:
        raise ValueError("returned run receipt solver_exit_code must be zero.")
    if _expect_int(receipt_payload.get("tee_exit_code")) != 0:
        raise ValueError("returned run receipt tee_exit_code must be zero.")

    _validate_timestamp(
        _expect_scalar(receipt_payload, "timestamp_utc", str), "timestamp_utc"
    )
    job_identity = _expect_mapping(
        receipt_payload.get("job_identity"), "returned receipt job_identity"
    )
    if _expect_scalar(job_identity, "job_id", str, fallback="") == "":
        raise ValueError("returned run receipt job_id must be a non-empty string.")
    if _expect_scalar(job_identity, "job_name", str, fallback="") == "":
        raise ValueError("returned run receipt job_name must be a non-empty string.")

    expected_input_map = _extract_hash_map(
        _expect_list(receipt_payload.get("input_hashes"), "input_hashes")
    )
    configured_input_map = _extract_hash_map(
        _expect_list(handoff_metadata.get("hashes"), "metadata hashes")
    )
    if expected_input_map != configured_input_map:
        raise ValueError(
            "returned run receipt input hashes do not match metadata hashes."
        )
    for name, expected in expected_input_map.items():
        observed = _compute_input_hash_entry(root, name)
        if expected != observed:
            raise ValueError(f"returned run receipt input hash mismatch for {name}.")

    if _expect_scalar(handoff_metadata, "route", str) != route:
        raise ValueError("handoff metadata route mismatch.")
    if _expect_scalar(run_metadata, "route", str) != route:
        raise ValueError("run metadata route mismatch.")

    output_files = _expect_list(receipt_payload.get("output_files"), "output_files")
    output_map = _extract_hash_map(output_files)
    required_outputs = _required_output_records(problem)
    for expected in required_outputs:
        rel = expected["path"]
        if rel not in output_map:
            raise ValueError(
                f"returned run receipt misses required output entry: {rel}"
            )

    for entry in output_files:
        data = _expect_mapping(entry, "returned receipt output entry")
        if _expect_scalar(data, "present", bool) is not True:
            raise ValueError("returned run receipt lists a missing required output.")

    for rel, expected in output_map.items():
        observed = _compute_hash_entry(_confined_path(root, rel))
        if observed[0] != expected[0] or observed[1] != expected[1]:
            raise ValueError(f"returned run receipt output hash mismatch for {rel}")

    log = _expect_mapping(receipt_payload.get("log"), "return_receipt.log")
    if not log:
        raise ValueError("returned run receipt log entry is missing.")
    log_path = _expect_scalar(log, "path", str)
    log_expected = _expect_scalar(log, "bytes", int), _expect_scalar(log, "sha256", str)
    log_observed = _compute_hash_entry(_confined_path(root, log_path))
    if log_expected != log_observed:
        raise ValueError(f"returned run receipt log hash mismatch for {log_path}.")
    if log_path == "":
        raise ValueError("returned run receipt log path must be non-empty.")
    if _expect_scalar(log, "present", bool) is not True:
        raise ValueError("returned run receipt log is not present.")

    receipt_identity = _expect_mapping(
        receipt_payload.get("solver_identity"), "receipt.solver_identity"
    )
    palace_identity = _read_json(root / "results/palace/palace.json")
    expected_identity = {
        key: palace_identity[key]
        for key in ("GitTag", "PeakMemoryMegabytes", "PeakNodeMemoryMegabytes")
        if key in palace_identity
    }
    if not expected_identity or receipt_identity != expected_identity:
        raise ValueError(
            "returned run receipt solver identity does not match palace.json."
        )


def _build_returned_receipt(payload: dict[str, Any]) -> PalaceReturnedReceipt:
    return PalaceReturnedReceipt(
        schema=_expect_scalar(payload, "schema", str),
        schema_version=_expect_int(payload.get("schema_version")),
        handoff_id=_expect_scalar(payload, "handoff_id", str),
        status=_expect_scalar(payload, "status", str),
        exit_code=_expect_int(payload.get("exit_code")),
        solver_exit_code=_expect_int(payload.get("solver_exit_code")),
        tee_exit_code=_expect_int(payload.get("tee_exit_code")),
        route=_expect_scalar(payload, "route", str),
        problem=_expect_scalar(payload, "problem", str),
        identity_verified=_expect_scalar(payload, "identity_verified", bool),
        timestamp_utc=_expect_scalar(payload, "timestamp_utc", str),
        job_identity=_expect_mapping(
            payload.get("job_identity"), "receipt.job_identity"
        ),
        input_hashes=tuple(
            _expect_mapping(entry, "returned_receipt.input_hash")
            for entry in _expect_list(
                payload.get("input_hashes"), "returned receipt input_hashes"
            )
        ),
        output_files=tuple(
            _expect_mapping(entry, "returned_receipt.output_file")
            for entry in _expect_list(
                payload.get("output_files"), "returned receipt output_files"
            )
        ),
        log=_expect_mapping(payload.get("log"), "receipt.log")
        if payload.get("log") is not None
        else None,
        solver_identity=_expect_mapping(
            payload.get("solver_identity"), "receipt.solver_identity"
        ),
    )


def _validate_returned_state_agreement(
    *,
    receipt_payload: dict[str, Any],
    run_metadata: dict[str, Any],
    resource_record: dict[str, Any],
) -> None:
    receipt_status = _expect_scalar(receipt_payload, "status", str)
    receipt_exit = _expect_int(receipt_payload.get("exit_code"))
    receipt_job = _expect_mapping(receipt_payload.get("job_identity"), "receipt job")
    for label, payload in (
        ("run metadata", run_metadata),
        ("resource record", resource_record),
    ):
        if _expect_scalar(payload, "status", str) != receipt_status:
            raise ValueError(f"{label} status does not match returned receipt.")
        if _expect_int(payload.get("exit_code")) != receipt_exit:
            raise ValueError(f"{label} exit_code does not match returned receipt.")
        if (
            _expect_mapping(payload.get("job_identity"), f"{label} job identity")
            != receipt_job
        ):
            raise ValueError(f"{label} job identity does not match returned receipt.")


def _required_output_records(problem: str) -> tuple[dict[str, Any], ...]:
    families = _PROBLEM_FAMILIES[problem]
    records: list[dict[str, Any]] = []
    for family in families:
        rel = f"results/palace/{family}.csv"
        records.append({"path": rel, "bytes": 0, "sha256": ""})
    records.append({"path": "results/palace/palace.json", "bytes": 0, "sha256": ""})
    return tuple(records)


def _index_counts(index_map: dict[str, Any]) -> dict[str, int]:
    entries = _expect_list(index_map.get("entries"), "palace_index_map.entries")
    counts = {
        "terminal": 0,
        "lumped_port": 0,
        "domain": 0,
        "surface": 0,
    }
    for entry in entries:
        data = _expect_mapping(entry, "palace_index_map.entry")
        section = _expect_scalar(data, "section", str, fallback="")
        if section.startswith("Boundaries.Terminal"):
            counts["terminal"] += 1
        elif section.startswith("Boundaries.LumpedPort"):
            counts["lumped_port"] += 1
        elif section.startswith("Domains.Postprocessing.Energy"):
            counts["domain"] += 1
        elif section == "Boundaries.Postprocessing.Dielectric":
            counts["surface"] += 1
    return counts


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _expect_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object.")
    return value


def _validate_palace_identity(payload: Mapping[str, Any]) -> None:
    if not payload:
        raise ValueError("palace_identity must be a non-empty mapping.")
    for key in ("config_schema", "runtime"):
        if _expect_scalar(payload, key, str, fallback=None) is None:
            raise ValueError("palace_identity must include config_schema and runtime.")


def _parse_cell(value: str) -> Any:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _require_finite_data(
    row: Mapping[str, Any],
    *,
    allow_infinite: bool = False,
    allow_non_numeric: bool = False,
) -> None:
    for key, value in row.items():
        if value is None:
            raise ValueError("result rows must not contain missing entries.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            if not allow_non_numeric:
                raise ValueError(f"result value in column {key!r} is not numeric.")
            continue
        if math.isnan(value) or (not allow_infinite and not math.isfinite(value)):
            raise ValueError(f"result value in column {key!r} is non-finite.")


def _require_surface_q_data(row: Mapping[str, Any]) -> None:
    """Permit only native lossless-Q infinities; all participation stays finite."""

    for key, value in row.items():
        if (
            value is None
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(f"result value in column {key!r} is not numeric.")
        if math.isnan(value):
            raise ValueError(f"result value in column {key!r} is non-finite.")
        if math.isinf(value) and (
            value < 0 or not (key == "Q" or key.startswith("Q_surf["))
        ):
            raise ValueError(
                f"only a positive solver-native Q may contain infinity; got {key!r}."
            )


def _require_ordered_index(
    table: ParsedTable, column: str, expected_count: int
) -> None:
    values = [row.get(column) for row in table.rows]
    expected = list(range(1, expected_count + 1))
    if values != expected:
        raise ValueError(
            f"{table.path} {column!r} values must be ordered 1..{expected_count}."
        )


def _require_positive_index(value: Any) -> None:
    if not isinstance(value, (int, float)) or value != int(value) or value <= 0:
        raise ValueError("matrix row index must be a positive integer.")


def _extract_hash_map(
    entries: list[dict[str, Any]],
    label: str = "hash entries",
    *,
    require_optional_values: bool = False,
) -> dict[str, tuple[int | None, str | None]]:
    result: dict[str, tuple[int | None, str | None]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(f"{label} must be mappings.")
        path = _expect_scalar(entry, "path", str)
        if path in result:
            raise ValueError(f"{label} contains duplicate path {path!r}.")
        if require_optional_values:
            bytes_value = _expect_int(entry.get("bytes"), optional=True)
            sha_value = _expect_scalar(entry, "sha256", str, fallback=None)
        else:
            bytes_value = _expect_int(entry.get("bytes"))
            sha_value = _expect_scalar(entry, "sha256", str)
        result[path] = (
            bytes_value,
            sha_value,
        )
    return result


def _compute_hash_entry(path: Path) -> tuple[int | None, str | None]:
    if not path.is_file():
        return None, None
    return path.stat().st_size, _sha256(path)


def _compute_input_hash_entry(root: Path, rel: str) -> tuple[int | None, str | None]:
    path = _confined_path(root, rel)
    return _compute_hash_entry(path)


def _validate_hash_entry(
    entry: Mapping[str, Any], label: str, *, optional: bool = False
) -> None:
    if optional and not entry:
        return
    _expect_scalar(entry, "path", str)
    _expect_int(entry.get("bytes"), optional=True)
    _expect_scalar(entry, "sha256", str)


def _validate_timestamp(value: str, label: str) -> None:
    try:
        parsed = value
        if value.endswith("Z"):
            parsed = value.replace("Z", "+00:00")
        __import__("datetime").datetime.fromisoformat(parsed)
    except Exception as exc:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp.") from exc


def _validate_contract_schema(payload: dict[str, Any], label: str) -> None:
    if _expect_scalar(payload, "schema_version", int, fallback=0) != 1:
        raise ValueError(f"{label} schema_version must be integer 1.")


def _read_json(path: Path) -> dict[str, Any]:
    payload = path.read_text(encoding="utf-8")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return parsed


def _expect_scalar(
    payload: dict[str, Any],
    field: str,
    _type: type,
    *,
    fallback: Any = None,
) -> Any:
    value = payload.get(field) if isinstance(payload, dict) else fallback
    if value is None:
        return fallback
    if (_type is int and isinstance(value, bool)) or not isinstance(value, _type):
        raise TypeError(f"field {field!r} must be {_type.__name__}.")
    return value


def _expect_list(payload: Any, label: str, *, allow_empty: bool = False) -> list[Any]:
    if not isinstance(payload, list):
        raise TypeError(f"{label} must be a list.")
    if not allow_empty and not payload:
        raise ValueError(f"{label} must not be empty.")
    return payload


def _expect_int(value: Any, *, optional: bool = False) -> int | None:
    if value is None:
        if optional:
            return None
        raise TypeError("value must be integer-compatible.")
    if isinstance(value, bool):
        raise TypeError("value must be integer-compatible.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError("value must be integer-compatible.")


def _to_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be float-compatible.")
    return float(value)


def _as_float_map(payload: Any, keys: tuple[str, ...]) -> dict[str, float | None]:
    if not isinstance(payload, dict):
        return {key: None for key in keys}
    return {
        key: _to_float(payload.get(key))
        if isinstance(payload.get(key), (int, float))
        else None
        for key in keys
    }


def _eigenmode_count(config: dict[str, Any]) -> int:
    solver = _expect_mapping(config.get("Solver"), "config.Solver")
    eigenmode = _expect_mapping(solver.get("Eigenmode"), "config.Solver.Eigenmode")
    count = _expect_int(eigenmode.get("N"))
    if count is None or count < 1:
        raise ValueError("config Solver.Eigenmode.N must be positive.")
    return count


def _optional_dict(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _confined_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("artifact paths must be non-empty relative paths.")
    candidate = (root / relative).resolve()
    root = root.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("artifact path escapes the run directory.")
    return candidate


__all__ = [
    "PalaceCost",
    "PalacePerformance",
    "PalaceProvenance",
    "PalaceReturnedReceipt",
    "ParsedTable",
    "ResolvedPalaceResult",
    "resolve_palace_result",
]
