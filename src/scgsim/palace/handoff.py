"""Manual-only Palace handoff adapted from gsim portable fixes @ 8cf5fa79fa3abb176940dbfc520ff34a44f4770e."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from ._mesh import (
    GSIM_SHA,
    SGB_DERIVATION_BASE_SHA,
    SGB_DERIVATION_IMPORTED_SHA,
    SGB_RUNTIME_AUTHORITY,
    MeshBuildResult,
)
from ._staged import RUNTIME_VERSION, SCHEMA_VERSION

HandoffProfile = Literal["ltlab-local", "ltlab-slurm", "f1-slurm"]
_PROFILES = {"ltlab-local", "ltlab-slurm", "f1-slurm"}
PORTABLE_HANDOFF_SHA = "8cf5fa79fa3abb176940dbfc520ff34a44f4770e"


@dataclass(frozen=True)
class HandoffPlan:
    """Prepared, manual-only files. This library never executes or submits them."""

    profile: HandoffProfile
    status: Literal["prepared"]
    kind: Literal["manual_handoff"]
    handoff_id: str
    run_dir: Path
    script_path: Path
    metadata_path: Path
    run_metadata_path: Path
    resource_record_path: Path
    manifest_path: Path
    archive_path: Path
    returned_receipt_path: Path
    returned_receipt_recorder_path: Path


def prepare_handoff(
    *,
    profile: str,
    mesh_result: MeshBuildResult,
    config_path: Path,
    executable: str,
    resources: Mapping[str, Any] | None = None,
    setup_commands: Sequence[str] = (),
    petsc_options: Sequence[str] = (),
    problem: Literal["Electrostatic", "Eigenmode"] = "Electrostatic",
) -> HandoffPlan:
    """Create only portable handoff artifacts; caller owns executable and resources."""
    if profile not in _PROFILES:
        raise ValueError(f"Unsupported handoff profile {profile!r}.")
    if not config_path.is_file() or not mesh_result.mesh_path.is_file():
        raise FileNotFoundError("handoff requires existing config.json and palace.msh.")

    executable = _single_line(executable, "executable")
    setup = tuple(_single_line(command, "setup command") for command in setup_commands)
    petsc = tuple(_single_line(option, "PETSc option") for option in petsc_options)
    resource_map = _validate_resources(profile, resources)
    run_dir = mesh_result.output_dir
    for directory in (
        run_dir / "logs",
        run_dir / "results" / "palace",
        run_dir / "metadata",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    script_path = run_dir / (
        "run_palace.sh" if profile == "ltlab-local" else "run_palace.sbatch"
    )
    (
        run_dir / ("run_palace.sbatch" if profile == "ltlab-local" else "run_palace.sh")
    ).unlink(missing_ok=True)

    metadata_path = run_dir / "metadata" / "palace_handoff_metadata.json"
    run_metadata_path = run_dir / "metadata" / "palace_run_metadata.json"
    resource_record_path = run_dir / "metadata" / "palace_resource_record.json"
    manifest_path = run_dir / "metadata" / "palace_handoff_archive_manifest.json"
    archive_path = run_dir / "palace_handoff.tar.gz"
    returned_receipt_path = run_dir / "metadata" / "palace_returned_run_receipt.json"
    returned_receipt_recorder_path = run_dir / "metadata" / "record_palace_return.py"
    index_map_path = run_dir / "metadata" / "palace_index_map.json"

    _clear_previous_returned_outputs(
        run_dir=run_dir,
        returned_receipt_path=returned_receipt_path,
    )
    if not index_map_path.is_file():
        raise FileNotFoundError("handoff requires metadata/palace_index_map.json.")

    route = _route(mesh_result)
    hashes = _hashes(
        run_dir,
        (
            mesh_result.mesh_path,
            config_path,
            mesh_result.mesh_manifest_path,
            index_map_path,
        ),
    )
    identity = _identity()

    if returned_receipt_recorder_path.is_file():
        returned_receipt_recorder_path.unlink()
    shutil.copy2(
        Path(__file__).with_name("returned_receipt.py"),
        returned_receipt_recorder_path,
    )
    returned_receipt_recorder_path.chmod(0o755)

    script_path.write_text(
        _render_script(
            profile,
            executable,
            resource_map,
            setup,
            petsc,
            returned_receipt_recorder_name=returned_receipt_recorder_path.name,
        )
        + "\n",
        encoding="utf-8",
    )
    script_path.chmod(script_path.stat().st_mode | 0o755)

    palace_identity = {
        "config_schema": SCHEMA_VERSION,
        "runtime": RUNTIME_VERSION,
    }
    execution_identity = _execution_identity(
        profile=profile,
        resources=resource_map,
        executable=executable,
        setup=setup,
        petsc=petsc,
        script_path=script_path,
        recorder_path=returned_receipt_recorder_path,
    )
    handoff_id = _compute_handoff_id(
        profile=profile,
        route=route,
        problem=problem,
        source_revisions=identity,
        palace_identity=palace_identity,
        hashes=hashes,
        execution_identity=execution_identity,
    )

    _write_json(
        metadata_path,
        {
            "schema_version": 1,
            "status": "prepared",
            "kind": "manual_handoff",
            "manual_intent": True,
            "profile": profile,
            "route": route,
            "problem": problem,
            "handoff_id": handoff_id,
            "returned_receipt_path": returned_receipt_path.relative_to(
                run_dir
            ).as_posix(),
            "returned_receipt_recorder": returned_receipt_recorder_path.relative_to(
                run_dir
            ).as_posix(),
            "palace_identity": palace_identity,
            "source_revisions": identity,
            "execution_identity": execution_identity,
            "command": _redacted_command(profile, executable, resource_map),
            "setup_commands": list(setup),
            "petsc_options": list(petsc),
            "requested_resources": resource_map,
            "resolved_resources": resource_map,
            "script": script_path.name,
            "config": config_path.name,
            "mesh": mesh_result.mesh_path.name,
            "hashes": hashes,
        },
    )

    _write_json(
        run_metadata_path,
        {
            "schema_version": 1,
            "status": "not_run",
            "kind": "manual_handoff",
            "route": route,
            "problem": problem,
            "handoff_id": handoff_id,
            "returned_receipt_path": returned_receipt_path.relative_to(
                run_dir
            ).as_posix(),
            "palace_identity": palace_identity,
            "source_revisions": identity,
            "execution_identity": execution_identity,
            "hashes": hashes,
        },
    )

    resource_record_path.unlink(missing_ok=True)
    archive_path.unlink(missing_ok=True)

    excluded = {archive_path, resource_record_path}
    members = _members(run_dir, exclude=excluded | {manifest_path})
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "archive": archive_path.name,
            "input_hashes": hashes,
            "members": members,
            "archive_members_excluded_from_self_listing": [
                manifest_path.relative_to(run_dir).as_posix()
            ],
            "archive_members_excluded": [
                resource_record_path.relative_to(run_dir).as_posix()
            ],
        },
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(run_dir.rglob("*")):
            if path in excluded:
                continue
            archive.add(
                path, arcname=path.relative_to(run_dir).as_posix(), recursive=False
            )
    _write_json(
        resource_record_path,
        {
            "schema_version": 1,
            "status": "not_submitted",
            "kind": "manual_handoff",
            "profile": profile,
            "route": route,
            "problem": problem,
            "handoff_id": handoff_id,
            "source_revisions": identity,
            "palace_identity": palace_identity,
            "execution_identity": execution_identity,
            "requested_resources": resource_map,
            "resolved_resources": resource_map,
            "archive": {"path": archive_path.name, "sha256": _sha256(archive_path)},
        },
    )

    return HandoffPlan(
        profile,
        "prepared",
        "manual_handoff",
        handoff_id,
        run_dir,
        script_path,
        metadata_path,
        run_metadata_path,
        resource_record_path,
        manifest_path,
        archive_path,
        returned_receipt_path,
        returned_receipt_recorder_path,
    )


def _validate_resources(
    profile: str, resources: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not isinstance(resources, Mapping) or not resources:
        raise ValueError("handoff requires caller-supplied resources.")
    result = dict(resources)
    if profile == "ltlab-local":
        _positive_int(result.get("threads", 1), "resources.threads")
        _positive_int(result.get("processes", 1), "resources.processes")
    else:
        _positive_int(result.get("ntasks"), "resources.ntasks")
        _positive_int(result.get("cpus_per_task"), "resources.cpus_per_task")
        nodes = _positive_int(result.get("nodes"), "resources.nodes")
        if profile == "ltlab-slurm" and nodes != 1:
            raise ValueError(
                "ltlab-slurm is a single-node profile and requires resources.nodes=1."
            )
        if profile == "f1-slurm":
            if nodes < 2:
                raise ValueError(
                    "f1-slurm is a multi-node profile and requires resources.nodes >= 2."
                )
            if not any(key in result for key in ("mem", "mem_per_cpu")):
                raise ValueError(
                    "f1-slurm requires caller-supplied high-memory resource (mem or mem_per_cpu)."
                )
    style = result.get("command_style", "binary")
    if style not in {"binary", "wrapper"}:
        raise ValueError("resources.command_style must be 'binary' or 'wrapper'.")
    if profile != "ltlab-local" and style == "wrapper":
        raise ValueError(
            "resources.command_style='wrapper' is supported only for ltlab-local; "
            "Slurm profiles require 'binary'."
        )
    return result


def _render_script(
    profile: str,
    executable: str,
    resources: Mapping[str, Any],
    setup: Sequence[str],
    petsc: Sequence[str],
    *,
    returned_receipt_recorder_name: str,
) -> str:
    style = str(resources.get("command_style", "binary"))
    threads = _positive_int(
        resources.get("threads", resources.get("cpus_per_task", 1)), "threads"
    )
    processes = _positive_int(
        resources.get("processes", resources.get("ntasks", 1)), "processes"
    )
    directives: list[str] = []
    if profile != "ltlab-local":
        for key, value in resources.items():
            if key in {"launcher", "command_style", "threads", "processes"}:
                continue
            _single_line(str(value), f"resource {key}")
            directives.append(f"#SBATCH --{key.replace('_', '-')}={value}")
    launcher = (
        []
        if profile == "ltlab-local"
        else _launcher_tokens(resources.get("launcher", ("srun",)))
    )
    if profile != "ltlab-local" and not launcher:
        raise ValueError("Slurm profile launcher must be non-empty.")

    arguments = ['"$PALACE_EXECUTABLE"']
    if style == "wrapper":
        arguments.extend(["-np", '"$PALACE_PROCESSES"', "-nt", '"$PALACE_THREADS"'])
    arguments.append('"$PALACE_CONFIG"')
    command = " ".join([*(shlex.quote(item) for item in launcher), *arguments])

    run_directory_lines = (
        [
            'SCRIPT_SOURCE="${BASH_SOURCE[0]}"',
            'RUN_DIR="$(CDPATH= cd -- "$(dirname -- "$SCRIPT_SOURCE")" && pwd -P)" || exit 2',
        ]
        if profile == "ltlab-local"
        else [
            '[[ -n "${SLURM_SUBMIT_DIR:-}" ]] || { echo "missing SLURM_SUBMIT_DIR" >&2; exit 2; }',
            'RUN_DIR="$(CDPATH= cd -- "$SLURM_SUBMIT_DIR" && pwd -P)" || exit 2',
        ]
    )

    lines = [
        "#!/usr/bin/env bash",
        *directives,
        "",
        "set -u",
        *run_directory_lines,
        '[[ -f "$RUN_DIR/config.json" ]] || { echo "missing $RUN_DIR/config.json" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/palace.msh" ]] || { echo "missing $RUN_DIR/palace.msh" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/metadata/palace_handoff_metadata.json" ]] || { echo "missing canonical handoff metadata" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/metadata/palace_run_metadata.json" ]] || { echo "missing canonical run metadata" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/metadata/mesh_manifest.json" ]] || { echo "missing canonical mesh manifest" >&2; exit 2; }',
        '[[ -f "$RUN_DIR/metadata/palace_index_map.json" ]] || { echo "missing canonical index map" >&2; exit 2; }',
        'cd "$RUN_DIR" || exit 2',
        "mkdir -p logs results/palace metadata",
        "rm -f metadata/palace_returned_run_receipt.json logs/palace-*.log",
        "rm -rf results/palace",
        "mkdir -p results/palace",
        'PALACE_CONFIG="config.json"',
        'PALACE_MESH="palace.msh"',
        f"PALACE_EXECUTABLE={shlex.quote(executable)}",
        f"PALACE_PROCESSES={processes}",
        f"PALACE_THREADS={threads}",
        'export OMP_NUM_THREADS="$PALACE_THREADS"',
        'PALACE_LOG="logs/palace-${SLURM_JOB_ID:-manual}.log"',
        '[[ -f "$PALACE_CONFIG" ]] || { echo "missing $PALACE_CONFIG" >&2; exit 2; }',
        '[[ -f "$PALACE_MESH" ]] || { echo "missing $PALACE_MESH" >&2; exit 2; }',
    ]
    if petsc:
        lines.extend(
            [
                f"PALACE_PETSC_OPTIONS={shlex.quote(' '.join(petsc))}",
                'export PETSC_OPTIONS="${PETSC_OPTIONS:-} ${PALACE_PETSC_OPTIONS}"',
            ]
        )

    if any(command.lstrip().startswith("module ") for command in setup):
        lines.extend(
            [
                "if ! type module >/dev/null 2>&1; then",
                '  [[ -r /etc/profile.d/modules.sh ]] || { echo "missing Environment Modules initialization" >&2; exit 2; }',
                "  set +u",
                "  source /etc/profile.d/modules.sh || { echo 'Environment Modules initialization failed' >&2; exit 2; }",
                "  set -u",
                "fi",
            ]
        )
    for setup_command in setup:
        lines.extend(
            [
                "set +u",
                f"{{ {setup_command}; }} || {{ echo 'setup command failed' >&2; exit 2; }}",
                "set -u",
            ]
        )

    lines.extend(
        [
            "set -o pipefail",
            f'{command} 2>&1 | tee "$PALACE_LOG"',
            'PALACE_PIPE_STATUS=("${PIPESTATUS[@]}")',
            "PALACE_EXIT_CODE=${PALACE_PIPE_STATUS[0]:-1}",
            "TEE_EXIT_CODE=${PALACE_PIPE_STATUS[1]:-1}",
            "PIPELINE_EXIT_CODE=$PALACE_EXIT_CODE",
            'if [ "$PIPELINE_EXIT_CODE" -eq 0 ] && [ "$TEE_EXIT_CODE" -ne 0 ]; then PIPELINE_EXIT_CODE=$TEE_EXIT_CODE; fi',
            (
                f'python3 metadata/{returned_receipt_recorder_name} "$(pwd)" '
                '"$PIPELINE_EXIT_CODE" --log-path "$PALACE_LOG" '
                '--solver-exit-code "$PALACE_EXIT_CODE" '
                '--tee-exit-code "$TEE_EXIT_CODE" '
                '--job-id "${SLURM_JOB_ID:-manual}" '
                '--job-name "${SLURM_JOB_NAME:-manual}" '
                " || RECEIPT_EXIT=$?"
            ),
            "RECEIPT_EXIT=${RECEIPT_EXIT:-0}",
            'if [ "$PIPELINE_EXIT_CODE" -ne 0 ]; then exit "$PIPELINE_EXIT_CODE"; fi',
            'if [ "$RECEIPT_EXIT" -ne 0 ]; then exit "$RECEIPT_EXIT"; fi',
            'exit "$PIPELINE_EXIT_CODE"',
        ]
    )
    return "\n".join(lines)


def _redacted_command(
    profile: str, executable: str, resources: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "manual_only": True,
        "script": "run_palace.sh" if profile == "ltlab-local" else "run_palace.sbatch",
        "executable_identity": Path(executable).name,
        "launcher": resources.get(
            "launcher", "local" if profile == "ltlab-local" else "srun"
        ),
        "arguments": ["[executable]", "config.json"],
    }


def _identity() -> dict[str, str]:
    try:
        scgsim = version("scgsim")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Cannot prepare handoff without installed scgsim package metadata."
        ) from exc
    return {
        "scgsim": scgsim,
        "candidate_state": "converging",
        "gsim_meshing": GSIM_SHA,
        "gsim_portable_handoff": PORTABLE_HANDOFF_SHA,
        "scgsim_sgb": SGB_RUNTIME_AUTHORITY,
        "sgb_derivation": {
            "base": SGB_DERIVATION_BASE_SHA,
            "imported_development": SGB_DERIVATION_IMPORTED_SHA,
        },
    }


def _compute_handoff_id(
    *,
    profile: str,
    route: str,
    problem: str,
    source_revisions: Mapping[str, Any],
    palace_identity: Mapping[str, Any],
    hashes: Sequence[Mapping[str, Any]],
    execution_identity: Mapping[str, Any],
) -> str:
    payload = {
        "route": route,
        "problem": problem,
        "profile": profile,
        "source_revisions": source_revisions,
        "palace_identity": palace_identity,
        "hashes": list(hashes),
        "execution_identity": execution_identity,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _execution_identity(
    *,
    profile: str,
    resources: Mapping[str, Any],
    executable: str,
    setup: Sequence[str],
    petsc: Sequence[str],
    script_path: Path,
    recorder_path: Path,
) -> dict[str, Any]:
    """Bind the generated execution surface without recording an absolute path."""

    return {
        "profile": profile,
        "resources": dict(resources),
        "setup_commands": list(setup),
        "petsc_options": list(petsc),
        "command": _redacted_command(profile, executable, resources),
        "script": {"path": script_path.name, "sha256": _sha256(script_path)},
        "recorder": {
            "path": f"metadata/{recorder_path.name}",
            "sha256": _sha256(recorder_path),
        },
    }


def _clear_previous_returned_outputs(
    *, run_dir: Path, returned_receipt_path: Path
) -> None:
    """A new manual handoff cannot reuse outputs or a receipt from an earlier run."""

    returned_receipt_path.unlink(missing_ok=True)
    results_path = run_dir / "results" / "palace"
    if results_path.exists():
        shutil.rmtree(results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    for log_path in (run_dir / "logs").glob("palace-*.log"):
        log_path.unlink()


def _launcher_tokens(value: Any) -> list[str]:
    """Treat a launcher string as one executable token, never as characters."""
    if isinstance(value, str):
        return [_single_line(value, "launcher")]
    if not isinstance(value, Sequence):
        raise TypeError(
            "resources.launcher must be a single string or a sequence of strings."
        )
    tokens = [_single_line(item, "launcher token") for item in value]
    if not tokens:
        raise ValueError("resources.launcher must be non-empty.")
    return tokens


def _members(root: Path, *, exclude: set[Path]) -> list[dict[str, str | int]]:
    members: list[dict[str, str | int]] = []
    for path in sorted(root.rglob("*")):
        if path in exclude:
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            members.append({"path": relative + "/", "type": "directory"})
        elif path.is_file():
            members.append(
                {
                    "path": relative,
                    "type": "file",
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return members


def _hashes(root: Path, paths: Sequence[Path]) -> list[dict[str, str | int]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route(mesh_result: MeshBuildResult) -> str:
    route = json.loads(mesh_result.provenance_path.read_text(encoding="utf-8")).get(
        "route"
    )
    if route not in {"A", "B"}:
        raise ValueError("mesh provenance must carry Route A or B.")
    return route


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _single_line(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise ValueError(f"{field} must be non-empty single-line text.")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


__all__ = ["HandoffPlan", "HandoffProfile", "prepare_handoff"]
