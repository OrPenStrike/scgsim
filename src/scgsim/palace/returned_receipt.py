"""Pure-stdlib recorder for completed manual Palace runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "palace-returned-run-receipt.v1"
RUNTIME_COMPATIBILITY_SCHEMA = "palace-runtime-compatibility.v1"
RUNTIME_COMPATIBILITY_PATH = "metadata/palace_runtime_compatibility.json"

try:
    from datetime import UTC
except ImportError:  # pragma: no cover
    UTC = __import__("datetime").timezone.utc

_REQUIRED_OUTPUT_FAMILIES = {
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = path.read_text(encoding="utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return data


def _required_output_paths(problem: str, log_path: str | None) -> list[str]:
    families = _REQUIRED_OUTPUT_FAMILIES.get(problem)
    if not families:
        raise ValueError("unsupported problem")
    paths = [f"results/palace/{family}.csv" for family in families]
    paths.append("results/palace/palace.json")
    if log_path:
        paths.append(log_path)
    return paths


def _iteration_output_paths(root: Path, problem: str) -> list[str]:
    families = _REQUIRED_OUTPUT_FAMILIES.get(problem)
    if not families:
        raise ValueError("unsupported problem")
    results = root / "results" / "palace"
    if not results.is_dir():
        return []
    paths: list[str] = []
    for directory in sorted(
        (
            path
            for path in results.iterdir()
            if path.is_dir()
            and path.name.startswith("iteration")
            and path.name.removeprefix("iteration").isdigit()
        ),
        key=lambda path: int(path.name.removeprefix("iteration")),
    ):
        for name in (*families, "palace"):
            path = directory / f"{name}.{'json' if name == 'palace' else 'csv'}"
            if path.is_file():
                paths.append(path.relative_to(root).as_posix())
    return paths


def _record_for_path(run_dir: Path, relative_path: str) -> dict[str, Any]:
    path = _confined_path(run_dir, relative_path)
    if not path.is_file():
        return {"path": relative_path, "bytes": None, "sha256": None, "present": False}
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "present": True,
    }


def _hashes_from_metadata(root: Path) -> list[dict[str, Any]]:
    metadata_path = root / "metadata" / "palace_handoff_metadata.json"
    metadata = _read_json(metadata_path)
    hashes = metadata.get("hashes")
    if not isinstance(hashes, list):
        raise TypeError("handoff metadata hashes must be a list.")
    return hashes


def _diagnostic_handoff_id(
    *, route: str, problem: str, input_hashes: list[dict[str, Any]]
) -> str:
    """Mark after-the-fact parser diagnostics as deliberately non-production."""

    canonical = json.dumps(
        {"route": route, "problem": problem, "input_hashes": input_hashes},
        sort_keys=True,
        separators=(",", ":"),
    )
    return "diagnostic-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _solver_identity(root: Path) -> dict[str, Any]:
    palace_json_path = root / "results" / "palace" / "palace.json"
    if not palace_json_path.is_file():
        return {}
    payload = _read_json(palace_json_path)
    return {
        k: payload[k]
        for k in ("GitTag", "PeakMemoryMegabytes", "PeakNodeMemoryMegabytes")
        if k in payload
    }


def _compatibility_required(handoff_metadata: dict[str, Any]) -> str | None:
    palace_identity = handoff_metadata.get("palace_identity")
    if not isinstance(palace_identity, dict):
        raise TypeError("handoff palace_identity must be a mapping.")
    probe = palace_identity.get("config_compatibility_probe")
    if probe is None:
        return None
    if probe != "native_dry_run":
        raise ValueError("unsupported Palace config compatibility probe.")
    if (
        palace_identity.get("config_compatibility_semantics")
        != "required_before_solve"
    ):
        raise ValueError(
            "handoff Palace config compatibility must be verified before solving."
        )
    if palace_identity.get("runtime_identity_semantics") != "observed_provenance_only":
        raise ValueError(
            "handoff Palace runtime identity must be provenance-only."
        )
    config_schema = palace_identity.get("config_schema")
    if not isinstance(config_schema, str) or not config_schema:
        raise ValueError("handoff Palace config schema must be non-empty.")
    return config_schema


def _runtime_compatibility(
    root: Path, handoff_metadata: dict[str, Any]
) -> dict[str, Any]:
    config_schema = _compatibility_required(handoff_metadata)
    if config_schema is None:
        return {}
    path = _confined_path(root, RUNTIME_COMPATIBILITY_PATH)
    if not path.is_file():
        return {}
    payload = _read_json(path)
    if payload.get("schema") != RUNTIME_COMPATIBILITY_SCHEMA:
        return payload
    if payload.get("config", {}).get("declared_schema") != config_schema:
        return payload
    return payload


def _runtime_compatibility_verified(
    *,
    root: Path,
    handoff_metadata: dict[str, Any],
    runtime_compatibility: dict[str, Any],
) -> bool:
    config_schema = _compatibility_required(handoff_metadata)
    if config_schema is None:
        return True
    if runtime_compatibility.get("schema") != RUNTIME_COMPATIBILITY_SCHEMA:
        return False
    config = runtime_compatibility.get("config")
    compatibility = runtime_compatibility.get("compatibility")
    if not isinstance(config, dict) or not isinstance(compatibility, dict):
        return False
    config_path = config.get("path")
    if config_path != "config.json" or config.get("declared_schema") != config_schema:
        return False
    expected_config = _record_for_path(root, config_path)
    if expected_config["present"] is not True:
        return False
    if config.get("bytes") != expected_config["bytes"] or config.get(
        "sha256"
    ) != expected_config["sha256"]:
        return False
    return (
        compatibility.get("method") == "native_dry_run"
        and compatibility.get("probe_exit_code") == 0
        and compatibility.get("verified") is True
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a returned Palace run receipt for manual handoff.",
    )
    parser.add_argument("run_dir", type=Path, help="Manual handoff run directory.")
    parser.add_argument("exit_code", type=int, help="Palace process exit code.")
    parser.add_argument("--log-path", dest="log_path", required=True)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument(
        "--execution-stage",
        choices=("config_compatibility", "solver"),
        required=True,
    )
    parser.add_argument("--preflight-exit-code", type=int, required=True)
    parser.add_argument("--solver-exit-code", type=int, default=None)
    parser.add_argument("--tee-exit-code", type=int, default=None)
    parser.add_argument("--diagnostic-after-the-fact", action="store_true")
    parser.add_argument(
        "--receipt-path", default="metadata/palace_returned_run_receipt.json"
    )
    args = parser.parse_args()

    solver_invoked = args.execution_stage == "solver"
    if solver_invoked:
        if args.preflight_exit_code != 0:
            raise ValueError("solver execution requires a successful preflight.")
        if args.solver_exit_code is None:
            raise ValueError("solver execution requires its exact exit code.")
    elif args.solver_exit_code is not None:
        raise ValueError("preflight failure must not declare a solver exit code.")

    root = args.run_dir.expanduser().resolve()
    handoff_metadata = _read_json(root / "metadata" / "palace_handoff_metadata.json")
    route = handoff_metadata.get("route")
    problem = handoff_metadata.get("problem")
    if not isinstance(route, str) or route not in {"A", "B"}:
        raise ValueError("run metadata route is missing or invalid.")
    if not isinstance(problem, str) or not problem:
        raise ValueError("run metadata problem is missing or invalid.")
    input_hashes = _hashes_from_metadata(root)
    handoff_id = handoff_metadata.get("handoff_id")
    if not isinstance(handoff_id, str) or not handoff_id:
        if not args.diagnostic_after_the_fact:
            raise ValueError("run metadata is missing handoff_id.")
        handoff_id = _diagnostic_handoff_id(
            route=route, problem=problem, input_hashes=input_hashes
        )

    solver_identity = _solver_identity(root)
    runtime_compatibility = _runtime_compatibility(root, handoff_metadata)
    compatibility_required = _compatibility_required(handoff_metadata)
    output_paths = [
        *_required_output_paths(problem, args.log_path),
        *_iteration_output_paths(root, problem),
        *(
            [RUNTIME_COMPATIBILITY_PATH]
            if compatibility_required is not None
            else []
        ),
    ]
    output_records = [_record_for_path(root, path) for path in output_paths]
    log_record = _record_for_path(root, args.log_path)
    completed_outputs = all(record["present"] is True for record in output_records)
    completed_inputs = _input_hashes_match(root, input_hashes)
    runtime_compatible = _runtime_compatibility_verified(
        root=root,
        handoff_metadata=handoff_metadata,
        runtime_compatibility=runtime_compatibility,
    )

    if args.diagnostic_after_the_fact:
        status = "diagnostic_unverified"
    elif (
        args.exit_code == 0
        and solver_invoked
        and args.preflight_exit_code == 0
        and args.solver_exit_code == 0
        and args.tee_exit_code == 0
        and completed_inputs
        and completed_outputs
        and runtime_compatible
    ):
        status = "completed"
    else:
        status = "failed"

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "handoff_id": handoff_id,
        "route": route,
        "problem": problem,
        "status": status,
        "exit_code": args.exit_code,
        "execution_stage": args.execution_stage,
        "solver_invoked": solver_invoked,
        "preflight_exit_code": args.preflight_exit_code,
        "solver_exit_code": args.solver_exit_code,
        "tee_exit_code": args.tee_exit_code,
        "identity_verified": status == "completed",
        "timestamp_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "job_identity": {
            "job_id": args.job_id,
            "job_name": args.job_name,
        },
        "input_hashes": input_hashes,
        "output_files": output_records,
        "log": log_record,
        "solver_identity": solver_identity,
    }
    if compatibility_required is not None:
        payload["runtime_compatibility"] = runtime_compatibility
    if handoff_metadata.get("route_a_thin_film") is not None:
        payload["route_a_thin_film"] = handoff_metadata["route_a_thin_film"]

    _update_resource_record(
        root=root,
        handoff_metadata=handoff_metadata,
        status=status,
        exit_code=args.exit_code,
        job_identity=payload["job_identity"],
    )
    _update_run_metadata(
        root=root,
        handoff_metadata=handoff_metadata,
        status=status,
        exit_code=args.exit_code,
        job_identity=payload["job_identity"],
    )
    receipt_path = _confined_path(root, args.receipt_path)
    _write_json_atomic(receipt_path, payload)
    if status == "completed" or args.diagnostic_after_the_fact:
        return 0
    return args.exit_code if args.exit_code != 0 else 3


def _input_hashes_match(root: Path, input_hashes: list[dict[str, Any]]) -> bool:
    for entry in input_hashes:
        path_name = entry.get("path")
        expected_bytes = entry.get("bytes")
        expected_sha = entry.get("sha256")
        if not isinstance(path_name, str) or not isinstance(expected_bytes, int):
            return False
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            return False
        path = _confined_path(root, path_name)
        if (
            not path.is_file()
            or path.stat().st_size != expected_bytes
            or _sha256(path) != expected_sha
        ):
            return False
    return bool(input_hashes)


def _update_resource_record(
    *,
    root: Path,
    handoff_metadata: dict[str, Any],
    status: str,
    exit_code: int,
    job_identity: dict[str, Any],
) -> None:
    path = root / "metadata" / "palace_resource_record.json"
    record = _read_json(path)
    if record.get("handoff_id") != handoff_metadata.get("handoff_id"):
        raise ValueError("resource record handoff identity does not match metadata.")
    record.update(
        {
            "status": status,
            "exit_code": exit_code,
            "job_identity": job_identity,
            "returned_receipt_path": "metadata/palace_returned_run_receipt.json",
        }
    )
    _write_json_atomic(path, record)


def _update_run_metadata(
    *,
    root: Path,
    handoff_metadata: dict[str, Any],
    status: str,
    exit_code: int,
    job_identity: dict[str, Any],
) -> None:
    path = root / "metadata" / "palace_run_metadata.json"
    record = _read_json(path)
    if record.get("handoff_id") != handoff_metadata.get("handoff_id"):
        raise ValueError("run metadata handoff identity does not match metadata.")
    record.update(
        {
            "status": status,
            "exit_code": exit_code,
            "job_identity": job_identity,
            "returned_receipt_path": "metadata/palace_returned_run_receipt.json",
        }
    )
    _write_json_atomic(path, record)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _confined_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("receipt artifact paths must be non-empty relative paths.")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("receipt artifact path escapes the run directory.")
    return candidate


def _run_probe(command: list[str]) -> tuple[int | None, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError:
        return None, ""
    return completed.returncode, completed.stdout


def _resolved_executable(executable: str) -> Path | None:
    candidate = (
        Path(executable)
        if Path(executable).parent != Path(".")
        else Path(shutil.which(executable) or "")
    )
    if not candidate:
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return resolved if resolved.is_file() else None


def _verify_config_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Verify that Palace accepts the sealed config before solving."
    )
    parser.add_argument("executable")
    parser.add_argument("command_style", choices=("binary", "wrapper"))
    parser.add_argument("output_path")
    parser.add_argument("config_path")
    parser.add_argument("--config-schema", required=True)
    args = parser.parse_args(argv)

    version_command = [args.executable]
    if args.command_style == "wrapper":
        version_command.append("-serial")
    version_command.append("--version")
    version_exit_code, version_output = _run_probe(version_command)
    matches = re.findall(r"(?m)^Palace version: ([^\s]+)\s*$", version_output)
    native_git_tag = matches[0] if len(matches) == 1 else None
    runtime_binary = (
        _resolved_executable(args.executable)
        if args.command_style == "binary"
        else None
    )
    binary_identity = runtime_binary.name if runtime_binary is not None else None
    binary_sha256 = _sha256(runtime_binary) if runtime_binary is not None else None

    root = Path.cwd().resolve()
    config_before = _record_for_path(root, args.config_path)
    dry_run_command = [args.executable]
    if args.command_style == "wrapper":
        dry_run_command.append("-serial")
    dry_run_command.extend(["--dry-run", args.config_path])
    dry_run_exit_code, _ = _run_probe(dry_run_command)

    config_record = _record_for_path(root, args.config_path)
    compatible = (
        dry_run_exit_code == 0
        and config_record["present"] is True
        and config_record == config_before
    )
    payload = {
        "schema": RUNTIME_COMPATIBILITY_SCHEMA,
        "config": {
            "path": config_record["path"],
            "bytes": config_record["bytes"],
            "sha256": config_record["sha256"],
            "declared_schema": args.config_schema,
        },
        "compatibility": {
            "method": "native_dry_run",
            "probe_exit_code": dry_run_exit_code,
            "verified": compatible,
        },
        "observed_identity": {
            "native_git_tag": native_git_tag,
            "version_probe_exit_code": version_exit_code,
            "runtime_binary_identity": binary_identity,
            "runtime_binary_sha256": binary_sha256,
        },
        "command": {
            "command_style": args.command_style,
            "executable_identity": Path(args.executable).name,
        },
    }
    _write_json_atomic(_confined_path(root, args.output_path), payload)
    observed_label = native_git_tag if native_git_tag is not None else "unavailable"
    print(
        "SCGSim Palace config compatibility: "
        f"schema={args.config_schema} dry_run_exit={dry_run_exit_code} "
        f"compatible={str(compatible).lower()} native_git_tag={observed_label} "
        f"binary_sha_recorded={str(binary_sha256 is not None).lower()}"
    )
    return 0 if compatible else 78


def _script_main() -> None:
    if sys.argv[1:2] == ["verify-config"]:
        raise SystemExit(_verify_config_main(sys.argv[2:]))
    raise SystemExit(main())


if __name__ == "__main__":
    _script_main()
