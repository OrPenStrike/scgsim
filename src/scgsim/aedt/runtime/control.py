"""Linux-local Q3D stop admission; only the recorder owns save and release.

The stable flock inode serializes one stop RPC against Analyze settlement.
A committed intent is never retried, including when its caller dies in the RPC.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from scgsim.aedt._io import file_sha256, read_json
from scgsim.aedt.preparation.handoff import HandoffPlan
from scgsim.aedt.results.provenance import (
    runtime_source_identity,
    validate_runtime_source,
)
from scgsim.aedt.specs.common import REQUIRED_AEDT_VERSION
from scgsim.aedt.specs.parse import parse_aedt_spec
from scgsim.aedt.specs.q3d import Q3dSpec

_SCHEMA = "scgsim.aedt.q3d-stop-control.v1"
_STATE = "q3d_stop_control.json"
_LOCK = "q3d_stop_control.lock"


def _require_linux() -> None:
    if sys.platform != "linux":
        raise RuntimeError("Q3D stop-and-save requires local Linux execution")


def _process_identity(pid: int) -> dict[str, Any]:
    """Read kernel creation identity; comm can itself contain closing parentheses."""
    path = Path("/proc") / str(pid)
    stat = (path / "stat").read_text()
    fields = stat[stat.rindex(")") + 2 :].split()
    if fields[0] in {"Z", "X"}:
        raise RuntimeError("Q3D control process is no longer live")
    return {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "uid": path.stat().st_uid,
    }


def _verify_process(expected: dict[str, Any]) -> None:
    if expected["uid"] != os.getuid() or _process_identity(expected["pid"]) != expected:
        raise RuntimeError("Q3D control process creation identity changed")


@contextmanager
def _locked(path: Path, *, create: bool = False) -> Iterator[None]:
    import fcntl

    flags = os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0)
    fd = os.open(path, flags, 0o600)
    try:
        if os.fstat(fd).st_uid != os.getuid():
            raise RuntimeError("Q3D control lock is not owned by the current user")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _store(path: Path, value: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _metadata_path(handoff: HandoffPlan | str | Path) -> Path:
    if isinstance(handoff, HandoffPlan):
        path = handoff.metadata_path
    else:
        path = Path(handoff)
        if path.is_dir():
            path = path / "metadata/aedt_handoff_metadata.json"
    path = path.resolve()
    if path != path.parent.parent / "metadata/aedt_handoff_metadata.json":
        raise RuntimeError("Q3D stop handoff metadata path is not canonical")
    return path


def _inputs(path: Path) -> tuple[dict[str, str], Q3dSpec]:
    run_dir = path.parent.parent
    spec_path = run_dir / "aedt_spec.json"
    spec = parse_aedt_spec(read_json(spec_path), base_dir=run_dir)
    if not isinstance(spec, Q3dSpec):
        raise RuntimeError("stop-and-save is only supported for Q3D")
    return {
        "metadata_sha256": file_sha256(path),
        "spec_sha256": file_sha256(spec_path),
        "manifest_sha256": file_sha256(run_dir / "metadata/aedt_handoff_manifest.json"),
    }, spec


def _native_identity(desktop: Any, state: dict[str, Any]) -> None:
    expected = state["desktop"]
    _verify_process(expected["process"])
    if (
        desktop.aedt_process_id != expected["process"]["pid"]
        or desktop.port != expected["port"]
        or desktop.odesktop.GetProcessID() != expected["process"]["pid"]
        or desktop.aedt_version_id != REQUIRED_AEDT_VERSION
    ):
        raise RuntimeError("Q3D stop requester did not bind the owned Desktop")
    owner = state["model"]
    if list(desktop.odesktop.GetProjectList()) != [owner["project_name"]]:
        raise RuntimeError("Q3D owned Desktop does not have the sole expected project")
    project = desktop.odesktop.GetActiveProject()
    if (
        project.GetName() != owner["project_name"]
        or (Path(project.GetPath()) / (project.GetName() + ".aedt")).resolve()
        != Path(owner["project_path"])
        or [name.rsplit(";", 1)[-1] for name in project.GetTopDesignList()]
        != [owner["design_name"]]
    ):
        raise RuntimeError("Q3D stop project/design identity changed")
    design = project.GetActiveDesign()
    if (
        design.GetName() != owner["design_name"]
        or design.GetDesignType() != "Q3D Extractor"
        or list(design.GetModule("AnalysisSetup").GetSetups()) != [owner["setup_name"]]
    ):
        raise RuntimeError("Q3D stop design/setup identity changed")


class Q3dStopControl:
    """Recorder-owned control state, published only at the actual Analyze call."""

    def __init__(self, metadata_path: Path, app: Any, runtime_source: dict[str, Any]):
        self.path = metadata_path.parent / _STATE
        self.lock = metadata_path.parent / _LOCK
        self.desktop = app.desktop_class
        self.state: dict[str, Any] | None = None
        self.analysis_call: dict[str, Any] = {"status": "not_started"}
        self.ledger_error: str | None = None
        inputs, spec = _inputs(metadata_path)
        self.initial = {
            "schema_version": _SCHEMA,
            "execution_nonce": uuid.uuid4().hex,
            "handoff": str(metadata_path),
            "inputs": inputs,
            "runtime_source": copy.deepcopy(runtime_source),
            "recorder": _process_identity(os.getpid()),
            "desktop": {
                "process": _process_identity(app.desktop_class.aedt_process_id),
                "port": app.desktop_class.port,
                "version": app.desktop_class.aedt_version_id,
            },
            "model": {
                "project_name": spec.project_name,
                "project_path": str(
                    (
                        metadata_path.parent.parent / (spec.project_name + ".aedt")
                    ).resolve()
                ),
                "design_name": spec.design_name,
                "setup_name": spec.run_control.setup_name,
            },
            "phase": "analysing",
            "intent": None,
        }

    def analyze(self, call: Callable[[], Any]) -> Any:
        try:
            with _locked(self.lock, create=True):
                if self.path.exists():
                    raise RuntimeError(
                        "Q3D control state already exists for this cohort"
                    )
                _store(self.path, self.initial)
        except Exception as exc:
            exc.add_note(
                "Q3D stop-control registration failed before Analyze; solver_invoked=False"
            )
            raise
        try:
            returned = call()
            self.analysis_call = {"status": "returned", "native_return": returned}
            return returned
        except Exception as exc:
            self.analysis_call = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            raise
        finally:
            primary = sys.exc_info()[1]
            try:
                with _locked(self.lock):
                    state = read_json(self.path)
                    if state["execution_nonce"] != self.initial["execution_nonce"]:
                        raise RuntimeError(
                            "Q3D execution nonce changed before settlement"
                        )
                    # Retain committed intent even if closing admission cannot be written.
                    self.state = copy.deepcopy(state)
                    state["phase"] = "closed"
                    state["analysis_call"] = copy.deepcopy(self.analysis_call)
                    if state["intent"] is not None:
                        try:
                            running = self.desktop.odesktop.AreThereSimulationsRunning(
                                True
                            )
                            state["native_settlement"] = {
                                "status": "inactive"
                                if running is False
                                else "active"
                                if running is True
                                else "unavailable",
                                "raw": running,
                                "raw_type": type(running).__name__,
                                "method": "AreThereSimulationsRunning",
                                "arguments": [True],
                            }
                        except Exception as exc:
                            state["native_settlement"] = {
                                "status": "unavailable",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                    self.state = copy.deepcopy(state)
                    _store(self.path, state)
                    self.state = copy.deepcopy(state)
            except Exception as exc:
                self.ledger_error = f"{type(exc).__name__}: {exc}"
                if self.state is not None:
                    self.state["closure_error"] = f"{type(exc).__name__}: {exc}"
                if primary is None:
                    raise
                primary.add_note(f"Q3D stop admission closure failed: {exc}")

    def snapshot(self) -> dict[str, Any]:
        state = copy.deepcopy(self.state or self.initial)
        state["analysis_call"] = copy.deepcopy(self.analysis_call)
        if self.ledger_error is not None:
            state["request_readback"] = {
                "status": "unavailable",
                "error": self.ledger_error,
            }
            if self.state is None:
                state.pop("intent", None)
        return state


def request_q3d_stop_and_save(handoff: HandoffPlan | str | Path) -> dict[str, Any]:
    """Request ONE clean stop of an exact owned Q3D run from another process.

    Save/export/release occur later in the original recorder after Analyze settles.
    This call neither resumes a consumed cohort nor resolves a physics result.
    """
    _require_linux()
    metadata_path = _metadata_path(handoff)
    path, lock = metadata_path.parent / _STATE, metadata_path.parent / _LOCK
    inputs, spec = _inputs(metadata_path)
    with _locked(lock):
        state = read_json(path)
        if state["schema_version"] != _SCHEMA or state["handoff"] != str(metadata_path):
            raise RuntimeError("Q3D stop control state is not bound to this handoff")
        if state["inputs"] != inputs:
            raise RuntimeError("Q3D stop handoff input bytes changed")
        validate_runtime_source(state["runtime_source"], stage="actual")
        actual = runtime_source_identity()
        if any(
            actual[key] != state["runtime_source"][key]
            for key in ("schema_version", "modules", "content_sha256")
        ):
            raise RuntimeError(
                "Q3D stop requester runtime bytes differ from the recorder"
            )
        receipt = read_json(metadata_path.parent / "aedt_run_receipt.json")
        if receipt.get("runtime_source") != state["runtime_source"]:
            raise RuntimeError("Q3D stop control/receipt execution identity differs")
        if state["recorder"]["pid"] == os.getpid():
            raise RuntimeError("Q3D stop requester must be a separate process")
        # Idempotence never attaches or repeats an RPC, even after recorder closure.
        if state["intent"] is not None:
            return copy.deepcopy(state)
        if state["phase"] != "analysing" or receipt.get("status") != "running":
            raise RuntimeError("Q3D stop admission is closed")
        _verify_process(state["recorder"])
        _verify_process(state["desktop"]["process"])
        if (
            state["model"]["project_name"] != spec.project_name
            or state["model"]["design_name"] != spec.design_name
            or state["model"]["setup_name"] != spec.run_control.setup_name
        ):
            raise RuntimeError("Q3D control model differs from the bound spec")
        # Pinned PyAEDT environment overrides must not redirect an exact attach.
        overrides = {
            "PYAEDT_PROCESS_ID": str(state["desktop"]["process"]["pid"]),
            "PYAEDT_DESKTOP_PORT": str(state["desktop"]["port"]),
            "PYAEDT_DESKTOP_VERSION": REQUIRED_AEDT_VERSION,
        }
        for name, expected in overrides.items():
            if os.environ.get(name) not in {None, "", expected}:
                raise RuntimeError(f"{name} redirects the owned Desktop attachment")
        if os.environ.get("PYAEDT_DOC_GENERATION", "false").lower() in {
            "true",
            "1",
            "t",
        }:
            raise RuntimeError("PyAEDT documentation mode would create a new Desktop")
        from scgsim.aedt.runtime.native.common import pyaedt_version

        pyaedt_version()
        from ansys.aedt.core import Desktop
        from ansys.aedt.core.generic.settings import settings
        from ansys.aedt.core.internal.desktop_sessions import _desktop_sessions

        if settings.remote_rpc_session is not None:
            raise RuntimeError(
                "Q3D stop control cannot attach through a remote RPC session"
            )
        if any(pid != state["desktop"]["process"]["pid"] for pid in _desktop_sessions):
            raise RuntimeError(
                "stop requester already has an unrelated Desktop session"
            )
        desktop = Desktop(
            version=REQUIRED_AEDT_VERSION,
            new_desktop=False,
            non_graphical=True,
            close_on_exit=False,
            aedt_process_id=state["desktop"]["process"]["pid"],
            port=state["desktop"]["port"],
        )
        _native_identity(desktop, state)
        _verify_process(state["recorder"])
        _native_identity(desktop, state)
        current_inputs, _ = _inputs(metadata_path)
        current_receipt = read_json(metadata_path.parent / "aedt_run_receipt.json")
        if (
            current_inputs != state["inputs"]
            or current_receipt.get("runtime_source") != state["runtime_source"]
        ):
            raise RuntimeError("Q3D stop input/execution binding changed before RPC")
        if not desktop.are_there_simulations_running:
            raise RuntimeError("Q3D owned Desktop has no active simulation to stop")
        state["intent"] = {
            "action": "stop_and_save",
            "requester": _process_identity(os.getpid()),
            "rpc_status": "unknown",
        }
        _store(path, state)  # Intent is committed immediately before the single RPC.
        try:
            response = desktop.stop_simulations(clean_stop=True)
            state["intent"].update(rpc_status="returned", native_response=response)
        except Exception as exc:
            state["intent"].update(
                rpc_status="error", error=f"{type(exc).__name__}: {exc}"
            )
            try:
                _store(path, state)
            except Exception as storage_error:
                exc.add_note(
                    f"Q3D stop RPC error state could not be stored: {storage_error}"
                )
            raise
        _store(path, state)
        return copy.deepcopy(state)
