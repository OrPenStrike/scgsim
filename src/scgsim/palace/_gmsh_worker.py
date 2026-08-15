"""Private fresh-process execution boundary for the existing Gmsh mesh path."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scgsim.palace._mesh import _mesh_xao_in_current_process


def main(arguments: list[str] | None = None) -> int:
    """Read one explicit request and write its JSON-serializable mesh result."""
    values = sys.argv[1:] if arguments is None else arguments
    if len(values) != 2:
        raise SystemExit("usage: python -m scgsim.palace._gmsh_worker REQUEST RESULT")
    request_path, result_path = (Path(value) for value in values)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    return _run_request(request, result_path)


def _run_request(request: Any, result_path: Path) -> int:
    """Validate and execute one decoded worker request."""
    if not isinstance(request, Mapping):
        raise TypeError("Gmsh worker request must be a mapping.")
    groups = _mesh_xao_in_current_process(
        xao_path=Path(_required(request, "xao_path", str)),
        route=_required(request, "route", str),
        records=_required(request, "records", list),
        stack=_required(request, "stack", dict),
        mesh_path=Path(_required(request, "mesh_path", str)),
        refined_mesh_size=float(_required(request, "refined_mesh_size", int | float)),
        max_mesh_size=float(_required(request, "max_mesh_size", int | float)),
    )
    result_path.write_text(
        json.dumps({"groups": groups}, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def _required(payload: Mapping[str, Any], key: str, expected: type[Any]) -> Any:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise TypeError(f"Gmsh worker request field {key!r} has an invalid type.")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
