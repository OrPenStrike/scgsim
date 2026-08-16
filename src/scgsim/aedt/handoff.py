"""Portable, manual-only handoff preparation for one two-port HFSS CPW run."""

from __future__ import annotations

import shutil
import tarfile
import time
from dataclasses import dataclass, replace
from pathlib import Path

from .spec import OFFICIAL_PYAEDT_SOURCE_URL, HfssSpec
from .util import file_sha256, write_json


@dataclass(frozen=True)
class HandoffPlan:
    """The fixed, portable file set. Preparing it never opens AEDT."""

    run_dir: Path
    script_path: Path
    spec_path: Path
    metadata_path: Path
    receipt_path: Path
    manifest_path: Path
    archive_path: Path


def prepare_handoff(*, spec: HfssSpec, output_dir: str | Path) -> HandoffPlan:
    """Copy and bind a checked public GDS into a new portable run directory."""
    source_gds = spec.gds_path.resolve()
    if not source_gds.is_file():
        raise FileNotFoundError(f"gds_path must be an existing file: {source_gds}")
    preflight = _gds_preflight(source_gds)
    actual_pairs = preflight["polygon_layer_datatypes"]
    declared_pairs = {(item.layer, item.datatype) for item in spec.layer_imports}
    if actual_pairs != declared_pairs:
        raise ValueError(
            f"GDS layer/datatype pairs {sorted(actual_pairs)!r} do not match spec {sorted(declared_pairs)!r}"
        )

    run_dir = Path(output_dir).expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(
            "output_dir must be new; prepared handoffs never reuse directories"
        )
    geometry_dir = run_dir / "geometry"
    metadata_dir = run_dir / "metadata"
    geometry_dir.mkdir(parents=True)
    metadata_dir.mkdir()
    copied_gds = geometry_dir / "design.gds"
    shutil.copy2(source_gds, copied_gds)

    portable_spec = replace(spec, gds_path=Path("geometry/design.gds"))
    script_path = run_dir / "run_aedt.sh"
    spec_path = run_dir / "aedt_spec.json"
    metadata_path = metadata_dir / "aedt_handoff_metadata.json"
    receipt_path = metadata_dir / "aedt_run_receipt.json"
    manifest_path = metadata_dir / "aedt_handoff_manifest.json"
    archive_path = run_dir / "aedt_handoff.tar.gz"
    prepared_at = _utc_now()
    started = time.perf_counter()

    _write_script(script_path)
    payload = portable_spec.to_payload()
    write_json(spec_path, payload)
    write_json(
        metadata_path,
        {
            "schema_version": "scgsim.aedt.handoff.v1",
            "status": "prepared",
            "mode": spec.mode,
            "project": payload["project"],
            "materials": payload["materials"],
            "vacuum_material_id": payload["vacuum_material_id"],
            "run_control": payload["run_control"],
            "pyaedt": payload["pyaedt"],
            "aedt": payload["aedt"],
            "files": {
                "spec": spec_path.name,
                "gds": "geometry/design.gds",
                "receipt": "metadata/aedt_run_receipt.json",
            },
            "gds_sha256": file_sha256(copied_gds),
            "gds_preflight": {
                "polygon_layer_datatypes": [
                    list(pair) for pair in sorted(actual_pairs)
                ],
                "path_count": preflight["path_count"],
                "label_count": preflight["label_count"],
                "labels": "ignored as non-geometry",
            },
            "prepared_at_utc": prepared_at,
            "preparation_seconds": round(time.perf_counter() - started, 6),
        },
    )
    write_json(
        receipt_path,
        {
            "schema_version": "scgsim.aedt.receipt.v1",
            "status": "not_run",
            "mode": spec.mode,
            "requested": {
                "aedt_version": spec.aedt_version,
                "pyaedt_version": spec.pyaedt_version,
                "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
            },
            "pdk_materials": payload["materials"],
            "vacuum_material_id": payload["vacuum_material_id"],
            "source": {
                "spec": spec_path.name,
                "spec_sha256": file_sha256(spec_path),
                "gds": "geometry/design.gds",
                "gds_sha256": file_sha256(copied_gds),
            },
            "outputs": {},
            "prepared_at_utc": prepared_at,
        },
    )
    allowed = (
        script_path,
        spec_path,
        copied_gds,
        metadata_path,
        receipt_path,
        manifest_path,
    )
    write_json(
        manifest_path,
        {
            "schema_version": "scgsim.aedt.handoff-manifest.v1",
            "allowed_paths": [path.relative_to(run_dir).as_posix() for path in allowed],
            "members": [
                _member(path, run_dir) for path in allowed if path != manifest_path
            ],
        },
    )
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in allowed:
            archive.add(
                path, arcname=path.relative_to(run_dir).as_posix(), recursive=False
            )
    return HandoffPlan(
        run_dir,
        script_path,
        spec_path,
        metadata_path,
        receipt_path,
        manifest_path,
        archive_path,
    )


def _gds_preflight(path: Path) -> dict[str, int | set[tuple[int, int]]]:
    try:
        import gdstk
    except ImportError as exc:  # pragma: no cover - dependency contract
        raise RuntimeError(
            "GDS preflight requires the scgsim[aedt] gdstk dependency"
        ) from exc
    path_pairs = _raw_gds_path_datatypes(path)
    if path_pairs:
        raise ValueError(
            "GDS preflight found PATH layer/datatypes "
            f"{sorted(path_pairs)!r}; V1 requires polygonized CPW geometry"
        )
    library = gdstk.read_gds(path)
    path_pairs.update(
        (int(layer), int(datatype))
        for cell in library.cells
        for path_item in cell.paths
        for layer, datatype in zip(path_item.layers, path_item.datatypes, strict=True)
    )
    if path_pairs:
        raise ValueError(
            "GDS preflight found PATH layer/datatypes "
            f"{sorted(path_pairs)!r}; V1 requires polygonized CPW geometry"
        )
    pairs = {
        (polygon.layer, polygon.datatype)
        for cell in library.cells
        for polygon in cell.polygons
    }
    if not pairs:
        raise ValueError("GDS preflight found no polygons")
    by_layer: dict[int, set[int]] = {}
    for layer, datatype in pairs:
        by_layer.setdefault(layer, set()).add(datatype)
    ambiguous = {
        layer: sorted(values) for layer, values in by_layer.items() if len(values) != 1
    }
    if ambiguous:
        raise ValueError(
            f"PyAEDT 1.3 import_gds_3d cannot select GDS datatypes: {ambiguous!r}"
        )
    return {
        "polygon_layer_datatypes": {
            (int(layer), int(datatype)) for layer, datatype in pairs
        },
        "path_count": 0,
        "label_count": sum(len(cell.labels) for cell in library.cells),
    }


def _raw_gds_path_datatypes(path: Path) -> set[tuple[int, int]]:
    """Read PATH records because gdstk expands them to polygons on GDS import."""
    data = path.read_bytes()
    offset = 0
    in_path = False
    layer: int | None = None
    datatype: int | None = None
    result: set[tuple[int, int]] = set()
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError("GDS preflight found a truncated record header")
        size = int.from_bytes(data[offset : offset + 2], "big")
        if size < 4 or offset + size > len(data):
            raise ValueError("GDS preflight found an invalid record length")
        record_type = data[offset + 2]
        payload = data[offset + 4 : offset + size]
        if record_type == 0x09:  # PATH
            in_path, layer, datatype = True, None, None
        elif in_path and record_type == 0x0D and len(payload) == 2:  # LAYER
            layer = int.from_bytes(payload, "big", signed=True)
        elif in_path and record_type == 0x0E and len(payload) == 2:  # DATATYPE
            datatype = int.from_bytes(payload, "big", signed=True)
        elif record_type == 0x11:  # ENDEL
            if in_path:
                if layer is None or datatype is None:
                    raise ValueError("GDS PATH record is missing layer or datatype")
                result.add((layer, datatype))
            in_path = False
        offset += size
    if in_path:
        raise ValueError("GDS preflight found an unterminated PATH record")
    return result


def _write_script(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'run_dir="$(cd "$(dirname "$0")" && pwd)"\n'
        'python -m scgsim.aedt.run --handoff "$run_dir/metadata/aedt_handoff_metadata.json" --execute "$@"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _member(path: Path, root: Path) -> dict[str, str | int]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = ["HandoffPlan", "prepare_handoff"]
