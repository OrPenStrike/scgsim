"""Portable, manual-only handoff preparation for one AEDT run."""

from __future__ import annotations

import shutil
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._runtime_provenance import (
    RECEIPT_V3,
    encode_initial_receipt,
    initial_receipt_payload,
    prepared_runtime_source,
)
from ._epr_models import (
    EprAnalysisRequest,
    PreparedPlanarGeometry,
    SavedSolution,
    detached,
)
from ._epr_geometry import validate_geometry_workers
from .spec import (
    LOCKED_PYAEDT,
    OFFICIAL_PYAEDT_SOURCE_URL,
    REQUIRED_AEDT_VERSION,
    AedtSpec,
    EigenmodeRunControl,
    HfssEprAnalysisSpec,
    HfssEprSpec,
    Q2dSpec,
)
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


def prepare_handoff(*, spec: AedtSpec, output_dir: str | Path) -> HandoffPlan:
    """Bind every required input into a new portable run directory."""
    source_gds: Path | None = None
    preflight: dict[str, int | set[tuple[int, int]]] | None = None
    if not isinstance(spec, Q2dSpec):
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
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    copied_gds: Path | None = None
    if source_gds is not None:
        geometry_dir = run_dir / "geometry"
        geometry_dir.mkdir()
        copied_gds = geometry_dir / "design.gds"
        shutil.copy2(source_gds, copied_gds)

    script_path = run_dir / "run_aedt.sh"
    spec_path = run_dir / "aedt_spec.json"
    metadata_path = metadata_dir / "aedt_handoff_metadata.json"
    receipt_path = metadata_dir / "aedt_run_receipt.json"
    manifest_path = metadata_dir / "aedt_handoff_manifest.json"
    archive_path = run_dir / "aedt_handoff.tar.gz"
    prepared_at = _utc_now()
    started = time.perf_counter()
    prepared_source = prepared_runtime_source()

    _write_script(script_path)
    payload = spec.to_payload()
    if not isinstance(spec, Q2dSpec):
        payload["gds"]["path"] = "geometry/design.gds"
    write_json(spec_path, payload)
    files = {
        "spec": spec_path.name,
        "receipt": "metadata/aedt_run_receipt.json",
    }
    metadata = {
        "schema_version": "scgsim.aedt.handoff.v1",
        "expected_receipt_schema": RECEIPT_V3,
        "status": "prepared",
        "mode": spec.mode,
        "project": payload["project"],
        "materials": payload["materials"],
        "vacuum_material_id": payload["vacuum_material_id"],
        "run_control": payload["run_control"],
        "pyaedt": payload["pyaedt"],
        "aedt": payload["aedt"],
        "files": files,
        "prepared_at_utc": prepared_at,
        "preparation_seconds": round(time.perf_counter() - started, 6),
    }
    source = {
        "spec": spec_path.name,
        "spec_sha256": file_sha256(spec_path),
    }
    if copied_gds is not None and preflight is not None:
        files["gds"] = "geometry/design.gds"
        metadata["gds_sha256"] = file_sha256(copied_gds)
        metadata["gds_preflight"] = {
            "polygon_layer_datatypes": [
                list(pair) for pair in sorted(preflight["polygon_layer_datatypes"])
            ],
            "path_count": preflight["path_count"],
            "label_count": preflight["label_count"],
            "labels": "ignored as non-geometry",
        }
        source["gds"] = "geometry/design.gds"
        source["gds_sha256"] = file_sha256(copied_gds)
    write_json(metadata_path, metadata)
    receipt_path.write_bytes(
        encode_initial_receipt(
            initial_receipt_payload(
                schema_version=RECEIPT_V3,
                mode=spec.mode,
                requested={
                    "aedt_version": spec.aedt_version,
                    "pyaedt_version": spec.pyaedt_version,
                    "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
                },
                pdk_materials=payload["materials"],
                vacuum_material_id=payload["vacuum_material_id"],
                source=source,
                prepared_runtime_source_value=prepared_source,
                outputs={},
                prepared_at_utc=prepared_at,
            )
        )
    )
    allowed = tuple(
        path
        for path in (
            script_path,
            spec_path,
            copied_gds,
            metadata_path,
            receipt_path,
            manifest_path,
        )
        if path is not None
    )
    write_json(
        manifest_path,
        {
            "schema_version": "scgsim.aedt.handoff-manifest.v1",
            "expected_receipt_schema": RECEIPT_V3,
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


def prepare_hfss_eigenmode_from_geometry(
    *,
    geometry: PreparedPlanarGeometry,
    project_name: str,
    design_name: str,
    run_control: EigenmodeRunControl,
    output_dir: str | Path,
    epr_request: EprAnalysisRequest | None = None,
    geometry_workers: int | None = None,
) -> HandoffPlan:
    """Prepare one portable body-first Eigenmode handoff without GDS."""

    spec = HfssEprSpec(
        project_name=project_name,
        design_name=design_name,
        geometry=geometry,
        run_control=run_control,
        epr_request=epr_request,
    )
    validate_geometry_workers(geometry_workers)
    run_dir = Path(output_dir).expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(
            "output_dir must be new; prepared handoffs never reuse directories"
        )
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    script_path = run_dir / "run_aedt.sh"
    spec_path = run_dir / "aedt_spec.json"
    metadata_path = metadata_dir / "aedt_handoff_metadata.json"
    receipt_path = metadata_dir / "aedt_run_receipt.json"
    manifest_path = metadata_dir / "aedt_handoff_manifest.json"
    archive_path = run_dir / "aedt_handoff.tar.gz"
    prepared_at = _utc_now()
    started = time.perf_counter()
    prepared_source = prepared_runtime_source()
    payload = spec.to_payload()
    materials = detached(geometry.source)["materials"]
    vacuum_ids = sorted(
        material_id
        for material_id, record in materials.items()
        if isinstance(record, dict) and record.get("kind") == "vacuum"
    )
    if len(vacuum_ids) != 1:
        raise ValueError("prepared EPR geometry requires exactly one vacuum material")

    _write_script(script_path)
    write_json(spec_path, payload)
    files = {
        "spec": spec_path.name,
        "receipt": "metadata/aedt_run_receipt.json",
    }
    workflow = "epr" if epr_request is not None else "body_first_eigenmode"
    metadata = {
        "schema_version": "scgsim.aedt.handoff.v2",
        "expected_receipt_schema": RECEIPT_V3,
        "status": "prepared",
        "mode": spec.mode,
        "workflow": workflow,
        "project": payload["project"],
        "materials": materials,
        "vacuum_material_id": vacuum_ids[0],
        "run_control": payload["run_control"],
        "execution": {"geometry_workers": geometry_workers},
        "pyaedt": payload["pyaedt"],
        "aedt": payload["aedt"],
        "files": files,
        "prepared_at_utc": prepared_at,
        "preparation_seconds": round(time.perf_counter() - started, 6),
    }
    source = {
        "spec": spec_path.name,
        "spec_sha256": file_sha256(spec_path),
        "planar_source_sha256": geometry.source_sha256,
        "planar_model_sha256": geometry.model_sha256,
    }
    write_json(metadata_path, metadata)
    receipt_path.write_bytes(
        encode_initial_receipt(
            initial_receipt_payload(
                schema_version=RECEIPT_V3,
                mode=spec.mode,
                requested={
                    "aedt_version": REQUIRED_AEDT_VERSION,
                    "pyaedt_version": LOCKED_PYAEDT,
                    "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
                    "workflow": workflow,
                },
                pdk_materials=materials,
                vacuum_material_id=vacuum_ids[0],
                source=source,
                prepared_runtime_source_value=prepared_source,
                outputs={},
                prepared_at_utc=prepared_at,
            )
        )
    )
    allowed = (
        script_path,
        spec_path,
        metadata_path,
        receipt_path,
        manifest_path,
    )
    write_json(
        manifest_path,
        {
            "schema_version": "scgsim.aedt.handoff-manifest.v2",
            "expected_receipt_schema": RECEIPT_V3,
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


def _prepare_epr_analysis_handoff(
    *,
    saved_solution: SavedSolution,
    geometry: PreparedPlanarGeometry,
    project_name: str,
    design_name: str,
    run_control: EigenmodeRunControl,
    epr_request: EprAnalysisRequest,
    output_dir: str | Path,
    geometry_workers: int | None,
) -> HandoffPlan:
    """Prepare one private workcopy-only saved-field analysis transaction."""

    if not isinstance(saved_solution, SavedSolution):
        raise TypeError("saved_solution must be SavedSolution")
    validate_geometry_workers(geometry_workers)
    run_dir = Path(output_dir).expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(
            "output_dir must be new; prepared handoffs never reuse directories"
        )
    saved_root = run_dir / "saved"
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    copied_paths: list[Path] = []
    for member in saved_solution.members:
        relative = Path(member["path"])
        source = (saved_solution.root / relative).resolve()
        if not source.is_relative_to(saved_solution.root) or source.is_symlink():
            raise RuntimeError("saved solution member escapes its sealed root")
        if (
            not source.is_file()
            or source.stat().st_size != member["bytes"]
            or file_sha256(source) != member["sha256"]
        ):
            raise RuntimeError(f"saved solution member changed: {relative.as_posix()}")
        target = saved_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied_paths.append(target)
    copied_solution = SavedSolution(
        root=saved_root,
        project_path=saved_root
        / saved_solution.project_path.relative_to(saved_solution.root),
        result_path=saved_root
        / saved_solution.result_path.relative_to(saved_solution.root),
        members=saved_solution.members,
        content_sha256=saved_solution.content_sha256,
        identity=saved_solution.identity,
    )
    spec = HfssEprAnalysisSpec(
        project_name=project_name,
        design_name=design_name,
        geometry=geometry,
        run_control=run_control,
        epr_request=epr_request,
        saved_solution=copied_solution,
    )
    script_path = run_dir / "run_aedt.sh"
    spec_path = run_dir / "aedt_spec.json"
    metadata_path = metadata_dir / "aedt_handoff_metadata.json"
    receipt_path = metadata_dir / "aedt_run_receipt.json"
    manifest_path = metadata_dir / "aedt_handoff_manifest.json"
    archive_path = run_dir / "aedt_handoff.tar.gz"
    prepared_at = _utc_now()
    started = time.perf_counter()
    prepared_source = prepared_runtime_source()
    payload = spec.to_payload()
    materials = detached(geometry.source)["materials"]
    vacuum_ids = sorted(
        material_id
        for material_id, record in materials.items()
        if isinstance(record, dict) and record.get("kind") == "vacuum"
    )
    if len(vacuum_ids) != 1:
        raise ValueError("prepared EPR geometry requires exactly one vacuum material")
    _write_analysis_script(script_path)
    write_json(spec_path, payload)
    files = {
        "spec": spec_path.name,
        "receipt": "metadata/aedt_run_receipt.json",
        "saved_project": (
            "saved/"
            + copied_solution.project_path.relative_to(saved_root).as_posix()
        ),
        "saved_results": (
            "saved/" + copied_solution.result_path.relative_to(saved_root).as_posix()
        ),
    }
    metadata = {
        "schema_version": "scgsim.aedt.handoff.v2",
        "expected_receipt_schema": RECEIPT_V3,
        "status": "prepared",
        "mode": spec.mode,
        "workflow": "epr_analysis",
        "project": payload["project"],
        "materials": materials,
        "vacuum_material_id": vacuum_ids[0],
        "run_control": payload["run_control"],
        "execution": {"geometry_workers": geometry_workers},
        "pyaedt": payload["pyaedt"],
        "aedt": payload["aedt"],
        "files": files,
        "prepared_at_utc": prepared_at,
        "preparation_seconds": round(time.perf_counter() - started, 6),
    }
    source = {
        "spec": spec_path.name,
        "spec_sha256": file_sha256(spec_path),
        "planar_source_sha256": geometry.source_sha256,
        "planar_model_sha256": geometry.model_sha256,
        "saved_solution_sha256": copied_solution.content_sha256,
    }
    write_json(metadata_path, metadata)
    receipt_path.write_bytes(
        encode_initial_receipt(
            initial_receipt_payload(
                schema_version=RECEIPT_V3,
                mode=spec.mode,
                requested={
                    "aedt_version": REQUIRED_AEDT_VERSION,
                    "pyaedt_version": LOCKED_PYAEDT,
                    "official_source": OFFICIAL_PYAEDT_SOURCE_URL,
                    "workflow": "epr_analysis",
                },
                pdk_materials=materials,
                vacuum_material_id=vacuum_ids[0],
                source=source,
                prepared_runtime_source_value=prepared_source,
                outputs={},
                prepared_at_utc=prepared_at,
            )
        )
    )
    allowed = (
        script_path,
        spec_path,
        *sorted(copied_paths),
        metadata_path,
        receipt_path,
        manifest_path,
    )
    write_json(
        manifest_path,
        {
            "schema_version": "scgsim.aedt.handoff-manifest.v2",
            "expected_receipt_schema": RECEIPT_V3,
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


def analyze_epr(
    *,
    saved_solution: SavedSolution,
    geometry: PreparedPlanarGeometry,
    project_name: str,
    design_name: str,
    run_control: EigenmodeRunControl,
    epr_request: EprAnalysisRequest,
    output_dir: str | Path,
    geometry_workers: int | None = None,
) -> Any:
    """Analyze one sealed saved-field cohort in an owned disposable workcopy."""

    plan = _prepare_epr_analysis_handoff(
        saved_solution=saved_solution,
        geometry=geometry,
        project_name=project_name,
        design_name=design_name,
        run_control=run_control,
        epr_request=epr_request,
        output_dir=output_dir,
        geometry_workers=geometry_workers,
    )
    from ._epr_results import resolve_epr_result
    from .run import _execute

    previous = Path.cwd()
    try:
        exit_code = _execute(plan.metadata_path, analyze_epr=True)
    finally:
        import os

        os.chdir(previous)
    if exit_code != 0:
        raise RuntimeError(
            "saved EPR analysis failed; inspect the prepared run receipt for the exact native error"
        )
    return resolve_epr_result(plan.run_dir / "results/epr/epr-result.json")


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


def _write_analysis_script(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'run_dir="$(cd "$(dirname "$0")" && pwd)"\n'
        'python -m scgsim.aedt.run --handoff "$run_dir/metadata/aedt_handoff_metadata.json" --execute --analyze-epr "$@"\n',
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


__all__ = [
    "analyze_epr",
    "HandoffPlan",
    "prepare_handoff",
    "prepare_hfss_eigenmode_from_geometry",
]
