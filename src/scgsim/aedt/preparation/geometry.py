"""GeometryPlan-to-AEDT request preparation adapters."""

from __future__ import annotations


import tarfile

import time

from collections.abc import Mapping, Sequence

from pathlib import Path

from tempfile import TemporaryDirectory

from typing import Literal

from scgsim.geometry import GeometryPlanSnapshot

from scgsim.aedt._io import file_sha256, write_json

from scgsim.aedt.epr.geometry import validate_geometry_workers

from scgsim.aedt.epr.models import (
    EprAnalysisRequest,
    ExpressionCacheConvergence,
    LumpedRlc,
    PreparedPlanarGeometry,
    detached,
)

from scgsim.aedt.preparation.handoff import (
    HandoffPlan,
    _member,
    _utc_now,
    _write_script,
    prepare_handoff,
)

from scgsim.aedt.results.provenance import (
    RECEIPT_V3,
    encode_initial_receipt,
    initial_receipt_payload,
    prepared_runtime_source,
)

from scgsim.aedt.specs.common import (
    AedtResources,
    LOCKED_PYAEDT,
    MatrixRunControl,
    OFFICIAL_PYAEDT_SOURCE_URL,
    PdkMaterial,
    REQUIRED_AEDT_VERSION,
)

from scgsim.aedt.specs.modeling import Modeling

from scgsim.aedt.specs.hfss import (EigenmodeRunControl, HfssEprSpec,
                                     HfssDrivenGeometrySpec, HfssRunControl)
from scgsim.aedt.specs.common import LumpedTerminalPort, TerminalPort


def prepare_q3d_from_geometry(
    snapshot: GeometryPlanSnapshot,
    *,
    modeling: Modeling,
    output_dir: str | Path,
    project_name: str,
    design_name: str,
    materials: Mapping[str, PdkMaterial],
    net_types: Mapping[str, Literal["Signal", "Ground"]],
    physical_ground_nets: Sequence[str],
    run_control: MatrixRunControl,
    region_padding_um: Sequence[float],
    resources: AedtResources | None = None,
) -> HandoffPlan:
    """Prepare finite C/G geometry from detached source facts, without AEDT.

    The caller owns final Net membership and native net types independently
    of physical ground roles. Component/PDK source geometry remains unchanged;
    generated import layers, source trace and originals travel in the handoff.
    """
    from scgsim.aedt.preparation.q3d_geometry import lower_q3d_geometry

    with TemporaryDirectory(prefix="scgsim-q3d-geometry-") as temporary:
        spec = lower_q3d_geometry(
            snapshot,
            modeling=modeling,
            directory=Path(temporary),
            project_name=project_name,
            design_name=design_name,
            materials=materials,
            net_types=net_types,
            physical_ground_nets=physical_ground_nets,
            run_control=run_control,
            region_padding_um=region_padding_um,
        )
        return prepare_handoff(spec=spec, output_dir=output_dir, resources=resources)


def prepare_hfss_eigenmode_from_geometry(
    *,
    modeling: Modeling,
    geometry: PreparedPlanarGeometry,
    project_name: str,
    design_name: str,
    run_control: EigenmodeRunControl,
    output_dir: str | Path,
    epr_request: EprAnalysisRequest | None = None,
    lumped_rlcs: Sequence[LumpedRlc] | None = None,
    expression_convergence: ExpressionCacheConvergence | None = None,
    geometry_workers: int | None = None,
    resources: AedtResources | None = None,
) -> HandoffPlan:
    """Prepare one portable body-first Eigenmode handoff without GDS."""

    if lumped_rlcs is not None:
        geometry = geometry.with_lumped_rlcs(lumped_rlcs)
    spec = HfssEprSpec(
        modeling=modeling,
        project_name=project_name,
        design_name=design_name,
        geometry=geometry,
        run_control=run_control,
        epr_request=epr_request,
        expression_convergence=expression_convergence,
    )
    return _prepare_body_hfss_handoff(
        spec=spec, geometry=geometry, output_dir=output_dir,
        workflow="epr" if epr_request is not None else "body_first_eigenmode",
        geometry_workers=geometry_workers, resources=resources,
    )


def prepare_hfss_driven_from_geometry(
    *, modeling: Modeling, geometry: PreparedPlanarGeometry,
    project_name: str, design_name: str, run_control: HfssRunControl,
    ports: Sequence[TerminalPort | LumpedTerminalPort], output_dir: str | Path,
    geometry_workers: int | None = None, resources: AedtResources | None = None,
) -> HandoffPlan:
    """Prepare a body-first Driven Terminal request with explicit port treatment."""
    spec = HfssDrivenGeometrySpec(
        modeling=modeling, geometry=geometry, project_name=project_name,
        design_name=design_name, run_control=run_control, ports=tuple(ports),
    )
    return _prepare_body_hfss_handoff(
        spec=spec, geometry=geometry, output_dir=output_dir,
        workflow="body_first_driven_terminal", geometry_workers=geometry_workers, resources=resources,
    )


def _prepare_body_hfss_handoff(*, spec, geometry, output_dir, workflow, geometry_workers, resources):
    """One canonical portable cohort writer for body-first HFSS requests."""
    validate_geometry_workers(geometry_workers)
    if resources is not None and not isinstance(resources, AedtResources):
        raise TypeError("resources must be AedtResources or None")
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
        "execution": {
            "geometry_workers": geometry_workers,
            "resources": resources.to_payload() if resources else None,
        },
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
