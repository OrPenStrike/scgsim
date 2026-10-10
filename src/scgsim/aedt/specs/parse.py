"""Explicit AEDT family schema dispatch."""

from __future__ import annotations


from pathlib import Path

from typing import Any

from scgsim.aedt.specs.common import (
    EIGENMODE_SCHEMA_VERSION,
    DRIVEN_GEOMETRY_SCHEMA_VERSION,
    EPR_ANALYSIS_SCHEMA_VERSION,
    EPR_ANALYSIS_SCHEMA_VERSION_V2,
    EPR_EIGENMODE_SCHEMA_VERSION,
    EPR_EIGENMODE_SCHEMA_VERSION_V2,
    EPR_EIGENMODE_SCHEMA_VERSION_V3,
    Q2D_SCHEMA_VERSION,
    Q3D_SCHEMA_VERSION,
    SCHEMA_VERSION,
)

from scgsim.aedt.specs.hfss import (
    HfssDrivenSpec,
    HfssDrivenGeometrySpec,
    HfssEigenmodeSpec,
    HfssEprAnalysisSpec,
    HfssEprSpec,
    HfssSpec,
)

from scgsim.aedt.specs.q2d import Q2dSpec

from scgsim.aedt.specs.q3d import Q3dSpec


AedtSpec = HfssSpec | Q3dSpec | Q2dSpec


def parse_aedt_spec(
    payload: dict[str, Any], *, base_dir: Path | None = None, allow_historical_modeling: bool = False
) -> AedtSpec:
    """Dispatch one explicit AEDT schema without inferring solver family."""
    if payload.get("schema_version") == DRIVEN_GEOMETRY_SCHEMA_VERSION:
        return HfssDrivenGeometrySpec.from_payload(payload)
    if payload.get("schema_version") == SCHEMA_VERSION:
        return HfssDrivenSpec.from_payload(payload, base_dir=base_dir, allow_historical_modeling=allow_historical_modeling)
    if payload.get("schema_version") == EIGENMODE_SCHEMA_VERSION:
        return HfssEigenmodeSpec.from_payload(payload, base_dir=base_dir, allow_historical_modeling=allow_historical_modeling)
    if payload.get("schema_version") in {
        EPR_EIGENMODE_SCHEMA_VERSION,
        EPR_EIGENMODE_SCHEMA_VERSION_V2,
        EPR_EIGENMODE_SCHEMA_VERSION_V3,
    }:
        return HfssEprSpec.from_payload(payload, allow_historical_modeling=allow_historical_modeling)
    if payload.get("schema_version") in {
        EPR_ANALYSIS_SCHEMA_VERSION,
        EPR_ANALYSIS_SCHEMA_VERSION_V2,
    }:
        return HfssEprAnalysisSpec.from_payload(payload, base_dir=base_dir)
    if payload.get("schema_version") == Q3D_SCHEMA_VERSION:
        return Q3dSpec.from_payload(payload, base_dir=base_dir, allow_historical_modeling=allow_historical_modeling)
    if payload.get("schema_version") == Q2D_SCHEMA_VERSION:
        return Q2dSpec.from_payload(payload)
    raise ValueError("unsupported AEDT schema")
