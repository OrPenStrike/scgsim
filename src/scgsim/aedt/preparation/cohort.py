"""Canonical prepared-input path and hash cohort ownership."""

from __future__ import annotations


from collections.abc import Mapping

from pathlib import Path

from scgsim.aedt._io import file_sha256, read_json

from scgsim.aedt.specs.hfss import HfssEprAnalysisSpec, HfssEprSpec

from scgsim.aedt.specs.parse import AedtSpec

from scgsim.aedt.specs.q2d import Q2dSpec

from scgsim.aedt.specs.q3d import Q3dSpec


MANIFEST_PATH = "metadata/aedt_handoff_manifest.json"

GEOMETRY_SOURCE_PATHS = {
    "canonical_gds": "geometry/source.gds",
    "stack": "metadata/geometry_stack.json",
    "trace": "metadata/geometry_trace.json",
}


def canonical_handoff_paths(spec: AedtSpec) -> list[str]:
    paths = ["run_aedt.sh", "aedt_spec.json"]
    if isinstance(spec, HfssEprAnalysisSpec):
        paths.extend(f"saved/{item['path']}" for item in spec.saved_solution.members)
    elif not isinstance(spec, (HfssEprSpec, Q2dSpec, Q3dSpec)):
        paths.append("geometry/design.gds")
        if spec.modeling is not None:
            paths.append("geometry/source.gds")
    paths.extend(
        (
            "metadata/aedt_handoff_metadata.json",
            "metadata/aedt_run_receipt.json",
            MANIFEST_PATH,
        )
    )
    if isinstance(spec, Q3dSpec) and spec.geometry_source is not None:
        paths.extend(GEOMETRY_SOURCE_PATHS.values())
    return paths


def canonical_member_paths(spec: AedtSpec) -> list[str]:
    return [path for path in canonical_handoff_paths(spec) if path != MANIFEST_PATH]


def validate_geometry_source(
    root: Path, receipt_source: Mapping, spec: Q3dSpec
) -> None:
    """Bind the three canonical source attachments and trace to the sealed spec."""
    geometry_source = spec.geometry_source
    if geometry_source is None:
        return
    verified = {}
    for key, expected_path in GEOMETRY_SOURCE_PATHS.items():
        reference = geometry_source["files"][key]
        if reference["path"] != expected_path:
            raise RuntimeError("Q3D geometry_source attachment path is not canonical")
        path = (root / expected_path).resolve()
        if not path.is_relative_to(root.resolve()):
            raise RuntimeError(
                f"Q3D geometry_source attachment escapes run directory: {key}"
            )
        if not path.is_file() or file_sha256(path) != reference["sha256"]:
            raise RuntimeError(f"Q3D geometry_source attachment hash mismatch: {key}")
        verified[key] = path
    if (
        geometry_source["files"]["canonical_gds"]["sha256"]
        != geometry_source["source_gds_sha256"]
        or geometry_source["files"]["stack"]["sha256"]
        != geometry_source["source_stack_sha256"]
    ):
        raise RuntimeError("Q3D geometry_source attachment digests are inconsistent")
    trace = read_json(verified["trace"])
    if not isinstance(trace, Mapping):
        raise RuntimeError("Q3D geometry trace mapping is invalid")
    expected_trace_source = dict(geometry_source)
    expected_trace_source.pop("files")
    if trace.get("geometry_source") != expected_trace_source:
        raise RuntimeError("Q3D geometry trace mapping differs from the sealed spec")


def validate_hfss_import_source(root: Path, raw_payload: Mapping, spec: AedtSpec) -> None:
    """Bind explicit native import lineage to immutable original GDS bytes."""
    if isinstance(spec, (HfssEprSpec, HfssEprAnalysisSpec, Q2dSpec, Q3dSpec)):
        return
    if spec.modeling is None:
        return  # Historical inventories retain their original source contract.
    gds = raw_payload["gds"]
    if gds.get("source_path") != "geometry/source.gds" or gds.get("path") != "geometry/design.gds":
        raise RuntimeError("HFSS source/import GDS paths are not canonical")
    if file_sha256(root / "geometry/source.gds") != gds.get("source_sha256"):
        raise RuntimeError("HFSS original GDS source identity differs")
    keys = ("layer", "datatype", "layer_name", "physical_layer_id", "native_import_layer")
    expected = [{key: item[key] for key in keys} for item in spec.effective_layer_imports]
    if gds.get("import_layer_map") != expected:
        raise RuntimeError("HFSS import layer mapping differs from declared source")
