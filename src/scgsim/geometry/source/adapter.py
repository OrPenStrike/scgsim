"""Lower GDS/stack and component facts into canonical source records. Preserve polygon holes and source IDs; never infer solver ownership."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from math import cos, isfinite, sin, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from scgsim.geometry.models.common import PathInput
from scgsim.geometry.models.input import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    SemanticEntitySpec,
)
from scgsim.geometry.models.regions import PortSheetOverlapRecord, PortSheetRegionRecord
from scgsim.geometry.source._occurrence import (
    _entity_from_layer_record,
    _fused_gdstk_components,
    _gds_polygons_for_entity,
    _is_record_sequence,
    _occurrence_ground_polygons_for_entity,
    _occurrence_include_polygon,
    _occurrence_polygons_for_entity,
    _occurrence_port_polygon,
    _point_in_layout_polygon,
    _point_in_ring,
    _polygons_by_layer,
    _resolve_material_kind,
    _ring_from_gdstk_polygon,
)
from scgsim.geometry.source.stack import build_component_stack

if TYPE_CHECKING:
    from gdsfactory import Component
    from gdsfactory.technology import LayerStack


def build_gds_stack_geometry_input(
    *,
    gds_file: PathInput,
    stack_file: PathInput,
    top_cell_name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GeometryBuildInput:
    """Level 0 adapter: build GeometryBuildInput from GDS plus stack JSON.

    This is the canonical frontend contract for v1. Tool-specific adapters
    such as GDSFactory, gsim, or direct KLayout support should
    lower into this same semantic stack shape instead of inventing their own
    geometry semantics.

    `gds_file` is the layout source path. The adapter must load this file,
    select `top_cell_name` when provided, apply an explicit hierarchy/flattening
    policy, and convert layer/datatype shapes to `LayoutPolygonSpec` records
    without leaking raw KLayout objects.

    `top_cell_name` optionally selects the GDS cell to adapt. If it is omitted
    and a later KLayout-backed extraction sees multiple plausible top cells, it
    should fail fast rather than guessing silently.

    `stack_file` is a path to a project stackup JSON file. It is the reviewed
    semantic contract for layer/datatype, material, z-range, route
    representation, and interface-recognition metadata. Unsupported suffixes
    fail clearly.

    Minimal JSON schema for this first slice:

    - `layers`: sequence of layer records. Each record maps one GDS
      `layer`/`datatype` pair to a semantic entity using `semantic_id`, `role`,
      `material_id`, optional `priority`, optional `geometry_kind`, optional
      `part_role`, optional `net_id`, optional `polygon_ids`, optional `labels`,
      optional `attached_face_metal_semantic_id`, optional
      `route_representations`, optional `host_void_semantic_id`, and either
      `z_um` plus `thickness_um` or a `geometry` mapping.
    - `solution_regions`: mapping from solution-region semantic id, such as
      `AIR` or `substrate`, to metadata. Each region may define `material_id`,
      `priority`, `geometry_kind`, and `geometry`; missing values default to
      the semantic id, `0`, `domain`, and the metadata mapping itself.
    - `metadata.port_sheet_source_layers`: optional sequence of Palace
      lumped-port sheet source layers. These become `PortSheetRegionRecord`s,
      not backend-live `SurfacePlanRecord`s.
    - `metadata`: optional mapping copied into `GeometryBuildInput.metadata`.

    `metadata` is copied to `GeometryBuildInput.metadata` for adapter version,
    source GDS/stack file paths, selected top cell, unit convention, KLayout
    database unit, flattening policy, and stack-file dialect/provenance.

    The implementation must return a fully frontend-normalized
    `GeometryBuildInput`: `polygons` from KLayout geometry, `entities` from the
    resolved stack-file material semantics, and `solution_regions` from
    solver-domain definitions such as AIR, substrate, dielectric, or enclosure
    boxes. It must not emit solver config or assume Ansys physical names are the
    final semantic ids.
    """
    import gdstk

    gds_path = Path(gds_file)
    if not gds_path.is_file():
        raise FileNotFoundError(gds_path)

    stack_mapping, stack_path = _load_stack_mapping(stack_file)
    raw_layers = stack_mapping.get("layers")
    if not _is_record_sequence(raw_layers):
        raise TypeError("stack_file must define sequence 'layers'")

    solution_regions = stack_mapping.get("solution_regions")
    if not isinstance(solution_regions, Mapping):
        raise TypeError("stack_file must define mapping 'solution_regions'")
    materials = stack_mapping.get("materials")
    if not isinstance(materials, Mapping):
        raise TypeError("stack_file must define mapping 'materials'")

    stack_metadata = stack_mapping.get("metadata", {})
    if not isinstance(stack_metadata, Mapping):
        raise TypeError("stack_file 'metadata' must be a mapping when provided")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping when provided")

    effective_top_cell_name = top_cell_name or stack_metadata.get("top_cell_name")
    library = gdstk.read_gds(str(gds_path))
    cell = _select_gds_cell(
        library,
        str(effective_top_cell_name) if effective_top_cell_name else None,
    )
    cell_bounds = _cell_bounds_um(cell)
    polygons_by_layer = _polygons_by_layer(cell)
    if stack_metadata.get("component_semantics_schema_version") == 2:
        if (
            stack_metadata.get("gds_sha256")
            != hashlib.sha256(gds_path.read_bytes()).hexdigest()
        ):
            raise ValueError("GeometryPlan GDS identity differs from its paired stack")
        source_cells = {source_cell.name: source_cell for source_cell in library.cells}
        occurrences = stack_metadata.get("source_occurrences")
        if not _is_record_sequence(occurrences):
            raise ValueError("GeometryPlan stack needs source occurrences")
        by_path = {
            row.get("path"): row for row in occurrences if isinstance(row, Mapping)
        }
        if len(by_path) != len(occurrences):
            raise ValueError("GeometryPlan source occurrence paths must be unique")
        root_path = (
            next(
                path for path, row in by_path.items() if row.get("parent_path") is None
            )
            if sum(row.get("parent_path") is None for row in by_path.values()) == 1
            else None
        )
        _validate_plan_occurrences(
            by_path,
            top_cell=cell,
            source_cells=source_cells,
            source_dbu_um=float(library.precision) * 1e6,
        )
        child_reference_indexes = {
            path: tuple(
                child["reference_index"]
                for child in occurrences
                if child["parent_path"] == path
            )
            for path in by_path
        }
        contribution_ids: set[tuple[str, str]] = set()
        physical_planes: set[str] = set()
        planes = {}
        for record in raw_layers:
            if not isinstance(record, Mapping):
                raise ValueError("GeometryPlan layers must be mappings")
            geometry = record.get("geometry", {})
            if geometry.get("geometry_source") == "die_face_minus_ground_mask":
                metadata = record.get("metadata", {})
                level = metadata.get("logical_layer_id")
                if level in physical_planes:
                    raise ValueError(
                        f"multiple ground planes claim physical level {level!r}"
                    )
                physical_planes.add(level)
                planes[record.get("semantic_id")] = record
        for record in raw_layers:
            if not isinstance(record, Mapping) or not isinstance(
                record.get("metadata"), Mapping
            ):
                raise ValueError("GeometryPlan layers need source occurrence metadata")
            path = record["metadata"].get("source_occurrence_path")
            occurrence = by_path.get(path)
            if occurrence is None or occurrence.get("cell_name") not in source_cells:
                raise ValueError("GeometryPlan layer has no recorded source occurrence")
            geometry = record.get("geometry", {})
            source_summaries = []
            for contribution in geometry.get("source_occurrence_includes", ()):
                if not isinstance(contribution, Mapping) or set(contribution) != {
                    "source_occurrence_path",
                    "source_local_id",
                    "level",
                    "layer",
                    "selector_point_um",
                    "polygon",
                }:
                    raise ValueError(
                        "GeometryPlan ground contribution has invalid fields"
                    )
                if record.get("semantic_id") not in planes:
                    raise ValueError(
                        "GeometryPlan ground contribution targets no root derived plane"
                    )
                if contribution["level"] != record["metadata"].get("logical_layer_id"):
                    raise ValueError(
                        "GeometryPlan ground contribution level differs from target plane"
                    )
                if (
                    not isinstance(contribution["source_local_id"], str)
                    or not contribution["source_local_id"]
                    or "/" in contribution["source_local_id"]
                ):
                    raise ValueError(
                        "GeometryPlan ground contribution local ID is invalid"
                    )
                source_path = contribution.get("source_occurrence_path")
                source_occurrence = by_path.get(source_path)
                if (
                    source_occurrence is None
                    or source_occurrence.get("cell_name") not in source_cells
                ):
                    raise ValueError(
                        "GeometryPlan ground contribution has no source occurrence"
                    )
                key = (source_path, contribution["source_local_id"])
                if key in contribution_ids:
                    raise ValueError(
                        "GeometryPlan ground contribution local ID is duplicated"
                    )
                contribution_ids.add(key)
                expected = _occurrence_include_polygon(
                    contribution,
                    cell=source_cells[source_occurrence["cell_name"]],
                    transform=source_occurrence["transform"],
                    excluded_reference_indexes=child_reference_indexes[source_path],
                )
                if contribution.get("polygon") != expected:
                    raise ValueError(
                        "GeometryPlan ground contribution differs from GDS source"
                    )
                source_summaries.append(
                    {
                        key: contribution[key]
                        for key in (
                            "source_occurrence_path",
                            "source_local_id",
                            "level",
                            "layer",
                        )
                    }
                )
            if (
                record["metadata"].get("ground_plane_contribution_sources", [])
                != source_summaries
            ):
                raise ValueError(
                    "GeometryPlan ground contribution lineage metadata differs"
                )
            if geometry.get("geometry_source", "gds_polygon") != "gds_polygon" and not (
                geometry.get("geometry_source") == "die_face_minus_ground_mask" and path != root_path
            ):
                continue
            local_ground = (geometry.get("geometry_source") == "die_face_minus_ground_mask"
                            and path != root_path)
            selector = (_occurrence_ground_polygons_for_entity if local_ground
                        else _occurrence_polygons_for_entity)
            expected = selector(
                _entity_from_layer_record(record, materials=materials),
                cell=source_cells[occurrence["cell_name"]],
                transform=occurrence["transform"],
                excluded_reference_indexes=child_reference_indexes[path],
                **({"plane_layer": solution_regions[geometry["plane_bounds_ref"]]
                    ["metadata"]["source_layer_datatype"]} if local_ground else {}),
            )
            if record["geometry"].get("source_occurrence_polygons_um") != expected:
                raise ValueError(
                    f"GeometryPlan polygon source differs for {record['semantic_id']!r}"
                )
        for record in stack_metadata.get("port_sheet_source_layers", ()):
            path = (
                record.get("source_occurrence_path")
                if isinstance(record, Mapping)
                else None
            )
            occurrence = by_path.get(path)
            if occurrence is None or occurrence.get("cell_name") not in source_cells:
                raise ValueError("GeometryPlan port has no recorded source occurrence")
            expected = _occurrence_port_polygon(
                record,
                cell=source_cells[occurrence["cell_name"]],
                transform=occurrence["transform"],
                excluded_reference_indexes=child_reference_indexes[path],
            )
            if record.get("source_occurrence_polygon_um") != expected:
                raise ValueError(
                    f"GeometryPlan port source differs for {record.get('name')!r}"
                )
        _validate_plan_ground_contribution_ownership(raw_layers)

    polygons: list[LayoutPolygonSpec] = []
    entities: list[SemanticEntitySpec] = [
        _solution_region_entity_from_record(
            semantic_id,
            record,
            materials=materials,
            cell_bounds_um=cell_bounds,
        )
        for semantic_id, record in solution_regions.items()
    ]
    domain_bounds_by_semantic_id = {
        entity.semantic_id: entity.geometry["domain_bounds_um"]
        for entity in entities
        if isinstance(entity.geometry.get("domain_bounds_um"), Mapping)
    }

    for record in raw_layers:
        for entity, entity_polygons in _entities_and_polygons_from_layer_record(
            record,
            materials=materials,
            polygons_by_layer=polygons_by_layer,
            cell_bounds_um=cell_bounds,
            domain_bounds_by_semantic_id=domain_bounds_by_semantic_id,
        ):
            polygons.extend(entity_polygons)
            entities.append(entity)

    combined_metadata = {
        **dict(stack_metadata),
        **dict(metadata or {}),
        "adapter": "gds_stack",
        "gds_file": str(gds_path),
        "stack_file": str(stack_path),
        "selected_cell_name": cell.name,
        "cell_bounds_um": cell_bounds,
        "source_dbu_um": float(library.precision) * 1e6,
    }
    from scgsim.geometry.source.intents import _route_a_sheet_interfaces

    combined_metadata["interface_intents_2d"] = _route_a_sheet_interfaces(
        entities, polygons
    )
    port_sheet_regions = _port_sheet_regions_from_stack_metadata(
        stack_metadata,
        polygons_by_layer=polygons_by_layer,
        host_entities=entities,
        host_polygons=polygons,
    )

    from scgsim.geometry.source.curves import boundary_binding

    return GeometryBuildInput(
        polygons=tuple(polygons),
        entities=tuple(entities),
        port_sheet_regions=port_sheet_regions,
        solution_regions=dict(solution_regions),
        metadata=combined_metadata,
        boundary_curves=tuple(
            boundary_binding(value)
            for value in stack_metadata.get("boundary_curves", ())
        ),
        boundary_reconstruction=tuple(
            boundary_binding(value, reconstruction=True)
            for value in stack_metadata.get("boundary_reconstruction", ())
        ),
    )


def build_gdsfactory_geometry_input(
    *,
    component: Component,
    layer_stack: LayerStack,
    materials: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    top_cell_name: str | None = None,
    work_dir: PathInput | None = None,
    padding_um: float = 100.0,
) -> GeometryBuildInput:
    """Build GeometryBuildInput from GDSFactory layout and technology objects.

    This adapter intentionally lowers GDSFactory/gsim objects into the reviewed
    Level-0 contract: a GDS file plus stack JSON, then delegates to
    `build_gds_stack_geometry_input`. The adapter does not invent a second
    semantic language.

    `layer_stack.layers[*].info` supplies PDK-owned material, fabrication,
    host-volume, and 3D-integration semantics. `component.info` supplies
    component-owned conductor identities, nets, and topology selectors.
    Missing or mixed authority fails before geometry lowering.

    `work_dir` makes the generated GDS and stack JSON reviewable. Without it,
    temporary files are used only long enough to call the Level-0 adapter.
    """
    if materials is not None and not isinstance(materials, Mapping):
        raise TypeError("materials must be a mapping when provided")

    if work_dir is None:
        with TemporaryDirectory() as tmp:
            return _build_gdsfactory_geometry_input_from_dir(
                component=component,
                layer_stack=layer_stack,
                materials=materials,
                metadata=metadata,
                top_cell_name=top_cell_name,
                work_dir=Path(tmp),
                padding_um=padding_um,
            )
    return _build_gdsfactory_geometry_input_from_dir(
        component=component,
        layer_stack=layer_stack,
        materials=materials,
        metadata=metadata,
        top_cell_name=top_cell_name,
        work_dir=Path(work_dir),
        padding_um=padding_um,
    )


def _build_gdsfactory_geometry_input_from_dir(
    *,
    component: Any,
    layer_stack: Any,
    materials: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    top_cell_name: str | None,
    work_dir: Path,
    padding_um: float,
) -> GeometryBuildInput:
    work_dir.mkdir(parents=True, exist_ok=True)
    component_name = str(getattr(component, "name", "component") or "component")
    safe_name = component_name.replace("/", "_").replace(":", "_")
    gds_path = work_dir / f"{safe_name}.gds"
    stack_path = work_dir / f"{safe_name}.stack.json"

    write_gds = getattr(component, "write_gds", None)
    if write_gds is None:
        raise TypeError("component must provide write_gds(path)")
    try:
        write_gds(gds_path)
    except TypeError:
        write_gds(str(gds_path))

    stack_mapping = _semantic_stack_mapping_from_layer_stack(
        component,
        layer_stack,
        materials=materials,
        source_gds=gds_path,
        padding_um=padding_um,
    )
    stack_path.write_text(
        json.dumps(stack_mapping, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return build_gds_stack_geometry_input(
        gds_file=gds_path,
        stack_file=stack_path,
        top_cell_name=top_cell_name or component_name,
        metadata={
            **dict(metadata or {}),
            "adapter": "gdsfactory",
            "component_name": component_name,
            "generated_gds_file": str(gds_path),
            "generated_stack_file": str(stack_path),
        },
    )


def _semantic_stack_mapping_from_layer_stack(
    component: Any,
    layer_stack: Any,
    *,
    materials: Mapping[str, Any] | None,
    source_gds: Path,
    padding_um: float,
) -> dict[str, Any]:
    stack = build_component_stack(
        component=component,
        layer_stack=layer_stack,
        material_records=materials,  # type: ignore[arg-type]
        coupon_padding_um=padding_um,
    )
    stack["metadata"].update(
        {
            "schema": "semantic_geometry_stack_v1",
            "units": "um",
            "source": str(source_gds),
            "adapter": "gdsfactory",
            "material_names": sorted(str(key) for key in (materials or {})),
        }
    )
    return stack


def _load_stack_mapping(
    stack_file: PathInput,
) -> tuple[Mapping[str, Any], Path]:
    stack_path = Path(stack_file)
    if not stack_path.is_file():
        raise FileNotFoundError(stack_path)
    if stack_path.suffix.lower() != ".json":
        raise ValueError(
            "unsupported stack_file suffix "
            f"{stack_path.suffix!r}; only JSON is supported for now"
        )

    data = json.loads(stack_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise TypeError("JSON stack_file root must be a mapping")
    return data, stack_path


def _select_gds_cell(library: Any, top_cell_name: str | None) -> Any:
    if top_cell_name is not None:
        for cell in library.cells:
            if cell.name == top_cell_name:
                return cell
        raise ValueError(f"GDS top_cell_name not found: {top_cell_name!r}")

    candidates = [
        cell
        for cell in library.cells
        if cell.name != "$$$CONTEXT_INFO$$$"
        and cell.get_polygons(apply_repetitions=True)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        names = ", ".join(sorted(cell.name for cell in candidates))
        raise ValueError(f"top_cell_name is required; candidates: {names}")

    top_cells = [
        cell
        for cell in library.top_level()
        if cell.get_polygons(apply_repetitions=True)
    ]
    if len(top_cells) == 1:
        return top_cells[0]
    names = ", ".join(sorted(cell.name for cell in top_cells))
    raise ValueError(f"top_cell_name is required; candidates: {names}")


def _cell_bounds_um(cell: Any) -> dict[str, float]:
    bbox = cell.bounding_box()
    if bbox is None:
        raise ValueError(f"GDS cell {cell.name!r} has no bounding box")
    (x_min, y_min), (x_max, y_max) = bbox
    return {
        "x_min_um": float(x_min),
        "y_min_um": float(y_min),
        "x_max_um": float(x_max),
        "y_max_um": float(y_max),
    }


def _validate_plan_occurrences(
    by_path: Mapping[str, Mapping[str, Any]],
    *,
    top_cell: Any,
    source_cells: Mapping[str, Any],
    source_dbu_um: float,
) -> None:
    """Bind every qualified path to one reference in the paired written GDS."""
    roots = [path for path, row in by_path.items() if row.get("parent_path") is None]
    if len(roots) != 1:
        raise ValueError("GeometryPlan GDS needs exactly one recorded root occurrence")
    root = roots[0]
    claimed: set[tuple[str, int]] = set()
    for path, row in by_path.items():
        if not isinstance(path, str) or not path or not isinstance(row, Mapping):
            raise ValueError("GeometryPlan occurrence path is invalid")
        raw_transform = row.get("transform")
        if (
            not _is_record_sequence(raw_transform)
            or len(raw_transform) != 6
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                for value in raw_transform
            )
        ):
            raise ValueError(f"GeometryPlan occurrence {path!r} has invalid transform")
        transform = tuple(float(value) for value in raw_transform)
        if path == root:
            if (
                row.get("cell_name") != top_cell.name
                or row.get("reference_index") is not None
                or any(
                    abs(actual - expected) > 1e-9
                    for actual, expected in zip(
                        transform, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                    )
                )
            ):
                raise ValueError("GeometryPlan root differs from selected GDS top cell")
            continue
        parent_path = path.rpartition("/")[0]
        parent = by_path.get(parent_path)
        index = row.get("reference_index")
        if (
            not parent_path.startswith(root)
            or row.get("parent_path") != parent_path
            or parent is None
            or parent.get("cell_name") not in source_cells
            or row.get("cell_name") not in source_cells
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
        ):
            raise ValueError(
                f"GeometryPlan occurrence {path!r} has no GDS parent reference"
            )
        references = source_cells[parent["cell_name"]].references
        if index >= len(references) or (parent_path, index) in claimed:
            raise ValueError(
                f"GeometryPlan occurrence {path!r} has duplicate or missing GDS reference"
            )
        claimed.add((parent_path, index))
        reference = references[index]
        target = getattr(reference.cell, "name", reference.cell)
        if target != row["cell_name"] or len(reference.repetition.get_offsets()) != 0:
            raise ValueError(
                f"GeometryPlan occurrence {path!r} differs from its GDS reference"
            )
        angle = float(reference.rotation or 0.0)
        scale = float(reference.magnification or 1.0)
        reflection = -1.0 if reference.x_reflection else 1.0
        local = (
            scale * cos(angle),
            -scale * sin(angle) * reflection,
            scale * sin(angle),
            scale * cos(angle) * reflection,
            float(reference.origin[0]),
            float(reference.origin[1]),
        )
        pa, pb, pc, pd, px, py = (float(value) for value in parent["transform"])
        a, b, c, d, x, y = local
        expected = (
            pa * a + pb * c,
            pa * b + pb * d,
            pc * a + pd * c,
            pc * b + pd * d,
            pa * x + pb * y + px,
            pc * x + pd * y + py,
        )
        if any(abs(transform[i] - expected[i]) > 1e-9 for i in range(4)) or any(
            abs(transform[i] - expected[i]) > source_dbu_um / 2 + 1e-9 for i in (4, 5)
        ):
            raise ValueError(
                f"GeometryPlan occurrence {path!r} transform differs from GDS"
            )


def _solution_region_entity_from_record(
    semantic_id: Any,
    record: Any,
    *,
    materials: Mapping[str, Any],
    cell_bounds_um: Mapping[str, float],
) -> SemanticEntitySpec:
    if not isinstance(semantic_id, str):
        raise TypeError("solution region ids must be strings")
    if not isinstance(record, Mapping):
        raise TypeError("stack_file 'solution_regions' values must be mappings")

    geometry = record.get("geometry", record)
    if not isinstance(geometry, Mapping):
        raise TypeError("solution region 'geometry' must be a mapping")

    geometry = dict(geometry)
    padding_um = float(geometry.get("padding_um", 0.0))
    geometry.setdefault(
        "domain_bounds_um",
        {
            "x_min_um": cell_bounds_um["x_min_um"] - padding_um,
            "y_min_um": cell_bounds_um["y_min_um"] - padding_um,
            "x_max_um": cell_bounds_um["x_max_um"] + padding_um,
            "y_max_um": cell_bounds_um["y_max_um"] + padding_um,
        },
    )

    return SemanticEntitySpec(
        semantic_id=semantic_id,
        role=record.get("role", "solution_region"),
        material_id=record.get("material_id", semantic_id),
        material_kind=_resolve_material_kind(
            record,
            material_id=record.get("material_id", semantic_id),
            materials=materials,
            context=f"solution region {semantic_id!r}",
        ),
        priority=record.get("priority", 0),
        geometry_kind=record.get("geometry_kind", "domain"),
        geometry=geometry,
        metadata=record.get("metadata", {}),
    )


def _entities_and_polygons_from_layer_record(
    record: Any,
    *,
    materials: Mapping[str, Any],
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
    cell_bounds_um: Mapping[str, float],
    domain_bounds_by_semantic_id: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[SemanticEntitySpec, tuple[LayoutPolygonSpec, ...]], ...]:
    entity = _entity_from_layer_record(record, materials=materials)
    geometry_source = str(entity.geometry.get("geometry_source", "gds_polygon"))
    if geometry_source == "die_face_minus_ground_mask":
        occurrence_polygons = entity.geometry.get("source_occurrence_polygons_um")
        if occurrence_polygons is not None:
            entity_polygons = tuple(LayoutPolygonSpec(
                polygon_id=f"{entity.semantic_id}__P{index:04d}",
                layer=f"{record['layer']}/{record['datatype']}",
                exterior=region["exterior"], holes=region["holes"],
                object_name=entity.semantic_id, net_name=entity.net_id,
                metadata={"source": "die_face_minus_ground_mask"},
            ) for index, region in enumerate(occurrence_polygons))
        else:
            entity_polygons = _derived_ground_polygons(
                entity,
                polygons_by_layer=polygons_by_layer,
                cell_bounds_um=cell_bounds_um,
                domain_bounds_by_semantic_id=domain_bounds_by_semantic_id,
            )
        if len(entity_polygons) > 1:
            return tuple(
                _split_polygon_entity(entity, polygon, index)
                for index, polygon in enumerate(entity_polygons)
            )
        return (
            (_entity_with_polygon_geometry(entity, entity_polygons), entity_polygons),
        )
    elif geometry_source == "gds_polygon":
        entity_polygons = _gds_polygons_for_entity(
            entity,
            polygons_by_layer=polygons_by_layer,
        )
    else:
        entity_polygons = ()

    if entity.geometry.get("split_polygons_as_entities") or (
        entity.geometry.get("route_ab_fused_selector_mode", False)
        and entity.geometry.get("selector_point_um") is None
    ):
        return tuple(
            _split_polygon_entity(entity, polygon, index)
            for index, polygon in enumerate(entity_polygons)
        )
    entity = _entity_with_polygon_geometry(entity, entity_polygons)
    return ((entity, entity_polygons),)


def _split_polygon_entity(
    entity: SemanticEntitySpec,
    polygon: LayoutPolygonSpec,
    index: int,
) -> tuple[SemanticEntitySpec, tuple[LayoutPolygonSpec, ...]]:
    semantic_id = f"{entity.semantic_id}_{index:04d}"
    split_polygon = LayoutPolygonSpec(
        polygon_id=f"{semantic_id}__P0000",
        layer=polygon.layer,
        exterior=polygon.exterior,
        holes=polygon.holes,
        object_name=semantic_id,
        net_name=polygon.net_name,
        port_name=polygon.port_name,
        metadata={
            **dict(polygon.metadata),
            "source_semantic_id": entity.semantic_id,
            "split_polygon_index": index,
        },
    )
    geometry = {
        **dict(entity.geometry),
        "outer_loop": split_polygon.exterior,
        "hole_loops": split_polygon.holes,
    }
    # Selection lists are consumed by this adapter; duplicating them onto each
    # split entity would turn generated-fill metadata quadratic.
    geometry.pop("include_selector_points_um", None)
    geometry.pop("exclude_selector_points_um", None)
    metadata = {
        **dict(entity.metadata),
        **(
            {"source_semantic_id": entity.semantic_id}
            if "source_occurrence_path" in entity.metadata
            else {"semantic_group_id": entity.semantic_id}
        ),
        "split_polygon_index": index,
    }
    return (
        SemanticEntitySpec(
            semantic_id=semantic_id,
            role=entity.role,
            material_id=entity.material_id,
            material_kind=entity.material_kind,
            priority=entity.priority,
            geometry_kind=entity.geometry_kind,
            part_role=entity.part_role,
            attached_face_metal_semantic_id=entity.attached_face_metal_semantic_id,
            net_id=entity.net_id,
            polygon_ids=(split_polygon.polygon_id,),
            labels=entity.labels,
            host_void_semantic_id=entity.host_void_semantic_id,
            requires_construction_body=entity.requires_construction_body,
            route_representations=entity.route_representations,
            geometry=geometry,
            metadata=metadata,
        ),
        (split_polygon,),
    )


def _validate_plan_ground_contribution_ownership(
    raw_layers: Sequence[Mapping[str, Any]],
) -> None:
    """A ground include cannot consume a separately named local conductor."""
    import gdstk

    for plane in raw_layers:
        for contribution in plane.get("geometry", {}).get(
            "source_occurrence_includes", ()
        ):
            region = contribution["polygon"]
            included = _layout_polygon_region(
                gdstk,
                LayoutPolygonSpec(
                    polygon_id="source_ground_contribution",
                    layer="0/0",
                    exterior=region["exterior"],
                    holes=region["holes"],
                ),
            )
            for other in raw_layers:
                if (
                    other.get("metadata", {}).get("source_occurrence_path")
                    != contribution["source_occurrence_path"]
                    or other["metadata"].get("logical_layer_id")
                    != contribution["level"]
                    or other.get("geometry", {}).get("geometry_source", "gds_polygon")
                    != "gds_polygon"
                ):
                    continue
                for polygon in other["geometry"]["source_occurrence_polygons_um"]:
                    claimed = _layout_polygon_region(
                        gdstk,
                        LayoutPolygonSpec(
                            polygon_id="named_local_conductor",
                            layer="0/0",
                            exterior=polygon["exterior"],
                            holes=polygon["holes"],
                        ),
                    )
                    if gdstk.boolean(included, claimed, "and", precision=1e-9):
                        raise ValueError(
                            f"ground contribution {contribution['source_local_id']!r} "
                            f"also contains local Entity {other['semantic_id']!r}"
                        )


def _derived_ground_polygons(
    entity: SemanticEntitySpec,
    *,
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
    cell_bounds_um: Mapping[str, float],
    domain_bounds_by_semantic_id: Mapping[str, Mapping[str, Any]],
    plane_polygons: Sequence[Any] | None = None,
) -> tuple[LayoutPolygonSpec, ...]:
    import gdstk

    from scgsim.geometry._primitives.loops import (
        _canonical_loop_sort_key,
        _split_gdstk_cutline_loop,
    )

    mask_layer = entity.geometry.get("mask_layer")
    if mask_layer is None:
        mask_key = (
            int(entity.geometry["gds_layer"]),
            int(entity.geometry["gds_datatype"]),
        )
    else:
        mask_key = (int(mask_layer[0]), int(mask_layer[1]))
    if plane_polygons is None:
        exterior = _rectangle_ring(
            _ground_plane_bounds(entity, domain_bounds_by_semantic_id, cell_bounds_um)
        )
        base = (gdstk.Polygon(exterior),)
    else:
        base = tuple(plane_polygons)
    without_mask = gdstk.boolean(
        base,
        _merged_gdstk_polygons(polygons_by_layer.get(mask_key, ())),
        "not",
        precision=1e-9,
    )
    include_layer = entity.geometry.get("include_layer")
    include_points = entity.geometry.get("include_selector_points_um", ())
    if include_layer is None and include_points:
        raise ValueError(
            f"{entity.semantic_id} include_selector_points_um requires include_layer"
        )
    included: list[Any] = []
    source_includes = entity.geometry.get("source_occurrence_includes", ())
    if source_includes and (include_layer is not None or include_points):
        raise ValueError(
            f"{entity.semantic_id} mixes source contributions with global includes"
        )
    for contribution in source_includes:
        polygon = (
            contribution.get("polygon") if isinstance(contribution, Mapping) else None
        )
        if not isinstance(polygon, Mapping):
            raise ValueError(
                f"{entity.semantic_id} has an incomplete source contribution"
            )
        included.extend(
            _layout_polygon_region(
                gdstk,
                LayoutPolygonSpec(
                    polygon_id=f"{entity.semantic_id}__INCLUDE__{len(included):04d}",
                    layer=f"{mask_key[0]}/{mask_key[1]}",
                    exterior=polygon["exterior"],
                    holes=polygon["holes"],
                ),
            )
        )
    if include_layer is not None:
        if (
            not isinstance(include_layer, Sequence)
            or isinstance(include_layer, str | bytes)
            or len(include_layer) != 2
        ):
            raise TypeError(
                f"{entity.semantic_id} include_layer must be a GDS layer pair"
            )
        candidates = polygons_by_layer.get(
            (int(include_layer[0]), int(include_layer[1])), ()
        )
        # The public PDK may represent one logical conductor with adjacent
        # source polygons.  Fuse exact geometric contacts before applying an
        # explicit topology selector; this is mechanical normalization, not
        # identity inference.
        components = _fused_gdstk_components(candidates)
        for point in include_points:
            if (
                not isinstance(point, Sequence)
                or isinstance(point, str | bytes)
                or len(point) != 2
            ):
                raise TypeError(
                    f"{entity.semantic_id} include_selector_points_um requires 2D points"
                )
            selector = (float(point[0]), float(point[1]))
            matches = [
                polygon
                for polygon, _, _ in components
                if _point_in_layout_polygon(selector, polygon)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{entity.semantic_id} include selector {tuple(point)!r} matched {len(matches)} source polygons"
                )
            included.extend(_layout_polygon_region(gdstk, matches[0]))
    regions = gdstk.boolean(without_mask or (), tuple(included), "or", precision=1e-9)
    normalized_regions = sorted(
        (
            _split_gdstk_cutline_loop(_ring_from_gdstk_polygon(region))
            for region in regions or ()
        ),
        key=lambda region: (
            _canonical_loop_sort_key(region[0]),
            tuple(_canonical_loop_sort_key(hole) for hole in region[1]),
        ),
    )
    if not normalized_regions:
        raise ValueError(
            f"{entity.semantic_id} derived ground must lower to at least one planar region"
        )
    return tuple(
        LayoutPolygonSpec(
            polygon_id=f"{entity.semantic_id}__P{index:04d}",
            layer=f"{mask_key[0]}/{mask_key[1]}",
            exterior=outer,
            holes=holes,
            object_name=entity.semantic_id,
            net_name=entity.net_id,
            metadata={
                "gds_layer": mask_key[0],
                "gds_datatype": mask_key[1],
                "source": "die_face_minus_ground_mask",
                "include_layer": include_layer,
            },
        )
        for index, (outer, holes) in enumerate(normalized_regions)
    )


def _merged_gdstk_polygons(polygons: Sequence[Any]) -> tuple[Any, ...]:
    if not polygons:
        return ()
    import gdstk

    merged = gdstk.boolean(
        polygons,
        (),
        "or",
        precision=1e-9,
    )
    return tuple(merged or ())


def _ground_plane_bounds(
    entity: SemanticEntitySpec,
    domain_bounds_by_semantic_id: Mapping[str, Mapping[str, Any]],
    cell_bounds_um: Mapping[str, float],
) -> Mapping[str, Any]:
    plane_bounds_ref = entity.geometry.get("plane_bounds_ref")
    if plane_bounds_ref is None:
        return cell_bounds_um
    if not isinstance(plane_bounds_ref, str):
        raise TypeError(f"{entity.semantic_id} plane_bounds_ref must be a string")
    try:
        return domain_bounds_by_semantic_id[plane_bounds_ref]
    except KeyError as exc:
        raise ValueError(
            f"{entity.semantic_id} plane_bounds_ref {plane_bounds_ref!r} "
            "does not match a solution region"
        ) from exc


def _entity_with_polygon_geometry(
    entity: SemanticEntitySpec,
    polygons: tuple[LayoutPolygonSpec, ...],
) -> SemanticEntitySpec:
    if not polygons:
        return entity
    geometry = dict(entity.geometry)
    geometry.pop("include_selector_points_um", None)
    geometry.pop("exclude_selector_points_um", None)
    if len(polygons) == 1:
        geometry.setdefault("outer_loop", polygons[0].exterior)
        geometry.setdefault("hole_loops", polygons[0].holes)
    polygon_ids = tuple(polygon.polygon_id for polygon in polygons)
    return SemanticEntitySpec(
        semantic_id=entity.semantic_id,
        role=entity.role,
        material_id=entity.material_id,
        material_kind=entity.material_kind,
        priority=entity.priority,
        geometry_kind=entity.geometry_kind,
        part_role=entity.part_role,
        attached_face_metal_semantic_id=entity.attached_face_metal_semantic_id,
        net_id=entity.net_id,
        polygon_ids=polygon_ids,
        labels=entity.labels,
        host_void_semantic_id=entity.host_void_semantic_id,
        requires_construction_body=entity.requires_construction_body,
        route_representations=entity.route_representations,
        geometry=geometry,
        metadata=entity.metadata,
    )


def _port_sheet_regions_from_stack_metadata(
    stack_metadata: Mapping[str, Any],
    *,
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
    host_entities: Sequence[SemanticEntitySpec],
    host_polygons: Sequence[LayoutPolygonSpec],
) -> tuple[PortSheetRegionRecord, ...]:
    records = stack_metadata.get("port_sheet_source_layers", ())
    if records is None:
        return ()
    if not _is_record_sequence(records):
        raise TypeError("metadata.port_sheet_source_layers must be a sequence")

    host_entity_by_polygon_id = {
        polygon_id: entity
        for entity in host_entities
        for polygon_id in entity.polygon_ids
    }
    host_polygons_by_id = {polygon.polygon_id: polygon for polygon in host_polygons}
    port_regions: list[PortSheetRegionRecord] = []
    for source_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError("metadata.port_sheet_source_layers items must be mappings")
        if record.get("source") != "palace_lumped_port_sheet":
            raise ValueError(
                "port_sheet_source_layers currently supports only "
                "source='palace_lumped_port_sheet'"
            )
        if "layer" not in record or "datatype" not in record:
            raise ValueError(
                "metadata.port_sheet_source_layers entries must define "
                "'layer' and 'datatype'"
            )
        source_name = record.get("name")
        if not isinstance(source_name, str) or not source_name:
            raise ValueError("port_sheet_source_layers entries must define name")
        port_index = record.get("port_index")
        if (
            isinstance(port_index, bool)
            or not isinstance(port_index, int)
            or port_index < 1
        ):
            raise ValueError(
                "port_sheet_source_layers entries must define 1-based port_index"
            )
        target_layer = record.get("target_layer")
        if not isinstance(target_layer, str) or not target_layer:
            raise ValueError(
                "port_sheet_source_layers entries must define target_layer"
            )
        direction_raw = record.get("direction")
        if (
            isinstance(direction_raw, str | bytes)
            or not isinstance(direction_raw, Sequence)
            or len(direction_raw) != 3
        ):
            raise ValueError(
                "port_sheet_source_layers entries must define a 3D direction"
            )
        direction = tuple(float(value) for value in direction_raw)
        if (
            not all(isfinite(value) for value in direction)
            or direction[2] != 0.0
            or direction[0] == direction[1] == 0.0
        ):
            raise ValueError(
                "port_sheet_source_layers direction must be finite, XY, and nonzero"
            )
        length = sqrt(direction[0] ** 2 + direction[1] ** 2)
        normalized_direction = (direction[0] / length, direction[1] / length, 0.0)
        sign_convention = record.get("direction_sign_convention", "authored")
        if not isinstance(sign_convention, str) or not sign_convention:
            raise ValueError(
                "port_sheet_source_layers direction_sign_convention must be non-empty"
            )
        layer = int(record["layer"])
        datatype = int(record["datatype"])
        occurrence_polygon = record.get("source_occurrence_polygon_um")
        if occurrence_polygon is not None:
            source_polygons = (
                (
                    0,
                    LayoutPolygonSpec(
                        polygon_id=f"PORT_SHEET__{source_name}__SOURCE__P0000",
                        layer=f"{layer}/{datatype}",
                        exterior=occurrence_polygon["exterior"],
                        holes=occurrence_polygon["holes"],
                        metadata={
                            "gds_layer": layer,
                            "gds_datatype": datatype,
                            "source": "palace_lumped_port_sheet",
                            "source_name": source_name,
                        },
                    ),
                ),
            )
        else:
            all_source_polygons = polygons_by_layer.get((layer, datatype), ())
            selector = record.get("selector_point_um")
            if selector is None:
                selected_polygons = tuple(enumerate(all_source_polygons))
            else:
                if (
                    isinstance(selector, (str, bytes))
                    or not isinstance(selector, Sequence)
                    or len(selector) != 2
                ):
                    raise ValueError("port sheet selector_point_um must be a 2D point")
                point = (float(selector[0]), float(selector[1]))
                selected_polygons = tuple(
                    (index, polygon)
                    for index, polygon in enumerate(all_source_polygons)
                    if _point_in_ring(point, _ring_from_gdstk_polygon(polygon))
                )
                if len(selected_polygons) != 1:
                    raise ValueError(
                        f"port sheet {source_name!r} selector matched "
                        f"{len(selected_polygons)} polygons"
                    )
            source_polygons = tuple(
                (
                    index,
                    LayoutPolygonSpec(
                        polygon_id=f"PORT_SHEET_{layer}_{datatype}__P{index:04d}",
                        layer=f"{layer}/{datatype}",
                        exterior=_ring_from_gdstk_polygon(polygon),
                        metadata={
                            "gds_layer": layer,
                            "gds_datatype": datatype,
                            "source": "palace_lumped_port_sheet",
                            "source_name": source_name,
                        },
                    ),
                )
                for index, polygon in selected_polygons
            )
        if not source_polygons:
            raise ValueError(
                f"port_sheet_source_layers entry {layer}/{datatype} has no polygons"
            )
        for polygon_index, source_polygon in source_polygons:
            port_sheet_id = f"PORT_SHEET__{source_name}__{polygon_index:04d}"
            scoped_hosts = host_entity_by_polygon_id
            if occurrence_polygon is not None:
                occurrence_path = record["source_occurrence_path"]
                scoped_hosts = {
                    polygon_id: entity
                    for polygon_id, entity in host_entity_by_polygon_id.items()
                    if entity.metadata.get("source_occurrence_path") == occurrence_path
                    or str(
                        entity.metadata.get("source_occurrence_path", "")
                    ).startswith(f"{occurrence_path}/")
                    or (
                        entity.geometry.get("geometry_source")
                        == "die_face_minus_ground_mask"
                        and entity.metadata.get("logical_layer_id") == target_layer
                        and any(
                            source.get("source_occurrence_path") == occurrence_path
                            for source in entity.metadata.get(
                                "ground_plane_contribution_sources", ()
                            )
                        )
                    )
                }
            overlaps = _port_sheet_overlaps(
                port_sheet_id=port_sheet_id,
                port_polygon=source_polygon,
                target_layer=target_layer,
                host_entity_by_polygon_id=scoped_hosts,
                host_polygons_by_id=host_polygons_by_id,
            )
            port_regions.append(
                PortSheetRegionRecord(
                    port_sheet_id=port_sheet_id,
                    source_layer=f"{layer}/{datatype}",
                    source_polygon_id=source_polygon.polygon_id,
                    exterior=source_polygon.exterior,
                    holes=source_polygon.holes,
                    overlaps=overlaps,
                    metadata={
                        "source_index": source_index,
                        "source_name": source_name,
                        "port_index": port_index,
                        "target_layer": target_layer,
                        "direction": normalized_direction,
                        "direction_raw": direction,
                        "direction_sign_convention": sign_convention,
                        "source": "palace_lumped_port_sheet",
                        **(
                            {"source_occurrence_path": record["source_occurrence_path"]}
                            if "source_occurrence_path" in record
                            else {}
                        ),
                    },
                )
            )
    return tuple(port_regions)


def _port_sheet_overlaps(
    *,
    port_sheet_id: str,
    port_polygon: LayoutPolygonSpec,
    target_layer: str,
    host_entity_by_polygon_id: Mapping[str, SemanticEntitySpec],
    host_polygons_by_id: Mapping[str, LayoutPolygonSpec],
) -> tuple[PortSheetOverlapRecord, ...]:
    import gdstk

    port_region = _layout_polygon_region(gdstk, port_polygon)
    overlaps: list[PortSheetOverlapRecord] = []
    for host_polygon_id, host_polygon in host_polygons_by_id.items():
        host_entity = host_entity_by_polygon_id.get(host_polygon_id)
        if (
            host_entity is None
            or not _is_port_sheet_host_entity(host_entity)
            or not _port_sheet_target_layer_matches(host_entity, target_layer)
        ):
            continue
        overlap_region = gdstk.boolean(
            port_region,
            _layout_polygon_region(gdstk, host_polygon),
            "and",
            precision=1e-9,
        )
        for index, overlap_polygon in enumerate(overlap_region or ()):
            if abs(float(overlap_polygon.area())) <= 1e-8:
                continue
            overlaps.append(
                PortSheetOverlapRecord(
                    overlap_id=(
                        f"PORT_SHEET_OVERLAP__{port_sheet_id}__"
                        f"{host_entity.semantic_id}__{len(overlaps):04d}"
                    ),
                    port_sheet_id=port_sheet_id,
                    port_polygon_id=port_polygon.polygon_id,
                    host_semantic_id=host_entity.semantic_id,
                    host_polygon_id=host_polygon_id,
                    overlap_loop=_ring_from_gdstk_polygon(overlap_polygon),
                    metadata={"overlap_polygon_index": index},
                )
            )
    return tuple(overlaps)


def _port_sheet_target_layer_matches(
    entity: SemanticEntitySpec, target_layer: str
) -> bool:
    semantic_group = entity.metadata.get("semantic_group_id")
    logical_layer = entity.metadata.get("logical_layer_id")
    return target_layer in {
        f"{entity.geometry.get('gds_layer')}/{entity.geometry.get('gds_datatype')}",
        semantic_group if isinstance(semantic_group, str) else "",
        logical_layer if isinstance(logical_layer, str) else "",
    }


def _is_port_sheet_host_entity(entity: SemanticEntitySpec) -> bool:
    if entity.material_kind != "conductor":
        return False
    if not entity.polygon_ids:
        return False
    geometry_source = str(entity.geometry.get("geometry_source", "gds_polygon"))
    return geometry_source in {"gds_polygon", "die_face_minus_ground_mask"}


def _layout_polygon_region(gdstk: Any, polygon: LayoutPolygonSpec) -> tuple[Any, ...]:
    outer = gdstk.Polygon(polygon.exterior)
    holes = tuple(gdstk.Polygon(hole) for hole in polygon.holes)
    if not holes:
        return (outer,)
    return tuple(gdstk.boolean((outer,), holes, "not", precision=1e-9) or ())


def _rectangle_ring(bounds: Mapping[str, float]) -> tuple[tuple[float, float], ...]:
    return (
        (float(bounds["x_min_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_max_um"])),
        (float(bounds["x_min_um"]), float(bounds["y_max_um"])),
    )
