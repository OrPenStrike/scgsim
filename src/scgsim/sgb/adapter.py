"""Frontend adapter contracts that build GeometryBuildInput records."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from math import isfinite, sqrt
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from scgsim.sgb.models import (
    GeometryBuildInput,
    LayoutPolygonSpec,
    PathInput,
    PortSheetOverlapRecord,
    PortSheetRegionRecord,
    SemanticEntitySpec,
)

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
    }
    combined_metadata["interface_intents_2d"] = _route_a_sheet_interfaces(
        entities,
        polygons,
    )
    port_sheet_regions = _port_sheet_regions_from_stack_metadata(
        stack_metadata,
        polygons_by_layer=polygons_by_layer,
        host_entities=entities,
        host_polygons=polygons,
    )

    return GeometryBuildInput(
        polygons=tuple(polygons),
        entities=tuple(entities),
        port_sheet_regions=port_sheet_regions,
        solution_regions=dict(solution_regions),
        metadata=combined_metadata,
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

    `layer_stack` must expose `layers` and `dielectrics` like gsim's
    `LayerStack`. Only conductor/via layout layers become semantic metal
    entities in this first slice; solution regions come from `dielectrics`.
    Missing GDS layer, z-range, material, air/vacuum region, or route
    representation data fails fast.

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
    layer_stack: Any,
    *,
    materials: Mapping[str, Any] | None,
    source_gds: Path,
    padding_um: float,
) -> dict[str, Any]:
    layers = getattr(layer_stack, "layers", None)
    if not isinstance(layers, Mapping):
        raise TypeError("layer_stack must expose a mapping 'layers'")

    if not materials:
        raise ValueError("GDSFactory adaptation requires explicit materials")
    solution_regions = _solution_regions_from_layer_stack(
        layer_stack,
        materials=materials,
        padding_um=padding_um,
    )
    layer_records = [
        _semantic_layer_record(name, layer, materials=materials)
        for name, layer in layers.items()
        if _layer_type(layer) in {"conductor", "via"}
    ]
    if not layer_records:
        raise ValueError("layer_stack has no conductor/via layers to adapt")

    return {
        "metadata": {
            "schema": "semantic_geometry_stack_v1",
            "units": "um",
            "source": str(source_gds),
            "adapter": "gdsfactory",
            "material_names": sorted(str(key) for key in (materials or {})),
        },
        "solution_regions": solution_regions,
        "materials": dict(materials),
        "layers": layer_records,
    }


def _solution_regions_from_layer_stack(
    layer_stack: Any,
    *,
    materials: Mapping[str, Any],
    padding_um: float,
) -> dict[str, Any]:
    raw_dielectrics = getattr(layer_stack, "dielectrics", None)
    if not _is_record_sequence(raw_dielectrics):
        raise TypeError("layer_stack must expose sequence 'dielectrics'")

    regions: dict[str, Any] = {}
    for index, raw in enumerate(raw_dielectrics):
        if not isinstance(raw, Mapping):
            raise TypeError("layer_stack.dielectrics items must be mappings")
        material_id = str(raw.get("material") or raw.get("material_id") or "")
        if not material_id:
            raise ValueError("dielectric records must define material")
        semantic_id = str(raw.get("name") or raw.get("domain") or material_id or index)
        z_min = raw.get("zmin", raw.get("z_min_um"))
        z_max = raw.get("zmax", raw.get("z_max_um"))
        if z_min is None or z_max is None:
            raise ValueError(f"solution region {semantic_id!r} needs zmin/zmax")
        is_airbox = raw.get("is_airbox", False)
        if not isinstance(is_airbox, bool):
            raise TypeError(f"solution region {semantic_id!r} is_airbox must be a bool")
        regions[semantic_id] = {
            "role": "solution_region",
            "material_id": material_id,
            "material_kind": _resolve_material_kind(
                raw,
                material_id=material_id,
                materials=materials,
                context=f"solution region {semantic_id!r}",
            ),
            "is_airbox": is_airbox,
            "geometry_kind": "domain",
            "geometry": {
                "domain": semantic_id,
                "padding_um": padding_um,
                "z_min_um": float(z_min),
                "z_max_um": float(z_max),
            },
        }
    return regions


def _semantic_layer_record(
    name: Any,
    layer: Any,
    *,
    materials: Mapping[str, Any],
) -> dict[str, Any]:
    gds_layer = _gds_layer_tuple(layer)
    layer_type = _layer_type(layer)
    z_min = getattr(layer, "zmin", None)
    thickness = getattr(layer, "thickness", None)
    material_id = str(getattr(layer, "material", "") or "")
    if z_min is None or thickness is None:
        raise ValueError(f"layer {name!r} needs zmin/thickness")
    if not material_id:
        raise ValueError(f"layer {name!r} needs material")
    info = getattr(layer, "info", None)
    raw_kind = getattr(layer, "material_kind", None)
    if raw_kind is None and isinstance(info, Mapping):
        raw_kind = info.get("material_kind")
    host_void_semantic_id = getattr(layer, "host_void_semantic_id", None)
    if host_void_semantic_id is None and isinstance(info, Mapping):
        host_void_semantic_id = info.get("host_void_semantic_id")
    if not isinstance(host_void_semantic_id, str) or not host_void_semantic_id:
        raise ValueError(f"layer {name!r} needs explicit host_void_semantic_id")

    semantic_id = str(name)
    is_via = layer_type == "via"
    return {
        "layer": gds_layer[0],
        "datatype": gds_layer[1],
        "semantic_id": semantic_id,
        "role": "metal",
        "material_id": material_id,
        "material_kind": _resolve_material_kind(
            {"material_kind": raw_kind},
            material_id=material_id,
            materials=materials,
            context=f"layer {name!r}",
        ),
        "priority": int(getattr(layer, "mesh_order", 0) or 0),
        "part_role": "bump_body" if is_via else "face_metal",
        "net_id": semantic_id,
        "geometry_kind": "layout_extrusion",
        "host_void_semantic_id": host_void_semantic_id,
        "geometry": {
            "z_um": float(z_min),
            "thickness_um": float(thickness),
            "geometry_source": "gds_polygon",
        },
        "route_representations": (
            {
                "A": "cutout_boundary_shell",
                "B": "cutout_boundary_shell",
                "C": "material_volume",
            }
            if is_via
            else {
                "A": "surface_sheet",
                "B": "cutout_boundary_shell",
                "C": "material_volume",
            }
        ),
        "metadata": {
            "source_layer_name": semantic_id,
            "source_layer_type": layer_type,
        },
    }


def _resolve_material_kind(
    record: Mapping[str, Any],
    *,
    material_id: Any,
    materials: Mapping[str, Any],
    context: str,
) -> str:
    """Resolve one entity kind from explicit record and stack material facts."""
    if not isinstance(material_id, str) or not material_id:
        raise ValueError(f"{context} needs a non-empty material_id")
    material = materials.get(material_id)
    if not isinstance(material, Mapping):
        raise TypeError(f"{context} material_id {material_id!r} is not in materials")
    stack_kind = material.get("kind")
    if stack_kind not in {"vacuum", "dielectric", "conductor"}:
        raise ValueError(
            f"{context} material {material_id!r} needs kind vacuum, dielectric, or conductor"
        )
    record_kind = record.get("material_kind")
    if record_kind is not None and record_kind not in {
        "vacuum",
        "dielectric",
        "conductor",
    }:
        raise ValueError(
            f"{context} material_kind must be vacuum, dielectric, or conductor"
        )
    if record_kind is not None and record_kind != stack_kind:
        raise ValueError(
            f"{context} material_kind disagrees with materials[{material_id!r}].kind"
        )
    return str(stack_kind)


def _gds_layer_tuple(layer: Any) -> tuple[int, int]:
    value = getattr(layer, "gds_layer", None)
    if value is None:
        value = getattr(layer, "layer", None)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes)
        and len(value) == 2
    ):
        return int(value[0]), int(value[1])
    raise ValueError("layer must define gds_layer as a 2-item tuple")


def _layer_type(layer: Any) -> str:
    value = getattr(layer, "layer_type", None)
    if value is None:
        info = getattr(layer, "info", None)
        if isinstance(info, Mapping):
            value = info.get("layer_type")
    if value is None:
        raise ValueError("layer must define layer_type")
    return str(value)


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


def _is_record_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


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


def _polygons_by_layer(cell: Any) -> dict[tuple[int, int], tuple[Any, ...]]:
    result: dict[tuple[int, int], list[Any]] = {}
    for polygon in cell.get_polygons(apply_repetitions=True):
        result.setdefault(
            (int(polygon.layer), int(polygon.datatype)),
            [],
        ).append(polygon)
    return {key: tuple(value) for key, value in result.items()}


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
        entity_polygons = _derived_ground_polygons(
            entity,
            polygons_by_layer=polygons_by_layer,
            cell_bounds_um=cell_bounds_um,
            domain_bounds_by_semantic_id=domain_bounds_by_semantic_id,
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
    metadata = {
        **dict(entity.metadata),
        "semantic_group_id": entity.semantic_id,
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


def _entity_from_layer_record(
    record: Any, *, materials: Mapping[str, Any]
) -> SemanticEntitySpec:
    if not isinstance(record, Mapping):
        raise TypeError("stack_file 'layers' items must be mappings")

    layer = record.get("layer")
    datatype = record.get("datatype")
    if layer is None or datatype is None:
        raise ValueError("stack_file layer records must define 'layer' and 'datatype'")

    for required_field in ("semantic_id", "role", "material_id"):
        if required_field not in record:
            raise ValueError(f"stack_file layer records must define {required_field!r}")

    raw_geometry = record.get("geometry", {})
    if not isinstance(raw_geometry, Mapping):
        raise TypeError("stack_file layer record 'geometry' must be a mapping")
    geometry = dict(raw_geometry)
    if "z_um" in record or "thickness_um" in record:
        if "z_um" not in record or "thickness_um" not in record:
            raise ValueError(
                "stack_file layer records must define both 'z_um' and "
                "'thickness_um', or use 'geometry'"
            )
        geometry.setdefault("z_um", record["z_um"])
        geometry.setdefault("thickness_um", record["thickness_um"])
    if not geometry:
        raise ValueError(
            "stack_file layer records must define 'geometry' or "
            "'z_um' plus 'thickness_um'"
        )
    geometry.setdefault("gds_layer", layer)
    geometry.setdefault("gds_datatype", datatype)

    polygon_ids = record.get("polygon_ids", ())
    labels = record.get("labels", ())
    if isinstance(polygon_ids, str | bytes) or not isinstance(polygon_ids, Sequence):
        raise TypeError("stack_file layer record 'polygon_ids' must be a sequence")
    if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
        raise TypeError("stack_file layer record 'labels' must be a sequence")

    return SemanticEntitySpec(
        semantic_id=record["semantic_id"],
        role=record["role"],
        material_id=record["material_id"],
        material_kind=_resolve_material_kind(
            record,
            material_id=record["material_id"],
            materials=materials,
            context=f"layer {record['semantic_id']!r}",
        ),
        priority=record.get("priority", 0),
        geometry_kind=record.get("geometry_kind", "layout_extrusion"),
        part_role=record.get("part_role"),
        attached_face_metal_semantic_id=record.get("attached_face_metal_semantic_id"),
        net_id=record.get("net_id"),
        polygon_ids=tuple(polygon_ids),
        labels=tuple(labels),
        host_void_semantic_id=record.get("host_void_semantic_id"),
        requires_construction_body=record.get("requires_construction_body", False),
        route_representations=record.get("route_representations", {}),
        geometry=geometry,
        metadata=record.get("metadata", {}),
    )


def _gds_polygons_for_entity(
    entity: SemanticEntitySpec,
    *,
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
) -> tuple[LayoutPolygonSpec, ...]:
    def selector_point(name: str, value: Any) -> tuple[float, float]:
        if (
            not isinstance(value, Sequence)
            or isinstance(value, str | bytes)
            or len(value) != 2
        ):
            raise ValueError(f"{entity.semantic_id} {name} must be a 2D point")
        return (float(value[0]), float(value[1]))

    def ordered(polygons: Sequence[Any]) -> tuple[Any, ...]:
        return tuple(
            sorted(
                polygons,
                key=lambda polygon: (
                    min(point[0] for point in _ring_from_gdstk_polygon(polygon)),
                    min(point[1] for point in _ring_from_gdstk_polygon(polygon)),
                    polygon.area(),
                ),
            )
        )

    layer = int(entity.geometry["gds_layer"])
    datatype = int(entity.geometry["gds_datatype"])
    candidates = polygons_by_layer.get((layer, datatype), ())
    if not candidates:
        return ()

    selector = entity.geometry.get("selector_point_um")
    if not entity.geometry.get("route_ab_fused_selector_mode", False):
        if selector is not None:
            selected = [
                polygon
                for polygon in candidates
                if _point_in_ring(
                    (float(selector[0]), float(selector[1])),
                    _ring_from_gdstk_polygon(polygon),
                )
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"{entity.semantic_id} selector_point_um matched "
                    f"{len(selected)} polygons"
                )
        else:
            selected = list(candidates)
        return tuple(
            LayoutPolygonSpec(
                polygon_id=f"POLY__{entity.semantic_id}__{index:04d}",
                layer=f"{layer}/{datatype}",
                exterior=_ring_from_gdstk_polygon(polygon),
                object_name=entity.semantic_id,
                net_name=entity.net_id,
            )
            for index, polygon in enumerate(selected)
        )

    excluded = entity.geometry.get("exclude_selector_points_um", ())
    if excluded is None:
        excluded = ()
    if isinstance(excluded, str | bytes) or not isinstance(excluded, Sequence):
        raise TypeError(
            f"{entity.semantic_id} exclude_selector_points_um must be a point sequence"
        )
    components = _fused_gdstk_components(ordered(candidates))
    points = (
        *(
            (selector_point("selector_point_um", selector),)
            if selector is not None
            else ()
        ),
        *(selector_point("exclude_selector_points_um", point) for point in excluded),
    )
    matched: list[int] = []
    for point in points:
        indexes = [
            index
            for index, (polygon, _, _) in enumerate(components)
            if _point_in_layout_polygon(point, polygon)
        ]
        if len(indexes) != 1:
            raise ValueError(
                f"{entity.semantic_id} selector point {point!r} matched "
                f"{len(indexes)} fused polygons"
            )
        if indexes[0] in matched:
            raise ValueError(
                f"{entity.semantic_id} selector points must match distinct polygons"
            )
        matched.append(indexes[0])
    if selector is not None:
        selected = [components[matched[0]]]
    else:
        selected = [
            component
            for index, component in enumerate(components)
            if index not in matched
        ]

    return tuple(
        LayoutPolygonSpec(
            polygon_id=f"{entity.semantic_id}__P{index:04d}",
            layer=f"{layer}/{datatype}",
            exterior=polygon.exterior,
            holes=polygon.holes,
            object_name=entity.semantic_id,
            net_name=entity.net_id,
            metadata={
                "gds_layer": layer,
                "gds_datatype": datatype,
                "source": "gds_polygon",
                "source_polygon_indexes": source_indexes,
                "source_polygon_ids": tuple(
                    f"POLY__{entity.semantic_id}__SOURCE__{source_index:04d}"
                    for source_index in source_indexes
                ),
                "source_area_um2": source_area,
            },
        )
        for index, (polygon, source_indexes, source_area) in enumerate(selected)
    )


def _fused_gdstk_components(
    candidates: Sequence[Any],
) -> tuple[tuple[LayoutPolygonSpec, tuple[int, ...], float], ...]:
    """Return stable occupied components for A/B selector ownership."""
    import gdstk

    from scgsim.sgb.planning import _split_gdstk_cutline_loop

    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    for index, polygon in enumerate(candidates):
        for other_index, other in enumerate(candidates[index + 1 :], index + 1):
            overlap = gdstk.boolean((polygon,), (other,), "and", precision=1e-9)
            if overlap or _rings_share_edge(
                _ring_from_gdstk_polygon(polygon), _ring_from_gdstk_polygon(other)
            ):
                union(index, other_index)

    members: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        members.setdefault(find(index), []).append(index)
    result: list[tuple[LayoutPolygonSpec, tuple[int, ...], float]] = []
    for indexes in members.values():
        regions = gdstk.boolean(
            tuple(candidates[index] for index in indexes),
            (),
            "or",
            precision=1e-9,
        )
        if not regions or len(regions) != 1:
            raise ValueError("fused selector component must lower to one polygon")
        outer, holes = _split_gdstk_cutline_loop(_ring_from_gdstk_polygon(regions[0]))
        source_indexes = tuple(indexes)
        result.append(
            (
                LayoutPolygonSpec(
                    polygon_id="",
                    layer="",
                    exterior=outer,
                    holes=holes,
                ),
                source_indexes,
                sum(float(candidates[index].area()) for index in source_indexes),
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda record: (
                min(point[0] for point in record[0].exterior),
                min(point[1] for point in record[0].exterior),
                record[2],
            ),
        )
    )


def _derived_ground_polygons(
    entity: SemanticEntitySpec,
    *,
    polygons_by_layer: Mapping[tuple[int, int], tuple[Any, ...]],
    cell_bounds_um: Mapping[str, float],
    domain_bounds_by_semantic_id: Mapping[str, Mapping[str, Any]],
) -> tuple[LayoutPolygonSpec, ...]:
    mask_layer = entity.geometry.get("mask_layer")
    if mask_layer is None:
        mask_key = (
            int(entity.geometry["gds_layer"]),
            int(entity.geometry["gds_datatype"]),
        )
    else:
        mask_key = (int(mask_layer[0]), int(mask_layer[1]))
    holes = tuple(
        _ring_from_gdstk_polygon(polygon)
        for polygon in _merged_gdstk_polygons(polygons_by_layer.get(mask_key, ()))
    )
    exterior = _rectangle_ring(
        _ground_plane_bounds(entity, domain_bounds_by_semantic_id, cell_bounds_um)
    )
    return (
        LayoutPolygonSpec(
            polygon_id=f"{entity.semantic_id}__P0000",
            layer=f"{mask_key[0]}/{mask_key[1]}",
            exterior=exterior,
            holes=holes,
            object_name=entity.semantic_id,
            net_name=entity.net_id,
            metadata={
                "gds_layer": mask_key[0],
                "gds_datatype": mask_key[1],
                "source": "die_face_minus_ground_mask",
            },
        ),
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
        source_polygons = polygons_by_layer.get((layer, datatype), ())
        if not source_polygons:
            raise ValueError(
                f"port_sheet_source_layers entry {layer}/{datatype} has no polygons"
            )
        for polygon_index, polygon in enumerate(source_polygons):
            source_polygon = LayoutPolygonSpec(
                polygon_id=f"PORT_SHEET_{layer}_{datatype}__P{polygon_index:04d}",
                layer=f"{layer}/{datatype}",
                exterior=_ring_from_gdstk_polygon(polygon),
                metadata={
                    "gds_layer": layer,
                    "gds_datatype": datatype,
                    "source": "palace_lumped_port_sheet",
                    "source_name": source_name,
                },
            )
            port_sheet_id = f"PORT_SHEET__{source_name}__{polygon_index:04d}"
            overlaps = _port_sheet_overlaps(
                port_sheet_id=port_sheet_id,
                port_polygon=source_polygon,
                target_layer=target_layer,
                host_entity_by_polygon_id=host_entity_by_polygon_id,
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
    return target_layer in {
        f"{entity.geometry.get('gds_layer')}/{entity.geometry.get('gds_datatype')}",
        semantic_group if isinstance(semantic_group, str) else "",
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


def _route_a_sheet_interfaces(
    entities: Sequence[SemanticEntitySpec],
    polygons: Sequence[LayoutPolygonSpec],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    polygons_by_id = {polygon.polygon_id: polygon for polygon in polygons}
    interfaces: list[Mapping[str, Any]] = []
    for entity in entities:
        if entity.route_representations.get("A") != "surface_sheet":
            continue
        for index, polygon_id in enumerate(entity.polygon_ids):
            polygon = polygons_by_id[polygon_id]
            z_um = float(entity.geometry.get("z_um", 0.0))
            interfaces.append(
                {
                    "interface_id": f"MA__{entity.semantic_id}__AIR__{index:04d}",
                    "kind": "MA",
                    "owner_semantic_ids": (entity.semantic_id, "AIR"),
                    "interface_kinds": ("MS", "MA"),
                    "recognition_rule": "route_a_surface_sheet_polygon",
                    "source_polygon_ids": (polygon_id,),
                    "valid_routes": ("A",),
                    "plane": {"axis": "z", "value_um": z_um},
                    "outer_loop": polygon.exterior,
                    "hole_loops": polygon.holes,
                }
            )
    return {"interfaces": tuple(interfaces)}


def _ring_from_gdstk_polygon(polygon: Any) -> tuple[tuple[float, float], ...]:
    points = tuple((float(x), float(y)) for x, y in polygon.points)
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("GDS polygon requires at least 3 unique points")
    return points


def _ring_edges(
    loop: Sequence[tuple[float, float]],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    return tuple(zip(loop, (*loop[1:], loop[0]), strict=True))


def _rings_share_edge(
    left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]
) -> bool:
    return any(
        _shared_segment_length(left_start, left_end, right_start, right_end) > 1e-9
        for left_start, left_end in _ring_edges(left)
        for right_start, right_end in _ring_edges(right)
    )


def _shared_segment_length(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> float:
    first_dx, first_dy = (
        first_end[0] - first_start[0],
        first_end[1] - first_start[1],
    )
    second_dx, second_dy = (
        second_end[0] - second_start[0],
        second_end[1] - second_start[1],
    )
    first_length = (first_dx * first_dx + first_dy * first_dy) ** 0.5
    second_length = (second_dx * second_dx + second_dy * second_dy) ** 0.5
    if first_length <= 1e-9 or second_length <= 1e-9:
        return 0.0
    if (
        abs(
            (second_start[0] - first_start[0]) * first_dy
            - (second_start[1] - first_start[1]) * first_dx
        )
        > 1e-9 * first_length
        or abs(first_dx * second_dy - first_dy * second_dx)
        > 1e-9 * first_length * second_length
    ):
        return 0.0
    axis = 0 if abs(first_dx) >= abs(first_dy) else 1
    first_low, first_high = sorted((first_start[axis], first_end[axis]))
    second_low, second_high = sorted((second_start[axis], second_end[axis]))
    overlap = min(first_high, second_high) - max(first_low, second_low)
    if overlap <= 1e-9:
        return 0.0
    return overlap * first_length / abs((first_dx, first_dy)[axis])


def _point_in_layout_polygon(
    point: tuple[float, float], polygon: LayoutPolygonSpec
) -> bool:
    return _point_in_ring(point, polygon.exterior) and not any(
        _point_in_ring(point, hole) for hole in polygon.holes
    )


def _rectangle_ring(bounds: Mapping[str, float]) -> tuple[tuple[float, float], ...]:
    return (
        (float(bounds["x_min_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_min_um"])),
        (float(bounds["x_max_um"]), float(bounds["y_max_um"])),
        (float(bounds["x_min_um"]), float(bounds["y_max_um"])),
    )


def _point_in_ring(
    point: tuple[float, float],
    ring: Sequence[tuple[float, float]],
) -> bool:
    x, y = point
    inside = False
    j = len(ring) - 1
    for i, (xi, yi) in enumerate(ring):
        xj, yj = ring[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside
