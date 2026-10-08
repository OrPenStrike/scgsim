"""Detach, transform and bind caller/component curve declarations to source identities. Prepared Z/material/solution ownership stays explicit."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import replace

from scgsim.geometry.models.input import (
    BoundaryCurveChainSpec,
    GdsBoundaryReconstructionSpec,
    SourceCurveSpec,
)


def source_curve(value) -> SourceCurveSpec:
    if isinstance(value, SourceCurveSpec):
        value = {name: getattr(value, name) for name in value.__dataclass_fields__}
    if not isinstance(value, Mapping):
        raise TypeError("source curve must be a SourceCurveSpec or mapping")
    record = SourceCurveSpec(**dict(value))
    if record.kind not in {
        "line_segment",
        "circular_arc",
        "interpolation_spline",
        "bspline",
    }:
        raise ValueError(f"unsupported source curve kind {record.kind!r}")
    points = tuple(tuple(float(x) for x in point) for point in record.points_um)
    if any(len(point) != 2 for point in points):
        raise ValueError("source curve points must be XY coordinates in micrometres")
    if record.kind == "line_segment" and len(points) != 2:
        raise ValueError("source line needs two endpoints")
    if record.kind == "circular_arc" and len(points) != 3:
        raise ValueError("source circular arc needs start, on-arc and end points")
    return replace(
        record,
        points_um=points,
        weights=tuple(record.weights),
        knots=tuple(record.knots),
        multiplicities=tuple(record.multiplicities),
        parameter_interval=(
            None
            if record.parameter_interval is None
            else tuple(record.parameter_interval)
        ),
    )


def boundary_binding(value, *, reconstruction=False):
    cls = GdsBoundaryReconstructionSpec if reconstruction else BoundaryCurveChainSpec
    if isinstance(value, cls):
        value = {name: getattr(value, name) for name in value.__dataclass_fields__}
    if not isinstance(value, Mapping):
        raise TypeError("boundary binding must be a source record or mapping")
    data = copy.deepcopy(dict(value))
    if not reconstruction:
        data["curves"] = tuple(source_curve(curve) for curve in data["curves"])
    else:
        data["anchor_indices"] = tuple(data.get("anchor_indices", ()))
        data["segment_indices"] = tuple(data.get("segment_indices", ()))
    data["selector_point_um"] = tuple(data["selector_point_um"])
    record = cls(**data)
    if not record.boundary_id or not record.entity_id:
        raise ValueError("boundary and Entity identities must be nonempty")
    if record.role not in {"outer", "hole"}:
        raise ValueError("source boundary role must be outer or hole")
    if record.role == "hole":
        if (
            isinstance(record.hole_index, bool)
            or not isinstance(record.hole_index, int)
            or record.hole_index < 0
        ):
            raise ValueError("hole boundary needs a zero-based source hole index")
    elif record.hole_index is not None:
        raise ValueError("outer boundary cannot have a hole index")
    return record


def transform_boundary(value, *, path, transform, point, reconstruction=False):
    """Qualify local ownership and transform geometry without changing order."""
    record = boundary_binding(value, reconstruction=reconstruction)
    if "/" in record.entity_id or "/" in record.boundary_id:
        raise ValueError(
            "component-authored boundary and Entity IDs must be local names"
        )
    updates = dict(
        boundary_id=f"{path}/{record.boundary_id}",
        entity_id=f"{path}/{record.entity_id}",
        selector_point_um=tuple(point(transform, record.selector_point_um)),
        source_occurrence_path=path,
        metadata={
            **record.metadata,
            "source_local_boundary_id": record.boundary_id,
            "source_transform": tuple(transform),
            "source_orientation": 1
            if transform[0] * transform[3] - transform[1] * transform[2] > 0
            else -1,
        },
    )
    if not reconstruction:
        updates["curves"] = tuple(
            replace(
                curve,
                points_um=tuple(tuple(point(transform, xy)) for xy in curve.points_um),
            )
            for curve in record.curves
        )
    return replace(record, **updates)


def bind_source_curves(prepared_input, source_input):
    """Carry detached authored XY bindings across a prepared-stack Z update.

    Entity/source polygon identities stay authored. Stack preparation owns Z and
    solution regions; this helper never recovers curves from prepared surfaces.
    """
    source_entities = {entity.semantic_id: entity for entity in source_input.entities}
    prepared_entities = {
        entity.semantic_id: entity for entity in prepared_input.entities
    }
    active = {
        record.entity_id
        for record in (
            *source_input.boundary_curves,
            *source_input.boundary_reconstruction,
        )
    }
    # Prepared adapters may split one source Entity into polygon children.
    # Reunite only explicit adapter lineage, never suffixes or geometric guesses.
    split_hosts = {}
    for entity_id in sorted(active - prepared_entities.keys()):
        children = tuple(
            entity
            for entity in prepared_input.entities
            if entity.metadata.get("source_semantic_id") == entity_id
            and "split_polygon_index" in entity.metadata
        )
        if not children:
            raise ValueError("prepared stack lost a curve-bound source Entity")
        first = children[0]
        source = source_entities[entity_id]
        fields = tuple(
            name
            for name in first.__dataclass_fields__
            if name not in {"semantic_id", "polygon_ids", "geometry", "metadata"}
        )
        z_fields = ("z_um", "z_min_um", "z_max_um", "thickness_um")
        if (
            any(
                any(getattr(child, name) != getattr(first, name) for name in fields)
                or any(
                    child.geometry.get(name) != first.geometry.get(name)
                    for name in z_fields
                )
                or child.metadata.get("source_occurrence_path")
                != source.metadata.get("source_occurrence_path")
                for child in children
            )
            or first.net_id != source.net_id
        ):
            raise ValueError(
                "prepared curve-bound Entity children have incompatible ownership"
            )
        metadata = dict(first.metadata)
        metadata.pop("split_polygon_index", None)
        prepared_entities[entity_id] = replace(
            first, semantic_id=entity_id, metadata=metadata
        )
        split_hosts.update((child.semantic_id, entity_id) for child in children)
    polygons = {polygon.polygon_id: polygon for polygon in prepared_input.polygons}
    source_polygons = {polygon.polygon_id: polygon for polygon in source_input.polygons}
    entities = []
    emitted = set()
    for prepared_entity in prepared_input.entities:
        entity_id = split_hosts.get(
            prepared_entity.semantic_id, prepared_entity.semantic_id
        )
        if entity_id in emitted:
            continue
        emitted.add(entity_id)
        entity = prepared_entities[entity_id]
        if entity.semantic_id in active:
            source = source_entities[entity.semantic_id]
            for polygon_id in source.polygon_ids:
                polygons[polygon_id] = source_polygons[polygon_id]
            geometry = dict(entity.geometry)
            for key in ("outer_loop", "hole_loops", "source_occurrence_polygons_um"):
                if key in source.geometry:
                    geometry[key] = copy.deepcopy(source.geometry[key])
            entity = replace(entity, polygon_ids=source.polygon_ids, geometry=geometry)
        entities.append(entity)
    source_ports = {
        region.port_sheet_id: region for region in source_input.port_sheet_regions
    }
    ports = []
    for region in prepared_input.port_sheet_regions:
        overlaps = []
        emitted_hosts = set()
        for overlap in region.overlaps:
            host_id = split_hosts.get(overlap.host_semantic_id)
            if host_id is None:
                overlaps.append(overlap)
                continue
            if host_id in emitted_hosts:
                continue
            original = source_ports.get(region.port_sheet_id)
            matches = (
                ()
                if original is None
                else tuple(
                    item
                    for item in original.overlaps
                    if item.host_semantic_id == host_id
                )
            )
            if not matches:
                raise ValueError(
                    "prepared curve-bound port lost its source host overlap"
                )
            overlaps.extend(copy.deepcopy(matches))
            emitted_hosts.add(host_id)
        ports.append(replace(region, overlaps=tuple(overlaps)))
    return replace(
        prepared_input,
        polygons=tuple(polygons.values()),
        entities=tuple(entities),
        port_sheet_regions=tuple(ports),
        boundary_curves=copy.deepcopy(source_input.boundary_curves),
        boundary_reconstruction=copy.deepcopy(source_input.boundary_reconstruction),
    )
