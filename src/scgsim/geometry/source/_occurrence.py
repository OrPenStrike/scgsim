"""Lower source Entity and occurrence polygon/port selections for adapter and plan. Never borrow compiler planning helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import isfinite
from typing import Any

from scgsim.geometry.models.input import LayoutPolygonSpec, SemanticEntitySpec
from scgsim.geometry.source._normalization import _gds_layer


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


def _is_record_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _polygons_by_layer(
    cell: Any, *, excluded_reference_indexes: Sequence[int] = ()
) -> dict[tuple[int, int], tuple[Any, ...]]:
    result: dict[tuple[int, int], list[Any]] = {}
    # Canonical public PDK cells use hierarchy.  SGB's Level-0 contract is
    # flattened layout polygons, so preserve every referenced source polygon
    # rather than silently lowering only the top cell's direct shapes.
    if excluded_reference_indexes:
        polygons = list(cell.get_polygons(apply_repetitions=True, depth=0))
        excluded = set(excluded_reference_indexes)
        for index, reference in enumerate(cell.references):
            if index not in excluded:
                polygons.extend(
                    reference.get_polygons(apply_repetitions=True, depth=None)
                )
    else:
        polygons = cell.get_polygons(apply_repetitions=True, depth=None)
    for polygon in polygons:
        result.setdefault(
            (int(polygon.layer), int(polygon.datatype)),
            [],
        ).append(polygon)
    return {key: tuple(value) for key, value in result.items()}


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
    occurrence_polygons = entity.geometry.get("source_occurrence_polygons_um")
    if occurrence_polygons is not None:
        if not _is_record_sequence(occurrence_polygons):
            raise TypeError(
                f"{entity.semantic_id} occurrence polygons must be a sequence"
            )
        return tuple(
            LayoutPolygonSpec(
                polygon_id=f"{entity.semantic_id}__P{index:04d}",
                layer=f"{entity.geometry['gds_layer']}/{entity.geometry['gds_datatype']}",
                exterior=record["exterior"],
                holes=record["holes"],
                object_name=entity.semantic_id,
                net_name=entity.net_id,
                metadata={
                    "source": "gds_source_occurrence",
                    "source_polygon_index": index,
                },
            )
            for index, record in enumerate(occurrence_polygons)
        )

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


def _occurrence_point(
    transform: Sequence[float], point: Sequence[float]
) -> list[float]:
    a, b, c, d, x, y = (float(value) for value in transform)
    px, py = float(point[0]), float(point[1])
    return [a * px + b * py + x, c * px + d * py + y]


def _occurrence_inverse_point(
    transform: Sequence[float], point: Sequence[float]
) -> list[float]:
    a, b, c, d, x, y = (float(value) for value in transform)
    determinant = a * d - b * c
    if not isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError("GeometryPlan occurrence transform is singular")
    px, py = float(point[0]) - x, float(point[1]) - y
    return [(d * px - b * py) / determinant, (-c * px + a * py) / determinant]


def _occurrence_polygon_record(
    polygon: LayoutPolygonSpec, transform: Sequence[float]
) -> dict[str, Any]:
    return {
        "exterior": [_occurrence_point(transform, point) for point in polygon.exterior],
        "holes": [
            [_occurrence_point(transform, point) for point in hole]
            for hole in polygon.holes
        ],
    }


def _occurrence_polygons_for_entity(
    entity: SemanticEntitySpec,
    *,
    cell: Any,
    transform: Sequence[float],
    excluded_reference_indexes: Sequence[int] = (),
) -> list[dict[str, Any]]:
    geometry = dict(entity.geometry)
    geometry.pop("source_occurrence_polygons_um", None)
    for key in ("selector_point_um",):
        if key in geometry:
            geometry[key] = _occurrence_inverse_point(transform, geometry[key])
    for key in ("include_selector_points_um", "exclude_selector_points_um"):
        if key in geometry:
            geometry[key] = [
                _occurrence_inverse_point(transform, point) for point in geometry[key]
            ]
    local_entity = replace(entity, geometry=geometry)
    selected = _gds_polygons_for_entity(
        local_entity,
        polygons_by_layer=_polygons_by_layer(
            cell, excluded_reference_indexes=excluded_reference_indexes
        ),
    )
    return [_occurrence_polygon_record(polygon, transform) for polygon in selected]


def _occurrence_port_polygon(
    record: Mapping[str, Any],
    *,
    cell: Any,
    transform: Sequence[float],
    excluded_reference_indexes: Sequence[int] = (),
) -> dict[str, Any]:
    layer = (record["layer"], record["datatype"])
    point = _occurrence_inverse_point(transform, record["selector_point_um"])
    selected = [
        polygon
        for polygon in _polygons_by_layer(
            cell, excluded_reference_indexes=excluded_reference_indexes
        ).get(layer, ())
        if _point_in_ring(point, _ring_from_gdstk_polygon(polygon))
    ]
    if len(selected) != 1:
        raise ValueError(
            f"GeometryPlan port {record['name']!r} matched {len(selected)} local sheet polygons"
        )
    local = LayoutPolygonSpec(
        polygon_id="source_port_sheet",
        layer=f"{layer[0]}/{layer[1]}",
        exterior=_ring_from_gdstk_polygon(selected[0]),
    )
    return _occurrence_polygon_record(local, transform)


def _occurrence_include_polygon(
    record: Mapping[str, Any],
    *,
    cell: Any,
    transform: Sequence[float],
    excluded_reference_indexes: Sequence[int] = (),
) -> dict[str, Any]:
    """Select one connected local metal region, including its authored pieces."""
    layer = _gds_layer(
        record.get("layer"), str(record.get("source_local_id", "ground contribution"))
    )
    point = _occurrence_inverse_point(transform, record["selector_point_um"])
    components = _fused_gdstk_components(
        _polygons_by_layer(
            cell, excluded_reference_indexes=excluded_reference_indexes
        ).get(layer, ())
    )
    selected = [
        polygon
        for polygon, _, _ in components
        if _point_in_layout_polygon((point[0], point[1]), polygon)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"ground contribution {record.get('source_local_id')!r} matched "
            f"{len(selected)} local source polygons"
        )
    return _occurrence_polygon_record(selected[0], transform)


def _fused_gdstk_components(
    candidates: Sequence[Any],
) -> tuple[tuple[LayoutPolygonSpec, tuple[int, ...], float], ...]:
    """Return stable occupied components for A/B selector ownership."""
    import gdstk

    from scgsim.geometry._primitives.loops import _split_gdstk_cutline_loop

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
        # GDS permits conductor fragments to share an exact edge.  gdstk's
        # boolean union may retain them separately; normalize this mechanical
        # source fact without using layout names or proximity inference.
        regions = gdstk.offset(regions or (), 0.0, join="miter", precision=1e-9)
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
