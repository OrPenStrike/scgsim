"""Own canonical point/curve/loop identity and orientation before native lowering. Shared topology is planned once."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from typing import Any

from scgsim.geometry._primitives.constants import _TOPOLOGY_EPS_UM
from scgsim.geometry._primitives.entities import _unique_ids
from scgsim.geometry._primitives.loops import (
    _clean_loop,
    _ordered_loop_signature,
    _polygon_area,
)
from scgsim.geometry._primitives.spatial import (
    _coordinate_key,
    _geometry_ref_surface_z_um,
)
from scgsim.geometry.models.topology import (
    CurvePlanRecord,
    CurveRefRecord,
    PointPlanRecord,
    SurfaceLoopRecord,
    SurfacePlanRecord,
)
from scgsim.semantics.ownership import (
    project_legacy_interface_record_owners,
    surface_declared_owner_ids,
)


def _surface_interface_record_owners(
    surface: SurfacePlanRecord,
    owners: tuple[str, ...],
) -> tuple[str, ...]:
    """Project structured sheet ownership onto the legacy two-owner record."""
    return project_legacy_interface_record_owners(
        owners,
        surface.metadata.get("boundary_volume_ids", ()),
        surface_id=surface.surface_id,
    )


def _structured_surface_boundary_volume_ids(
    surface: SurfacePlanRecord,
    owners: tuple[str, ...],
) -> tuple[str, ...]:
    raw = surface.metadata.get("boundary_volume_ids", ())
    boundary_ids = () if isinstance(raw, str) else tuple(str(value) for value in raw)
    expected_count = 1 if surface.metadata.get("sheet_contact_cap") else 2
    if len(boundary_ids) != expected_count or not set(boundary_ids).issubset(owners):
        raise ValueError(
            f"{surface.surface_id} structured interface requires {expected_count} ordered "
            "boundary_volume_ids from its owners"
        )
    return _unique_ids(boundary_ids)


def plan_canonical_topology(
    *,
    surfaces: tuple[SurfacePlanRecord, ...],
) -> tuple[
    tuple[PointPlanRecord, ...],
    tuple[CurvePlanRecord, ...],
    tuple[SurfaceLoopRecord, ...],
    tuple[SurfacePlanRecord, ...],
]:
    """Canonicalize planned surface boundaries into compiler-owned topology.

    This is the v1 implementation boundary that turns surface geometry into a
    topology registry. It must:

    - collect every outer/hole/quad boundary from planned surfaces;
    - create one `PointPlanRecord` per unique live coordinate;
    - split collinear overlapping edges and T-junctions into shared atomic
      curves;
    - create `CurvePlanRecord`s and ordered `SurfaceLoopRecord`s;
    - assign `outer_loop_ref` / `hole_loop_refs` on each surface;
    - reject parent-plus-child live overlap after surface partitioning;
    - ensure every interface surface is backed by `InterfacePlanRecord`;
    - reject duplicate live surfaces unless they are intentionally merged into
      one surface id before tagging; and
    - make volume closure checkable without asking OCC to discover topology.

    Raw `geometry_ref` lowering is not a v1 conformal-geometry contract; the
    backend consumes the planned point/curve/surface-loop refs produced here.
    """
    point_ids: dict[tuple[float, float, float], str] = {}
    point_coordinates: dict[str, tuple[float, float, float]] = {}
    point_curve_ids: dict[str, set[str]] = {}
    curve_ids: dict[tuple[str, str], str] = {}
    curve_owner_ids: dict[str, set[str]] = {}
    curve_interface_ids: dict[str, set[str]] = {}
    curve_surface_ids: dict[str, set[str]] = {}
    curve_volume_ids: dict[str, set[str]] = {}
    loop_ids: dict[tuple[str, tuple[tuple[str, int], ...]], str] = {}
    loops: dict[str, SurfaceLoopRecord] = {}
    canonical_surfaces: list[SurfacePlanRecord] = []
    surface_specs: list[
        tuple[
            SurfacePlanRecord,
            tuple[tuple[str, int, tuple[tuple[float, float, float], ...]], ...],
        ]
    ] = []

    def point_id(coordinate: tuple[float, float, float]) -> str:
        key = _coordinate_key(coordinate)
        existing = point_ids.get(key)
        if existing is not None:
            return existing
        new_id = f"P__{len(point_ids):06d}"
        point_ids[key] = new_id
        point_coordinates[new_id] = key
        point_curve_ids[new_id] = set()
        return new_id

    for surface in surfaces:
        if surface.construction_only:
            canonical_surfaces.append(surface)
            continue
        specs = _surface_ring3d_specs(surface)
        surface_specs.append((surface, specs))
        for _, _, ring in specs:
            for coordinate in ring:
                point_id(coordinate)

    point_axis_index = _point_axis_index(point_coordinates.values())
    point_line_index = _axis_aligned_point_index(point_coordinates.values())

    def split_edge_points(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> tuple[tuple[float, float, float], ...]:
        points_on_edge = [
            (parameter, coordinate)
            for coordinate in _segment_candidate_points(
                start,
                end,
                point_axis_index,
                point_line_index,
            )
            for parameter in (_segment_parameter(coordinate, start, end),)
            if parameter is not None
        ]
        return tuple(coordinate for _, coordinate in sorted(points_on_edge))

    def curve_ref(
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        surface: SurfacePlanRecord,
    ) -> CurveRefRecord:
        start_id = point_id(start)
        end_id = point_id(end)
        if start_id == end_id:
            raise ValueError(f"{surface.surface_id} has zero-length curve")
        key = tuple(sorted((start_id, end_id)))
        curve_id = curve_ids.get(key)
        if curve_id is None:
            curve_id = f"C__{len(curve_ids):06d}"
            curve_ids[key] = curve_id
        point_curve_ids[start_id].add(curve_id)
        point_curve_ids[end_id].add(curve_id)
        curve_owner_ids.setdefault(curve_id, set()).update(
            str(owner_id) for owner_id in _surface_owner_ids(surface)
        )
        if surface.interface_id is not None:
            curve_interface_ids.setdefault(curve_id, set()).add(surface.interface_id)
        curve_surface_ids.setdefault(curve_id, set()).add(surface.surface_id)
        curve_volume_ids.setdefault(curve_id, set()).update(
            str(volume_id)
            for volume_id in surface.metadata.get("boundary_volume_ids", ())
        )
        return CurveRefRecord(
            curve_id=curve_id,
            orientation=1 if key == (start_id, end_id) else -1,
            role="boundary",
        )

    for surface, specs in surface_specs:
        loop_refs: list[str] = []
        for role, index, ring in specs:
            raw_curve_refs = tuple(
                curve_ref(segment_start, segment_end, surface)
                for start, end in _ring3d_edges(ring)
                for split_points in (split_edge_points(start, end),)
                for segment_start, segment_end in pairwise(split_points)
            )
            curve_refs = _cancel_backtracking_curve_refs(raw_curve_refs)
            if not curve_refs:
                raise ValueError(f"{surface.surface_id} has empty canonical loop")
            loop_key = (role, _ordered_loop_signature(curve_refs))
            loop_id = loop_ids.get(loop_key)
            if loop_id is None:
                suffix = "OUTER" if role == "outer" else f"HOLE_{index:04d}"
                loop_id = f"LOOP__{surface.surface_id}__{suffix}"
                loop_ids[loop_key] = loop_id
                loops[loop_id] = SurfaceLoopRecord(
                    loop_id=loop_id,
                    curve_refs=curve_refs,
                    role=role,
                    surface_id=surface.surface_id,
                )
            loop_refs.append(loop_id)
        if not loop_refs:
            raise ValueError(f"{surface.surface_id} has no planned loops")
        canonical_surfaces.append(
            replace(
                surface,
                outer_loop_ref=loop_refs[0],
                hole_loop_refs=tuple(loop_refs[1:]),
            )
        )

    retained_loop_surface_ids: dict[str, set[str]] = {}
    for surface in canonical_surfaces:
        if surface.construction_only:
            continue
        for loop_id in (surface.outer_loop_ref, *surface.hole_loop_refs):
            if loop_id is not None:
                retained_loop_surface_ids.setdefault(loop_id, set()).add(
                    surface.surface_id
                )
    retained_curve_surface_ids: dict[str, set[str]] = {}
    for loop_id, loop in loops.items():
        for ref in loop.curve_refs:
            retained_curve_surface_ids.setdefault(ref.curve_id, set()).update(
                retained_loop_surface_ids.get(loop_id, ())
            )
    retained_curve_ids = set(retained_curve_surface_ids)
    retained_surfaces = {surface.surface_id: surface for surface in canonical_surfaces}
    curve_owner_ids = {
        curve_id: {
            owner_id
            for surface_id in surface_ids
            for owner_id in _surface_owner_ids(retained_surfaces[surface_id])
        }
        for curve_id, surface_ids in retained_curve_surface_ids.items()
    }
    curve_interface_ids = {
        curve_id: {
            retained_surfaces[surface_id].interface_id
            for surface_id in surface_ids
            if retained_surfaces[surface_id].interface_id is not None
        }
        for curve_id, surface_ids in retained_curve_surface_ids.items()
    }
    curve_volume_ids = {
        curve_id: {
            str(volume_id)
            for surface_id in surface_ids
            for volume_id in retained_surfaces[surface_id].metadata.get(
                "boundary_volume_ids", ()
            )
        }
        for curve_id, surface_ids in retained_curve_surface_ids.items()
    }
    # `curve_ref()` records provisional ownership before Boolean residual
    # cancellation.  Rebuild every public curve claim from the retained loops,
    # then make the invariant explicit so canceled cut lines cannot leak into
    # the canonical ledger.
    for curve_id, surface_ids in retained_curve_surface_ids.items():
        retained = tuple(retained_surfaces[surface_id] for surface_id in surface_ids)
        expected_owners = {
            owner_id for surface in retained for owner_id in _surface_owner_ids(surface)
        }
        expected_interfaces = {
            surface.interface_id
            for surface in retained
            if surface.interface_id is not None
        }
        expected_volumes = {
            str(volume_id)
            for surface in retained
            for volume_id in surface.metadata.get("boundary_volume_ids", ())
        }
        if (
            curve_owner_ids[curve_id] != expected_owners
            or curve_interface_ids[curve_id] != expected_interfaces
            or curve_volume_ids[curve_id] != expected_volumes
        ):
            raise AssertionError(
                f"{curve_id} retains claims absent from its retained surfaces"
            )
    point_curve_ids = {
        point_id_: {
            curve_id for curve_id in curve_ids_ if curve_id in retained_curve_ids
        }
        for point_id_, curve_ids_ in point_curve_ids.items()
    }
    points = tuple(
        PointPlanRecord(
            point_id=point_id_,
            coordinate=point_coordinates[point_id_],
            used_by_curve_ids=tuple(sorted(point_curve_ids[point_id_])),
        )
        for point_id_ in sorted(point_coordinates)
    )
    curves = tuple(
        CurvePlanRecord(
            curve_id=curve_id,
            curve_kind="line_segment",
            start_point_id=start_id,
            end_point_id=end_id,
            owner_semantic_ids=tuple(sorted(curve_owner_ids.get(curve_id, ()))),
            interface_ids=tuple(sorted(curve_interface_ids.get(curve_id, ()))),
            used_by_surface_ids=tuple(
                sorted(retained_curve_surface_ids.get(curve_id, ()))
            ),
            boundary_volume_ids=tuple(sorted(curve_volume_ids.get(curve_id, ()))),
        )
        for (start_id, end_id), curve_id in sorted(
            curve_ids.items(),
            key=lambda item: item[1],
        )
        if curve_id in retained_curve_ids
    )
    return points, curves, tuple(loops.values()), tuple(canonical_surfaces)


def _cancel_backtracking_curve_refs(
    refs: tuple[CurveRefRecord, ...],
) -> tuple[CurveRefRecord, ...]:
    """Remove immediate A→B→A artifacts from Boolean residual splitting."""
    result: list[CurveRefRecord] = []
    for ref in refs:
        if (
            result
            and result[-1].curve_id == ref.curve_id
            and result[-1].orientation == -ref.orientation
        ):
            result.pop()
        else:
            result.append(ref)
    if (
        len(result) > 1
        and result[0].curve_id == result[-1].curve_id
        and result[0].orientation == -result[-1].orientation
    ):
        result = result[1:-1]
    return tuple(result)


def _surface_ring3d_specs(
    surface: SurfacePlanRecord,
) -> tuple[tuple[str, int, tuple[tuple[float, float, float], ...]], ...]:
    geometry_ref = surface.geometry_ref
    if "quad_points" in geometry_ref:
        return (("outer", 0, _clean_ring3d(geometry_ref["quad_points"])),)
    if "outer_loop" not in geometry_ref:
        raise ValueError(f"{surface.surface_id} requires outer_loop or quad_points")
    z_um = _geometry_ref_surface_z_um(geometry_ref)
    outer_loop = _canonical_planar_loop_orientation(
        _clean_loop(geometry_ref["outer_loop"])
    )
    specs = [
        (
            "outer",
            0,
            tuple((x, y, z_um) for x, y in outer_loop),
        )
    ]
    specs.extend(
        (
            "hole",
            index,
            tuple(
                (x, y, z_um)
                for x, y in _canonical_planar_loop_orientation(_clean_loop(hole_loop))
            ),
        )
        for index, hole_loop in enumerate(geometry_ref.get("hole_loops", ()))
    )
    return tuple(specs)


def _canonical_planar_loop_orientation(
    loop: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Use one XY loop direction for OCC plane surfaces and their holes."""
    if _polygon_area(loop) < 0:
        return tuple(reversed(loop))
    return loop


def _clean_ring3d(ring: Any) -> tuple[tuple[float, float, float], ...]:
    points = tuple(
        (float(point[0]), float(point[1]), float(point[2])) for point in ring
    )
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("3D loop requires at least 3 unique points")
    return points


def _ring3d_edges(
    ring: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    return tuple(
        (ring[index], ring[(index + 1) % len(ring)]) for index in range(len(ring))
    )


def _segment_parameter(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> float | None:
    vector = tuple(end[index] - start[index] for index in range(3))
    offset = tuple(point[index] - start[index] for index in range(3))
    length_sq = sum(value * value for value in vector)
    if length_sq <= 1e-18:
        return None
    parameter = sum(offset[index] * vector[index] for index in range(3)) / length_sq
    if parameter < -1e-9 or parameter > 1.0 + 1e-9:
        return None
    closest = tuple(start[index] + parameter * vector[index] for index in range(3))
    distance_sq = sum((point[index] - closest[index]) ** 2 for index in range(3))
    if distance_sq > 1e-18:
        return None
    return max(0.0, min(1.0, parameter))


def _point_axis_index(
    coordinates: Sequence[tuple[float, float, float]],
) -> tuple[
    tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]],
    ...,
]:
    indexes: list[tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]]] = []
    for axis in range(3):
        items = sorted((coordinate[axis], coordinate) for coordinate in coordinates)
        indexes.append(
            (
                tuple(value for value, _ in items),
                tuple(coordinate for _, coordinate in items),
            )
        )
    return tuple(indexes)


def _axis_aligned_point_index(
    coordinates: Sequence[tuple[float, float, float]],
) -> dict[
    tuple[int, tuple[float, float]],
    tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]],
]:
    records: dict[
        tuple[int, tuple[float, float]],
        list[tuple[float, tuple[float, float, float]]],
    ] = {}
    for coordinate in coordinates:
        key = _coordinate_key(coordinate)
        for varying_axis in range(3):
            fixed_key = tuple(key[axis] for axis in range(3) if axis != varying_axis)
            records.setdefault((varying_axis, fixed_key), []).append(
                (key[varying_axis], key)
            )
    return {
        line_key: (
            tuple(value for value, _ in sorted_items),
            tuple(coordinate for _, coordinate in sorted_items),
        )
        for line_key, items in records.items()
        for sorted_items in (tuple(sorted(items)),)
    }


def _segment_candidate_points(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    point_axis_index: Sequence[
        tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]]
    ],
    point_line_index: Mapping[
        tuple[int, tuple[float, float]],
        tuple[tuple[float, ...], tuple[tuple[float, float, float], ...]],
    ],
) -> tuple[tuple[float, float, float], ...]:
    start_key = _coordinate_key(start)
    end_key = _coordinate_key(end)
    varying_axes = tuple(
        axis
        for axis in range(3)
        if abs(start_key[axis] - end_key[axis]) > _TOPOLOGY_EPS_UM
    )
    if len(varying_axes) == 1:
        varying_axis = varying_axes[0]
        fixed_key = tuple(start_key[axis] for axis in range(3) if axis != varying_axis)
        values, coordinates = point_line_index.get(
            (varying_axis, fixed_key),
            ((), ()),
        )
        lower = min(start_key[varying_axis], end_key[varying_axis]) - _TOPOLOGY_EPS_UM
        upper = max(start_key[varying_axis], end_key[varying_axis]) + _TOPOLOGY_EPS_UM
        left = bisect_left(values, lower)
        right = bisect_right(values, upper)
        candidates = set(coordinates[left:right])
        candidates.update((start_key, end_key))
        return tuple(sorted(candidates))
    bounds = tuple(
        (
            min(start_key[axis], end_key[axis]) - _TOPOLOGY_EPS_UM,
            max(start_key[axis], end_key[axis]) + _TOPOLOGY_EPS_UM,
        )
        for axis in range(3)
    )
    ranges: list[tuple[int, int, int]] = []
    for axis, (values, _) in enumerate(point_axis_index):
        lower, upper = bounds[axis]
        left = bisect_left(values, lower)
        right = bisect_right(values, upper)
        ranges.append((right - left, left, axis))
    _, left, axis = min(ranges)
    right = left + min(ranges)[0]
    coordinates = point_axis_index[axis][1][left:right]
    candidates = {
        coordinate
        for coordinate in coordinates
        if all(
            bounds[coordinate_axis][0]
            <= coordinate[coordinate_axis]
            <= bounds[coordinate_axis][1]
            for coordinate_axis in range(3)
        )
    }
    candidates.update((start_key, end_key))
    return tuple(sorted(candidates))


def _surface_owner_ids(surface: SurfacePlanRecord) -> tuple[str, ...]:
    return surface_declared_owner_ids(
        surface.metadata.get("owner_semantic_ids", (surface.owner_semantic_id,)),
        surface.owner_semantic_id,
    )
