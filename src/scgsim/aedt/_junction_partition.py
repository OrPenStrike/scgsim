"""AEDT-only junction partitioning before conductor/interface preparation.

The local DBU grid selects cuts, while the existing SGB Boolean precision
represents geometry. Neither is a native contact tolerance. The constant
candidate argument applies only after closed-corridor clearance is verified.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from decimal import ROUND_FLOOR, Decimal
from fractions import Fraction
from itertools import pairwise
from typing import Any

import gdstk

from scgsim.sgb import GeometryBuildInput, LayoutPolygonSpec
from scgsim.sgb.planning import (
    _cancel_reversed_planar_edges,
    _simple_planar_loops_from_edges,
)

from ._epr_models import PlanarJunction, canonical_sha256

METHOD = "scgsim.aedt.junction-partition.v1"
PRECISION_UM = 1e-9


def _plain(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _edges(rings):
    for ring in rings:
        yield from zip(ring, (*ring[1:], ring[0]))


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on(point, a, b):
    return _cross(a, b, point) == 0 and all(
        min(a[i], b[i]) <= point[i] <= max(a[i], b[i]) for i in (0, 1)
    )


def _inside_ring(point, ring):
    inside = False
    for a, b in _edges((ring,)):
        if _on(point, a, b):
            return 0  # Boundary, distinct from inside and outside.
        if (a[1] > point[1]) != (b[1] > point[1]):
            x = Fraction(b[0] - a[0], b[1] - a[1]) * (point[1] - a[1]) + a[0]
            if point[0] < x:
                inside = not inside
    return 1 if inside else -1


def _inside(point, rings):
    return _inside_ring(point, rings[0]) >= 0 and not any(
        _inside_ring(point, hole) == 1 for hole in rings[1:]
    )


def _section(rings, axis, coordinate, low, high):
    """Exact closed contact on a specified edge in Boolean integer coordinates."""
    other = 1 - axis
    marks = {Fraction(low), Fraction(high)}
    for a, b in _edges(rings):
        if a[axis] == b[axis] == coordinate:
            marks.update(Fraction(v[other]) for v in (a, b) if low <= v[other] <= high)
        elif a[axis] != b[axis] and min(a[axis], b[axis]) <= coordinate <= max(
            a[axis], b[axis]
        ):
            v = (
                Fraction(coordinate - a[axis], b[axis] - a[axis])
                * (b[other] - a[other])
                + a[other]
            )
            if low <= v <= high:
                marks.add(v)
    ordered = sorted(marks)
    point = lambda v: (coordinate, v) if axis == 0 else (v, coordinate)
    intervals = [
        (a, b) for a, b in pairwise(ordered) if _inside(point((a + b) / 2), rings)
    ]
    points = [v for v in ordered if _inside(point(v), rings)]
    return intervals, points


def _covered(polygons, axis, coordinate, low, high):
    intervals = sorted(
        interval
        for rings in polygons
        for interval in _section(rings, axis, coordinate, low, high)[0]
    )
    end = Fraction(low)
    for a, b in intervals:
        if a > end:
            return False
        end = max(end, b)
    return end >= high


def _line_intervals(polygons):
    result = defaultdict(list)
    for rings in polygons:
        for a, b in _edges(rings):
            dx, dy = b[0] - a[0], b[1] - a[1]
            divisor = math.gcd(dx, dy)
            if not divisor:
                continue
            dx, dy = dx // divisor, dy // divisor
            if dx < 0 or (dx == 0 and dy < 0):
                dx, dy = -dx, -dy
            key = (dx, dy, dx * a[1] - dy * a[0])
            u, v = a[0] * dx + a[1] * dy, b[0] * dx + b[1] * dy
            result[key].append((min(u, v), max(u, v)))
    return result


def _positive_boundary_contact(left, right):
    a, b = _line_intervals(left), _line_intervals(right)
    return any(
        max(x0, y0) < min(x1, y1)
        for key in a.keys() & b.keys()
        for x0, x1 in a[key]
        for y0, y1 in b[key]
    )


def _closed_contact(left, right):
    for a in left:
        for b in right:
            ax = [p[0] for p in a[0]]
            ay = [p[1] for p in a[0]]
            bx = [p[0] for p in b[0]]
            by = [p[1] for p in b[0]]
            if (
                max(ax) < min(bx)
                or max(bx) < min(ax)
                or max(ay) < min(by)
                or max(by) < min(ay)
            ):
                continue
            if {p for ring in a for p in ring} & {p for ring in b for p in ring}:
                return True
            if any(_inside(p, b) for p in a[0]) or any(_inside(p, a) for p in b[0]):
                return True
            for p, q in _edges(a):
                for r, s in _edges(b):
                    c = (
                        _cross(p, q, r),
                        _cross(p, q, s),
                        _cross(r, s, p),
                        _cross(r, s, q),
                    )
                    if any((_on(r, p, q), _on(s, p, q), _on(p, r, s), _on(q, r, s))):
                        return True
                    if c[0] * c[1] < 0 and c[2] * c[3] < 0:
                        return True
    return False


def _region(polygons):
    result = []
    for rings in polygons:
        exterior = gdstk.Polygon(
            [(x * PRECISION_UM, y * PRECISION_UM) for x, y in rings[0]]
        )
        holes = [
            gdstk.Polygon([(x * PRECISION_UM, y * PRECISION_UM) for x, y in ring])
            for ring in rings[1:]
        ]
        result.extend(_boolean([exterior], holes, "not"))
    return result


def _boolean(left, right, operation):
    # SGB's Boolean wrapper filters small areas. Partitioning must retain every
    # residual, so reuse its precision and loop lowering without that filter.
    return list(gdstk.boolean(left, right, operation, precision=PRECISION_UM) or ())


def _rings(region):
    result = []
    for polygon in region:
        points = tuple(
            (round(x / PRECISION_UM), round(y / PRECISION_UM))
            for x, y in polygon.points
        )
        # Atomize at every retained vertex, including vertices inside another
        # edge. This preserves SGB minimum-coherence/point-fragment semantics.
        ordered = sorted(set(points))
        xs = [point[0] for point in ordered]
        edges = []
        for a, b in _edges((points,)):
            candidates = ordered[
                bisect_left(xs, min(a[0], b[0])) : bisect_right(xs, max(a[0], b[0]))
            ]
            on_edge = sorted(
                (p for p in candidates if _on(p, a, b)),
                key=lambda p: (
                    (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])
                ),
            )
            edges.extend(pairwise(on_edge))
        loops = _simple_planar_loops_from_edges(_cancel_reversed_planar_edges(edges))
        area = lambda ring: abs(
            sum(a[0] * b[1] - b[0] * a[1] for a, b in _edges((ring,)))
        )
        outer = max(loops, key=area)
        holes = tuple(ring for ring in loops if ring is not outer)
        if any(_inside_ring(hole[0], outer) != 1 for hole in holes):
            raise ValueError(
                "junction residual has point-connected or non-interior components"
            )
        result.append((outer, *holes))
    return result


def _rectangle(x0, y0, x1, y1):
    return ((((x0, y0), (x1, y0), (x1, y1), (x0, y1)),),)


def _contacts(polygons, span, width):
    """Find extrema of area, segment and point contact with closed authored S."""
    region = _boolean(_region(polygons), _region(_rectangle(0, 0, span, width)), "and")
    positions = [round(x / PRECISION_UM) for p in region for x, _ in p.points]
    for rings in polygons:
        for axis, c, low, high in (
            (0, 0, 0, width),
            (0, span, 0, width),
            (1, 0, 0, span),
            (1, width, 0, span),
        ):
            intervals, points = _section(rings, axis, c, low, high)
            if axis == 0 and (intervals or points):
                positions.append(c)
            elif axis == 1:
                positions.extend(points)
                positions.extend(v for pair in intervals for v in pair)
    return (min(positions), max(positions)) if positions else None


def _bound_decimal(bound):
    rational = Fraction(bound)
    return (
        Decimal(rational.numerator)
        / Decimal(rational.denominator)
        * Decimal(str(PRECISION_UM))
    )


def _candidate_indices(bound, step, *, left):
    q = _bound_decimal(bound) / Decimal(str(step))
    floor = int(q.to_integral_value(rounding=ROUND_FLOOR))
    aligned = q == floor
    if left:
        return ([floor] if aligned else []) + [floor + 1]
    return ([floor] if aligned else []) + [floor if not aligned else floor - 1]


def _cut(index, step):
    requested = float(Decimal(index) * Decimal(str(step)))
    return requested, round(requested / PRECISION_UM)


def _end_candidate(free, own, foreign, cut, span, width, *, left):
    clip = _rectangle(0, 0, cut, width) if left else _rectangle(cut, 0, span, width)
    region = (
        _boolean(free, _region(clip), "and")
        if (cut > 0 if left else cut < span)
        else []
    )
    try:
        end = _rings(region)
    except ValueError as exc:
        return None, {"valid": False, "reason": str(exc)}
    if len(end) > 1:
        return None, {
            "valid": False,
            "reason": "nonempty end has disconnected components",
            "components": len(end),
        }
    if end and not _positive_boundary_contact(end, own):
        return None, {
            "valid": False,
            "reason": "end lacks positive-length own-arm contact",
        }
    if _closed_contact(end, foreign):
        return None, {
            "valid": False,
            "reason": "end contacts wrong terminal or third net",
        }
    if not _covered([*own, *end], 0, cut, 0, width):
        return None, {
            "valid": False,
            "reason": "designated end edge is not wholly covered",
        }
    return end, {"valid": True, "empty_residual": not end, "components": len(end)}


def partition_junctions(
    build_input: GeometryBuildInput, junctions: Sequence[PlanarJunction], step: float
):
    """Detach AEDT ends under their source Entities, retaining authored polygons."""
    if not junctions:
        return build_input
    if "aedt_junction_partitions" in build_input.metadata:
        raise ValueError(
            "junction input is already partitioned; use immutable authored input"
        )
    originals = {p.polygon_id: p for p in build_input.polygons}
    originals.update(
        {
            r.source_polygon_id: LayoutPolygonSpec(
                r.source_polygon_id, r.source_layer, r.exterior, r.holes
            )
            for r in build_input.port_sheet_regions
        }
    )
    authored_ids = {j.source_polygon_id for j in junctions}
    entities = {e.semantic_id: e for e in build_input.entities}
    additions = []
    records = {}
    appended = defaultdict(list)
    support_clips = defaultdict(list)
    for junction in junctions:
        sheet = originals[junction.source_polygon_id]
        dx, dy = junction.direction_xy
        px, py = -dy, dx
        along = [x * dx + y * dy for x, y in sheet.exterior]
        across = [x * px + y * py for x, y in sheet.exterior]
        origin = (min(along), min(across))
        span = round((max(along) - origin[0]) / PRECISION_UM)
        width = round((max(across) - origin[1]) / PRECISION_UM)
        if span <= 0 or width <= 0:
            raise ValueError(
                "junction authored geometry collapses at Boolean precision"
            )
        deviations = []

        def local(
            polygon, *, dx=dx, dy=dy, px=px, py=py, origin=origin, deviations=deviations
        ):
            rings = []
            for ring in (polygon.exterior, *polygon.holes):
                converted = []
                for x, y in ring:
                    u, v = x * dx + y * dy - origin[0], x * px + y * py - origin[1]
                    point = (round(u / PRECISION_UM), round(v / PRECISION_UM))
                    deviations.append(
                        max(
                            abs(point[0] * PRECISION_UM - u),
                            abs(point[1] * PRECISION_UM - v),
                        )
                    )
                    converted.append(point)
                rings.append(tuple(converted))
            for index, ring in enumerate(rings):
                area = sum(a[0] * b[1] - b[0] * a[1] for a, b in _edges((ring,)))
                if len(set(ring)) < 3 or area == 0:
                    raise ValueError(
                        "numerical junction geometry failure: source polygon/hole collapses at Boolean precision"
                    )
                if index and any(_inside_ring(point, rings[0]) != 1 for point in ring):
                    raise ValueError(
                        "numerical junction geometry failure: source hole loses interior topology"
                    )
            return tuple(rings)

        by_entity = {
            e.semantic_id: [
                local(originals[p]) for p in e.polygon_ids if p not in authored_ids
            ]
            for e in entities.values()
            if e.material_kind == "conductor"
        }
        by_net = defaultdict(list)
        for eid, polygons in by_entity.items():
            by_net[entities[eid].net_id].extend(polygons)
        own_a, own_b = by_net[junction.terminal_a_net], by_net[junction.terminal_b_net]
        if not own_a or not own_b:
            raise ValueError(
                "junction requires both explicit terminal conductor geometries"
            )
        other = [
            p
            for net, polygons in by_net.items()
            if net not in {junction.terminal_a_net, junction.terminal_b_net}
            for p in polygons
        ]
        if _contacts(other, span, width) is not None:
            raise ValueError("third-net closed contact with authored junction sheet")
        ca, cb = _contacts(own_a, span, width), _contacts(own_b, span, width)
        if ca is None or cb is None or ca[1] >= cb[0]:
            raise ValueError("unsupported junction contact ordering or clear corridor")
        lower, upper = ca[1], cb[0]
        for polygons in (own_a, own_b, other):
            for rings in polygons:
                for y in (0, width):
                    intervals, points = _section(rings, 1, y, lower, upper)
                    if any(a < b for a, b in intervals) or any(
                        lower < v < upper for v in points
                    ):
                        raise ValueError(
                            "unsupported corridor side-boundary metal contact"
                        )
        sheet_region = _region(_rectangle(0, 0, span, width))
        free = _boolean(sheet_region, _region([*own_a, *own_b]), "not")
        corridor = _region(_rectangle(lower, 0, upper, width))
        if _boolean(corridor, free, "not"):
            raise ValueError(
                "unsupported corridor containment; no general search performed"
            )
        choices = []
        checks = []
        for left, bound, own, foreign in (
            (True, lower, own_a, own_b + other),
            (False, upper, own_b, own_a + other),
        ):
            chosen = None
            for index in _candidate_indices(bound, step, left=left):
                requested, cut = _cut(index, step)
                interior = (
                    (Decimal(index) * Decimal(str(step)) > _bound_decimal(bound))
                    if left
                    else (Decimal(index) * Decimal(str(step)) < _bound_decimal(bound))
                )
                if interior and cut == bound:
                    raise ValueError(
                        "numerical junction geometry failure: interior grid cut collapses to boundary"
                    )
                if (
                    cut < lower
                    or cut > upper
                    or (interior and (cut <= lower or cut >= upper))
                ):
                    continue
                end, evidence = _end_candidate(
                    free, own, foreign, cut, span, width, left=left
                )
                checks.append(
                    {
                        "side": "A" if left else "B",
                        "index": index,
                        "requested_s_um": requested,
                        "actual_s_um": cut * PRECISION_UM,
                        "deviation_um": cut * PRECISION_UM - requested,
                        **evidence,
                    }
                )
                if end is not None:
                    chosen = (cut, end, index)
                    break
            if chosen is None:
                raise ValueError(
                    "unsupported junction end topology on verified corridor/grid"
                )
            choices.append(chosen)
        (a, end_a, ka), (b, end_b, kb) = choices
        if a >= b:
            raise ValueError("junction DBU grid has no positive central span")
        central = _rectangle(a, 0, b, width)
        for y in (0, width):
            for rings in [*own_a, *own_b, *other, *end_a, *end_b]:
                intervals, points = _section(rings, 1, y, a, b)
                if any(x < z for x, z in intervals) or any(a < v < b for v in points):
                    raise ValueError(
                        "RLC open side has forbidden line or point contact"
                    )
        for cut, wrong in ((a, own_b + other + end_b), (b, own_a + other + end_a)):
            if any(any(_section(rings, 0, cut, 0, width)) for rings in wrong):
                raise ValueError("wrong-net RLC end/corner contact")
        parts = [_region(end_a), _region(central), _region(end_b)]
        union = _boolean([p for part in parts for p in part], [], "or")
        if _boolean(union, free, "xor") or any(
            _boolean(parts[i], parts[j], "and") for i, j in ((0, 1), (0, 2), (1, 2))
        ):
            raise ValueError(
                "numerical junction geometry failure: partition coverage/interiors differ"
            )

        def world(rings, *, dx=dx, dy=dy, px=px, py=py, origin=origin):
            return [
                [
                    [
                        (origin[0] + u * PRECISION_UM) * dx
                        + (origin[1] + v * PRECISION_UM) * px,
                        (origin[0] + u * PRECISION_UM) * dy
                        + (origin[1] + v * PRECISION_UM) * py,
                    ]
                    for u, v in ring
                ]
                for ring in rings
            ]

        end_records = []
        for label, end, own, cut, net in (
            ("A", end_a, own_a, a, junction.terminal_a_net),
            ("B", end_b, own_b, b, junction.terminal_b_net),
        ):
            owners = [
                eid
                for eid, polygons in by_entity.items()
                if entities[eid].net_id == net
                and (
                    _positive_boundary_contact(end, polygons)
                    if end
                    else _covered(polygons, 0, cut, 0, width)
                )
            ]
            if len(owners) != 1:
                raise ValueError(
                    "junction end lacks an unambiguous source Entity/material/Z lineage"
                )
            owner = entities[owners[0]]
            polygon_id = None
            if end:
                polygon_id = f"aedt_junction_{canonical_sha256([junction.junction_id, label, owner.semantic_id])[:24]}"
                loops = world(end[0])
                additions.append(
                    LayoutPolygonSpec(
                        polygon_id,
                        sheet.layer,
                        tuple(map(tuple, loops[0])),
                        tuple(tuple(map(tuple, ring)) for ring in loops[1:]),
                        net_name=net,
                        metadata={
                            "aedt_junction_id": junction.junction_id,
                            "source_entity_id": owner.semantic_id,
                        },
                    )
                )
                appended[owner.semantic_id].append(polygon_id)
                # A union (F intersect left) equals A union (S intersect left)
                # under verified terminal ordering. Build owner support in one
                # union, avoiding subtract/reunion intersection roundoff.
                clip = (
                    _rectangle(0, 0, cut, width)
                    if label == "A"
                    else _rectangle(cut, 0, span, width)
                )
                support_clips[owner.semantic_id].append(world(clip[0]))
            end_records.append(
                {
                    "side": label,
                    "source_entity_id": owner.semantic_id,
                    "net_id": net,
                    "material_id": owner.material_id,
                    "geometry": _plain(owner.geometry),
                    "representation": dict(owner.route_representations),
                    "polygon_id": polygon_id,
                    "empty_residual": not end,
                    "local_regions": _plain(end),
                }
            )
        cid = (
            f"aedt_junction_{canonical_sha256([junction.junction_id, 'central'])[:24]}"
        )
        loops = world(central[0])
        additions.append(
            LayoutPolygonSpec(
                cid,
                sheet.layer,
                tuple(map(tuple, loops[0])),
                metadata={
                    "aedt_junction_id": junction.junction_id,
                    "part": "central_rlc",
                },
            )
        )
        records[junction.junction_id] = {
            "method": METHOD,
            "authored_polygon_id": junction.source_polygon_id,
            "central_polygon_id": cid,
            "direction_xy": [dx, dy],
            "local_origin_projection_um": list(origin),
            "source_dbu_um": step,
            "boolean_precision_um": PRECISION_UM,
            "span_um": span * PRECISION_UM,
            "width_um": width * PRECISION_UM,
            "corridor_um": [lower * PRECISION_UM, upper * PRECISION_UM],
            "cut_indices": [ka, kb],
            "actual_cuts_um": [a * PRECISION_UM, b * PRECISION_UM],
            "candidate_checks": checks,
            "candidate_count": len(checks),
            "proof_conditions": {
                "full_width_corridor_containment": True,
                "closed_corridor_side_clearance": True,
                "terminal_contact_ordering": "A_left_B_right",
                "partition_coverage_and_disjoint_interiors": True,
            },
            "maximum_input_coordinate_deviation_um": max(deviations, default=0.0),
            "ends": end_records,
            "free_local_regions": _plain(_rings(free)),
            "central_local_regions": _plain(central),
        }
    metadata = dict(build_input.metadata)
    metadata["aedt_authored_input_sha256"] = canonical_sha256(_plain(build_input))
    metadata["aedt_junction_partitions"] = records
    polygons_by_id = {p.polygon_id: p for p in (*build_input.polygons, *additions)}
    # A junction's proof considered original conductors. Other junctions add
    # metal afterwards, so revalidate every final RLC edge/interior and every
    # end against the complete detached conductor set before any catalogue.
    all_conductors = []
    for entity in build_input.entities:
        for pid in (*entity.polygon_ids, *appended[entity.semantic_id]):
            if entity.material_kind == "conductor" and pid not in authored_ids:
                all_conductors.append((entity.net_id, pid, polygons_by_id[pid]))
    for record in records.values():
        dx, dy = record["direction_xy"]
        px, py = -dy, dx
        origin = record["local_origin_projection_um"]

        def final_local(polygon, *, dx=dx, dy=dy, px=px, py=py, origin=origin):
            return tuple(
                tuple(
                    (
                        round((x * dx + y * dy - origin[0]) / PRECISION_UM),
                        round((x * px + y * py - origin[1]) / PRECISION_UM),
                    )
                    for x, y in ring
                )
                for ring in (polygon.exterior, *polygon.holes)
            )

        metal = [(net, pid, final_local(p)) for net, pid, p in all_conductors]
        a, b = [round(value / PRECISION_UM) for value in record["actual_cuts_um"]]
        width = round(record["width_um"] / PRECISION_UM)
        central = _region(_rectangle(a, 0, b, width))
        if _boolean(central, _region([rings for _, _, rings in metal]), "and"):
            raise ValueError("final cross-junction PEC intersects RLC interior")
        nets = [end["net_id"] for end in record["ends"]]
        for cut, own_net in ((a, nets[0]), (b, nets[1])):
            if not _covered(
                [rings for net, _, rings in metal if net == own_net], 0, cut, 0, width
            ):
                raise ValueError(
                    "final junction designated end edge is not wholly covered"
                )
            if any(
                any(_section(rings, 0, cut, 0, width))
                for net, _, rings in metal
                if net != own_net
            ):
                raise ValueError("final cross-junction wrong-net end/corner contact")
        for y in (0, width):
            for _, _, rings in metal:
                intervals, points = _section(rings, 1, y, a, b)
                if intervals or any(a < point < b for point in points):
                    raise ValueError("final cross-junction open-side contact")
        for end in record["ends"]:
            if end["polygon_id"] is not None:
                own = final_local(polygons_by_id[end["polygon_id"]])
                if _closed_contact(
                    [own], [rings for net, _, rings in metal if net != end["net_id"]]
                ):
                    raise ValueError("final cross-junction end contacts another net")
        record["proof_conditions"]["all_derived_junctions_checked"] = True
    derived_entities = []
    for entity in build_input.entities:
        extra = appended[entity.semantic_id]
        owner_ids = tuple(pid for pid in entity.polygon_ids if pid not in authored_ids)
        if not extra and owner_ids == entity.polygon_ids:
            derived_entities.append(entity)
            continue
        # Authored junction sheets remain in the immutable polygon backing, but
        # are never conductor support, including when explicitly listed by an owner.
        ids = (*owner_ids, *extra)
        if not ids:
            continue
        rings = [
            tuple(
                tuple(
                    (round(x / PRECISION_UM), round(y / PRECISION_UM)) for x, y in ring
                )
                for ring in (polygons_by_id[pid].exterior, *polygons_by_id[pid].holes)
            )
            for pid in owner_ids
        ]
        rings.extend(
            tuple(
                tuple(
                    (round(x / PRECISION_UM), round(y / PRECISION_UM)) for x, y in ring
                )
                for ring in clip
            )
            for clip in support_clips[entity.semantic_id]
        )
        merged = _rings(_boolean(_region(rings), [], "or"))
        if len(merged) != 1:
            raise ValueError(
                "junction source Entity support cannot be represented as one connected normalized region"
            )
        geometry = dict(entity.geometry)
        geometry["outer_loop"] = tuple(
            (x * PRECISION_UM, y * PRECISION_UM) for x, y in merged[0][0]
        )
        geometry["hole_loops"] = tuple(
            tuple((x * PRECISION_UM, y * PRECISION_UM) for x, y in ring)
            for ring in merged[0][1:]
        )
        derived_entities.append(replace(entity, polygon_ids=ids, geometry=geometry))
    return replace(
        build_input,
        polygons=(*build_input.polygons, *additions),
        entities=tuple(derived_entities),
        metadata=metadata,
    )
