"""HFSS field-expression authoring and raw integral evaluation for EPR.

This module owns one bound field context at a time.  It does not select modes,
iterate adaptive passes, own Desktop, or derive participation ratios.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parse_native_scalar(value: Any) -> dict[str, Any]:
    """Parse one native scalar without treating false/missing as numeric zero."""

    if value is False or value is None or isinstance(value, bool):
        raise RuntimeError("HFSS field calculator returned no scalar value")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise RuntimeError("HFSS field calculator returned a nonfinite scalar")
        return {"value": number, "unit": "", "raw": value}
    text = str(value).strip()
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*([^\s]*)",
        text,
    )
    if match is None:
        raise RuntimeError(f"HFSS field calculator scalar is invalid: {value!r}")
    number = float(match.group(1))
    if not math.isfinite(number):
        raise RuntimeError("HFSS field calculator returned a nonfinite scalar")
    return {"value": number, "unit": match.group(2), "raw": text}


def _constant(value: float) -> list[str]:
    return [f"Scalar_Constant({value:.17g})"]


def _function(name: str) -> list[str]:
    return [f"Scalar_Function(FuncValue='{name}')"]


def _binary(left: Sequence[str], right: Sequence[str], operation: str) -> list[str]:
    return [*left, *right, f"Operation('{operation}')"]


def _square(value: Sequence[str]) -> list[str]:
    return [*value, "Scalar_Constant(2)", "Operation('Pow')"]


def _absolute(value: Sequence[str]) -> list[str]:
    return [*value, "Operation('Abs')"]


def _clamp_unit_interval(value: Sequence[str]) -> list[str]:
    # max(0, t) - max(0, t - 1), with max(0, x) = (x + |x|) / 2.
    above_zero = _binary(_binary(value, _absolute(value), "+"), _constant(2.0), "/")
    shifted = _binary(value, _constant(1.0), "-")
    above_one = _binary(
        _binary(shifted, _absolute(shifted), "+"), _constant(2.0), "/"
    )
    return _binary(above_zero, above_one, "-")


def _hard_step(value: Sequence[str]) -> list[str]:
    # H(s >= 0) = 1 - atan2(+0, s) / atan2(+0, -1). Native signed-zero
    # behavior is part of the operation evidence and is never epsilon-shifted.
    canonical = _binary(value, _constant(0.0), "+")
    numerator = [*_constant(0.0), *canonical, "Operation('BMathFunc', 'Atan2')"]
    denominator = [*_constant(0.0), *_constant(-1.0), "Operation('BMathFunc', 'Atan2')"]
    return _binary(_constant(1.0), _binary(numerator, denominator, "/"), "-")


def _patch_token(patch_id: str) -> str:
    return hashlib.sha256(patch_id.encode("utf-8")).hexdigest()[:16]


def _coordinate_name(patch_id: str, ring: int, vertex: int, axis: str) -> str:
    return f"scgsim_mask_{_patch_token(patch_id)}_r{ring}_v{vertex}_{axis}_um"


def _margin_name(patch_id: str, margin_um: float) -> str:
    identity = f"{patch_id}\0{float(margin_um):.17g}"
    token = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"scgsim_mask_margin_{token}_um"


def _origin_name(patch_id: str, axis: str) -> str:
    return f"scgsim_mask_{_patch_token(patch_id)}_origin_{axis}_um"


def _plane_basis(
    origin_um: Sequence[float], u: Sequence[float], v: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if len(origin_um) != 3 or len(u) != 3 or len(v) != 3:
        raise ValueError("mask plane origin/u/v must each contain three values")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (*origin_um, *u, *v)
    ):
        raise TypeError("mask plane origin/u/v values must be numeric and non-boolean")
    origin = tuple(float(value) for value in origin_um)
    u_vector = tuple(float(value) for value in u)
    v_vector = tuple(float(value) for value in v)
    if not all(math.isfinite(value) for value in (*origin, *u_vector, *v_vector)):
        raise ValueError("mask plane origin/u/v must be finite")
    if not math.isclose(sum(value * value for value in u_vector), 1.0, abs_tol=1e-12):
        raise ValueError("mask plane u must be a unit vector")
    if not math.isclose(sum(value * value for value in v_vector), 1.0, abs_tol=1e-12):
        raise ValueError("mask plane v must be a unit vector")
    if not math.isclose(sum(a * b for a, b in zip(u_vector, v_vector)), 0.0, abs_tol=1e-12):
        raise ValueError("mask plane u and v must be perpendicular")
    return origin, u_vector, v_vector


def _plane_coordinate(
    patch_id: str, basis: Sequence[float]
) -> list[str]:
    result = _constant(0.0)
    for axis, weight in zip(("x", "y", "z"), basis):
        if weight == 0.0:
            continue
        shifted = _binary(
            _function(axis.upper()), _function(_origin_name(patch_id, axis)), "-"
        )
        result = _binary(result, _binary(shifted, _constant(weight), "*"), "+")
    return result


def _ring_points(
    ring: Sequence[Sequence[float]], *, label: str
) -> tuple[tuple[float, float], ...]:
    if isinstance(ring, (str, bytes)):
        raise TypeError(f"mask ring {label} must be a coordinate sequence")
    points: list[tuple[float, float]] = []
    for index, point in enumerate(ring):
        if isinstance(point, (str, bytes)) or len(point) != 2:
            raise ValueError(
                f"mask ring {label} vertex {index} must contain exactly two coordinates"
            )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point):
            raise TypeError(
                f"mask ring {label} vertex {index} coordinates must be numeric"
            )
        pair = (float(point[0]), float(point[1]))
        if not all(math.isfinite(value) for value in pair):
            raise ValueError(
                f"mask ring {label} vertex {index} coordinates must be finite"
            )
        points.append(pair)
    return tuple(points)


def mask_variables(
    patch_id: str,
    exterior: Sequence[Sequence[float]],
    holes: Sequence[Sequence[Sequence[float]]],
    margins_um: Sequence[float],
    *,
    plane_origin_um: Sequence[float],
    plane_u: Sequence[float],
    plane_v: Sequence[float],
) -> dict[str, str]:
    origin, _, _ = _plane_basis(plane_origin_um, plane_u, plane_v)
    values = {"scgsim_mask_reference_um": "1um"}
    for axis, coordinate in zip(("x", "y", "z"), origin):
        values[_origin_name(patch_id, axis)] = f"{coordinate:.17g}um"
    for margin in margins_um:
        values[_margin_name(patch_id, margin)] = f"{float(margin):.17g}um"
    if isinstance(holes, (str, bytes)):
        raise TypeError("mask holes must be a sequence of rings")
    rings = tuple(
        _ring_points(ring, label=f"{patch_id}/{ring_index}")
        for ring_index, ring in enumerate((exterior, *holes))
    )
    for ring_index, ring in enumerate(rings):
        for vertex_index, (x, y) in enumerate(ring):
            values[_coordinate_name(patch_id, ring_index, vertex_index, "x")] = f"{float(x):.17g}um"
            values[_coordinate_name(patch_id, ring_index, vertex_index, "y")] = f"{float(y):.17g}um"
    return values


def _validated_postprocessing_readback(
    values: Mapping[str, str], postprocessing: Mapping[str, Any]
) -> dict[str, str]:
    """Preserve exact expressions; accept rewritten finite literals only at exact parity."""

    from ansys.aedt.core.generic.numbers_utils import decompose_variable_value

    observed: dict[str, str] = {}
    for name, requested in values.items():
        if name not in postprocessing:
            raise RuntimeError(
                f"HFSS postprocessing variable {name!r} is missing from readback"
            )
        actual = str(postprocessing[name])
        requested_number, requested_unit = decompose_variable_value(requested)
        if requested == actual:
            if (
                isinstance(requested_number, (int, float))
                and not math.isfinite(requested_number)
            ) or re.fullmatch(
                r"[+-]?(?:nan|inf(?:inity)?)[A-Za-z]*", requested.strip(), re.I
            ):
                raise RuntimeError(
                    f"HFSS postprocessing variable {name!r} is nonfinite"
                )
            observed[name] = actual
            continue
        actual_number, actual_unit = decompose_variable_value(actual)
        if not (
            isinstance(requested_number, (int, float))
            and not isinstance(requested_number, bool)
            and isinstance(actual_number, (int, float))
            and not isinstance(actual_number, bool)
            and math.isfinite(requested_number)
            and math.isfinite(actual_number)
            and requested_unit == actual_unit
            and requested_number == actual_number
        ):
            raise RuntimeError(
                f"HFSS postprocessing variable {name!r} differs from requested value"
            )
        observed[name] = actual
    return observed


def install_variables(app: Any, values: Mapping[str, str]) -> dict[str, str]:
    manager = app.variable_manager
    for name, value in values.items():
        if not manager.set_variable(
            name,
            expression=value,
            sweep=False,
            overwrite=True,
            is_post_processing=True,
        ):
            raise RuntimeError(f"HFSS postprocessing variable failed: {name!r}")
    return _validated_postprocessing_readback(values, manager.post_processing_variables)


def _signed_area(ring: Sequence[Sequence[float]]) -> float:
    return sum(
        float(ring[index][0]) * float(ring[(index + 1) % len(ring)][1])
        - float(ring[(index + 1) % len(ring)][0]) * float(ring[index][1])
        for index in range(len(ring))
    ) / 2.0


def _cross(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (
        float(b[1]) - float(a[1])
    ) * (float(c[0]) - float(a[0]))


def _point_in_triangle(
    point: Sequence[float], triangle: Sequence[Sequence[float]], orientation: float
) -> bool:
    return all(
        orientation * _cross(triangle[index], triangle[(index + 1) % 3], point)
        >= 0.0
        for index in range(3)
    )


def triangulate_ring(
    ring: Sequence[Sequence[float]], *, label: str
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Deterministically ear-clip one simple concave or convex ring."""

    points = _ring_points(ring, label=label)
    if len(points) < 3 or len(set(points)) != len(points):
        raise ValueError(f"mask ring {label} must have distinct vertices")
    area = _signed_area(points)
    if area == 0.0:
        raise ValueError(f"mask ring {label} has zero area")
    orientation = 1.0 if area > 0.0 else -1.0
    indices = list(range(len(points)))
    triangles: list[tuple[tuple[float, float], ...]] = []
    while len(indices) > 3:
        ear: tuple[int, tuple[tuple[float, float], ...]] | None = None
        for position, current in enumerate(indices):
            previous = indices[position - 1]
            following = indices[(position + 1) % len(indices)]
            triangle = (points[previous], points[current], points[following])
            if orientation * _cross(*triangle) <= 0.0:
                continue
            if any(
                _point_in_triangle(points[index], triangle, orientation)
                for index in indices
                if index not in {previous, current, following}
            ):
                continue
            ear = position, triangle
            break
        if ear is None:
            raise ValueError(f"mask ring {label} is self-intersecting or degenerate")
        position, triangle = ear
        triangles.append(triangle)
        indices.pop(position)
    final = tuple(points[index] for index in indices)
    if orientation * _cross(*final) <= 0.0:
        raise ValueError(f"mask ring {label} has a degenerate final triangle")
    triangles.append(final)
    return tuple(triangles)


def _triangle_characteristic(
    patch_id: str,
    ring_index: int,
    triangle_index: int,
    ring: Sequence[Sequence[float]],
    triangle: Sequence[Sequence[float]],
    x: Sequence[str],
    y: Sequence[str],
) -> list[str]:
    orientation = 1.0 if _signed_area(triangle) > 0.0 else -1.0
    result = _constant(1.0)
    reference_squared = _square(_function("scgsim_mask_reference_um"))
    for index in range(3):
        following = (index + 1) % 3
        source_a = (float(triangle[index][0]), float(triangle[index][1]))
        source_b = (float(triangle[following][0]), float(triangle[following][1]))
        normalized_ring = [
            (float(point[0]), float(point[1])) for point in ring
        ]
        a_index = normalized_ring.index(source_a)
        b_index = normalized_ring.index(source_b)
        ax = _function(_coordinate_name(patch_id, ring_index, a_index, "x"))
        ay = _function(_coordinate_name(patch_id, ring_index, a_index, "y"))
        bx = _function(_coordinate_name(patch_id, ring_index, b_index, "x"))
        by = _function(_coordinate_name(patch_id, ring_index, b_index, "y"))
        vx, vy = _binary(bx, ax, "-"), _binary(by, ay, "-")
        cross = _binary(
            _binary(vx, _binary(y, ay, "-"), "*"),
            _binary(vy, _binary(x, ax, "-"), "*"),
            "-",
        )
        signed = cross if orientation > 0.0 else _binary(_constant(0.0), cross, "-")
        result = _binary(result, _hard_step(_binary(signed, reference_squared, "/")), "*")
    return result


def _ring_characteristic(
    patch_id: str,
    ring_index: int,
    ring: Sequence[Sequence[float]],
    x: Sequence[str],
    y: Sequence[str],
) -> list[str]:
    triangles = triangulate_ring(ring, label=f"{patch_id}/{ring_index}")
    # Union of nonoverlapping ear triangles: 1 - product(1 - triangle_i).
    outside = _constant(1.0)
    for index, triangle in enumerate(triangles):
        outside = _binary(
            outside,
            _binary(
                _constant(1.0),
                _triangle_characteristic(
                    patch_id, ring_index, index, ring, triangle, x, y
                ),
                "-",
            ),
            "*",
        )
    return _binary(_constant(1.0), outside, "-")


def _segment_distances_squared(
    patch_id: str,
    ring_index: int,
    ring: Sequence[Sequence[float]],
    x: Sequence[str],
    y: Sequence[str],
) -> tuple[list[str], ...]:
    distances: list[list[str]] = []
    for index in range(len(ring)):
        following = (index + 1) % len(ring)
        ax = _function(_coordinate_name(patch_id, ring_index, index, "x"))
        ay = _function(_coordinate_name(patch_id, ring_index, index, "y"))
        bx = _function(_coordinate_name(patch_id, ring_index, following, "x"))
        by = _function(_coordinate_name(patch_id, ring_index, following, "y"))
        vx, vy = _binary(bx, ax, "-"), _binary(by, ay, "-")
        px, py = _binary(x, ax, "-"), _binary(y, ay, "-")
        parameter = _clamp_unit_interval(
            _binary(
                _binary(_binary(px, vx, "*"), _binary(py, vy, "*"), "+"),
                _binary(_square(vx), _square(vy), "+"),
                "/",
            )
        )
        dx = _binary(px, _binary(parameter, vx, "*"), "-")
        dy = _binary(py, _binary(parameter, vy, "*"), "-")
        distances.append(_binary(_square(dx), _square(dy), "+"))
    return tuple(distances)


def compile_mask_operations(
    *,
    patch_id: str,
    exterior: Sequence[Sequence[float]],
    holes: Sequence[Sequence[Sequence[float]]],
    margin_um: float,
    plane_origin_um: Sequence[float],
    plane_u: Sequence[float],
    plane_v: Sequence[float],
) -> list[str]:
    """Compile a hard polygon-with-holes and finite-original-boundary mask."""

    _, u, v = _plane_basis(plane_origin_um, plane_u, plane_v)
    x = _plane_coordinate(patch_id, u)
    y = _plane_coordinate(patch_id, v)
    characteristic = _ring_characteristic(patch_id, 0, exterior, x, y)
    for index, ring in enumerate(holes, start=1):
        characteristic = _binary(
            characteristic,
            _binary(
                _constant(1.0),
                _ring_characteristic(patch_id, index, ring, x, y),
                "-",
            ),
            "*",
        )
    distances = list(_segment_distances_squared(patch_id, 0, exterior, x, y))
    for index, ring in enumerate(holes, start=1):
        distances.extend(_segment_distances_squared(patch_id, index, ring, x, y))
    margin = _function(_margin_name(patch_id, margin_um))
    reference_squared = _square(_function("scgsim_mask_reference_um"))
    retained = _constant(1.0)
    # min(distance_i) >= margin iff every original boundary segment satisfies
    # distance_i >= margin.  Multiplying the hard criteria keeps the native
    # expression linear and excludes triangulation seams from the margin.
    for distance in distances:
        criterion = _binary(
            _binary(distance, _square(margin), "-"), reference_squared, "/"
        )
        retained = _binary(retained, _hard_step(criterion), "*")
    return _binary(characteristic, retained, "*")


def mask_region_union_variables(
    patch_id: str,
    support_regions: Sequence[Mapping[str, Any]],
    attribution_regions: Sequence[Mapping[str, Any]],
    margin_um: float,
    *,
    plane_origin_um: Sequence[float],
    plane_u: Sequence[float],
    plane_v: Sequence[float],
) -> dict[str, str]:
    """Bind original support-boundary coordinates and one requested margin."""

    values: dict[str, str] = {}
    for prefix, regions in (
        ("support", support_regions),
        ("attribution", attribution_regions),
    ):
        for index, region in enumerate(regions):
            if not isinstance(region, Mapping):
                raise TypeError("mask regions must contain mappings")
            item = mask_variables(
                f"{patch_id}_{prefix}_{index}",
                region.get("exterior", ()),
                region.get("holes", ()),
                (margin_um,) if prefix == "support" else (),
                plane_origin_um=plane_origin_um,
                plane_u=plane_u,
                plane_v=plane_v,
            )
            for name, value in item.items():
                if name in values and values[name] != value:
                    raise RuntimeError(f"mask variable {name!r} is contradictory")
                values[name] = value
    return values


def compile_mask_region_union_operations(
    *,
    patch_id: str,
    support_regions: Sequence[Mapping[str, Any]],
    attribution_regions: Sequence[Mapping[str, Any]],
    margin_um: float,
    plane_origin_um: Sequence[float],
    plane_u: Sequence[float],
    plane_v: Sequence[float],
) -> list[str]:
    """Compile exact support-distance mask intersected with owner attribution."""

    _, u, v = _plane_basis(plane_origin_um, plane_u, plane_v)

    def region_union(
        prefix: str,
        regions: Sequence[Mapping[str, Any]],
        *,
        exact_margin_um: float | None = None,
    ) -> list[str]:
        outside = _constant(1.0)
        for index, region in enumerate(regions):
            region_id = f"{patch_id}_{prefix}_{index}"
            x = _plane_coordinate(region_id, u)
            y = _plane_coordinate(region_id, v)
            if exact_margin_um is None:
                characteristic = _ring_characteristic(
                    region_id, 0, region["exterior"], x, y
                )
                for hole_index, ring in enumerate(region.get("holes", ()), start=1):
                    characteristic = _binary(
                        characteristic,
                        _binary(
                            _constant(1.0),
                            _ring_characteristic(region_id, hole_index, ring, x, y),
                            "-",
                        ),
                        "*",
                    )
            else:
                characteristic = compile_mask_operations(
                    patch_id=region_id,
                    exterior=region["exterior"],
                    holes=region.get("holes", ()),
                    margin_um=exact_margin_um,
                    plane_origin_um=plane_origin_um,
                    plane_u=plane_u,
                    plane_v=plane_v,
                )
            outside = _binary(
                outside, _binary(_constant(1.0), characteristic, "-"), "*"
            )
        return _binary(_constant(1.0), outside, "-")

    if not support_regions or not attribution_regions:
        return _constant(0.0)
    return _binary(
        region_union("support", support_regions, exact_margin_um=margin_um),
        region_union("attribution", attribution_regions),
        "*",
    )


def field_integral_operations(
    *,
    quantity: str,
    selection_name: str,
    mask_operations: Sequence[str] | None = None,
    adjacent_side: bool = False,
    normal_vector: Sequence[float] | None = None,
) -> list[str]:
    """Return the documented HFSS stack for one raw integral quantity."""

    if quantity == "effective_volume":
        value = _constant(1.0)
        terminal = [
            f"EnterVolume('{selection_name}')",
            "Operation('VolumeValue')",
            "Operation('Integrate')",
        ]
    elif quantity == "electric_volume":
        value = [
            "Fundamental_Quantity('E')",
            "Operation('Conj')",
            "Fundamental_Quantity('E')",
            "Operation('Dot')",
            "Operation('Real')",
        ]
        terminal = [
            f"EnterVolume('{selection_name}')",
            "Operation('VolumeValue')",
            "Operation('Integrate')",
        ]
    elif quantity == "magnetic_energy":
        value = [
            "NameOfExpression('<Hx,Hy,Hz>')",
            "Operation('Conj')",
            "NameOfExpression('<Hx,Hy,Hz>')",
            "Operation('Dot')",
            "Operation('Real')",
        ]
        terminal = [
            f"EnterVolume('{selection_name}')",
            "Operation('VolumeValue')",
            "Operation('Integrate')",
        ]
    elif quantity in {"electric_normal", "electric_tangential", "masked_area"}:
        # Area is independent of which field side is sampled.
        enter_surface = (
            "EnterAdjacentSurf"
            if adjacent_side and quantity != "masked_area"
            else "EnterSurface"
        )
        if quantity == "masked_area":
            value = _constant(1.0)
            terminal = [
                f"{enter_surface}('{selection_name}')",
                "Operation('SurfaceValue')",
                "Operation('Integrate')",
            ]
            if mask_operations:
                value = [*value, *mask_operations, "Operation('*')"]
            return [*value, *terminal]
        if normal_vector is None or len(normal_vector) != 3 or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in normal_vector
        ):
            raise ValueError("surface field integral requires a finite native normal")
        length = math.sqrt(sum(float(item) ** 2 for item in normal_vector))
        if not math.isfinite(length) or length == 0.0:
            raise ValueError("surface field integral requires a finite nonzero native normal")
        vector = "Vector_Constant(" + ", ".join(
            f"{float(item) / length:.17g}" for item in normal_vector
        ) + ")"
        # A bound Normal scalar cannot be multiplied by the spatial mask in CLC.
        normal_real = [
            "Fundamental_Quantity('E')",
            "Operation('Real')",
            vector,
            "Operation('Dot')",
        ]
        normal_imag = [
            "Fundamental_Quantity('E')",
            "Operation('Imag')",
            vector,
            "Operation('Dot')",
        ]
        normal_square = _binary(_square(normal_real), _square(normal_imag), "+")
        if quantity == "electric_normal":
            value = normal_square
        else:
            full_real = ["Fundamental_Quantity('E')", "Operation('Real')"]
            full_imag = ["Fundamental_Quantity('E')", "Operation('Imag')"]
            full_square = _binary(
                [*full_real, *full_real, "Operation('Dot')"],
                [*full_imag, *full_imag, "Operation('Dot')"],
                "+",
            )
            value = _binary(full_square, normal_square, "-")
        # This records side intent in the expression identity.  The adjacent
        # expression is authored through the native reporter, not this CLC
        # spelling, whose saved-file form is not established by PyAEDT.
        terminal = [
            f"{enter_surface}('{selection_name}')",
            "Operation('SurfaceValue')",
            "Operation('Integrate')",
        ]
    elif quantity in {"junction_voltage_real", "junction_voltage_imag"}:
        raise ValueError(
            "junction voltage requires an explicit directed vector; use "
            "junction_voltage_operations"
        )
    else:
        raise ValueError(f"unsupported HFSS EPR field quantity {quantity!r}")
    if adjacent_side and quantity not in {"electric_normal", "electric_tangential"}:
        raise ValueError("adjacent_side is valid only for surface field quantities")
    if mask_operations:
        value = [*value, *mask_operations, "Operation('*')"]
    return [*value, *terminal]


def junction_voltage_operations(
    *, quantity: str, selection_name: str, direction_xy: Sequence[float]
) -> list[str]:
    """Integrate Re/Im(E) dot the declared in-plane terminal direction."""

    if quantity not in {"junction_voltage_real", "junction_voltage_imag"}:
        raise ValueError("junction voltage quantity must be real or imaginary")
    if len(direction_xy) != 2:
        raise ValueError("junction direction requires two values")
    dx, dy = float(direction_xy[0]), float(direction_xy[1])
    norm = math.hypot(dx, dy)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("junction direction must be finite and nonzero")
    component = "Real" if quantity.endswith("real") else "Imag"
    return [
        "Fundamental_Quantity('E')",
        f"Operation('{component}')",
        f"Vector_Constant({dx / norm:.17g}, {dy / norm:.17g}, 0)",
        "Operation('Dot')",
        f"EnterSurface('{selection_name}')",
        "Operation('SurfaceValue')",
        "Operation('Integrate')",
    ]


def create_verified_face_list(
    app: Any, *, name: str, face_ids: Sequence[int]
) -> dict[str, Any]:
    ids = [int(value) for value in face_ids]
    if not ids or len(ids) != len(set(ids)) or any(value <= 0 for value in ids):
        raise ValueError("face list requires distinct positive native face ids")
    if name in {str(item.name) for item in app.modeler.user_lists}:
        raise RuntimeError(f"HFSS face-list name already exists: {name!r}")
    item = app.modeler.create_face_list(ids, name=name)
    if item is False or item is None or item.name != name or item.props.get("Type") != "Face":
        raise RuntimeError(f"HFSS face-list creation failed: {name!r}")
    native_id = app.modeler.get_entitylist_id(name)
    if type(native_id) is not int or native_id <= 0 or native_id != item.props.get("ID"):
        raise RuntimeError(f"HFSS face-list readback failed: {name!r}")
    members = item.props.get("EntityList")
    if isinstance(members, (int, str)):
        members = [members]
    if [int(value) for value in members or ()] != ids:
        raise RuntimeError(f"HFSS face-list membership differs: {name!r}")
    return {"name": name, "native_id": native_id, "face_ids": ids}


def evaluate_named_expression(
    app: Any,
    *,
    purpose: str,
    operations: Sequence[str],
    solution: str,
    phase_degrees: float,
    dependencies: Mapping[str, str],
    selection: Mapping[str, Any],
    evidence_dir: Path,
) -> dict[str, Any]:
    """Author and evaluate one immutable named expression in a bound context."""

    authored = author_named_expression(
        app,
        purpose=purpose,
        operations=operations,
        solution=solution,
        phase_degrees=phase_degrees,
        dependencies=dependencies,
        selection=selection,
        evidence_dir=evidence_dir,
    )
    name = authored["name"]
    raw = app.post.fields_calculator.evaluate(
        name,
        setup=solution,
        intrinsics={"Phase": f"{phase_degrees:g}deg"},
    )
    return {
        **authored,
        "scalar": parse_native_scalar(raw),
    }


def author_named_expression(
    app: Any,
    *,
    purpose: str,
    operations: Sequence[str],
    solution: str,
    phase_degrees: float,
    dependencies: Mapping[str, str],
    selection: Mapping[str, Any],
    evidence_dir: Path,
    namespace: str = "prepared",
    adjacent_selection_name: str | None = None,
) -> dict[str, Any]:
    """Author one immutable expression without requiring solved fields."""

    if solution != str(solution).strip() or not solution:
        raise ValueError("solution must be non-empty canonical text")
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("purpose must be non-empty text")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", namespace):
        raise ValueError("expression namespace must be a safe native identifier")
    if isinstance(operations, (str, bytes)) or any(
        not isinstance(item, str) or not item or "\n" in item or "\0" in item
        for item in operations
    ):
        raise ValueError("expression operations must be canonical text entries")
    for key in ("object_name", "selection_name"):
        value = selection.get(key)
        if value is not None and (
            not isinstance(value, str)
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value)
        ):
            raise ValueError(f"expression selection {key} is not a safe native name")
    identity = {
        "schema_version": "scgsim.aedt.epr-expression.v1",
        "namespace": namespace,
        "purpose": purpose,
        "operations": list(operations),
        "solution": solution,
        "phase_degrees": float(phase_degrees),
        "dependencies": dict(sorted(dependencies.items())),
        "selection": dict(selection),
    }
    identity["sha256"] = _digest(identity)
    name = f"scgsim_epr_{identity['sha256'][:24]}"
    calculator = app.post.fields_calculator
    if calculator.is_expression_defined(name):
        raise RuntimeError(f"HFSS expression already exists: {name!r}")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    retained = evidence_dir / f"{name}.clc"
    if adjacent_selection_name is None:
        authored = calculator.create_expression_file(name, list(operations))
        if authored is False or not isinstance(authored, str):
            raise RuntimeError(f"HFSS expression authoring failed: {name!r}")
        source = Path(authored)
        retained.write_bytes(source.read_bytes())
        calculator.ofieldsreporter.LoadNamedExpressions(str(source), "Fields", [name])
    else:
        if (
            selection.get("selection_name") != adjacent_selection_name
            or list(operations[-3:])
            != [
                f"EnterAdjacentSurf('{adjacent_selection_name}')",
                "Operation('SurfaceValue')",
                "Operation('Integrate')",
            ]
        ):
            raise ValueError("adjacent surface expression selection is inconsistent")
        integrand_name = f"{name}_integrand"
        authored = calculator.create_expression_file(
            integrand_name, list(operations[:-3])
        )
        if authored is False or not isinstance(authored, str):
            raise RuntimeError(f"HFSS adjacent integrand authoring failed: {name!r}")
        integrand_bytes = Path(authored).read_bytes()
        (evidence_dir / f"{integrand_name}.clc").write_bytes(integrand_bytes)
        reporter = calculator.ofieldsreporter
        reporter.LoadNamedExpressions(str(authored), "Fields", [integrand_name])
        if not calculator.is_expression_defined(integrand_name):
            raise RuntimeError(f"HFSS adjacent integrand was not defined: {name!r}")
        reporter.CalcStack("clear")
        reporter.CopyNamedExprToStack(integrand_name)
        reporter.EnterAdjacentSurf(adjacent_selection_name)
        reporter.CalcOp("SurfaceValue")
        reporter.CalcOp("Integrate")
        reporter.AddNamedExpression(name, "Fields")
        if not calculator.is_expression_defined(name):
            raise RuntimeError(f"HFSS adjacent expression was not defined: {name!r}")
        reporter.SaveNamedExpressions(str(retained), [name], True)
        reporter.CalcStack("clear")
        if not retained.is_file() or retained.stat().st_size == 0:
            raise RuntimeError(f"HFSS adjacent expression was not saved: {name!r}")
    if not calculator.is_expression_defined(name):
        raise RuntimeError(f"HFSS expression was not defined: {name!r}")
    return {
        "name": name,
        "identity": identity,
        "definition_sha256": hashlib.sha256(retained.read_bytes()).hexdigest(),
        "definition_status": "authored_and_loaded",
        "solution": solution,
        "phase_degrees": float(phase_degrees),
    }


__all__ = [
    "author_named_expression",
    "compile_mask_operations",
    "compile_mask_region_union_operations",
    "create_verified_face_list",
    "evaluate_named_expression",
    "field_integral_operations",
    "install_variables",
    "mask_variables",
    "mask_region_union_variables",
    "parse_native_scalar",
    "junction_voltage_operations",
    "triangulate_ring",
]
