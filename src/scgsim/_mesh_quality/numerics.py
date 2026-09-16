"""Range-safe tetrahedron orientation, volume, center, and condition metrics.

For vertices p0..p3, B=[p1-p0 p2-p0 p3-p0], D=det(B), signed
volume is D/6, and volume magnitude is abs(D)/6.  The reference edge matrix
W has rows (1,1/2,1/2), (0,sqrt(3)/2,sqrt(3)/6), and
(0,0,sqrt(2/3)); J=B inv(W) and kappa(J)=smax/smin.

The floating orientation filter assumes IEEE-754 binary64 round-to-nearest.
One exactly reversible, element-wide power-of-two scaling occurs before edge
subtraction.  The explicit determinant expansion uses the matching bound
E=(7+56u)uP with u=2**-53.  This certifies sign only, not relative volume
accuracy. Unsafe scaling, subnormal intermediates, inconclusive signs, and
representation-boundary ambiguity use exact dyadic arithmetic from the twelve
original parsed binary64 coordinates.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .reader import MeshData


_U = 2.0**-53
_EPS = 2.0**-52
_MIN_NORMAL = sys.float_info.min
_W = np.asarray(
    (
        (1.0, 0.5, 0.5),
        (0.0, math.sqrt(3.0) / 2.0, math.sqrt(3.0) / 6.0),
        (0.0, 0.0, math.sqrt(2.0 / 3.0)),
    ),
    dtype=np.float64,
)
_INV_W = np.linalg.inv(_W)
_INV_W.flags.writeable = False


@dataclass(frozen=True, slots=True)
class Measurements:
    orientation: np.ndarray
    orientation_method: np.ndarray
    determinant_estimate: np.ndarray
    determinant_error_bound: np.ndarray
    determinant_scale_exponent: np.ndarray
    signed_volume: np.ndarray
    volume: np.ndarray
    volume_status: np.ndarray
    volume_method: np.ndarray
    volume_error_bound: np.ndarray
    signed_volume_m3: np.ndarray
    volume_m3: np.ndarray
    volume_m3_status: np.ndarray
    volume_m3_method: np.ndarray
    volume_m3_error_bound: np.ndarray
    kappa_j: np.ndarray
    condition_status: np.ndarray
    centers: np.ndarray
    center_status: np.ndarray
    fallback_used: np.ndarray


def _readonly(value: np.ndarray) -> np.ndarray:
    value.flags.writeable = False
    return value


def _exact_determinant(points: np.ndarray) -> Fraction:
    exact = [
        [Fraction.from_float(float(points[row, axis])) for axis in range(3)]
        for row in range(4)
    ]
    a = [exact[1][axis] - exact[0][axis] for axis in range(3)]
    b = [exact[2][axis] - exact[0][axis] for axis in range(3)]
    c = [exact[3][axis] - exact[0][axis] for axis in range(3)]
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - b[0] * (a[1] * c[2] - a[2] * c[1])
        + c[0] * (a[1] * b[2] - a[2] * b[1])
    )


def _normal_rows(values: np.ndarray) -> np.ndarray:
    """Return rows whose nonzero binary64 values are finite and normal."""

    axes = tuple(range(1, values.ndim))
    valid = np.isfinite(values) & ((values == 0.0) | (np.abs(values) >= _MIN_NORMAL))
    return np.all(valid, axis=axes)


def _filtered_determinants(
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorize the explicit determinant and matching sign bound for one chunk."""

    a = edges[:, :, 0]
    b = edges[:, :, 1]
    c = edges[:, :, 2]
    products = np.column_stack(
        (
            b[:, 1] * c[:, 2],
            b[:, 2] * c[:, 1],
            a[:, 1] * c[:, 2],
            a[:, 2] * c[:, 1],
            a[:, 1] * b[:, 2],
            a[:, 2] * b[:, 1],
        )
    )
    differences = np.column_stack(
        (
            products[:, 0] - products[:, 1],
            products[:, 2] - products[:, 3],
            products[:, 4] - products[:, 5],
        )
    )
    terms = np.column_stack(
        (
            a[:, 0] * differences[:, 0],
            -b[:, 0] * differences[:, 1],
            c[:, 0] * differences[:, 2],
        )
    )
    determinant = (terms[:, 0] + terms[:, 1]) + terms[:, 2]
    permanent = (
        np.abs(a[:, 0]) * (np.abs(products[:, 0]) + np.abs(products[:, 1]))
        + np.abs(b[:, 0]) * (np.abs(products[:, 2]) + np.abs(products[:, 3]))
        + np.abs(c[:, 0]) * (np.abs(products[:, 4]) + np.abs(products[:, 5]))
    )
    bound = (7.0 + 56.0 * _U) * _U * permanent
    intermediates = np.column_stack(
        (products, differences, terms, determinant, permanent, bound)
    )
    accepted = _normal_rows(intermediates) & (np.abs(determinant) > bound)
    return determinant, bound, accepted


def _fraction_float(value: Fraction) -> tuple[float, str]:
    if value == 0:
        return 0.0, "exact_zero"
    try:
        converted = float(value)
    except OverflowError:
        return math.nan, "overflow"
    if not math.isfinite(converted):
        return math.nan, "overflow"
    if converted == 0.0:
        return math.nan, "underflow"
    return converted, "available"


def _restore_interval(
    estimate: float,
    bound: float,
    exponent: int,
    *,
    scale_m: float | None,
) -> tuple[float, float, str]:
    factor = 1.0 / 6.0
    restore_exponent = 3 * exponent
    if scale_m is not None:
        mantissa, scale_exponent = math.frexp(scale_m)
        factor *= mantissa**3
        restore_exponent += 3 * scale_exponent
    low = max(0.0, abs(estimate) - bound) * factor
    middle = abs(estimate) * factor
    high = (abs(estimate) + bound) * factor
    try:
        restored = math.ldexp(middle, restore_exponent)
        restored_low = math.ldexp(low, restore_exponent)
        restored_high = math.ldexp(high, restore_exponent)
    except OverflowError:
        return math.nan, math.nan, "ambiguous_representation"
    if (
        not math.isfinite(restored)
        or not math.isfinite(restored_high)
        or (low > 0.0 and restored_low == 0.0)
        or (middle > 0.0 and restored == 0.0)
    ):
        return math.nan, math.nan, "ambiguous_representation"
    try:
        restored_bound = math.ldexp(bound * factor, restore_exponent)
    except OverflowError:
        return math.nan, math.nan, "ambiguous_representation"
    if not math.isfinite(restored_bound):
        return math.nan, math.nan, "ambiguous_representation"
    restored_bound += (
        16.0 * _U * (abs(restored) + abs(restored_bound)) + math.ulp(restored)
    )
    if restored <= restored_bound:
        return math.nan, math.nan, "ambiguous_representation"
    return restored, restored_bound, "available"


def _exact_center(points: np.ndarray) -> tuple[np.ndarray, str]:
    values = np.empty(3, dtype=np.float64)
    for axis in range(3):
        exact = sum(
            (Fraction.from_float(float(points[row, axis])) for row in range(4)),
            Fraction(),
        ) / 4
        value, status = _fraction_float(exact)
        if status not in {"available", "exact_zero"}:
            return np.full(3, np.nan), f"center_{status}"
        values[axis] = value
    return values, "available"


def _scaled_center(
    scaled_points: np.ndarray, original_points: np.ndarray, exponent: int
) -> tuple[np.ndarray, str]:
    totals = np.asarray(
        [
            math.fsum(float(value) for value in scaled_points[:, axis])
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    try:
        restored = np.asarray(
            [math.ldexp(float(value), exponent - 2) for value in totals],
            dtype=np.float64,
        )
    except OverflowError:
        return _exact_center(original_points)
    if not np.all(np.isfinite(restored)):
        return _exact_center(original_points)
    if np.any((totals != 0.0) & (restored == 0.0)):
        return _exact_center(original_points)
    if np.any((restored != 0.0) & (np.abs(restored) < _MIN_NORMAL)):
        return _exact_center(original_points)
    return restored, "available"


def _condition_chunk(
    edges: np.ndarray,
    rows: np.ndarray,
    condition_status: np.ndarray,
    kappa_j: np.ndarray,
) -> None:
    """Measure one bounded SVD chunk and isolate a failing LAPACK element."""

    batch = edges @ _INV_W
    try:
        singular_values = np.linalg.svd(batch, compute_uv=False)
        results = [
            (int(index), values, None)
            for index, values in zip(rows, singular_values, strict=True)
        ]
    except np.linalg.LinAlgError:
        results = []
        for index, matrix in zip(rows, batch, strict=True):
            try:
                values = np.linalg.svd(matrix, compute_uv=False)
                results.append((int(index), values, None))
            except np.linalg.LinAlgError as error:
                results.append((int(index), None, error))
    for index, values, error in results:
        if condition_status[index] == "exact_singular":
            continue
        if error is not None or values is None or not np.all(np.isfinite(values)):
            condition_status[index] = "svd_failed"
            continue
        maximum = float(values[0])
        minimum = float(values[-1])
        if maximum == 0.0 or minimum == 0.0 or minimum <= 3.0 * _EPS * maximum:
            condition_status[index] = "numerically_unresolved"
            continue
        kappa_j[index] = maximum / minimum
        condition_status[index] = "available"


def measure(mesh: MeshData, length_scale_m: float | None) -> Measurements:
    """Measure all locatable type-4 elements without changing their orientation."""

    count = len(mesh.element_ids)
    orientation = np.zeros(count, dtype=np.int8)
    orientation_method = np.full(count, "unavailable", dtype="U32")
    determinant_estimate = np.full(count, np.nan, dtype=np.float64)
    determinant_error_bound = np.full(count, np.nan, dtype=np.float64)
    determinant_scale_exponent = np.zeros(count, dtype=np.int32)
    signed_volume = np.full(count, np.nan, dtype=np.float64)
    volume = np.full(count, np.nan, dtype=np.float64)
    volume_status = np.full(count, "unavailable", dtype="U32")
    volume_method = np.full(count, "unavailable", dtype="U32")
    volume_error_bound = np.full(count, np.nan, dtype=np.float64)
    signed_volume_m3 = np.full(count, np.nan, dtype=np.float64)
    volume_m3 = np.full(count, np.nan, dtype=np.float64)
    volume_m3_status = np.full(
        count, "not_requested" if length_scale_m is None else "unavailable", dtype="U32"
    )
    volume_m3_method = np.full(
        count, "not_requested" if length_scale_m is None else "unavailable", dtype="U32"
    )
    volume_m3_error_bound = np.full(count, np.nan, dtype=np.float64)
    kappa_j = np.full(count, np.nan, dtype=np.float64)
    condition_status = np.full(count, "unavailable", dtype="U32")
    centers = np.full((count, 3), np.nan, dtype=np.float64)
    center_status = np.full(count, "unavailable", dtype="U32")
    fallback_used = np.zeros(count, dtype=np.bool_)
    chunk_size = 4096
    for start in range(0, count, chunk_size):
        rows = np.arange(start, min(start + chunk_size, count), dtype=np.int64)
        indices = mesh.node_indices[rows]
        missing = np.any(indices < 0, axis=1)
        missing_rows = rows[missing]
        orientation_method[missing_rows] = "missing_nodes"
        volume_status[missing_rows] = "missing_nodes"
        volume_method[missing_rows] = "missing_nodes"
        condition_status[missing_rows] = "missing_nodes"
        center_status[missing_rows] = "missing_nodes"
        if length_scale_m is not None:
            volume_m3_status[missing_rows] = "missing_nodes"
            volume_m3_method[missing_rows] = "missing_nodes"

        present_rows = rows[~missing]
        if not len(present_rows):
            continue
        present_points = mesh.coordinates[mesh.node_indices[present_rows]]
        finite = np.all(np.isfinite(present_points), axis=(1, 2))
        nonfinite_rows = present_rows[~finite]
        orientation_method[nonfinite_rows] = "nonfinite_coordinates"
        volume_status[nonfinite_rows] = "nonfinite_coordinates"
        volume_method[nonfinite_rows] = "nonfinite_coordinates"
        condition_status[nonfinite_rows] = "nonfinite_coordinates"
        center_status[nonfinite_rows] = "nonfinite_coordinates"
        if length_scale_m is not None:
            volume_m3_status[nonfinite_rows] = "nonfinite_coordinates"
            volume_m3_method[nonfinite_rows] = "nonfinite_coordinates"

        valid_rows = present_rows[finite]
        points = present_points[finite]
        if not len(valid_rows):
            continue
        maximum = np.max(np.abs(points), axis=(1, 2))
        exponents = np.frexp(maximum)[1].astype(np.int32)
        exponents[maximum == 0.0] = 0
        determinant_scale_exponent[valid_rows] = exponents
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            scaled = np.ldexp(points, -exponents[:, None, None])
            restored = np.ldexp(scaled, exponents[:, None, None])
            edges = np.stack(
                (
                    scaled[:, 1] - scaled[:, 0],
                    scaled[:, 2] - scaled[:, 0],
                    scaled[:, 3] - scaled[:, 0],
                ),
                axis=2,
            )
        scaled_normal = _normal_rows(scaled)
        reversible = scaled_normal & np.all(restored == points, axis=(1, 2))
        reversible &= np.all((points == 0.0) | (scaled != 0.0), axis=(1, 2))
        edge_normal = _normal_rows(edges)
        safe = reversible & edge_normal

        condition_status[valid_rows[~scaled_normal]] = "scaling_subnormal"
        condition_status[valid_rows[scaled_normal & ~reversible]] = (
            "scaling_not_reversible"
        )
        condition_status[valid_rows[reversible & ~edge_normal]] = (
            "unsafe_subtraction"
        )
        for local, row in enumerate(valid_rows):
            if safe[local]:
                centers[row], center_status[row] = _scaled_center(
                    scaled[local], points[local], int(exponents[local])
                )
            else:
                centers[row], center_status[row] = _exact_center(points[local])

        safe_rows = valid_rows[safe]
        safe_edges = edges[safe]
        accepted_rows = np.empty(0, dtype=np.int64)
        if len(safe_rows):
            estimates, bounds, accepted = _filtered_determinants(safe_edges)
            determinant_estimate[safe_rows] = estimates
            determinant_error_bound[safe_rows] = bounds
            accepted_rows = safe_rows[accepted]
            orientation[accepted_rows] = np.where(estimates[accepted] > 0.0, 1, -1)
            orientation_method[accepted_rows] = "floating_filter"

        exact_cache: dict[int, Fraction] = {}

        def exact_for(row: int) -> Fraction:
            if row not in exact_cache:
                exact_cache[row] = _exact_determinant(
                    mesh.coordinates[mesh.node_indices[row]]
                )
            return exact_cache[row]

        def apply_exact_mesh(row: int) -> None:
            exact_determinant = exact_for(row)
            fallback_used[row] = True
            sign = 1 if exact_determinant > 0 else -1 if exact_determinant < 0 else 0
            orientation[row] = sign
            orientation_method[row] = "exact_dyadic"
            exact_signed = exact_determinant / 6
            magnitude, status = _fraction_float(abs(exact_signed))
            signed, signed_status = _fraction_float(exact_signed)
            volume_status[row] = status
            volume_method[row] = "exact_rounded"
            if status in {"available", "exact_zero"}:
                volume[row] = magnitude
            if signed_status in {"available", "exact_zero"}:
                signed_volume[row] = signed
            if sign == 0:
                condition_status[row] = "exact_singular"
                kappa_j[row] = math.inf

        rejected_rows = np.concatenate(
            (
                valid_rows[~safe],
                safe_rows[orientation_method[safe_rows] != "floating_filter"],
            )
        )
        for row_value in rejected_rows:
            apply_exact_mesh(int(row_value))
        for row_value in accepted_rows:
            row = int(row_value)
            restored_volume, restored_bound, restored_status = _restore_interval(
                float(determinant_estimate[row]),
                float(determinant_error_bound[row]),
                int(determinant_scale_exponent[row]),
                scale_m=None,
            )
            if restored_status == "available":
                volume[row] = restored_volume
                signed_volume[row] = math.copysign(restored_volume, orientation[row])
                volume_status[row] = "available"
                volume_method[row] = "floating_estimate"
                volume_error_bound[row] = restored_bound
            else:
                apply_exact_mesh(row)

        if length_scale_m is not None:
            exact_scale = Fraction.from_float(length_scale_m) ** 3
            for row_value in valid_rows:
                row = int(row_value)
                if row not in exact_cache:
                    restored_si, restored_si_bound, status_si = _restore_interval(
                        float(determinant_estimate[row]),
                        float(determinant_error_bound[row]),
                        int(determinant_scale_exponent[row]),
                        scale_m=length_scale_m,
                    )
                    if status_si == "available":
                        volume_m3[row] = restored_si
                        signed_volume_m3[row] = math.copysign(
                            restored_si, orientation[row]
                        )
                        volume_m3_status[row] = "available"
                        volume_m3_method[row] = "floating_estimate"
                        volume_m3_error_bound[row] = restored_si_bound
                        continue
                    exact_for(row)
                    fallback_used[row] = True
                exact_signed_si = exact_cache[row] * exact_scale / 6
                value_si, status_si = _fraction_float(abs(exact_signed_si))
                signed_si, signed_si_status = _fraction_float(exact_signed_si)
                volume_m3_status[row] = status_si
                volume_m3_method[row] = "exact_rounded"
                if status_si in {"available", "exact_zero"}:
                    volume_m3[row] = value_si
                if signed_si_status in {"available", "exact_zero"}:
                    signed_volume_m3[row] = signed_si

        if len(safe_rows):
            _condition_chunk(safe_edges, safe_rows, condition_status, kappa_j)

    return Measurements(
        orientation=_readonly(orientation),
        orientation_method=_readonly(orientation_method),
        determinant_estimate=_readonly(determinant_estimate),
        determinant_error_bound=_readonly(determinant_error_bound),
        determinant_scale_exponent=_readonly(determinant_scale_exponent),
        signed_volume=_readonly(signed_volume),
        volume=_readonly(volume),
        volume_status=_readonly(volume_status),
        volume_method=_readonly(volume_method),
        volume_error_bound=_readonly(volume_error_bound),
        signed_volume_m3=_readonly(signed_volume_m3),
        volume_m3=_readonly(volume_m3),
        volume_m3_status=_readonly(volume_m3_status),
        volume_m3_method=_readonly(volume_m3_method),
        volume_m3_error_bound=_readonly(volume_m3_error_bound),
        kappa_j=_readonly(kappa_j),
        condition_status=_readonly(condition_status),
        centers=_readonly(centers),
        center_status=_readonly(center_status),
        fallback_used=_readonly(fallback_used),
    )
