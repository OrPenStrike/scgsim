"""Immutable report and presentation surface for offline mesh diagnostics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from importlib import metadata
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .numerics import Measurements, _decode_label, _StatusCode
from .reader import MeshData


_SCHEMA = "scgsim.mesh-quality.v1"


def _distribution_version() -> str:
    try:
        return metadata.version("scgsim")
    except metadata.PackageNotFoundError:
        return "unavailable"


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _stats(values: np.ndarray, included: np.ndarray, total: int) -> dict[str, Any]:
    selected = values[included]
    finite = selected[np.isfinite(selected)]
    if not len(finite):
        return {
            "sample_count": 0,
            "excluded_count": total,
            "min": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    def position(q: float) -> float:
        rank = (len(finite) - 1) * q
        lower = math.floor(rank)
        upper = min(lower + 1, len(finite) - 1)
        working = np.array(finite, dtype=np.float64, copy=True)
        endpoints = np.partition(working, (lower, upper))
        if lower == upper:
            return float(endpoints[lower])
        weight = Fraction.from_float(rank - lower)
        exact = (
            (1 - weight) * Fraction.from_float(float(endpoints[lower]))
            + weight * Fraction.from_float(float(endpoints[upper]))
        )
        result = float(exact)
        if not math.isfinite(result):
            raise RuntimeError("finite statistic interpolation produced a nonfinite value")
        return result

    return {
        "sample_count": int(len(finite)),
        "excluded_count": int(total - len(finite)),
        "min": float(np.min(finite)),
        "median": position(0.5),
        "p95": position(0.95),
        "p99": position(0.99),
        "max": float(np.max(finite)),
    }


def _make_summary(mesh: MeshData, metrics: Measurements) -> Mapping[str, Any]:
    total = len(mesh.element_ids)
    measured = np.isin(
        metrics.orientation_method,
        (_StatusCode.FLOATING_FILTER, _StatusCode.EXACT_DYADIC),
    )
    volume_available = np.isin(
        metrics.volume_status, (_StatusCode.AVAILABLE, _StatusCode.EXACT_ZERO)
    )
    volume_m3_available = np.isin(
        metrics.volume_m3_status, (_StatusCode.AVAILABLE, _StatusCode.EXACT_ZERO)
    )
    kappa_available = metrics.condition_status == _StatusCode.AVAILABLE
    warnings: list[str] = []
    exact_zero = int(
        np.count_nonzero(
            (metrics.orientation == 0)
            & (metrics.orientation_method == _StatusCode.EXACT_DYADIC)
        )
    )
    missing = int(
        np.count_nonzero(metrics.orientation_method == _StatusCode.MISSING_NODES)
    )
    nonfinite = int(
        np.count_nonzero(
            metrics.orientation_method == _StatusCode.NONFINITE_COORDINATES
        )
    )
    condition_codes, condition_values = np.unique(
        metrics.condition_status, return_counts=True
    )
    condition_counts = {
        _decode_label(code): int(value)
        for code, value in zip(condition_codes, condition_values, strict=True)
    }
    condition_unavailable = ~np.isin(
        metrics.condition_status,
        (_StatusCode.AVAILABLE, _StatusCode.EXACT_SINGULAR),
    )
    unavailable_conditions = int(np.count_nonzero(condition_unavailable))
    unresolved = condition_counts.get("numerically_unresolved", 0)
    if total == 0:
        warnings.append("No four-node type-4 tetrahedra were present; no 3D quality metric was computed.")
    if exact_zero:
        warnings.append(f"{exact_zero} tetrahedra have exact zero volume.")
    if missing or nonfinite:
        warnings.append(
            f"{missing + nonfinite} tetrahedra are unmeasurable because coordinates are missing or nonfinite."
        )
    uncovered = int(mesh.element_counts["other_known_3d"])
    if uncovered:
        warnings.append(
            f"{uncovered} known 3D elements are not four-node tetrahedra and are outside this metric."
        )
    unknown = int(mesh.element_counts["unknown"])
    if unknown:
        warnings.append(
            f"{unknown} elements use unknown Gmsh element types and were counted without inferred dimension or arity."
        )
    if unavailable_conditions:
        warnings.append(
            f"{unavailable_conditions} tetrahedra have unavailable condition numbers; status counts remain explicit."
        )
    representation_failures = int(
        np.count_nonzero(
            np.isin(
                metrics.volume_status,
                (_StatusCode.UNDERFLOW, _StatusCode.OVERFLOW),
            )
        )
        + np.count_nonzero(
            np.isin(
                metrics.volume_m3_status,
                (_StatusCode.UNDERFLOW, _StatusCode.OVERFLOW),
            )
        )
    )
    if representation_failures:
        warnings.append(
            f"{representation_failures} mesh-unit or SI volume results are unavailable because binary64 cannot represent them."
        )
    summary = {
        "counts": {
            "type4_tetrahedra": total,
            "checked_tetrahedra": int(np.count_nonzero(measured)),
            "unchecked_tetrahedra": int(total - np.count_nonzero(measured)),
            "available_volume": int(np.count_nonzero(volume_available)),
            "unavailable_volume": int(total - np.count_nonzero(volume_available)),
            "negative_orientation": int(np.count_nonzero(metrics.orientation < 0)),
            "exact_zero": exact_zero,
            "missing_nodes": missing,
            "nonfinite_coordinates": nonfinite,
            "condition_exact_singular": int(
                np.count_nonzero(
                    metrics.condition_status == _StatusCode.EXACT_SINGULAR
                )
            ),
            "condition_numerically_unresolved": unresolved,
            "condition_svd_failed": int(
                np.count_nonzero(metrics.condition_status == _StatusCode.SVD_FAILED)
            ),
            "condition_unavailable": unavailable_conditions,
            "exact_fallback": int(np.count_nonzero(metrics.fallback_used)),
            "volume_underflow": int(
                np.count_nonzero(metrics.volume_status == _StatusCode.UNDERFLOW)
            ),
            "volume_overflow": int(
                np.count_nonzero(metrics.volume_status == _StatusCode.OVERFLOW)
            ),
            "volume_m3_underflow": int(
                np.count_nonzero(metrics.volume_m3_status == _StatusCode.UNDERFLOW)
            ),
            "volume_m3_overflow": int(
                np.count_nonzero(metrics.volume_m3_status == _StatusCode.OVERFLOW)
            ),
            **{key: int(value) for key, value in mesh.element_counts.items()},
        },
        "statistics": {
            "volume": _stats(metrics.volume, volume_available, total),
            "volume_m3": _stats(metrics.volume_m3, volume_m3_available, total),
            "kappa_j": _stats(metrics.kappa_j, kappa_available, total),
        },
        "condition_status_counts": condition_counts,
        "warnings": warnings,
    }
    return _freeze(summary)


@dataclass(frozen=True, slots=True)
class MeshQualityReport:
    """Detached diagnostic snapshot; it never rereads or modifies its source mesh."""

    _mesh: MeshData
    _metrics: Measurements
    _length_scale_m: float | None
    _summary: Mapping[str, Any]

    @classmethod
    def create(
        cls, mesh: MeshData, metrics: Measurements, length_scale_m: float | None
    ) -> "MeshQualityReport":
        return cls(mesh, metrics, length_scale_m, _make_summary(mesh, metrics))

    @property
    def summary(self) -> dict[str, Any]:
        """Return detached aggregate data, not aliases to the retained snapshot."""

        return _plain(self._summary)

    def _tags(self, row: int) -> list[int]:
        start, size = (int(value) for value in self._mesh.tag_offsets[row])
        return [int(value) for value in self._mesh.raw_tags[start : start + size]]

    def _coordinates(self, row: int) -> tuple[list[list[float | None]] | None, str]:
        indices = self._mesh.node_indices[row]
        if np.any(indices < 0):
            return None, "missing_nodes"
        values = self._mesh.coordinates[indices]
        output = [
            [_finite(float(value)) for value in point]
            for point in values
        ]
        if not np.all(np.isfinite(values)):
            return output, "nonfinite_coordinates"
        return output, "available"

    def _element_record(self, row: int) -> dict[str, Any]:
        physical_tag = int(self._mesh.physical_tags[row]) or None
        elementary_tag = int(self._mesh.elementary_tags[row]) or None
        coordinates, coordinate_status = self._coordinates(row)
        condition_status = _decode_label(self._metrics.condition_status[row])
        orientation_method = _decode_label(self._metrics.orientation_method[row])
        orientation_value = (
            int(self._metrics.orientation[row])
            if orientation_method in {"floating_filter", "exact_dyadic"}
            else None
        )
        orientation_status = (
            "positive"
            if orientation_value == 1
            else "negative"
            if orientation_value == -1
            else "exact_zero"
            if orientation_value == 0
            else "indeterminate"
        )
        kappa: float | str | None
        if condition_status == "exact_singular":
            kappa = "+inf"
        else:
            kappa = _finite(float(self._metrics.kappa_j[row]))
        center = [
            _finite(float(value)) for value in self._metrics.centers[row]
        ]
        center_status = _decode_label(self._metrics.center_status[row])
        if center_status != "available":
            center = [None, None, None]
        return {
            "element_id": int(self._mesh.element_ids[row]),
            "connectivity": [int(value) for value in self._mesh.connectivity[row]],
            "raw_tags": self._tags(row),
            "physical_tag": physical_tag,
            "elementary_tag": elementary_tag,
            "physical_name": (
                self._mesh.physical_names.get((3, physical_tag))
                if physical_tag is not None
                else None
            ),
            "coordinates": coordinates,
            "coordinate_status": coordinate_status,
            "center": center,
            "center_status": center_status,
            "orientation": orientation_value,
            "orientation_status": orientation_status,
            "orientation_method": orientation_method,
            "scaled_determinant_estimate": _finite(
                float(self._metrics.determinant_estimate[row])
            ),
            "scaled_determinant_error_bound": _finite(
                float(self._metrics.determinant_error_bound[row])
            ),
            "determinant_scale_exponent": int(
                self._metrics.determinant_scale_exponent[row]
            ),
            "signed_volume": _finite(float(self._metrics.signed_volume[row])),
            "volume": _finite(float(self._metrics.volume[row])),
            "volume_status": _decode_label(self._metrics.volume_status[row]),
            "volume_method": _decode_label(self._metrics.volume_method[row]),
            "volume_error_bound": _finite(
                float(self._metrics.volume_error_bound[row])
            ),
            "signed_volume_m3": _finite(float(self._metrics.signed_volume_m3[row])),
            "volume_m3": _finite(float(self._metrics.volume_m3[row])),
            "volume_m3_status": _decode_label(self._metrics.volume_m3_status[row]),
            "volume_m3_method": _decode_label(self._metrics.volume_m3_method[row]),
            "volume_m3_error_bound": _finite(
                float(self._metrics.volume_m3_error_bound[row])
            ),
            "kappa_j": kappa,
            "condition_status": condition_status,
            "exact_fallback": bool(self._metrics.fallback_used[row]),
        }

    def _header(self) -> dict[str, Any]:
        identity = self._mesh.file_identity
        return {
            "schema": _SCHEMA,
            "producer": {
                "package": "scgsim",
                "version": _distribution_version(),
                "numpy_version": np.__version__,
            },
            "source": {
                "path": str(self._mesh.source_path),
                "sha256": self._mesh.sha256,
                "bytes": self._mesh.bytes_read,
                "file_identity": {
                    "device": identity.device,
                    "inode": identity.inode,
                    "size": identity.size,
                    "mtime_ns": identity.mtime_ns,
                },
            },
            "reader": {
                "format": "gmsh-msh-2.2-ascii",
                "element_counts": dict(self._mesh.element_counts),
                "optional_sections": dict(self._mesh.optional_sections),
                "physical_names": [
                    {"dimension": dim, "tag": tag, "name": name}
                    for (dim, tag), name in sorted(self._mesh.physical_names.items())
                ],
            },
            "metrics": {
                "orientation": "explicit triple product with Shewchuk-style sign filter or exact dyadic fallback",
                "volume": "abs(det(B))/6; signed volume retains det(B) sign",
                "condition": "singular values of B @ inv(W), with numerical-resolution status kept separate",
                "floating_error_bounds": "absolute bounds include determinant-filter and conservative conversion-rounding terms",
            },
            "units": {
                "coordinates": "mesh_unit",
                "volume": "mesh_unit^3",
                "length_scale_m": self._length_scale_m,
                "volume_m3": "m^3" if self._length_scale_m is not None else None,
            },
            "summary": self.summary,
        }

    def _node_record(self, row: int) -> dict[str, Any]:
        values = self._mesh.coordinates[row]
        finite = bool(np.all(np.isfinite(values)))
        return {
            "node_id": int(self._mesh.node_ids[row]),
            "coordinates": [_finite(float(value)) for value in values],
            "coordinate_status": "available" if finite else "nonfinite_coordinate",
        }

    def to_dict(self) -> dict[str, Any]:
        """Materialize the complete schema; this intentionally scales with mesh size."""

        return {
            **self._header(),
            "nodes": [self._node_record(row) for row in range(len(self._mesh.node_ids))],
            "elements": [
                self._element_record(row) for row in range(len(self._mesh.element_ids))
            ],
        }

    def save_json(self, path: str | Path) -> Path:
        """Stream the complete report to a newly created JSON file."""

        target = Path(path).expanduser().resolve()
        if target == self._mesh.source_path:
            raise ValueError("mesh-quality JSON target must not be the source mesh")
        header = self._header()
        serialized_header = [
            (
                json.dumps(key),
                json.dumps(value, allow_nan=False, separators=(",", ":")),
            )
            for key, value in header.items()
        ]
        with target.open("x", encoding="utf-8") as handle:
            handle.write("{")
            for index, (key, value) in enumerate(serialized_header):
                if index:
                    handle.write(",")
                handle.write(key)
                handle.write(":")
                handle.write(value)
            handle.write(',"nodes":[')
            for row in range(len(self._mesh.node_ids)):
                if row:
                    handle.write(",")
                handle.write(
                    json.dumps(
                        self._node_record(row), allow_nan=False, separators=(",", ":")
                    )
                )
            handle.write('],"elements":[')
            for row in range(len(self._mesh.element_ids)):
                if row:
                    handle.write(",")
                handle.write(
                    json.dumps(
                        self._element_record(row),
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                )
            handle.write("]}\n")
        return target

    def _selection(self, metric: str, limit: int) -> "_Selection":
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        element_ids = self._mesh.element_ids
        if metric == "kappa_j":
            singular = [
                row
                for row in range(len(element_ids))
                if self._metrics.condition_status[row] == _StatusCode.EXACT_SINGULAR
            ]
            singular.sort(key=lambda row: int(element_ids[row]))
            finite = [
                row
                for row in range(len(element_ids))
                if self._metrics.condition_status[row] == _StatusCode.AVAILABLE
            ]
            finite.sort(
                key=lambda row: (-float(self._metrics.kappa_j[row]), int(element_ids[row]))
            )
            unresolved = [
                row
                for row in range(len(element_ids))
                if self._metrics.condition_status[row]
                not in {_StatusCode.AVAILABLE, _StatusCode.EXACT_SINGULAR}
            ]
            unresolved.sort(key=lambda row: int(element_ids[row]))
            ranked = singular + finite
        elif metric == "volume":
            finite = [
                row
                for row in range(len(element_ids))
                if self._metrics.volume_status[row]
                in {_StatusCode.AVAILABLE, _StatusCode.EXACT_ZERO}
            ]
            finite.sort(
                key=lambda row: (float(self._metrics.volume[row]), int(element_ids[row]))
            )
            finite_rows = set(finite)
            unresolved = [row for row in range(len(element_ids)) if row not in finite_rows]
            unresolved.sort(key=lambda row: int(element_ids[row]))
            ranked = finite
        else:
            raise ValueError("metric must be 'kappa_j' or 'volume'")
        return _Selection(
            ranked=tuple(ranked[:limit]),
            remainder=tuple(unresolved[:limit]),
            ranked_total=len(ranked),
            remainder_total=len(unresolved),
        )

    def show_summary(self) -> str:
        counts = self._summary["counts"]
        lines = [
            f"Mesh quality: {counts['checked_tetrahedra']} tetrahedra; "
            f"{counts['available_volume']} measurable volumes; "
            f"{counts['exact_zero']} exact-zero; "
            f"{counts['negative_orientation']} negative orientation."
        ]
        statistics = self._summary["statistics"]
        lines.append(_format_statistics("Volume", "mesh_unit^3", statistics["volume"]))
        if self._length_scale_m is not None:
            lines.append(_format_statistics("SI volume", "m^3", statistics["volume_m3"]))
        lines.append(_format_statistics("Kappa(J)", "dimensionless", statistics["kappa_j"]))
        warnings = self._summary["warnings"]
        if warnings:
            lines.extend(f"Warning: {item}" for item in warnings)
        text = "\n".join(lines)
        print(text)
        return text

    def show_worst_elements(self, metric: str = "kappa_j", limit: int = 10) -> str:
        selection = self._selection(metric, limit)
        lines = [
            f"Worst elements by {metric}",
            (
                f"ranked selected={len(selection.ranked)}/{selection.ranked_total}; "
                f"remainder selected={len(selection.remainder)}/{selection.remainder_total}"
            ),
            "group\telement_id\tvalue\tstatus",
        ]
        for group, rows in (
            ("ranked", selection.ranked),
            ("remainder", selection.remainder),
        ):
            for row in rows:
                record = self._element_record(row)
                value = record[metric]
                status = (
                    record["condition_status"]
                    if metric == "kappa_j"
                    else record["volume_status"]
                )
                lines.append(f"{group}\t{record['element_id']}\t{value}\t{status}")
        text = "\n".join(lines)
        print(text)
        return text

    def show_elements(
        self,
        element_ids: Sequence[int] | None = None,
        metric: str = "kappa_j",
        limit: int = 10,
    ) -> Any:
        """Return a detached Plotly figure without opening a browser or writing files."""

        from plotly import graph_objects as go

        if metric not in {"kappa_j", "volume"}:
            raise ValueError("metric must be 'kappa_j' or 'volume'")
        if element_ids is None:
            selection = self._selection(metric, limit)
            grouped_rows = (
                ("ranked", selection.ranked),
                ("remainder", selection.remainder),
            )
        else:
            row_by_id = {
                int(element_id): row
                for row, element_id in enumerate(self._mesh.element_ids)
            }
            requested = []
            for element_id in element_ids:
                if isinstance(element_id, (bool, np.bool_)) or not isinstance(
                    element_id, Integral
                ):
                    raise TypeError(
                        "element_ids must contain only Python or NumPy integers, excluding booleans"
                    )
                requested.append(int(element_id))
            unknown = [element_id for element_id in requested if element_id not in row_by_id]
            if unknown:
                raise KeyError(f"unknown tetrahedron element IDs: {unknown}")
            requested_rows = tuple(row_by_id[element_id] for element_id in requested)
            grouped_rows = (("requested", requested_rows),)
        figure = go.Figure()
        unavailable: list[tuple[int, str]] = []
        drawn_counts = {group: 0 for group, _ in grouped_rows}
        edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        for group, rows in grouped_rows:
            for row in rows:
                record = self._element_record(row)
                coordinates = record["coordinates"]
                if coordinates is None or record["coordinate_status"] != "available":
                    unavailable.append(
                        (record["element_id"], record["coordinate_status"])
                    )
                    continue
                x: list[float | None] = []
                y: list[float | None] = []
                z: list[float | None] = []
                for first, second in edges:
                    x.extend((coordinates[first][0], coordinates[second][0], None))
                    y.extend((coordinates[first][1], coordinates[second][1], None))
                    z.extend((coordinates[first][2], coordinates[second][2], None))
                hover = (
                    f"element={record['element_id']}<br>nodes={record['connectivity']}"
                    f"<br>coordinates={coordinates}<br>physical={record['physical_name']}"
                    f"<br>volume={record['volume']}"
                    f" ({record['volume_status']})<br>kappa={record['kappa_j']}"
                    f" ({record['condition_status']})<extra></extra>"
                )
                figure.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode="lines",
                        name=f"element {record['element_id']}",
                        hovertemplate=hover,
                    )
                )
                figure.add_trace(
                    go.Scatter3d(
                        x=[point[0] for point in coordinates],
                        y=[point[1] for point in coordinates],
                        z=[point[2] for point in coordinates],
                        mode="markers",
                        name=f"element {record['element_id']} vertices",
                        showlegend=False,
                        hovertemplate=hover,
                    )
                )
                drawn_counts[group] += 1
        selected_counts = {group: len(rows) for group, rows in grouped_rows}
        count_text = "; ".join(
            f"{group} selected={selected_counts[group]} drawn={drawn_counts[group]}"
            for group, _ in grouped_rows
        )
        unavailable_text = ", ".join(
            f"{element_id} ({reason})" for element_id, reason in unavailable
        )
        figure.update_layout(
            scene={"aspectmode": "data"},
            title=f"Tetrahedra by {metric} — {count_text}",
            annotations=(
                [
                    {
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.0,
                        "y": 1.0,
                        "showarrow": False,
                        "text": f"Undrawable: {unavailable_text}",
                        "align": "left",
                    }
                ]
                if unavailable
                else []
            ),
            meta={
                "selected_group_counts": selected_counts,
                "drawn_group_counts": drawn_counts,
                "unavailable_element_ids": [
                    element_id for element_id, _ in unavailable
                ],
                "unavailable_elements": [
                    {"element_id": element_id, "reason": reason}
                    for element_id, reason in unavailable
                ],
            },
        )
        return figure


@dataclass(frozen=True, slots=True)
class _Selection:
    ranked: tuple[int, ...]
    remainder: tuple[int, ...]
    ranked_total: int
    remainder_total: int


def _format_statistics(label: str, units: str, values: Mapping[str, Any]) -> str:
    samples = int(values["sample_count"])
    excluded = int(values["excluded_count"])
    if not samples:
        return f"{label} [{units}]: no samples; excluded={excluded}."
    return (
        f"{label} [{units}]: samples={samples}; excluded={excluded}; "
        f"min={values['min']}; median={values['median']}; "
        f"p95={values['p95']}; p99={values['p99']}; max={values['max']}."
    )
