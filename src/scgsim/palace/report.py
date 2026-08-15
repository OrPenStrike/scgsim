"""Native and optional Plotly summaries for resolved Palace results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .resolve import ParsedTable, ResolvedPalaceResult


@dataclass(frozen=True)
class NativeTabularSummary:
    """Compact native view of a resolved tabular result."""

    name: str
    headers: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


_ELECTROSTATIC_RESULT_TABLES = (
    "terminal-C",
    "terminal-Cm",
    "terminal-Cinv",
    "terminal-V",
    "domain-E",
    "surface-Q",
)
_EIGENMODE_RESULT_TABLES = (
    "eig",
    "port-EPR",
    "port-I",
    "port-V",
    "domain-E",
    "surface-Q",
)


def _show_all_results(
    result: ResolvedPalaceResult, *, render_plotly: bool = False
) -> dict[str, Any]:
    """Return selected parsed result families as native summaries."""

    if not isinstance(result, ResolvedPalaceResult):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")

    selected = _native_result_families(result.problem, result.tables)
    summaries = {
        name: NativeTabularSummary(
            name=table.name, headers=table.headers, rows=table.rows
        )
        for name, table in selected.items()
    }

    if not render_plotly:
        return summaries

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "plotly is not available; install plotly or call render_plotly=False"
        ) from exc

    rendered: dict[str, Any] = {}
    for name, table in selected.items():
        rendered[name] = _table_to_plotly(go, table)
    return rendered


def _show_simulation_benchmark(
    result: ResolvedPalaceResult,
) -> dict[str, dict[str, Any]]:
    """Return native cost, performance, resources, and receipt metadata."""

    if not isinstance(result, ResolvedPalaceResult):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")

    return {
        "cost": {
            "problem_degrees_of_freedom": result.cost.problem_degrees_of_freedom,
            "mesh_elements": result.cost.mesh_elements,
            "mpi_size": result.cost.mpi_size,
            "openmp_threads": result.cost.openmp_threads,
            "peak_memory_megabytes": result.cost.peak_memory_megabytes,
            "peak_node_memory_megabytes": result.cost.peak_node_memory_megabytes,
            "linear_solver": result.cost.linear_solver,
            "git_tag": result.cost.git_tag,
        },
        "performance": {
            "counts": result.performance.counts,
            "durations": result.performance.durations,
        },
        "resources": {
            "requested_resources": result.provenance.resource_record.get(
                "requested_resources", {}
            ),
            "resolved_resources": result.provenance.resource_record.get(
                "resolved_resources", {}
            ),
        },
        "performance_metadata": {
            "route": result.route,
            "problem": result.problem,
            "status": result.status,
            "has_returned_outputs": result.has_returned_outputs,
            "receipt_status": result.returned_receipt.status,
            "receipt_exit_code": result.returned_receipt.exit_code,
        },
        "error_indicators": {
            "rows": len(result.tables["error-indicators"].rows),
            "headers": result.tables["error-indicators"].headers,
        },
    }


def _native_result_families(
    problem: str, tables: dict[str, ParsedTable]
) -> dict[str, ParsedTable]:
    if problem == "Electrostatic":
        return {
            name: tables[name]
            for name in _ELECTROSTATIC_RESULT_TABLES
            if name in tables
        }
    if problem == "Eigenmode":
        return {
            name: tables[name] for name in _EIGENMODE_RESULT_TABLES if name in tables
        }
    return {name: table for name, table in tables.items() if name != "error-indicators"}


def _table_to_plotly(go_module: Any, table: ParsedTable):
    return go_module.Figure(
        data=[
            go_module.Table(
                header={"values": list(table.headers)},
                cells={
                    "values": [
                        [row.get(column) for row in table.rows]
                        for column in table.headers
                    ]
                },
            )
        ],
        layout={"title_text": table.name},
    )


__all__ = ["NativeTabularSummary"]
