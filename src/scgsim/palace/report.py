"""Trustworthiness and optional Plotly summaries for Palace runs.

The first report layer answers whether a run is usable: identity, AMR
convergence, and cost. Physics-result figures are a later layer.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .resolve import ParsedTable, ResolvedPalaceResult, _read_csv_table, _read_json

_ITERATION_DIR = re.compile(r"^iteration(\d+)$")
_FREQ_HEADER = "Re{f} (GHz)"
_CAP_HEADER_PREFIX = "C[i]["
_INDEX_COLUMNS = {"m", "i"}
_SKIP_EIG_COLUMNS = {"Error (Bkwd.)", "Error (Abs.)"}
_ERROR_INDICATOR_TRACES = ("Norm", "Maximum", "Mean")
_MUTED = "#8b949e"
_COLORWAY = (
    "#56B4E9",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#0072B2",
    "#D55E00",
    "#F0E442",
)
_PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

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

_DURATION_BARS = (
    "MeshPreprocessing",
    "Setup",
    "Construction",
    "OperatorConstruction",
    "Preconditioner",
    "LinearSolve",
    "EigenvalueSolve",
    "Div.-FreeProjection",
    "Solve",
    "Adaptation",
    "Rebalancing",
    "Postprocessing",
    "Paraview",
    "DiskIO",
)


@dataclass(frozen=True)
class NativeTabularSummary:
    """Compact native view of a resolved tabular result."""

    name: str
    headers: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AmrPassSnapshot:
    """One solver snapshot from an AMR pass or the parent results folder."""

    pass_index: int
    source: str
    path: Path
    frequencies_ghz: tuple[float, ...] | None
    eig_columns: dict[str, tuple[float, ...]] | None
    port_epr: dict[str, tuple[float, ...]] | None
    capacitance_matrix_f: tuple[tuple[float, ...], ...] | None
    error_indicators: dict[str, float] | None
    error_norm: float | None
    degrees_of_freedom: int | None
    mesh_elements: int | None
    elapsed_total_s: float | None
    peak_node_memory_mb: float | None


@dataclass(frozen=True)
class PassCostRecord:
    """One AMR pass worth of solver cost, for this run and later accumulation."""

    pass_index: int
    source: str
    degrees_of_freedom: int | None
    mesh_elements: int | None
    elapsed_cumulative_s: float | None
    elapsed_pass_s: float | None
    seconds_per_million_dof: float | None
    peak_node_memory_mb: float | None


@dataclass(frozen=True)
class PalaceTrustReport:
    """Run-identity, AMR evidence, and cost for one Palace package folder."""

    run_dir: Path
    problem: str
    route: str
    profile: str
    completeness: Literal["complete", "partial"]
    latest_source: str
    identity: dict[str, Any]
    passes: tuple[AmrPassSnapshot, ...]
    amr_tolerance: float | None
    amr_max_passes: int | None
    durations: dict[str, float]
    cost: dict[str, Any]

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display

        display(HTML(self._identity_html()))
        items = self._convergence_items()
        if len(self.passes) < 2:
            display(HTML(items[0]))
        else:
            display(HTML(self._convergence_heading_html()))
            for item in items:
                if isinstance(item, str):
                    display(HTML(item))
                else:
                    _show_figure(item)
        display(HTML(self._benchmark_cards_html()))
        benchmark = self._benchmark_time_figure()
        if isinstance(benchmark, str):
            display(HTML(benchmark))
        else:
            _show_figure(benchmark)
        display(HTML(self._benchmark_pass_table_html()))
        for figure in self._benchmark_pass_figures():
            _show_figure(figure)

    def _identity_html(self) -> str:
        cards = (
            ("Problem", self.problem),
            ("Route", self.route),
            ("Profile", self.profile),
            ("Completeness", self.completeness),
            ("Latest snapshot", self.latest_source),
            ("AMR passes recorded", str(len(self.passes))),
            ("AMR MaxIts", _fmt(self.amr_max_passes)),
            ("AMR Tol", _fmt(self.amr_tolerance)),
            ("MPI × OpenMP", self.identity.get("resources")),
            ("Receipt", self.identity.get("receipt")),
            ("Palace git", self.identity.get("git_tag")),
            ("Handoff id", self.identity.get("handoff_id_short")),
        )
        items = "".join(
            (
                "<div style='min-width:11rem;flex:1 1 11rem;border:1px solid var(--border,#d0d7de);"
                "border-radius:8px;padding:0.6rem 0.75rem;margin:0.25rem;'>"
                f"<div style='font-size:0.75rem;opacity:0.7'>{html.escape(str(label))}</div>"
                f"<div style='font-size:0.95rem;font-weight:600'>{html.escape(str(value))}</div>"
                "</div>"
            )
            for label, value in cards
            if value not in {None, ""}
        )
        return (
            "<section><h3>Run Identity</h3>"
            "<p style='opacity:0.75;margin-top:0'>Package folder and solver identity. "
            "This is not a physics result.</p>"
            f"<div style='display:flex;flex-wrap:wrap'>{items}</div></section>"
        )

    def _convergence_items(self) -> list[Any]:
        if len(self.passes) < 2:
            return [self._convergence_status_html()]
        xs = tuple(pass_.pass_index for pass_ in self.passes)
        items: list[Any] = []
        for header in _union_mapping_keys(self.passes, "eig_columns"):
            if header in _SKIP_EIG_COLUMNS:
                continue
            traces = _mode_traces(
                self.passes,
                lambda pass_, column=header: (
                    None if pass_.eig_columns is None else pass_.eig_columns.get(column)
                ),
            )
            if header == "Q" and not _traces_are_finite(traces):
                items.append(
                    "<p style='opacity:0.75'>Q is non-finite in this run; "
                    "the loss model may be disabled. The Q convergence plot is omitted.</p>"
                )
                continue
            figure = _line_figure(
                title=f"{header} vs AMR pass",
                xlabel="AMR pass",
                ylabel=header,
                xs=xs,
                traces=traces,
                yaxis_type=_yaxis_type(header),
            )
            if figure is not None:
                items.append(figure)
            if header == _FREQ_HEADER:
                delta = _line_figure(
                    title=f"|Δ {header}| vs AMR pass",
                    xlabel="AMR pass",
                    ylabel=f"|Δ {header}|",
                    xs=xs[1:],
                    traces=_delta_traces(traces),
                    yaxis_type="log",
                )
                if delta is not None:
                    items.append(delta)
        for header in _union_mapping_keys(self.passes, "port_epr"):
            traces = _mode_traces(
                self.passes,
                lambda pass_, column=header: (
                    None if pass_.port_epr is None else pass_.port_epr.get(column)
                ),
            )
            figure = _line_figure(
                title=f"Port EPR {header} vs AMR pass",
                xlabel="AMR pass",
                ylabel=header,
                xs=xs,
                traces=traces,
            )
            if figure is not None:
                items.append(figure)
        for label, traces in _capacitance_traces(self.passes):
            figure = _line_figure(
                title=f"{label} vs AMR pass",
                xlabel="AMR pass",
                ylabel="fF",
                xs=xs,
                traces=traces,
            )
            if figure is not None:
                items.append(figure)
            if label == "Maxwell C_ij":
                continue
            delta = _line_figure(
                title=f"|Δ {label}| vs AMR pass",
                xlabel="AMR pass",
                ylabel="|Δ| (fF)",
                xs=xs[1:],
                traces=_delta_traces(traces),
                yaxis_type="log",
            )
            if delta is not None:
                items.append(delta)
        traces = [
            (
                name,
                [
                    None
                    if pass_.error_indicators is None
                    else pass_.error_indicators.get(name)
                    for pass_ in self.passes
                ],
            )
            for name in _ERROR_INDICATOR_TRACES
        ]
        hline = None
        if self.amr_tolerance is not None:
            hline = (self.amr_tolerance, "configured AMR Tol")
        figure = _line_figure(
            title="AMR error indicator vs AMR pass",
            xlabel="AMR pass",
            ylabel="error indicator",
            xs=xs,
            traces=traces,
            yaxis_type="log",
            hline=hline,
        )
        if figure is not None:
            items.append(figure)
        return items

    def _convergence_heading_html(self) -> str:
        return (
            "<section><h3>Numerical Evidence</h3>"
            "<p style='opacity:0.75;margin-top:0'>Each readable AMR scalar is its own "
            "figure. AMR error Norm, Maximum, and Mean share one log plot because they "
            "are the same local indicator. Minimum is omitted. The dashed line is the "
            "configured Palace AMR Tol, not a newly invented acceptance Gate. "
            "Surface-Q and domain-E participation remain the later physics layer.</p>"
            "</section>"
        )

    def _convergence_status_html(self) -> str:
        latest = self.passes[-1] if self.passes else None
        quantity = _scalar_physics(latest, self.problem) if latest is not None else None
        if self.problem == "Eigenmode" and quantity is not None:
            quantity_line = f"Latest Re(f): {quantity:.6g} GHz"
        elif quantity is not None:
            quantity_line = f"Latest max |C_ii|: {quantity * 1e15:.6g} fF"
        else:
            quantity_line = "Latest physics quantity: unavailable"
        error_line = (
            f"Estimated error indicator (Norm): {_fmt(latest.error_norm)}"
            if latest is not None
            else "Estimated error indicator: unavailable"
        )
        body = (
            f"<p>{html.escape(quantity_line)}</p>"
            f"<p>{html.escape(error_line)}</p>"
            f"<p>Configured AMR Tol: {html.escape(_fmt(self.amr_tolerance))}</p>"
            f"<p>Configured MaxIts: {html.escape(_fmt(self.amr_max_passes))}</p>"
            f"<p>Latest snapshot: {html.escape(self.latest_source)}</p>"
        )
        title = (
            "No solver snapshots in results/palace"
            if not self.passes
            else "AMR stopped after the initial solve"
        )
        return (
            "<section><h3>Numerical Evidence</h3>"
            f"<div style='border:1px solid var(--border,#d0d7de);border-radius:8px;"
            f"padding:0.9rem 1rem'><strong>{html.escape(title)}</strong>{body}</div>"
            "<p style='opacity:0.75'>A trend plot is omitted because a single pass "
            "cannot show adaptation.</p></section>"
        )

    def _benchmark_cards_html(self) -> str:
        cards = (
            ("Total wall", _fmt_seconds(self.durations.get("Total"))),
            ("Peak node memory", _fmt_mb(self.cost.get("peak_node_memory_megabytes"))),
            ("Peak rank memory", _fmt_mb(self.cost.get("peak_memory_megabytes"))),
            ("DOFs", _fmt(self.cost.get("problem_degrees_of_freedom"))),
            ("Mesh elements", _fmt(self.cost.get("mesh_elements"))),
            ("MPI × OpenMP", self.identity.get("resources")),
        )
        items = "".join(
            (
                "<div style='min-width:11rem;flex:1 1 11rem;border:1px solid var(--border,#d0d7de);"
                "border-radius:8px;padding:0.6rem 0.75rem;margin:0.25rem;'>"
                f"<div style='font-size:0.75rem;opacity:0.7'>{html.escape(label)}</div>"
                f"<div style='font-size:0.95rem;font-weight:600'>{html.escape(str(value))}</div>"
                "</div>"
            )
            for label, value in cards
            if value not in {None, ""}
        )
        return (
            "<section><h3>Simulation Benchmark</h3>"
            "<p style='opacity:0.75;margin-top:0'>Cost and timing. This does not "
            "decide whether the physics is correct. Palace timers can overlap; they "
            "are not a partition of wall time.</p>"
            f"<div style='display:flex;flex-wrap:wrap'>{items}</div></section>"
        )

    def _benchmark_time_figure(self) -> Any:
        labels: list[str] = []
        values: list[float] = []
        for name in _DURATION_BARS:
            duration = self.durations.get(name)
            if duration is None or duration <= 0:
                continue
            labels.append(name)
            values.append(duration)
        if not labels:
            return (
                "<p style='opacity:0.75'>Palace elapsed-time timers are unavailable "
                "in this package.</p>"
            )
        go = _plotly()
        fig = go.Figure(
            data=[
                go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    marker={"color": _COLORWAY[0], "opacity": 0.88},
                    text=[_fmt_seconds(value) for value in values],
                    textposition="outside",
                    textfont={"color": _MUTED, "size": 12},
                    hovertemplate="%{x:.3g} s<extra>%{y}</extra>",
                )
            ]
        )
        _style_figure(
            fig,
            title="Elapsed time by Palace timer",
            height=max(260, 36 * len(labels) + 90),
            margin={"l": 150, "r": 72, "t": 56, "b": 40},
            hovermode="closest",
            showlegend=False,
        )
        fig.update_xaxes(title_text="seconds", rangemode="tozero")
        fig.update_yaxes(categoryorder="total ascending")
        return fig

    def pass_cost_records(self) -> tuple[PassCostRecord, ...]:
        records: list[PassCostRecord] = []
        previous_elapsed: float | None = None
        for pass_ in self.passes:
            elapsed_pass = None
            if pass_.elapsed_total_s is not None:
                if previous_elapsed is None:
                    elapsed_pass = pass_.elapsed_total_s
                else:
                    elapsed_pass = pass_.elapsed_total_s - previous_elapsed
                previous_elapsed = pass_.elapsed_total_s
            seconds_per_million = None
            if (
                elapsed_pass is not None
                and pass_.degrees_of_freedom is not None
                and pass_.degrees_of_freedom > 0
            ):
                seconds_per_million = elapsed_pass / (pass_.degrees_of_freedom / 1e6)
            records.append(
                PassCostRecord(
                    pass_index=pass_.pass_index,
                    source=pass_.source,
                    degrees_of_freedom=pass_.degrees_of_freedom,
                    mesh_elements=pass_.mesh_elements,
                    elapsed_cumulative_s=pass_.elapsed_total_s,
                    elapsed_pass_s=elapsed_pass,
                    seconds_per_million_dof=seconds_per_million,
                    peak_node_memory_mb=pass_.peak_node_memory_mb,
                )
            )
        return tuple(records)

    def _benchmark_pass_table_html(self) -> str:
        records = self.pass_cost_records()
        if not records:
            return (
                "<p style='opacity:0.75'>No AMR pass cost snapshots are available.</p>"
            )
        header = (
            "<tr>"
            + "".join(
                _html_cell("th", name)
                for name in (
                    "Pass",
                    "Snapshot",
                    "DOFs",
                    "Mesh elems",
                    "This pass",
                    "Cumulative",
                    "s / MDoF",
                    "Peak node mem",
                )
            )
            + "</tr>"
        )
        rows = []
        for record in records:
            rows.append(
                "<tr>"
                + _html_cell("td", str(record.pass_index))
                + _html_cell("td", record.source)
                + _html_cell("td", _fmt(record.degrees_of_freedom))
                + _html_cell("td", _fmt(record.mesh_elements))
                + _html_cell("td", _fmt_seconds(record.elapsed_pass_s))
                + _html_cell("td", _fmt_seconds(record.elapsed_cumulative_s))
                + _html_cell("td", _fmt(record.seconds_per_million_dof))
                + _html_cell("td", _fmt_mb(record.peak_node_memory_mb))
                + "</tr>"
            )
        return (
            "<section><h4>Cost by AMR pass</h4>"
            "<p style='opacity:0.75;margin-top:0'>Palace <code>ElapsedTime.Total</code> "
            "is cumulative; this-pass time is the difference between snapshots. "
            "These rows are the per-run cost records to accumulate later. "
            "They do not decide physics correctness.</p>"
            "<table style='border-collapse:collapse;font-size:0.9rem'>"
            f"<thead>{header}</thead><tbody>{''.join(rows)}</tbody></table></section>"
        )

    def _benchmark_pass_figures(self) -> list[Any]:
        records = self.pass_cost_records()
        if len(records) < 2:
            return []
        figures: list[Any] = []
        this_pass = [record.elapsed_pass_s for record in records]
        dofs = [record.degrees_of_freedom for record in records]
        labels = [record.source for record in records]
        dof_cost = _line_figure(
            title="This-pass wall time vs DOFs",
            xlabel="DOFs",
            ylabel="seconds this pass",
            xs=dofs,
            traces=[("this pass", this_pass)],
            point_labels=labels,
        )
        if dof_cost is not None:
            figures.append(dof_cost)
        return figures


def inspect_run_trustworthiness(run_dir: str | Path) -> PalaceTrustReport:
    """Build the trustworthiness report from a package folder.

    This path is for complete or incomplete returned runs. It does not replace
    ``resolve_palace_result`` identity verification for a finished package.
    """

    root = Path(run_dir).expanduser().resolve()
    return _build_trust_report(root)


def _show_run_trustworthiness(result: ResolvedPalaceResult) -> PalaceTrustReport:
    if not isinstance(result, ResolvedPalaceResult):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")
    report = _build_trust_report(result.run_dir, resolved=result)
    return report


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

    go = _plotly()
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


def _build_trust_report(
    root: Path, *, resolved: ResolvedPalaceResult | None = None
) -> PalaceTrustReport:
    handoff_path = root / "metadata" / "palace_handoff_metadata.json"
    if not handoff_path.is_file():
        raise FileNotFoundError(f"required artifact missing: {handoff_path}")
    handoff = _read_json(handoff_path)
    problem = str(handoff.get("problem") or (resolved.problem if resolved else ""))
    route = str(handoff.get("route") or (resolved.route if resolved else ""))
    profile = str(handoff.get("profile") or "")
    if problem not in {"Eigenmode", "Electrostatic"}:
        raise ValueError(f"unsupported problem {problem!r} for trustworthiness report.")

    config = _read_optional_json(root / "config.json")
    receipt = _read_optional_json(
        root / "metadata" / "palace_returned_run_receipt.json"
    )
    refinement = _refinement(config)
    passes = _collect_amr_passes(root, problem)
    latest = passes[-1] if passes else None
    parent_complete = _parent_has_physics(root / "results" / "palace", problem)
    completeness: Literal["complete", "partial"] = (
        "complete" if parent_complete else "partial"
    )
    latest_source = latest.source if latest is not None else "none"
    palace_payload = _latest_palace_json(passes, resolved)
    durations = _durations(palace_payload)
    cost = _cost_cards(palace_payload, resolved)
    mpi = cost.get("mpi_size")
    omp = cost.get("openmp_threads")
    resources = (
        f"{mpi} MPI × {omp} OMP" if mpi is not None and omp is not None else None
    )
    receipt_status = None
    if receipt is not None:
        receipt_status = (
            f"{receipt.get('status', 'unknown')} / exit {receipt.get('exit_code', '?')}"
        )
    elif resolved is not None:
        receipt_status = f"{resolved.returned_receipt.status} / exit {resolved.returned_receipt.exit_code}"
    handoff_id = str(handoff.get("handoff_id") or "")
    identity = {
        "handoff_id": handoff_id,
        "handoff_id_short": handoff_id[:12] if handoff_id else None,
        "resources": resources,
        "receipt": receipt_status,
        "git_tag": cost.get("git_tag"),
    }
    return PalaceTrustReport(
        run_dir=root,
        problem=problem,
        route=route,
        profile=profile,
        completeness=completeness,
        latest_source=latest_source,
        identity=identity,
        passes=tuple(passes),
        amr_tolerance=refinement.get("Tol"),
        amr_max_passes=_as_int(refinement.get("MaxIts")),
        durations=durations,
        cost=cost,
    )


def _collect_amr_passes(root: Path, problem: str) -> list[AmrPassSnapshot]:
    results = root / "results" / "palace"
    if not results.is_dir():
        return []
    iteration_dirs = []
    for child in results.iterdir():
        match = _ITERATION_DIR.match(child.name)
        if match and child.is_dir():
            iteration_dirs.append((int(match.group(1)), child))
    iteration_dirs.sort()
    snapshots = [
        _load_snapshot(
            pass_index=index - 1, source=path.name, path=path, problem=problem
        )
        for index, path in iteration_dirs
    ]
    if not _parent_has_physics(results, problem):
        return snapshots
    parent = _load_snapshot(
        pass_index=len(snapshots),
        source="final",
        path=results,
        problem=problem,
    )
    if snapshots and _same_physics(snapshots[-1], parent):
        snapshots[-1] = replace(parent, pass_index=snapshots[-1].pass_index)
        return snapshots
    return [*snapshots, parent]


def _load_snapshot(
    *, pass_index: int, source: str, path: Path, problem: str
) -> AmrPassSnapshot:
    eig_columns = (
        _read_numeric_table_columns(path / "eig.csv")
        if problem == "Eigenmode"
        else None
    )
    port_epr = (
        _read_numeric_table_columns(path / "port-EPR.csv")
        if problem == "Eigenmode"
        else None
    )
    capacitance = (
        _read_capacitance(path / "terminal-C.csv")
        if problem == "Electrostatic"
        else None
    )
    error_indicators = _read_error_indicators(path / "error-indicators.csv")
    frequencies = None if eig_columns is None else eig_columns.get(_FREQ_HEADER)
    error_norm = None if error_indicators is None else error_indicators.get("Norm")
    palace_payload = _read_optional_json(path / "palace.json")
    problem_block = palace_payload.get("Problem") if palace_payload else None
    elapsed = None
    if palace_payload is not None:
        durations = palace_payload.get("ElapsedTime", {})
        if isinstance(durations, dict):
            duration_map = durations.get("Durations")
            if isinstance(duration_map, dict) and isinstance(
                duration_map.get("Total"), (int, float)
            ):
                elapsed = float(duration_map["Total"])
    return AmrPassSnapshot(
        pass_index=pass_index,
        source=source,
        path=path,
        frequencies_ghz=frequencies,
        eig_columns=eig_columns,
        port_epr=port_epr,
        capacitance_matrix_f=capacitance,
        error_indicators=error_indicators,
        error_norm=error_norm,
        degrees_of_freedom=_as_int(
            problem_block.get("DegreesOfFreedom")
            if isinstance(problem_block, dict)
            else None
        ),
        mesh_elements=_as_int(
            problem_block.get("MeshElements")
            if isinstance(problem_block, dict)
            else None
        ),
        elapsed_total_s=elapsed,
        peak_node_memory_mb=_mapping_max(
            palace_payload.get("PeakNodeMemoryMegabytes") if palace_payload else None
        ),
    )


def _read_numeric_table_columns(path: Path) -> dict[str, tuple[float, ...]] | None:
    if not path.is_file():
        return None
    table = _read_csv_table(path)
    columns: dict[str, tuple[float, ...]] = {}
    for header in table.headers:
        if header in _INDEX_COLUMNS:
            continue
        values: list[float] = []
        usable = True
        for row in table.rows:
            value = row.get(header)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                usable = False
                break
            values.append(float(value))
        if usable and values:
            columns[header] = tuple(values)
    return columns or None


def _read_capacitance(path: Path) -> tuple[tuple[float, ...], ...] | None:
    if not path.is_file():
        return None
    table = _read_csv_table(path)
    matrix: list[tuple[float, ...]] = []
    value_headers = [
        header for header in table.headers if header.startswith(_CAP_HEADER_PREFIX)
    ]
    if not value_headers:
        return None
    for row in table.rows:
        values: list[float] = []
        for header in value_headers:
            value = row.get(header)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None
            values.append(float(value))
        matrix.append(tuple(values))
    return tuple(matrix) if matrix else None


def _read_error_indicators(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    table = _read_csv_table(path)
    if not table.rows:
        return None
    payload: dict[str, float] = {}
    for header in table.headers:
        value = table.rows[0].get(header)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            payload[header] = float(value)
    return payload or None


def _parent_has_physics(results: Path, problem: str) -> bool:
    if problem == "Eigenmode":
        return (results / "eig.csv").is_file()
    return (results / "terminal-C.csv").is_file()


def _same_physics(left: AmrPassSnapshot, right: AmrPassSnapshot) -> bool:
    if left.frequencies_ghz is not None and right.frequencies_ghz is not None:
        return left.frequencies_ghz == right.frequencies_ghz
    if left.capacitance_matrix_f is not None and right.capacitance_matrix_f is not None:
        return left.capacitance_matrix_f == right.capacitance_matrix_f
    return False


def _scalar_physics(pass_: AmrPassSnapshot | None, problem: str) -> float | None:
    if pass_ is None:
        return None
    if problem == "Eigenmode":
        if not pass_.frequencies_ghz:
            return None
        return pass_.frequencies_ghz[0]
    if pass_.capacitance_matrix_f is None:
        return None
    return _max_abs_diagonal(pass_.capacitance_matrix_f)


def _max_abs_diagonal(matrix: Sequence[Sequence[float]]) -> float:
    values = []
    for index, row in enumerate(matrix):
        if index < len(row):
            values.append(abs(row[index]))
    return max(values) if values else 0.0


def _refinement(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    model = config.get("Model")
    if not isinstance(model, dict):
        return {}
    refinement = model.get("Refinement")
    return refinement if isinstance(refinement, dict) else {}


def _latest_palace_json(
    passes: Sequence[AmrPassSnapshot], resolved: ResolvedPalaceResult | None
) -> dict[str, Any] | None:
    if resolved is not None:
        return resolved.provenance.palace_json
    for pass_ in reversed(passes):
        payload = _read_optional_json(pass_.path / "palace.json")
        if payload is not None:
            return payload
    return None


def _durations(palace_payload: dict[str, Any] | None) -> dict[str, float]:
    if not palace_payload:
        return {}
    elapsed = palace_payload.get("ElapsedTime")
    if not isinstance(elapsed, dict):
        return {}
    raw = elapsed.get("Durations")
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): float(value)
        for key, value in raw.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _cost_cards(
    palace_payload: dict[str, Any] | None, resolved: ResolvedPalaceResult | None
) -> dict[str, Any]:
    if resolved is not None:
        return {
            "problem_degrees_of_freedom": resolved.cost.problem_degrees_of_freedom,
            "mesh_elements": resolved.cost.mesh_elements,
            "mpi_size": resolved.cost.mpi_size,
            "openmp_threads": resolved.cost.openmp_threads,
            "peak_memory_megabytes": _mapping_max(resolved.cost.peak_memory_megabytes),
            "peak_node_memory_megabytes": _mapping_max(
                resolved.cost.peak_node_memory_megabytes
            ),
            "git_tag": resolved.cost.git_tag,
        }
    if not palace_payload:
        return {}
    problem_block = palace_payload.get("Problem")
    problem_block = problem_block if isinstance(problem_block, dict) else {}
    return {
        "problem_degrees_of_freedom": _as_int(problem_block.get("DegreesOfFreedom")),
        "mesh_elements": _as_int(problem_block.get("MeshElements")),
        "mpi_size": _as_int(problem_block.get("MPISize")),
        "openmp_threads": _as_int(problem_block.get("OpenMPThreads")),
        "peak_memory_megabytes": _mapping_max(
            palace_payload.get("PeakMemoryMegabytes")
        ),
        "peak_node_memory_megabytes": _mapping_max(
            palace_payload.get("PeakNodeMemoryMegabytes")
        ),
        "git_tag": palace_payload.get("GitTag")
        if isinstance(palace_payload.get("GitTag"), str)
        else None,
    }


def _mapping_max(payload: Any) -> float | None:
    if isinstance(payload, dict):
        value = payload.get("Max")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None
    if isinstance(payload, (int, float)) and not isinstance(payload, bool):
        return float(payload)
    return None


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


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _read_json(path)


def _union_mapping_keys(
    passes: Sequence[AmrPassSnapshot], attribute: str
) -> tuple[str, ...]:
    keys: list[str] = []
    seen: set[str] = set()
    for pass_ in passes:
        mapping = getattr(pass_, attribute)
        if not isinstance(mapping, dict):
            continue
        for key in mapping:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return tuple(keys)


def _mode_traces(
    passes: Sequence[AmrPassSnapshot],
    getter: Callable[[AmrPassSnapshot], Sequence[float] | None],
) -> list[tuple[str, list[float | None]]]:
    mode_count = max((len(getter(pass_) or ()) for pass_ in passes), default=0)
    traces: list[tuple[str, list[float | None]]] = []
    for mode in range(mode_count):
        values = []
        for pass_ in passes:
            column = getter(pass_)
            if column is None or mode >= len(column):
                values.append(None)
            else:
                values.append(column[mode])
        traces.append((f"Mode {mode + 1}", values))
    return traces


def _traces_are_finite(traces: Sequence[tuple[str, Sequence[float | None]]]) -> bool:
    numbers = [
        value
        for _name, ys in traces
        for value in ys
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return bool(numbers) and all(math.isfinite(value) for value in numbers)


def _delta_traces(
    traces: Sequence[tuple[str, Sequence[float | None]]],
) -> list[tuple[str, list[float | None]]]:
    deltas: list[tuple[str, list[float | None]]] = []
    for name, ys in traces:
        series: list[float | None] = []
        previous: float | None = None
        for value in ys:
            if previous is None or value is None:
                series.append(None)
            else:
                series.append(abs(value - previous))
            if value is not None:
                previous = value
        deltas.append((name, series[1:]))
    return deltas


def _capacitance_traces(
    passes: Sequence[AmrPassSnapshot],
) -> list[tuple[str, list[tuple[str, list[float | None]]]]]:
    rows = max(
        (
            len(pass_.capacitance_matrix_f)
            for pass_ in passes
            if pass_.capacitance_matrix_f
        ),
        default=0,
    )
    cols = max(
        (
            len(row)
            for pass_ in passes
            if pass_.capacitance_matrix_f
            for row in pass_.capacitance_matrix_f
        ),
        default=0,
    )
    if rows == 0 or cols == 0:
        return []
    all_traces: list[tuple[str, list[float | None]]] = []
    per_element: list[tuple[str, list[tuple[str, list[float | None]]]]] = []
    for row in range(rows):
        for col in range(cols):
            name = f"C[{row + 1}][{col + 1}]"
            values = []
            for pass_ in passes:
                matrix = pass_.capacitance_matrix_f
                if matrix is None or row >= len(matrix) or col >= len(matrix[row]):
                    values.append(None)
                else:
                    values.append(matrix[row][col] * 1e15)
            trace = (name, values)
            all_traces.append(trace)
            per_element.append((name, [trace]))
    if len(all_traces) > 9:
        return [("Maxwell C_ij", all_traces)]
    return per_element


def _yaxis_type(name: str) -> str:
    lowered = name.lower()
    if any(
        marker in lowered
        for marker in ("error", "q", "norm", "minimum", "maximum", "mean")
    ):
        return "log"
    return "linear"


def _line_figure(
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xs: Sequence[int | float | None],
    traces: Sequence[tuple[str, Sequence[float | None]]],
    point_labels: Sequence[str] | None = None,
    yaxis_type: str = "linear",
    hline: tuple[float, str] | None = None,
) -> Any | None:
    go = _plotly()
    fig = go.Figure()
    _style_figure(fig, title=title)
    plotted = False
    for name, ys in traces:
        plot_ys = _positive_or_none(ys) if yaxis_type == "log" else list(ys)
        pairs: list[tuple[int | float, float]] = []
        labels: list[str] = []
        for index, (x_value, y_value) in enumerate(zip(xs, plot_ys, strict=False)):
            if x_value is None or y_value is None:
                continue
            pairs.append((x_value, y_value))
            if point_labels is not None and index < len(point_labels):
                labels.append(point_labels[index])
        if not pairs:
            continue
        plot_x = [item[0] for item in pairs]
        plot_y = [item[1] for item in pairs]
        trace: dict[str, Any] = {
            "x": plot_x,
            "y": plot_y,
            "mode": "lines+markers+text" if labels else "lines+markers",
            "name": name,
            "line": {"width": 2.4},
            "marker": {"size": 8, "line": {"width": 0}},
            "hovertemplate": "%{y:.6g}<extra>%{fullData.name}</extra>",
        }
        if labels:
            trace["text"] = labels
            trace["textposition"] = "top center"
            trace["textfont"] = {"size": 11, "color": _MUTED}
        fig.add_trace(go.Scatter(**trace))
        plotted = True
    if not plotted:
        return None
    if hline is not None:
        fig.add_hline(
            y=hline[0],
            line_dash="dot",
            line_color=_MUTED,
            line_width=1.2,
            annotation_text=hline[1],
            annotation_font={"size": 11, "color": _MUTED},
            annotation_position="top right",
        )
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel, type=yaxis_type)
    _maybe_integer_xticks(fig, xs)
    return fig


def _style_figure(
    fig: Any,
    *,
    title: str,
    height: int = 380,
    margin: dict[str, int] | None = None,
    hovermode: str = "x unified",
    showlegend: bool = True,
) -> None:
    fig.update_layout(
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "ui-sans-serif, system-ui, sans-serif",
            "size": 13,
            "color": _MUTED,
        },
        title={
            "text": title,
            "font": {"size": 15, "color": _MUTED},
            "x": 0,
            "xanchor": "left",
        },
        colorway=list(_COLORWAY),
        hovermode=hovermode,
        hoverlabel={
            "bgcolor": "rgba(22, 27, 34, 0.92)",
            "bordercolor": "rgba(139, 148, 158, 0.35)",
            "font": {"color": "#e6edf3", "size": 12},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "xanchor": "left",
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "font": {"size": 12, "color": _MUTED},
        },
        height=height,
        margin=margin or {"l": 64, "r": 28, "t": 72, "b": 48},
        showlegend=showlegend,
    )
    axis = {
        "showgrid": True,
        "gridcolor": "rgba(139, 148, 158, 0.22)",
        "zeroline": False,
        "showline": False,
        "ticks": "",
        "automargin": True,
        "title": {"font": {"size": 12, "color": _MUTED}},
        "tickfont": {"size": 11, "color": _MUTED},
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)


def _maybe_integer_xticks(fig: Any, xs: Sequence[int | float | None]) -> None:
    numeric = [float(value) for value in xs if isinstance(value, (int, float))]
    if not numeric or not all(value.is_integer() for value in numeric):
        return
    span = max(numeric) - min(numeric)
    if span <= 24:
        fig.update_xaxes(dtick=1)


def _show_figure(fig: Any) -> None:
    fig.show(config=_PLOTLY_CONFIG)


def _plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "plotly is not available; install plotly to render Palace trustworthiness figures."
        ) from exc
    return go


def _html_cell(tag: str, text: str) -> str:
    return (
        f"<{tag} style='border:1px solid var(--border,#d0d7de);"
        f"padding:0.35rem 0.65rem;text-align:left'>{html.escape(text)}</{tag}>"
    )


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return int(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _fmt_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "unavailable"
    seconds = float(value)
    if seconds >= 120:
        return f"{seconds / 60:.2f} min"
    return f"{seconds:.3g} s"


def _positive_or_none(values: Sequence[float | None]) -> list[float | None]:
    return [None if value is None or value <= 0 else value for value in values]


def _fmt_mb(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "unavailable"
    mb = float(value)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GiB"
    return f"{mb:.3g} MiB"


__all__ = [
    "AmrPassSnapshot",
    "NativeTabularSummary",
    "PalaceTrustReport",
    "PassCostRecord",
    "inspect_run_trustworthiness",
]
