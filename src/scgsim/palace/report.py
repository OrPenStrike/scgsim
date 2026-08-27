"""Trust, physics-quantity, and benchmark surfaces for Palace runs.

Trust answers whether a run is usable from identity and AMR evidence. Latest
physical interpretation and simulation cost remain explicit separate calls.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .resolve import (
    ResolvedPalaceResult,
    _compute_hash_entry,
    _confined_path,
    _extract_hash_map,
    _read_csv_table,
    _read_json,
    _validate_index_entries,
    _validate_surface_table,
)

_ITERATION_DIR = re.compile(r"^iteration(\d+)$")
_FREQ_HEADER = "Re{f} (GHz)"
_CAP_HEADER_PREFIX = "C[i]["
_INDEX_COLUMNS = {"m", "i"}
_SKIP_EIG_COLUMNS = {"Error (Bkwd.)", "Error (Abs.)"}
_ERROR_INDICATOR_TRACES = ("Norm", "Maximum", "Mean")
_SURFACE_TYPES = ("MA", "MS", "SA")
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
_PLOT_HEIGHT = 460
_PLOT_FONT = 16
_PLOT_TITLE_FONT = 18
_PLOT_AXIS_FONT = 16
_PLOT_TICK_FONT = 14
_PLOT_LEGEND_FONT = 15
ReportTheme = Literal["light", "dark"]


@dataclass(frozen=True)
class _PlotTheme:
    paper_bgcolor: str
    plot_bgcolor: str
    font: str
    muted: str
    grid: str
    hover_bg: str
    hover_border: str
    hover_font: str


_THEMES = {
    "light": _PlotTheme(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f6f8fa",
        font="#1f2328",
        muted="#59636e",
        grid="rgba(31, 35, 40, 0.12)",
        hover_bg="#ffffff",
        hover_border="#d0d7de",
        hover_font="#1f2328",
    ),
    "dark": _PlotTheme(
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font="#e6edf3",
        muted="#8b949e",
        grid="rgba(139, 148, 158, 0.22)",
        hover_bg="#161b22",
        hover_border="#30363d",
        hover_font="#e6edf3",
    ),
}

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
class SurfaceEprRecord:
    """One solver-native surface result bound to structured SGB provenance."""

    index: int
    interface_type: str
    surface_id: str
    face_kind: str
    owner_semantic_ids: tuple[str, ...]
    net_id: str | None
    equipotential_id: str | None
    source_provenance: dict[str, Any]
    participation: float
    quality_factor: float
    loss_tangent: float | None


@dataclass(frozen=True)
class SurfaceEprSeriesSnapshot:
    """Surface-EPR values for one mode or excitation at one solver snapshot."""

    pass_index: int
    source: str
    series_index: int
    series_kind: Literal["mode", "excitation"]
    records: tuple[SurfaceEprRecord, ...]
    quality_factor_total: float | None
    t1_seconds: float | None
    loss_status: Literal["available", "unavailable_missing", "unavailable_nonfinite"]


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
    surface_epr: tuple[SurfaceEprSeriesSnapshot, ...] | None
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
class PalaceResultSelection:
    """The readable snapshot selected independently from the run outcome.

    ``selected_path`` is always relative to the run directory.
    """

    final_snapshot_status: Literal["readable", "missing", "unreadable"]
    selected_source: str | None
    selected_path: Path | None
    selected_pass_index: int | None
    reason: Literal[
        "final_snapshot",
        "latest_complete_iteration_after_failed_attempt",
        "no_complete_snapshot",
    ]
    integrity: Literal["receipt_bound", "observed_unsealed", "unavailable"]


@dataclass(frozen=True)
class PalaceFailureDiagnosis:
    """Evidence-based diagnosis that never changes the returned run status."""

    category: Literal[
        "out_of_memory",
        "signal_killed",
        "solver_error",
        "output_capture_error",
        "unknown",
    ]
    exit_code: int | None
    solver_exit_code: int | None
    tee_exit_code: int | None
    summary: str
    evidence: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _CollectedSnapshots:
    passes: tuple[AmrPassSnapshot, ...]
    final_snapshot_status: Literal["readable", "missing", "unreadable"]


@dataclass(frozen=True)
class PalaceTrustReport:
    """Run-identity, AMR evidence, and cost for one Palace package folder."""

    run_dir: Path
    problem: str
    route: str
    profile: str
    completeness: Literal["complete", "partial"]
    latest_source: str
    selection: PalaceResultSelection
    failure: PalaceFailureDiagnosis | None
    identity: dict[str, Any]
    passes: tuple[AmrPassSnapshot, ...]
    amr_tolerance: float | None
    amr_max_passes: int | None
    durations: dict[str, float]
    cost: dict[str, Any]
    provenance: dict[str, Any]
    theme: ReportTheme = "light"

    def with_theme(self, theme: ReportTheme) -> PalaceTrustReport:
        checked = _checked_theme(theme)
        return self if self.theme == checked else replace(self, theme=checked)

    def show_run_trustworthiness(
        self, *, theme: ReportTheme = "light"
    ) -> PalaceTrustReport:
        """Return this complete or partial trust report with the selected theme."""

        return self.with_theme(theme)

    def show_simulation_benchmark(self) -> dict[str, dict[str, Any]]:
        """Return every available cost, timing, resource, and run-state field."""

        attempted = _read_optional_json(self.run_dir / "results/palace/palace.json")
        return {
            "cost": dict(self.cost),
            "performance": {"counts": {}, "durations": dict(self.durations)},
            "attempted_run": {
                "cost": _cost_cards(attempted, None),
                "durations": _durations(attempted),
            },
            "selected_snapshot": {
                "source": self.selection.selected_source,
                "cost": dict(self.cost),
                "durations": dict(self.durations),
            },
            "resources": {
                "requested_resources": self.provenance.get("requested_resources", {}),
                "resolved_resources": self.provenance.get("resolved_resources", {}),
            },
            "performance_metadata": {
                "route": self.route,
                "problem": self.problem,
                "status": self.identity.get("receipt"),
                "completeness": self.completeness,
                "latest_source": self.latest_source,
                "selection": _selection_payload(self.selection),
                "failure": _failure_payload(self.failure),
            },
        }

    def show_all_results(
        self, *, theme: ReportTheme = "light", ranking_limit: int | None = 20
    ) -> None:
        """Display trust, benchmark, and available physics in the shared order."""

        from IPython.display import display

        trust = self.show_run_trustworthiness(theme=theme)
        display(trust)
        display(trust.show_simulation_benchmark())
        display(
            trust.show_physics_quantities(
                theme=theme,
                ranking_limit=ranking_limit,
            )
        )

    def _tokens(self) -> _PlotTheme:
        return _theme_tokens(self.theme)

    def _card_html(self, label: str, value: Any) -> str:
        return (
            "<div style='min-width:11rem;flex:1 1 11rem;border:1px solid var(--border,#d0d7de);"
            "border-radius:8px;padding:0.6rem 0.75rem;margin:0.25rem;'>"
            f"<div style='font-size:0.75rem;opacity:0.7'>{html.escape(str(label))}</div>"
            f"<div style='font-size:0.95rem;font-weight:600'>{html.escape(str(value))}</div>"
            "</div>"
        )

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
        for item in self._surface_convergence_items():
            if isinstance(item, str):
                display(HTML(item))
            else:
                _show_figure(item)
        display(HTML(self._provenance_html()))

    def show_physics_quantities(
        self,
        *,
        theme: ReportTheme = "light",
        ranking_limit: int | None = 20,
    ) -> PhysicsQuantitiesReport:
        """Return latest readable physics without redrawing convergence."""

        return PhysicsQuantitiesReport(
            self.with_theme(theme),
            _checked_ranking_limit(ranking_limit),
        )

    def _identity_html(self) -> str:
        cards = (
            ("Problem", self.problem),
            ("Route", self.route),
            ("Profile", self.profile),
            ("Completeness", self.completeness),
            ("Latest snapshot", self.latest_source),
            ("Final snapshot", self.selection.final_snapshot_status),
            ("Selection integrity", self.selection.integrity),
            ("AMR passes recorded", str(len(self.passes))),
            ("AMR MaxIts", _fmt(self.amr_max_passes)),
            ("AMR Tol", _fmt(self.amr_tolerance)),
            ("MPI × OpenMP", self.identity.get("resources")),
            ("Receipt", self.identity.get("receipt")),
            ("Palace git", self.identity.get("git_tag")),
            ("Handoff id", self.identity.get("handoff_id_short")),
        )
        items = "".join(
            self._card_html(str(label), value)
            for label, value in cards
            if value not in {None, ""}
        )
        return (
            "<section><h3>Run Identity</h3>"
            "<p style='opacity:0.75;margin-top:0'>Package folder and solver identity. "
            "This is not a physics result.</p>"
            f"<div style='display:flex;flex-wrap:wrap'>{items}</div>"
            f"{self._run_state_html()}</section>"
        )

    def _run_state_html(self) -> str:
        notices: list[str] = []
        if self.failure is not None:
            notices.append(
                "<div style='border-left:4px solid #cf222e;background:rgba(207,34,46,0.10);"
                "padding:0.65rem 0.8rem;margin:0.7rem 0'>"
                f"<strong>Run failed — {html.escape(self.failure.category.replace('_', ' '))}</strong><br>"
                f"{html.escape(self.failure.summary)}<br>"
                f"Exit: {_fmt(self.failure.exit_code)}; solver: "
                f"{_fmt(self.failure.solver_exit_code)}; output capture: "
                f"{_fmt(self.failure.tee_exit_code)}."
                "<details><summary>Failure evidence</summary><code>"
                f"{html.escape(json.dumps(self.failure.evidence, sort_keys=True))}"
                "</code></details></div>"
            )
        if self.selection.reason == "latest_complete_iteration_after_failed_attempt":
            source = self.selection.selected_source or "unavailable"
            path = self.selection.selected_path
            notices.append(
                "<div style='border-left:4px solid #bf8700;background:rgba(191,135,0,0.10);"
                "padding:0.65rem 0.8rem;margin:0.7rem 0'>"
                "<strong>Fallback result selected</strong><br>"
                f"Using {html.escape(source)} as partial evidence; the attempted final "
                f"snapshot is {html.escape(self.selection.final_snapshot_status)}. "
                f"Integrity: {html.escape(self.selection.integrity)}."
                + (
                    f"<br><code>{html.escape(str(path))}</code>"
                    if path is not None
                    else ""
                )
                + "</div>"
            )
        elif self.selection.reason == "no_complete_snapshot":
            notices.append(
                "<div style='border-left:4px solid #bf8700;background:rgba(191,135,0,0.10);"
                "padding:0.65rem 0.8rem;margin:0.7rem 0'>"
                "<strong>No complete result snapshot is available.</strong></div>"
            )
        return "".join(notices)

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
                theme=self.theme,
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
                    theme=self.theme,
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
                theme=self.theme,
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
                theme=self.theme,
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
                theme=self.theme,
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
            theme=self.theme,
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
            "Surface participation convergence follows below; latest ranking and "
            "loss interpretation remain a separate physics call.</p>"
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
            "<div style='border:1px solid var(--border,#d0d7de);border-radius:8px;"
            f"padding:0.9rem 1rem'><strong>{html.escape(title)}</strong>{body}</div>"
            "<p style='opacity:0.75'>A trend plot is omitted because a single pass "
            "cannot show adaptation.</p></section>"
        )

    def _surface_convergence_items(self) -> list[Any]:
        snapshots = _surface_snapshots(self.passes)
        if not snapshots:
            return [
                (
                    "<section><h3>Surface-EPR Numerical Convergence</h3>"
                    "<p style='opacity:0.75'>MA/MS/SA participation convergence is "
                    "unavailable because no readable structured surface-Q snapshot "
                    "was returned.</p></section>"
                )
            ]
        items: list[Any] = [
            (
                "<section><h3>Surface-EPR Numerical Convergence</h3>"
                "<p style='opacity:0.75;margin-top:0'>Each mode or excitation is kept "
                "separate. Percentages use the MA+MS+SA sum from the same source and "
                "AMR pass.</p></section>"
            )
        ]
        for series_index in sorted({item.series_index for item in snapshots}):
            series = tuple(
                item for item in snapshots if item.series_index == series_index
            )
            items.append(f"<h4>{html.escape(_series_label(series[-1]))}</h4>")
            total = _surface_total_figure(series, self.theme)
            if total is not None:
                items.append(total)
            items.append(_surface_percentage_figure(series, self.theme))
        return items

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
            self._card_html(label, value)
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
        tokens = self._tokens()
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
                    textfont={"color": tokens.muted, "size": _PLOT_TICK_FONT},
                    hovertemplate="%{x:.3g} s<extra>%{y}</extra>",
                )
            ]
        )
        _style_figure(
            fig,
            title="Elapsed time by Palace timer",
            height=max(320, 42 * len(labels) + 110),
            margin={"l": 196, "r": 88, "t": 56, "b": 52},
            hovermode="closest",
            showlegend=False,
            theme=self.theme,
        )
        fig.update_xaxes(title=_axis_title("seconds", tokens), rangemode="tozero")
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
            theme=self.theme,
        )
        if dof_cost is not None:
            figures.append(dof_cost)
        return figures

    def _provenance_html(self) -> str:
        payload = html.escape(json.dumps(self.provenance, indent=2, sort_keys=True))
        return (
            "<section><h3>Provenance</h3>"
            "<p style='opacity:0.75;margin-top:0'>Exact source, Palace, input, and "
            "returned-run identities recorded by the handoff package.</p>"
            f"<pre style='white-space:pre-wrap'>{payload}</pre></section>"
        )


@dataclass(frozen=True)
class PhysicsQuantitiesReport:
    """Latest readable physical quantities built from structured solver data."""

    trust: PalaceTrustReport
    ranking_limit: int | None = 20

    @property
    def snapshots(self) -> tuple[SurfaceEprSeriesSnapshot, ...]:
        return _surface_snapshots(self.trust.passes)

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display

        display(HTML(self._heading_html()))
        snapshots = self.snapshots
        if not snapshots:
            display(
                HTML(
                    "<p style='opacity:0.75'>Surface-EPR physics is unavailable: "
                    "no readable surface-Q snapshot could be bound to structured "
                    "index-map semantics.</p>"
                )
            )
            return
        for series_index in sorted({item.series_index for item in snapshots}):
            series = tuple(
                item for item in snapshots if item.series_index == series_index
            )
            latest = series[-1]
            label = _series_label(latest)
            display(HTML(f"<h4>{html.escape(label)}</h4>"))
            ranking = _surface_ranking_figure(
                latest, self.ranking_limit, self.trust.theme
            )
            if ranking is not None:
                _show_figure(ranking)
            display(HTML(_surface_loss_html(latest, self.trust.problem)))

    def _heading_html(self) -> str:
        state = (
            "complete returned run"
            if self.trust.completeness == "complete"
            else "partial / convergence not established"
        )
        limit = (
            "all surfaces" if self.ranking_limit is None else str(self.ranking_limit)
        )
        latest_source = self.trust.selection.selected_source or "unavailable"
        return (
            "<section><h3>Physics Quantities</h3>"
            "<p style='opacity:0.75;margin-top:0'>Latest-readable individual-surface "
            "participation and loss interpretation are shown only after run identity "
            "and numerical evidence. Modes and excitations remain separate. "
            f"Run state: {html.escape(state)}; latest readable snapshot: "
            f"{html.escape(latest_source)}; ranking limit: "
            f"{html.escape(limit)}; snapshot integrity: "
            f"{html.escape(self.trust.selection.integrity)}. Complete bound records "
            "remain available through "
            "<code>snapshots</code>.</p></section>"
        )


def inspect_run_trustworthiness(
    run_dir: str | Path, *, theme: ReportTheme = "light"
) -> PalaceTrustReport:
    """Build the trustworthiness report from a package folder.

    This path is for complete or incomplete returned runs. It does not replace
    ``resolve_palace_result`` identity verification for a finished package.
    ``theme`` is ``light`` (default) or ``dark`` and applies only to Plotly
    figures.
    """

    root = Path(run_dir).expanduser().resolve()
    return _build_trust_report(root, theme=theme)


def _show_run_trustworthiness(
    result: ResolvedPalaceResult, *, theme: ReportTheme = "light"
) -> PalaceTrustReport:
    if not isinstance(result, ResolvedPalaceResult):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")
    report = _build_trust_report(result.run_dir, resolved=result, theme=theme)
    return report


def _show_physics_quantities(
    result: ResolvedPalaceResult,
    *,
    theme: ReportTheme = "light",
    ranking_limit: int | None = 20,
) -> PhysicsQuantitiesReport:
    report = _show_run_trustworthiness(result, theme=theme)
    return report.show_physics_quantities(
        theme=theme,
        ranking_limit=ranking_limit,
    )


def _show_all_results(
    result: ResolvedPalaceResult,
    *,
    theme: ReportTheme = "light",
    ranking_limit: int | None = 20,
) -> None:
    """Display trust, benchmark, and physics in the Human-defined order."""

    if not isinstance(result, ResolvedPalaceResult):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")

    from IPython.display import display

    trust = _show_run_trustworthiness(result, theme=theme)
    display(trust)
    display(_show_simulation_benchmark(result))
    display(
        trust.show_physics_quantities(
            theme=theme,
            ranking_limit=ranking_limit,
        )
    )


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
    root: Path,
    *,
    resolved: ResolvedPalaceResult | None = None,
    theme: ReportTheme = "light",
) -> PalaceTrustReport:
    handoff_path = root / "metadata" / "palace_handoff_metadata.json"
    if not handoff_path.is_file():
        raise FileNotFoundError(f"required artifact missing: {handoff_path}")
    handoff = _read_json(handoff_path)
    mesh_manifest = _read_optional_json(root / "metadata" / "mesh_manifest.json")
    mesh_thin_film = (
        mesh_manifest.get("route_a_thin_film")
        if isinstance(mesh_manifest, dict)
        else None
    )
    if mesh_thin_film != handoff.get("route_a_thin_film"):
        raise ValueError(
            "mesh manifest and handoff Route-A thin-film identity mismatch."
        )
    problem = str(handoff.get("problem") or (resolved.problem if resolved else ""))
    route = str(handoff.get("route") or (resolved.route if resolved else ""))
    profile = str(handoff.get("profile") or "")
    if problem not in {"Eigenmode", "Electrostatic"}:
        raise ValueError(f"unsupported problem {problem!r} for trustworthiness report.")

    config = _read_optional_json(root / "config.json")
    receipt = _read_optional_json(
        root / "metadata" / "palace_returned_run_receipt.json"
    )
    receipt_paths = _validate_inspection_receipt(root, handoff, receipt)
    refinement = _refinement(config)
    surface_bindings = _read_surface_bindings(
        resolved.provenance.index_map
        if resolved is not None
        else _read_optional_json(root / "metadata" / "palace_index_map.json")
    )
    collected = _collect_amr_passes(root, problem, surface_bindings)
    passes = collected.passes
    receipt_status = receipt.get("status") if receipt is not None else None
    selection = _result_selection(
        root=root,
        problem=problem,
        collected=collected,
        receipt_paths=receipt_paths,
    )
    parent_complete = (
        receipt_status == "completed"
        and collected.final_snapshot_status == "readable"
        and selection.selected_source == "final"
    )
    completeness: Literal["complete", "partial"] = (
        "complete" if parent_complete else "partial"
    )
    latest_source = selection.selected_source or "none"
    failure = _failure_diagnosis(root, receipt, receipt_paths)
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
    provenance = {
        key: handoff.get(key)
        for key in (
            "source_revisions",
            "palace_identity",
            "hashes",
            "requested_resources",
            "resolved_resources",
            "route_a_thin_film",
        )
        if handoff.get(key) is not None
    }
    if receipt is not None:
        provenance["returned_receipt"] = receipt
    return PalaceTrustReport(
        run_dir=root,
        problem=problem,
        route=route,
        profile=profile,
        completeness=completeness,
        latest_source=latest_source,
        selection=selection,
        failure=failure,
        identity=identity,
        passes=tuple(passes),
        amr_tolerance=refinement.get("Tol"),
        amr_max_passes=_as_int(refinement.get("MaxIts")),
        durations=durations,
        cost=cost,
        provenance=provenance,
        theme=_checked_theme(theme),
    )


def _validate_inspection_receipt(
    root: Path,
    handoff: dict[str, Any],
    receipt: dict[str, Any] | None,
) -> frozenset[str]:
    if receipt is None:
        return frozenset()
    for field in ("handoff_id", "route", "problem"):
        value = receipt.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"returned receipt {field} must be a non-empty string.")
        if field in handoff and handoff.get(field) != value:
            raise ValueError(
                f"returned receipt {field} does not match handoff metadata."
            )
    if receipt.get("route_a_thin_film") != handoff.get("route_a_thin_film"):
        raise ValueError("returned receipt Route-A thin-film identity mismatch.")
    if not isinstance(receipt.get("status"), str) or not receipt["status"]:
        raise ValueError("returned receipt status must be a non-empty string.")
    for field in ("exit_code", "solver_exit_code", "tee_exit_code"):
        _receipt_exit_code(receipt, field)

    input_entries = receipt.get("input_hashes")
    handoff_entries = handoff.get("hashes")
    if not isinstance(input_entries, list) or not isinstance(handoff_entries, list):
        raise TypeError("returned receipt and handoff input hashes must be lists.")
    receipt_inputs = _extract_hash_map(input_entries)
    handoff_inputs = _extract_hash_map(handoff_entries)
    if receipt_inputs != handoff_inputs:
        raise ValueError("returned receipt input hashes do not match handoff metadata.")
    for relative, expected in receipt_inputs.items():
        observed = _compute_hash_entry(_confined_path(root, relative))
        if observed != expected:
            raise ValueError(f"returned receipt input hash mismatch for {relative}.")

    output_entries = receipt.get("output_files")
    if not isinstance(output_entries, list):
        raise TypeError("returned receipt output_files must be a list.")
    verified: set[str] = set()
    seen: set[str] = set()
    for index, entry in enumerate(output_entries):
        relative = _verify_receipt_file_record(root, entry, f"output_files[{index}]")
        if relative in seen:
            raise ValueError(f"returned receipt contains duplicate path {relative!r}.")
        seen.add(relative)
        if entry["present"] is True:
            verified.add(relative)

    log = receipt.get("log")
    if not isinstance(log, dict):
        raise TypeError("returned receipt log must be a mapping.")
    log_relative = _verify_receipt_file_record(root, log, "log")
    if log["present"] is True:
        verified.add(log_relative)
    return frozenset(verified)


def _verify_receipt_file_record(root: Path, entry: Any, label: str) -> str:
    if not isinstance(entry, dict):
        raise TypeError(f"returned receipt {label} must be a mapping.")
    relative = entry.get("path")
    present = entry.get("present")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"returned receipt {label} path must be non-empty.")
    if not isinstance(present, bool):
        raise TypeError(f"returned receipt {label} present must be bool.")
    observed = _compute_hash_entry(_confined_path(root, relative))
    if present:
        size = entry.get("bytes")
        sha256 = entry.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"returned receipt {label} bytes must be non-negative.")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"returned receipt {label} sha256 must have 64 digits.")
        if observed != (size, sha256):
            raise ValueError(f"returned receipt output hash mismatch for {relative}.")
    elif entry.get("bytes") is not None or entry.get("sha256") is not None:
        raise ValueError(f"returned receipt missing {label} must not carry a hash.")
    elif observed != (None, None):
        raise ValueError(f"returned receipt missing output now exists: {relative}.")
    return relative


def _receipt_exit_code(receipt: dict[str, Any], field: str) -> int:
    value = receipt.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"returned receipt {field} must be an integer.")
    return value


def _result_selection(
    *,
    root: Path,
    problem: str,
    collected: _CollectedSnapshots,
    receipt_paths: frozenset[str],
) -> PalaceResultSelection:
    if not collected.passes:
        return PalaceResultSelection(
            final_snapshot_status=collected.final_snapshot_status,
            selected_source=None,
            selected_path=None,
            selected_pass_index=None,
            reason="no_complete_snapshot",
            integrity="unavailable",
        )
    selected = collected.passes[-1]
    reason = (
        "final_snapshot"
        if selected.source == "final"
        else "latest_complete_iteration_after_failed_attempt"
    )
    artifact_paths = _snapshot_artifact_paths(selected.path, problem)
    relative_paths = {path.relative_to(root).as_posix() for path in artifact_paths}
    integrity: Literal["receipt_bound", "observed_unsealed"] = (
        "receipt_bound"
        if relative_paths and relative_paths.issubset(receipt_paths)
        else "observed_unsealed"
    )
    return PalaceResultSelection(
        final_snapshot_status=collected.final_snapshot_status,
        selected_source=selected.source,
        selected_path=selected.path.relative_to(root),
        selected_pass_index=selected.pass_index,
        reason=reason,
        integrity=integrity,
    )


def _snapshot_artifact_paths(path: Path, problem: str) -> tuple[Path, ...]:
    names = (
        (
            "eig.csv",
            "port-EPR.csv",
            "surface-Q.csv",
            "error-indicators.csv",
            "palace.json",
        )
        if problem == "Eigenmode"
        else ("terminal-C.csv", "surface-Q.csv", "error-indicators.csv", "palace.json")
    )
    return tuple(candidate for name in names if (candidate := path / name).is_file())


def _failure_diagnosis(
    root: Path,
    receipt: dict[str, Any] | None,
    receipt_paths: frozenset[str],
) -> PalaceFailureDiagnosis | None:
    if receipt is None or receipt.get("status") != "failed":
        return None
    exit_code = _receipt_exit_code(receipt, "exit_code")
    solver_exit = _receipt_exit_code(receipt, "solver_exit_code")
    tee_exit = _receipt_exit_code(receipt, "tee_exit_code")
    evidence: list[dict[str, Any]] = [
        {
            "source": "returned_receipt",
            "exit_code": exit_code,
            "solver_exit_code": solver_exit,
            "tee_exit_code": tee_exit,
        }
    ]
    oom_evidence = _slurm_oom_evidence(root, receipt, receipt_paths)
    if oom_evidence is not None:
        evidence.append(oom_evidence)
        category = "out_of_memory"
        summary = "Slurm reported an out-of-memory event."
    elif solver_exit == 137 or exit_code == 137:
        category = "signal_killed"
        summary = (
            "The solver was killed with exit 137; out of memory is possible but "
            "not confirmed by scheduler evidence."
        )
    elif solver_exit != 0:
        category = "solver_error"
        summary = f"The Palace solver exited with status {solver_exit}."
    elif tee_exit != 0:
        category = "output_capture_error"
        summary = f"Solver output capture exited with status {tee_exit}."
    else:
        category = "unknown"
        summary = "The returned receipt reports failure without a classified cause."
    return PalaceFailureDiagnosis(
        category=category,
        exit_code=exit_code,
        solver_exit_code=solver_exit,
        tee_exit_code=tee_exit,
        summary=summary,
        evidence=tuple(evidence),
    )


def _slurm_oom_evidence(
    root: Path,
    receipt: dict[str, Any],
    receipt_paths: frozenset[str],
) -> dict[str, Any] | None:
    log = receipt.get("log")
    if not isinstance(log, dict):
        return None
    relative = log.get("path")
    if not isinstance(relative, str) or relative not in receipt_paths:
        return None
    path = _confined_path(root, relative)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lowered = line.lower()
            if "slurm" not in lowered:
                continue
            if (
                "oom_kill event" in lowered
                or "oom-kill event" in lowered
                or "out_of_memory" in lowered
            ):
                return {
                    "source": "hash_verified_log",
                    "path": relative,
                    "marker": "slurm_out_of_memory",
                    "sha256": log.get("sha256"),
                }
    return None


def _selection_payload(selection: PalaceResultSelection) -> dict[str, Any]:
    return {
        "final_snapshot_status": selection.final_snapshot_status,
        "selected_source": selection.selected_source,
        "selected_path": (
            None if selection.selected_path is None else str(selection.selected_path)
        ),
        "selected_pass_index": selection.selected_pass_index,
        "reason": selection.reason,
        "integrity": selection.integrity,
    }


def _failure_payload(
    failure: PalaceFailureDiagnosis | None,
) -> dict[str, Any] | None:
    if failure is None:
        return None
    return {
        "category": failure.category,
        "exit_code": failure.exit_code,
        "solver_exit_code": failure.solver_exit_code,
        "tee_exit_code": failure.tee_exit_code,
        "summary": failure.summary,
        "evidence": failure.evidence,
    }


def _collect_amr_passes(
    root: Path,
    problem: str,
    surface_bindings: tuple[dict[str, Any], ...] | None,
) -> _CollectedSnapshots:
    results = root / "results" / "palace"
    if not results.is_dir():
        return _CollectedSnapshots((), "missing")
    iteration_dirs = []
    for child in results.iterdir():
        match = _ITERATION_DIR.match(child.name)
        if match and child.is_dir():
            iteration_dirs.append((int(match.group(1)), child))
    iteration_dirs.sort()
    snapshots: list[AmrPassSnapshot] = []
    for index, path in iteration_dirs:
        snapshot = _load_optional_snapshot(
            pass_index=index - 1,
            source=path.name,
            path=path,
            problem=problem,
            surface_bindings=surface_bindings,
        )
        if snapshot is not None:
            snapshots.append(snapshot)
    if not _parent_has_physics(results, problem):
        return _CollectedSnapshots(tuple(snapshots), "missing")
    parent = _load_optional_snapshot(
        pass_index=max((index for index, _path in iteration_dirs), default=0),
        source="final",
        path=results,
        problem=problem,
        surface_bindings=surface_bindings,
    )
    if parent is None:
        return _CollectedSnapshots(tuple(snapshots), "unreadable")
    if snapshots and _same_physics(snapshots[-1], parent):
        snapshots[-1] = replace(parent, pass_index=snapshots[-1].pass_index)
        return _CollectedSnapshots(tuple(snapshots), "readable")
    return _CollectedSnapshots((*snapshots, parent), "readable")


def _load_optional_snapshot(
    *,
    pass_index: int,
    source: str,
    path: Path,
    problem: str,
    surface_bindings: tuple[dict[str, Any], ...] | None,
) -> AmrPassSnapshot | None:
    try:
        return _load_snapshot(
            pass_index=pass_index,
            source=source,
            path=path,
            problem=problem,
            surface_bindings=surface_bindings,
        )
    except (OSError, TypeError, ValueError):
        return None


def _load_snapshot(
    *,
    pass_index: int,
    source: str,
    path: Path,
    problem: str,
    surface_bindings: tuple[dict[str, Any], ...] | None,
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
    expected_rows = (
        len(frequencies)
        if frequencies is not None
        else len(capacitance)
        if capacitance is not None
        else 0
    )
    surface_epr = _read_surface_epr(
        path / "surface-Q.csv",
        problem=problem,
        pass_index=pass_index,
        source=source,
        frequencies_ghz=frequencies,
        expected_rows=expected_rows,
        bindings=surface_bindings,
    )
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
        surface_epr=surface_epr,
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


def _read_surface_bindings(
    index_map: dict[str, Any] | None,
) -> tuple[dict[str, Any], ...] | None:
    if index_map is None:
        return None
    try:
        _validate_index_entries(index_map)
        entries = [
            entry
            for entry in index_map["entries"]
            if entry["section"] == "Boundaries.Postprocessing.Dielectric"
        ]
        entries.sort(key=lambda entry: entry["index"])
        bindings: list[dict[str, Any]] = []
        for entry in entries:
            metadata = entry["metadata"]
            interface_type = metadata.get("interface_type")
            surface_id = metadata.get("surface_id")
            face_kind = metadata.get("face_kind")
            owners = metadata.get("owner_semantic_ids")
            provenance = metadata.get("source_provenance")
            if interface_type not in _SURFACE_TYPES:
                raise ValueError("surface interface_type must be MA, MS, or SA.")
            if not isinstance(surface_id, str) or not surface_id:
                raise ValueError("surface_id must be a non-empty string.")
            if not isinstance(face_kind, str) or not face_kind:
                raise ValueError("face_kind must be a non-empty string.")
            if (
                not isinstance(owners, list)
                or not owners
                or not all(isinstance(owner, str) and owner for owner in owners)
            ):
                raise ValueError("owner_semantic_ids must be non-empty strings.")
            if not isinstance(provenance, dict):
                raise TypeError("source_provenance must be a mapping.")
            epr_spec = entry.get("epr_spec")
            loss_tangent = (
                epr_spec.get("loss_tangent") if isinstance(epr_spec, dict) else None
            )
            if loss_tangent is not None and (
                isinstance(loss_tangent, bool)
                or not isinstance(loss_tangent, (int, float))
                or not math.isfinite(float(loss_tangent))
                or loss_tangent < 0
            ):
                raise ValueError(
                    "surface loss_tangent must be finite and non-negative."
                )
            bindings.append(
                {
                    "index": entry["index"],
                    "interface_type": interface_type,
                    "surface_id": surface_id,
                    "face_kind": face_kind,
                    "owner_semantic_ids": tuple(owners),
                    "net_id": _optional_string(metadata.get("net_id")),
                    "equipotential_id": _optional_string(
                        metadata.get("equipotential_id")
                    ),
                    "source_provenance": provenance,
                    "loss_tangent": (
                        None if loss_tangent is None else float(loss_tangent)
                    ),
                }
            )
        return tuple(bindings) or None
    except (KeyError, TypeError, ValueError):
        return None


def _read_surface_epr(
    path: Path,
    *,
    problem: str,
    pass_index: int,
    source: str,
    frequencies_ghz: tuple[float, ...] | None,
    expected_rows: int,
    bindings: tuple[dict[str, Any], ...] | None,
) -> tuple[SurfaceEprSeriesSnapshot, ...] | None:
    if not path.is_file() or not bindings or expected_rows <= 0:
        return None
    try:
        table = _read_csv_table(path)
        _validate_surface_table(table, len(bindings), expected_rows)
        snapshots: list[SurfaceEprSeriesSnapshot] = []
        index_name = "m" if problem == "Eigenmode" else "i"
        for row in table.rows:
            series_index = int(row[index_name])
            records = tuple(
                SurfaceEprRecord(
                    **binding,
                    participation=float(row[f"p_surf[{binding['index']}]"]),
                    quality_factor=float(row[f"Q_surf[{binding['index']}]"]),
                )
                for binding in bindings
            )
            inverse_loss = _surface_inverse_loss(records)
            quality_factor = (
                1.0 / inverse_loss
                if inverse_loss is not None and inverse_loss > 0
                else None
            )
            frequency_hz = None
            if frequencies_ghz is not None and 0 < series_index <= len(frequencies_ghz):
                frequency_hz = frequencies_ghz[series_index - 1] * 1e9
            t1_seconds = (
                quality_factor / (2.0 * math.pi * frequency_hz)
                if quality_factor is not None
                and frequency_hz is not None
                and frequency_hz > 0
                else None
            )
            snapshots.append(
                SurfaceEprSeriesSnapshot(
                    pass_index=pass_index,
                    source=source,
                    series_index=series_index,
                    series_kind="mode" if problem == "Eigenmode" else "excitation",
                    records=records,
                    quality_factor_total=quality_factor,
                    t1_seconds=t1_seconds,
                    loss_status=(
                        "unavailable_missing"
                        if inverse_loss is None
                        else "unavailable_nonfinite"
                        if quality_factor is None
                        else "available"
                    ),
                )
            )
        return tuple(snapshots)
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _surface_inverse_loss(records: Sequence[SurfaceEprRecord]) -> float | None:
    if any(record.loss_tangent is None for record in records):
        return None
    inverse_loss = sum(
        record.participation * float(record.loss_tangent) for record in records
    )
    return inverse_loss if math.isfinite(inverse_loss) and inverse_loss >= 0 else None


def _surface_totals(snapshot: SurfaceEprSeriesSnapshot) -> dict[str, float]:
    return {
        interface_type: sum(
            record.participation
            for record in snapshot.records
            if record.interface_type == interface_type
        )
        for interface_type in _SURFACE_TYPES
    }


def _surface_total_figure(
    snapshots: Sequence[SurfaceEprSeriesSnapshot], theme: ReportTheme
) -> Any | None:
    return _line_figure(
        title=f"{_series_label(snapshots[-1])}: MA/MS/SA participation vs AMR pass",
        xlabel="AMR pass",
        ylabel="total participation",
        xs=[snapshot.pass_index for snapshot in snapshots],
        traces=[
            (
                interface_type,
                [_surface_totals(snapshot)[interface_type] for snapshot in snapshots],
            )
            for interface_type in _SURFACE_TYPES
        ],
        theme=theme,
    )


def _surface_snapshots(
    passes: Sequence[AmrPassSnapshot],
) -> tuple[SurfaceEprSeriesSnapshot, ...]:
    return tuple(snapshot for pass_ in passes for snapshot in (pass_.surface_epr or ()))


def _surface_percentage_figure(
    snapshots: Sequence[SurfaceEprSeriesSnapshot], theme: ReportTheme
) -> Any | str:
    totals = [_surface_totals(snapshot) for snapshot in snapshots]
    denominators = [sum(values.values()) for values in totals]
    figure = _line_figure(
        title=f"{_series_label(snapshots[-1])}: normalized MA/MS/SA participation",
        xlabel="AMR pass",
        ylabel="share of MA+MS+SA (%)",
        xs=[snapshot.pass_index for snapshot in snapshots],
        traces=[
            (
                interface_type,
                [
                    None
                    if denominator <= 0
                    else 100.0 * values[interface_type] / denominator
                    for values, denominator in zip(totals, denominators, strict=True)
                ],
            )
            for interface_type in _SURFACE_TYPES
        ],
        theme=theme,
    )
    if figure is None:
        return (
            "<p style='opacity:0.75'>Normalized MA/MS/SA percentages are unavailable "
            "because their same-snapshot denominator is zero.</p>"
        )
    figure.update_yaxes(range=[0, 100])
    return figure


def _surface_ranking_figure(
    snapshot: SurfaceEprSeriesSnapshot,
    ranking_limit: int | None,
    theme: ReportTheme,
) -> Any | None:
    records = sorted(
        snapshot.records, key=lambda record: (-record.participation, record.index)
    )
    visible = records if ranking_limit is None else records[:ranking_limit]
    if not visible:
        return None
    tokens = _theme_tokens(theme)
    go = _plotly()
    labels = [
        f"#{record.index} {record.interface_type} · {record.face_kind}"
        f"<br>{record.owner_semantic_ids[0]}"
        for record in visible
    ]
    custom = [
        [
            record.surface_id,
            ", ".join(record.owner_semantic_ids),
            record.face_kind,
            record.net_id or "unassigned",
            record.equipotential_id or "unassigned",
            _source_provenance_label(record.source_provenance),
        ]
        for record in visible
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=[record.participation for record in visible],
                y=labels,
                orientation="h",
                marker={
                    "color": [
                        _COLORWAY[_SURFACE_TYPES.index(record.interface_type)]
                        for record in visible
                    ]
                },
                customdata=custom,
                hovertemplate=(
                    "%{x:.6g}<br>surface=%{customdata[0]}"
                    "<br>owners=%{customdata[1]}<br>face=%{customdata[2]}"
                    "<br>net=%{customdata[3]}<br>equipotential=%{customdata[4]}"
                    "<br>source=%{customdata[5]}<extra>%{y}</extra>"
                ),
            )
        ]
    )
    count = len(snapshot.records)
    shown = len(visible)
    _style_figure(
        fig,
        title=(
            f"{_series_label(snapshot)}: latest surface participation ranking "
            f"({snapshot.source}; {shown} of {count})"
        ),
        height=max(360, 29 * shown + 130),
        margin={"l": 240, "r": 52, "t": 72, "b": 64},
        hovermode="closest",
        showlegend=False,
        theme=theme,
    )
    fig.update_xaxes(title=_axis_title("participation", tokens), rangemode="tozero")
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def _surface_loss_html(snapshot: SurfaceEprSeriesSnapshot, problem: str) -> str:
    if snapshot.loss_status == "available":
        q_value = _fmt(snapshot.quality_factor_total)
        detail = "Q_total uses 1 / Σ(p_i tanδ_i); individual Q values are never summed."
    elif snapshot.loss_status == "unavailable_missing":
        q_value = "unavailable"
        detail = "At least one structured surface loss tangent is unavailable."
    else:
        q_value = "unavailable / non-finite"
        detail = (
            "The configured surface-loss sum is zero, so native +inf Q is not "
            "converted to zero. Participation remains available."
        )
    cards = [
        ("Latest source", snapshot.source),
        ("Surface-loss Q_total", q_value),
    ]
    if problem == "Eigenmode":
        cards.append(
            (
                "Surface-loss T1",
                _fmt_seconds(snapshot.t1_seconds)
                if snapshot.t1_seconds is not None
                else "unavailable / non-finite",
            )
        )
    items = "".join(
        "<div style='min-width:11rem;flex:1 1 11rem;border:1px solid "
        "var(--border,#d0d7de);border-radius:8px;padding:0.6rem 0.75rem;margin:0.25rem'>"
        f"<div style='font-size:0.75rem;opacity:0.7'>{html.escape(label)}</div>"
        f"<div style='font-size:0.95rem;font-weight:600'>{html.escape(value)}</div></div>"
        for label, value in cards
    )
    return (
        f"<div style='display:flex;flex-wrap:wrap'>{items}</div>"
        f"<p style='opacity:0.75'>{html.escape(detail)}</p>"
    )


def _series_label(snapshot: SurfaceEprSeriesSnapshot) -> str:
    noun = "Mode" if snapshot.series_kind == "mode" else "Excitation"
    return f"{noun} {snapshot.series_index}"


def _source_provenance_label(provenance: dict[str, Any]) -> str:
    record_ids = provenance.get("source_record_ids")
    if (
        isinstance(record_ids, list)
        and record_ids
        and all(isinstance(record_id, str) for record_id in record_ids)
    ):
        suffix = f" (+{len(record_ids) - 1})" if len(record_ids) > 1 else ""
        return f"{record_ids[0]}{suffix}"
    return "structured provenance retained"


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
    theme: ReportTheme = "light",
) -> Any | None:
    tokens = _theme_tokens(theme)
    go = _plotly()
    fig = go.Figure()
    _style_figure(fig, title=title, theme=theme)
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
            "line": {"width": 3},
            "marker": {"size": 10, "line": {"width": 0}},
            "hovertemplate": "%{y:.6g}<extra>%{fullData.name}</extra>",
        }
        if labels:
            trace["text"] = labels
            trace["textposition"] = "top center"
            trace["textfont"] = {"size": _PLOT_TICK_FONT, "color": tokens.muted}
        fig.add_trace(go.Scatter(**trace))
        plotted = True
    if not plotted:
        return None
    if hline is not None:
        fig.add_hline(
            y=hline[0],
            line_dash="dot",
            line_color=tokens.muted,
            line_width=1.5,
            annotation_text=hline[1],
            annotation_font={"size": _PLOT_TICK_FONT, "color": tokens.muted},
            annotation_position="top right",
        )
    fig.update_xaxes(title=_axis_title(xlabel, tokens))
    fig.update_yaxes(title=_axis_title(ylabel, tokens), type=yaxis_type)
    _maybe_integer_xticks(fig, xs)
    return fig


def _style_figure(
    fig: Any,
    *,
    title: str,
    height: int = _PLOT_HEIGHT,
    margin: dict[str, int] | None = None,
    hovermode: str = "x unified",
    showlegend: bool = True,
    theme: ReportTheme = "light",
) -> None:
    tokens = _theme_tokens(theme)
    if margin is None:
        margin = (
            {"l": 84, "r": 168, "t": 72, "b": 64}
            if showlegend
            else {"l": 72, "r": 36, "t": 72, "b": 64}
        )
    fig.update_layout(
        template="none",
        paper_bgcolor=tokens.paper_bgcolor,
        plot_bgcolor=tokens.plot_bgcolor,
        font={
            "family": "ui-sans-serif, system-ui, sans-serif",
            "size": _PLOT_FONT,
            "color": tokens.font,
        },
        title={
            "text": title,
            "font": {"size": _PLOT_TITLE_FONT, "color": tokens.font},
            "x": 0,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "pad": {"t": 0, "b": 10, "l": 0},
        },
        colorway=list(_COLORWAY),
        hovermode=hovermode,
        hoverlabel={
            "bgcolor": tokens.hover_bg,
            "bordercolor": tokens.hover_border,
            "font": {"color": tokens.hover_font, "size": _PLOT_TICK_FONT},
        },
        legend={
            "orientation": "v",
            "xref": "paper",
            "yref": "paper",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "font": {"size": _PLOT_LEGEND_FONT, "color": tokens.font},
            "tracegroupgap": 6,
            "itemsizing": "constant",
            "itemwidth": 36,
        },
        height=height,
        margin=margin,
        showlegend=showlegend,
    )
    axis = {
        "showgrid": True,
        "gridcolor": tokens.grid,
        "zeroline": False,
        "showline": False,
        "ticks": "",
        "automargin": False,
        "tickfont": {"size": _PLOT_TICK_FONT, "color": tokens.muted},
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)


def _axis_title(text: str, tokens: _PlotTheme) -> dict[str, Any]:
    return {
        "text": text,
        "font": {"size": _PLOT_AXIS_FONT, "color": tokens.font},
        "standoff": 8,
    }


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


def _checked_theme(theme: str) -> ReportTheme:
    if theme == "light" or theme == "dark":
        return theme
    raise ValueError("theme must be 'light' or 'dark'.")


def _checked_ranking_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("ranking_limit must be a positive integer or None.")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("optional semantic identity must be a non-empty string.")
    return value


def _theme_tokens(theme: str) -> _PlotTheme:
    return _THEMES[_checked_theme(theme)]


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
    "PalaceFailureDiagnosis",
    "PalaceResultSelection",
    "PalaceTrustReport",
    "PassCostRecord",
    "PhysicsQuantitiesReport",
    "inspect_run_trustworthiness",
]
