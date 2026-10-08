"""Palace-specific report projections, displays, and detached figures.

Trust answers whether a run is usable from identity and AMR evidence. Latest
physical interpretation and simulation cost remain explicit separate calls.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from ..results.report_data import (
    AmrPassSnapshot,
    PalaceFailureDiagnosis,
    PalaceResultSelection,
    PassCostRecord,
    SurfaceEprSeriesSnapshot,
    SurfaceMaskEprRecord,
    SurfaceMaskEprSeriesSnapshot,
)
from ..results.report_data import (
    _collect_amr_passes,
    _declares_surface_masks,
    _failure_diagnosis,
    _failure_payload,
    _as_int,
    _mapping_max,
    _read_optional_json,
    _read_surface_bindings,
    _read_surface_mask_bindings,
    _result_selection,
    _selection_payload,
    _surface_mask_snapshots,
    _surface_snapshots,
    _validate_inspection_receipt,
)
from ..results.resolve import (
    ResolvedPalaceResult,
    _read_json,
    _validate_index_entries,
    _validate_surface_mask_config,
)

_FREQ_HEADER = "Re{f} (GHz)"
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
    show_details: bool = False

    def mesh_summary(self) -> dict[str, Any]:
        """Read the same detached mesh authority used by resolved results."""
        import copy

        return copy.deepcopy(self.provenance["mesh_summary"])

    def with_theme(self, theme: ReportTheme) -> PalaceTrustReport:
        checked = _checked_theme(theme)
        return self if self.theme == checked else replace(self, theme=checked)

    def show_run_trustworthiness(
        self, *, theme: ReportTheme = "light", show_details: bool = False
    ) -> PalaceTrustReport:
        """Return this complete or partial trust report with the selected theme."""

        report = self.with_theme(theme)
        checked = _checked_show_details(show_details)
        return (
            report
            if report.show_details == checked
            else replace(report, show_details=checked)
        )

    def show_simulation_benchmark(
        self, *, show_details: bool = False
    ) -> SimulationBenchmarkReport:
        """Return the readable benchmark while retaining every machine field."""

        attempted = _read_optional_json(self.run_dir / "results/palace/palace.json")
        data = {
            "cost": dict(self.cost),
            "mesh_summary": self.mesh_summary(),
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
        return SimulationBenchmarkReport(
            trust=self,
            data=data,
            show_details=_checked_show_details(show_details),
        )

    def show_all_results(
        self,
        *,
        theme: ReportTheme = "light",
        ranking_limit: int | None = 20,
        show_details: bool = False,
    ) -> None:
        """Display trust, benchmark, and available physics in the shared order."""

        from IPython.display import display

        trust = self.show_run_trustworthiness(
            theme=theme,
            show_details=show_details,
        )
        display(trust)
        display(trust.show_simulation_benchmark(show_details=show_details))
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
        if len(self.passes) >= 2:
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
        if self.show_details:
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
                f"Stage: {_fmt(self.failure.execution_stage)}; preflight: "
                f"{_fmt(self.failure.preflight_exit_code)}; solver invoked: "
                f"{_fmt(self.failure.solver_invoked)}; exit: "
                f"{_fmt(self.failure.exit_code)}; solver: "
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
            items: list[Any] = [self._convergence_status_html()]
            q_traces = _mode_traces(
                self.passes,
                lambda pass_: (
                    None if pass_.eig_columns is None else pass_.eig_columns.get("Q")
                ),
            )
            q_notice = _native_positive_infinity_q_notice(q_traces)
            if q_notice is not None:
                items.append(q_notice)
            return items
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
                    _native_positive_infinity_q_notice(traces)
                    or "<p style='opacity:0.75'>A finite Q convergence plot is "
                    "unavailable because the Q trace contains non-finite or "
                    "unavailable data.</p>"
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
class SimulationBenchmarkReport:
    """Human-readable simulation cost with complete machine data in ``data``."""

    trust: PalaceTrustReport
    data: dict[str, Any]
    show_details: bool = False

    def _ipython_display_(self) -> None:
        from IPython.display import HTML, display

        display(HTML(self.trust._benchmark_cards_html()))
        display(
            HTML(
                _mesh_summary_html(
                    self.data["mesh_summary"], card_html=self.trust._card_html
                )
            )
        )
        timing = self.trust._benchmark_time_figure()
        if isinstance(timing, str):
            display(HTML(timing))
        else:
            _show_figure(timing)
        display(HTML(self.trust._benchmark_pass_table_html()))
        for figure in self.trust._benchmark_pass_figures():
            _show_figure(figure)
        if self.show_details:
            payload = html.escape(json.dumps(self.data, indent=2, sort_keys=True))
            display(
                HTML(
                    "<section><h4>Benchmark metadata</h4>"
                    "<p style='opacity:0.75;margin-top:0'>Complete machine-readable "
                    "cost, timing, resource, and run-state fields.</p>"
                    f"<pre style='white-space:pre-wrap'>{payload}</pre></section>"
                )
            )


@dataclass(frozen=True)
class PhysicsQuantitiesReport:
    """Latest readable physical quantities built from structured solver data."""

    trust: PalaceTrustReport
    ranking_limit: int | None = 20

    @property
    def snapshots(self) -> tuple[SurfaceEprSeriesSnapshot, ...]:
        return _surface_snapshots(self.trust.passes)

    @property
    def masked_snapshots(self) -> tuple[SurfaceMaskEprSeriesSnapshot, ...]:
        """All readable native mask passes; ``snapshots`` remains baseline-only."""
        return _surface_mask_snapshots(self.trust.passes)

    @property
    def masks_configured(self) -> bool:
        index_map = _read_optional_json(
            self.trust.run_dir / "metadata" / "palace_index_map.json"
        )
        return bool(_read_surface_mask_bindings(index_map))

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
            mask_latest = _latest_mask_snapshot(
                self.masked_snapshots,
                pass_index=latest.pass_index,
                source=latest.source,
                series_index=latest.series_index,
            )
            if mask_latest is not None:
                for interface_type in _SURFACE_TYPES:
                    figure = _surface_mask_participation_figure(
                        mask_latest,
                        latest,
                        interface_type,
                        self.trust.theme,
                    )
                    if figure is not None:
                        _show_figure(figure)
                display(HTML(_surface_mask_html(mask_latest)))
            elif self.masks_configured:
                display(
                    HTML(
                        "<p style='opacity:0.75'>Native masked Surface EPR is "
                        "unavailable for the selected source/pass; baseline "
                        "Surface-EPR remains separately readable.</p>"
                    )
                )

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
    result: ResolvedPalaceResult,
    *,
    theme: ReportTheme = "light",
    show_details: bool = False,
) -> PalaceTrustReport:
    if not isinstance(result, ResolvedPalaceResult):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")
    report = _build_trust_report(result.run_dir, resolved=result, theme=theme)
    return report.show_run_trustworthiness(theme=theme, show_details=show_details)


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
    show_details: bool = False,
) -> None:
    """Display trust, benchmark, and physics in the Human-defined order."""

    if not isinstance(result, ResolvedPalaceResult):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")

    from IPython.display import display

    trust = _show_run_trustworthiness(
        result,
        theme=theme,
        show_details=show_details,
    )
    display(trust)
    display(
        SimulationBenchmarkReport(
            trust=trust,
            data=_resolved_benchmark_data(result),
            show_details=_checked_show_details(show_details),
        )
    )
    display(
        trust.show_physics_quantities(
            theme=theme,
            ranking_limit=ranking_limit,
        )
    )


def _show_simulation_benchmark(
    result: ResolvedPalaceResult,
    *,
    show_details: bool = False,
) -> SimulationBenchmarkReport:
    """Return the readable benchmark and its complete machine data."""

    if not isinstance(result, ResolvedPalaceResult):
        raise TypeError("resolved result report requires ResolvedPalaceResult.")

    return SimulationBenchmarkReport(
        trust=_show_run_trustworthiness(result),
        data=_resolved_benchmark_data(result),
        show_details=_checked_show_details(show_details),
    )


def _resolved_benchmark_data(result: ResolvedPalaceResult) -> dict[str, Any]:
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
        "mesh_summary": result.mesh_summary(),
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
    index_map = (
        resolved.provenance.index_map
        if resolved is not None
        else _read_optional_json(root / "metadata" / "palace_index_map.json")
    )
    if _declares_surface_masks(config, index_map):
        if not isinstance(config, dict) or not isinstance(index_map, dict):
            raise ValueError(
                "declared surface masks require config and index evidence."
            )
        _validate_index_entries(index_map)
        _validate_surface_mask_config(config, index_map)
    surface_bindings = _read_surface_bindings(index_map)
    surface_mask_bindings = _read_surface_mask_bindings(index_map)
    collected = _collect_amr_passes(
        root,
        problem,
        surface_bindings,
        surface_mask_bindings,
        failed_attempt=receipt is not None and receipt.get("status") == "failed",
    )
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
    from ..mesh.summary import mesh_summary

    hashes = {entry["path"]: entry["sha256"] for entry in (handoff.get("hashes") or ())}
    solver = config.get("Solver", {}) if config is not None else {}
    fem_order = solver.get("Order")
    provenance["mesh_summary"] = mesh_summary(
        mesh_manifest if mesh_manifest is not None else {},
        problem=problem,
        fem_order=fem_order,
        mesh_sha256=hashes.get("palace.msh"),
        manifest_sha256=hashes.get("metadata/mesh_manifest.json"),
    )
    if fem_order is None and problem == "Eigenmode":
        provenance["mesh_summary"]["dof_estimate"]["reason"] = "FEM order not recorded"
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


def _latest_mask_snapshot(
    snapshots: Sequence[SurfaceMaskEprSeriesSnapshot],
    *,
    pass_index: int,
    source: str,
    series_index: int,
) -> SurfaceMaskEprSeriesSnapshot | None:
    return next(
        (
            snapshot
            for snapshot in reversed(snapshots)
            if snapshot.pass_index == pass_index
            and snapshot.source == source
            and snapshot.series_index == series_index
        ),
        None,
    )


def _surface_mask_participation_figure(
    snapshot: SurfaceMaskEprSeriesSnapshot,
    baseline: SurfaceEprSeriesSnapshot,
    interface_type: str,
    theme: ReportTheme,
) -> Any | None:
    records = [
        record for record in snapshot.records if record.interface_type == interface_type
    ]
    if not records:
        return None
    baseline_by_index = {record.index: record for record in baseline.records}
    go = _plotly()
    figure = go.Figure()
    grouped: dict[int, list[SurfaceMaskEprRecord]] = {}
    for record in records:
        grouped.setdefault(record.baseline_index, []).append(record)
    for color_index, (baseline_index, series) in enumerate(sorted(grouped.items())):
        ordered = sorted(series, key=lambda item: item.margin_index)
        first = ordered[0]
        label = f"#{baseline_index} {first.face_kind} · {first.owner_semantic_ids[0]}"
        figure.add_scatter(
            x=[record.margin_um for record in ordered],
            y=[record.participation for record in ordered],
            mode="lines+markers",
            name=label,
            line={"color": _COLORWAY[color_index % len(_COLORWAY)]},
            customdata=[
                [
                    record.index,
                    record.margin_index,
                    record.energy_j,
                    (
                        "unavailable / +inf"
                        if math.isinf(record.quality_factor)
                        else f"{record.quality_factor:.6g}"
                    ),
                    record.contribution_status,
                    record.retained_area_status,
                    record.surface_id,
                    ", ".join(record.owner_semantic_ids),
                ]
                for record in ordered
            ],
            hovertemplate=(
                "margin=%{x:.6g} µm<br>participation=%{y:.6g}"
                "<br>masked index=%{customdata[0]}<br>ordinal=%{customdata[1]}"
                "<br>energy=%{customdata[2]:.6g} J<br>native Q=%{customdata[3]}"
                "<br>contribution=%{customdata[4]}<br>retained area=%{customdata[5]}"
                "<br>surface=%{customdata[6]}<br>owners=%{customdata[7]}"
                "<extra>%{fullData.name}</extra>"
            ),
        )
        baseline_record = baseline_by_index.get(baseline_index)
        if baseline_record is not None:
            figure.add_scatter(
                x=[0.0],
                y=[baseline_record.participation],
                mode="markers",
                name=f"baseline {label}",
                marker={
                    "symbol": "diamond",
                    "size": 9,
                    "color": _COLORWAY[color_index % len(_COLORWAY)],
                },
                hovertemplate="baseline participation=%{y:.6g}<extra>%{fullData.name}</extra>",
            )
    _style_figure(
        figure,
        title=(
            f"{_series_label(snapshot)}: {interface_type} native Inset-mask participation "
            f"vs margin ({snapshot.source})"
        ),
        height=460,
        margin={"l": 78, "r": 220, "t": 72, "b": 64},
        hovermode="closest",
        showlegend=True,
        theme=theme,
    )
    figure.update_xaxes(title="Inset margin (µm)", rangemode="tozero")
    figure.update_yaxes(title="participation", rangemode="tozero")
    figure.update_layout(legend={"x": 1.02, "xanchor": "left", "y": 1})
    return figure


def _surface_mask_html(snapshot: SurfaceMaskEprSeriesSnapshot) -> str:
    zeroes = sum(
        record.contribution_status == "zero_contribution" for record in snapshot.records
    )
    return (
        "<p style='opacity:0.75'>Native mask Q and energy remain available in "
        "<code>masked_snapshots</code>. The native Inset output does not retain "
        "surface area support; zero participation or energy is reported as "
        "<code>zero_contribution</code>, not as proof of excluded geometry. "
        f"This snapshot contains {zeroes} zero-contribution mask record(s).</p>"
    )


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
            _source_provenance_label(
                record.source_provenance,
                interface_type=record.interface_type,
            ),
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
        height=max(420, 48 * shown + 150),
        margin={"l": 240, "r": 52, "t": 72, "b": 64},
        hovermode="closest",
        showlegend=False,
        theme=theme,
    )
    fig.update_layout(bargap=0.38)
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


def _series_label(
    snapshot: SurfaceEprSeriesSnapshot | SurfaceMaskEprSeriesSnapshot,
) -> str:
    noun = "Mode" if snapshot.series_kind == "mode" else "Excitation"
    return f"{noun} {snapshot.series_index}"


def _source_provenance_label(
    provenance: dict[str, Any],
    *,
    interface_type: str | None = None,
) -> str:
    record_ids = provenance.get("source_record_ids")
    label = "structured provenance retained"
    if (
        isinstance(record_ids, list)
        and record_ids
        and all(isinstance(record_id, str) for record_id in record_ids)
    ):
        suffix = f" (+{len(record_ids) - 1})" if len(record_ids) > 1 else ""
        label = f"{record_ids[0]}{suffix}"
    contributions = _surface_contribution_records(
        provenance,
        interface_type=interface_type,
    )
    if not contributions:
        return label
    sides = tuple(
        dict.fromkeys(
            str(record["side"])
            for record in contributions
            if isinstance(record.get("side"), str)
        )
    )
    suffix = f"; {len(contributions)} contribution(s)"
    if sides:
        suffix += f" ({'/'.join(sides)})"
    return f"{label}{suffix}"


def _surface_contribution_records(
    provenance: Mapping[str, Any],
    *,
    interface_type: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    ledger = provenance.get("surface_contribution_ledger")
    if isinstance(ledger, (list, tuple)):
        records.extend(record for record in ledger if isinstance(record, Mapping))
    else:
        sources = provenance.get("sources", ())
        if isinstance(sources, (list, tuple)):
            for source in sources:
                if isinstance(source, Mapping):
                    records.extend(_surface_contribution_records(source))
    deduplicated: dict[str, Mapping[str, Any]] = {}
    identities: dict[str, str] = {}
    for record in records:
        contribution_id = record.get("contribution_id")
        if not isinstance(contribution_id, str) or not contribution_id:
            continue
        identity = _serialized_report_contribution(record)
        previous = identities.setdefault(contribution_id, identity)
        if previous != identity:
            raise ValueError(
                f"conflicting repeated surface contribution {contribution_id!r}"
            )
        deduplicated.setdefault(contribution_id, record)
    return tuple(
        record
        for record in deduplicated.values()
        if interface_type is None or record.get("classification") == interface_type
    )


def _serialized_report_contribution(record: Mapping[str, Any]) -> str:
    """Canonicalize one complete serialized EvidenceResult transport record."""
    return json.dumps(
        dict(record),
        sort_keys=True,
        separators=(",", ":"),
    )


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


def _native_positive_infinity_q_notice(
    traces: Sequence[tuple[str, Sequence[float | None]]],
) -> str | None:
    values = [value for _name, ys in traces for value in ys]
    contains_positive_infinity = any(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == math.inf
        for value in values
    )
    finite_or_positive_infinity_only = all(
        value is None
        or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (math.isfinite(value) or value == math.inf)
        )
        for value in values
    )
    if not contains_positive_infinity or not finite_or_positive_infinity_only:
        return None
    return (
        "<p style='opacity:0.75'>Native-reported Q includes +inf; "
        "a finite-Q convergence view is unavailable.</p>"
    )


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
            "pad": {"t": 12, "b": 12, "l": 16, "r": 0},
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


def _checked_show_details(value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError("show_details must be a bool.")
    return value


def _theme_tokens(theme: str) -> _PlotTheme:
    return _THEMES[_checked_theme(theme)]


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


def _mesh_summary_html(
    summary: Mapping[str, Any], *, card_html: Callable[[str, Any], str]
) -> str:
    """Keep geometric discretization distinct from native solver measurements."""
    if summary["status"] == "not_recorded":
        return "<section><h4>Geometry mesh</h4><p>Mesh observations were not recorded in this historical manifest.</p></section>"
    meshing = summary["meshing"]
    estimate = summary["dof_estimate"]
    stats, effective = meshing["statistics"], meshing["effective"]
    cards = [
        (
            "Geometry / FEM order",
            f"{effective['geometry_order']} / {_fmt(summary['finite_element_order'])}",
        ),
        (
            "Native nodes / elements",
            f"{stats['node_count']} / {stats['element_count']}",
        ),
        ("3D algorithm", effective["algorithm_3d"]),
        (
            "Volume / surface thread options",
            f"{_fmt(effective['volume_threads'])} / {_fmt(effective['surface_threads_2d'])}",
        ),
        ("HighOrder optimization", meshing["high_order_optimization"]["status"]),
        ("ND topological DOF estimate", _fmt(estimate.get("value"))),
    ]
    items = "".join(card_html(label, value) for label, value in cards)
    rows = []
    for region in stats["physical_regions"]:
        chords = region.get("corner_chord_edge_lengths")
        values = [
            region["name"],
            region["dimension"],
            region["element_count"],
            ", ".join(
                f"{code}: {count}" for code, count in region["element_types"].items()
            ),
        ]
        values.extend(
            [
                chords["count"],
                *(_fmt(chords[key]) for key in ("minimum", "mean", "maximum")),
                chords["unit"],
            ]
            if chords is not None
            else ["not recorded"] * 5
        )
        rows.append(
            "<tr>" + "".join(_html_cell("td", str(value)) for value in values) + "</tr>"
        )
    header = "".join(
        _html_cell("th", label)
        for label in (
            "Region",
            "Dim",
            "Elements",
            "Gmsh type: count",
            "Edges",
            "Min",
            "Mean",
            "Max",
            "Unit",
        )
    )
    types = ", ".join(
        f"{record['name']}: {record['count']}" for record in stats["element_types"]
    )
    curve_context = "; ".join(
        f"{label}: {'recorded' if summary.get(key) is not None else 'not recorded'}"
        for label, key in (
            ("Source curves", "curve_source"),
            ("Native curve arrangement", "curved_arrangement"),
        )
    )
    return (
        "<section><h4>Geometry mesh</h4>"
        f'<div style="display:flex;flex-wrap:wrap">{items}</div>'
        "<p>The ND estimate uses the configured FEM order and tetrahedral primary-corner topology before boundary constraints; it is not actual solver DOF. Electrostatic H1 estimation is unavailable. Thread options are not measured thread utilization.</p>"
        "<details><summary>Physical regions and mesh discretization</summary>"
        f"<p>Gmsh {html.escape(str(meshing['gmsh_version']))}. {html.escape(types)}.</p>"
        "<p>Edge statistics count unique PRIMARY CORNER CHORDS within each region, not curved arc lengths. Point regions have no edges; missing historical statistics are not recorded.</p>"
        f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        f"<p>{html.escape(curve_context)}. Full controls and provenance remain available in mesh_summary() and report data.</p>"
        "</details></section>"
    )
