"""Trustworthiness and optional Plotly summaries for Palace runs.

The first report layer answers whether a run is usable: identity, AMR
convergence, and cost. Physics-result figures are a later layer.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .resolve import ParsedTable, ResolvedPalaceResult, _read_csv_table, _read_json

_ITERATION_DIR = re.compile(r"^iteration(\d+)$")
_FREQ_HEADER = "Re{f} (GHz)"
_CAP_HEADER_PREFIX = "C[i]["

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
    capacitance_matrix_f: tuple[tuple[float, ...], ...] | None
    error_norm: float | None
    degrees_of_freedom: int | None
    mesh_elements: int | None
    elapsed_total_s: float | None


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
        convergence = self._convergence_figure()
        if isinstance(convergence, str):
            display(HTML(convergence))
        else:
            display(HTML(self._convergence_heading_html()))
            display(convergence)
        display(HTML(self._benchmark_cards_html()))
        benchmark = self._benchmark_time_figure()
        if isinstance(benchmark, str):
            display(HTML(benchmark))
        else:
            display(benchmark)

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

    def _convergence_figure(self) -> Any:
        if len(self.passes) < 2:
            return self._convergence_status_html()
        go = _plotly()
        from plotly.subplots import make_subplots

        xs = [pass_.pass_index for pass_ in self.passes]
        physics_traces, physics_title, physics_ylabel = self._physics_series()
        error_norms = [pass_.error_norm for pass_ in self.passes]
        deltas = self._physics_deltas()
        dofs = [pass_.degrees_of_freedom for pass_ in self.passes]
        elements = [pass_.mesh_elements for pass_ in self.passes]

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                physics_title,
                "Estimated error / pass-to-pass delta",
                "Numerical cost",
            ),
            specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]],
        )
        for name, ys in physics_traces:
            fig.add_trace(
                go.Scatter(x=xs, y=ys, mode="lines+markers", name=name),
                row=1,
                col=1,
            )
        fig.update_yaxes(title_text=physics_ylabel, row=1, col=1)

        if any(value is not None for value in error_norms):
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=_positive_or_none(error_norms),
                    mode="lines+markers",
                    name="error indicator (Norm)",
                ),
                row=2,
                col=1,
            )
        if any(value is not None for value in deltas):
            fig.add_trace(
                go.Scatter(
                    x=xs[1:],
                    y=_positive_or_none(deltas),
                    mode="lines+markers",
                    name=self._delta_trace_name(),
                ),
                row=2,
                col=1,
                secondary_y=True,
            )
        if self.amr_tolerance is not None:
            fig.add_hline(
                y=self.amr_tolerance,
                line_dash="dash",
                row=2,
                col=1,
                annotation_text="configured AMR Tol",
            )
        fig.update_yaxes(title_text="error indicator", type="log", row=2, col=1)
        fig.update_yaxes(
            title_text=self._delta_trace_name(),
            type="log",
            row=2,
            col=1,
            secondary_y=True,
        )

        fig.add_trace(
            go.Scatter(x=xs, y=dofs, mode="lines+markers", name="DOFs"),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=elements,
                mode="lines+markers",
                name="mesh elements",
            ),
            row=3,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(title_text="DOFs", row=3, col=1)
        fig.update_yaxes(title_text="mesh elements", row=3, col=1, secondary_y=True)
        fig.update_xaxes(title_text="AMR pass", row=3, col=1)
        fig.update_layout(
            template="plotly_white",
            height=820,
            legend={"orientation": "h", "y": 1.08},
            margin={"l": 70, "r": 50, "t": 60, "b": 40},
        )
        return fig

    def _physics_series(self) -> tuple[list[tuple[str, list[float | None]]], str, str]:
        if self.problem == "Eigenmode":
            mode_count = max(
                (len(pass_.frequencies_ghz) for pass_ in self.passes if pass_.frequencies_ghz),
                default=0,
            )
            traces = []
            for mode in range(mode_count):
                traces.append(
                    (
                        f"Mode {mode + 1}",
                        [
                            None
                            if pass_.frequencies_ghz is None or mode >= len(pass_.frequencies_ghz)
                            else pass_.frequencies_ghz[mode]
                            for pass_ in self.passes
                        ],
                    )
                )
            return traces, "Eigenfrequency vs AMR pass", "Re(f) (GHz)"
        traces = [
            (
                "max |C_ii|",
                [
                    None
                    if pass_.capacitance_matrix_f is None
                    else _max_abs_diagonal(pass_.capacitance_matrix_f) * 1e15
                    for pass_ in self.passes
                ],
            )
        ]
        return traces, "Capacitance vs AMR pass", "max |C_ii| (fF)"

    def _delta_trace_name(self) -> str:
        if self.problem == "Eigenmode":
            return "|Δ Re(f)| (GHz)"
        return "|Δ max |C_ii|| (fF)"

    def _physics_deltas(self) -> list[float | None]:
        scale = 1.0 if self.problem == "Eigenmode" else 1e15
        values: list[float | None] = []
        previous: float | None = None
        for pass_ in self.passes:
            current = _scalar_physics(pass_, self.problem)
            if previous is None or current is None:
                values.append(None)
            else:
                values.append(abs(current - previous) * scale)
            previous = current if current is not None else previous
        return values[1:]

    def _convergence_heading_html(self) -> str:
        return (
            "<section><h3>Numerical Evidence</h3>"
            "<p style='opacity:0.75;margin-top:0'>Physics quantity, estimated error, "
            "and cost versus AMR pass. The dashed line is the configured Palace AMR Tol, "
            "not a newly invented acceptance Gate.</p></section>"
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
                    text=[_fmt_seconds(value) for value in values],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            template="plotly_white",
            title="Elapsed time by Palace timer",
            xaxis_title="seconds",
            yaxis={"categoryorder": "total ascending"},
            height=max(240, 48 * len(labels) + 80),
            margin={"l": 140, "r": 80, "t": 50, "b": 40},
        )
        fig.update_xaxes(rangemode="tozero")
        return fig


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
        name: NativeTabularSummary(name=table.name, headers=table.headers, rows=table.rows)
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
            "requested_resources": result.provenance.resource_record.get("requested_resources", {}),
            "resolved_resources": result.provenance.resource_record.get("resolved_resources", {}),
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
    receipt = _read_optional_json(root / "metadata" / "palace_returned_run_receipt.json")
    refinement = _refinement(config)
    passes = _collect_amr_passes(root, problem)
    latest = passes[-1] if passes else None
    parent_complete = _parent_has_physics(root / "results" / "palace", problem)
    completeness: Literal["complete", "partial"] = "complete" if parent_complete else "partial"
    latest_source = latest.source if latest is not None else "none"
    palace_payload = _latest_palace_json(passes, resolved)
    durations = _durations(palace_payload)
    cost = _cost_cards(palace_payload, resolved)
    mpi = cost.get("mpi_size")
    omp = cost.get("openmp_threads")
    resources = f"{mpi} MPI × {omp} OMP" if mpi is not None and omp is not None else None
    receipt_status = None
    if receipt is not None:
        receipt_status = (
            f"{receipt.get('status', 'unknown')} / exit {receipt.get('exit_code', '?')}"
        )
    elif resolved is not None:
        receipt_status = (
            f"{resolved.returned_receipt.status} / exit {resolved.returned_receipt.exit_code}"
        )
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
        _load_snapshot(pass_index=index - 1, source=path.name, path=path, problem=problem)
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
        previous = snapshots[-1]
        snapshots[-1] = AmrPassSnapshot(
            pass_index=previous.pass_index,
            source="final",
            path=parent.path,
            frequencies_ghz=parent.frequencies_ghz,
            capacitance_matrix_f=parent.capacitance_matrix_f,
            error_norm=parent.error_norm,
            degrees_of_freedom=parent.degrees_of_freedom,
            mesh_elements=parent.mesh_elements,
            elapsed_total_s=parent.elapsed_total_s,
        )
        return snapshots
    return [*snapshots, parent]


def _load_snapshot(*, pass_index: int, source: str, path: Path, problem: str) -> AmrPassSnapshot:
    frequencies = _read_frequencies(path / "eig.csv") if problem == "Eigenmode" else None
    capacitance = _read_capacitance(path / "terminal-C.csv") if problem == "Electrostatic" else None
    error_norm = _read_error_norm(path / "error-indicators.csv")
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
        capacitance_matrix_f=capacitance,
        error_norm=error_norm,
        degrees_of_freedom=_as_int(
            problem_block.get("DegreesOfFreedom") if isinstance(problem_block, dict) else None
        ),
        mesh_elements=_as_int(
            problem_block.get("MeshElements") if isinstance(problem_block, dict) else None
        ),
        elapsed_total_s=elapsed,
    )


def _read_frequencies(path: Path) -> tuple[float, ...] | None:
    if not path.is_file():
        return None
    table = _read_csv_table(path)
    values = []
    for row in table.rows:
        value = row.get(_FREQ_HEADER)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        values.append(float(value))
    return tuple(values) if values else None


def _read_capacitance(path: Path) -> tuple[tuple[float, ...], ...] | None:
    if not path.is_file():
        return None
    table = _read_csv_table(path)
    matrix: list[tuple[float, ...]] = []
    value_headers = [header for header in table.headers if header.startswith(_CAP_HEADER_PREFIX)]
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


def _read_error_norm(path: Path) -> float | None:
    if not path.is_file():
        return None
    table = _read_csv_table(path)
    if not table.rows:
        return None
    value = table.rows[0].get("Norm")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value)


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
            "peak_node_memory_megabytes": _mapping_max(resolved.cost.peak_node_memory_megabytes),
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
        "peak_memory_megabytes": _mapping_max(palace_payload.get("PeakMemoryMegabytes")),
        "peak_node_memory_megabytes": _mapping_max(palace_payload.get("PeakNodeMemoryMegabytes")),
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


def _native_result_families(problem: str, tables: dict[str, ParsedTable]) -> dict[str, ParsedTable]:
    if problem == "Electrostatic":
        return {name: tables[name] for name in _ELECTROSTATIC_RESULT_TABLES if name in tables}
    if problem == "Eigenmode":
        return {name: tables[name] for name in _EIGENMODE_RESULT_TABLES if name in tables}
    return {name: table for name, table in tables.items() if name != "error-indicators"}


def _table_to_plotly(go_module: Any, table: ParsedTable):
    return go_module.Figure(
        data=[
            go_module.Table(
                header={"values": list(table.headers)},
                cells={
                    "values": [[row.get(column) for row in table.rows] for column in table.headers]
                },
            )
        ],
        layout={"title_text": table.name},
    )


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _read_json(path)


def _plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "plotly is not available; install plotly to render Palace trustworthiness figures."
        ) from exc
    return go


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
    "inspect_run_trustworthiness",
]
