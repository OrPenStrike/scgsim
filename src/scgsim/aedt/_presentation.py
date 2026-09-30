"""Offline AEDT result projections and notebook rendering.

Native execution and artifact verification stay with their existing owners.
"""

from __future__ import annotations

import html
import json
import math
from typing import Any

from ._epr_models import EprResult


def _history_view(
    result: EprResult,
    *,
    mode: int | None,
    native_pass: int | None,
    show_convergence: bool,
) -> dict[str, Any]:
    requested_modes_raw = result.provenance.get(
        "requested_modes", sorted({int(row["mode"]) for row in result.rows})
    )
    known_modes = {
        int(item) for item in requested_modes_raw if type(item) is int and item > 0
    }
    if mode is not None and mode not in known_modes:
        raise ValueError(f"mode {mode} was not requested by this EPR result")
    requested_modes = [
        int(item)
        for item in requested_modes_raw
        if type(item) is int and item > 0 and (mode is None or item == mode)
    ]
    target_pass = native_pass
    if result.result_kind == "adaptive_history" and target_pass is None:
        observed_last = result.provenance.get("solver_last_completed_pass")
        if type(observed_last) is not int or observed_last <= 0:
            raise ValueError("adaptive EPR result lacks solver_last_completed_pass")
        target_pass = observed_last
    selected = [
        row for row in result.rows
        if (mode is None or row["mode"] == mode)
        and (result.result_kind != "adaptive_history" or row["native_pass"] == target_pass)
    ]
    selected_by_mode = {int(row["mode"]): row for row in selected}
    unavailable: list[str] = []
    for requested_mode in requested_modes:
        if requested_mode not in selected_by_mode:
            unavailable.append(
                f"mode {requested_mode}: requested pass {target_pass} is missing"
                if target_pass is not None
                else f"mode {requested_mode}: result row is missing"
            )
    bars: list[dict[str, Any]] = []
    for row in selected:
        row_mode = row["mode"]
        pass_label = "" if "native_pass" not in row else f", pass {row['native_pass']:g}"
        if row.get("status") == "partial" or "normalization_energy_j" not in row:
            unavailable.append(
                f"mode {row_mode}{pass_label}: "
                f"{row.get('combination_error', 'incomplete native evidence')}"
            )
            continue
        labels: list[str] = []
        values: list[float] = []
        for surface in row.get("surface_contributions", ()):
            labels.append(
                f"surface:{surface.get('group_id', surface.get('contribution_id'))}"
                f"@{surface['margin_um']:g}um"
            )
            values.append(float(surface["participation"]))
        for junction in row.get("junctions", ()):
            labels.extend([
                f"junction-L:{junction['junction_id']}",
                f"junction-C:{junction['junction_id']}",
            ])
            values.extend([
                float(junction["inductive_participation"]),
                float(junction["capacitive_participation"]),
            ])
        bars.append({"name": f"mode {row_mode}{pass_label}", "labels": labels, "values": values})
    convergence_view = show_convergence and result.result_kind == "adaptive_history"
    lines: list[dict[str, Any]] = []
    if convergence_view:
        last_pass = result.provenance.get("solver_last_completed_pass")
        if type(last_pass) is not int or last_pass <= 0:
            raise ValueError("adaptive EPR result lacks solver_last_completed_pass")
        pass_axis = list(range(1, last_pass + 1))
        for row_mode in requested_modes:
            complete = sorted(
                (row for row in result.rows if row["mode"] == row_mode and row["status"] == "complete"),
                key=lambda item: item["native_pass"],
            )
            series: dict[str, dict[int, float]] = {}
            frequency_by_pass: dict[int, float] = {}
            for row in result.rows:
                if row["mode"] != row_mode:
                    continue
                frequency = row.get("frequency_hz")
                if not isinstance(frequency, (int, float)) or isinstance(frequency, bool):
                    continue
                try:
                    frequency_hz = float(frequency)
                except OverflowError:
                    continue
                if math.isfinite(frequency_hz) and frequency_hz > 0:
                    frequency_by_pass[int(row["native_pass"])] = frequency_hz
            for row in complete:
                native = int(row["native_pass"])
                series.setdefault("magnetic+inductive / normalization", {})[native] = (
                    float(row["magnetic_energy_balance_j"]) / float(row["normalization_energy_j"])
                )
                for surface in row["surface_contributions"]:
                    label = (
                        f"surface:{surface.get('group_id', surface.get('contribution_id'))}@"
                        f"{surface['margin_um']:g}um"
                    )
                    series.setdefault(label, {})[native] = float(surface["participation"])
                for junction in row["junctions"]:
                    series.setdefault(f"junction-L:{junction['junction_id']}", {})[native] = float(
                        junction["inductive_participation"]
                    )
                    series.setdefault(f"junction-C:{junction['junction_id']}", {})[native] = float(
                        junction["capacitive_participation"]
                    )
            for label, values in sorted(series.items()):
                lines.append({
                    "name": f"mode {row_mode} {label}", "x": pass_axis,
                    "y": [values.get(item) for item in pass_axis], "secondary_y": False,
                })
            lines.append({
                "name": f"mode {row_mode} frequency", "x": pass_axis,
                "y": [frequency_by_pass.get(item) for item in pass_axis], "secondary_y": True,
            })
    return {
        "result_kind": result.result_kind, "mode": mode, "native_pass": target_pass,
        "show_convergence": show_convergence, "convergence_view": convergence_view,
        "bars": bars, "lines": lines, "unavailable": unavailable,
    }


def plot_epr_result(
    result: EprResult,
    *,
    mode: int | None = None,
    native_pass: int | None = None,
    show_convergence: bool = True,
) -> Any:
    """Render the exact selected pass plus an optional separate convergence view."""

    if not isinstance(result, EprResult):
        raise TypeError("result must be EprResult")
    if mode is not None and (type(mode) is not int or mode <= 0):
        raise ValueError("mode must be a positive integer or None")
    if native_pass is not None and (type(native_pass) is not int or native_pass <= 0):
        raise ValueError("native_pass must be a positive integer or None")
    if native_pass is not None and result.result_kind != "adaptive_history":
        raise ValueError("native_pass selection requires adaptive-history results")
    if type(show_convergence) is not bool:
        raise TypeError("show_convergence must be bool")
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    view = _history_view(
        result, mode=mode, native_pass=native_pass, show_convergence=show_convergence
    )
    convergence_view = view["convergence_view"]
    figure = make_subplots(
        rows=2 if convergence_view else 1,
        cols=1,
        specs=([[{}], [{"secondary_y": True}]] if convergence_view else [[{}]]),
        shared_xaxes=False,
        vertical_spacing=0.16,
        subplot_titles=(
            ("Selected-pass participation", "Pass convergence")
            if convergence_view else ("Participation",)
        ),
    )
    for bar in view["bars"]:
        figure.add_trace(go.Bar(name=bar["name"], x=bar["labels"], y=bar["values"]), row=1, col=1)
    if convergence_view:
        for line in view["lines"]:
            figure.add_trace(
                go.Scatter(
                    name=line["name"], x=line["x"], y=line["y"],
                    mode="lines+markers", connectgaps=False,
                ),
                row=2, col=1, **({"secondary_y": True} if line["secondary_y"] else {}),
            )
    figure.update_layout(
        barmode="group",
        title=f"AEDT EPR — {view['result_kind'].replace('_', ' ')}",
        meta={
            "schema_version": "scgsim.aedt.epr-plot.v1",
            "result_kind": view["result_kind"],
            "selected_mode": mode,
            "selected_native_pass": view["native_pass"],
            "show_convergence": show_convergence,
            "unavailable": view["unavailable"],
        },
    )
    figure.update_xaxes(title_text="Contribution", row=1, col=1)
    figure.update_yaxes(title_text="Participation", row=1, col=1)
    if convergence_view:
        figure.update_xaxes(title_text="Adaptive pass", row=2, col=1)
        figure.update_yaxes(
            title_text="Participation / energy-balance ratio", row=2, col=1, secondary_y=False
        )
        figure.update_yaxes(
            title_text="Frequency (Hz)", row=2, col=1, secondary_y=True, showgrid=False
        )
    if view["unavailable"]:
        figure.add_annotation(
            text="<br>".join(view["unavailable"]), xref="paper", yref="paper",
            x=0.0, y=1.0, xanchor="left", yanchor="bottom", showarrow=False,
        )
    return figure


def _show_view(result: EprResult, *, mode: int | None, native_pass: int | None) -> dict[str, Any]:
    if not isinstance(result, EprResult):
        raise TypeError("result must be EprResult")
    selected = [
        row for row in result.rows
        if (mode is None or row["mode"] == mode)
        and (native_pass is None or row.get("native_pass") == native_pass)
    ]
    if len(selected) != 1:
        raise ValueError("show_epr requires one exact mode/pass selection")
    row = selected[0]
    if row["status"] != "complete":
        frequency = row.get("frequency_hz")
        return {
            "row": row, "complete": False,
            "frequency_text": (
                f"; verified frequency {frequency:.6g} Hz"
                if isinstance(frequency, (int, float)) and math.isfinite(frequency) and frequency > 0
                else ""
            ),
        }
    surfaces = row["surface_contributions"]
    domains = row["electric_domains"]
    junctions = row["junctions"]
    baselines = [item for item in surfaces if item.get("evaluation_kind") == "unmasked_baseline"]
    known_baselines = [item["inverse_q"] for item in baselines if item.get("inverse_q") is not None]
    return {
        "row": row, "complete": True, "surfaces": surfaces, "domains": domains,
        "junctions": junctions,
        "bulk_participations": [
            item["energy_j"] / row["normalization_energy_j"] for item in domains
        ],
        "coverage": {
            "surface_rows": len(surfaces),
            "bindings": sum(len(item.get("members", item.get("binding_ids", ()))) for item in surfaces),
            "bulk_domains": len(domains), "junctions": len(junctions),
            "known_baseline_subtotal": (
                f"{sum(known_baselines):.6g}" if known_baselines else "unavailable"
            ),
            "unknown_baseline_losses": sum(item.get("inverse_q") is None for item in baselines),
        },
    }


def show_epr(result: EprResult, *, mode: int | None = None, native_pass: int | None = None) -> Any:
    """Show one exact pass/mode with separate surface, bulk, and junction axes."""

    view = _show_view(result, mode=mode, native_pass=native_pass)
    row = view["row"]
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(rows=3, cols=1, subplot_titles=("Surface", "Bulk", "Junction"))
    if view["complete"]:
        surfaces = view["surfaces"]
        surface_labels = [
            f"{item.get('owner_id', item.get('contribution_id', 'owner'))} / "
            f"{item['interface_kind']} / {item.get('evaluation_kind', 'requested_margin')} "
            f"{item['margin_um']:g} µm"
            for item in surfaces
        ]
        figure.add_trace(go.Bar(
            x=[item["participation"] for item in surfaces], y=surface_labels,
            orientation="h", name="surface p",
            customdata=[json.dumps({
                "group_id": item.get("group_id"),
                "members": item.get("members", item.get("binding_ids")),
                "inverse_q": item.get("inverse_q"),
                "assumptions": item.get("assumptions"),
                "original_assumptions": item.get("original_assumptions"),
                "assumption_update": item.get("assumption_update"),
            }, sort_keys=True, default=str) for item in surfaces],
            hovertemplate="%{y}<br>p=%{x:.6g}<br>%{customdata}<extra></extra>",
        ), row=1, col=1)
        domains = view["domains"]
        figure.add_trace(go.Bar(
            x=view["bulk_participations"], y=[item["domain_id"] for item in domains],
            orientation="h", name="bulk p",
        ), row=2, col=1)
        junctions = view["junctions"]
        figure.add_trace(go.Bar(
            x=[item["inductive_participation"] for item in junctions],
            y=[item["junction_id"] for item in junctions], orientation="h", name="junction pL",
        ), row=3, col=1)
        figure.add_trace(go.Bar(
            x=[item["capacitive_participation"] for item in junctions],
            y=[item["junction_id"] for item in junctions], orientation="h", name="junction pC",
        ), row=3, col=1)
        coverage = view["coverage"]
        figure.add_annotation(
            text=(
                f"Coverage: {coverage['surface_rows']} surface rows / "
                f"{coverage['bindings']} bindings; "
                f"{coverage['bulk_domains']} bulk domains; {coverage['junctions']} junctions. "
                f"Known baseline 1/Q subtotal: {coverage['known_baseline_subtotal']}; "
                f"{coverage['unknown_baseline_losses']} baseline losses unknown. "
                "Sidewalls uncomputed; requested margins are alternatives."
            ), xref="paper", yref="paper", x=0, y=1.08, showarrow=False,
        )
    else:
        figure.add_annotation(
            text=f"Partial native result{view['frequency_text']}; EPR quantities unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
    figure.update_layout(
        title=f"EPR mode {row['mode']}" + (f" pass {row['native_pass']}" if "native_pass" in row else " saved field"),
        height=900, showlegend=True,
    )
    for axis in ("xaxis", "xaxis2", "xaxis3"):
        figure.layout[axis].title = "participation"
    return figure


def display_resolved_run(resolved: Any, *, show_details: bool) -> None:
    """Compose receipt-bound tables and existing EPR figures for a notebook."""

    from IPython.display import HTML, display

    rows = resolved.physics_results()
    benchmark = resolved.simulation_benchmark()
    result = resolved.epr_result()
    project_sha = dict(resolved._output_hashes)[resolved.project_path.name]
    source_sha = (
        result.provenance["analysis_source_sha256"] if result is not None else None
    )
    final_pass = (
        result.provenance["solver_last_completed_pass"] if result is not None else None
    )
    display(HTML(
        "<section><h3>AEDT completed run</h3>"
        f"<p>Mode: {html.escape(resolved.mode)}; "
        f"project: {html.escape(resolved.project_path.name)}; "
        f"receipt: {html.escape(resolved.receipt_path.relative_to(resolved.receipt_path.parent.parent).as_posix())}"
        + (f"; observed final pass: {final_pass}" if final_pass is not None else "")
        + ".</p>"
        f"<p>Spec SHA-256: {html.escape(str(resolved._spec_sha256))}; "
        f"project SHA-256: {html.escape(project_sha)}; "
        f"EPR analysis source SHA-256: {html.escape(str(source_sha))}.</p>"
        "</section>"
    ))
    display_benchmark(benchmark, show_details=show_details)
    headings = tuple(rows[0]) if rows else ()
    shown = rows if show_details else rows[:20]
    display(HTML(
        "<section><h3>AEDT resolved results</h3>"
        f"<p>{html.escape(resolved.mode)}; {len(rows)} primary result rows "
        "verified against the completed receipt.</p>"
        + (
            "<table><thead><tr>"
            + "".join(f"<th>{html.escape(key)}</th>" for key in headings)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(
                    f"<td>{html.escape(str(row.get(key, '')))}</td>"
                    for key in headings
                ) + "</tr>"
                for row in shown
            ) + "</tbody></table>"
            if headings else "<p>No primary result rows were recorded.</p>"
        )
        + (
            f"<p>Showing {len(shown)} of {len(rows)} rows; "
            "set show_details=True for the full table.</p>"
            if len(shown) != len(rows) else ""
        )
        + "</section>"
    ))
    if result is None:
        display(HTML("<p>Adaptive EPR was not requested for this run.</p>"))
        return
    last_pass = result.provenance["solver_last_completed_pass"]
    requested_modes = result.provenance["requested_modes"]
    display(HTML(
        f"<h3>Adaptive EPR at observed pass {last_pass}</h3>"
        "<p>Requested modes are shown individually with surface, bulk, and "
        "junction evidence. Missing or partial values remain unavailable.</p>"
    ))
    try:
        display(plot_epr_result(result, show_convergence=True))
    except ImportError:
        display(HTML("<p>Plotly is unavailable; the adaptive history figure cannot be shown.</p>"))
    for mode in requested_modes:
        selected = [
            row for row in result.rows
            if row["mode"] == mode and row["native_pass"] == last_pass
        ]
        if not selected:
            display(HTML(
                f"<p>Mode {mode}: observed pass {last_pass} has no EPR row.</p>"
            ))
            continue
        if len(selected) != 1:
            raise RuntimeError("adaptive EPR contains duplicate mode/pass rows")
        try:
            display(show_epr(result, mode=mode, native_pass=last_pass))
        except ImportError:
            status = html.escape(str(selected[0]["status"]))
            display(HTML(
                f"<p>Mode {mode}, observed pass {last_pass}: {status}; "
                "Plotly is unavailable for the separate EPR axes.</p>"
            ))


def _benchmark_view(data: dict[str, Any], *, show_details: bool) -> dict[str, Any]:
    """Project the recorded native profile tree into display-only rows and series."""

    benchmark = data["benchmark"]
    if benchmark["status"] != "complete":
        return {"table_rows": [], "adaptive_series": [], "sweep_series": []}
    rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []

    def visit(node: dict[str, Any], profile: str, group: str, branch: str, pass_id: int | None = None) -> None:
        name = str(node["name"])
        pass_id = node.get("adaptive_pass", pass_id)
        kind = (
            "adaptive" if pass_id is not None else
            "sweep" if "sweep" in name.lower() else
            "subproblem" if data["mode"] in {"q2d", "q3d"} else branch
        )
        if node.get("metrics"):
            rows.append({
                "profile": profile, "process_group": group, "scope": kind,
                "stage": name, "pass": pass_id,
                "frequency_native": node.get("native_properties", {}).get("Frequency"),
                **node["metrics"],
            })
        if "adaptive_pass" in node:
            peak = node.get("stage_memory_peak", {})
            pass_rows.append({
                "profile": profile, "process_group": group, "scope": "adaptive",
                "stage": name, "pass": pass_id,
                "memory_native": peak.get("memory_native", ""),
                **node["metrics"],
            })
        for child in node.get("children", []):
            visit(child, profile, group, kind, pass_id)

    for profile in benchmark["profiles"]:
        for group in profile["process_groups"]:
            visit(group, profile["native_setup_profile"], group["name"], "stage")
    summary_rows = pass_rows + [
        row for row in rows
        if row["pass"] is None and row["scope"] in {"sweep", "subproblem"}
    ]
    table_rows = rows if show_details else (summary_rows or rows)
    adaptive_series: list[dict[str, Any]] = []
    sweep_series: list[dict[str, Any]] = []
    if table_rows:
        for profile in benchmark["profiles"]:
            for group in profile["process_groups"]:
                stages = {
                    row["stage"] for row in rows
                    if row["profile"] == profile["native_setup_profile"]
                    and row["process_group"] == group["name"]
                    and row["scope"] == "adaptive" and row["pass"] is not None
                    and ("real_seconds" in row or "elapsed_seconds" in row)
                }
                for stage in sorted(stages):
                    for metric in ("real_seconds", "elapsed_seconds"):
                        selected = [
                            row for row in rows
                            if row["profile"] == profile["native_setup_profile"]
                            and row["process_group"] == group["name"]
                            and row["stage"] == stage and row["pass"] is not None
                            and metric in row
                            and (metric == "real_seconds" or "real_seconds" not in row)
                        ]
                        if selected:
                            adaptive_series.append({
                                "x": [row["pass"] for row in selected],
                                "y": [row[metric] for row in selected],
                                "name": f"{group['name']} / {stage} ({metric.removesuffix('_seconds')})",
                            })
        for profile in benchmark["profiles"]:
            for group in profile["process_groups"]:
                for metric in ("real_seconds", "elapsed_seconds"):
                    selected = [
                        row for row in rows
                        if row["profile"] == profile["native_setup_profile"]
                        and row["process_group"] == group["name"]
                        and row["scope"] == "sweep" and row["frequency_native"] is not None
                        and metric in row
                        and (metric == "real_seconds" or "real_seconds" not in row)
                    ]
                    if selected:
                        sweep_series.append({
                            "x": [row["frequency_native"] for row in selected],
                            "y": [row[metric] for row in selected],
                            "name": f"{group['name']} ({metric.removesuffix('_seconds')})",
                        })
    return {
        "table_rows": table_rows,
        "adaptive_series": adaptive_series,
        "sweep_series": sweep_series,
    }


def display_benchmark(data: dict[str, Any], *, show_details: bool) -> None:
    """Display recorded benchmark evidence; preserve the table without Plotly."""

    from IPython.display import HTML, display

    benchmark = data["benchmark"]
    status = html.escape(str(benchmark["status"]))
    mode = html.escape(str(data["mode"]))
    display(HTML(f"<section><h3>AEDT simulation benchmark</h3><p>{mode} · {status}</p>"
                 f"<p>SCGSim execution: {data['execution_seconds']} s; "
                 f"project: {data['project_bytes']} bytes; "
                 f"primary CSV: {data['primary_csv_bytes']} bytes.</p></section>"))
    if benchmark["status"] != "complete":
        if benchmark.get("reason"):
            display(HTML(f"<p>{html.escape(str(benchmark['reason']))}</p>"))
        return
    view = _benchmark_view(data, show_details=show_details)
    table_rows = view["table_rows"]
    if table_rows:
        headings = ("profile", "process_group", "scope", "pass", "stage", "real_seconds", "elapsed_seconds", "cpu_seconds", "memory_native", "tetrahedra", "solved_elements", "elements", "linear_matrix_size")
        head = "".join(f"<th>{html.escape(key)}</th>" for key in headings)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in headings) + "</tr>" for row in table_rows)
        display(HTML(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"))
        try:
            import plotly.graph_objects as go
        except ImportError:
            display(HTML("<p>Plotly is unavailable; the native pass table remains available.</p>"))
        else:
            figure = go.Figure()
            for series in view["adaptive_series"]:
                figure.add_scatter(x=series["x"], y=series["y"], mode="lines+markers", name=series["name"])
            if figure.data:
                figure.update_layout(xaxis_title="Native adaptive pass", yaxis_title="Native reported time (s; real/elapsed labelled)")
                display(figure)
            sweep = go.Figure()
            for series in view["sweep_series"]:
                sweep.add_scatter(x=series["x"], y=series["y"], mode="lines+markers", name=series["name"])
            if sweep.data:
                sweep.update_layout(xaxis_title="Native sweep frequency", yaxis_title="Native reported time (s; real/elapsed labelled)")
                display(sweep)
    if show_details:
        display(HTML("<pre>" + html.escape(json.dumps(benchmark, indent=2)) + "</pre>"))
