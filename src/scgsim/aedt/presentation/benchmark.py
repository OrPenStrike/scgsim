"""AEDT benchmark presentation only."""

from __future__ import annotations


import html

from typing import Any

from scgsim.presentation.notebook import (
    cards,
    checked_theme,
    details,
    json_details,
    section,
    style_figure,
    table,
)


def _benchmark_view(data: dict[str, Any], *, show_details: bool) -> dict[str, Any]:
    """Project the recorded native profile tree into display-only rows and series."""

    benchmark = data["benchmark"]
    if benchmark["status"] != "complete":
        return {"table_rows": [], "adaptive_series": [], "sweep_series": []}
    rows: list[dict[str, Any]] = []
    pass_rows: list[dict[str, Any]] = []

    def visit(
        node: dict[str, Any],
        profile: str,
        group: str,
        branch: str,
        pass_id: int | None = None,
    ) -> None:
        name = str(node["name"])
        pass_id = node.get("adaptive_pass", pass_id)
        kind = (
            "adaptive"
            if pass_id is not None
            else "sweep"
            if "sweep" in name.lower()
            else "subproblem"
            if data["mode"] in {"q2d", "q3d"}
            else branch
        )
        if node.get("metrics"):
            rows.append(
                {
                    "profile": profile,
                    "process_group": group,
                    "scope": kind,
                    "stage": name,
                    "pass": pass_id,
                    "frequency_native": node.get("native_properties", {}).get(
                        "Frequency"
                    ),
                    **node["metrics"],
                }
            )
        if "adaptive_pass" in node:
            peak = node.get("stage_memory_peak", {})
            pass_rows.append(
                {
                    "profile": profile,
                    "process_group": group,
                    "scope": "adaptive",
                    "stage": name,
                    "pass": pass_id,
                    "memory_native": peak.get("memory_native", ""),
                    **node["metrics"],
                }
            )
        for child in node.get("children", []):
            visit(child, profile, group, kind, pass_id)

    for profile in benchmark["profiles"]:
        for group in profile["process_groups"]:
            visit(group, profile["native_setup_profile"], group["name"], "stage")
    summary_rows = pass_rows + [
        row
        for row in rows
        if row["pass"] is None and row["scope"] in {"sweep", "subproblem"}
    ]
    table_rows = rows if show_details else (summary_rows or rows)
    adaptive_series: list[dict[str, Any]] = []
    sweep_series: list[dict[str, Any]] = []
    if table_rows:
        for profile in benchmark["profiles"]:
            for group in profile["process_groups"]:
                stages = {
                    row["stage"]
                    for row in rows
                    if row["profile"] == profile["native_setup_profile"]
                    and row["process_group"] == group["name"]
                    and row["scope"] == "adaptive"
                    and row["pass"] is not None
                    and ("real_seconds" in row or "elapsed_seconds" in row)
                }
                for stage in sorted(stages):
                    for metric in ("real_seconds", "elapsed_seconds"):
                        selected = [
                            row
                            for row in rows
                            if row["profile"] == profile["native_setup_profile"]
                            and row["process_group"] == group["name"]
                            and row["stage"] == stage
                            and row["pass"] is not None
                            and metric in row
                            and (metric == "real_seconds" or "real_seconds" not in row)
                        ]
                        if selected:
                            adaptive_series.append(
                                {
                                    "x": [row["pass"] for row in selected],
                                    "y": [row[metric] for row in selected],
                                    "name": f"{group['name']} / {stage} ({metric.removesuffix('_seconds')})",
                                }
                            )
        for profile in benchmark["profiles"]:
            for group in profile["process_groups"]:
                for metric in ("real_seconds", "elapsed_seconds"):
                    selected = [
                        row
                        for row in rows
                        if row["profile"] == profile["native_setup_profile"]
                        and row["process_group"] == group["name"]
                        and row["scope"] == "sweep"
                        and row["frequency_native"] is not None
                        and metric in row
                        and (metric == "real_seconds" or "real_seconds" not in row)
                    ]
                    if selected:
                        sweep_series.append(
                            {
                                "x": [row["frequency_native"] for row in selected],
                                "y": [row[metric] for row in selected],
                                "name": f"{group['name']} ({metric.removesuffix('_seconds')})",
                            }
                        )
    return {
        "table_rows": table_rows,
        "adaptive_series": adaptive_series,
        "sweep_series": sweep_series,
    }


def display_benchmark(
    data: dict[str, Any], *, show_details: bool, theme: str = "light"
) -> None:
    """Display recorded benchmark evidence; preserve the table without Plotly."""

    checked_theme(theme)
    from IPython.display import HTML, display

    benchmark = data["benchmark"]
    display(
        HTML(
            section(
                "AEDT simulation benchmark",
                cards(
                    (
                        ("Analysis", data["mode"]),
                        ("Native profile", benchmark["status"]),
                        ("SCGSim execution (s)", data["execution_seconds"]),
                        ("Project bytes", data["project_bytes"]),
                        ("Primary CSV bytes", data["primary_csv_bytes"]),
                    )
                ),
                theme=theme,
            )
        )
    )
    if benchmark["status"] != "complete":
        if benchmark.get("reason"):
            display(
                HTML(
                    section(
                        "Native profile unavailable",
                        f"<p>{html.escape(str(benchmark['reason']))}</p>",
                        theme=theme,
                    )
                )
            )
        return
    view = _benchmark_view(data, show_details=show_details)
    table_rows = view["table_rows"]
    if table_rows:
        headings = (
            "profile",
            "process_group",
            "scope",
            "pass",
            "stage",
            "real_seconds",
            "elapsed_seconds",
            "cpu_seconds",
            "memory_native",
            "tetrahedra",
            "solved_elements",
            "elements",
            "linear_matrix_size",
        )
        body = table(
            headings, [tuple(row.get(key) for key in headings) for row in table_rows]
        )
        display(
            HTML(
                section(
                    "Native stage and pass evidence",
                    details(
                        "Native profile table",
                        body,
                        opened=show_details or len(table_rows) <= 20,
                    ),
                    theme=theme,
                )
            )
        )
        try:
            import plotly.graph_objects as go
        except ImportError:
            display(
                HTML(
                    section(
                        "Benchmark figures unavailable",
                        "<p>Plotly is unavailable; the native pass table remains available.</p>",
                        theme=theme,
                    )
                )
            )
        else:
            figure = go.Figure()
            for series in view["adaptive_series"]:
                figure.add_scatter(
                    x=series["x"],
                    y=series["y"],
                    mode="lines+markers",
                    name=html.escape(series["name"]),
                )
            if figure.data:
                figure.update_layout(
                    xaxis_title="Native adaptive pass",
                    yaxis_title="Native reported time (s; real/elapsed labelled)",
                )
                display(style_figure(figure, theme=theme))
            sweep = go.Figure()
            for series in view["sweep_series"]:
                sweep.add_scatter(
                    x=series["x"],
                    y=series["y"],
                    mode="lines+markers",
                    name=html.escape(series["name"]),
                )
            if sweep.data:
                sweep.update_layout(
                    xaxis_title="Native sweep frequency",
                    yaxis_title="Native reported time (s; real/elapsed labelled)",
                )
                display(style_figure(sweep, theme=theme))
    if show_details:
        display(
            HTML(
                section(
                    "Native benchmark details",
                    json_details("Recorded native profile", benchmark, opened=True),
                    theme=theme,
                )
            )
        )
