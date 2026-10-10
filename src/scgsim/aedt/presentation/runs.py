"""Resolved AEDT run result presentation only."""

from __future__ import annotations


import html

import json

from typing import Any

from scgsim.presentation.notebook import (
    INTERFACE_COLORS,
    cards,
    checked_theme,
    details,
    json_details,
    section,
    style_figure,
    table,
)

from scgsim.aedt.epr.models import EprResult, detached

from scgsim.aedt.presentation.benchmark import display_benchmark

from scgsim.aedt.presentation.epr import (
    _history_view,
    _native_reference_html,
    _ranking_height,
    _show_view,
    _surface_hover,
    _surface_key,
    _surface_label,
    _surface_metadata,
    _surface_tick_label,
    _usable_frequency,
)


def _history_figures(result: EprResult, *, theme: str) -> tuple[Any, ...]:
    """Separate the existing pass projections without changing eligibility."""

    from plotly import graph_objects as go

    view = _history_view(result, mode=None, native_pass=None, show_convergence=True)
    surface_groups = sorted(
        {
            (line["mode"], line["interface_kind"])
            for line in view["lines"]
            if line["kind"] == "surface"
        },
        key=lambda item: (
            item[0],
            ("MA", "MS", "SA").index(item[1]) if item[1] in {"MA", "MS", "SA"} else 3,
            item[1],
        ),
    )
    groups = [
        ("frequency", None, None, "Frequency by adaptive pass", "Frequency (Hz)"),
        *(
            (
                "surface",
                mode,
                interface,
                f"Surface participation · mode {mode} / {interface}",
                "Participation",
            )
            for mode, interface in surface_groups
        ),
        (
            "junction",
            None,
            None,
            "Junction participation by adaptive pass",
            "Participation",
        ),
        (
            "energy",
            None,
            None,
            "Magnetic and inductive energy balance",
            "Balance / normalization",
        ),
    ]
    figures = []
    for kind, selected_mode, selected_interface, title, y_title in groups:
        figure = go.Figure()
        seen: dict[str, int] = {}
        for line in view["lines"]:
            if not (
                line["kind"] == kind
                or kind == "junction"
                and line["kind"].startswith("junction")
            ):
                continue
            if selected_mode is not None and (
                line["mode"] != selected_mode
                or line.get("interface_kind") != selected_interface
            ):
                continue
            index = seen.setdefault(line["identity"], len(seen))
            color = (
                INTERFACE_COLORS.get(line.get("interface_kind"), "#CC79A7")
                if kind == "surface"
                else "#0072B2"
                if kind == "frequency"
                else "#D55E00"
                if kind == "junction"
                else "#E69F00"
            )
            figure.add_scatter(
                x=line["x"],
                y=line["y"],
                name=line["name"],
                mode="lines+markers",
                connectgaps=False,
                line={
                    "color": color,
                    "dash": ("solid", "dash", "dot", "dashdot")[index % 4],
                },
                marker={"symbol": ("circle", "diamond", "square", "cross")[index % 4]},
                customdata=[line.get("hover", line["identity"])] * len(line["x"]),
                meta=line.get("source_metadata", {"identity": line["identity"]}),
                hovertemplate="%{fullData.name}<br>pass %{x}<br>value %{y:.6g}<br>"
                "%{customdata}<extra></extra>",
            )
        figure.update_layout(
            title=title,
            xaxis_title="Adaptive pass",
            yaxis_title=y_title,
            meta={
                "requested_modes": list(result.provenance["requested_modes"]),
                "observed_pass": view["native_pass"],
                "kind": kind,
                "mode": selected_mode,
                "interface_kind": selected_interface,
            },
        )
        figure.update_xaxes(
            tickmode="array", tickvals=list(range(1, view["native_pass"] + 1))
        )
        legend_rows = len(figure.data)
        style_figure(figure, theme=theme, height=max(460, 440 + 28 * legend_rows))
        figure.update_layout(
            margin={"l": 70, "r": 20, "t": 65, "b": max(115, 75 + 29 * legend_rows)}
        )
        figures.append(figure)
    return tuple(figures)


def _margin_figure(row: Any, *, theme: str) -> Any | None:
    """Keep requested margins within exact source/film lineages; baseline is a marker."""

    from plotly import graph_objects as go

    surfaces = row["surface_contributions"]
    if not surfaces:
        return None
    groups: dict[str, list[Any]] = {}
    for item in surfaces:
        members = item.get("members", item.get("binding_ids", ()))
        lineage = json.dumps(
            {
                "owner_id": item.get("owner_id"),
                "interface_kind": item["interface_kind"],
                "field_side": item.get("field_side"),
                "members": detached(members),
                "assumptions": detached(item.get("assumptions")),
            },
            sort_keys=True,
        )
        groups.setdefault(lineage, []).append(item)
    figure = go.Figure()
    for index, items in enumerate(groups.values()):
        first = items[0]
        color = INTERFACE_COLORS.get(first["interface_kind"], "#CC79A7")
        requested = sorted(
            (
                item
                for item in items
                if item.get("evaluation_kind") != "unmasked_baseline"
            ),
            key=lambda item: item["margin_um"],
        )
        baseline = [
            item for item in items if item.get("evaluation_kind") == "unmasked_baseline"
        ]
        label = _surface_label(first).split(" · ")[0] + f" · {first['interface_kind']}"
        for evaluation, values, mode in (
            ("Requested", requested, "lines+markers"),
            ("Unmasked", baseline, "markers"),
        ):
            if not values:
                continue
            figure.add_scatter(
                x=[item["margin_um"] for item in values],
                y=[item["participation"] for item in values],
                mode=mode,
                name=f"{label} · {evaluation}",
                connectgaps=False,
                line={
                    "color": color,
                    "dash": ("solid", "dash", "dot", "dashdot")[index % 4],
                },
                marker={
                    "color": color,
                    "symbol": ("circle", "diamond", "square", "cross")[index % 4],
                },
                customdata=[_surface_hover(item) for item in values],
                meta=[_surface_metadata(item) for item in values],
                hovertemplate="%{fullData.name}<br>margin %{x:g} µm<br>p=%{y:.6g}<br>"
                "%{customdata}<extra></extra>",
            )
    figure.update_layout(
        title=f"Mode {row['mode']} requested margins and unmasked baseline",
        xaxis_title="Requested margin (µm)",
        yaxis_title="Surface participation",
        meta={"mode": row["mode"], "native_pass": row.get("native_pass")},
    )
    return style_figure(figure, theme=theme)


def _final_quantity_figures(row: Any, *, theme: str) -> tuple[Any, ...]:
    """Present each final-pass quantity independently without recombination."""

    from plotly import graph_objects as go

    surfaces = sorted(
        row["surface_contributions"], key=lambda item: -float(item["participation"])
    )
    surface_keys = [_surface_key(item) for item in surfaces]
    surface = go.Figure(
        go.Bar(
            x=[item["participation"] for item in surfaces],
            y=surface_keys,
            orientation="h",
            name="surface p",
            marker_color=[
                INTERFACE_COLORS.get(item["interface_kind"], "#CC79A7")
                for item in surfaces
            ],
            customdata=[_surface_hover(item) for item in surfaces],
            meta=[_surface_metadata(item) for item in surfaces],
            hovertemplate="p=%{x:.6g}<br>%{customdata}<extra></extra>",
        )
    )
    surface.update_yaxes(
        tickmode="array",
        tickvals=surface_keys,
        ticktext=[_surface_tick_label(item) for item in surfaces],
        autorange="reversed",
    )
    surface.update_layout(
        title=f"Mode {row['mode']} surface owners · pass {row['native_pass']}",
        xaxis_title="Surface participation",
    )
    style_figure(surface, theme=theme, height=_ranking_height(len(surfaces)))
    surface.update_layout(
        showlegend=False, margin={"l": 180, "r": 20, "t": 65, "b": 75}
    )
    surface.update_yaxes(automargin=False, tickfont={"size": 12})
    figures = [surface]
    bulk = go.Figure(
        go.Bar(
            x=[
                item["energy_j"] / row["normalization_energy_j"]
                for item in row["electric_domains"]
            ],
            y=[html.escape(str(item["domain_id"])) for item in row["electric_domains"]],
            orientation="h",
            name="bulk p",
            marker_color="#0072B2",
        )
    )
    bulk.update_layout(
        title=f"Mode {row['mode']} bulk domains · pass {row['native_pass']}",
        xaxis_title="Bulk participation",
    )
    style_figure(
        bulk, theme=theme, height=_ranking_height(len(row["electric_domains"]))
    )
    bulk.update_layout(showlegend=False, margin={"l": 140, "r": 20, "t": 65, "b": 75})
    figures.append(bulk)
    for field, label, color in (
        ("inductive_participation", "junction pL", "#D55E00"),
        ("capacitive_participation", "junction pC", "#CC79A7"),
    ):
        junction = go.Figure(
            go.Bar(
                x=[item[field] for item in row["junctions"]],
                y=[html.escape(str(item["junction_id"])) for item in row["junctions"]],
                orientation="h",
                name=label,
                marker_color=color,
            )
        )
        junction.update_layout(
            title=f"Mode {row['mode']} {label} · pass {row['native_pass']}",
            xaxis_title=label,
        )
        style_figure(
            junction, theme=theme, height=_ranking_height(len(row["junctions"]))
        )
        junction.update_layout(
            showlegend=False, margin={"l": 140, "r": 20, "t": 65, "b": 75}
        )
        figures.append(junction)
    return tuple(figures)


def _physics_tables(
    result: EprResult, row: Any, *, show_details: bool, plotly_available: bool
) -> str:
    view = _show_view(result, mode=row["mode"], native_pass=row["native_pass"])
    coverage = view["coverage"]
    surfaces = sorted(view["surfaces"], key=lambda item: -float(item["participation"]))
    surface_rows = [
        (
            item.get("owner_id", item.get("contribution_id")),
            item["interface_kind"],
            "Unmasked"
            if item.get("evaluation_kind") == "unmasked_baseline"
            else "Requested",
            item["margin_um"],
            item["participation"],
            item.get("assumptions", {}).get("loss_tangent"),
            item.get("inverse_q"),
            item.get("group_id"),
        )
        for item in surfaces
    ]
    surface_table = table(
        (
            "Owner",
            "Interface",
            "Evaluation",
            "Margin (µm)",
            "p",
            "Loss tangent",
            "1/Q",
            "Group",
        ),
        surface_rows,
    )
    bulk_table = table(
        ("Domain", "Energy (J)", "Participation"),
        [
            (item["domain_id"], item["energy_j"], participation)
            for item, participation in zip(view["domains"], view["bulk_participations"])
        ],
    )
    junction_table = table(
        ("Junction", "pL", "pC"),
        [
            (
                item["junction_id"],
                item["inductive_participation"],
                item["capacitive_participation"],
            )
            for item in view["junctions"]
        ],
    )
    return (
        cards(
            (
                ("Surface rows", coverage["surface_rows"]),
                ("Source bindings", coverage["bindings"]),
                ("Bulk domains", coverage["bulk_domains"]),
                ("Junctions", coverage["junctions"]),
                ("Known baseline 1/Q subtotal", coverage["known_baseline_subtotal"]),
                ("Unknown baseline loss", coverage["unknown_baseline_losses"]),
            )
        )
        + "<p>Sidewalls uncomputed; requested margins are alternative evaluations. "
        "No total device Q or T1 is inferred.</p>"
        + _native_reference_html(row, opened=show_details)
        + details(
            "Surface owner, interface, margin, and loss table",
            surface_table,
            opened=show_details or not plotly_available or len(surfaces) <= 12,
        )
        + details(
            "Bulk participation table",
            bulk_table,
            opened=show_details or not plotly_available,
        )
        + details(
            "Junction inductive and capacitive table",
            junction_table,
            opened=show_details or not plotly_available,
        )
    )


def display_resolved_run(
    resolved: Any, *, show_details: bool, theme: str = "light"
) -> None:
    """Show verified identity, numerical evidence, cost, and final-pass physics."""

    checked_theme(theme)
    from IPython.display import HTML, display

    rows = resolved.physics_results()
    benchmark = resolved.simulation_benchmark()
    result = resolved.epr_result()
    final_pass = (
        result.provenance["solver_last_completed_pass"] if result is not None else None
    )
    modes = result.provenance["requested_modes"] if result is not None else None
    display(
        HTML(
            section(
                "AEDT completed run",
                cards(
                    (
                        ("Backend", "AEDT"),
                        ("Project", resolved.project_path.name),
                        ("Analysis", resolved.mode),
                        ("Setup", resolved._setup_name),
                        (
                            "Requested modes",
                            ", ".join(map(str, modes))
                            if modes is not None
                            else "not requested",
                        ),
                        ("Observed final pass", final_pass),
                        ("Completion", "completed"),
                    )
                ),
                theme=theme,
            )
        )
    )

    lumped = resolved._lumped_boundaries
    if lumped:
        summaries = []
        for record in lumped:
            request = record["requested"]
            if "topology" in request:
                treatment = f"{request['topology']} R={request['resistance_ohm']} ohm; L={request['inductance_h']} H; C={request['capacitance_f']} F (None disabled)"
            else:
                treatment = f"Lumped Terminal {request['impedance_ohm']} ohm; renormalize={request['renormalize']}; deembed={request['deembed_um']} um"
            summaries.append(
                (
                    record.get("support_id", request.get("support_id")),
                    record["boundary"],
                    treatment,
                )
            )
        display(
            HTML(
                section(
                    "Source-bound lumped elements",
                    table(
                        ("Support", "Native boundary", "Authored treatment"), summaries
                    )
                    + json_details(
                        "Source and native assignment evidence",
                        {"boundaries": list(lumped)},
                        opened=show_details,
                    ),
                    theme=theme,
                )
            )
        )

    headings = tuple(rows[0]) if rows else ()
    primary_table = (
        table(headings, [tuple(row.get(key, "") for key in headings) for row in rows])
        if headings
        else "<p>No primary result rows were recorded.</p>"
    )
    display(
        HTML(
            section(
                "AEDT numerical evidence",
                f"<p>{len(rows)} primary result rows verified against the completed receipt.</p>"
                + details(
                    "Primary result table",
                    primary_table,
                    opened=show_details or len(rows) <= 20,
                ),
                theme=theme,
            )
        )
    )
    plotly_available = True
    if result is not None:
        try:
            for figure in _history_figures(result, theme=theme):
                display(figure)
        except ImportError:
            plotly_available = False
            display(
                HTML(
                    section(
                        "Numerical figures unavailable",
                        "<p>Plotly is unavailable; recorded frequencies and complete-row "
                        "quantities remain in the tables below.</p>",
                        theme=theme,
                    )
                )
            )
    display_benchmark(benchmark, show_details=show_details, theme=theme)

    if result is None:
        display(
            HTML(
                section(
                    "AEDT physics",
                    "<p>Adaptive EPR was not requested for this run.</p>",
                    theme=theme,
                )
            )
        )
    else:
        for mode in modes:
            selected = [
                row
                for row in result.rows
                if row["mode"] == mode and row["native_pass"] == final_pass
            ]
            if not selected:
                display(
                    HTML(
                        section(
                            f"Mode {mode} · observed pass {final_pass}",
                            "<p>The requested final-pass EPR row is missing. Earlier passes are not substituted.</p>",
                            theme=theme,
                        )
                    )
                )
                continue
            if len(selected) != 1:
                raise RuntimeError("adaptive EPR contains duplicate mode/pass rows")
            row = selected[0]
            if row["status"] != "complete":
                frequency = _usable_frequency(row.get("frequency_hz"))
                display(
                    HTML(
                        section(
                            f"Mode {mode} · observed pass {final_pass}",
                            cards(
                                (("Status", "partial"), ("Frequency (Hz)", frequency))
                            )
                            + "<p>Surface, bulk, and junction EPR quantities are unavailable: "
                            + html.escape(
                                str(
                                    row.get(
                                        "combination_error",
                                        "incomplete native evidence",
                                    )
                                )
                            )
                            + ".</p>"
                            + _native_reference_html(row, opened=show_details),
                            theme=theme,
                        )
                    )
                )
                continue
            display(
                HTML(
                    section(
                        f"Mode {mode} · observed pass {final_pass}",
                        _physics_tables(
                            result,
                            row,
                            show_details=show_details,
                            plotly_available=plotly_available,
                        ),
                        theme=theme,
                    )
                )
            )
            if plotly_available:
                try:
                    for figure in _final_quantity_figures(row, theme=theme):
                        display(figure)
                    margin = _margin_figure(row, theme=theme)
                    if margin is not None:
                        display(margin)
                except ImportError:
                    plotly_available = False
                    display(
                        HTML(
                            section(
                                "Physics figures unavailable",
                                "<p>Plotly is unavailable; final-pass tables remain available.</p>",
                                theme=theme,
                            )
                        )
                    )
    project_sha = dict(resolved._output_hashes)[resolved.project_path.name]
    source_sha = (
        result.provenance["analysis_source_sha256"] if result is not None else None
    )
    receipt = resolved.receipt_path.relative_to(
        resolved.receipt_path.parent.parent
    ).as_posix()
    identity = {
        "receipt": receipt,
        "spec_sha256": resolved._spec_sha256,
        "project_sha256": project_sha,
        "analysis_source_sha256": source_sha,
    }
    display(
        HTML(
            section(
                "AEDT source and receipt details",
                json_details(
                    "Recorded source and receipt identity",
                    identity,
                    opened=show_details,
                ),
                theme=theme,
            )
        )
    )
