"""EPR-specific plots and notebook presentation only."""

from __future__ import annotations


import html

import json

import math

import textwrap

from typing import Any

from scgsim.presentation.notebook import (
    INTERFACE_COLORS,
    checked_theme,
    display_text,
    style_figure,
)

from scgsim.aedt.epr.models import EprResult, detached


def _usable_frequency(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    try:
        frequency_hz = float(value)
    except OverflowError:
        return None
    return frequency_hz if math.isfinite(frequency_hz) and frequency_hz > 0 else None


def _surface_key(surface: Any) -> str:
    """Use recorded scientific identity, never a shortened display label."""

    members = surface.get("members", surface.get("binding_ids", ()))
    return json.dumps(
        {
            "group_id": surface.get("group_id"),
            "contribution_id": surface.get("contribution_id"),
            "owner_id": surface.get("owner_id"),
            "interface_kind": surface.get("interface_kind"),
            "evaluation_kind": surface.get("evaluation_kind", "requested_margin"),
            "margin_um": surface.get("margin_um"),
            "field_side": surface.get("field_side"),
            "members": detached(members),
        },
        sort_keys=True,
    )


def _surface_label_parts(surface: Any) -> tuple[str, str, str]:
    owner = surface.get("owner_id", surface.get("contribution_id", "owner"))
    readable_owner = str(owner).replace("_", " ")
    evaluation = (
        "Unmasked"
        if surface.get("evaluation_kind") == "unmasked_baseline"
        else f"Requested {surface['margin_um']:g} µm"
    )
    return readable_owner, str(surface["interface_kind"]), evaluation


def _surface_label(surface: Any) -> str:
    return " · ".join(
        html.escape(part, quote=False) for part in _surface_label_parts(surface)
    )


def _surface_tick_label(surface: Any) -> str:
    """Wrap category labels so long source names cannot consume the plot area."""

    raw = " · ".join(_surface_label_parts(surface))
    parts = textwrap.wrap(raw, width=28, break_long_words=True, break_on_hyphens=False)
    return "<br>".join(html.escape(part, quote=False) for part in parts)


def _surface_hover(surface: Any) -> str:
    metadata = {
        "group_id": surface.get("group_id"),
        "owner_id": surface.get("owner_id"),
        "interface_kind": surface.get("interface_kind"),
        "evaluation_kind": surface.get("evaluation_kind"),
        "margin_um": surface.get("margin_um"),
        "field_side": surface.get("field_side"),
        "members": detached(surface.get("members", surface.get("binding_ids"))),
        "source": surface.get("source"),
        "assumptions": detached(surface.get("assumptions")),
        "original_assumptions": detached(surface.get("original_assumptions")),
        "assumption_update": detached(surface.get("assumption_update")),
        "inverse_q": surface.get("inverse_q"),
    }
    members = metadata["members"] or []
    lines = [
        ("Owner", metadata["owner_id"]),
        ("Group", metadata["group_id"]),
        ("Interface", metadata["interface_kind"]),
        ("Evaluation", metadata["evaluation_kind"]),
        ("Margin (µm)", metadata["margin_um"]),
        ("Side", metadata["field_side"]),
        ("Source", metadata["source"]),
        ("Members", len(members)),
    ]
    for index, member in enumerate(members[:3], 1):
        if isinstance(member, dict):
            identity = ", ".join(
                f"{key}={member[key]}"
                for key in (
                    "binding_id",
                    "contribution_id",
                    "source_polygon_id",
                    "field_side",
                )
                if key in member
            )
        else:
            identity = str(member)
        lines.append((f"Member {index}", identity))
    if len(members) > 3:
        lines.append(
            (
                "Additional members",
                f"{len(members) - 3}; full records in trace metadata",
            )
        )
    assumptions = metadata["assumptions"]
    if isinstance(assumptions, dict):
        lines.extend(
            (name.replace("_", " "), value) for name, value in assumptions.items()
        )
    rendered = []
    for label, value in lines:
        wrapped = textwrap.wrap(
            str(display_text(value)),
            width=58,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        rendered.append(
            f"<b>{html.escape(label, quote=False)}</b>: "
            + "<br>".join(html.escape(part, quote=False) for part in wrapped)
        )
    return "<br>".join(rendered)


def _surface_metadata(surface: Any) -> dict[str, Any]:
    """Keep the full source record independently of a shortened hover display."""

    return detached(surface)


def _ranking_height(count: int) -> int:
    return max(460, 180 + 52 * count)


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
        row
        for row in result.rows
        if (mode is None or row["mode"] == mode)
        and (
            result.result_kind != "adaptive_history"
            or row["native_pass"] == target_pass
        )
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
        pass_label = (
            "" if "native_pass" not in row else f", pass {row['native_pass']:g}"
        )
        if row.get("status") == "partial" or "normalization_energy_j" not in row:
            unavailable.append(
                f"mode {row_mode}{pass_label}: "
                f"{row.get('combination_error', 'incomplete native evidence')}"
            )
            continue
        labels: list[str] = []
        keys: list[str] = []
        kinds: list[str] = []
        values: list[float] = []
        for surface in row.get("surface_contributions", ()):
            labels.append(_surface_tick_label(surface))
            keys.append(_surface_key(surface))
            kinds.append(f"surface:{surface['interface_kind']}")
            values.append(float(surface["participation"]))
        for junction in row.get("junctions", ()):
            labels.extend(
                [
                    f"junction-L:{html.escape(str(junction['junction_id']))}",
                    f"junction-C:{html.escape(str(junction['junction_id']))}",
                ]
            )
            keys.extend(
                [
                    f"junction-L:{junction['junction_id']}",
                    f"junction-C:{junction['junction_id']}",
                ]
            )
            kinds.extend(["junction-L", "junction-C"])
            values.extend(
                [
                    float(junction["inductive_participation"]),
                    float(junction["capacitive_participation"]),
                ]
            )
        bars.append(
            {
                "name": f"mode {row_mode}{pass_label}",
                "labels": labels,
                "keys": keys,
                "kinds": kinds,
                "values": values,
            }
        )
    convergence_view = show_convergence and result.result_kind == "adaptive_history"
    lines: list[dict[str, Any]] = []
    if convergence_view:
        last_pass = result.provenance.get("solver_last_completed_pass")
        if type(last_pass) is not int or last_pass <= 0:
            raise ValueError("adaptive EPR result lacks solver_last_completed_pass")
        pass_axis = list(range(1, last_pass + 1))
        for row_mode in requested_modes:
            complete = sorted(
                (
                    row
                    for row in result.rows
                    if row["mode"] == row_mode and row["status"] == "complete"
                ),
                key=lambda item: item["native_pass"],
            )
            series: dict[str, dict[str, Any]] = {}
            frequency_by_pass: dict[int, float] = {}
            for row in result.rows:
                if row["mode"] != row_mode:
                    continue
                frequency_hz = _usable_frequency(row.get("frequency_hz"))
                if frequency_hz is not None:
                    frequency_by_pass[int(row["native_pass"])] = frequency_hz
            for row in complete:
                native = int(row["native_pass"])
                energy = series.setdefault(
                    "energy",
                    {
                        "name": f"mode {row_mode} magnetic+inductive / normalization",
                        "kind": "energy",
                        "identity": "energy",
                        "values": {},
                    },
                )
                energy["values"][native] = float(
                    row["magnetic_energy_balance_j"]
                ) / float(row["normalization_energy_j"])
                for surface in row["surface_contributions"]:
                    identity = _surface_key(surface)
                    item = series.setdefault(
                        identity,
                        {
                            "name": f"mode {row_mode} {_surface_label(surface)}",
                            "kind": "surface",
                            "interface_kind": surface["interface_kind"],
                            "identity": identity,
                            "hover": _surface_hover(surface),
                            "source_metadata": _surface_metadata(surface),
                            "mode": row_mode,
                            "values": {},
                        },
                    )
                    item["values"][native] = float(surface["participation"])
                for junction in row["junctions"]:
                    for kind, field in (
                        ("junction-L", "inductive_participation"),
                        ("junction-C", "capacitive_participation"),
                    ):
                        identity = f"{kind}:{junction['junction_id']}"
                        item = series.setdefault(
                            identity,
                            {
                                "name": f"mode {row_mode} {html.escape(identity)}",
                                "kind": kind,
                                "identity": identity,
                                "mode": row_mode,
                                "values": {},
                            },
                        )
                        item["values"][native] = float(junction[field])
            for item in sorted(
                series.values(), key=lambda entry: entry["name"] + entry["identity"]
            ):
                lines.append(
                    {
                        **{
                            key: value for key, value in item.items() if key != "values"
                        },
                        "x": pass_axis,
                        "y": [item["values"].get(native) for native in pass_axis],
                        "secondary_y": False,
                    }
                )
            lines.append(
                {
                    "name": f"mode {row_mode} frequency",
                    "x": pass_axis,
                    "y": [frequency_by_pass.get(item) for item in pass_axis],
                    "kind": "frequency",
                    "identity": f"mode:{row_mode}:frequency",
                    "mode": row_mode,
                    "secondary_y": True,
                }
            )
    return {
        "result_kind": result.result_kind,
        "mode": mode,
        "native_pass": target_pass,
        "show_convergence": show_convergence,
        "convergence_view": convergence_view,
        "bars": bars,
        "lines": lines,
        "unavailable": unavailable,
    }


def plot_epr_result(
    result: EprResult,
    *,
    mode: int | None = None,
    native_pass: int | None = None,
    show_convergence: bool = True,
    theme: str = "light",
) -> Any:
    """Render exact selected-pass values and independent adaptive history axes."""

    checked_theme(theme)
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
    titles = (
        (
            "Selected-pass participation",
            "Frequency by pass",
            "Participation by pass",
            "Energy balance by pass",
        )
        if convergence_view
        else ("Selected-pass participation",)
    )
    figure = make_subplots(
        rows=len(titles),
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.09 if convergence_view else 0.12,
        subplot_titles=titles,
    )
    ticks: list[str] = []
    tick_labels: list[str] = []
    for bar in view["bars"]:
        keys = [f"{bar['name']}:{key}" for key in bar["keys"]]
        ticks.extend(keys)
        tick_labels.extend(bar["labels"])
        figure.add_trace(
            go.Bar(
                name=bar["name"],
                x=keys,
                y=bar["values"],
                marker_color=[
                    INTERFACE_COLORS.get(kind.removeprefix("surface:"), "#CC79A7")
                    for kind in bar["kinds"]
                ],
                customdata=bar["labels"],
                hovertemplate="%{customdata}<br>p=%{y:.6g}<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=1,
        )
    figure.update_xaxes(
        tickmode="array", tickvals=ticks, ticktext=tick_labels, row=1, col=1
    )
    if convergence_view:
        owner_styles: dict[str, int] = {}
        dashes = ("solid", "dash", "dot", "dashdot")
        symbols = ("circle", "diamond", "square", "cross")
        for line in view["lines"]:
            kind = line["kind"]
            row_number = 2 if kind == "frequency" else 4 if kind == "energy" else 3
            style_index = owner_styles.setdefault(line["identity"], len(owner_styles))
            color = (
                INTERFACE_COLORS.get(line.get("interface_kind"), "#CC79A7")
                if kind == "surface"
                else "#0072B2"
                if kind == "frequency"
                else "#D55E00"
                if kind.startswith("junction")
                else "#E69F00"
            )
            figure.add_trace(
                go.Scatter(
                    name=line["name"],
                    x=line["x"],
                    y=line["y"],
                    mode="lines+markers",
                    connectgaps=False,
                    line={"color": color, "dash": dashes[style_index % len(dashes)]},
                    marker={"symbol": symbols[style_index % len(symbols)]},
                    customdata=[line.get("hover", line["identity"])] * len(line["x"]),
                    meta=line.get("source_metadata", {"identity": line["identity"]}),
                    hovertemplate="%{fullData.name}<br>pass %{x}<br>value %{y:.6g}<br>"
                    "%{customdata}<extra></extra>",
                ),
                row=row_number,
                col=1,
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
        for row_number in (2, 3, 4):
            figure.update_xaxes(
                title_text="Adaptive pass",
                tickmode="array",
                tickvals=list(range(1, view["native_pass"] + 1)),
                row=row_number,
                col=1,
            )
        figure.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
        figure.update_yaxes(title_text="Participation", row=3, col=1)
        figure.update_yaxes(
            title_text="Magnetic+inductive / normalization", row=4, col=1
        )
    if view["unavailable"]:
        figure.add_annotation(
            text="<br>".join(html.escape(message) for message in view["unavailable"]),
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.0,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
        )
    style_figure(figure, theme=theme, height=1320 if convergence_view else 460)
    figure.update_layout(
        margin={"l": 165, "r": 20, "t": 65, "b": 210 if convergence_view else 110}
    )
    return figure


def _show_view(
    result: EprResult, *, mode: int | None, native_pass: int | None
) -> dict[str, Any]:
    if not isinstance(result, EprResult):
        raise TypeError("result must be EprResult")
    selected = [
        row
        for row in result.rows
        if (mode is None or row["mode"] == mode)
        and (native_pass is None or row.get("native_pass") == native_pass)
    ]
    if len(selected) != 1:
        raise ValueError("show_epr requires one exact mode/pass selection")
    row = selected[0]
    if row["status"] != "complete":
        frequency = _usable_frequency(row.get("frequency_hz"))
        return {
            "row": row,
            "complete": False,
            "frequency_text": (
                f"; verified frequency {frequency:.6g} Hz"
                if frequency is not None
                else ""
            ),
        }
    surfaces = row["surface_contributions"]
    domains = row["electric_domains"]
    junctions = row["junctions"]
    baselines = [
        item for item in surfaces if item.get("evaluation_kind") == "unmasked_baseline"
    ]
    known_baselines = [
        item["inverse_q"] for item in baselines if item.get("inverse_q") is not None
    ]
    return {
        "row": row,
        "complete": True,
        "surfaces": surfaces,
        "domains": domains,
        "junctions": junctions,
        "bulk_participations": [
            item["energy_j"] / row["normalization_energy_j"] for item in domains
        ],
        "coverage": {
            "surface_rows": len(surfaces),
            "bindings": sum(
                len(item.get("members", item.get("binding_ids", ())))
                for item in surfaces
            ),
            "bulk_domains": len(domains),
            "junctions": len(junctions),
            "known_baseline_subtotal": (
                f"{sum(known_baselines):.6g}" if known_baselines else "unavailable"
            ),
            "unknown_baseline_losses": sum(
                item.get("inverse_q") is None for item in baselines
            ),
        },
    }


def show_epr(
    result: EprResult,
    *,
    mode: int | None = None,
    native_pass: int | None = None,
    theme: str = "light",
) -> Any:
    """Show one exact pass/mode with separate surface, bulk, and junction axes."""

    checked_theme(theme)
    view = _show_view(result, mode=mode, native_pass=native_pass)
    row = view["row"]
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=3, cols=1, subplot_titles=("Surface", "Bulk", "Junction")
    )
    top_margin = 65
    if view["complete"]:
        surfaces = sorted(
            view["surfaces"], key=lambda item: -float(item["participation"])
        )
        surface_keys = [_surface_key(item) for item in surfaces]
        surface_labels = [_surface_tick_label(item) for item in surfaces]
        figure.add_trace(
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
            ),
            row=1,
            col=1,
        )
        figure.update_yaxes(
            tickmode="array",
            tickvals=surface_keys,
            ticktext=surface_labels,
            autorange="reversed",
            row=1,
            col=1,
        )
        domains = view["domains"]
        figure.add_trace(
            go.Bar(
                x=view["bulk_participations"],
                y=[html.escape(str(item["domain_id"])) for item in domains],
                orientation="h",
                name="bulk p",
            ),
            row=2,
            col=1,
        )
        junctions = view["junctions"]
        figure.add_trace(
            go.Bar(
                x=[item["inductive_participation"] for item in junctions],
                y=[html.escape(str(item["junction_id"])) for item in junctions],
                orientation="h",
                name="junction pL",
            ),
            row=3,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=[item["capacitive_participation"] for item in junctions],
                y=[html.escape(str(item["junction_id"])) for item in junctions],
                orientation="h",
                name="junction pC",
            ),
            row=3,
            col=1,
        )
        coverage = view["coverage"]
        coverage_text = (
            f"Coverage: {coverage['surface_rows']} surface rows / "
            f"{coverage['bindings']} bindings; "
            f"{coverage['bulk_domains']} bulk domains; {coverage['junctions']} junctions. "
            f"Known baseline 1/Q subtotal: {coverage['known_baseline_subtotal']}; "
            f"{coverage['unknown_baseline_losses']} baseline losses unknown. "
            "Sidewalls uncomputed; requested margins are alternatives."
        )
        coverage_lines = textwrap.wrap(coverage_text, width=32, break_long_words=True)
        top_margin = max(240, 135 + 14 * len(coverage_lines))
        figure.add_annotation(
            text="<br>".join(html.escape(line) for line in coverage_lines),
            xref="paper",
            yref="paper",
            x=0,
            y=1,
            yshift=top_margin - 110,
            xanchor="left",
            yanchor="top",
            align="left",
            font={"size": 11},
            showarrow=False,
        )
    else:
        figure.add_annotation(
            text=f"Partial native result{view['frequency_text']}; EPR quantities unavailable",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    title = f"EPR mode {row['mode']}" + (
        f" pass {row['native_pass']}" if "native_pass" in row else " saved field"
    )
    figure.update_layout(
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
            "yref": "container",
        }
        if view["complete"]
        else title,
        showlegend=True,
        meta={
            "surface_identities": [
                _surface_key(item) for item in view.get("surfaces", ())
            ],
            "mode": row["mode"],
            "native_pass": row.get("native_pass"),
        },
    )
    for axis in ("xaxis", "xaxis2", "xaxis3"):
        figure.layout[axis].title = "participation"
    style_figure(
        figure,
        theme=theme,
        height=max(
            900 + top_margin - 65,
            _ranking_height(len(view.get("surfaces", ()))) + 320 + top_margin - 65,
        )
        if view["complete"]
        else 900,
    )
    figure.update_layout(margin={"l": 180, "r": 20, "t": top_margin, "b": 125})
    figure.update_yaxes(automargin=False, tickfont={"size": 12}, row=1, col=1)
    return figure
