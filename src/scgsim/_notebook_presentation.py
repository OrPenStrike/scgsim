"""Small, scoped HTML and Figure styling for SCGSim notebook reports.

The caller owns scientific projections. These helpers only format detached
values and style newly created figures; no global Plotly state is changed.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

ReportTheme = Literal["light", "dark"]

_PLOT_COLORS = ("#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2", "#D55E00")
INTERFACE_COLORS = {"MA": "#56B4E9", "MS": "#E69F00", "SA": "#009E73"}

_THEMES = {
    "light": {
        "paper": "#ffffff", "plot": "#f6f8fa", "text": "#1f2328",
        "muted": "#59636e", "border": "#d0d7de", "grid": "rgba(31,35,40,.12)",
    },
    "dark": {
        "paper": "#0d1117", "plot": "#161b22", "text": "#e6edf3",
        "muted": "#8b949e", "border": "#30363d", "grid": "rgba(139,148,158,.22)",
    },
}


def checked_theme(theme: str) -> ReportTheme:
    """Validate one public presentation choice before rendering or reading data."""

    if not isinstance(theme, str) or theme not in _THEMES:
        raise ValueError("theme must be 'light' or 'dark'")
    return theme  # type: ignore[return-value]


def display_text(value: Any, *, unknown: str = "unknown") -> str:
    """Keep an absent value visibly different from numeric zero."""

    if value is None:
        return unknown
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def selection_text(value: Any, *, absent: bool = False) -> str:
    """None/all and empty/none apply only to configured selections."""

    if absent:
        return "not requested"
    if value is None:
        return "all"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and not value:
        return "none"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ", ".join(display_text(item) for item in value)
    return display_text(value)


def section(title: str, body: str, *, theme: str | None = None, class_name: str = "") -> str:
    """Wrap trusted markup in one locally styled notebook section."""

    if theme is None:
        style = (
            "color-scheme:light dark;color:var(--vscode-editor-foreground,CanvasText);"
            "background:var(--vscode-editor-background,Canvas);"
            "border:1px solid var(--vscode-panel-border,GrayText)"
        )
    else:
        tokens = _THEMES[checked_theme(theme)]
        style = (
            f"color:{tokens['text']};background:{tokens['paper']};"
            f"border:1px solid {tokens['border']}"
        )
    return (
        f'<section class="scgsim-notebook {html.escape(class_name, quote=True)}" '
        f'style="{style};border-radius:8px;padding:16px;margin:10px 0;'
        'font-family:system-ui,sans-serif;font-size:15px">'
        f'<h3 style="font-size:18px;margin:0 0 12px">{html.escape(title)}</h3>'
        f"{body}</section>"
    )


def cards(items: Sequence[tuple[str, Any]]) -> str:
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 12px">'
        + "".join(
            '<div style="border:1px solid var(--vscode-panel-border,#8b949e);'
            'border-radius:8px;padding:8px 12px;min-width:0;flex:1 1 110px">'
            f'<div style="font-size:14px;opacity:.75">{html.escape(str(label))}</div>'
            f'<div style="font-size:16px;font-weight:600;overflow-wrap:anywhere">'
            f'{html.escape(display_text(value))}</div>'
            "</div>"
            for label, value in items
        )
        + "</div>"
    )


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a compact escaped data table; cells are never trusted markup."""

    header = "".join(f"<th style='text-align:left;padding:6px'>{html.escape(str(item))}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(
            f"<td style='padding:6px;border-top:1px solid rgba(128,128,128,.25)'>"
            f"{html.escape(display_text(value))}</td>"
            for value in row
        ) + "</tr>"
        for row in rows
    )
    return (
        '<div style="overflow-x:auto"><table style="border-collapse:collapse;'
        f'width:100%;font-size:14px"><thead><tr>{header}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def details(title: str, body: str, *, opened: bool = False) -> str:
    return (
        f"<details{' open' if opened else ''}>"
        f"<summary>{html.escape(title)}</summary>{body}</details>"
    )


def json_details(title: str, data: Mapping[str, Any], *, opened: bool = False) -> str:
    return details(
        title,
        "<pre style='white-space:pre-wrap;overflow-wrap:anywhere'>"
        + html.escape(json.dumps(data, indent=2, sort_keys=True, default=str))
        + "</pre>",
        opened=opened,
    )


def style_figure(figure: Any, *, theme: str, height: int = 460) -> Any:
    """Apply per-Figure Plotly colors and typography without setting a template."""

    tokens = _THEMES[checked_theme(theme)]
    figure.update_layout(
        height=height, paper_bgcolor=tokens["paper"], plot_bgcolor=tokens["plot"],
        font={"family": "system-ui, sans-serif", "size": 16, "color": tokens["text"]},
        title_font={"size": 18, "color": tokens["text"]},
        colorway=list(_PLOT_COLORS),
        legend={
            "orientation": "h", "x": 0, "xanchor": "left", "y": -0.22,
            "yanchor": "top", "font": {"size": 14, "color": tokens["text"]},
        },
        hoverlabel={"bgcolor": tokens["paper"], "font": {"size": 14, "color": tokens["text"]}},
        margin={"l": 95, "r": 25, "t": 65, "b": 115},
    )
    figure.update_xaxes(
        title_font={"size": 16}, tickfont={"size": 14, "color": tokens["muted"]},
        gridcolor=tokens["grid"], zerolinecolor=tokens["grid"],
    )
    figure.update_yaxes(
        title_font={"size": 16}, tickfont={"size": 14, "color": tokens["muted"]},
        gridcolor=tokens["grid"], zerolinecolor=tokens["grid"],
    )
    return figure
