"""Offline Palace EPR projection and Plotly rendering."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def _plain_view(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_view(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_view(item) for item in value]
    return value


def _show_view(result: Any, *, mode: int | None) -> dict[str, Any]:
    selected = [row for row in result.rows if mode is None or row["mode"] == mode]
    if len(selected) != 1:
        raise ValueError("show_epr requires one exact mode")
    row = selected[0]
    surface = row["surface"]
    baselines = [
        item for item in surface if item["evaluation_kind"] == "unmasked_baseline"
    ]
    known_baselines = [
        item["inverse_q"] for item in baselines if item["inverse_q"] is not None
    ]
    return {
        "row": row,
        "coverage": {
            "surface_rows": len(surface),
            "source_surfaces": len({item["surface_id"] for item in surface}),
            "bulk_domains": len(row["bulk"]),
            "ports": len(row["junction"]),
            "known_baseline_subtotal": (
                f"{sum(known_baselines):.6g}" if known_baselines else "unavailable"
            ),
            "unknown_baseline_losses": sum(
                item["inverse_q"] is None for item in baselines
            ),
        },
    }


def show_epr(result: Any, *, mode: int | None = None) -> Any:
    """Plot one mode with separate surface, bulk, and signed Palace port axes."""

    view = _show_view(result, mode=mode)
    row = view["row"]
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=3, cols=1, subplot_titles=("Surface", "Bulk", "Palace port")
    )
    surface = row["surface"]
    figure.add_trace(
        go.Bar(
            x=[item["participation"] for item in surface],
            y=[
                f"{','.join(item['owner_semantic_ids'])} / {item['interface_kind']} / {item['evaluation_kind']} {item['margin_um']:g} µm"
                for item in surface
            ],
            customdata=[
                json.dumps(
                    _plain_view(
                        {
                            "group_id": item["group_id"],
                            "index": item["index"],
                            "surface_id": item["surface_id"],
                            "field_side": item["field_side"],
                            "members": item["members"],
                            "inverse_q": item["inverse_q"],
                            "assumptions": item["assumptions"],
                            "original_assumptions": item["original_assumptions"],
                            "recorded_native_assumptions": item.get(
                                "recorded_native_assumptions"
                            ),
                            "loss_provenance": item.get("loss_provenance"),
                            "assumption_update": item.get("assumption_update"),
                        }
                    ),
                    sort_keys=True,
                )
                for item in surface
            ],
            hovertemplate="%{y}<br>p=%{x:.6g}<br>%{customdata}<extra></extra>",
            orientation="h",
            name="surface p",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=[item["participation"] for item in row["bulk"]],
            y=[item["domain_id"] for item in row["bulk"]],
            orientation="h",
            name="bulk p",
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=[item["participation"] for item in row["junction"]],
            y=[item["port_name"] for item in row["junction"]],
            orientation="h",
            name="signed port p",
        ),
        row=3,
        col=1,
    )
    coverage = view["coverage"]
    figure.add_annotation(
        text=(
            f"Coverage: {coverage['surface_rows']} surface rows / "
            f"{coverage['source_surfaces']} source surfaces; "
            f"{coverage['bulk_domains']} bulk domains; {coverage['ports']} ports. "
            f"Known baseline 1/Q subtotal: {coverage['known_baseline_subtotal']}; "
            f"{coverage['unknown_baseline_losses']} baseline losses unknown. "
            "Sidewalls uncomputed when unselected; requested margins are alternatives."
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=1.08,
        showarrow=False,
    )
    figure.update_layout(title=f"Palace EPR mode {row['mode']}", height=900)
    return figure
