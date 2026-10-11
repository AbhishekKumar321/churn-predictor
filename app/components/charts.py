"""Lightweight, dependency-free chart components (rendered as HTML/CSS bars)."""

from __future__ import annotations

from fasthtml.common import Div, H3, P, Section, Span


def horizontal_bars(
    title: str,
    rows: list[tuple[str, float]],
    subtitle: str = "",
    fmt: str = "pct",
) -> Section:
    """Render a horizontal bar chart from label → value pairs.

    fmt='pct' formats as 0-100%, fmt='int' formats as integer, fmt='raw' as-is.
    """
    if not rows:
        return Section(
            H3(title),
            P("No data available.", cls="empty-state"),
            cls="card chart-card",
        )
    max_val = max(v for _, v in rows) or 1.0
    bars = []
    for label, value in rows:
        width = int(round((value / max_val) * 100))
        display = _format(value, fmt)
        bars.append(
            Div(
                Div(
                    Span(label, cls="bar-label"),
                    Span(display, cls="bar-value"),
                    cls="bar-header",
                ),
                Div(Div(cls="bar-fill", style=f"width: {width}%;"), cls="bar-track"),
                cls="importance-row",
            )
        )
    header = [H3(title)]
    if subtitle:
        header.append(P(subtitle, cls="card-subtext"))
    return Section(*header, Div(*bars, cls="importance-list"), cls="card chart-card")


def _format(value: float, fmt: str) -> str:
    if fmt == "pct":
        return f"{value * 100:.1f}%"
    if fmt == "int":
        return f"{int(value):,}"
    return str(value)
