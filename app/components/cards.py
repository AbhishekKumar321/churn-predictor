"""Result cards: risk verdict, factors, recommendations, empty/error states."""

from __future__ import annotations

from fasthtml.common import (
    H2,
    H3,
    H4,
    Div,
    Li,
    P,
    Section,
    Span,
    Strong,
    Ul,
)

from app.services.prediction_service import PredictionResult
from app.utils.schema import field_by_name


def result_panel(result: PredictionResult) -> Div:
    """The primary results panel — verdict + factors + recommendations."""
    return Div(
        _verdict_card(result),
        _factors_card(result),
        _recommendations_card(result),
        cls="result-grid",
    )


def _verdict_card(result: PredictionResult) -> Section:
    pct = result.probability_pct
    bucket = result.risk_bucket
    return Section(
        H2("Prediction"),
        Div(
            Div(
                Span(pct, cls="probability-value"),
                Span("churn probability", cls="probability-caption"),
                cls="probability-block",
                style=f"--accent: {result.risk_color};",
            ),
            Div(
                Span(f"{bucket} risk", cls="risk-badge", style=f"background: {result.risk_color};"),
                P(
                    f"The model estimates a {pct} chance this customer will churn.",
                    cls="verdict-desc",
                ),
                cls="verdict-text",
            ),
            cls="verdict-body",
        ),
        cls="card verdict-card",
    )


def _factors_card(result: PredictionResult) -> Section:
    """Feature-importance bars for the top-6 drivers."""
    bars = []
    for name, value in result.importance_bars:
        label = field_by_name(name).label if _in_schema(name) else name
        pct = int(round(value * 100))
        bars.append(
            Div(
                Div(
                    Span(label, cls="bar-label"),
                    Span(f"{pct}%", cls="bar-value"),
                    cls="bar-header",
                ),
                Div(
                    Div(cls="bar-fill", style=f"width: {pct}%;"),
                    cls="bar-track",
                ),
                cls="importance-row",
            )
        )

    body = bars or [P("No importance data available.", cls="empty-state")]
    return Section(
        H3("Top drivers (global)"),
        P(
            "These are the features that most influence the model's decisions across all customers.",
            cls="card-subtext",
        ),
        Div(*body, cls="importance-list"),
        cls="card factors-card",
    )


def _recommendations_card(result: PredictionResult) -> Section:
    items = [Li(text) for text in result.recommendations]
    return Section(
        H3("Recommended next steps"),
        Ul(*items, cls="recommendations"),
        cls="card recommendations-card",
    )


def error_panel(message: str) -> Div:
    return Div(
        Section(
            H3("Something went wrong"),
            P(message, cls="error-text"),
            cls="card error-card",
        ),
        cls="result-grid",
    )


def empty_result() -> Div:
    return Div(
        Section(
            H4("Waiting for input"),
            P(
                "Fill in the customer details on the left and click "
                "\"Analyze customer\" to see the churn risk breakdown.",
                cls="empty-state",
            ),
            cls="card empty-card",
        ),
        cls="result-grid",
    )


def kpi_card(label: str, value: str, tone: str = "default") -> Section:
    return Section(
        P(label, cls="kpi-label"),
        Strong(value, cls="kpi-value"),
        cls=f"card kpi-card kpi-{tone}",
    )


def _in_schema(name: str) -> bool:
    try:
        field_by_name(name)
        return True
    except KeyError:
        return False
