"""Form component: renders the prediction input form from the schema."""

from __future__ import annotations

from typing import Any

from fasthtml.common import (
    Button,
    Div,
    Form,
    Input,
    Label,
    Option,
    Select,
    Small,
    Span,
)

from app.utils.schema import CATEGORICAL_FIELDS, NUMERIC_FIELDS, FeatureField


def _field_error(field: FeatureField, errors: dict[str, str]):
    msg = errors.get(field.name)
    if not msg:
        return None
    return Small(msg, cls="field-error", role="alert")


def _numeric_field(field: FeatureField, value: Any, errors: dict[str, str]):
    attrs = {
        "type": "number",
        "id": field.name,
        "name": field.name,
        "value": "" if value is None else str(value),
        "aria-describedby": f"{field.name}-help",
    }
    if field.min_value is not None:
        attrs["min"] = str(field.min_value)
    if field.max_value is not None:
        attrs["max"] = str(field.max_value)
    if field.step is not None:
        attrs["step"] = str(field.step)
    if errors.get(field.name):
        attrs["aria-invalid"] = "true"

    return Div(
        Label(field.label, fr=field.name),
        Div(
            Input(**attrs),
            Span(field.unit, cls="input-unit") if field.unit else None,
            cls="input-wrap",
        ),
        Small(field.help_text, id=f"{field.name}-help", cls="field-help") if field.help_text else None,
        _field_error(field, errors),
        cls="form-field",
    )


def _select_field(field: FeatureField, value: Any, errors: dict[str, str]):
    current = value if value is not None else field.default
    options = [
        Option(choice, value=choice, selected=(choice == current))
        for choice in field.choices
    ]
    attrs = {
        "id": field.name,
        "name": field.name,
        "aria-describedby": f"{field.name}-help" if field.help_text else None,
    }
    if errors.get(field.name):
        attrs["aria-invalid"] = "true"
    attrs = {k: v for k, v in attrs.items() if v is not None}

    return Div(
        Label(field.label, fr=field.name),
        Select(*options, **attrs),
        Small(field.help_text, id=f"{field.name}-help", cls="field-help") if field.help_text else None,
        _field_error(field, errors),
        cls="form-field",
    )


def prediction_form(
    values: dict[str, Any] | None = None,
    errors: dict[str, str] | None = None,
) -> Form:
    """Build the full prediction form.

    - values: prefilled inputs (falls back to each field's default)
    - errors: field-level validation messages
    """
    values = values or {}
    errors = errors or {}

    def resolve(field: FeatureField) -> Any:
        return values.get(field.name, field.default)

    numeric_cards = [_numeric_field(f, resolve(f), errors) for f in NUMERIC_FIELDS]
    categorical_cards = [_select_field(f, resolve(f), errors) for f in CATEGORICAL_FIELDS]

    return Form(
        Div(
            Div(*numeric_cards, cls="form-grid"),
            cls="form-section",
        ),
        Div(
            Div(*categorical_cards, cls="form-grid"),
            cls="form-section",
        ),
        Div(
            Button(
                Span("Analyze customer", cls="btn-label"),
                Span(cls="htmx-indicator spinner"),
                type="submit",
                cls="btn btn-primary",
            ),
            Button(
                "Reset",
                type="reset",
                cls="btn btn-secondary",
            ),
            cls="form-actions",
        ),
        method="post",
        action="/predict",
        hx_post="/predict",
        hx_target="#result",
        hx_swap="innerHTML",
        hx_indicator=".spinner",
        cls="prediction-form",
        id="prediction-form",
    )
