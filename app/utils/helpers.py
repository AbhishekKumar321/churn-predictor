"""Cross-cutting helpers: validation, coercion, logging setup."""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.utils.schema import ALL_FIELDS, CATEGORICAL_FIELDS, NUMERIC_FIELDS, field_by_name


class ValidationError(ValueError):
    """Raised when a form payload fails validation. `errors` maps field-name → message."""

    def __init__(self, errors: dict[str, str]):
        super().__init__("Invalid form input")
        self.errors = errors


def setup_logging() -> logging.Logger:
    """Configure root logger once and return the app logger."""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    return logging.getLogger("churn_predictor")


def coerce_and_validate(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate & coerce a raw form dict into typed values.

    Raises ValidationError with a per-field error map on failure.
    """
    errors: dict[str, str] = {}
    cleaned: dict[str, Any] = {}

    for field in NUMERIC_FIELDS:
        value = raw.get(field.name)
        if value in (None, ""):
            errors[field.name] = "Required."
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            errors[field.name] = "Must be a number."
            continue
        if field.min_value is not None and num < field.min_value:
            errors[field.name] = f"Must be ≥ {field.min_value}."
            continue
        if field.max_value is not None and num > field.max_value:
            errors[field.name] = f"Must be ≤ {field.max_value}."
            continue
        cleaned[field.name] = num

    for field in CATEGORICAL_FIELDS:
        value = raw.get(field.name)
        if value in (None, ""):
            errors[field.name] = "Required."
            continue
        if value not in field.choices:
            errors[field.name] = f"Must be one of: {', '.join(field.choices)}."
            continue
        cleaned[field.name] = value

    if errors:
        raise ValidationError(errors)
    return cleaned


def risk_bucket(probability: float) -> str:
    """Return a coarse risk label based on configured thresholds."""
    if probability >= settings.high_risk_threshold:
        return "High"
    if probability >= settings.medium_risk_threshold:
        return "Medium"
    return "Low"


def risk_color(bucket: str) -> str:
    """Return a CSS-friendly color name for a risk bucket."""
    return {"High": "#dc3545", "Medium": "#f59f00", "Low": "#2f9e44"}.get(bucket, "#6c757d")


def human_recommendations(bucket: str, top_factors: list[tuple[str, float]]) -> list[str]:
    """Return retention actions appropriate for the risk bucket."""
    factor_labels = [field_by_name(name).label for name, _ in top_factors[:3]]
    if bucket == "High":
        return [
            "Reach out within 48 hours with a retention offer.",
            f"Focus the conversation on: {', '.join(factor_labels) if factor_labels else 'top drivers'}.",
            "Consider offering a contract upgrade with a discount.",
        ]
    if bucket == "Medium":
        return [
            "Add to the next monthly retention campaign.",
            "Send a satisfaction survey to surface latent issues.",
            f"Monitor: {', '.join(factor_labels) if factor_labels else 'top drivers'}.",
        ]
    return [
        "No immediate action required.",
        "Include in the loyalty program touchpoints.",
        "Re-score if any account details change.",
    ]


def assert_known_fields(payload: dict[str, Any]) -> None:
    """Reject payloads with unexpected keys to prevent silent field drift."""
    known = {f.name for f in ALL_FIELDS}
    unexpected = set(payload) - known
    if unexpected:
        raise ValidationError({name: "Unknown field." for name in unexpected})
