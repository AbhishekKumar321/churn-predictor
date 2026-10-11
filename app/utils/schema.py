"""Single source of truth for the feature schema.

Every module that touches customer data — training, preprocessing, forms,
inference — imports the field definitions from here. If a field is added
or renamed, only this file needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FieldKind = Literal["numeric", "categorical"]


@dataclass(frozen=True)
class FeatureField:
    """Describes a single input feature."""

    name: str
    label: str
    kind: FieldKind
    help_text: str = ""
    # numeric-only
    min_value: float | None = None
    max_value: float | None = None
    default: float | str | None = None
    step: float | None = None
    unit: str = ""
    # categorical-only
    choices: tuple[str, ...] = ()


NUMERIC_FIELDS: tuple[FeatureField, ...] = (
    FeatureField(
        name="tenure_months",
        label="Tenure (months)",
        kind="numeric",
        help_text="Number of months the customer has been with the company.",
        min_value=0,
        max_value=120,
        default=12,
        step=1,
        unit="months",
    ),
    FeatureField(
        name="monthly_charges",
        label="Monthly charges",
        kind="numeric",
        help_text="Total monthly bill in USD.",
        min_value=0,
        max_value=500,
        default=70.0,
        step=0.5,
        unit="USD",
    ),
    FeatureField(
        name="total_charges",
        label="Total charges",
        kind="numeric",
        help_text="Lifetime revenue from this customer in USD.",
        min_value=0,
        max_value=20000,
        default=840.0,
        step=1.0,
        unit="USD",
    ),
    FeatureField(
        name="num_support_calls",
        label="Support calls (last 6 months)",
        kind="numeric",
        help_text="Number of times the customer contacted support recently.",
        min_value=0,
        max_value=50,
        default=1,
        step=1,
    ),
)

CATEGORICAL_FIELDS: tuple[FeatureField, ...] = (
    FeatureField(
        name="contract_type",
        label="Contract type",
        kind="categorical",
        help_text="Length of the current contract.",
        choices=("Month-to-month", "One year", "Two year"),
        default="Month-to-month",
    ),
    FeatureField(
        name="payment_method",
        label="Payment method",
        kind="categorical",
        choices=(
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card",
        ),
        default="Electronic check",
    ),
    FeatureField(
        name="internet_service",
        label="Internet service",
        kind="categorical",
        choices=("DSL", "Fiber optic", "No"),
        default="Fiber optic",
    ),
    FeatureField(
        name="tech_support",
        label="Tech support add-on",
        kind="categorical",
        choices=("Yes", "No"),
        default="No",
    ),
    FeatureField(
        name="online_security",
        label="Online security add-on",
        kind="categorical",
        choices=("Yes", "No"),
        default="No",
    ),
    FeatureField(
        name="paperless_billing",
        label="Paperless billing",
        kind="categorical",
        choices=("Yes", "No"),
        default="Yes",
    ),
    FeatureField(
        name="senior_citizen",
        label="Senior citizen",
        kind="categorical",
        choices=("Yes", "No"),
        default="No",
    ),
)

ALL_FIELDS: tuple[FeatureField, ...] = NUMERIC_FIELDS + CATEGORICAL_FIELDS

NUMERIC_FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in NUMERIC_FIELDS)
CATEGORICAL_FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in CATEGORICAL_FIELDS)
ALL_FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in ALL_FIELDS)

TARGET_COLUMN: str = "churn"


def field_by_name(name: str) -> FeatureField:
    """Return the FeatureField whose name matches, or raise KeyError."""
    for f in ALL_FIELDS:
        if f.name == name:
            return f
    raise KeyError(f"Unknown feature: {name}")


def default_input() -> dict[str, float | str]:
    """Return a fully-populated example input using each field's default."""
    return {f.name: f.default for f in ALL_FIELDS if f.default is not None}
