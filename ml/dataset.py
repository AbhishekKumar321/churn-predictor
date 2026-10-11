"""Synthetic dataset generation.

We generate a realistic churn dataset in-repo so the project runs with zero
manual downloads. The generator embeds sensible priors (short-tenure, high-bill,
month-to-month, high support-call customers churn more often) so a trained
model learns non-trivial patterns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.utils.schema import (
    CATEGORICAL_FIELDS,
    NUMERIC_FIELDS,
    TARGET_COLUMN,
    field_by_name,
)


def generate_dataset(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic churn dataset with realistic feature relationships."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 72, size=n_rows)
    monthly_charges = np.clip(rng.normal(70, 30, size=n_rows), 15, 200)
    total_charges = np.clip(
        monthly_charges * np.maximum(tenure, 1) * rng.uniform(0.8, 1.15, size=n_rows),
        0,
        None,
    )
    num_support_calls = rng.poisson(lam=1.5, size=n_rows).clip(0, 30)

    contract_type = rng.choice(
        field_by_name("contract_type").choices,
        size=n_rows,
        p=[0.55, 0.25, 0.20],
    )
    payment_method = rng.choice(
        field_by_name("payment_method").choices,
        size=n_rows,
        p=[0.35, 0.20, 0.25, 0.20],
    )
    internet_service = rng.choice(
        field_by_name("internet_service").choices,
        size=n_rows,
        p=[0.35, 0.5, 0.15],
    )
    tech_support = rng.choice(("Yes", "No"), size=n_rows, p=[0.35, 0.65])
    online_security = rng.choice(("Yes", "No"), size=n_rows, p=[0.30, 0.70])
    paperless_billing = rng.choice(("Yes", "No"), size=n_rows, p=[0.6, 0.4])
    senior_citizen = rng.choice(("Yes", "No"), size=n_rows, p=[0.16, 0.84])

    logit = (
        -1.2
        + (-0.04) * tenure
        + 0.015 * (monthly_charges - 70)
        + 0.35 * num_support_calls
        + np.where(contract_type == "Month-to-month", 1.2, 0.0)
        + np.where(contract_type == "Two year", -0.8, 0.0)
        + np.where(payment_method == "Electronic check", 0.6, 0.0)
        + np.where(internet_service == "Fiber optic", 0.4, 0.0)
        + np.where(tech_support == "No", 0.3, 0.0)
        + np.where(online_security == "No", 0.25, 0.0)
        + np.where(paperless_billing == "Yes", 0.15, 0.0)
        + np.where(senior_citizen == "Yes", 0.4, 0.0)
        + rng.normal(0, 0.3, size=n_rows)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, size=n_rows) < prob).astype(int)

    df = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": np.round(monthly_charges, 2),
            "total_charges": np.round(total_charges, 2),
            "num_support_calls": num_support_calls,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "online_security": online_security,
            "paperless_billing": paperless_billing,
            "senior_citizen": senior_citizen,
            TARGET_COLUMN: churn,
        }
    )

    expected = {f.name for f in NUMERIC_FIELDS + CATEGORICAL_FIELDS} | {TARGET_COLUMN}
    assert set(df.columns) == expected, "Generated dataset columns drifted from schema."
    return df


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_or_generate(path: Path, n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Load the dataset from disk, generating it if it's missing."""
    if path.exists():
        return pd.read_csv(path)
    df = generate_dataset(n_rows=n_rows, seed=seed)
    save_dataset(df, path)
    return df
