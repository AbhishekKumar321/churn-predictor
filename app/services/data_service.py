"""Data service: read-only access to the training dataset for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.config import settings
from app.utils.schema import TARGET_COLUMN


@dataclass
class DatasetStats:
    n_rows: int
    n_churned: int
    churn_rate: float
    contract_breakdown: list[tuple[str, int, float]]        # (contract, churn_count, churn_rate)
    tenure_buckets: list[tuple[str, int, float]]            # (bucket, count, churn_rate)


@lru_cache(maxsize=1)
def dataset_stats(path: Path | None = None) -> DatasetStats:
    """Compute summary statistics for the dashboard. Cached for the process."""
    csv_path = path or settings.dataset_path
    if not csv_path.exists():
        return DatasetStats(0, 0, 0.0, [], [])
    df = pd.read_csv(csv_path)

    n_rows = len(df)
    n_churned = int(df[TARGET_COLUMN].sum())
    churn_rate = n_churned / n_rows if n_rows else 0.0

    contract_rows: list[tuple[str, int, float]] = []
    if "contract_type" in df.columns:
        grouped = df.groupby("contract_type")[TARGET_COLUMN].agg(["sum", "count"])
        for contract, row in grouped.iterrows():
            rate = float(row["sum"]) / float(row["count"]) if row["count"] else 0.0
            contract_rows.append((str(contract), int(row["sum"]), rate))
        contract_rows.sort(key=lambda t: t[2], reverse=True)

    tenure_rows: list[tuple[str, int, float]] = []
    if "tenure_months" in df.columns:
        bins = [-1, 6, 12, 24, 48, 72, 120]
        labels = ["0-6", "7-12", "13-24", "25-48", "49-72", "73+"]
        buckets = pd.cut(df["tenure_months"], bins=bins, labels=labels)
        grouped = df.groupby(buckets, observed=True)[TARGET_COLUMN].agg(["sum", "count"])
        for label, row in grouped.iterrows():
            rate = float(row["sum"]) / float(row["count"]) if row["count"] else 0.0
            tenure_rows.append((str(label), int(row["count"]), rate))

    return DatasetStats(
        n_rows=n_rows,
        n_churned=n_churned,
        churn_rate=churn_rate,
        contract_breakdown=contract_rows,
        tenure_buckets=tenure_rows,
    )


def clear_stats_cache() -> None:
    dataset_stats.cache_clear()
