"""Low-level model IO + prediction primitives.

The service layer wraps these. Tests can call them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelBundle:
    pipeline: Any
    feature_columns: list[str]
    metrics: dict
    trained_on_rows: int
    sklearn_class: str


def load_model(model_path: Path) -> ModelBundle:
    """Load a serialized bundle. Raises FileNotFoundError with an actionable hint."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Train it first with:  python -m ml.train"
        )
    obj = joblib.load(model_path)
    return ModelBundle(**obj)


def predict_proba_one(bundle: ModelBundle, row: pd.DataFrame) -> float:
    """Return P(churn=1) for a one-row DataFrame."""
    proba = bundle.pipeline.predict_proba(row[bundle.feature_columns])[0, 1]
    return float(proba)


def raw_feature_importance(bundle: ModelBundle) -> list[tuple[str, float]]:
    """Return (transformed_feature_name, importance) sorted descending.

    Transformed names include one-hot expansions like 'contract_type_Two year'.
    """
    clf = bundle.pipeline.named_steps["clf"]
    pre = bundle.pipeline.named_steps["preprocess"]
    try:
        names = pre.get_feature_names_out().tolist()
    except AttributeError:
        names = [f"feature_{i}" for i in range(len(clf.feature_importances_))]
    importances = clf.feature_importances_.tolist()
    return sorted(zip(names, importances), key=lambda t: t[1], reverse=True)


def grouped_feature_importance(bundle: ModelBundle) -> list[tuple[str, float]]:
    """Collapse one-hot expansions back into their source feature names.

    e.g. 'contract_type_Two year' + 'contract_type_One year' → 'contract_type'.
    Sorted descending. Only original schema names appear.
    """
    grouped: dict[str, float] = {}
    for name, imp in raw_feature_importance(bundle):
        base = _base_feature(name, bundle.feature_columns)
        grouped[base] = grouped.get(base, 0.0) + float(imp)
    return sorted(grouped.items(), key=lambda t: t[1], reverse=True)


def _base_feature(transformed: str, feature_columns: list[str]) -> str:
    """Find the schema column that transformed name came from."""
    # exact match for numeric (StandardScaler keeps the name)
    if transformed in feature_columns:
        return transformed
    # OneHotEncoder emits '<column>_<category>'
    matches = [col for col in feature_columns if transformed.startswith(col + "_")]
    if matches:
        # longest prefix wins (defensive against overlapping names)
        return max(matches, key=len)
    return transformed


def top_contributing_factors(
    bundle: ModelBundle, row: pd.DataFrame, k: int = 3
) -> list[tuple[str, float]]:
    """Return the k schema-level features with the highest overall importance.

    NOTE: This is a global (dataset-level) attribution, not a per-row SHAP-style
    attribution. It's fast, requires no extra dependency, and communicates
    "which fields typically move the needle." For per-instance attributions,
    add SHAP later and swap this out.
    """
    row_values = row.iloc[0].to_dict()
    ranked = grouped_feature_importance(bundle)
    result: list[tuple[str, float]] = []
    for name, importance in ranked:
        if name in row_values:
            result.append((name, importance))
        if len(result) >= k:
            break
    return result


def normalized_importance_bars(bundle: ModelBundle, k: int = 6) -> list[tuple[str, float]]:
    """Top-k schema features scaled so the largest is 1.0. UI-friendly."""
    ranked = grouped_feature_importance(bundle)[:k]
    if not ranked:
        return []
    max_val = max(v for _, v in ranked) or 1.0
    return [(n, float(np.round(v / max_val, 4))) for n, v in ranked]
