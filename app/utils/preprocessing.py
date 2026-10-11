"""Preprocessing pipeline factory.

Builds a scikit-learn ColumnTransformer that scales numeric features and
one-hot-encodes categorical features. Used by both training and inference,
so predictions see exactly the same transformations as training data.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.utils.schema import (
    CATEGORICAL_FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
)


def build_preprocessor() -> ColumnTransformer:
    """Return a fresh preprocessor. Fit it on training data before use."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(NUMERIC_FEATURE_NAMES)),
            ("cat", categorical_pipeline, list(CATEGORICAL_FEATURE_NAMES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def to_dataframe(payload: dict) -> pd.DataFrame:
    """Convert a single form payload dict into a one-row DataFrame with all expected columns.

    Missing keys become NaN so the imputer can fill them.
    """
    row = {name: payload.get(name) for name in NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES}
    return pd.DataFrame([row])
