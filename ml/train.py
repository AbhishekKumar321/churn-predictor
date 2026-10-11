"""Train the churn model and serialize it to disk.

Algorithm choice — RandomForestClassifier:
- Handles mixed numeric/categorical (after one-hot) without gradient tuning.
- Native, calibrated feature importances → easy explainability in the UI.
- Robust to unscaled numerics and mild class imbalance (with class_weight).
- Fast to train on tens of thousands of rows without a GPU.

Run:
    python -m ml.train
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Allow running as a script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.utils.helpers import setup_logging  # noqa: E402
from app.utils.preprocessing import build_preprocessor  # noqa: E402
from app.utils.schema import (  # noqa: E402
    CATEGORICAL_FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    TARGET_COLUMN,
)
from ml.dataset import load_or_generate  # noqa: E402
from ml.evaluate import evaluate  # noqa: E402

logger = setup_logging()


def build_pipeline(random_state: int) -> Pipeline:
    """Compose preprocessor + classifier into a single fit-once pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_and_save(n_rows: int = 5000) -> dict:
    """End-to-end: load data → split → fit → evaluate → save. Returns metrics."""
    logger.info("Loading dataset from %s", settings.dataset_path)
    df = load_or_generate(settings.dataset_path, n_rows=n_rows, seed=settings.random_seed)

    feature_cols = list(NUMERIC_FEATURE_NAMES) + list(CATEGORICAL_FEATURE_NAMES)
    X = df[feature_cols]
    y = df[TARGET_COLUMN].astype(int)

    logger.info("Splitting: n=%d, test_size=%.2f", len(df), settings.test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_seed,
        stratify=y,
    )

    pipeline = build_pipeline(random_state=settings.random_seed)

    start = time.perf_counter()
    logger.info("Training %s", pipeline.named_steps["clf"].__class__.__name__)
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    logger.info("Training complete in %.2fs", fit_seconds)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    report = evaluate(y_test.to_numpy(), y_pred, y_proba)
    logger.info("\n%s", report.pretty())

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": feature_cols,
            "metrics": report.to_dict(),
            "trained_on_rows": len(X_train),
            "sklearn_class": "RandomForestClassifier",
        },
        settings.model_path,
    )
    logger.info("Model saved to %s", settings.model_path)

    metrics_path = settings.models_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report.to_dict(), indent=2))
    logger.info("Metrics written to %s", metrics_path)

    return report.to_dict()


if __name__ == "__main__":
    train_and_save()
