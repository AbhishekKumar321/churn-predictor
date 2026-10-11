"""ML service: single owner of the loaded model.

Loads the model bundle lazily on first use and caches it for the process
lifetime. All prediction paths must go through get_model() so we never load
the same pickle twice.
"""

from __future__ import annotations

import threading
from functools import lru_cache

from app.config import settings
from app.utils.helpers import setup_logging
from ml.inference import ModelBundle, load_model

logger = setup_logging()

_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_model() -> ModelBundle:
    """Return the cached model bundle. Loads it on first call.

    Thread-safe: lru_cache handles the memoization race, and joblib.load
    is safe under GIL contention because the RandomForest is read-only after fit.
    """
    with _LOCK:
        logger.info("Loading model from %s", settings.model_path)
        bundle = load_model(settings.model_path)
        logger.info(
            "Model loaded (%s trained on %d rows, AUC=%.3f)",
            bundle.sklearn_class,
            bundle.trained_on_rows,
            bundle.metrics.get("roc_auc", float("nan")),
        )
        return bundle


def clear_cache() -> None:
    """Drop the cached model — used by tests that swap the model file."""
    get_model.cache_clear()
