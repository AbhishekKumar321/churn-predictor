"""Pytest fixtures.

Trains the model once per test session (into a tmp path) and points every
service at it, so tests never touch the real /models directory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Add project root to sys.path so `import app...` works when tests run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def trained_model_env(tmp_path_factory):
    """Train a small model into a tmp directory and expose env vars."""
    tmp_dir = tmp_path_factory.mktemp("churn_model")
    data_dir = tmp_dir / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MODEL_FILENAME"] = "test_model.joblib"

    # Force config to reload with these overrides
    from importlib import reload
    from app import config as cfg_module
    reload(cfg_module)

    # Point paths at the tmp dir
    cfg_module.settings.__dict__["models_dir"] = tmp_dir / "models"
    cfg_module.settings.__dict__["data_dir"] = tmp_dir / "data"
    (tmp_dir / "models").mkdir(parents=True, exist_ok=True)

    from ml.dataset import generate_dataset, save_dataset
    from ml.train import train_and_save

    # Small dataset for speed
    df = generate_dataset(n_rows=800, seed=7)
    save_dataset(df, cfg_module.settings.dataset_path)
    train_and_save(n_rows=800)

    # Reset cached model in service layer
    from app.services import ml_service
    ml_service.clear_cache()
    from app.services import data_service
    data_service.clear_stats_cache()

    yield cfg_module.settings


@pytest.fixture
def valid_payload():
    from app.utils.schema import default_input
    return default_input()
