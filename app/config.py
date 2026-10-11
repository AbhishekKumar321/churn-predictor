"""Central configuration loaded from environment variables.

All tunable paths, thresholds, and runtime settings live here so no other
module has to reach for os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings.

    All values can be overridden through environment variables (see .env.example).
    """

    app_name: str = os.getenv("APP_NAME", "Churn Predictor")
    environment: str = os.getenv("APP_ENV", "development")
    debug: bool = _env_bool("DEBUG", True)

    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = _env_int("PORT", 5001)

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")

    model_filename: str = os.getenv("MODEL_FILENAME", "churn_model.joblib")
    dataset_filename: str = os.getenv("DATASET_FILENAME", "churn_dataset.csv")

    random_seed: int = _env_int("RANDOM_SEED", 42)
    test_size: float = _env_float("TEST_SIZE", 0.2)

    high_risk_threshold: float = _env_float("HIGH_RISK_THRESHOLD", 0.66)
    medium_risk_threshold: float = _env_float("MEDIUM_RISK_THRESHOLD", 0.33)

    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / "raw" / self.dataset_filename


settings = Settings()
