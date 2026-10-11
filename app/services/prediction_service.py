"""Prediction service: turn a validated form payload into a rich result object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.ml_service import get_model
from app.utils.helpers import (
    human_recommendations,
    risk_bucket,
    risk_color,
)
from app.utils.preprocessing import to_dataframe
from ml.inference import (
    normalized_importance_bars,
    predict_proba_one,
    top_contributing_factors,
)


@dataclass
class PredictionResult:
    probability: float                              # P(churn=1)
    label: int                                      # 0 or 1 at 0.5 cutoff
    risk_bucket: str                                # "Low" | "Medium" | "High"
    risk_color: str                                 # hex color
    top_factors: list[tuple[str, float]]            # [(schema_name, importance), ...]
    importance_bars: list[tuple[str, float]]        # normalized to 0..1
    recommendations: list[str]
    inputs: dict[str, Any] = field(default_factory=dict)

    @property
    def probability_pct(self) -> str:
        return f"{self.probability * 100:.1f}%"

    def to_dict(self) -> dict:
        return {
            "probability": self.probability,
            "probability_pct": self.probability_pct,
            "label": self.label,
            "risk_bucket": self.risk_bucket,
            "top_factors": self.top_factors,
            "importance_bars": self.importance_bars,
            "recommendations": self.recommendations,
            "inputs": self.inputs,
        }


def predict(payload: dict[str, Any]) -> PredictionResult:
    """Score a single validated payload. Payload keys must match the schema."""
    bundle = get_model()
    row = to_dataframe(payload)
    proba = predict_proba_one(bundle, row)
    bucket = risk_bucket(proba)
    top = top_contributing_factors(bundle, row, k=3)
    bars = normalized_importance_bars(bundle, k=6)
    return PredictionResult(
        probability=proba,
        label=int(proba >= 0.5),
        risk_bucket=bucket,
        risk_color=risk_color(bucket),
        top_factors=top,
        importance_bars=bars,
        recommendations=human_recommendations(bucket, top),
        inputs=dict(payload),
    )
