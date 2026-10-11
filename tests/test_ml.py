"""Tests for the ML training + inference pipeline."""

from __future__ import annotations

from app.services.prediction_service import predict


def test_training_produces_reasonable_metrics(trained_model_env):
    from ml.inference import load_model
    bundle = load_model(trained_model_env.model_path)
    metrics = bundle.metrics
    # On synthetic data with strong signal, AUC should comfortably beat random.
    assert metrics["roc_auc"] > 0.75
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0


def test_predict_returns_valid_probability(trained_model_env, valid_payload):
    result = predict(valid_payload)
    assert 0.0 <= result.probability <= 1.0
    assert result.label in (0, 1)
    assert result.risk_bucket in ("Low", "Medium", "High")
    assert result.top_factors, "expected top factors to be populated"
    assert result.importance_bars, "expected importance bars to be populated"
    assert result.recommendations, "expected recommendations to be populated"


def test_high_risk_profile_scores_higher_than_low_risk(trained_model_env, valid_payload):
    low_risk = dict(valid_payload)
    low_risk.update(
        tenure_months=60,
        contract_type="Two year",
        monthly_charges=50.0,
        num_support_calls=0,
        tech_support="Yes",
        online_security="Yes",
        payment_method="Credit card",
    )
    high_risk = dict(valid_payload)
    high_risk.update(
        tenure_months=1,
        contract_type="Month-to-month",
        monthly_charges=110.0,
        num_support_calls=8,
        tech_support="No",
        online_security="No",
        payment_method="Electronic check",
    )
    assert predict(high_risk).probability > predict(low_risk).probability


def test_importance_bars_are_normalized(trained_model_env, valid_payload):
    result = predict(valid_payload)
    values = [v for _, v in result.importance_bars]
    assert max(values) == 1.0
    assert all(0.0 <= v <= 1.0 for v in values)
