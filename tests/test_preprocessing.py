"""Tests for preprocessing, schema validation, and helpers."""

from __future__ import annotations

import pytest

from app.utils.helpers import (
    ValidationError,
    coerce_and_validate,
    human_recommendations,
    risk_bucket,
)
from app.utils.preprocessing import build_preprocessor, to_dataframe
from app.utils.schema import (
    ALL_FEATURE_NAMES,
    CATEGORICAL_FEATURE_NAMES,
    NUMERIC_FEATURE_NAMES,
    default_input,
    field_by_name,
)


def test_default_input_matches_schema():
    payload = default_input()
    assert set(payload) == set(ALL_FEATURE_NAMES)


def test_field_by_name_raises_for_unknown():
    with pytest.raises(KeyError):
        field_by_name("nope")


def test_to_dataframe_has_expected_columns():
    df = to_dataframe(default_input())
    assert df.shape == (1, len(ALL_FEATURE_NAMES))
    for col in NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES:
        assert col in df.columns


def test_coerce_and_validate_happy_path():
    cleaned = coerce_and_validate(default_input())
    assert isinstance(cleaned["monthly_charges"], float)
    assert cleaned["contract_type"] in field_by_name("contract_type").choices


def test_coerce_missing_field_raises_with_error_map():
    payload = default_input()
    del payload["monthly_charges"]
    with pytest.raises(ValidationError) as exc:
        coerce_and_validate(payload)
    assert "monthly_charges" in exc.value.errors


def test_coerce_bounds_are_enforced():
    payload = default_input()
    payload["monthly_charges"] = 999999  # above max
    with pytest.raises(ValidationError) as exc:
        coerce_and_validate(payload)
    assert "monthly_charges" in exc.value.errors


def test_coerce_rejects_bad_categorical():
    payload = default_input()
    payload["contract_type"] = "Not a real contract"
    with pytest.raises(ValidationError) as exc:
        coerce_and_validate(payload)
    assert "contract_type" in exc.value.errors


def test_coerce_rejects_non_numeric_strings():
    payload = default_input()
    payload["tenure_months"] = "twelve"
    with pytest.raises(ValidationError) as exc:
        coerce_and_validate(payload)
    assert "tenure_months" in exc.value.errors


def test_risk_bucket_thresholds():
    assert risk_bucket(0.9) == "High"
    assert risk_bucket(0.5) == "Medium"
    assert risk_bucket(0.1) == "Low"


def test_recommendations_are_non_empty_per_bucket():
    for bucket in ("Low", "Medium", "High"):
        recs = human_recommendations(bucket, [("tenure_months", 0.3)])
        assert isinstance(recs, list) and len(recs) >= 2


def test_build_preprocessor_fits_and_transforms():
    from ml.dataset import generate_dataset
    df = generate_dataset(n_rows=200, seed=1)
    X = df[list(NUMERIC_FEATURE_NAMES) + list(CATEGORICAL_FEATURE_NAMES)]
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] > len(NUMERIC_FEATURE_NAMES)  # one-hot expands columns
