"""End-to-end tests for FastHTML routes using Starlette's TestClient."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from app.main import create_app
from app.utils.schema import default_input


@pytest.fixture(scope="module")
def client(trained_model_env):
    app = create_app()
    return TestClient(app)


def test_home_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Churn" in response.text or "churn" in response.text


def test_predict_page_renders_form(client):
    response = client.get("/predict")
    assert response.status_code == 200
    for name in ("tenure_months", "contract_type", "monthly_charges"):
        assert name in response.text


def test_about_page_renders(client):
    response = client.get("/about")
    assert response.status_code == 200
    assert "About" in response.text


def test_dashboard_page_renders(client):
    response = client.get("/dashboard")
    assert response.status_code == 200


def test_post_predict_returns_result_partial(client):
    payload = {k: str(v) for k, v in default_input().items()}
    response = client.post("/predict", data=payload)
    assert response.status_code == 200
    text = response.text
    assert "churn probability" in text
    assert "risk" in text.lower()


def test_post_predict_with_bad_input_shows_validation(client):
    payload = {k: str(v) for k, v in default_input().items()}
    payload["monthly_charges"] = "-9999"
    response = client.post("/predict", data=payload)
    assert response.status_code == 200
    assert "fix the following" in response.text.lower() or "must be" in response.text.lower()


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code in (200, 503)
    body = response.json()
    assert "status" in body


def test_api_predict_json(client):
    payload = default_input()
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_bucket"] in ("Low", "Medium", "High")


def test_api_predict_rejects_unknown_field(client):
    payload = default_input()
    payload["mystery_field"] = 1
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "fields" in body


def test_api_predict_rejects_empty_body(client):
    # Empty object is valid JSON but has no required fields.
    response = client.post("/api/predict", json={})
    assert response.status_code == 422
    body = response.json()
    assert "fields" in body
    # every required field should show up in the error map
    assert "tenure_months" in body["fields"]
