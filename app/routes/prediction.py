"""HTMX prediction endpoint — returns a result partial to swap into #result."""

from __future__ import annotations

import logging

from fasthtml.common import Div, H3, P, Section
from starlette.requests import Request

from app.components.cards import error_panel, result_panel
from app.services.prediction_service import predict
from app.utils.helpers import ValidationError, coerce_and_validate

logger = logging.getLogger("churn_predictor")


def register(app) -> None:
    @app.post("/predict")
    async def predict_endpoint(request: Request):
        form = await request.form()
        raw = {k: v for k, v in form.items()}
        try:
            payload = coerce_and_validate(raw)
        except ValidationError as e:
            return _validation_panel(e.errors)
        except Exception:
            logger.exception("Unexpected error during input coercion")
            return error_panel("We couldn't read the form. Please refresh and try again.")

        try:
            result = predict(payload)
        except FileNotFoundError as e:
            logger.error("Model not available: %s", e)
            return error_panel(
                "The prediction model isn't trained yet. "
                "Run  python -m ml.train  and try again."
            )
        except Exception:
            logger.exception("Prediction failed")
            return error_panel("Something went wrong scoring this customer. Please try again.")

        return result_panel(result)


def _validation_panel(errors: dict[str, str]):
    items = [P(f"{field}: {msg}", cls="validation-item") for field, msg in errors.items()]
    return Div(
        Section(
            H3("Please fix the following"),
            *items,
            cls="card error-card",
        ),
        cls="result-grid",
    )
