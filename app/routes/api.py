"""JSON API endpoints — programmatic prediction + health check."""

from __future__ import annotations

import logging

from starlette.requests import Request
from starlette.responses import JSONResponse

from app.services.ml_service import get_model
from app.services.prediction_service import predict
from app.utils.helpers import ValidationError, assert_known_fields, coerce_and_validate

logger = logging.getLogger("churn_predictor")


def register(app) -> None:
    @app.get("/api/health")
    def health():
        try:
            bundle = get_model()
            return JSONResponse(
                {
                    "status": "ok",
                    "model_loaded": True,
                    "trained_on_rows": bundle.trained_on_rows,
                    "metrics": bundle.metrics,
                }
            )
        except FileNotFoundError as e:
            return JSONResponse(
                {"status": "degraded", "model_loaded": False, "detail": str(e)},
                status_code=503,
            )

    @app.post("/api/predict")
    async def api_predict(request: Request):
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

        if not isinstance(payload, dict):
            return JSONResponse({"error": "Body must be a JSON object."}, status_code=400)

        try:
            assert_known_fields(payload)
            cleaned = coerce_and_validate(payload)
        except ValidationError as e:
            return JSONResponse({"error": "Validation failed", "fields": e.errors}, status_code=422)

        try:
            result = predict(cleaned)
        except FileNotFoundError as e:
            return JSONResponse({"error": str(e)}, status_code=503)
        except Exception:
            logger.exception("API prediction failed")
            return JSONResponse({"error": "Prediction failed."}, status_code=500)

        return JSONResponse(result.to_dict())
