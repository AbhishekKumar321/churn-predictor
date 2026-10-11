# Churn Predictor

A production-quality AI/ML web application built entirely in Python and FastHTML.
It predicts whether a subscription customer is likely to churn, explains the
main drivers behind each prediction, and suggests retention actions.

## Features

- Server-rendered UI in Python via **FastHTML** — no separate JS build.
- **HTMX**-powered form: predictions swap into the page without a full reload.
- **RandomForestClassifier** trained end-to-end from a reproducible synthetic dataset.
- Feature-importance visualisation for explainability.
- Dashboard with dataset-level stats (churn rate by contract / by tenure bucket).
- JSON API (`GET /api/health`, `POST /api/predict`) for programmatic use.
- Fully typed schema + strict form validation.
- Test suite covering preprocessing, ML pipeline, and every route.
- Config via `.env` — no secrets in source.

## Architecture

```
UI (FastHTML components)  ─┐
                           │
Routes (pages/prediction/api) ──► Services (ml_service, prediction_service, data_service)
                                    │
                                    ▼
                          ML pipeline (train / evaluate / inference)
                                    │
                                    ▼
                          Model artifact (joblib) + Dataset (CSV)
```

Layers are strictly one-way: UI depends on services, services depend on the ML pipeline, the ML pipeline depends on the schema. Nothing lower depends on anything above it.

## Technology stack

| Layer          | Choice                              |
| -------------- | ----------------------------------- |
| Language       | Python 3.11+                        |
| Web framework  | FastHTML (server-rendered) + HTMX   |
| ML             | scikit-learn (RandomForestClassifier)|
| Data           | pandas, numpy                       |
| Serialization  | joblib                              |
| Config         | python-dotenv                       |
| Tests          | pytest, httpx (via Starlette TestClient) |

## Project structure

```
churn_predictor/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastHTML app factory + entrypoint
│   ├── config.py                  # Settings loaded from .env
│   ├── routes/
│   │   ├── pages.py               # Home, Predict, Dashboard, About
│   │   ├── prediction.py          # HTMX POST /predict
│   │   └── api.py                 # /api/health, /api/predict
│   ├── components/
│   │   ├── layout.py              # navbar, footer, page shell
│   │   ├── forms.py               # schema-driven form renderer
│   │   ├── cards.py               # verdict / factors / recommendations
│   │   └── charts.py              # HTML/CSS horizontal bars
│   ├── services/
│   │   ├── ml_service.py          # cached model loader
│   │   ├── prediction_service.py  # form → PredictionResult
│   │   └── data_service.py        # dataset stats for dashboard
│   └── utils/
│       ├── schema.py              # Single source of truth for features
│       ├── preprocessing.py       # ColumnTransformer factory
│       └── helpers.py             # validation, risk bucketing, logging
├── ml/
│   ├── dataset.py                 # Synthetic dataset generator
│   ├── train.py                   # Fit and serialize the model
│   ├── evaluate.py                # Metric report
│   └── inference.py               # Model IO + prediction primitives
├── data/
│   ├── raw/                       # Generated CSV lives here
│   └── processed/
├── models/                        # Serialized model + metrics.json
├── static/
│   └── styles.css                 # All UI styling
├── tests/
│   ├── conftest.py                # Trains a fresh model in tmp dir
│   ├── test_preprocessing.py
│   ├── test_ml.py
│   └── test_routes.py
├── requirements.txt
├── .env.example
├── .gitignore
├── run.py
└── README.md
```

## Installation

### 1. Prerequisites

- Python 3.11 or newer
- pip

### 2. Clone / open the project

```bash
cd churn_predictor
```

### 3. Create a virtual environment

macOS / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (bash / Git Bash):
```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Copy the environment file

```bash
cp .env.example .env
```

Edit `.env` if you want to change the port, thresholds, or model filename.

## Train the model

The training script generates a synthetic 5,000-row dataset (if one doesn't
already exist) and fits a RandomForest classifier:

```bash
python -m ml.train
# or:  python run.py train
```

Artifacts produced:

- `data/raw/churn_dataset.csv`
- `models/churn_model.joblib`
- `models/metrics.json`

Typical metrics on the default seed (ROC-AUC ≈ 0.85+).

## Run the app

```bash
python run.py
```

Open <http://127.0.0.1:5001> in your browser.

- `/` — Landing page
- `/predict` — Fill in the form → get a churn prediction
- `/dashboard` — Dataset-level breakdowns
- `/about` — Model + limitations
- `/api/health` — JSON status
- `POST /api/predict` — JSON prediction API

## Run the tests

The test suite trains a small model into a temp directory (session-scoped),
so it does not touch your `./models/` folder.

```bash
pytest
```

Expected: all tests pass in a few seconds.

## Example API usage

```bash
curl -X POST http://127.0.0.1:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 3,
    "monthly_charges": 89.0,
    "total_charges": 267.0,
    "num_support_calls": 5,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "tech_support": "No",
    "online_security": "No",
    "paperless_billing": "Yes",
    "senior_citizen": "No"
  }'
```

Response:

```json
{
  "probability": 0.83,
  "probability_pct": "83.0%",
  "label": 1,
  "risk_bucket": "High",
  "top_factors": [["tenure_months", 0.28], ["contract_type", 0.15], ...],
  "importance_bars": [["tenure_months", 1.0], ...],
  "recommendations": ["Reach out within 48 hours ..."],
  "inputs": { ... }
}
```

## Configuration

All settings are read from environment variables (see `.env.example`). The
`Settings` dataclass in `app/config.py` is the single source of truth — no
module should read `os.environ` directly.

| Variable                | Default         | Purpose                                  |
| ----------------------- | --------------- | ---------------------------------------- |
| `APP_NAME`              | Churn Predictor | Shown in nav + page titles                |
| `DEBUG`                 | true            | Autoreload for FastHTML dev server        |
| `HOST` / `PORT`         | 127.0.0.1 / 5001| Bind address                              |
| `MODEL_FILENAME`        | churn_model.joblib | Serialized model file                  |
| `DATASET_FILENAME`      | churn_dataset.csv  | Raw dataset CSV                         |
| `RANDOM_SEED`           | 42              | Deterministic training + splits           |
| `TEST_SIZE`             | 0.2             | Test fraction for `train_test_split`     |
| `HIGH_RISK_THRESHOLD`   | 0.66            | Boundary between Medium and High         |
| `MEDIUM_RISK_THRESHOLD` | 0.33            | Boundary between Low and Medium          |
| `LOG_LEVEL`             | INFO            | Root logger level                         |

## Troubleshooting

- **"Model not found" error on `/predict`** — run `python -m ml.train` first.
- **Port already in use** — set `PORT=<something else>` in `.env`.
- **`ModuleNotFoundError: app`** — you're outside the project root; `cd churn_predictor` first, or use `python run.py`.
- **Slow first prediction** — the model is loaded lazily on the first call; subsequent calls hit the cache.

## Future improvements

- Add SHAP for per-instance (not global) feature attributions.
- Batch prediction endpoint (CSV upload → CSV download).
- Add async background training via a task queue (RQ / Celery).
- Replace SQLite-less design with a real DB if you want to persist scored customers.
- Model registry + A/B testing between model versions.
- Fairness audit + calibration curves in the About page.
- Dockerfile + docker-compose for one-command deployment.

## License

MIT — do whatever you like with it, no warranty.
