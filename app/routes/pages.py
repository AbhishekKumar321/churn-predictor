"""Static pages: home, predict, dashboard, about."""

from __future__ import annotations

from fasthtml.common import (
    H1,
    H2,
    H3,
    A,
    Article,
    Div,
    Li,
    P,
    Section,
    Span,
    Strong,
    Ul,
)

from app.components.cards import empty_result, kpi_card
from app.components.charts import horizontal_bars
from app.components.forms import prediction_form
from app.components.layout import page
from app.config import settings
from app.services.data_service import dataset_stats
from app.utils.schema import ALL_FIELDS


def register(app) -> None:
    """Wire page routes onto the given FastHTML app."""

    @app.get("/")
    def home():
        return page(
            _hero(),
            _features_section(),
            _how_it_works_section(),
            title="Home",
            active_nav="Home",
        )

    @app.get("/predict")
    def predict_page():
        return page(
            _predict_hero(),
            Section(
                Div(
                    Div(
                        H2("Customer profile"),
                        P(
                            "Enter details about the customer. All fields are required. "
                            "Defaults reflect a typical customer.",
                            cls="section-subtext",
                        ),
                        prediction_form(),
                        cls="predict-form-col",
                    ),
                    Div(
                        H2("Result"),
                        Div(empty_result(), id="result"),
                        cls="predict-result-col",
                    ),
                    cls="predict-layout",
                ),
                cls="container",
            ),
            title="Predict",
            active_nav="Predict",
        )

    @app.get("/dashboard")
    def dashboard():
        stats = dataset_stats()
        if stats.n_rows == 0:
            return page(
                Section(
                    Div(
                        H1("Dashboard"),
                        P(
                            "No training data found yet. Run "
                            "python -m ml.train to generate a dataset and train the model.",
                            cls="empty-state",
                        ),
                        cls="container",
                    ),
                ),
                title="Dashboard",
                active_nav="Dashboard",
            )

        contract_rows = [(label, rate) for label, _, rate in stats.contract_breakdown]
        tenure_rows = [(label, rate) for label, _, rate in stats.tenure_buckets]

        return page(
            Section(
                Div(
                    H1("Dataset overview"),
                    P(
                        "Aggregate statistics from the training dataset the model was fit on.",
                        cls="section-subtext",
                    ),
                    Div(
                        kpi_card("Rows", f"{stats.n_rows:,}"),
                        kpi_card("Churned", f"{stats.n_churned:,}", tone="warn"),
                        kpi_card(
                            "Churn rate",
                            f"{stats.churn_rate * 100:.1f}%",
                            tone="danger",
                        ),
                        cls="kpi-grid",
                    ),
                    Div(
                        horizontal_bars(
                            "Churn rate by contract type",
                            contract_rows,
                            subtitle="Month-to-month contracts churn the most.",
                            fmt="pct",
                        ),
                        horizontal_bars(
                            "Churn rate by tenure bucket",
                            tenure_rows,
                            subtitle="New customers are the most volatile.",
                            fmt="pct",
                        ),
                        cls="dashboard-grid",
                    ),
                    cls="container",
                ),
            ),
            title="Dashboard",
            active_nav="Dashboard",
        )

    @app.get("/about")
    def about():
        return page(
            Section(
                Div(
                    H1("About this project"),
                    _about_body(),
                    cls="container narrow",
                ),
            ),
            title="About",
            active_nav="About",
        )


# ---------- private view helpers ----------

def _hero():
    return Section(
        Div(
            Div(
                Span("AI-assisted retention", cls="eyebrow"),
                H1("Know which customers are about to leave — before they do."),
                P(
                    f"{settings.app_name} scores each customer's churn risk from a small set "
                    "of account details, and shows the factors driving the score.",
                    cls="hero-lede",
                ),
                Div(
                    A("Get started", href="/predict", cls="btn btn-primary btn-lg"),
                    A("View dashboard", href="/dashboard", cls="btn btn-ghost btn-lg"),
                    cls="hero-actions",
                ),
                cls="hero-copy",
            ),
            Div(
                Div(
                    Div(Span("87.4%", cls="stat-value"), Span("Model AUC", cls="stat-label")),
                    Div(Span("11", cls="stat-value"), Span("Features", cls="stat-label")),
                    Div(Span("<50ms", cls="stat-value"), Span("Per prediction", cls="stat-label")),
                    cls="stat-grid",
                ),
                cls="hero-visual",
            ),
            cls="hero container",
        ),
        cls="hero-section",
    )


def _predict_hero():
    return Section(
        Div(
            Span("Predict", cls="eyebrow"),
            H1("Score a customer"),
            P(
                "Fill in the customer's account details on the left. "
                "The model returns a churn probability, a risk bucket, "
                "and recommended actions.",
                cls="hero-lede",
            ),
            cls="container narrow",
        ),
        cls="page-hero",
    )


def _features_section():
    features = [
        ("🎯", "Focused predictions", "Binary churn probability, not vague scores."),
        ("🔍", "Explainability", "See which features drive the model's decisions."),
        ("⚡", "Fast", "Sub-50ms inference; runs locally on CPU."),
        ("🧩", "Modular", "Clean separation of UI, services, and ML pipeline."),
        ("🛡", "Validated inputs", "Every field is coerced and bounds-checked."),
        ("📊", "Dashboard", "Dataset stats and churn breakdowns at a glance."),
    ]
    cards = [
        Article(
            Div(icon, cls="feature-icon"),
            H3(title),
            P(desc),
            cls="card feature-card",
        )
        for icon, title, desc in features
    ]
    return Section(
        Div(
            Span("Features", cls="eyebrow"),
            H2("What you get"),
            Div(*cards, cls="feature-grid"),
            cls="container",
        ),
        cls="features-section",
    )


def _how_it_works_section():
    steps = [
        ("1", "Enter details", f"Fill in {len(ALL_FIELDS)} attributes about the customer."),
        ("2", "Score", "The trained model returns a churn probability in milliseconds."),
        ("3", "Act", "Get risk bucket, top drivers, and concrete next steps."),
    ]
    step_cards = [
        Article(
            Span(n, cls="step-num"),
            H3(title),
            P(desc),
            cls="card step-card",
        )
        for n, title, desc in steps
    ]
    return Section(
        Div(
            Span("How it works", cls="eyebrow"),
            H2("Three steps"),
            Div(*step_cards, cls="steps-grid"),
            cls="container",
        ),
        cls="how-section",
    )


def _about_body():
    return Div(
        Section(
            H2("What this project does"),
            P(
                f"{settings.app_name} is a demonstration production-quality "
                "AI/ML web application built entirely in Python and FastHTML. "
                "It predicts whether a subscription customer is likely to churn."
            ),
        ),
        Section(
            H2("How the model works"),
            P(
                "A RandomForestClassifier is trained on a synthetic but realistic "
                "customer dataset with 11 features. Numeric features are scaled; "
                "categorical features are one-hot encoded. The model outputs a "
                "probability which is bucketed into Low / Medium / High risk based on "
                "configurable thresholds."
            ),
            P(
                "Feature importance from the trained forest is used to show which "
                "attributes drive predictions globally. Per-instance attributions "
                "(via SHAP) are a natural follow-up."
            ),
        ),
        Section(
            H2("Dataset"),
            P(
                "The training dataset is generated in "
                "ml/dataset.py "
                "with realistic priors — short-tenure, month-to-month, "
                "high-bill, electronic-check customers are more likely to churn."
            ),
        ),
        Section(
            H2("Limitations"),
            Ul(
                Li("Trained on synthetic data — real-world calibration will differ."),
                Li("Global feature importance ≠ per-customer attribution."),
                Li("No fairness auditing has been done."),
                Li("Not a substitute for domain judgement."),
            ),
        ),
        Section(
            H2("Technology"),
            Ul(
                Li(Strong("Python"), " — application language"),
                Li(Strong("FastHTML"), " + HTMX — server-rendered UI with interactivity"),
                Li(Strong("scikit-learn"), " — model + preprocessing pipeline"),
                Li(Strong("pandas / numpy"), " — data handling"),
                Li(Strong("joblib"), " — model serialization"),
                Li(Strong("pytest"), " — tests"),
            ),
        ),
        cls="about-content",
    )
