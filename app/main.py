"""FastHTML application factory + entrypoint.

Import order matters: config must be imported before anything that logs,
so log level is set correctly.
"""

from __future__ import annotations

from fasthtml.common import FastHTML, Link, Meta, Script, serve
from starlette.staticfiles import StaticFiles

from app.config import settings
from app.routes import api as api_routes
from app.routes import pages as page_routes
from app.routes import prediction as prediction_routes
from app.utils.helpers import setup_logging

logger = setup_logging()


def create_app() -> FastHTML:
    """Build and configure a FastHTML app instance."""
    app = FastHTML(
        hdrs=(
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            Meta(name="description", content=f"{settings.app_name} — AI-assisted retention"),
            Link(rel="stylesheet", href="/static/styles.css"),
            Script(src="https://unpkg.com/htmx.org@1.9.12"),
        ),
        pico=False,
    )

    static_dir = settings.project_root / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    else:
        logger.warning("Static directory not found: %s", static_dir)

    page_routes.register(app)
    prediction_routes.register(app)
    api_routes.register(app)

    logger.info(
        "%s ready — env=%s debug=%s",
        settings.app_name,
        settings.environment,
        settings.debug,
    )
    return app


app = create_app()


def main() -> None:
    """Console entrypoint — used by run.py."""
    serve(
        appname="app.main",
        app="app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
