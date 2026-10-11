"""Page layout primitives: navbar, footer, page shell."""

from __future__ import annotations

from fasthtml.common import (
    A,
    Div,
    Footer,
    Header,
    Main,
    Nav,
    P,
    Span,
    Strong,
    Title,
)

from app.config import settings

NAV_LINKS: tuple[tuple[str, str], ...] = (
    ("Home", "/"),
    ("Predict", "/predict"),
    ("Dashboard", "/dashboard"),
    ("About", "/about"),
)


def navbar(active: str = "") -> Header:
    """Top navigation bar. `active` should match one of NAV_LINKS labels."""
    return Header(
        Nav(
            Div(
                A(
                    Span("◈", cls="brand-mark"),
                    Strong(settings.app_name),
                    href="/",
                    cls="brand",
                ),
                cls="nav-brand",
            ),
            Div(
                *[
                    A(
                        label,
                        href=href,
                        cls=f"nav-link{' active' if label == active else ''}",
                    )
                    for label, href in NAV_LINKS
                ],
                cls="nav-links",
            ),
            cls="nav-inner",
        ),
        cls="site-header",
    )


def footer() -> Footer:
    return Footer(
        Div(
            P(
                f"© {settings.app_name}. Built with Python + FastHTML.",
                cls="footer-text",
            ),
            P(
                "Predictions are estimates, not guarantees. Use as decision support only.",
                cls="footer-note",
            ),
            cls="footer-inner",
        ),
        cls="site-footer",
    )


def page(*children, title: str = "", active_nav: str = ""):
    """Wrap page content with title, navbar, footer."""
    full_title = f"{title} | {settings.app_name}" if title else settings.app_name
    return (
        Title(full_title),
        navbar(active=active_nav),
        Main(*children, cls="site-main"),
        footer(),
    )
