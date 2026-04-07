"""
Global theme and styling utilities for the Streamlit application.

This module centralizes CSS and visual behavior that should be shared
across all pages. Keeping styling here prevents duplication and gives
the full app a consistent look and feel.

Why this matters:
- Recruiters should feel they are viewing one polished product, not
  four unrelated mini-apps.
- Shared theming makes maintenance easier.
- Visual consistency improves the perceived professionalism of the app.
"""

import streamlit as st


def apply_app_theme() -> None:
    """
    Apply a lightweight custom CSS theme across the Streamlit app.

    This function should be called near the top of every page.
    The styling here is intentionally moderate: clean cards, slightly
    tighter spacing, and polished typography without over-designing.
    """
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                padding-left: 2rem;
                padding-right: 2rem;
                max-width: 1200px;
            }

            .project-card {
                border: 1px solid rgba(120, 120, 120, 0.18);
                border-radius: 16px;
                padding: 1.2rem;
                margin-bottom: 1rem;
                background-color: rgba(250, 250, 250, 0.03);
            }

            .project-title {
                font-size: 1.2rem;
                font-weight: 700;
                margin-bottom: 0.4rem;
            }

            .project-summary {
                font-size: 0.95rem;
                margin-bottom: 0.6rem;
                color: rgba(240, 240, 240, 0.85);
            }

            .skill-badge {
                display: inline-block;
                padding: 0.25rem 0.55rem;
                border-radius: 999px;
                margin-right: 0.35rem;
                margin-bottom: 0.35rem;
                font-size: 0.78rem;
                border: 1px solid rgba(120, 120, 120, 0.25);
                background-color: rgba(255, 255, 255, 0.04);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )