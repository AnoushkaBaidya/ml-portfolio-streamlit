"""
Global theme and styling utilities for the Streamlit application.

This module centralizes CSS and visual behavior that should be shared
across all pages. Keeping styling here prevents duplication and gives
the full app a consistent look and feel.
"""

import streamlit as st


def apply_app_theme() -> None:
    """
    Apply a lightweight custom CSS theme across the Streamlit app.
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

            .info-card {
                border: 1px solid rgba(120, 120, 120, 0.18);
                border-radius: 16px;
                padding: 1rem 1.1rem;
                margin-bottom: 1rem;
                background-color: rgba(250, 250, 250, 0.03);
                min-height: 120px;
            }

            .info-card-title {
                font-size: 0.85rem;
                color: rgba(240, 240, 240, 0.70);
                margin-bottom: 0.55rem;
                text-transform: uppercase;
                letter-spacing: 0.03rem;
            }

            .info-card-value {
                font-size: 1.35rem;
                font-weight: 700;
                line-height: 1.25;
                margin-bottom: 0.45rem;
            }

            .info-card-subtitle {
                font-size: 0.82rem;
                color: rgba(240, 240, 240, 0.72);
            }

            .kv-table {
                margin-top: 0.5rem;
            }

            .kv-row {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                padding: 0.55rem 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            }

            .kv-row:last-child {
                border-bottom: none;
            }

            .kv-key {
                font-size: 0.92rem;
                color: rgba(240, 240, 240, 0.78);
            }

            .kv-value {
                font-size: 0.92rem;
                font-weight: 600;
                text-align: right;
                color: rgba(250, 250, 250, 0.96);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )