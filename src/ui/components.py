"""
Reusable UI components for the ML portfolio application.

These components are intentionally presentation-focused and should not
contain any project-specific model logic. Their job is to keep the
Streamlit page files cleaner and easier to read.
"""

from __future__ import annotations

import streamlit as st


def render_project_card(
    emoji: str,
    title: str,
    summary: str,
    skills: list[str],
) -> None:
    """
    Render a reusable project summary card.
    """
    badges_html = "".join(
        [f'<span class="skill-badge">{skill}</span>' for skill in skills]
    )

    st.markdown(
        f"""
        <div class="project-card">
            <div class="project-title">{emoji} {title}</div>
            <div class="project-summary">{summary}</div>
            <div>{badges_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(
    title: str,
    value: str,
    subtitle: str = "",
) -> None:
    """
    Render a compact information card for production model details.
    """
    subtitle_html = f'<div class="info-card-subtitle">{subtitle}</div>' if subtitle else ""

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">{title}</div>
            <div class="info-card-value">{value}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_card(title: str) -> None:
    """
    Start a visual section with a styled title.
    """
    st.markdown(
        f"""
        <div class="project-card">
            <div class="project-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_key_value_rows(items: list[tuple[str, str]], columns: int = 2) -> None:
    """
    Render key-value items in a clean Streamlit-native grid.

    Parameters
    ----------
    items : list[tuple[str, str]]
        List of (label, value) pairs to display.
    columns : int
        Number of columns to use in the grid.
    """
    if not items:
        st.info("No details available.")
        return

    for start in range(0, len(items), columns):
        row_items = items[start:start + columns]
        cols = st.columns(columns)

        for idx, (label, value) in enumerate(row_items):
            with cols[idx]:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-card-title">{label}</div>
                        <div class="info-card-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_notes_card(title: str, text: str) -> None:
    """
    Render a styled notes block.
    """
    st.markdown(
        f"""
        <div class="project-card">
            <div class="project-title">{title}</div>
            <div class="project-summary">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )