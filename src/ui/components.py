"""
Reusable UI components for the ML portfolio application.

These components are intentionally presentation-focused and should not
contain any project-specific model logic. Their job is to keep the
Streamlit page files cleaner and easier to read.
"""

import streamlit as st


def render_project_card(
    emoji: str,
    title: str,
    summary: str,
    skills: list[str],
) -> None:
    """
    Render a reusable project summary card.

    Parameters
    ----------
    emoji : str
        Small visual identifier for the project.
    title : str
        Project title shown prominently on the card.
    summary : str
        Short recruiter-friendly explanation of the project.
    skills : list[str]
        Key ML concepts or engineering skills highlighted as badges.
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