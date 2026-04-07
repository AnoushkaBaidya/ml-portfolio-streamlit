"""
Main landing page for the ML Portfolio Streamlit application.

This file is the top-level entry point for the entire portfolio app.
It serves as the homepage that introduces the portfolio, explains the
included projects, and guides recruiters or viewers to the individual
project pages in the Streamlit sidebar.

Why this file exists:
- Streamlit uses `app.py` as the primary launch target.
- In a multi-page application, this file acts like the "home screen."
- We keep the landing page lightweight and focused on portfolio framing,
  while each project page lives in the `pages/` directory.

Author:
- Your name here
"""

import streamlit as st

from src.ui.theme import apply_app_theme
from src.ui.components import render_project_card


# ---------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ML Portfolio",
    page_icon="🤖",
    layout="wide",
)


def main() -> None:
    """
    Render the landing page for the ML portfolio application.

    This function is intentionally simple and presentation-focused.
    Business logic, model training, and plotting for each project
    are handled in their respective modules and page files.
    """
    apply_app_theme()

    st.title("🤖 Machine Learning Portfolio")
    st.caption(
        "A multi-project interactive portfolio built with Streamlit, "
        "featuring supervised and unsupervised machine learning workflows."
    )

    st.markdown(
        """
        Welcome to my interactive ML portfolio. This application combines four
        end-to-end machine learning demonstrations into one deployable web app.

        The projects included are:

        - **Diabetes Prediction** — binary classification, model comparison, feature scaling
        - **Netflix Clustering** — unsupervised learning, KMeans, PCA, cluster interpretation
        - **Spotify Popularity Prediction** — regression, model benchmarking, SHAP explainability
        - **Customer Churn Prediction** — classification, imbalance handling, threshold tuning

        Use the **left sidebar** to navigate between projects.
        """
    )

    st.divider()

    st.subheader("Portfolio Projects")

    col1, col2 = st.columns(2)

    with col1:
        render_project_card(
            emoji="🩺",
            title="Diabetes Prediction",
            summary=(
                "Compare multiple classification models, study feature scaling, "
                "evaluate ROC-AUC and F1, and make live patient-level predictions."
            ),
            skills=[
                "Classification",
                "Feature Scaling",
                "Model Evaluation",
                "Hyperparameter Tuning",
            ],
        )

        render_project_card(
            emoji="🎵",
            title="Spotify Popularity Prediction",
            summary=(
                "Predict song popularity from audio features using regression models, "
                "compare performance, and explain predictions with SHAP."
            ),
            skills=[
                "Regression",
                "Model Comparison",
                "Explainability",
                "Feature Analysis",
            ],
        )

    with col2:
        render_project_card(
            emoji="🎬",
            title="Netflix Clustering",
            summary=(
                "Cluster Netflix titles using KMeans, determine the optimal number "
                "of clusters, and visualize high-dimensional data with PCA."
            ),
            skills=[
                "Clustering",
                "KMeans",
                "PCA",
                "Unsupervised Learning",
            ],
        )

        render_project_card(
            emoji="📉",
            title="Customer Churn Prediction",
            summary=(
                "Predict telecom churn using engineered features, compare models, "
                "handle class imbalance, and tune classification thresholds."
            ),
            skills=[
                "Classification",
                "SMOTE",
                "Threshold Tuning",
                "Business Metrics",
            ],
        )

    st.divider()

    st.info(
        "This portfolio is designed for recruiter-friendly exploration: "
        "one link, four live machine learning applications, and no local setup required."
    )


if __name__ == "__main__":
    main()