"""
Main landing page for the ML Portfolio Streamlit application.
"""

import streamlit as st

from src.ui.theme import apply_app_theme
from src.ui.components import render_project_card


st.set_page_config(
    page_title="ML Portfolio",
    page_icon="🤖",
    layout="wide",
)


def main() -> None:
    apply_app_theme()

    st.title("🤖 Machine Learning Portfolio")
    st.caption(
        "A production-aware, multi-project machine learning portfolio built with Streamlit."
    )

    st.markdown(
        """
        Welcome to my interactive ML portfolio. This application combines four end-to-end
        machine learning demonstrations into one deployable web app.

        It is designed to show both:
        - **interactive ML exploration**
        - **production-style inference using saved artifacts**

        ### Included Projects
        - **Diabetes Prediction** — classification, scaling, model benchmarking, production inference
        - **Netflix Clustering** — KMeans, K-selection, PCA visualization, production clustering artifacts
        - **Spotify Popularity Prediction** — regression, explainability, saved production model
        - **Customer Churn Prediction** — imbalance handling, threshold tuning, artifact-backed scoring

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
                "Binary classification with feature scaling, model comparison, "
                "hyperparameter tuning, and saved production artifacts for inference."
            ),
            skills=[
                "Classification",
                "Feature Scaling",
                "Model Evaluation",
                "Production Inference",
            ],
        )

        render_project_card(
            emoji="🎵",
            title="Spotify Popularity Prediction",
            summary=(
                "Regression on audio features with model comparison, SHAP explainability, "
                "and artifact-backed production scoring."
            ),
            skills=[
                "Regression",
                "Explainability",
                "Model Comparison",
                "Artifact Deployment",
            ],
        )

    with col2:
        render_project_card(
            emoji="🎬",
            title="Netflix Clustering",
            summary=(
                "Unsupervised clustering with production-selected K, PCA visualization, "
                "and saved clustering artifacts."
            ),
            skills=[
                "Clustering",
                "KMeans",
                "PCA",
                "Production Artifacts",
            ],
        )

        render_project_card(
            emoji="📉",
            title="Customer Churn Prediction",
            summary=(
                "Customer churn classification with feature engineering, imbalance handling, "
                "threshold tuning, and saved production model scoring."
            ),
            skills=[
                "Classification",
                "SMOTE",
                "Threshold Tuning",
                "Production Scoring",
            ],
        )

    st.divider()

    st.success(
        "This portfolio is designed to be friendly: one link, live interaction, "
        "and production-aware ML architecture with saved model artifacts."
    )


if __name__ == "__main__":
    main()