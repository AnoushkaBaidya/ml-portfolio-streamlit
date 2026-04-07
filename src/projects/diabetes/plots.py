"""
Plotting helpers for the diabetes prediction project.

This module isolates all matplotlib/seaborn plotting logic from the page
rendering layer so that the Streamlit page stays easier to read.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def plot_diabetes_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Plot the correlation heatmap for the diabetes dataset.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
    )
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    return fig


def plot_scaling_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """
    Plot grouped bars for scaling strategy performance.
    """
    melted = results_df.melt(id_vars="Scaling", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.barplot(data=melted, x="Scaling", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Scaling Impact on Logistic Regression")
    fig.tight_layout()
    return fig


def plot_model_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """
    Plot grouped bars comparing diabetes model metrics.
    """
    melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, title: str) -> plt.Figure:
    """
    Plot a confusion matrix for one trained model.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_roc_curves(X_test_scaled, y_test, fitted_models: dict) -> plt.Figure:
    """
    Plot overlaid ROC curves for all diabetes models.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for model_name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, name=model_name)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC = 0.5)")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig