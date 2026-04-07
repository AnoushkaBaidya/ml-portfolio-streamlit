"""
Plotting helpers for the churn project.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_churn_distribution(series: pd.Series) -> plt.Figure:
    """
    Plot churn vs retained class counts.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    counts = series.value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white")

    for idx, value in enumerate(counts.values):
        ax.text(idx, value + 40, f"{value:,}", ha="center", fontweight="bold", fontsize=10)

    ax.set_ylabel("Customers")
    ax.set_title("Churn vs Retained")
    sns.despine()
    fig.tight_layout()
    return fig


def plot_churn_rate_by_contract(df: pd.DataFrame, target_col: str) -> plt.Figure:
    """
    Plot churn rate grouped by contract type.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    churn_by_contract = (
        df.groupby("Contract")[target_col]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )
    churn_by_contract.plot.barh(ax=ax, color="#3498db", edgecolor="white")
    ax.set_xlabel("Churn Rate (%)")
    ax.set_title("Month-to-month contracts churn the most")
    sns.despine()
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_series: pd.Series) -> plt.Figure:
    """
    Plot top feature importances for the churn model.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    importance_series.plot.barh(ax=ax, color="#8e44ad", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Features")
    sns.despine()
    fig.tight_layout()
    return fig


def plot_before_after_smote(y_train, y_resampled) -> plt.Figure:
    """
    Plot class distribution before and after SMOTE.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    for ax, data, title in [
        (axes[0], y_train, "Before SMOTE"),
        (axes[1], y_resampled, "After SMOTE"),
    ]:
        counts = pd.Series(data).value_counts().sort_index()
        ax.bar(
            ["No Churn (0)", "Churn (1)"],
            counts.values,
            color=["#2ecc71", "#e74c3c"],
            edgecolor="white",
        )
        ax.set_title(title)
        ax.set_ylabel("Samples")

        for idx, value in enumerate(counts.values):
            ax.text(idx, value + 20, f"{value:,}", ha="center", fontsize=9)

    sns.despine()
    fig.tight_layout()
    return fig


def plot_precision_recall_curve_with_marker(recalls, precisions, marker_recall, marker_precision, threshold: float) -> plt.Figure:
    """
    Plot the precision-recall curve and highlight the current threshold point.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recalls, precisions, linewidth=2, label="PR curve")
    ax.scatter(
        [marker_recall],
        [marker_precision],
        s=120,
        color="#e74c3c",
        zorder=5,
        label=f"Threshold = {threshold:.2f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    sns.despine()
    fig.tight_layout()
    return fig


def plot_threshold_sweep(sweep_df: pd.DataFrame, current_threshold: float) -> plt.Figure:
    """
    Plot precision, recall, and F1 across all thresholds.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(sweep_df["Threshold"], sweep_df["Precision"], label="Precision", linewidth=2)
    ax.plot(sweep_df["Threshold"], sweep_df["Recall"], label="Recall", linewidth=2)
    ax.plot(sweep_df["Threshold"], sweep_df["F1"], label="F1", linewidth=2, linestyle="--")
    ax.axvline(
        current_threshold,
        color="#e74c3c",
        linestyle=":",
        linewidth=1.5,
        label=f"Current ({current_threshold:.2f})",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("How Metrics Change with Threshold")
    ax.legend(fontsize=8)
    sns.despine()
    fig.tight_layout()
    return fig


def plot_confusion_matrix_heatmap(confusion_matrix_values) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        confusion_matrix_values,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig