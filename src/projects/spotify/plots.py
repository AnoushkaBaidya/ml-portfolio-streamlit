"""
Plotting helpers for the Spotify popularity prediction project.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def plot_spotify_popularity_distribution(series: pd.Series) -> plt.Figure:
    """
    Plot the target popularity distribution.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(series, bins=40, edgecolor="white")
    ax.set_xlabel("Popularity")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Song Popularity")
    fig.tight_layout()
    return fig


def plot_spotify_correlation_heatmap(corr_df: pd.DataFrame) -> plt.Figure:
    """
    Plot the Spotify feature correlation heatmap.
    """
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    return fig


def plot_target_correlations(corr_df: pd.DataFrame, target_col: str) -> plt.Figure:
    """
    Plot feature correlations with the target column.
    """
    target_corr = corr_df[target_col].drop(target_col).sort_values(key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    target_corr.plot.barh(ax=ax)
    ax.set_xlabel(f"Pearson Correlation with {target_col}")
    ax.set_title("Feature Correlation with Popularity")
    ax.axvline(0, color="grey", linewidth=0.8)
    fig.tight_layout()
    return fig


def plot_spotify_model_metrics(metrics_df: pd.DataFrame) -> plt.Figure:
    """
    Plot side-by-side bar charts for MAE, RMSE, and R².
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, metric in zip(axes, ["MAE", "RMSE", "R²"]):
        metrics_df[metric].plot.bar(ax=ax, edgecolor="white")
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Model Performance Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_spotify_shap_summary(shap_values, X_test, feature_names: list[str]) -> plt.Figure:
    """
    Plot the SHAP summary plot and return the matplotlib figure.
    """
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_spotify_mean_abs_shap(mean_abs_shap: np.ndarray, feature_names: list[str]) -> plt.Figure:
    """
    Plot mean absolute SHAP feature importance.
    """
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs_shap})
        .sort_values("Mean |SHAP|", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["Feature"], importance_df["Mean |SHAP|"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    fig.tight_layout()
    return fig
