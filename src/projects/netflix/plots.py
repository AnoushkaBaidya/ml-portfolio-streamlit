"""
Plotting helpers for the Netflix clustering project.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_value_distribution(series: pd.Series, title: str, top_n: int = 10) -> plt.Figure:
    """
    Plot a horizontal bar chart of the top-N value counts in a series.
    """
    counts = series.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(counts) + 1))
    counts.sort_values().plot.barh(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Count")
    fig.tight_layout()
    return fig


def plot_elbow_curve(metrics_df: pd.DataFrame) -> plt.Figure:
    """
    Plot inertia vs K for the elbow method.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(metrics_df["K"], metrics_df["Inertia"], "o-")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Elbow Method – Inertia vs K")
    ax.set_xticks(metrics_df["K"])
    fig.tight_layout()
    return fig


def plot_silhouette_curve(metrics_df: pd.DataFrame) -> plt.Figure:
    """
    Plot silhouette score vs K.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(metrics_df["K"], metrics_df["Silhouette"], "s-")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs K")
    ax.set_xticks(metrics_df["K"])
    fig.tight_layout()
    return fig


def plot_pca_cluster_scatter(pca_result, labels) -> plt.Figure:
    """
    Plot a 2D PCA scatter colored by cluster labels.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        pca_result[:, 0],
        pca_result[:, 1],
        c=labels,
        cmap="Set2",
        alpha=0.6,
        edgecolors="w",
        linewidth=0.3,
        s=30,
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA – Cluster Visualisation")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    return fig