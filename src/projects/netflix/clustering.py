"""
Clustering utilities for the Netflix project.

This module contains:
- K selection metrics
- final KMeans fitting
- PCA projection helpers
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


@st.cache_data
def compute_k_selection_metrics(
    scaled_features: np.ndarray,
    k_values: tuple | list = (2, 3, 4, 5, 6, 7, 8, 9, 10),
) -> pd.DataFrame:
    """
    Compute multiple clustering quality metrics for candidate K values.

    Metrics
    -------
    Inertia
        Lower is better, but usually decreases monotonically.
    Silhouette
        Higher is better.
    Davies-Bouldin
        Lower is better.
    Calinski-Harabasz
        Higher is better.
    """
    records = []

    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled_features)

        records.append(
            {
                "K": int(k),
                "Inertia": float(model.inertia_),
                "Silhouette": float(silhouette_score(scaled_features, labels)),
                "Davies-Bouldin": float(davies_bouldin_score(scaled_features, labels)),
                "Calinski-Harabasz": float(calinski_harabasz_score(scaled_features, labels)),
            }
        )

    return pd.DataFrame(records)


@st.cache_data
def run_netflix_kmeans(
    scaled_features: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Fit KMeans and return cluster labels.
    """
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    return model.fit_predict(scaled_features)


def fit_final_netflix_kmeans(
    scaled_features: np.ndarray,
    k: int,
) -> KMeans:
    """
    Fit and return a final KMeans model for production use.
    """
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(scaled_features)
    return model


@st.cache_data
def run_netflix_pca(
    scaled_features: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """
    Reduce high-dimensional clustering features using PCA.
    """
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(scaled_features)


def fit_netflix_pca(
    scaled_features: np.ndarray,
    n_components: int = 2,
) -> PCA:
    """
    Fit and return a PCA transformer for production use.
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(scaled_features)
    return pca