"""
Clustering utilities for the Netflix project.

This module contains:
- elbow/silhouette computations
- KMeans fitting
- PCA projection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


@st.cache_data
def compute_k_selection_metrics(
    scaled_features: np.ndarray,
    k_range: tuple = (2, 11),
) -> pd.DataFrame:
    """
    Compute inertia and silhouette score across a range of K values.
    """
    records = []

    for k in range(k_range[0], k_range[1]):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled_features)

        records.append(
            {
                "K": k,
                "Inertia": model.inertia_,
                "Silhouette": silhouette_score(scaled_features, labels),
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


@st.cache_data
def run_netflix_pca(
    scaled_features: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """
    Reduce high-dimensional clustering features into 2D using PCA.
    """
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(scaled_features)