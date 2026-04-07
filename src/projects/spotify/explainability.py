"""
Explainability utilities for the Spotify project.

This module uses SHAP to explain the best tree-based regression model.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
import shap


def get_best_tree_model(results: dict) -> tuple[str, object]:
    """
    Select the best tree-based model from the trained Spotify models.

    Linear Regression is excluded because TreeExplainer is meant for tree ensembles.
    """
    tree_models = {
        model_name: result
        for model_name, result in results.items()
        if model_name != "Linear Regression"
    }

    best_name = max(tree_models, key=lambda model_name: tree_models[model_name]["r2"])
    best_model = tree_models[best_name]["model"]
    return best_name, best_model


@st.cache_data
def compute_spotify_shap_values(_model, X_test):
    """
    Compute SHAP values for a fitted tree-based model.

    The model argument is prefixed with an underscore so Streamlit
    will exclude it from cache hashing.
    """
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_test)
    return shap_values


def compute_mean_absolute_shap(shap_values: np.ndarray, feature_names: list[str]):
    """
    Compute mean absolute SHAP importance for each feature.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return mean_abs_shap, feature_names