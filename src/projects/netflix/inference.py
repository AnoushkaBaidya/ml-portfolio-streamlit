"""
Artifact-backed inference helpers for the Netflix clustering project.

For clustering, "inference" means assigning cluster labels using the
saved scaler, saved feature schema, and saved KMeans model.
"""

from __future__ import annotations

import pandas as pd

from src.projects.netflix.artifact_io import load_netflix_artifact_bundle


def load_netflix_production_bundle(version: str | None = None) -> dict:
    """
    Convenience wrapper for loading the Netflix production artifact bundle.
    """
    return load_netflix_artifact_bundle(version=version)


def assign_clusters_with_artifact(
    encoded_df: pd.DataFrame,
    version: str | None = None,
) -> dict:
    """
    Assign cluster labels to an already-encoded Netflix feature matrix using
    the saved production artifacts.

    Parameters
    ----------
    encoded_df : pd.DataFrame
        Numeric feature matrix with columns in the expected training schema.
    version : str | None
        Optional explicit artifact version.

    Returns
    -------
    dict
        {
            "version": ...,
            "model_name": ...,
            "labels": ...,
            "pca_2d": ...,
            "pca_3d": ...
        }
    """
    bundle = load_netflix_artifact_bundle(version=version)

    model = bundle["model"]
    preprocessors = bundle["preprocessors"]
    scaler = preprocessors["scaler"]
    pca_2d = preprocessors["pca_2d"]
    pca_3d = preprocessors["pca_3d"]
    feature_order = preprocessors["feature_order"]

    aligned_df = encoded_df.reindex(columns=feature_order, fill_value=0)
    scaled_features = scaler.transform(aligned_df)

    labels = model.predict(scaled_features)
    pca_result_2d = pca_2d.transform(scaled_features)
    pca_result_3d = pca_3d.transform(scaled_features)

    return {
        "version": bundle["version"],
        "model_name": bundle["model_card"]["model_name"],
        "labels": labels,
        "pca_2d": pca_result_2d,
        "pca_3d": pca_result_3d,
        "model_card": bundle["model_card"],
        "metrics": bundle["metrics"],
    }