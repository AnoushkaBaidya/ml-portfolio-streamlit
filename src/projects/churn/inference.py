"""
Artifact-backed inference utilities for the churn project.
"""

from __future__ import annotations

import pandas as pd

from src.projects.churn.artifact_io import load_churn_artifact_bundle


def load_churn_production_bundle(version: str | None = None) -> dict:
    """
    Convenience wrapper for loading the churn production artifact bundle.
    """
    return load_churn_artifact_bundle(version=version)


def predict_with_churn_artifact(
    row_df: pd.DataFrame,
    version: str | None = None,
) -> dict:
    """
    Run churn prediction using a saved artifact bundle.

    Parameters
    ----------
    row_df : pd.DataFrame
        Single-row churn input DataFrame containing raw engineered input fields.
    version : str | None
        Optional explicit artifact version.

    Returns
    -------
    dict
        {
            "version": ...,
            "model_name": ...,
            "probability": ...,
            "prediction": ...,
            "threshold": ...
        }
    """
    bundle = load_churn_artifact_bundle(version=version)

    model = bundle["model"]
    preprocessors = bundle["preprocessors"]
    scaler = preprocessors["scaler"]
    feature_order = preprocessors["feature_order"]
    threshold = preprocessors["recommended_threshold"]

    row_encoded = pd.get_dummies(row_df, drop_first=True)
    row_encoded = row_encoded.reindex(columns=feature_order, fill_value=0)
    row_scaled = pd.DataFrame(
        scaler.transform(row_encoded),
        columns=feature_order,
    )

    churn_probability = model.predict_proba(row_scaled)[0][1]
    prediction = int(churn_probability >= threshold)

    return {
        "version": bundle["version"],
        "model_name": bundle["model_card"]["model_name"],
        "probability": float(churn_probability),
        "prediction": prediction,
        "threshold": float(threshold),
        "model_card": bundle["model_card"],
        "metrics": bundle["metrics"],
    }