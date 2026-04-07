"""
Artifact-backed inference utilities for the diabetes project.

This module is responsible for production-style prediction using a
saved model artifact and saved preprocessing objects.
"""

from __future__ import annotations

import numpy as np

from src.projects.diabetes.artifact_io import load_diabetes_artifact_bundle
from src.projects.diabetes.data import FEATURE_COLS
from src.shared.schema import validate_required_features


def build_diabetes_input_payload(
    pregnancies: int,
    glucose: int,
    blood_pressure: int,
    skin_thickness: int,
    insulin: int,
    bmi: float,
    dpf: float,
    age: int,
) -> dict:
    """
    Build a standardized diabetes input payload dictionary.
    """
    return {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }


def predict_with_diabetes_artifact(input_payload: dict, version: str | None = None) -> dict:
    """
    Run diabetes prediction using a saved artifact bundle.

    Parameters
    ----------
    input_payload : dict
        Dictionary containing all required diabetes input features.
    version : str | None
        Optional explicit version. If omitted, latest version is used.

    Returns
    -------
    dict
        {
            "version": ...,
            "model_name": ...,
            "prediction": ...,
            "probabilities": ...
        }
    """
    is_valid, missing_features = validate_required_features(input_payload, FEATURE_COLS)
    if not is_valid:
        raise ValueError(f"Missing required features: {missing_features}")

    bundle = load_diabetes_artifact_bundle(version=version)
    model = bundle["model"]
    preprocessors = bundle["preprocessors"]
    scaler = preprocessors["scaler"]

    row = np.array([[input_payload[feature] for feature in FEATURE_COLS]])
    row_scaled = scaler.transform(row)

    prediction = model.predict(row_scaled)[0]
    probabilities = model.predict_proba(row_scaled)[0]

    return {
        "version": bundle["version"],
        "model_name": bundle["model_card"]["model_name"],
        "prediction": int(prediction),
        "probabilities": probabilities,
        "model_card": bundle["model_card"],
        "metrics": bundle["metrics"],
    }


def load_diabetes_production_bundle(version: str | None = None) -> dict:
    """
    Convenience wrapper for loading the diabetes production artifact bundle.
    """
    return load_diabetes_artifact_bundle(version=version)