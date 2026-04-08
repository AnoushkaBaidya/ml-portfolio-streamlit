"""
Artifact-backed inference utilities for the Spotify project.
"""

from __future__ import annotations

import numpy as np

from src.projects.spotify.artifact_io import load_spotify_artifact_bundle
from src.projects.spotify.data import AUDIO_FEATURES
from src.shared.schema import validate_required_features


def build_spotify_input_payload(
    duration_ms: int,
    explicit: int,
    danceability: float,
    energy: float,
    key: int,
    loudness: float,
    mode: int,
    speechiness: float,
    acousticness: float,
    instrumentalness: float,
    liveness: float,
    valence: float,
    tempo: float,
    time_signature: int,
) -> dict:
    """
    Build a standardized Spotify input payload.
    """
    return {
        "duration_ms": duration_ms,
        "explicit": explicit,
        "danceability": danceability,
        "energy": energy,
        "key": key,
        "loudness": loudness,
        "mode": mode,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "time_signature": time_signature,
    }


def predict_with_spotify_artifact(input_payload: dict, version: str | None = None) -> dict:
    """
    Run Spotify prediction using a saved artifact bundle.
    """
    is_valid, missing_features = validate_required_features(input_payload, AUDIO_FEATURES)
    if not is_valid:
        raise ValueError(f"Missing required features: {missing_features}")

    bundle = load_spotify_artifact_bundle(version=version)
    model = bundle["model"]
    preprocessors = bundle["preprocessors"]
    scaler = preprocessors["scaler"]

    row = np.array([[input_payload[feature] for feature in AUDIO_FEATURES]])
    row_scaled = scaler.transform(row)

    prediction = model.predict(row_scaled)[0]
    prediction = float(np.clip(prediction, 0, 100))

    return {
        "version": bundle["version"],
        "model_name": bundle["model_card"]["model_name"],
        "prediction": prediction,
        "model_card": bundle["model_card"],
        "metrics": bundle["metrics"],
    }


def load_spotify_production_bundle(version: str | None = None) -> dict:
    """
    Convenience wrapper for loading the Spotify production artifact bundle.
    """
    return load_spotify_artifact_bundle(version=version)