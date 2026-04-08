"""
Offline training pipeline for the Spotify project.

Run this script manually to:
- load data
- split and scale
- train candidate regression models
- select the best model
- save versioned production artifacts
"""

from __future__ import annotations

from pathlib import Path

from src.projects.spotify.artifact_io import save_spotify_artifact_bundle
from src.projects.spotify.data import AUDIO_FEATURES, TARGET_COL, load_spotify_data
from src.projects.spotify.features import split_and_scale_spotify_data
from src.projects.spotify.models import train_best_spotify_model
from src.shared.metadata import build_artifact_manifest, build_model_card
from src.shared.schema import build_feature_schema
from src.shared.utils import load_project_training_config


def run_spotify_training_pipeline() -> dict:
    """
    Execute the full offline Spotify training pipeline and save artifacts.
    """
    config = load_project_training_config("spotify_train.json")
    version = config["version"]

    df, dataset_source = load_spotify_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale_spotify_data(df)

    result = train_best_spotify_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        selection_metric=config["selection_metric"],
    )

    model_name = result["model_name"]
    model = result["model"]
    metrics = result["test_metrics"]

    preprocessors = {
        "scaler": scaler,
        "feature_order": AUDIO_FEATURES,
    }

    feature_schema = build_feature_schema(
        feature_names=AUDIO_FEATURES,
        target_name=TARGET_COL,
        numeric_ranges={
            "duration_ms": [60000, 420000],
            "explicit": [0, 1],
            "danceability": [0.0, 1.0],
            "energy": [0.0, 1.0],
            "key": [0, 11],
            "loudness": [-60.0, 0.0],
            "mode": [0, 1],
            "speechiness": [0.0, 1.0],
            "acousticness": [0.0, 1.0],
            "instrumentalness": [0.0, 1.0],
            "liveness": [0.0, 1.0],
            "valence": [0.0, 1.0],
            "tempo": [50.0, 200.0],
            "time_signature": [3, 7]
        },
    )

    model_card = build_model_card(
        project_name="spotify",
        version=version,
        model_name=model_name,
        dataset_source=dataset_source,
        metrics=metrics,
        hyperparameters={},
        notes="Selected offline using candidate regression model comparison and saved for production inference.",
    )

    artifact_manifest = build_artifact_manifest(
        project_name="spotify",
        version=version,
        model_name=model_name,
        dataset_source=dataset_source,
        artifact_paths={
            "model": "model.joblib",
            "preprocessors": "preprocessors.joblib",
            "metrics": "metrics.json",
            "feature_schema": "feature_schema.json",
            "training_config": "training_config.json",
            "model_card": "model_card.json",
            "artifact_manifest": "artifact_manifest.json",
        },
    )

    file_map = save_spotify_artifact_bundle(
        version=version,
        model=model,
        preprocessors=preprocessors,
        metrics=metrics,
        feature_schema=feature_schema,
        training_config=config,
        model_card=model_card,
        artifact_manifest=artifact_manifest,
    )

    leaderboard_path = Path(file_map["version_dir"]) / "leaderboard.csv"
    result["leaderboard"].to_csv(leaderboard_path, index=False)

    return {
        "project_name": "spotify",
        "version": version,
        "dataset_source": dataset_source,
        "selected_model": model_name,
        "metrics": metrics,
        "artifact_dir": str(file_map["version_dir"]),
        "leaderboard_path": str(leaderboard_path),
    }


if __name__ == "__main__":
    summary = run_spotify_training_pipeline()
    print("Spotify training pipeline completed.")
    for key, value in summary.items():
        print(f"{key}: {value}")