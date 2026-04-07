"""
Offline training pipeline for the diabetes project.

Run this script manually to:
- load data
- split into train/test
- train and compare candidate models
- select the best model
- save versioned production artifacts

This script is intentionally separate from Streamlit so that inference
does not retrain models at runtime.
"""

from __future__ import annotations

from pathlib import Path
import traceback

print("[train.py] File imported successfully.")

from src.projects.diabetes.artifact_io import save_diabetes_artifact_bundle
from src.projects.diabetes.data import FEATURE_COLS, TARGET_COL, load_diabetes_data
from src.projects.diabetes.models import split_diabetes_data, train_best_diabetes_model
from src.shared.metadata import build_artifact_manifest, build_model_card
from src.shared.schema import build_feature_schema
from src.shared.utils import load_project_training_config


def run_diabetes_training_pipeline() -> dict:
    """
    Execute the full offline diabetes training pipeline and save artifacts.
    """
    print("[train.py] Entered run_diabetes_training_pipeline()")

    config = load_project_training_config("diabetes_train.json")
    print(f"[train.py] Loaded config: {config}")

    version = config["version"]

    df, dataset_source = load_diabetes_data()
    print(f"[train.py] Loaded dataset. Shape: {df.shape}")
    print(f"[train.py] Dataset source: {dataset_source}")

    X_train, X_test, y_train, y_test = split_diabetes_data(
        df=df,
        test_size=config["test_size"],
        random_state=config["random_state"],
    )
    print("[train.py] Data split complete.")

    result = train_best_diabetes_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        selection_metric=config["selection_metric"],
    )
    print("[train.py] Model training complete.")

    model_name = result["model_name"]
    model = result["model"]
    scaler = result["scaler"]
    best_params = result["best_params"]
    metrics = result["test_metrics"]

    preprocessors = {
        "scaler": scaler,
        "feature_order": FEATURE_COLS,
    }

    feature_schema = build_feature_schema(
        feature_names=FEATURE_COLS,
        target_name=TARGET_COL,
        numeric_ranges={
            "Pregnancies": [0, 17],
            "Glucose": [0, 200],
            "BloodPressure": [0, 122],
            "SkinThickness": [0, 99],
            "Insulin": [0, 846],
            "BMI": [0.0, 67.0],
            "DiabetesPedigreeFunction": [0.078, 2.42],
            "Age": [21, 81],
        },
    )
    print("[train.py] Feature schema built.")

    model_card = build_model_card(
        project_name="diabetes",
        version=version,
        model_name=model_name,
        dataset_source=dataset_source,
        metrics=metrics,
        hyperparameters=best_params,
        notes="Selected offline using GridSearchCV and saved for production inference.",
    )
    print("[train.py] Model card built.")

    artifact_manifest = build_artifact_manifest(
        project_name="diabetes",
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
    print("[train.py] Artifact manifest built.")

    file_map = save_diabetes_artifact_bundle(
        version=version,
        model=model,
        preprocessors=preprocessors,
        metrics=metrics,
        feature_schema=feature_schema,
        training_config=config,
        model_card=model_card,
        artifact_manifest=artifact_manifest,
    )
    print(f"[train.py] Artifacts saved: {file_map}")

    leaderboard_path = Path(file_map["version_dir"]) / "leaderboard.csv"
    result["leaderboard"].to_csv(leaderboard_path, index=False)
    print(f"[train.py] Leaderboard saved to: {leaderboard_path}")

    summary = {
        "project_name": "diabetes",
        "version": version,
        "dataset_source": dataset_source,
        "selected_model": model_name,
        "best_params": best_params,
        "metrics": metrics,
        "artifact_dir": str(file_map["version_dir"]),
        "leaderboard_path": str(leaderboard_path),
    }

    print("[train.py] Pipeline returning summary.")
    return summary


if __name__ == "__main__":
    print("[train.py] __main__ block started.")
    try:
        summary = run_diabetes_training_pipeline()
        print("Diabetes training pipeline completed.")
        for key, value in summary.items():
            print(f"{key}: {value}")
    except Exception as exc:
        print("[train.py] ERROR while running training pipeline:")
        print(exc)
        traceback.print_exc()