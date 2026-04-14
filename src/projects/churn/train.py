"""
Offline training pipeline for the churn project.

Run this script manually to:
- load data
- engineer features
- preprocess
- train candidate classifiers
- select the best model
- save versioned production artifacts
"""

from __future__ import annotations

from pathlib import Path

from src.projects.churn.artifact_io import save_churn_artifact_bundle
from src.projects.churn.data import TARGET_COL, load_churn_data
from src.projects.churn.features import engineer_churn_features, preprocess_churn_data
from src.projects.churn.models import train_best_churn_model
from src.shared.metadata import build_artifact_manifest, build_model_card
from src.shared.schema import build_feature_schema
from src.shared.utils import load_project_training_config


def run_churn_training_pipeline() -> dict:
    """
    Execute the full offline churn training pipeline and save artifacts.
    """
    config = load_project_training_config("churn_train.json")
    version = config["version"]

    raw_df, dataset_source = load_churn_data()
    df = engineer_churn_features(raw_df)

    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_churn_data(df)

    result = train_best_churn_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        selection_metric=config["selection_metric"],
        use_smote=config["use_smote_for_production"],
        default_threshold=config["default_threshold"],
    )

    model_name = result["model_name"]
    model = result["model"]
    metrics = result["test_metrics"]
    recommended_threshold = result["recommended_threshold"]

    preprocessors = {
        "scaler": scaler,
        "feature_order": feature_names,
        "recommended_threshold": recommended_threshold,
    }

    feature_schema = build_feature_schema(
        feature_names=feature_names,
        target_name=TARGET_COL,
    )

    model_card = build_model_card(
        project_name="churn",
        version=version,
        model_name=model_name,
        dataset_source=dataset_source,
        metrics=metrics,
        hyperparameters={"recommended_threshold": recommended_threshold},
        notes="Selected offline using candidate churn model comparison and saved for production inference.",
    )

    artifact_manifest = build_artifact_manifest(
        project_name="churn",
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

    file_map = save_churn_artifact_bundle(
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
        "project_name": "churn",
        "version": version,
        "dataset_source": dataset_source,
        "selected_model": model_name,
        "metrics": metrics,
        "recommended_threshold": recommended_threshold,
        "artifact_dir": str(file_map["version_dir"]),
        "leaderboard_path": str(leaderboard_path),
    }


if __name__ == "__main__":
    summary = run_churn_training_pipeline()
    print("Churn training pipeline completed.")
    for key, value in summary.items():
        print(f"{key}: {value}")