"""
Offline training pipeline for the Netflix clustering project.

Run this script manually to:
- load data
- engineer features
- scale features
- evaluate candidate K values
- choose the best production K
- fit final KMeans and PCA artifacts
- save versioned production artifacts
"""

from __future__ import annotations

from pathlib import Path

from sklearn.preprocessing import StandardScaler

from src.projects.netflix.artifact_io import save_netflix_artifact_bundle
from src.projects.netflix.clustering import (
    compute_k_selection_metrics,
    fit_final_netflix_kmeans,
    fit_netflix_pca,
)
from src.projects.netflix.data import load_netflix_data
from src.projects.netflix.features import engineer_netflix_features
from src.shared.metadata import build_artifact_manifest, build_model_card
from src.shared.schema import build_feature_schema
from src.shared.utils import load_project_training_config


def run_netflix_training_pipeline() -> dict:
    """
    Execute the full offline Netflix clustering pipeline and save artifacts.
    """
    config = load_project_training_config("netflix_train.json")
    version = config["version"]

    df, dataset_source = load_netflix_data()
    clean_df, encoded_df, genre_cols = engineer_netflix_features(df)

    feature_order = encoded_df.columns.tolist()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(encoded_df)

    metrics_df = compute_k_selection_metrics(
        scaled_features=scaled_features,
        k_values=config["candidate_k_values"],
    )

    best_row = metrics_df.sort_values(by="Silhouette", ascending=False).iloc[0]
    best_k = int(best_row["K"])

    final_model = fit_final_netflix_kmeans(
        scaled_features=scaled_features,
        k=best_k,
    )

    final_pca_2d = fit_netflix_pca(
        scaled_features=scaled_features,
        n_components=config["pca_components_2d"],
    )

    final_pca_3d = fit_netflix_pca(
        scaled_features=scaled_features,
        n_components=config["pca_components_3d"],
    )

    metrics = {
        "Best K": best_k,
        "Silhouette": round(float(best_row["Silhouette"]), 4),
        "Davies-Bouldin": round(float(best_row["Davies-Bouldin"]), 4),
        "Calinski-Harabasz": round(float(best_row["Calinski-Harabasz"]), 4),
        "Inertia": round(float(best_row["Inertia"]), 4),
    }

    preprocessors = {
        "scaler": scaler,
        "pca_2d": final_pca_2d,
        "pca_3d": final_pca_3d,
        "feature_order": feature_order,
        "genre_columns": genre_cols,
    }

    feature_schema = build_feature_schema(
        feature_names=feature_order,
        target_name=None,
    )

    model_card = build_model_card(
        project_name="netflix",
        version=version,
        model_name="KMeans",
        dataset_source=dataset_source,
        metrics=metrics,
        hyperparameters={"n_clusters": best_k, "n_init": 10, "random_state": 42},
        notes="Selected offline using candidate K comparison across multiple clustering metrics.",
    )

    artifact_manifest = build_artifact_manifest(
        project_name="netflix",
        version=version,
        model_name="KMeans",
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

    file_map = save_netflix_artifact_bundle(
        version=version,
        model=final_model,
        preprocessors=preprocessors,
        metrics=metrics,
        feature_schema=feature_schema,
        training_config=config,
        model_card=model_card,
        artifact_manifest=artifact_manifest,
    )

    metrics_path = Path(file_map["version_dir"]) / "k_selection_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    return {
        "project_name": "netflix",
        "version": version,
        "dataset_source": dataset_source,
        "selected_model": "KMeans",
        "best_k": best_k,
        "metrics": metrics,
        "artifact_dir": str(file_map["version_dir"]),
        "k_selection_metrics_path": str(metrics_path),
    }


if __name__ == "__main__":
    summary = run_netflix_training_pipeline()
    print("Netflix training pipeline completed.")
    for key, value in summary.items():
        print(f"{key}: {value}")