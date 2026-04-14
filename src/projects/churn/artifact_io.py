"""
Artifact I/O helpers for the churn project.
"""

from __future__ import annotations

from src.shared.artifact_store import (
    load_standard_artifact_bundle,
    save_standard_artifact_bundle,
)

PROJECT_NAME = "churn"


def save_churn_artifact_bundle(
    version: str,
    model,
    preprocessors: dict,
    metrics: dict,
    feature_schema: dict,
    training_config: dict,
    model_card: dict,
    artifact_manifest: dict,
):
    """
    Save a standardized churn artifact bundle.
    """
    return save_standard_artifact_bundle(
        project_name=PROJECT_NAME,
        version=version,
        model=model,
        preprocessors=preprocessors,
        metrics=metrics,
        feature_schema=feature_schema,
        training_config=training_config,
        model_card=model_card,
        artifact_manifest=artifact_manifest,
    )


def load_churn_artifact_bundle(version: str | None = None) -> dict:
    """
    Load a standardized churn artifact bundle.

    If version is None, the latest version is loaded.
    """
    return load_standard_artifact_bundle(
        project_name=PROJECT_NAME,
        version=version,
    )