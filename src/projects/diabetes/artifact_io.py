"""
Artifact I/O helpers for the diabetes project.

This module defines how diabetes-specific artifact bundles are saved
and loaded using the shared artifact infrastructure.
"""

from __future__ import annotations

from src.shared.artifact_store import (
    load_standard_artifact_bundle,
    save_standard_artifact_bundle,
)


PROJECT_NAME = "diabetes"


def save_diabetes_artifact_bundle(
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
    Save a standardized diabetes artifact bundle.
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


def load_diabetes_artifact_bundle(version: str | None = None) -> dict:
    """
    Load a standardized diabetes artifact bundle.

    If version is None, the latest version is loaded.
    """
    return load_standard_artifact_bundle(
        project_name=PROJECT_NAME,
        version=version,
    )