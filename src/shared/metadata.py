"""
Metadata builders for ML artifact bundles.

This module creates structured metadata files that describe:
- what model was selected
- when it was trained
- what version it belongs to
- what dataset source was used
- what metrics were achieved
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_timestamp() -> str:
    """
    Return the current UTC timestamp in ISO 8601 format.
    """
    return datetime.now(timezone.utc).isoformat()


def build_artifact_manifest(
    project_name: str,
    version: str,
    model_name: str,
    dataset_source: str,
    artifact_paths: dict[str, str],
) -> dict:
    """
    Build a high-level manifest describing an artifact bundle.
    """
    return {
        "project_name": project_name,
        "version": version,
        "model_name": model_name,
        "dataset_source": dataset_source,
        "created_at_utc": utc_timestamp(),
        "artifact_paths": artifact_paths,
    }


def build_model_card(
    project_name: str,
    version: str,
    model_name: str,
    dataset_source: str,
    metrics: dict,
    hyperparameters: dict,
    notes: str = "",
) -> dict:
    """
    Build a model card summarizing the selected production artifact.
    """
    return {
        "project_name": project_name,
        "version": version,
        "model_name": model_name,
        "dataset_source": dataset_source,
        "created_at_utc": utc_timestamp(),
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "notes": notes,
    }