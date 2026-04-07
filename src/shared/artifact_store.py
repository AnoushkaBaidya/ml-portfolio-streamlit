"""
Artifact store helpers for versioned ML model bundles.

This module standardizes where artifacts live and how the application
finds the latest or a specific model version for each project.

Artifact layout
---------------
artifacts/
└── processed/
    └── <project_name>/
        └── <version>/
            ├── model.joblib
            ├── preprocessors.joblib
            ├── metrics.json
            ├── feature_schema.json
            ├── training_config.json
            ├── model_card.json
            └── artifact_manifest.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.shared.paths import ARTIFACTS_PROCESSED_DIR
from src.shared.serialization import save_joblib_artifact, save_json

def get_project_artifact_root(project_name: str) -> Path:
    """
    Return the artifact root folder for a project.

    Example
    -------
    artifacts/processed/diabetes
    """
    return ARTIFACTS_PROCESSED_DIR / project_name


def get_version_dir(project_name: str, version: str) -> Path:
    """
    Return the artifact directory for a specific project version.
    """
    return get_project_artifact_root(project_name) / version


def list_available_versions(project_name: str) -> list[str]:
    """
    List available version directories for a project.

    Returns versions sorted lexicographically.
    """
    project_root = get_project_artifact_root(project_name)
    if not project_root.exists():
        return []

    versions = [path.name for path in project_root.iterdir() if path.is_dir()]
    return sorted(versions)


def get_latest_version(project_name: str) -> Optional[str]:
    """
    Return the latest available version for a project.

    Current strategy:
    - uses lexicographic sorting
    - assumes versions are named consistently like v1, v2, v3
      or sortable timestamps

    Returns None if no version exists.
    """
    versions = list_available_versions(project_name)
    if not versions:
        return None
    return versions[-1]


def ensure_version_dir(project_name: str, version: str) -> Path:
    """
    Create and return the version directory for a project.
    """
    version_dir = get_version_dir(project_name, version)
    version_dir.mkdir(parents=True, exist_ok=True)
    return version_dir


def get_artifact_file_map(project_name: str, version: str) -> dict[str, Path]:
    """
    Return the standard file paths for an artifact bundle.
    """
    version_dir = get_version_dir(project_name, version)

    return {
        "version_dir": version_dir,
        "model": version_dir / "model.joblib",
        "preprocessors": version_dir / "preprocessors.joblib",
        "metrics": version_dir / "metrics.json",
        "feature_schema": version_dir / "feature_schema.json",
        "training_config": version_dir / "training_config.json",
        "model_card": version_dir / "model_card.json",
        "artifact_manifest": version_dir / "artifact_manifest.json",
    }


def artifact_bundle_exists(project_name: str, version: str) -> bool:
    """
    Check whether the minimum required files for a bundle exist.
    """
    files = get_artifact_file_map(project_name, version)
    required = [
        files["model"],
        files["preprocessors"],
        files["metrics"],
        files["feature_schema"],
        files["artifact_manifest"],
    ]
    return all(path.exists() for path in required)


def save_standard_artifact_bundle(
    project_name: str,
    version: str,
    model,
    preprocessors: dict,
    metrics: dict,
    feature_schema: dict,
    training_config: dict,
    model_card: dict,
    artifact_manifest: dict,
) -> dict[str, Path]:
    """
    Save a complete standard artifact bundle for a project version.

    Returns
    -------
    dict[str, Path]
        File map pointing to the saved bundle files.
    """
    ensure_version_dir(project_name, version)
    file_map = get_artifact_file_map(project_name, version)

    save_joblib_artifact(model, file_map["model"])
    save_joblib_artifact(preprocessors, file_map["preprocessors"])
    save_json(metrics, file_map["metrics"])
    save_json(feature_schema, file_map["feature_schema"])
    save_json(training_config, file_map["training_config"])
    save_json(model_card, file_map["model_card"])
    save_json(artifact_manifest, file_map["artifact_manifest"])

    return file_map


def load_standard_artifact_bundle(project_name: str, version: str | None = None) -> dict:
    """
    Load a standard artifact bundle for a project.

    If version is None, the latest available version is loaded.
    """
    resolved_version = version or get_latest_version(project_name)
    if resolved_version is None:
        raise FileNotFoundError(f"No artifact versions found for project '{project_name}'.")

    if not artifact_bundle_exists(project_name, resolved_version):
        raise FileNotFoundError(
            f"Artifact bundle for project '{project_name}' version '{resolved_version}' is incomplete."
        )

    file_map = get_artifact_file_map(project_name, resolved_version)

    from src.shared.serialization import load_joblib_artifact, load_json

    return {
        "version": resolved_version,
        "model": load_joblib_artifact(file_map["model"]),
        "preprocessors": load_joblib_artifact(file_map["preprocessors"]),
        "metrics": load_json(file_map["metrics"]),
        "feature_schema": load_json(file_map["feature_schema"]),
        "training_config": load_json(file_map["training_config"]),
        "model_card": load_json(file_map["model_card"]),
        "artifact_manifest": load_json(file_map["artifact_manifest"]),
        "file_map": file_map,
    }