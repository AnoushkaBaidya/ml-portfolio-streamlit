"""
Tests for the shared artifact store helpers.
"""

from src.shared.artifact_store import (
    get_artifact_file_map,
    get_project_artifact_root,
    get_version_dir,
)


def test_project_artifact_root_path():
    path = get_project_artifact_root("diabetes")
    assert "artifacts" in str(path)
    assert "processed" in str(path)
    assert "diabetes" in str(path)


def test_version_dir_path():
    path = get_version_dir("spotify", "v1")
    assert "spotify" in str(path)
    assert "v1" in str(path)


def test_artifact_file_map_keys():
    file_map = get_artifact_file_map("churn", "v2")
    expected_keys = {
        "version_dir",
        "model",
        "preprocessors",
        "metrics",
        "feature_schema",
        "training_config",
        "model_card",
        "artifact_manifest",
    }
    assert expected_keys == set(file_map.keys())