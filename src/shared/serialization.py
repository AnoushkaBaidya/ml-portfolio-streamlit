"""
Serialization helpers for saving and loading ML artifacts.

This module centralizes how Python objects such as trained models,
scalers, encoders, and pipelines are written to disk and restored later.

Why this matters
----------------
A production-style ML project should not retrain at inference time.
Instead, it should load versioned, pre-trained artifacts from disk.
"""

from __future__ import annotations

from pathlib import Path
import json
import joblib


def save_joblib_artifact(obj, path: Path) -> None:
    """
    Save a Python object to disk using joblib.

    Parameters
    ----------
    obj : Any
        Python object to serialize.
    path : Path
        Target file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib_artifact(path: Path):
    """
    Load a joblib artifact from disk.

    Parameters
    ----------
    path : Path
        Path to the serialized artifact.

    Returns
    -------
    Any
        Deserialized Python object.
    """
    return joblib.load(path)


def save_json(data: dict, path: Path) -> None:
    """
    Save a dictionary as a formatted JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    """
    Load a dictionary from a JSON file.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)