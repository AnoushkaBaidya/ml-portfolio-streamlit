"""
General-purpose utility helpers for the ML portfolio application.
"""

from __future__ import annotations

from pathlib import Path

from src.shared.serialization import load_json
from src.shared.paths import PROJECT_ROOT


def load_project_training_config(filename: str) -> dict:
    """
    Load a project training config JSON from the configs directory.

    Parameters
    ----------
    filename : str
        Example: "diabetes_train.json"

    Returns
    -------
    dict
        Parsed config dictionary.
    """
    config_path = PROJECT_ROOT / "configs" / filename
    return load_json(config_path)