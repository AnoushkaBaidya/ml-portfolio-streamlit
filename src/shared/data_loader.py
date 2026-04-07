"""
Shared dataset-loading utilities for the ML portfolio application.

This module centralizes the logic used by all projects to locate datasets.
The goal is to keep dataset discovery consistent across the portfolio.

Dataset lookup order
--------------------
1. artifacts/processed/
2. artifacts/raw/
3. artifacts/
4. data/

Why this matters
----------------
- Keeps data-loading behavior predictable across projects
- Allows easy swapping between real datasets and synthetic fallbacks
- Avoids duplicating path logic inside every project module
"""

from pathlib import Path
from typing import Iterable


def find_dataset_path(
    candidate_filenames: Iterable[str],
    project_root: Path,
) -> Path | None:
    """
    Search common portfolio dataset locations for the first matching file.

    Parameters
    ----------
    candidate_filenames : Iterable[str]
        One or more possible dataset filenames to search for.
        Example: ["diabetes.csv", "pima_diabetes.csv"]
    project_root : Path
        Absolute path to the repository root.

    Returns
    -------
    Path | None
        The first matching dataset path if found, otherwise None.

    Notes
    -----
    Search order is intentionally fixed so all projects behave consistently.
    """
    search_dirs = [
        project_root / "artifacts" / "processed",
        project_root / "artifacts" / "raw",
        project_root / "artifacts",
        project_root / "data",
    ]

    for directory in search_dirs:
        for filename in candidate_filenames:
            candidate_path = directory / filename
            if candidate_path.exists():
                return candidate_path

    return None