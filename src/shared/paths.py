"""
Path helper utilities for the ML portfolio application.

This module will later provide shared logic for:
- finding datasets in artifacts/ and data/
- resolving project-relative paths
- supporting local and deployed environments cleanly
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_RAW_DIR = ARTIFACTS_DIR / "raw"
ARTIFACTS_PROCESSED_DIR = ARTIFACTS_DIR / "processed"
DATA_DIR = PROJECT_ROOT / "data"