"""
Schema helpers for feature validation and artifact metadata.

This module stores and validates model input expectations such as:
- feature names
- feature order
- categorical options
- numeric ranges
"""

from __future__ import annotations


def build_feature_schema(
    feature_names: list[str],
    target_name: str | None = None,
    numeric_ranges: dict | None = None,
    categorical_options: dict | None = None,
) -> dict:
    """
    Build a structured feature schema dictionary.
    """
    return {
        "feature_names": feature_names,
        "target_name": target_name,
        "numeric_ranges": numeric_ranges or {},
        "categorical_options": categorical_options or {},
    }


def validate_required_features(input_payload: dict, feature_names: list[str]) -> tuple[bool, list[str]]:
    """
    Validate that all required features are present in an input payload.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, missing_features)
    """
    missing = [feature for feature in feature_names if feature not in input_payload]
    return len(missing) == 0, missing