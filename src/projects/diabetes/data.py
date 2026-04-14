"""
Data-loading utilities for the diabetes prediction project.

This module is responsible only for:
- defining dataset columns
- locating and loading a diabetes dataset if available
- generating a realistic synthetic fallback dataset if not

It does NOT train models or render Streamlit UI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.shared.data_loader import find_dataset_path
from src.shared.paths import PROJECT_ROOT


# ---------------------------------------------------------------------
# Project-specific constants
# ---------------------------------------------------------------------
FEATURE_COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COL = "Outcome"


def generate_synthetic_diabetes_data(
    n_rows: int = 768,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic diabetes dataset with realistic distributions.

    The synthetic data is designed to roughly mimic the shape and ranges
    of the Pima Indians Diabetes dataset so the ML workflow remains
    meaningful even when a real CSV is not available.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic diabetes dataset with the expected feature and target columns.
    """
    rng = np.random.RandomState(random_state)

    data = {
        "Pregnancies": rng.poisson(lam=3.8, size=n_rows).clip(0, 17),
        "Glucose": rng.normal(loc=121, scale=32, size=n_rows).clip(0, 200).astype(int),
        "BloodPressure": rng.normal(loc=69, scale=19, size=n_rows).clip(0, 122).astype(int),
        "SkinThickness": rng.normal(loc=21, scale=16, size=n_rows).clip(0, 99).astype(int),
        "Insulin": rng.normal(loc=80, scale=115, size=n_rows).clip(0, 846).astype(int),
        "BMI": rng.normal(loc=32, scale=8, size=n_rows).clip(0, 67).round(1),
        "DiabetesPedigreeFunction": rng.exponential(scale=0.47, size=n_rows).clip(0.078, 2.42).round(3),
        "Age": rng.normal(loc=33, scale=12, size=n_rows).clip(21, 81).astype(int),
    }

    df = pd.DataFrame(data)

    # Create a probabilistic target correlated with clinically meaningful drivers.
    logit = (
        -8
        + 0.03 * df["Glucose"]
        + 0.05 * df["BMI"]
        + 0.02 * df["Age"]
        + 0.15 * df["Pregnancies"]
    )
    probability = 1 / (1 + np.exp(-logit))
    df[TARGET_COL] = rng.binomial(1, probability)

    return df


def _load_from_local_csv(csv_path: Path) -> pd.DataFrame | None:
    """
    Load a diabetes dataset from a local CSV if it matches the expected schema.
    """
    try:
        df = pd.read_csv(csv_path)
        expected_cols = set(FEATURE_COLS + [TARGET_COL])
        if expected_cols.issubset(df.columns):
            return df
    except Exception:
        return None

    return None


def _load_from_kaggle() -> pd.DataFrame | None:
    """
    Attempt to download the diabetes dataset from Kaggle using kagglehub.

    Returns None when:
    - kagglehub is not installed
    - Kaggle credentials are not configured
    - the download fails
    - the CSV schema does not match expectations
    """
    try:
        import kagglehub  # optional dependency

        dataset_path = kagglehub.dataset_download(
            "uciml/pima-indians-diabetes-database"
        )
        csv_path = Path(dataset_path) / "diabetes.csv"
        if csv_path.exists():
            return _load_from_local_csv(csv_path)
    except Exception:
        return None

    return None


@st.cache_data
def load_diabetes_data() -> tuple[pd.DataFrame, str]:
    """
    Load the diabetes dataset using the portfolio-wide lookup strategy.

    Load priority
    -------------
    1. artifacts/processed/diabetes.csv
    2. artifacts/raw/diabetes.csv
    3. artifacts/diabetes.csv
    4. data/diabetes.csv
    5. Kaggle download
    6. Synthetic fallback

    Returns
    -------
    tuple[pd.DataFrame, str]
        A tuple of:
        - loaded DataFrame
        - human-readable data source description
    """
    dataset_path = find_dataset_path(
        candidate_filenames=["diabetes.csv", "pima_diabetes.csv"],
        project_root=PROJECT_ROOT,
    )

    if dataset_path is not None:
        local_df = _load_from_local_csv(dataset_path)
        if local_df is not None:
            return local_df, f"Loaded local dataset"

    kaggle_df = _load_from_kaggle()
    if kaggle_df is not None:
        return kaggle_df, "Loaded dataset from Kaggle"

    synthetic_df = generate_synthetic_diabetes_data()
    return synthetic_df, "Loaded synthetic dataset"