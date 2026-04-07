"""
Data-loading utilities for the telecom customer churn project.

This module is responsible only for:
- loading churn data from local files, Kaggle, or synthetic fallback
- defining project-specific constants
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.shared.data_loader import find_dataset_path
from src.shared.paths import PROJECT_ROOT


RANDOM_STATE = 42
TARGET_COL = "Churn"


def generate_synthetic_churn_data(n_rows: int = 7000) -> pd.DataFrame:
    """
    Generate a realistic synthetic telecom churn dataset.

    The feature relationships are designed so churn is more likely for:
    - short-tenure customers
    - month-to-month contracts
    - no tech support / no security
    - higher monthly charges
    - electronic check customers
    """
    rng = np.random.default_rng(RANDOM_STATE)

    genders = rng.choice(["Male", "Female"], n_rows)
    senior = rng.choice([0, 1], n_rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n_rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n_rows, p=[0.30, 0.70])

    tenure = rng.exponential(scale=32, size=n_rows).clip(1, 72).astype(int)

    phone_service = rng.choice(["Yes", "No"], n_rows, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["Yes", "No"], n_rows, p=[0.42, 0.58]),
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], n_rows, p=[0.34, 0.44, 0.22]
    )

    def internet_dependent(yes_prob: float) -> np.ndarray:
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], n_rows, p=[yes_prob, 1 - yes_prob]),
        )

    online_security = internet_dependent(0.29)
    online_backup = internet_dependent(0.34)
    device_protection = internet_dependent(0.34)
    tech_support = internet_dependent(0.29)
    streaming_tv = internet_dependent(0.38)
    streaming_movies = internet_dependent(0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.21, 0.24]
    )
    paperless = rng.choice(["Yes", "No"], n_rows, p=[0.60, 0.40])
    payment = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n_rows,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    base_charge = np.where(
        internet_service == "Fiber optic",
        rng.normal(80, 15, n_rows),
        np.where(
            internet_service == "DSL",
            rng.normal(55, 12, n_rows),
            rng.normal(25, 8, n_rows),
        ),
    ).clip(18, 120)
    monthly_charges = np.round(base_charge, 2)

    total_charges = np.round(
        monthly_charges * tenure * rng.uniform(0.95, 1.05, n_rows), 2
    )

    logit = (
        -2.0
        + (-0.03 * tenure)
        + (1.2 * (contract == "Month-to-month"))
        + (-0.7 * (contract == "Two year"))
        + (0.5 * (internet_service == "Fiber optic"))
        + (-0.4 * (online_security == "Yes"))
        + (-0.4 * (tech_support == "Yes"))
        + (0.3 * (paperless == "Yes"))
        + (0.5 * (payment == "Electronic check"))
        + (0.01 * monthly_charges)
        + (0.3 * senior)
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = np.where(rng.random(n_rows) < probability, "Yes", "No")

    return pd.DataFrame(
        {
            "customerID": [f"C{str(i).zfill(5)}" for i in range(1, n_rows + 1)],
            "gender": genders,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            TARGET_COL: churn,
        }
    )


def _load_from_local_csv(csv_path: Path) -> pd.DataFrame | None:
    """
    Load churn data from a local CSV and coerce TotalCharges to numeric.
    """
    try:
        df = pd.read_csv(csv_path)
        if TARGET_COL not in df.columns:
            return None

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.dropna(subset=["TotalCharges"], inplace=True)
        return df
    except Exception:
        return None


def _load_from_kaggle() -> pd.DataFrame | None:
    """
    Attempt to download the telco churn dataset from Kaggle.
    """
    try:
        import kagglehub
    except Exception:
        return None

    try:
        dataset_path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        for file_path in Path(dataset_path).iterdir():
            if file_path.suffix.lower() != ".csv":
                continue
            df = _load_from_local_csv(file_path)
            if df is not None:
                return df
    except Exception:
        return None

    return None


@st.cache_data
def load_churn_data() -> tuple[pd.DataFrame, str]:
    """
    Load churn data using the portfolio-wide lookup strategy.
    """
    dataset_path = find_dataset_path(
        candidate_filenames=["churn.csv", "telco_churn.csv"],
        project_root=PROJECT_ROOT,
    )

    if dataset_path is not None:
        local_df = _load_from_local_csv(dataset_path)
        if local_df is not None:
            return local_df, f"Loaded local dataset from: {dataset_path}"

    kaggle_df = _load_from_kaggle()
    if kaggle_df is not None:
        return kaggle_df, "Loaded dataset from Kaggle"

    synthetic_df = generate_synthetic_churn_data()
    return synthetic_df, "Loaded synthetic dataset"