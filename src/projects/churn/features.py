"""
Feature engineering and preprocessing utilities for the churn project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.projects.churn.data import RANDOM_STATE, TARGET_COL


@st.cache_data
def engineer_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed features for churn prediction.
    """
    out = df.copy()

    out["tenure_bin"] = pd.cut(
        out["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["New", "Mid", "Loyal", "Long-term"],
    )

    out["AvgMonthlyCharge"] = np.round(out["TotalCharges"] / out["tenure"].clip(1), 2)
    out["ChargeRatio"] = np.round(out["MonthlyCharges"] / (out["TotalCharges"] + 1), 4)

    service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    out["NumServices"] = out[service_cols].apply(lambda row: (row == "Yes").sum(), axis=1)

    out["HasInternet"] = (out["InternetService"] != "No").astype(int)
    out["IsAutoPayment"] = (
        out["PaymentMethod"].isin(
            ["Bank transfer (automatic)", "Credit card (automatic)"]
        )
    ).astype(int)

    return out


@st.cache_data
def preprocess_churn_data(df: pd.DataFrame):
    """
    One-hot encode categoricals, split into train/test, and scale features.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    drop_cols = ["customerID", "tenure_bin"]
    model_df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    y = (model_df[TARGET_COL] == "Yes").astype(int)
    X = model_df.drop(columns=[TARGET_COL])

    X = pd.get_dummies(X, drop_first=True)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_names,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_names,
    )

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        feature_names,
        scaler,
    )