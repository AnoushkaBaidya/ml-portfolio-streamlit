"""
Feature engineering and train/test utilities for the Spotify project.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.projects.spotify.data import AUDIO_FEATURES, TARGET_COL


@st.cache_data
def get_spotify_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix for Spotify numeric features.
    """
    return df[AUDIO_FEATURES + [TARGET_COL]].corr()


@st.cache_data
def split_and_scale_spotify_data(df: pd.DataFrame):
    """
    Split Spotify data into train/test sets and apply standard scaling.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, scaler
    """
    X = df[AUDIO_FEATURES].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler