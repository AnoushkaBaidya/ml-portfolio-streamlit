"""
Feature engineering utilities for the Netflix clustering project.

This module transforms raw Netflix catalog data into a numeric feature
matrix suitable for clustering.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler


@st.cache_data
def engineer_netflix_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Convert raw Netflix data into numeric clustering features.

    Steps
    -----
    1. Label-encode the rating column
    2. One-hot encode the top genres
    3. Extract numeric duration
    4. Add binary movie flag

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, list[str]]
        clean_df     : enriched human-readable DataFrame
        encoded_df   : numeric feature matrix for clustering
        genre_cols   : one-hot encoded genre columns
    """
    clean = df.copy()

    # Rating encoding
    rating_encoder = LabelEncoder()
    clean["rating_encoded"] = rating_encoder.fit_transform(clean["rating"].fillna("NR"))

    # One-hot encode top genres
    top_genres = (
        clean["listed_in"]
        .fillna("")
        .str.split(", ")
        .explode()
        .value_counts()
        .head(12)
        .index.tolist()
    )

    for genre in top_genres:
        clean[f"genre_{genre}"] = clean["listed_in"].fillna("").str.contains(genre, regex=False).astype(int)

    genre_cols = [col for col in clean.columns if col.startswith("genre_")]

    # Numeric duration
    clean["duration_num"] = (
        clean["duration"]
        .fillna("0")
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
        .fillna(0)
    )

    # Binary movie flag
    clean["is_movie"] = (clean["type"] == "Movie").astype(int)

    feature_cols = ["release_year", "rating_encoded", "duration_num", "is_movie"] + genre_cols
    encoded_df = clean[feature_cols].copy()

    return clean, encoded_df, genre_cols


@st.cache_data
def scale_netflix_features(encoded_df: pd.DataFrame) -> tuple:
    """
    Standard-scale the numeric feature matrix for KMeans clustering.

    Returns
    -------
    tuple
        scaled_array, feature_names
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(encoded_df)
    return scaled, encoded_df.columns.tolist()