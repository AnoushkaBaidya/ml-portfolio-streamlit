"""
Data-loading utilities for the Spotify popularity prediction project.

This module is responsible only for:
- loading Spotify track data from local files, API, Kaggle, or synthetic fallback
- defining the feature schema used by the project

It does NOT handle feature scaling, model training, explainability, plotting,
or Streamlit rendering.
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
AUDIO_FEATURES = [
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

TARGET_COL = "popularity"
META_COLS = ["track_name", "artist_name", "album_name"]


def _validate_spotify_schema(df: pd.DataFrame) -> bool:
    """
    Validate whether a DataFrame contains the required Spotify project columns.
    """
    required = set(AUDIO_FEATURES + [TARGET_COL] + META_COLS)
    return required.issubset(df.columns)


def generate_synthetic_spotify_data(
    n_rows: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic Spotify-like dataset with realistic audio feature distributions.

    Parameters
    ----------
    n_rows : int
        Number of tracks to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic Spotify track dataset.
    """
    rng = np.random.RandomState(random_state)

    df = pd.DataFrame(
        {
            "track_name": [f"Track_{i}" for i in range(n_rows)],
            "artist_name": [f"Artist_{rng.randint(0, 500)}" for _ in range(n_rows)],
            "album_name": [f"Album_{rng.randint(0, 1000)}" for _ in range(n_rows)],
            "duration_ms": rng.randint(60_000, 420_000, n_rows),
            "explicit": rng.choice([0, 1], n_rows, p=[0.7, 0.3]),
            "danceability": rng.beta(5, 3, n_rows),
            "energy": rng.beta(5, 3, n_rows),
            "key": rng.randint(0, 12, n_rows),
            "loudness": rng.uniform(-60, 0, n_rows),
            "mode": rng.choice([0, 1], n_rows, p=[0.4, 0.6]),
            "speechiness": rng.beta(1.5, 10, n_rows),
            "acousticness": rng.beta(1.5, 5, n_rows),
            "instrumentalness": rng.beta(1, 10, n_rows),
            "liveness": rng.beta(2, 8, n_rows),
            "valence": rng.beta(3, 3, n_rows),
            "tempo": rng.uniform(50, 200, n_rows),
            "time_signature": rng.choice([3, 4, 5, 6, 7], n_rows, p=[0.05, 0.75, 0.1, 0.05, 0.05]),
        }
    )

    # Create a learnable popularity signal from audio features plus noise.
    df[TARGET_COL] = np.clip(
        (
            20
            + 30 * df["danceability"]
            + 20 * df["energy"]
            + 10 * df["valence"]
            + 0.15 * df["loudness"]
            - 15 * df["acousticness"]
            - 10 * df["instrumentalness"]
            + 5 * df["explicit"]
            + rng.normal(0, 8, n_rows)
        ),
        0,
        100,
    ).astype(int)

    return df


def _load_from_local_csv(csv_path: Path) -> pd.DataFrame | None:
    """
    Load Spotify data from a local CSV and normalize some common alternative columns.
    """
    try:
        df = pd.read_csv(csv_path)

        if "track_name" not in df.columns and "name" in df.columns:
            df = df.rename(columns={"name": "track_name"})

        if "artist_name" not in df.columns and "artists" in df.columns:
            df = df.rename(columns={"artists": "artist_name"})

        for meta_col in META_COLS:
            if meta_col not in df.columns:
                df[meta_col] = "Unknown"

        if _validate_spotify_schema(df):
            return df
    except Exception:
        return None

    return None


def _fetch_from_spotify_api() -> pd.DataFrame | None:
    """
    Attempt to fetch Spotify tracks and audio features using the Spotify Web API.

    Returns None when:
    - spotipy is not installed
    - credentials are missing
    - API calls fail
    - audio feature retrieval is incomplete
    """
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
    except Exception:
        return None

    try:
        spotify_client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(),
            requests_timeout=10,
        )
    except Exception:
        return None

    playlist_ids = [
        "37i9dQZF1DXcBWIGoYBM5M",
        "37i9dQZF1DX0XUsuxWHRQd",
        "37i9dQZF1DWXRqgorJj26U",
        "37i9dQZF1DX4sWSpwq3LiO",
        "37i9dQZF1DX1lVhptIYRda",
        "37i9dQZF1DXa8NOEUWPn9W",
        "37i9dQZF1DX4JAvHpjipBk",
        "37i9dQZF1DX10zKzsJ2jva",
    ]

    tracks: list[dict] = []
    seen_ids: set[str] = set()

    for playlist_id in playlist_ids:
        try:
            results = spotify_client.playlist_tracks(playlist_id, limit=100)
            for item in results.get("items", []):
                track = item.get("track")
                if not track or not track.get("id"):
                    continue

                track_id = track["id"]
                if track_id in seen_ids:
                    continue

                seen_ids.add(track_id)

                tracks.append(
                    {
                        "track_id": track_id,
                        "track_name": track.get("name", "Unknown"),
                        "artist_name": track["artists"][0]["name"] if track.get("artists") else "Unknown",
                        "album_name": track["album"]["name"] if track.get("album") else "Unknown",
                        "popularity": track.get("popularity", 0),
                        "explicit": int(track.get("explicit", False)),
                        "duration_ms": track.get("duration_ms", 0),
                    }
                )
        except Exception:
            continue

    if not tracks:
        return None

    audio_keys = [feature for feature in AUDIO_FEATURES if feature not in ("duration_ms", "explicit")]
    track_ids = [track["track_id"] for track in tracks]

    for batch_start in range(0, len(track_ids), 100):
        batch_ids = track_ids[batch_start: batch_start + 100]

        try:
            features_list = spotify_client.audio_features(batch_ids)
        except Exception:
            return None

        for offset, features in enumerate(features_list or []):
            if features is None:
                continue

            idx = batch_start + offset
            for key in audio_keys:
                tracks[idx][key] = features.get(key, 0)

    df = pd.DataFrame(tracks)

    required = AUDIO_FEATURES + [TARGET_COL] + META_COLS
    if not all(column in df.columns for column in required):
        return None

    df = df.dropna(subset=AUDIO_FEATURES).reset_index(drop=True)
    return df if len(df) >= 50 else None


def _load_from_kaggle() -> pd.DataFrame | None:
    """
    Attempt to download a Spotify tracks dataset from Kaggle.

    Returns None when Kaggle is unavailable or the schema is invalid.
    """
    try:
        import kagglehub
    except Exception:
        return None

    try:
        dataset_path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")

        for file_path in Path(dataset_path).iterdir():
            if file_path.suffix.lower() != ".csv":
                continue

            df = pd.read_csv(file_path)

            if "track_name" not in df.columns and "name" in df.columns:
                df = df.rename(columns={"name": "track_name"})

            if "artist_name" not in df.columns and "artists" in df.columns:
                df = df.rename(columns={"artists": "artist_name"})

            for meta_col in META_COLS:
                if meta_col not in df.columns:
                    df[meta_col] = "Unknown"

            if _validate_spotify_schema(df):
                return df
    except Exception:
        return None

    return None


@st.cache_data
def load_spotify_data() -> tuple[pd.DataFrame, str]:
    """
    Load Spotify data using the portfolio-wide lookup strategy.

    Load priority
    -------------
    1. artifacts/processed/spotify_tracks.csv
    2. artifacts/raw/spotify_tracks.csv
    3. artifacts/spotify_tracks.csv
    4. data/spotify_tracks.csv
    5. Spotify API
    6. Kaggle download
    7. Synthetic fallback

    Returns
    -------
    tuple[pd.DataFrame, str]
        Loaded DataFrame and a human-readable source label.
    """
    dataset_path = find_dataset_path(
        candidate_filenames=["spotify_tracks.csv", "spotify.csv"],
        project_root=PROJECT_ROOT,
    )

    if dataset_path is not None:
        local_df = _load_from_local_csv(dataset_path)
        if local_df is not None:
            return local_df, f"Loaded local dataset"

    api_df = _fetch_from_spotify_api()
    if api_df is not None:
        return api_df, "Loaded dataset from Spotify API"

    kaggle_df = _load_from_kaggle()
    if kaggle_df is not None:
        return kaggle_df, "Loaded dataset from Kaggle"

    synthetic_df = generate_synthetic_spotify_data()
    return synthetic_df, "Loaded synthetic dataset"