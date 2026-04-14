"""
Data-loading utilities for the Netflix clustering project.

This module is responsible only for:
- loading Netflix catalog data from local files, Kaggle, or synthetic fallback
- storing project-specific constants used for realistic data generation

It does NOT handle feature engineering, clustering, plotting, or UI rendering.
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
RATINGS = [
    "TV-MA",
    "TV-14",
    "TV-PG",
    "R",
    "PG-13",
    "TV-Y7",
    "TV-Y",
    "PG",
    "TV-G",
    "NR",
    "G",
    "NC-17",
]

GENRE_POOL = [
    "International Movies",
    "Dramas",
    "Comedies",
    "Action & Adventure",
    "Documentaries",
    "Children & Family Movies",
    "Romantic Movies",
    "Horror Movies",
    "Thrillers",
    "Stand-Up Comedy",
    "Sci-Fi & Fantasy",
    "Crime TV Shows",
    "TV Dramas",
    "Reality TV",
    "Kids' TV",
    "Anime Series",
    "Classic Movies",
    "Music & Musicals",
    "Independent Movies",
    "British TV Shows",
]

COUNTRY_POOL = [
    "United States",
    "India",
    "United Kingdom",
    "Canada",
    "France",
    "Japan",
    "South Korea",
    "Spain",
    "Mexico",
    "Australia",
    "Germany",
    "Nigeria",
    "Brazil",
    "Turkey",
    "Egypt",
]


def generate_synthetic_netflix_data(
    n_rows: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic Netflix-style dataset with realistic distributions.

    Parameters
    ----------
    n_rows : int
        Number of rows to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic Netflix catalog dataset.
    """
    rng = np.random.RandomState(random_state)

    content_types = rng.choice(["Movie", "TV Show"], size=n_rows, p=[0.70, 0.30])
    release_years = rng.normal(loc=2015, scale=6, size=n_rows).clip(1970, 2023).astype(int)

    rating_weights = np.array([0.22, 0.21, 0.12, 0.10, 0.09, 0.06, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01])
    ratings = rng.choice(RATINGS, size=n_rows, p=rating_weights)

    genres_list = []
    for _ in range(n_rows):
        num_genres = rng.choice([1, 2, 3], p=[0.30, 0.50, 0.20])
        selected_genres = rng.choice(GENRE_POOL, size=num_genres, replace=False)
        genres_list.append(", ".join(selected_genres))

    durations = []
    for content_type in content_types:
        if content_type == "Movie":
            minutes = int(np.clip(rng.normal(100, 25), 30, 240))
            durations.append(f"{minutes} min")
        else:
            seasons = int(rng.choice([1, 2, 3, 4, 5], p=[0.45, 0.25, 0.15, 0.10, 0.05]))
            durations.append(f"{seasons} Season{'s' if seasons != 1 else ''}")

    first_names = ["Alex", "Jordan", "Sam", "Chris", "Pat", "Morgan", "Taylor", "Casey", "Riley", "Jamie", "Drew", "Quinn"]
    last_names = ["Smith", "Kim", "Garcia", "Chen", "Patel", "Müller", "Tanaka", "Silva", "Johansson", "Ali", "Brown", "Okafor"]
    adjectives = ["thrilling", "heartwarming", "gripping", "hilarious", "thought-provoking", "visually stunning", "intense", "charming", "dark", "inspiring"]
    nouns = ["journey", "story", "adventure", "mystery", "tale", "saga", "drama", "comedy", "documentary", "series"]

    directors = []
    casts = []
    descriptions = []
    countries = []
    dates_added = []

    for _ in range(n_rows):
        directors.append(f"{rng.choice(first_names)} {rng.choice(last_names)}")

        cast_size = rng.randint(2, 6)
        casts.append(
            ", ".join(f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(cast_size))
        )

        descriptions.append(
            f"A {rng.choice(adjectives)} {rng.choice(nouns)} that explores {rng.choice(adjectives)} themes."
        )

        countries.append(rng.choice(COUNTRY_POOL))

        month = rng.randint(1, 13)
        day = rng.randint(1, 29)
        year = rng.choice([2019, 2020, 2021, 2022, 2023])
        dates_added.append(f"{month:02d}/{day:02d}/{year}")

    df = pd.DataFrame(
        {
            "show_id": [f"s{i + 1}" for i in range(n_rows)],
            "type": content_types,
            "title": [f"Title {i + 1}" for i in range(n_rows)],
            "director": directors,
            "cast": casts,
            "country": countries,
            "date_added": dates_added,
            "release_year": release_years,
            "rating": ratings,
            "duration": durations,
            "listed_in": genres_list,
            "description": descriptions,
        }
    )

    return df


def _load_from_local_csv(csv_path: Path) -> pd.DataFrame | None:
    """
    Load a Netflix dataset from a local CSV if the expected columns exist.
    """
    try:
        df = pd.read_csv(csv_path)
        expected = {"type", "title", "rating", "listed_in", "release_year"}
        if expected.issubset(df.columns):
            return df
    except Exception:
        return None

    return None


def _load_from_kaggle() -> pd.DataFrame | None:
    """
    Attempt to download the Netflix titles dataset from Kaggle.

    Returns None when Kaggle access is unavailable or the schema is invalid.
    """
    try:
        import kagglehub  # optional dependency

        dataset_path = kagglehub.dataset_download("shivamb/netflix-shows")
        csv_path = Path(dataset_path) / "netflix_titles.csv"
        if csv_path.exists():
            return _load_from_local_csv(csv_path)
    except Exception:
        return None

    return None


@st.cache_data
def load_netflix_data() -> tuple[pd.DataFrame, str]:
    """
    Load the Netflix dataset using the portfolio-wide lookup strategy.

    Load priority
    -------------
    1. artifacts/processed/netflix_titles.csv
    2. artifacts/raw/netflix_titles.csv
    3. artifacts/netflix_titles.csv
    4. data/netflix_titles.csv
    5. Kaggle download
    6. Synthetic fallback

    Returns
    -------
    tuple[pd.DataFrame, str]
        Loaded DataFrame and a human-readable source label.
    """
    dataset_path = find_dataset_path(
        candidate_filenames=["netflix_titles.csv", "netflix.csv"],
        project_root=PROJECT_ROOT,
    )

    if dataset_path is not None:
        local_df = _load_from_local_csv(dataset_path)
        if local_df is not None:
            return local_df, f"Loaded local dataset"

    kaggle_df = _load_from_kaggle()
    if kaggle_df is not None:
        return kaggle_df, "Loaded dataset from Kaggle"

    synthetic_df = generate_synthetic_netflix_data()
    return synthetic_df, "Loaded synthetic dataset"