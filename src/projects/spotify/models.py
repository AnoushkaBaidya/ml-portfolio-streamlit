"""
Model-training and evaluation utilities for the Spotify popularity prediction project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


@st.cache_resource
def train_spotify_models(X_train, X_test, y_train, y_test) -> dict:
    """
    Train all Spotify regression models and return fitted models plus metrics.

    Returns
    -------
    dict
        Mapping of model name to:
        {
            "model": fitted_model,
            "mae": ...,
            "rmse": ...,
            "r2": ...
        }
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        ),
    }

    results: dict[str, dict] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results[model_name] = {
            "model": model,
            "mae": mean_absolute_error(y_test, predictions),
            "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "r2": r2_score(y_test, predictions),
        }

    return results


def build_spotify_metrics_dataframe(results: dict) -> pd.DataFrame:
    """
    Convert the model results dictionary into a comparison DataFrame.
    """
    metrics_df = pd.DataFrame(
        {
            model_name: {
                "MAE": result["mae"],
                "RMSE": result["rmse"],
                "R²": result["r2"],
            }
            for model_name, result in results.items()
        }
    ).T.round(4)

    metrics_df.index.name = "Model"
    return metrics_df


def get_best_spotify_model_name(results: dict) -> str:
    """
    Return the model name with the highest R² score.
    """
    return max(results, key=lambda model_name: results[model_name]["r2"])