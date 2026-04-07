"""
Model-training and evaluation utilities for the diabetes prediction project.

This module contains:
- train/test splitting
- scaling comparisons
- model benchmarking
- hyperparameter tuning
- offline best-model selection utilities
- exploratory runtime helpers for UI tabs

This module does NOT render Streamlit UI directly.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from src.projects.diabetes.data import FEATURE_COLS, TARGET_COL


def get_diabetes_models() -> dict:
    """
    Return the baseline model collection used in the comparison tab.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }


def get_diabetes_training_grids() -> dict:
    """
    Return model objects and hyperparameter grids used for offline training.
    """
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.01, 0.1, 1, 10]},
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, None],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
            },
        },
    }


@st.cache_data
def split_diabetes_data(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
):
    """
    Split the diabetes dataset into train and test partitions.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


@st.cache_data
def evaluate_scaling_strategies(
    X_train,
    X_test,
    y_train,
    y_test,
) -> pd.DataFrame:
    """
    Compare logistic regression performance under different scaling strategies.
    """
    scalers = {
        "No Scaling": None,
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
    }

    rows = []

    for scaler_name, scaler in scalers.items():
        X_train_current = X_train.copy()
        X_test_current = X_test.copy()

        if scaler is not None:
            X_train_current = scaler.fit_transform(X_train_current)
            X_test_current = scaler.transform(X_test_current)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_current, y_train)

        predictions = model.predict(X_test_current)
        probabilities = model.predict_proba(X_test_current)[:, 1]

        rows.append(
            {
                "Scaling": scaler_name,
                "Accuracy": round(accuracy_score(y_test, predictions), 4),
                "F1": round(f1_score(y_test, predictions), 4),
                "ROC-AUC": round(roc_auc_score(y_test, probabilities), 4),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data
def compare_diabetes_models(
    X_train,
    X_test,
    y_train,
    y_test,
) -> pd.DataFrame:
    """
    Train all diabetes models on standardized data and compare performance.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rows = []

    for model_name, model in get_diabetes_models().items():
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]

        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(accuracy_score(y_test, predictions), 4),
                "F1": round(f1_score(y_test, predictions), 4),
                "ROC-AUC": round(roc_auc_score(y_test, probabilities), 4),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data
def run_diabetes_grid_search(
    X_train,
    y_train,
    model_name: str,
    test_size: float,
    random_state: int,
) -> dict:
    """
    Run GridSearchCV for the selected diabetes model.

    Parameters include test_size and random_state so Streamlit cache keys
    update when the user changes sidebar settings.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    grids = get_diabetes_training_grids()
    config = grids[model_name]

    search = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    search.fit(X_train_scaled, y_train)

    return {
        "best_params": search.best_params_,
        "best_score": round(search.best_score_, 4),
        "cv_results": pd.DataFrame(search.cv_results_)[
            ["params", "mean_test_score", "rank_test_score"]
        ].sort_values("rank_test_score"),
    }


def fit_models_for_visuals(
    X_train,
    X_test,
    y_train,
):
    """
    Fit diabetes models on standardized data for confusion matrices and ROC curves.

    Returns
    -------
    tuple
        scaler, X_test_scaled, fitted_model_dict
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    fitted_models = {}

    for model_name, model in get_diabetes_models().items():
        model.fit(X_train_scaled, y_train)
        fitted_models[model_name] = model

    return scaler, X_test_scaled, fitted_models


def predict_diabetes_outcome(
    X_train,
    y_train,
    patient_features,
) -> dict:
    """
    Exploratory runtime prediction helper.

    This is retained for educational comparison purposes, but production
    inference should use the artifact-backed inference module instead.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)

    patient_scaled = scaler.transform(patient_features)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(scaler.transform(X_train), y_train)

    prediction = model.predict(patient_scaled)[0]
    probabilities = model.predict_proba(patient_scaled)[0]

    return {
        "prediction": int(prediction),
        "probabilities": probabilities,
    }


def train_best_diabetes_model(
    X_train,
    X_test,
    y_train,
    y_test,
    selection_metric: str = "ROC-AUC",
) -> dict:
    """
    Train and select the best diabetes model offline using GridSearchCV.

    Returns
    -------
    dict
        {
            "model_name": ...,
            "model": fitted_best_model,
            "scaler": fitted_scaler,
            "best_params": ...,
            "test_metrics": ...,
            "leaderboard": ...
        }
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    grids = get_diabetes_training_grids()
    leaderboard_rows = []
    best_package = None
    best_score = float("-inf")

    for model_name, config in grids.items():
        search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )
        search.fit(X_train_scaled, y_train)

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test_scaled)
        probabilities = best_model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "Accuracy": round(accuracy_score(y_test, predictions), 4),
            "F1": round(f1_score(y_test, predictions), 4),
            "ROC-AUC": round(roc_auc_score(y_test, probabilities), 4),
        }

        leaderboard_rows.append(
            {
                "Model": model_name,
                "Best Params": search.best_params_,
                "Accuracy": metrics["Accuracy"],
                "F1": metrics["F1"],
                "ROC-AUC": metrics["ROC-AUC"],
            }
        )

        score = metrics.get(selection_metric, metrics["ROC-AUC"])
        if score > best_score:
            best_score = score
            best_package = {
                "model_name": model_name,
                "model": best_model,
                "scaler": scaler,
                "best_params": search.best_params_,
                "test_metrics": metrics,
            }

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(
        by=selection_metric,
        ascending=False,
    )

    best_package["leaderboard"] = leaderboard_df
    return best_package