"""
Model-training and evaluation utilities for the churn project.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.projects.churn.data import RANDOM_STATE


def get_churn_model_registry() -> dict:
    """
    Return the candidate churn models.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }


@st.cache_data
def train_churn_models(X_train, y_train, use_smote: bool = False) -> dict:
    """
    Train churn models, optionally applying SMOTE before fitting.
    """
    X_train_current = X_train.copy()
    y_train_current = y_train.copy()

    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_current, y_train_current = smote.fit_resample(X_train_current, y_train_current)

    models = get_churn_model_registry()

    for model in models.values():
        model.fit(X_train_current, y_train_current)

    return models


@st.cache_data
def evaluate_churn_models(_models: dict, X_test, y_test, threshold: float = 0.5) -> pd.DataFrame:
    """
    Evaluate churn models at a chosen decision threshold.
    """
    rows = []

    for model_name, model in _models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(accuracy_score(y_test, predictions), 4),
                "Precision": round(precision_score(y_test, predictions, zero_division=0), 4),
                "Recall": round(recall_score(y_test, predictions, zero_division=0), 4),
                "F1": round(f1_score(y_test, predictions, zero_division=0), 4),
                "ROC-AUC": round(roc_auc_score(y_test, probabilities), 4),
            }
        )

    return pd.DataFrame(rows)


def get_churn_feature_importance(model, feature_names: list[str], top_n: int = 15) -> pd.Series:
    """
    Extract top feature importances from a fitted tree-based churn model.
    """
    importance_series = pd.Series(model.feature_importances_, index=feature_names)
    return importance_series.nlargest(top_n).sort_values()


def train_best_churn_model(
    X_train,
    X_test,
    y_train,
    y_test,
    selection_metric: str = "ROC-AUC",
    use_smote: bool = True,
    default_threshold: float = 0.5,
) -> dict:
    """
    Train churn models offline and select the best one.

    Returns
    -------
    dict
        {
            "model_name": ...,
            "model": fitted_model,
            "test_metrics": ...,
            "leaderboard": ...,
            "recommended_threshold": ...
        }
    """
    X_train_current = X_train.copy()
    y_train_current = y_train.copy()

    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_current, y_train_current = smote.fit_resample(X_train_current, y_train_current)

    registry = get_churn_model_registry()
    leaderboard_rows = []
    best_package = None
    best_score = float("-inf")

    for model_name, model in registry.items():
        model.fit(X_train_current, y_train_current)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= default_threshold).astype(int)

        metrics = {
            "Accuracy": round(accuracy_score(y_test, predictions), 4),
            "Precision": round(precision_score(y_test, predictions, zero_division=0), 4),
            "Recall": round(recall_score(y_test, predictions, zero_division=0), 4),
            "F1": round(f1_score(y_test, predictions, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, probabilities), 4),
        }

        leaderboard_rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"],
                "ROC-AUC": metrics["ROC-AUC"],
            }
        )

        score = metrics[selection_metric]
        if score > best_score:
            best_score = score
            best_package = {
                "model_name": model_name,
                "model": model,
                "test_metrics": metrics,
                "recommended_threshold": default_threshold,
            }

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(
        by=selection_metric,
        ascending=False,
    )

    best_package["leaderboard"] = leaderboard_df
    return best_package