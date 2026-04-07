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


@st.cache_resource
def train_churn_models(X_train, y_train, use_smote: bool = False) -> dict:
    """
    Train churn models, optionally applying SMOTE before fitting.
    """
    X_train_current = X_train.copy()
    y_train_current = y_train.copy()

    if use_smote:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_current, y_train_current = smote.fit_resample(
            X_train_current, y_train_current
        )

    models = {
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

    for model in models.values():
        model.fit(X_train_current, y_train_current)

    return models


@st.cache_data
def evaluate_churn_models(_models: dict, X_test, y_test, threshold: float = 0.5) -> pd.DataFrame:
    """
    Evaluate churn models at a chosen decision threshold.

    The models argument is prefixed with an underscore so Streamlit
    excludes it from cache hashing.
    """
    rows = []

    for model_name, model in _models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(accuracy_score(y_test, predictions), 4),
                "Precision": round(
                    precision_score(y_test, predictions, zero_division=0), 4
                ),
                "Recall": round(
                    recall_score(y_test, predictions, zero_division=0), 4
                ),
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