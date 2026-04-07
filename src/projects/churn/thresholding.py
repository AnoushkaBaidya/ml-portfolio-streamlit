"""
Threshold-tuning helpers for the churn project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def compute_threshold_metrics(probabilities, y_true, threshold: float) -> dict:
    """
    Compute classification metrics at a specific threshold.
    """
    predictions = (probabilities >= threshold).astype(int)

    return {
        "predictions": predictions,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "accuracy": accuracy_score(y_true, predictions),
        "confusion_matrix": confusion_matrix(y_true, predictions),
    }


def compute_precision_recall_curve_data(probabilities, y_true):
    """
    Compute precision-recall curve arrays.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
    return precisions, recalls, thresholds


def build_threshold_sweep_dataframe(probabilities, y_true) -> pd.DataFrame:
    """
    Evaluate precision, recall, and F1 across a range of thresholds.
    """
    threshold_values = np.arange(0.05, 0.96, 0.01)
    rows = []

    for threshold in threshold_values:
        predictions = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "Threshold": round(float(threshold), 2),
                "Precision": precision_score(y_true, predictions, zero_division=0),
                "Recall": recall_score(y_true, predictions, zero_division=0),
                "F1": f1_score(y_true, predictions, zero_division=0),
            }
        )

    return pd.DataFrame(rows)