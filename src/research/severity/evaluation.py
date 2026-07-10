"""Severity-specific wrappers around general regression metrics."""

from __future__ import annotations

import numpy as np

from src.research.common import regression_metrics


def severity_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Score continuous disease-severity predictions using project conventions."""
    return regression_metrics(y_true, y_pred)
