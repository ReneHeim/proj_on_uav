"""Unit tests for the canonical research-workflow support library."""

from __future__ import annotations

import math

import numpy as np

from src.research.common import (
    grouped_rmse_delta_bootstrap,
    load_artifact_catalog,
    markdown_table,
    project_root,
    regression_metrics,
)


def test_project_root_contains_repository_metadata() -> None:
    root = project_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "src").is_dir()


def test_artifact_catalog_is_named_and_relative() -> None:
    catalog = load_artifact_catalog()
    assert "severity.current.selected_multiangular_predictions" in catalog
    assert all(not path.startswith("/") for path in catalog.values())


def test_regression_metrics_are_consistent() -> None:
    metrics = regression_metrics(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 3.0]))
    assert math.isclose(metrics["rmse"], math.sqrt(1 / 3))
    assert math.isclose(metrics["mae"], 1 / 3)
    assert math.isclose(metrics["bias"], 1 / 3)


def test_grouped_bootstrap_reports_positive_improvement() -> None:
    y_true = np.array([0.0, 1.0, 0.0, 1.0])
    candidate = np.array([0.0, 1.0, 0.0, 1.0])
    baseline = np.array([1.0, 0.0, 1.0, 0.0])
    result = grouped_rmse_delta_bootstrap(
        y_true, candidate, baseline, np.array(["a", "a", "b", "b"]), iterations=100, seed=7
    )
    assert result["rmse_reduction"] > 0
    assert result["rmse_reduction_prob_gt_zero"] == 1.0


def test_markdown_table_supports_record_lists() -> None:
    rendered = markdown_table([{"model": "ridge", "rmse": 3.4567}], float_digits=2)
    assert "| model | rmse |" in rendered
    assert "3.46" in rendered
