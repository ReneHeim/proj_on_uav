"""Shared infrastructure for ONCERCO analysis and plotting scripts.

This is the single import surface for recurring workflow concerns: paths,
output directories, logging, metrics, report tables, artifact lookup, and
standard figure/prediction persistence. Model-specific algorithms remain in
their analysis modules.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr


def project_root() -> Path:
    """Return the repository root without relying on the current directory."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunPaths:
    """Standard output locations for one reproducible workflow run."""

    root: Path
    results: Path
    figures: Path
    reports: Path
    logs: Path
    manifests: Path

    @classmethod
    def create(cls, relative_root: str | Path) -> "RunPaths":
        root = project_root() / "outputs" / Path(relative_root)
        paths = cls(
            root=root,
            results=root / "results",
            figures=root / "figures",
            reports=root / "reports",
            logs=root / "logs",
            manifests=root / "manifests",
        )
        for path in (paths.results, paths.figures, paths.reports, paths.logs, paths.manifests):
            path.mkdir(parents=True, exist_ok=True)
        return paths


def configure_logging(log_dir: Path, workflow: str) -> Path:
    """Configure stdout and per-run file logging for a named workflow."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{workflow}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    logging.info("Log file: %s", log_path)
    return log_path


def log_phase(name: str, started: float, *, wall_clock: bool = False) -> float:
    """Log and return elapsed seconds for a major workflow phase."""
    elapsed = (time.time() if wall_clock else time.perf_counter()) - started
    logging.info("[PHASE] %s: %.1fs", name, elapsed)
    return elapsed


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Spearman correlation, or NaN when either input is constant."""
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return math.nan
    value = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return float(value) if value is not None else math.nan


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the project-standard continuous severity metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "rmse": math.sqrt(float(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": 1.0 - float(np.sum(residual**2) / total) if total else math.nan,
        "spearman": safe_spearman(y_true, y_pred),
        "bias": float(np.mean(residual)),
    }


def grouped_rmse_delta_bootstrap(
    y_true: np.ndarray,
    candidate_pred: np.ndarray,
    baseline_pred: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int = 1_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap RMSE reduction, resampling whole experimental groups."""
    y_true = np.asarray(y_true, dtype=float)
    candidate_pred = np.asarray(candidate_pred, dtype=float)
    baseline_pred = np.asarray(baseline_pred, dtype=float)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        raise ValueError("Grouped bootstrap requires at least two distinct groups.")

    def rmse(prediction: np.ndarray, rows: np.ndarray | None = None) -> float:
        rows = np.arange(y_true.size) if rows is None else rows
        return math.sqrt(float(np.mean((prediction[rows] - y_true[rows]) ** 2)))

    observed = rmse(baseline_pred) - rmse(candidate_pred)
    rng = np.random.default_rng(seed)
    group_rows = [np.flatnonzero(groups == group) for group in unique_groups]
    deltas = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = rng.integers(0, len(group_rows), size=len(group_rows))
        rows = np.concatenate([group_rows[item] for item in sampled])
        deltas[index] = rmse(baseline_pred, rows) - rmse(candidate_pred, rows)
    return {
        "rmse_reduction": observed,
        "rmse_reduction_ci_low": float(np.quantile(deltas, alpha / 2)),
        "rmse_reduction_ci_high": float(np.quantile(deltas, 1 - alpha / 2)),
        "rmse_reduction_prob_gt_zero": float(np.mean(deltas > 0)),
        "n_bootstrap": float(iterations),
        "n_groups": float(unique_groups.size),
    }


def markdown_table(rows: Any, *, float_digits: int = 3, max_rows: int | None = None) -> str:
    """Render pandas/Polars tables or record lists as compact Markdown."""
    if hasattr(rows, "to_dicts"):
        records = rows.to_dicts()
    elif hasattr(rows, "to_dict"):
        records = rows.to_dict(orient="records")
    else:
        records = list(rows)
    if not records:
        return "_No rows._"
    if max_rows is not None:
        records = records[:max_rows]
    columns = list(records[0])

    def format_value(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, (float, np.floating)):
            return "" if math.isnan(float(value)) else f"{float(value):.{float_digits}f}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(format_value(row.get(column, "")) for column in columns) + " |" for row in records]
    return "\n".join([header, divider, *body])


def write_report(path: Path, content: str) -> Path:
    """Write a UTF-8 Markdown report and log its exact location."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logging.info("Wrote report: %s", path)
    return path


def save_figure(
    figure: Any,
    stem: Path,
    formats: tuple[str, ...] = ("png", "pdf", "svg"),
    dpi: int = 300,
) -> list[Path]:
    """Save a Matplotlib-compatible figure in the requested publication formats."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in formats:
        path = stem.with_suffix(f".{suffix}")
        figure.savefig(path, dpi=dpi)
        paths.append(path)
    return paths


def prediction_path(directory: Path, *parts: str, suffix: str = ".csv") -> Path:
    """Build a normalized prediction filename from model descriptor parts."""
    normalized = ["".join(char if char.isalnum() or char == "_" else "_" for char in value).strip("_") for value in parts]
    return directory / ("_".join(part.lower() for part in normalized if part) + suffix)


def save_predictions(frame: Any, path: Path) -> Path:
    """Persist a pandas or Polars prediction table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(frame, "write_csv"):
        frame.write_csv(path)
    else:
        frame.to_csv(path, index=False)
    logging.info("Wrote predictions: %s", path)
    return path


def load_artifact_catalog(catalog_path: Path | None = None) -> dict[str, str]:
    """Load artifact IDs to repository-relative paths."""
    path = catalog_path or project_root() / "configs" / "outputs.yaml"
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    artifacts = data.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError(f"Invalid artifact catalog: {path}")
    return {str(key): str(value) for key, value in artifacts.items()}


def artifact_path(name: str, catalog_path: Path | None = None) -> Path:
    """Resolve a named artifact and fail with an actionable message."""
    catalog = load_artifact_catalog(catalog_path)
    if name not in catalog:
        raise KeyError(f"Unknown artifact '{name}'. Available: {', '.join(sorted(catalog))}")
    return project_root() / catalog[name]
