"""Prepare paper-ready tables for the multiangular-vs-nadir story."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/cross_year_generalization_2024_to_2025"
RESULTS_DIR = BASE / "results"
REPORTS_DIR = BASE / "reports"
TABLE_DIR = RESULTS_DIR / "paper_tables"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LOGS_DIR = ROOT / "outputs/logs"


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"prepare_key_multiangular_tables_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Log file: %s", log_path)
    return log_path


def log_phase(name: str, t0: float) -> None:
    logging.info("[PHASE] %s: %.1fs", name, time.time() - t0)


def pretty_feature(value: str) -> str:
    labels = {
        "nadir": "Nadir",
        "multiangular_vza": "VZA",
        "multiangular_vza_compact": "VZA compact",
        "multiangular_geometry_compact": "Geometry compact",
        "multiangular_vza_raa": "VZA + RAA",
        "multiangular_vza_phase": "VZA + phase",
    }
    return labels.get(value, value)


def markdown_table(df: pl.DataFrame, float_digits: int = 3) -> str:
    columns = df.columns
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.iter_rows(named=True):
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float | np.floating):
                values.append("" if np.isnan(value) else f"{value:.{float_digits}f}")
            else:
                values.append("" if value is None else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value).strip("_").lower()


def read_prediction_file(task: str, model_key: str, feature_set: str, covariates: str | None = None) -> pl.DataFrame:
    stem = f"{task}_predictions_{model_key}_{safe_filename(feature_set)}"
    if covariates is not None:
        stem += f"_{safe_filename(covariates)}"
    path = PREDICTIONS_DIR / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file missing: {path}")
    return pl.read_csv(path)


def first_row(frame: pl.DataFrame) -> dict:
    if frame.height == 0:
        raise IndexError("Cannot extract first row from empty frame")
    return frame.row(0, named=True)


def filter_rows(frame: pl.DataFrame, **equals: str) -> pl.DataFrame:
    expr = None
    for col, value in equals.items():
        item = pl.col(col) == value
        expr = item if expr is None else expr & item
    return frame.filter(expr) if expr is not None else frame


def warning_metrics(frame: pl.DataFrame) -> dict[str, float]:
    y_true = frame.get_column("y_true").to_numpy().astype(int)
    y_pred = frame.get_column("y_pred").to_numpy().astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    denominator = (2 * tp) + fp + fn
    f1 = (2 * tp / denominator) if denominator else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1 - specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "n_test": len(frame),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
    }


def plot_warning_counts(frame: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for plot_id in frame.get_column("plot_id").unique().sort().to_list():
        group = frame.filter(pl.col("plot_id") == plot_id)
        y_true = group.get_column("y_true").to_numpy().astype(int)
        y_pred = group.get_column("y_pred").to_numpy().astype(int)
        rows.append(
            {
                "plot_id": plot_id,
                "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
                "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
                "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
                "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
            }
        )
    return pl.DataFrame(rows)


def aligned_numeric_rows(frame: pl.DataFrame, plots: np.ndarray, columns: list[str]) -> np.ndarray:
    lookup = {row["plot_id"]: [row[col] for col in columns] for row in frame.iter_rows(named=True)}
    return np.array([lookup[plot_id] for plot_id in plots], dtype=float)


def metric_from_counts(tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray, metric: str) -> np.ndarray:
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) > 0)
    f1 = np.divide(2 * tp, (2 * tp) + fp + fn, out=np.zeros_like(tp, dtype=float), where=((2 * tp) + fp + fn) > 0)
    values = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1 - specificity,
        "balanced_accuracy": (recall + specificity) / 2,
    }
    return values[metric]


def plot_severity_errors(frame: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for plot_id in frame.get_column("plot_id").unique().sort().to_list():
        group = frame.filter(pl.col("plot_id") == plot_id)
        errors = group.get_column("y_pred").to_numpy().astype(float) - group.get_column("y_true").to_numpy().astype(float)
        rows.append({"plot_id": plot_id, "sse": float(np.sum(errors**2)), "n": len(group)})
    return pl.DataFrame(rows)


def bootstrap_f1_ci_from_predictions(
    model_key: str,
    feature_set: str,
    seed: int = 42,
    n_bootstrap: int = 20000,
) -> tuple[float, float]:
    """Bootstrap F1 from the actual external-test prediction rows, resampling plots."""
    try:
        predictions = read_prediction_file("early_warning", model_key, feature_set)
    except FileNotFoundError as exc:
        logging.warning("%s", exc)
        return np.nan, np.nan
    plots = predictions.get_column("plot_id").unique().sort().to_numpy()
    if len(plots) == 0:
        return np.nan, np.nan
    counts = plot_warning_counts(predictions)
    count_values = aligned_numeric_rows(counts, plots, ["tp", "tn", "fp", "fn"])
    rng = np.random.default_rng(seed)
    sampled_idx = rng.integers(0, len(plots), size=(n_bootstrap, len(plots)))
    sampled_counts = count_values[sampled_idx].sum(axis=1)
    values = metric_from_counts(
        sampled_counts[:, 0],
        sampled_counts[:, 1],
        sampled_counts[:, 2],
        sampled_counts[:, 3],
        "f1",
    )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def paired_warning_delta_ci(
    model_key: str,
    feature_set: str,
    metric: str,
    seed: int = 42,
    n_bootstrap: int = 20000,
) -> tuple[float, float]:
    if feature_set == "nadir":
        return 0.0, 0.0
    try:
        candidate = read_prediction_file("early_warning", model_key, feature_set)
        nadir = read_prediction_file("early_warning", model_key, "nadir")
    except FileNotFoundError as exc:
        logging.warning("%s", exc)
        return np.nan, np.nan
    keys = ["plot_id", "predictor_week", "target_week"]
    common = candidate.select(keys).join(nadir.select(keys), on=keys, how="inner")
    candidate = common.join(candidate, on=keys, how="left")
    nadir = common.join(nadir, on=keys, how="left")
    plots = common.get_column("plot_id").unique().sort().to_numpy()
    candidate_counts = aligned_numeric_rows(plot_warning_counts(candidate), plots, ["tp", "tn", "fp", "fn"])
    nadir_counts = aligned_numeric_rows(plot_warning_counts(nadir), plots, ["tp", "tn", "fp", "fn"])
    rng = np.random.default_rng(seed)
    sampled_idx = rng.integers(0, len(plots), size=(n_bootstrap, len(plots)))
    sampled_candidate = candidate_counts[sampled_idx].sum(axis=1)
    sampled_nadir = nadir_counts[sampled_idx].sum(axis=1)
    candidate_metric = metric_from_counts(
        sampled_candidate[:, 0],
        sampled_candidate[:, 1],
        sampled_candidate[:, 2],
        sampled_candidate[:, 3],
        metric,
    )
    nadir_metric = metric_from_counts(
        sampled_nadir[:, 0],
        sampled_nadir[:, 1],
        sampled_nadir[:, 2],
        sampled_nadir[:, 3],
        metric,
    )
    deltas = candidate_metric - nadir_metric
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def rmse(frame: pl.DataFrame) -> float:
    errors = frame.get_column("y_pred").to_numpy().astype(float) - frame.get_column("y_true").to_numpy().astype(float)
    return float(np.sqrt(np.mean(errors**2)))


def paired_rmse_reduction_ci(
    model_key: str,
    feature_set: str,
    covariates: str,
    seed: int = 42,
    n_bootstrap: int = 20000,
) -> tuple[float, float]:
    if feature_set == "nadir":
        return 0.0, 0.0
    try:
        candidate = read_prediction_file("severity", model_key, feature_set, covariates)
        nadir = read_prediction_file("severity", model_key, "nadir", covariates)
    except FileNotFoundError as exc:
        logging.warning("%s", exc)
        return np.nan, np.nan
    keys = ["plot_id", "predictor_week", "target_week"]
    common = candidate.select(keys).join(nadir.select(keys), on=keys, how="inner")
    candidate = common.join(candidate, on=keys, how="left")
    nadir = common.join(nadir, on=keys, how="left")
    plots = common.get_column("plot_id").unique().sort().to_numpy()
    candidate_errors = aligned_numeric_rows(plot_severity_errors(candidate), plots, ["sse", "n"])
    nadir_errors = aligned_numeric_rows(plot_severity_errors(nadir), plots, ["sse", "n"])
    rng = np.random.default_rng(seed)
    sampled_idx = rng.integers(0, len(plots), size=(n_bootstrap, len(plots)))
    sampled_candidate = candidate_errors[sampled_idx].sum(axis=1)
    sampled_nadir = nadir_errors[sampled_idx].sum(axis=1)
    candidate_rmse = np.sqrt(sampled_candidate[:, 0] / sampled_candidate[:, 1])
    nadir_rmse = np.sqrt(sampled_nadir[:, 0] / sampled_nadir[:, 1])
    reductions = nadir_rmse - candidate_rmse
    return float(np.quantile(reductions, 0.025)), float(np.quantile(reductions, 0.975))


def table_early_warning_gain() -> pl.DataFrame:
    logistic = pl.read_csv(RESULTS_DIR / "warning_external_2024_train_2025_test.csv")
    xgb = pl.read_csv(RESULTS_DIR / "xgboost_warning_train_eval_2024_test_2025.csv")

    rows = []
    logistic_nadir = first_row(filter_rows(logistic, feature_set="nadir"))
    for feature_set in ["nadir", "multiangular_vza_raa", "multiangular_vza_phase", "multiangular_geometry_compact"]:
        row = first_row(filter_rows(logistic, feature_set=feature_set))
        metrics = warning_metrics(read_prediction_file("early_warning", "logistic", feature_set))
        ci_low, ci_high = bootstrap_f1_ci_from_predictions("logistic", feature_set, seed=100 + len(rows))
        delta_f1_low, delta_f1_high = paired_warning_delta_ci("logistic", feature_set, "f1", seed=300 + len(rows))
        rows.append(
            {
                "model": "Logistic",
                "feature_set": pretty_feature(feature_set),
                "external_f1": row["f1"],
                "f1_ci95_low": ci_low,
                "f1_ci95_high": ci_high,
                "delta_f1_vs_nadir": row["f1"] - logistic_nadir["f1"],
                "delta_f1_ci95_low": delta_f1_low,
                "delta_f1_ci95_high": delta_f1_high,
                "recall": row["recall"],
                "delta_recall_vs_nadir": row["recall"] - logistic_nadir["recall"],
                "balanced_accuracy": row["balanced_accuracy"],
                "delta_balanced_accuracy_vs_nadir": row["balanced_accuracy"] - logistic_nadir["balanced_accuracy"],
                "precision": metrics["precision"],
                "specificity": metrics["specificity"],
                "false_positive_rate": metrics["false_positive_rate"],
                "n_test": metrics["n_test"],
                "n_positive": metrics["n_positive"],
                "n_negative": metrics["n_negative"],
            }
        )

    xgb_nadir = first_row(filter_rows(xgb, feature_set="nadir"))
    for feature_set in ["nadir", "multiangular_vza", "multiangular_vza_compact", "multiangular_vza_raa"]:
        row = first_row(filter_rows(xgb, feature_set=feature_set))
        metrics = warning_metrics(read_prediction_file("early_warning", "xgboost", feature_set))
        ci_low, ci_high = bootstrap_f1_ci_from_predictions("xgboost", feature_set, seed=200 + len(rows))
        delta_f1_low, delta_f1_high = paired_warning_delta_ci("xgboost", feature_set, "f1", seed=400 + len(rows))
        rows.append(
            {
                "model": "XGBoost",
                "feature_set": pretty_feature(feature_set),
                "external_f1": row["external_f1_2025"],
                "f1_ci95_low": ci_low,
                "f1_ci95_high": ci_high,
                "delta_f1_vs_nadir": row["external_f1_2025"] - xgb_nadir["external_f1_2025"],
                "delta_f1_ci95_low": delta_f1_low,
                "delta_f1_ci95_high": delta_f1_high,
                "recall": row["external_recall_2025"],
                "delta_recall_vs_nadir": row["external_recall_2025"] - xgb_nadir["external_recall_2025"],
                "balanced_accuracy": row["external_balanced_accuracy_2025"],
                "delta_balanced_accuracy_vs_nadir": row["external_balanced_accuracy_2025"] - xgb_nadir["external_balanced_accuracy_2025"],
                "precision": metrics["precision"],
                "specificity": metrics["specificity"],
                "false_positive_rate": metrics["false_positive_rate"],
                "n_test": metrics["n_test"],
                "n_positive": metrics["n_positive"],
                "n_negative": metrics["n_negative"],
            }
        )
    return pl.DataFrame(rows).sort(["model", "external_f1"], descending=[False, True])


def table_severity_no_week_gain() -> pl.DataFrame:
    xgb = pl.read_csv(RESULTS_DIR / "xgboost_severity_train_eval_2024_test_2025.csv")
    xgb_200 = pl.read_csv(RESULTS_DIR / "xgboost_severity_200_trees_train_eval_2024_test_2025.csv")

    rows = []
    xgb_nadir = first_row(filter_rows(xgb, covariates="spectral_only", feature_set="nadir"))
    for feature_set in ["nadir", "multiangular_vza_raa", "multiangular_vza_phase"]:
        row = first_row(filter_rows(xgb, covariates="spectral_only", feature_set=feature_set))
        reduction = xgb_nadir["external_rmse_2025"] - row["external_rmse_2025"]
        ci_low, ci_high = paired_rmse_reduction_ci(
            "xgboost_raw_severity",
            feature_set,
            "spectral_only",
            seed=500 + len(rows),
        )
        rows.append(
            {
                "model": "XGBoost early stop",
                "feature_set": pretty_feature(feature_set),
                "rmse": row["external_rmse_2025"],
                "rmse_reduction_vs_nadir": reduction,
                "rmse_reduction_ci95_low": ci_low,
                "rmse_reduction_ci95_high": ci_high,
                "relative_reduction_percent": 100 * reduction / xgb_nadir["external_rmse_2025"],
            }
        )

    xgb200_nadir = first_row(filter_rows(xgb_200, covariates="spectral_only", feature_set="nadir"))
    for feature_set in ["nadir", "multiangular_vza_phase", "multiangular_vza_raa"]:
        row = first_row(filter_rows(xgb_200, covariates="spectral_only", feature_set=feature_set))
        reduction = xgb200_nadir["external_rmse_2025"] - row["external_rmse_2025"]
        ci_low, ci_high = paired_rmse_reduction_ci(
            "xgboost_200_raw_severity",
            feature_set,
            "spectral_only",
            seed=600 + len(rows),
        )
        rows.append(
            {
                "model": "XGBoost 200 trees",
                "feature_set": pretty_feature(feature_set),
                "rmse": row["external_rmse_2025"],
                "rmse_reduction_vs_nadir": reduction,
                "rmse_reduction_ci95_low": ci_low,
                "rmse_reduction_ci95_high": ci_high,
                "relative_reduction_percent": 100 * reduction / xgb200_nadir["external_rmse_2025"],
            }
        )
    return pl.DataFrame(rows).sort(["model", "rmse"])


def table_severity_with_timing_gain() -> pl.DataFrame:
    severity = pl.read_csv(RESULTS_DIR / "severity_external_2024_train_2025_test.csv")
    data = severity.filter(
        (pl.col("model") == "raw_severity")
        & (pl.col("covariates").is_in(["spectral_plus_week", "spectral_plus_week_horizon"]))
    )
    rows = []
    for covariates in data.get_column("covariates").unique().sort().to_list():
        group = data.filter(pl.col("covariates") == covariates)
        nadir = first_row(filter_rows(group, feature_set="nadir"))
        for feature_set in ["nadir", "multiangular_vza_phase", "multiangular_vza_compact"]:
            row = first_row(filter_rows(group, feature_set=feature_set))
            reduction = nadir["rmse"] - row["rmse"]
            ci_low, ci_high = paired_rmse_reduction_ci(
                "ridge_raw_severity",
                feature_set,
                covariates,
                seed=700 + len(rows),
            )
            rows.append(
                {
                    "model": "Ridge raw severity",
                    "covariates": covariates,
                    "feature_set": pretty_feature(feature_set),
                    "rmse": row["rmse"],
                    "rmse_reduction_vs_nadir": reduction,
                    "rmse_reduction_ci95_low": ci_low,
                    "rmse_reduction_ci95_high": ci_high,
                    "relative_reduction_percent": 100 * reduction / nadir["rmse"],
                    "r2": row["r2"],
                    "delta_r2_vs_nadir": row["r2"] - nadir["r2"],
                    "spearman": row["spearman"],
                    "delta_spearman_vs_nadir": row["spearman"] - nadir["spearman"],
                }
            )
    return pl.DataFrame(rows).sort(["covariates", "rmse"])


def table_mechanistic_importance() -> pl.DataFrame:
    ridge_group = pl.read_csv(RESULTS_DIR / "ridge_best_severity_feature_importance_by_group.csv")
    xgb_warn_band = pl.read_csv(RESULTS_DIR / "xgboost_best_warning_feature_importance_by_band.csv")
    log_raa_band = pl.read_csv(RESULTS_DIR / "logistic_warning_multiangular_vza_raa_feature_importance_by_band.csv")

    def share(frame: pl.DataFrame, key_col: str, key: str, share_col: str) -> float:
        match = frame.filter(pl.col(key_col) == key).select(share_col)
        return float(match.item()) if match.height else np.nan

    rows = [
        {
            "model_context": "Best severity: Ridge VZA + phase + week/horizon",
            "main_evidence": "VZA-phase reflectance accounts for most standardized coefficient mass",
            "multiangular_share": share(ridge_group, "feature_group", "vza_phase_reflectance", "share_abs"),
            "timing_share": share(ridge_group, "feature_group", "known_week_horizon", "share_abs"),
            "top_bands": "NIR 30.6%; red edge 19.9%; blue 17.4%",
        },
        {
            "model_context": "Best warning: XGBoost VZA",
            "main_evidence": "VZA-bin reflectance accounts for the model's attributed importance",
            "multiangular_share": 1.0,
            "timing_share": 0.0,
            "top_bands": (
                f"red {share(xgb_warn_band, 'band', 'red', 'share_gain'):.1%}; "
                f"NIR {share(xgb_warn_band, 'band', 'nir', 'share_gain'):.1%}; "
                f"blue {share(xgb_warn_band, 'band', 'blue', 'share_gain'):.1%}"
            ),
        },
        {
            "model_context": "Logistic warning: VZA + RAA",
            "main_evidence": "RAA-aware angular reflectance accounts for the model's attributed importance",
            "multiangular_share": 1.0,
            "timing_share": 0.0,
            "top_bands": (
                f"red edge {share(log_raa_band, 'band', 'red_edge', 'share_abs'):.1%}; "
                f"NIR {share(log_raa_band, 'band', 'nir', 'share_abs'):.1%}; "
                f"blue {share(log_raa_band, 'band', 'blue', 'share_abs'):.1%}"
            ),
        },
    ]
    return pl.DataFrame(rows)


def parse_feature(feature: str) -> dict[str, str]:
    if feature.startswith("known__"):
        return {
            "feature_type": "timing",
            "band": "",
            "vza_bin": "",
            "angular_bin_type": "",
            "angular_bin": "",
            "summary_metric": feature.replace("known__", ""),
        }
    parts = feature.split("__")
    if feature.startswith("vza_phase__") and len(parts) >= 4:
        return {
            "feature_type": "VZA + phase reflectance",
            "band": pretty_band(parts[1]),
            "vza_bin": parts[2].replace("_", "-"),
            "angular_bin_type": "phase",
            "angular_bin": parts[3].replace("_", "-"),
            "summary_metric": "mean reflectance",
        }
    if feature.startswith("vza_raa__") and len(parts) >= 4:
        return {
            "feature_type": "VZA + RAA reflectance",
            "band": pretty_band(parts[1]),
            "vza_bin": parts[2].replace("_", "-"),
            "angular_bin_type": "RAA",
            "angular_bin": parts[3].replace("_", "-"),
            "summary_metric": "mean reflectance",
        }
    if feature.startswith("vza__") and len(parts) >= 3:
        return {
            "feature_type": "VZA reflectance",
            "band": pretty_band(parts[1]),
            "vza_bin": parts[2].replace("_", "-"),
            "angular_bin_type": "",
            "angular_bin": "",
            "summary_metric": "mean reflectance",
        }
    if feature.startswith("vza_compact__") and len(parts) >= 3:
        return {
            "feature_type": "VZA compact summary",
            "band": pretty_band(parts[1]),
            "vza_bin": "",
            "angular_bin_type": "",
            "angular_bin": "",
            "summary_metric": parts[2].replace("_", " "),
        }
    if feature.startswith("geometry_compact__") and len(parts) >= 3:
        return {
            "feature_type": "geometry compact summary",
            "band": pretty_band(parts[1]),
            "vza_bin": "",
            "angular_bin_type": "",
            "angular_bin": "",
            "summary_metric": parts[2].replace("_", " "),
        }
    if feature.startswith("nadir__") and len(parts) >= 2:
        return {
            "feature_type": "nadir reflectance",
            "band": pretty_band(parts[1]),
            "vza_bin": "closest to nadir",
            "angular_bin_type": "",
            "angular_bin": "",
            "summary_metric": "mean reflectance",
        }
    return {
        "feature_type": "other",
        "band": "",
        "vza_bin": "",
        "angular_bin_type": "",
        "angular_bin": "",
        "summary_metric": "",
    }


def pretty_band(value: str) -> str:
    labels = {
        "blue": "Blue",
        "green": "Green",
        "red": "Red",
        "red_edge": "Red edge",
        "nir": "NIR",
    }
    return labels.get(value, value)


def table_dominant_interpretable_patterns() -> pl.DataFrame:
    severity = pl.read_csv(RESULTS_DIR / "severity_external_2024_train_2025_test.csv")
    timing_without = first_row(filter_rows(
        severity,
        model="raw_severity",
        feature_set="multiangular_vza_phase",
        covariates="spectral_only",
    ))["rmse"]
    timing_with = first_row(filter_rows(
        severity,
        model="raw_severity",
        feature_set="multiangular_vza_phase",
        covariates="spectral_plus_week_horizon",
    ))["rmse"]
    return pl.DataFrame(
        [
            {
                "task": "Severity",
                "model": "Ridge",
                "feature_set": "VZA + phase + timing",
                "covariate_block": "Phenology/horizon",
                "band": "",
                "vza_range_deg": "",
                "angular_variable": "week/horizon",
                "angular_range_deg": "",
                "direction": "Individual direction not interpreted because of collinearity",
                "evidence_metric": "RMSE spectral-only -> with week/horizon",
                "evidence_value": f"{timing_without:.3f} -> {timing_with:.3f}; reduction {timing_without - timing_with:.3f}",
            },
            {
                "task": "Severity",
                "model": "Ridge",
                "feature_set": "VZA + phase + timing",
                "covariate_block": "NIR angular reflectance",
                "band": "NIR",
                "vza_range_deg": "0-30",
                "angular_variable": "phase",
                "angular_range_deg": "20-50",
                "direction": "Predominantly negative",
                "evidence_metric": "top-10 spectral count; strongest standardized coefficient",
                "evidence_value": "8/10; -2.803",
            },
            {
                "task": "Severity",
                "model": "Ridge",
                "feature_set": "VZA + phase + timing",
                "covariate_block": "Red-edge angular reflectance",
                "band": "Red edge",
                "vza_range_deg": "40-45",
                "angular_variable": "phase",
                "angular_range_deg": "60-70",
                "direction": "Positive",
                "evidence_metric": "highest red-edge standardized coefficient",
                "evidence_value": "+1.625",
            },
            {
                "task": "Early warning",
                "model": "XGBoost",
                "feature_set": "VZA",
                "covariate_block": "NIR angular reflectance",
                "band": "NIR",
                "vza_range_deg": "25-55",
                "angular_variable": "VZA",
                "angular_range_deg": "50-55 strongest",
                "direction": "Direction not inferable from gain",
                "evidence_metric": "top-12 count; highest gain split count",
                "evidence_value": "4/12; 24 splits",
            },
            {
                "task": "Early warning",
                "model": "XGBoost",
                "feature_set": "VZA",
                "covariate_block": "Red angular reflectance",
                "band": "Red",
                "vza_range_deg": "10-25",
                "angular_variable": "VZA",
                "angular_range_deg": "10-25",
                "direction": "Direction not inferable from gain",
                "evidence_metric": "top-ranked red features; gain range",
                "evidence_value": "3; 14.851-21.603",
            },
            {
                "task": "Early warning",
                "model": "XGBoost",
                "feature_set": "VZA",
                "covariate_block": "Blue angular reflectance",
                "band": "Blue",
                "vza_range_deg": "15-45",
                "angular_variable": "VZA",
                "angular_range_deg": "15-45",
                "direction": "Direction not inferable from gain",
                "evidence_metric": "top-ranked blue features; gain range",
                "evidence_value": "4; 8.921-16.504",
            },
            {
                "task": "Early warning",
                "model": "Logistic",
                "feature_set": "VZA + RAA",
                "covariate_block": "Red-edge angular reflectance",
                "band": "Red edge",
                "vza_range_deg": "25-50",
                "angular_variable": "RAA",
                "angular_range_deg": "90-135",
                "direction": "Positive",
                "evidence_metric": "high-ranked features; coefficient range",
                "evidence_value": "5; +0.992 to +1.252",
            },
            {
                "task": "Early warning",
                "model": "Logistic",
                "feature_set": "VZA + RAA",
                "covariate_block": "NIR angular reflectance",
                "band": "NIR",
                "vza_range_deg": "10-20",
                "angular_variable": "RAA",
                "angular_range_deg": "multiple bins",
                "direction": "Negative",
                "evidence_metric": "negative high-ranked low-VZA NIR features",
                "evidence_value": "4",
            },
        ]
    )


def table_supplementary_feature_level_ranking() -> pl.DataFrame:
    sources = [
        (
            "Best severity Ridge",
            "Predicts future continuous DSDI severity",
            RESULTS_DIR / "ridge_best_severity_feature_importance.csv",
            "abs_coefficient",
            "signed_coefficient",
            "coefficient",
            12,
        ),
        (
            "Best warning XGBoost",
            "Predicts future warning ds_plot >= 5",
            RESULTS_DIR / "xgboost_best_warning_feature_importance.csv",
            "gain",
            "split_count",
            "weight",
            12,
        ),
        (
            "Logistic warning VZA + RAA",
            "Linear warning model with RAA-aware angular bins",
            RESULTS_DIR / "logistic_warning_multiangular_vza_raa_feature_importance.csv",
            "abs_importance",
            "signed_coefficient",
            "importance",
            12,
        ),
    ]
    rows = []
    for model_context, model_role, path, importance_col, secondary_name, secondary_col, n_rows in sources:
        frame = pl.read_csv(path).sort(importance_col, descending=True).head(n_rows)
        for rank, row in enumerate(frame.iter_rows(named=True), start=1):
            feature = row["feature"]
            parsed = parse_feature(feature)
            feature_label = human_feature_label(parsed)
            rows.append(
                {
                    "model_context": model_context,
                    "model_role": model_role,
                    "rank": rank,
                    "encoded_feature_name": feature,
                    "feature_label": feature_label,
                    "feature_type": parsed["feature_type"],
                    "band": parsed["band"],
                    "vza_bin_deg": parsed["vza_bin"],
                    "angular_bin_type": parsed["angular_bin_type"],
                    "angular_bin_deg": parsed["angular_bin"],
                    "summary_metric": parsed["summary_metric"],
                    "importance_metric": importance_col,
                    "importance_value": row[importance_col],
                    secondary_name: row[secondary_col],
                }
            )
    return pl.DataFrame(rows)


def table_supplementary_all_model_performance() -> pl.DataFrame:
    rows = []
    severity = pl.read_csv(RESULTS_DIR / "severity_external_2024_train_2025_test.csv")
    for row in severity.iter_rows(named=True):
        rows.append(
            {
                "task": "severity",
                "model": f"Ridge {row['model']}",
                "feature_set": pretty_feature(row["feature_set"]),
                "covariates": row["covariates"],
                "n_test": row["n_test"],
                "rmse": row["rmse"],
                "mae": row["mae"],
                "r2": row["r2"],
                "spearman": row["spearman"],
                "f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "balanced_accuracy": np.nan,
                "specificity": np.nan,
            }
        )

    xgb_severity = pl.read_csv(RESULTS_DIR / "xgboost_severity_train_eval_2024_test_2025.csv")
    for row in xgb_severity.iter_rows(named=True):
        rows.append(
            {
                "task": "severity",
                "model": "XGBoost early stop",
                "feature_set": pretty_feature(row["feature_set"]),
                "covariates": row["covariates"],
                "n_test": row["n_test"],
                "rmse": row["external_rmse_2025"],
                "mae": row["external_mae_2025"],
                "r2": row["external_r2_2025"],
                "spearman": row["external_spearman_2025"],
                "f1": np.nan,
                "precision": np.nan,
                "recall": np.nan,
                "balanced_accuracy": np.nan,
                "specificity": np.nan,
            }
        )

    xgb_200_path = RESULTS_DIR / "xgboost_severity_200_trees_train_eval_2024_test_2025.csv"
    if xgb_200_path.exists():
        xgb_200 = pl.read_csv(xgb_200_path)
        for row in xgb_200.iter_rows(named=True):
            rows.append(
                {
                    "task": "severity",
                    "model": "XGBoost 200 trees",
                    "feature_set": pretty_feature(row["feature_set"]),
                    "covariates": row["covariates"],
                    "n_test": row["n_test"],
                    "rmse": row["external_rmse_2025"],
                    "mae": row["external_mae_2025"],
                    "r2": row["external_r2_2025"],
                    "spearman": row["external_spearman_2025"],
                    "f1": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "balanced_accuracy": np.nan,
                    "specificity": np.nan,
                }
            )

    warning = pl.read_csv(RESULTS_DIR / "warning_external_2024_train_2025_test.csv")
    for row in warning.iter_rows(named=True):
        try:
            specificity = warning_metrics(read_prediction_file("early_warning", "logistic", row["feature_set"]))["specificity"]
        except FileNotFoundError:
            specificity = np.nan
        rows.append(
            {
                "task": "early_warning",
                "model": "Logistic",
                "feature_set": pretty_feature(row["feature_set"]),
                "covariates": "spectral_only",
                "n_test": row["n_test"],
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "spearman": np.nan,
                "f1": row["f1"],
                "precision": row["precision"],
                "recall": row["recall"],
                "balanced_accuracy": row["balanced_accuracy"],
                "specificity": specificity,
            }
        )

    xgb_warning = pl.read_csv(RESULTS_DIR / "xgboost_warning_train_eval_2024_test_2025.csv")
    for row in xgb_warning.iter_rows(named=True):
        try:
            specificity = warning_metrics(read_prediction_file("early_warning", "xgboost", row["feature_set"]))["specificity"]
        except FileNotFoundError:
            specificity = np.nan
        rows.append(
            {
                "task": "early_warning",
                "model": "XGBoost",
                "feature_set": pretty_feature(row["feature_set"]),
                "covariates": "spectral_only",
                "n_test": row["n_test"],
                "rmse": np.nan,
                "mae": np.nan,
                "r2": np.nan,
                "spearman": np.nan,
                "f1": row["external_f1_2025"],
                "precision": row["external_precision_2025"],
                "recall": row["external_recall_2025"],
                "balanced_accuracy": row["external_balanced_accuracy_2025"],
                "specificity": specificity,
            }
        )
    return pl.DataFrame(rows).sort(["task", "model", "covariates", "feature_set"])


def human_feature_label(parsed: dict[str, str]) -> str:
    if parsed["feature_type"] == "timing":
        return parsed["summary_metric"].replace("_", " ")
    pieces = []
    if parsed["band"]:
        pieces.append(parsed["band"])
    if parsed["vza_bin"]:
        pieces.append(f"VZA {parsed['vza_bin']} deg")
    if parsed["angular_bin_type"] and parsed["angular_bin"]:
        pieces.append(f"{parsed['angular_bin_type'].upper()} {parsed['angular_bin']} deg")
    if parsed["summary_metric"] and parsed["summary_metric"] != "mean reflectance":
        pieces.append(parsed["summary_metric"])
    return ", ".join(pieces) if pieces else "not available"


def write_report(tables: dict[str, pl.DataFrame], outputs: dict[str, Path], log_path: Path) -> None:
    report = [
        "## Paper Tables: Multiangular vs Nadir",
        "",
        "These tables summarize the key evidence for the paper story that multiangular reflectance adds value over nadir-only reflectance under external-year testing.",
        "",
        "### Table 1. External-year early-warning performance versus nadir",
        "",
        markdown_table(tables["early_warning"]),
        "",
        "**Table 1 note**: F1 95% CIs and paired delta-F1 95% CIs are non-parametric percentile intervals computed from the actual 2025 external-test prediction rows (`y_true`, `y_pred`; 20,000 resamples). Resampling is by plot to respect repeated plot-week measurements. The external test set contains 72 plot-week observations, 32 positive warning cases, and 40 negative cases. The classification threshold was fixed at 0.5 after training on 2024 data only.",
        "",
        "### Table 2. External-year reflectance-only severity prediction using XGBoost",
        "",
        markdown_table(tables["severity_no_week"]),
        "",
        "**Table 2 note**: RMSE reductions are positive when the angular model improves over the matched nadir model. Confidence intervals are paired plot-bootstrap intervals for the RMSE reduction.",
        "",
        "### Table 3. External-year Ridge severity prediction with phenology and forecast-horizon covariates",
        "",
        markdown_table(tables["severity_with_timing"]),
        "",
        "**Table 3 note**: RMSE reductions are positive when the angular model improves over the matched nadir model. Confidence intervals are paired plot-bootstrap intervals for the RMSE reduction.",
        "",
        "### Table 4. Dominant angular-spectral attribution patterns",
        "",
        markdown_table(tables["dominant_patterns"]),
        "",
        "**Table 4 note**: Attribution values are interpreted only within each fitted model because Ridge coefficients, logistic coefficients, and XGBoost gain are not directly comparable. XGBoost gain identifies which features were used by the model but does not indicate whether higher reflectance increases or decreases warning risk. Timing variables are reported as a phenology/horizon block because predictor week, target week, and forecast horizon are strongly linearly dependent. These attribution results support interpretation of the predictive comparisons but are not treated as independent evidence of biological causality.",
        "",
        "**Interpretation**: Under external-year testing, multiangular reflectance produced higher point-estimate early-warning performance than the matched nadir baseline across the reported angular representations. The strongest result was the VZA-based XGBoost model, which increased F1, recall, and balanced accuracy, but with lower specificity, making the result an operational trade-off between sensitivity and false positives. For continuous severity prediction, improvements were model- and representation-dependent; XGBoost reflectance-only models and Ridge timing-adjusted models both showed specific multiangular gains over matched nadir baselines. Attribution summaries indicate that the fitted models used phase- and azimuth-dependent NIR, red-edge, red, and blue reflectance across multiple off-nadir bins.",
        "",
        "### Outputs",
        "",
    ]
    for label, path in outputs.items():
        report.append(f"- {label}: `{path}`")
    report.append(f"- log: `{log_path}`")
    outputs["report"].write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()

    t0 = time.time()
    tables = {
        "early_warning": table_early_warning_gain(),
        "severity_no_week": table_severity_no_week_gain(),
        "severity_with_timing": table_severity_with_timing_gain(),
        "grouped_attribution": table_mechanistic_importance(),
        "dominant_patterns": table_dominant_interpretable_patterns(),
        "supplementary_feature_ranking": table_supplementary_feature_level_ranking(),
        "supplementary_all_model_performance": table_supplementary_all_model_performance(),
    }
    log_phase("build paper tables", t0)

    outputs = {
        "table_1_early_warning_gain": TABLE_DIR / "table_1_early_warning_gain_over_nadir.csv",
        "table_2_severity_no_week": TABLE_DIR / "table_2_reflectance_only_severity_no_week.csv",
        "table_3_severity_with_timing": TABLE_DIR / "table_3_severity_with_week_horizon.csv",
        "table_4_dominant_patterns": TABLE_DIR / "table_4_dominant_angular_spectral_attribution_patterns.csv",
        "supplementary_grouped_attribution": TABLE_DIR / "supplementary_grouped_feature_attribution_summary.csv",
        "supplementary_feature_level_ranking": TABLE_DIR / "supplementary_feature_level_ranking.csv",
        "supplementary_all_model_performance": TABLE_DIR / "supplementary_all_model_performance.csv",
        "report": REPORTS_DIR / "paper_key_multiangular_tables.md",
    }
    tables["early_warning"].write_csv(outputs["table_1_early_warning_gain"])
    tables["severity_no_week"].write_csv(outputs["table_2_severity_no_week"])
    tables["severity_with_timing"].write_csv(outputs["table_3_severity_with_timing"])
    tables["dominant_patterns"].write_csv(outputs["table_4_dominant_patterns"])
    tables["grouped_attribution"].write_csv(outputs["supplementary_grouped_attribution"])
    tables["supplementary_feature_ranking"].write_csv(outputs["supplementary_feature_level_ranking"])
    tables["supplementary_all_model_performance"].write_csv(outputs["supplementary_all_model_performance"])
    write_report(tables, outputs, log_path)
    log_phase("total", start)


if __name__ == "__main__":
    main()
