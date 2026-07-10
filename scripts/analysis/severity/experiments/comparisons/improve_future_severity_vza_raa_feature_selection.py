#!/usr/bin/env python3
"""Try more stable VZA+RAA feature families for future severity prediction.

The raw VZA+RAA feature set overfits because stability selection can pick a few
narrow RAA bins that are correlated in 2024 but unstable in 2025. This script
tests lower-dimensional, clustered RAA summaries combined with the strong
compact VZA distribution feature family.
"""

from __future__ import annotations

from src.research.common import write_report as persist_report

import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[5]
MPLCONFIG_DIR = ROOT / "outputs/.matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.early_warning.analyze_early_warning_severity_2024 import (  # noqa: E402
    TARGET,
    build_model_table,
)
from scripts.analysis.severity import (  # noqa: E402
    debug_multiangular_rmse_bottleneck as residual_pipeline,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (  # noqa: E402
    build_feature_sets,
    clean_token,
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/future/vza_raa_feature_selection_improvement"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

DIST_RESULTS_DIR = ROOT / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/results"
RAA_2024 = (
    ROOT
    / "outputs/runs/analysis/reflectance/raa_sun_geometry/2024/ground_filtered/results/plot_week_vza_raa_features_2024.parquet"
)
RAA_2025 = (
    ROOT
    / "outputs/runs/analysis/reflectance/raa_sun_geometry/2025/ground_filtered/results/plot_week_vza_raa_features_2025.parquet"
)
DISEASE_2024 = ROOT / "outputs/shared/disease/clean_disease_scores_2024.csv"
DISEASE_2025 = ROOT / "outputs/shared/disease/clean_disease_scores_2025.csv"

META_COLS = ["year", "week", "plot_id", "cult", "trt"]
COMPACT_BANDS = {"red", "red_edge", "nir", "osavi"}
COVARIATES = "spectral_plus_week_horizon"
VZA_BASELINE_RMSE = 8.089186
SEED = residual_pipeline.SEED


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"improve_future_severity_vza_raa_feature_selection_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    logging.info("Log file: %s", log_path)
    return log_path


def log_phase(name: str, started: float) -> None:
    logging.info("[PHASE] %s: %.1fs", name, time.perf_counter() - started)


def configure_residual_pipeline_paths() -> None:
    residual_pipeline.ROOT = ROOT
    residual_pipeline.COVARIATES = COVARIATES
    residual_pipeline.OUTPUT_ROOT = OUTPUT_ROOT
    residual_pipeline.RESULTS_DIR = RESULTS_DIR
    residual_pipeline.REPORTS_DIR = REPORTS_DIR
    residual_pipeline.FIGURES_DIR = FIGURES_DIR
    residual_pipeline.PREDICTIONS_DIR = PREDICTIONS_DIR


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return math.nan
    value = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return float(value) if value is not None else math.nan


def metric_row(
    predictions: pd.DataFrame,
    model: str,
    feature_set: str,
    n_train: int,
    n_features: int,
    fit_time_s: float,
) -> dict[str, object]:
    y = predictions["y_true"].to_numpy(float)
    pred = predictions["y_pred"].to_numpy(float)
    return {
        "model": model,
        "feature_set": feature_set,
        "covariates": COVARIATES,
        "n_train": n_train,
        "n_test": len(predictions),
        "n_features": n_features,
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred) if len(np.unique(y)) > 1 else math.nan,
        "spearman": safe_spearman(y, pred),
        "fit_time_s": fit_time_s,
    }


def write_predictions(predictions: pd.DataFrame, model: str, feature_set: str) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"severity_predictions_{model}_{feature_set}_{COVARIATES}.csv".replace(
        "/", "_"
    )
    predictions.to_csv(path, index=False)


def read_inputs() -> (
    tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame, pd.DataFrame]
):
    started = time.perf_counter()
    long_2024 = pd.read_csv(DIST_RESULTS_DIR / "distribution_features_long_2024.csv")
    long_2025 = pd.read_csv(DIST_RESULTS_DIR / "distribution_features_long_2025.csv")
    feature_sets = build_feature_sets(long_2024, long_2025)
    raa_2024 = pl.read_parquet(RAA_2024).to_pandas()
    raa_2025 = pl.read_parquet(RAA_2025).to_pandas()
    log_phase("read compact VZA features and RAA parquets", started)
    return feature_sets, raa_2024, raa_2025


def merge_feature_frames(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return left.merge(right, on=META_COLS, how="left")


def raa_feature_name(prefix: str, band: object, *parts: object) -> str:
    tokens = [prefix, clean_token(str(band))]
    tokens.extend(clean_token(str(part)) for part in parts)
    return "__".join(tokens)


def pivot_rows(rows: list[dict[str, object]], prefix: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=META_COLS)
    data = pd.DataFrame(rows)
    pivot = data.pivot_table(
        index=META_COLS,
        columns="feature",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    logging.info(
        "%s compact features: rows=%d cols=%d",
        prefix,
        len(pivot),
        len(pivot.columns) - len(META_COLS),
    )
    return pivot


def build_raa_global_features(raa: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in raa.groupby(META_COLS + ["band_name"], sort=False):
        group = group.dropna(subset=["reflectance", "raa_midpoint"]).copy()
        if group.empty:
            continue
        values = group["reflectance"].to_numpy(float)
        low = group[group["raa_midpoint"] <= 90]["reflectance"]
        high = group[group["raa_midpoint"] > 90]["reflectance"]
        front = group[group["raa_midpoint"] <= 45]["reflectance"]
        back = group[group["raa_midpoint"] >= 135]["reflectance"]
        metrics = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "range": float(np.nanmax(values) - np.nanmin(values)),
            "low_minus_high": (
                float(low.mean() - high.mean()) if not low.empty and not high.empty else np.nan
            ),
            "front_minus_back": (
                float(front.mean() - back.mean()) if not front.empty and not back.empty else np.nan
            ),
        }
        base = dict(zip(META_COLS, keys[: len(META_COLS)], strict=True))
        band = keys[-1]
        for metric, value in metrics.items():
            rows.append(
                {**base, "feature": raa_feature_name("raa_global", band, metric), "value": value}
            )
    return pivot_rows(rows, "raa_global")


def build_raa_cluster_features(raa: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in raa.groupby(META_COLS + ["band_name", "vza_class"], sort=False):
        group = group.dropna(subset=["reflectance", "raa_midpoint"]).copy()
        if group.empty:
            continue
        values = group["reflectance"].to_numpy(float)
        low = group[group["raa_midpoint"] <= 90]["reflectance"]
        high = group[group["raa_midpoint"] > 90]["reflectance"]
        metrics = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "range": float(np.nanmax(values) - np.nanmin(values)),
            "low_minus_high": (
                float(low.mean() - high.mean()) if not low.empty and not high.empty else np.nan
            ),
        }
        base = dict(zip(META_COLS, keys[: len(META_COLS)], strict=True))
        band = keys[len(META_COLS)]
        vza_class = keys[len(META_COLS) + 1]
        for metric, value in metrics.items():
            rows.append(
                {
                    **base,
                    "feature": raa_feature_name("raa_cluster", band, vza_class, metric),
                    "value": value,
                }
            )
    return pivot_rows(rows, "raa_cluster")


def build_raa_anisotropy_curve_features(raa: pd.DataFrame) -> pd.DataFrame:
    cluster = build_raa_cluster_features(raa)
    rows: list[dict[str, object]] = []
    feature_cols = [c for c in cluster.columns if c.startswith("raa_cluster__")]
    grouped: dict[tuple[str, str], list[tuple[float, str]]] = {}
    for col in feature_cols:
        parts = col.split("__")
        if len(parts) != 4:
            continue
        _, band, vza_class, metric = parts
        if metric not in {"low_minus_high", "range", "std"}:
            continue
        try:
            low, high = vza_class.split("_")
            midpoint = (float(low) + float(high)) / 2
        except ValueError:
            continue
        grouped.setdefault((band, metric), []).append((midpoint, col))
    for row in cluster[META_COLS].itertuples(index=False):
        mask = (cluster[META_COLS] == pd.Series(row._asdict())).all(axis=1)
        source = cluster.loc[mask].iloc[0]
        base = row._asdict()
        for (band, metric), entries in grouped.items():
            entries = sorted(entries)
            cols = [col for _, col in entries]
            mids = np.array([mid for mid, _ in entries], dtype=float)
            vals = source[cols].to_numpy(float)
            valid = np.isfinite(vals)
            if valid.sum() == 0:
                continue
            values = {
                "mean": float(np.nanmean(vals)),
                "max_abs": float(np.nanmax(np.abs(vals))),
                "sd_over_vza": float(np.nanstd(vals)),
            }
            if valid.sum() >= 2:
                x = mids[valid] - mids[valid].mean()
                y = vals[valid] - vals[valid].mean()
                denom = float(np.sum(x**2))
                if denom > 0:
                    values["slope_over_vza"] = float(np.sum(x * y) / denom)
            for summary, value in values.items():
                rows.append(
                    {
                        **base,
                        "feature": raa_feature_name("raa_curve", band, metric, summary),
                        "value": value,
                    }
                )
    return pivot_rows(rows, "raa_curve")


def filter_red_edge_nir(frame: pd.DataFrame) -> pd.DataFrame:
    keep = META_COLS.copy()
    for col in frame.columns:
        if col in META_COLS:
            continue
        parts = col.split("__")
        if len(parts) >= 2 and parts[1] in COMPACT_BANDS:
            keep.append(col)
    return frame[keep].copy()


def build_candidate_feature_sets(
    vza_features: tuple[pd.DataFrame, pd.DataFrame],
    raa_2024: pd.DataFrame,
    raa_2025: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    started = time.perf_counter()
    vza_train, vza_test = vza_features
    raa_global_train = filter_red_edge_nir(build_raa_global_features(raa_2024))
    raa_global_test = filter_red_edge_nir(build_raa_global_features(raa_2025))
    raa_cluster_train = filter_red_edge_nir(build_raa_cluster_features(raa_2024))
    raa_cluster_test = filter_red_edge_nir(build_raa_cluster_features(raa_2025))
    raa_curve_train = filter_red_edge_nir(build_raa_anisotropy_curve_features(raa_2024))
    raa_curve_test = filter_red_edge_nir(build_raa_anisotropy_curve_features(raa_2025))
    candidates = {
        "compact_vza_only_recomputed": (vza_train, vza_test),
        "compact_vza_plus_raa_global": (
            merge_feature_frames(vza_train, raa_global_train),
            merge_feature_frames(vza_test, raa_global_test),
        ),
        "compact_vza_plus_raa_curve": (
            merge_feature_frames(vza_train, raa_curve_train),
            merge_feature_frames(vza_test, raa_curve_test),
        ),
        "compact_vza_plus_raa_global_curve": (
            merge_feature_frames(
                merge_feature_frames(vza_train, raa_global_train), raa_curve_train
            ),
            merge_feature_frames(merge_feature_frames(vza_test, raa_global_test), raa_curve_test),
        ),
        "compact_vza_plus_raa_cluster": (
            merge_feature_frames(vza_train, raa_cluster_train),
            merge_feature_frames(vza_test, raa_cluster_test),
        ),
    }
    for name, (train, test) in candidates.items():
        n_train_features = len([c for c in train.columns if c not in META_COLS])
        n_test_features = len([c for c in test.columns if c not in META_COLS])
        logging.info(
            "%s: train_features=%d test_features=%d", name, n_train_features, n_test_features
        )
    log_phase("build candidate VZA+RAA compact feature sets", started)
    return candidates


def ridge_model(
    train: pd.DataFrame, test: pd.DataFrame, feature_set: str
) -> tuple[dict, pd.DataFrame]:
    started = time.perf_counter()
    cols, train_aligned, test_aligned = residual_pipeline.prepare_aligned(train, test)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=residual_pipeline.ALPHAS)),
        ]
    )
    y_train = train_aligned[TARGET].to_numpy(float)
    pipeline.fit(train_aligned[cols], y_train)
    pred = residual_pipeline.clip_predictions(pipeline.predict(test_aligned[cols]), y_train)
    predictions = residual_pipeline.prediction_frame(
        test_aligned, pred, "ridge_all_features", feature_set
    )
    result = metric_row(
        predictions,
        "ridge_all_features",
        feature_set,
        len(train_aligned),
        len(cols),
        time.perf_counter() - started,
    )
    return result, predictions


def huber_model(
    train: pd.DataFrame, test: pd.DataFrame, feature_set: str
) -> tuple[dict, pd.DataFrame]:
    started = time.perf_counter()
    cols, train_aligned, test_aligned = residual_pipeline.prepare_aligned(train, test)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("huber", HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=1000)),
        ]
    )
    y_train = train_aligned[TARGET].to_numpy(float)
    pipeline.fit(train_aligned[cols], y_train)
    pred = residual_pipeline.clip_predictions(pipeline.predict(test_aligned[cols]), y_train)
    predictions = residual_pipeline.prediction_frame(
        test_aligned, pred, "huber_all_features", feature_set
    )
    result = metric_row(
        predictions,
        "huber_all_features",
        feature_set,
        len(train_aligned),
        len(cols),
        time.perf_counter() - started,
    )
    return result, predictions


def evaluate_candidates(
    candidates: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    selected_rows: list[pd.DataFrame] = []
    for feature_set, (train_features, test_features) in candidates.items():
        table_started = time.perf_counter()
        train = build_model_table(train_features, disease_2024)
        test = build_model_table(test_features, disease_2025)
        logging.info("%s model table: train=%d test=%d", feature_set, len(train), len(test))
        log_phase(f"build model table {feature_set}", table_started)
        for fit_name, func in [
            ("ridge_all_features", ridge_model),
            ("huber_all_features", huber_model),
        ]:
            try:
                result, pred = func(train, test, feature_set)
                rows.append(result)
                write_predictions(pred, fit_name, feature_set)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Failed %s / %s", fit_name, feature_set)
                rows.append({"model": fit_name, "feature_set": feature_set, "error": str(exc)})
        for fit_name, func in [
            (
                "residual_reliability_filtered_xgboost",
                residual_pipeline.fit_residual_reliability_filtered_xgboost,
            ),
            ("residual_xgboost_stability", residual_pipeline.fit_residual_xgboost_stability),
        ]:
            try:
                result, pred, tuning = func(train, test, feature_set)
                rows.append(result)
                write_predictions(pred, fit_name, feature_set)
                tuning["model"] = fit_name
                tuning["feature_set"] = feature_set
                selected_rows.append(tuning)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Failed %s / %s", fit_name, feature_set)
                rows.append({"model": fit_name, "feature_set": feature_set, "error": str(exc)})
    results = pd.DataFrame(rows).sort_values("rmse", na_position="last")
    tuning = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    return results, tuning


def build_report(results: pd.DataFrame, tuning: pd.DataFrame, log_path: Path) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best = results.dropna(subset=["rmse"]).sort_values("rmse").head(15)
    best_row = best.iloc[0] if not best.empty else pd.Series(dtype=object)
    beat_text = "No candidate beat the current VZA-only slide baseline."
    if not best.empty and float(best_row["rmse"]) < VZA_BASELINE_RMSE:
        beat_text = (
            f"Best candidate `{best_row['model']}` / `{best_row['feature_set']}` "
            f"beat the VZA-only baseline by {VZA_BASELINE_RMSE - float(best_row['rmse']):.3f} RMSE."
        )
    report = [
        "## Results: Future Severity VZA+RAA Feature-Selection Improvement",
        "",
        "This experiment tests clustered/compact VZA+RAA features inspired by cluster stability selection: instead of letting the model choose isolated narrow RAA bins, RAA information is summarized as lower-dimensional band and curve descriptors before fitting the same future-severity residual model family.",
        "",
        markdown_table(best.round(4), max_rows=20),
        "",
        f"**Interpretation**: {beat_text}",
        "",
        "### Reproducibility",
        "",
        "- Train year: `2024`",
        "- Test year: `2025`",
        "- Target: next observed disease-severity week after each predictor week",
        f"- VZA-only reference RMSE used for comparison: `{VZA_BASELINE_RMSE:.3f}`",
        "- Candidate features: compact VZA distribution features plus global, curve, or cluster summaries of RAA reflectance.",
        "- Models: RidgeCV, Huber, residual reliability-filtered XGBoost, and residual XGBoost stability.",
        f"- Log: `{log_path.relative_to(ROOT)}`",
        "",
        "### Outputs",
        "",
        f"- Results: `{(RESULTS_DIR / 'vza_raa_feature_selection_improvement_results.csv').relative_to(ROOT)}`",
        f"- Tuning audit: `{(RESULTS_DIR / 'vza_raa_feature_selection_improvement_tuning.csv').relative_to(ROOT)}`",
    ]
    path = REPORTS_DIR / "vza_raa_feature_selection_improvement_summary.md"
    persist_report(path, report)
    return path


def main() -> None:
    total = time.perf_counter()
    log_path = setup_logging()
    configure_residual_pipeline_paths()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_sets, raa_2024, raa_2025 = read_inputs()
    candidates = build_candidate_feature_sets(
        feature_sets["compact_anomaly_multiangular"],
        raa_2024,
        raa_2025,
    )
    disease_2024 = pd.read_csv(DISEASE_2024)
    disease_2025 = pd.read_csv(DISEASE_2025)
    results, tuning = evaluate_candidates(candidates, disease_2024, disease_2025)
    result_path = RESULTS_DIR / "vza_raa_feature_selection_improvement_results.csv"
    tuning_path = RESULTS_DIR / "vza_raa_feature_selection_improvement_tuning.csv"
    results.to_csv(result_path, index=False)
    tuning.to_csv(tuning_path, index=False)
    build_report(results, tuning, log_path)
    logging.info("Wrote results: %s", result_path)
    logging.info("Wrote tuning: %s", tuning_path)
    log_phase("total", total)


if __name__ == "__main__":
    main()
