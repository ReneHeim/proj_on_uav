#!/usr/bin/env python3
"""External-year test of multiangular image-distribution features.

This is a separate experiment family from the mean-reflectance tables. It uses
per-plot pixel parquets to summarize distributional image information by
plot/week/angle/band, then tests whether those distribution features improve
2025 severity prediction over a nadir-only distribution baseline.
"""

from __future__ import annotations

import logging
import math
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity.analyze_cross_year_generalization_2024_to_2025 import (
    ALPHAS,
    MIN_NON_NULL_FRACTION,
    SEED,
    TARGET,
    TARGET_LOG,
    WARNING_TARGET,
    add_known_covariates,
    add_lagged_disease_features,
    add_lagged_known_covariates,
    build_model_table,
    feature_columns,
    load_2024_disease_with_fallback,
    load_2025_disease_with_fallback,
    safe_filename,
)
from src.analysis.result_01_reflectance_distributions import (
    BANDS,
    FINE_VZA_MAX,
    FINE_VZA_MIN,
    POLYGON_PATHS,
    YEAR_PLOT_DIRS,
    assign_fine_vza_bins,
    ensure_indices,
    ground_filter_expressions,
    load_polygon_meta,
    sample_equal_vza_bins,
)

OUTPUT_ROOT = ROOT / "outputs/multiangular_distribution_feature_family"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/logs"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

GROUND_FILTER = True
OSAVI_THRESHOLD = 0.2
EXCESS_GREEN_THRESHOLD: float | None = None
MAX_PLOT_SAMPLE = 2_000_000
MIN_BIN_PIXELS = 500
LOW_OSAVI_THRESHOLDS = [0.25, 0.35]
LOW_NIR_QUANTILE = 0.15
LOW_RED_EDGE_QUANTILE = 0.15
COMPACT_BANDS = {"nir", "red_edge", "red", "osavi"}
COMPACT_METRICS = {
    "p10",
    "p25",
    "iqr",
    "cv",
    "osavi_p10",
    "osavi_iqr",
    "osavi_frac_lt_025",
    "osavi_frac_lt_035",
}
COMPACT_SHAPE_METRICS = {
    "range",
    "std",
    "offnadir_minus_nadir",
    "relative_offnadir_minus_nadir",
    "slope",
}
PCA_EXPLAINED_VARIANCE = 0.95
SELECT_K_FEATURES = 30
STABILITY_REPEATS = 10
STABILITY_TEST_SIZE = 0.25
STABILITY_MIN_FREQUENCY = 0.40
STABILITY_FALLBACK_TOP_K = 25
STABILITY_L1_RATIOS = [0.5, 0.9]
STABILITY_ALPHAS = np.logspace(-2, 2, 12)
PAIRED_BOOTSTRAP_ITERATIONS = 500
PAIRED_BOOTSTRAP_ALPHA = 0.05


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_multiangular_distribution_feature_family_{timestamp}.log"
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


def clean_token(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return math.nan
    value = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return float(value) if value is not None else math.nan


def align_train_test(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    common_cols = sorted(set(feature_columns(train)).intersection(feature_columns(test)))
    train = train.copy()
    test = test.copy()
    train[common_cols] = train[common_cols].replace([np.inf, -np.inf], np.nan)
    test[common_cols] = test[common_cols].replace([np.inf, -np.inf], np.nan)
    common_cols = [col for col in common_cols if train[col].notna().mean() >= MIN_NON_NULL_FRACTION]
    common_cols = [col for col in common_cols if test[col].notna().mean() > 0]
    keep = [
        "plot_id",
        "predictor_week",
        "target_week",
        TARGET,
        TARGET_LOG,
        WARNING_TARGET,
    ] + common_cols
    return common_cols, train[keep].copy(), test[keep].copy()


def split_train_eval_by_plot(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, eval_idx = next(
        splitter.split(table, table[TARGET], groups=table["plot_id"].to_numpy())
    )
    return train_idx, eval_idx


def markdown_table(df: pd.DataFrame, float_digits: int = 3, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.head(max_rows)
    columns = list(shown.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in shown.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float) or isinstance(value, np.floating):
                values.append("" if pd.isna(value) else f"{value:.{float_digits}f}")
            else:
                values.append("" if pd.isna(value) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def plot_week_dirs_available(year: int) -> dict[int, Path]:
    available = {}
    for week, directory in YEAR_PLOT_DIRS[year].items():
        if directory.exists():
            available[week] = directory
        else:
            logging.warning("Skipping missing %s week %s directory: %s", year, week, directory)
    return available


def summarize_plot_file(
    path: Path, year: int, week: int, meta: dict[str, dict[str, object]]
) -> tuple[list[dict], dict]:
    plot_id = path.stem
    read_t0 = time.perf_counter()
    frame = pl.read_parquet(path)
    read_seconds = time.perf_counter() - read_t0
    original_rows = frame.height
    band_columns = list(BANDS)
    quality = (
        pl.col("vza").is_not_nan()
        & (pl.col("vza") >= FINE_VZA_MIN)
        & (pl.col("vza") < FINE_VZA_MAX)
    )
    for band in band_columns:
        quality = quality & pl.col(band).is_not_nan() & (pl.col(band) > 0)
    frame = frame.filter(quality)
    rows_after_quality = frame.height
    frame = ensure_indices(frame)
    if GROUND_FILTER:
        frame = frame.filter(ground_filter_expressions(OSAVI_THRESHOLD, EXCESS_GREEN_THRESHOLD))
    rows_after_ground = frame.height
    frame = assign_fine_vza_bins(frame)
    frame = sample_equal_vza_bins(frame, max_rows=MAX_PLOT_SAMPLE)
    rows_after_sample = frame.height
    if frame.is_empty():
        return [], {
            "year": year,
            "week": week,
            "plot_id": plot_id,
            "read_seconds": read_seconds,
            "original_rows": original_rows,
            "rows_after_quality": rows_after_quality,
            "rows_after_ground_filter": rows_after_ground,
            "rows_after_sampling": rows_after_sample,
        }

    long = frame.unpivot(
        index=["vza_class", "vza_midpoint"],
        on=band_columns,
        variable_name="band",
        value_name="reflectance",
    ).with_columns(pl.col("reflectance").cast(pl.Float64))

    summary = (
        long.group_by("vza_class", "vza_midpoint", "band")
        .agg(
            pl.len().alias("n_pixels"),
            pl.col("reflectance").mean().alias("mean"),
            pl.col("reflectance").median().alias("median"),
            pl.col("reflectance").quantile(0.05).alias("p05"),
            pl.col("reflectance").quantile(0.10).alias("p10"),
            pl.col("reflectance").quantile(0.25).alias("p25"),
            pl.col("reflectance").quantile(0.75).alias("p75"),
            pl.col("reflectance").quantile(0.90).alias("p90"),
            pl.col("reflectance").quantile(0.95).alias("p95"),
            pl.col("reflectance").std().alias("sd"),
        )
        .with_columns(
            (pl.col("p75") - pl.col("p25")).alias("iqr"),
            (pl.col("sd") / pl.col("mean")).alias("cv"),
        )
        .filter(pl.col("n_pixels") >= MIN_BIN_PIXELS)
    )

    osavi = (
        frame.group_by("vza_class", "vza_midpoint")
        .agg(
            pl.len().alias("osavi_n_pixels"),
            pl.col("OSAVI").mean().alias("osavi_mean"),
            pl.col("OSAVI").quantile(0.10).alias("osavi_p10"),
            pl.col("OSAVI").quantile(0.25).alias("osavi_p25"),
            pl.col("OSAVI").quantile(0.75).alias("osavi_p75"),
            (pl.col("OSAVI") < LOW_OSAVI_THRESHOLDS[0]).mean().alias("osavi_frac_lt_025"),
            (pl.col("OSAVI") < LOW_OSAVI_THRESHOLDS[1]).mean().alias("osavi_frac_lt_035"),
        )
        .with_columns((pl.col("osavi_p75") - pl.col("osavi_p25")).alias("osavi_iqr"))
        .filter(pl.col("osavi_n_pixels") >= MIN_BIN_PIXELS)
    )

    rows = []
    for record in summary.to_dicts():
        band = str(record["band"])
        band_name = BANDS[band]
        for metric in ["mean", "median", "p05", "p10", "p25", "p75", "p90", "p95", "iqr", "cv"]:
            value = record[metric]
            if value is not None and np.isfinite(value):
                rows.append(
                    {
                        "year": year,
                        "week": week,
                        "plot_id": plot_id,
                        "cult": meta[plot_id]["cult"],
                        "trt": meta[plot_id]["trt"],
                        "band": band,
                        "band_name": band_name,
                        "vza_class": record["vza_class"],
                        "vza_midpoint": record["vza_midpoint"],
                        "metric": metric,
                        "value": float(value),
                    }
                )
    for record in osavi.to_dicts():
        for metric in [
            "osavi_mean",
            "osavi_p10",
            "osavi_iqr",
            "osavi_frac_lt_025",
            "osavi_frac_lt_035",
        ]:
            value = record[metric]
            if value is not None and np.isfinite(value):
                rows.append(
                    {
                        "year": year,
                        "week": week,
                        "plot_id": plot_id,
                        "cult": meta[plot_id]["cult"],
                        "trt": meta[plot_id]["trt"],
                        "band": "index",
                        "band_name": "OSAVI",
                        "vza_class": record["vza_class"],
                        "vza_midpoint": record["vza_midpoint"],
                        "metric": metric,
                        "value": float(value),
                    }
                )

    return rows, {
        "year": year,
        "week": week,
        "plot_id": plot_id,
        "read_seconds": read_seconds,
        "original_rows": original_rows,
        "rows_after_quality": rows_after_quality,
        "rows_after_ground_filter": rows_after_ground,
        "rows_after_sampling": rows_after_sample,
        "feature_rows": len(rows),
    }


def build_distribution_long(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    meta = load_polygon_meta(year)
    all_rows: list[dict] = []
    audit_rows: list[dict] = []
    for week, directory in plot_week_dirs_available(year).items():
        files = sorted(directory.glob("plot_*.parquet"))
        logging.info(
            "%s week %s: reading %d plot parquets from %s", year, week, len(files), directory
        )
        for path in files:
            if path.stem not in meta:
                continue
            rows, audit = summarize_plot_file(path, year, week, meta)
            all_rows.extend(rows)
            audit_rows.append(audit)
    if not all_rows:
        raise RuntimeError(f"No distribution features created for {year}.")
    audit = pd.DataFrame(audit_rows)
    read_times = audit["read_seconds"].dropna().to_numpy(float)
    if read_times.size:
        logging.info(
            "[PHASE] parquet read summary %s: min=%.4fs median=%.4fs mean=%.4fs max=%.4fs",
            year,
            float(np.min(read_times)),
            float(np.median(read_times)),
            float(np.mean(read_times)),
            float(np.max(read_times)),
        )
    frame = pd.DataFrame(all_rows)
    log_phase(f"build distribution long {year}", started)
    return frame, audit


def pivot_distribution_features(
    long: pd.DataFrame, feature_set: str, include_all_angles: bool
) -> pd.DataFrame:
    data = long.copy()
    if not include_all_angles:
        closest = (
            data[["year", "week", "plot_id", "band_name", "metric", "vza_midpoint", "vza_class"]]
            .sort_values("vza_midpoint")
            .groupby(["year", "week", "plot_id", "band_name", "metric"], as_index=False)
            .head(1)[["year", "week", "plot_id", "band_name", "metric", "vza_class"]]
        )
        data = data.merge(
            closest, on=["year", "week", "plot_id", "band_name", "metric", "vza_class"], how="inner"
        )
        data["feature"] = (
            feature_set
            + "__"
            + data["band_name"].map(clean_token)
            + "__"
            + data["metric"].map(clean_token)
        )
    else:
        data["feature"] = (
            feature_set
            + "__"
            + data["band_name"].map(clean_token)
            + "__"
            + data["vza_class"].map(clean_token)
            + "__"
            + data["metric"].map(clean_token)
        )
    pivot = data.pivot_table(
        index=["year", "week", "plot_id", "cult", "trt"],
        columns="feature",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def add_angular_distribution_shape_features(multiangular: pd.DataFrame) -> pd.DataFrame:
    started = time.perf_counter()
    out = multiangular.copy()
    feature_cols = [col for col in out.columns if col.startswith("dist_multiangular__")]
    groups: dict[tuple[str, str], list[tuple[float, str]]] = {}
    for col in feature_cols:
        match = re.match(r"dist_multiangular__(.+?)__(\d+)_(\d+)__(.+)$", col)
        if not match:
            continue
        band, low, high, metric = match.groups()
        midpoint = (int(low) + int(high)) / 2
        groups.setdefault((band, metric), []).append((midpoint, col))
    shape_columns: dict[str, np.ndarray] = {}
    for (band, metric), entries in groups.items():
        entries = sorted(entries)
        cols = [col for _, col in entries]
        mids = np.array([mid for mid, _ in entries], dtype=float)
        values = out[cols].to_numpy(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            shape_columns[f"dist_shape__{band}__{metric}__range"] = np.nanmax(
                values, axis=1
            ) - np.nanmin(values, axis=1)
            shape_columns[f"dist_shape__{band}__{metric}__std"] = np.nanstd(values, axis=1)
        if len(cols) >= 2:
            first = values[:, 0]
            last = values[:, -1]
            shape_columns[f"dist_shape__{band}__{metric}__offnadir_minus_nadir"] = last - first
            shape_columns[f"dist_shape__{band}__{metric}__relative_offnadir_minus_nadir"] = (
                np.divide(
                    last - first,
                    first,
                    out=np.full_like(first, np.nan, dtype=float),
                    where=np.isfinite(first) & (first != 0),
                )
            )
        if len(cols) >= 3:
            centered_x = mids - mids.mean()
            denom = float(np.sum(centered_x**2))
            if denom > 0:
                y_centered = values - np.nanmean(values, axis=1, keepdims=True)
                shape_columns[f"dist_shape__{band}__{metric}__slope"] = (
                    np.nansum(y_centered * centered_x, axis=1) / denom
                )
    if shape_columns:
        out = pd.concat([out, pd.DataFrame(shape_columns, index=out.index)], axis=1)
    log_phase("add angular distribution shape features", started)
    return out


def is_compact_distribution_feature(column: str) -> bool:
    if not column.startswith(("dist_nadir__", "dist_multiangular__")):
        return False
    parts = column.split("__")
    if len(parts) == 3:
        _, band, metric = parts
    elif len(parts) == 4:
        _, band, _angle, metric = parts
    else:
        return False
    return band in COMPACT_BANDS and metric in COMPACT_METRICS


def is_compact_shape_feature(column: str) -> bool:
    if not column.startswith("dist_shape__"):
        return False
    parts = column.split("__")
    if len(parts) != 4:
        return False
    _, band, metric, shape_metric = parts
    return (
        band in COMPACT_BANDS
        and metric in COMPACT_METRICS
        and shape_metric in COMPACT_SHAPE_METRICS
    )


def compact_feature_frame(frame: pd.DataFrame, include_shape: bool) -> pd.DataFrame:
    meta = [col for col in ["year", "week", "plot_id", "cult", "trt"] if col in frame.columns]
    keep = []
    for col in frame.columns:
        if col in meta:
            continue
        if is_compact_distribution_feature(col) or (
            include_shape and is_compact_shape_feature(col)
        ):
            keep.append(col)
    return frame[meta + keep].copy()


def build_feature_sets(
    long_2024: pd.DataFrame, long_2025: pd.DataFrame
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    started = time.perf_counter()
    train_nadir = pivot_distribution_features(long_2024, "dist_nadir", include_all_angles=False)
    test_nadir = pivot_distribution_features(long_2025, "dist_nadir", include_all_angles=False)
    train_multi = pivot_distribution_features(
        long_2024, "dist_multiangular", include_all_angles=True
    )
    test_multi = pivot_distribution_features(
        long_2025, "dist_multiangular", include_all_angles=True
    )
    train_shape = add_angular_distribution_shape_features(train_multi)
    test_shape = add_angular_distribution_shape_features(test_multi)
    train_nadir_compact = compact_feature_frame(train_nadir, include_shape=False)
    test_nadir_compact = compact_feature_frame(test_nadir, include_shape=False)
    train_multi_compact = compact_feature_frame(train_multi, include_shape=False)
    test_multi_compact = compact_feature_frame(test_multi, include_shape=False)
    train_shape_compact = compact_feature_frame(train_shape, include_shape=True)
    test_shape_compact = compact_feature_frame(test_shape, include_shape=True)
    feature_sets = {
        "distribution_nadir": (train_nadir, test_nadir),
        "distribution_multiangular": (train_multi, test_multi),
        "distribution_multiangular_shape": (train_shape, test_shape),
        "compact_anomaly_nadir": (train_nadir_compact, test_nadir_compact),
        "compact_anomaly_multiangular": (train_multi_compact, test_multi_compact),
        "compact_anomaly_multiangular_shape": (train_shape_compact, test_shape_compact),
    }
    for name, (train, test) in feature_sets.items():
        logging.info(
            "%s: train rows=%d test rows=%d train features=%d test features=%d",
            name,
            len(train),
            len(test),
            len(feature_columns(train)),
            len(feature_columns(test)),
        )
    log_phase("build feature sets", started)
    return feature_sets


def save_predictions(
    predictions: pd.DataFrame, model: str, feature_set: str, covariates: str
) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        PREDICTIONS_DIR
        / f"severity_predictions_{safe_filename(model)}_{safe_filename(feature_set)}_{safe_filename(covariates)}.csv",
        index=False,
    )


def prediction_path(model: str, feature_set: str, covariates: str) -> Path:
    return (
        PREDICTIONS_DIR
        / f"severity_predictions_{safe_filename(model)}_{safe_filename(feature_set)}_{safe_filename(covariates)}.csv"
    )


def evaluate_ridge(
    train: pd.DataFrame, test: pd.DataFrame, feature_set: str, covariate_mode: str, use_log: bool
) -> dict:
    cols, train_aligned, test_aligned = align_train_test(train, test)
    train_aligned, test_aligned, cols = add_known_covariates(
        train_aligned, test_aligned, cols, covariate_mode
    )
    target_col = TARGET_LOG if use_log else TARGET
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    fit_t0 = time.perf_counter()
    pipeline.fit(train_aligned[cols], train_aligned[target_col].to_numpy(float))
    fit_time = time.perf_counter() - fit_t0
    pred_t0 = time.perf_counter()
    pred = pipeline.predict(test_aligned[cols])
    if use_log:
        pred = np.clip(np.expm1(pred), 0, None)
    y_train = train_aligned[TARGET].to_numpy(float)
    pred = np.clip(pred, float(np.nanmin(y_train)), float(np.nanmax(y_train)))
    predict_time = time.perf_counter() - pred_t0
    y = test_aligned[TARGET].to_numpy(float)
    model = "ridge_log_severity" if use_log else "ridge_raw_severity"
    predictions = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    predictions["model"] = model
    predictions["feature_set"] = feature_set
    predictions["covariates"] = covariate_mode
    predictions["y_true"] = y
    predictions["y_pred"] = pred
    save_predictions(predictions, model, feature_set, covariate_mode)
    return {
        "feature_set": feature_set,
        "model": model,
        "covariates": covariate_mode,
        "n_train": len(train_aligned),
        "n_test": len(test_aligned),
        "n_features": len(cols),
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred) if len(np.unique(y)) > 1 else math.nan,
        "spearman": safe_spearman(y, pred),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }


def evaluate_reduced_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
    covariate_mode: str,
    reduction: str,
) -> dict:
    cols, train_aligned, test_aligned = align_train_test(train, test)
    train_aligned, test_aligned, cols = add_known_covariates(
        train_aligned, test_aligned, cols, covariate_mode
    )
    if reduction == "pca95":
        reducer = PCA(n_components=PCA_EXPLAINED_VARIANCE, svd_solver="full", random_state=SEED)
        model_name = "ridge_pca95_raw_severity"
    elif reduction == "selectk":
        k = min(SELECT_K_FEATURES, len(cols), max(1, len(train_aligned) - 2))
        reducer = SelectKBest(score_func=f_regression, k=k)
        model_name = "ridge_selectk_raw_severity"
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reducer", reducer),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    fit_t0 = time.perf_counter()
    pipeline.fit(train_aligned[cols], train_aligned[TARGET].to_numpy(float))
    fit_time = time.perf_counter() - fit_t0
    pred_t0 = time.perf_counter()
    pred = pipeline.predict(test_aligned[cols])
    y_train = train_aligned[TARGET].to_numpy(float)
    pred = np.clip(pred, float(np.nanmin(y_train)), float(np.nanmax(y_train)))
    predict_time = time.perf_counter() - pred_t0
    y = test_aligned[TARGET].to_numpy(float)
    fitted_reducer = pipeline.named_steps["reducer"]
    n_reduced = int(
        getattr(fitted_reducer, "n_components_", getattr(fitted_reducer, "k", len(cols)))
    )
    predictions = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    predictions["model"] = model_name
    predictions["feature_set"] = feature_set
    predictions["covariates"] = covariate_mode
    predictions["reduction"] = reduction
    predictions["y_true"] = y
    predictions["y_pred"] = pred
    save_predictions(predictions, model_name, feature_set, covariate_mode)
    return {
        "feature_set": feature_set,
        "model": model_name,
        "covariates": covariate_mode,
        "reduction": reduction,
        "n_train": len(train_aligned),
        "n_test": len(test_aligned),
        "n_features": len(cols),
        "n_reduced_features": n_reduced,
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred) if len(np.unique(y)) > 1 else math.nan,
        "spearman": safe_spearman(y, pred),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }


def select_stable_features(
    train_aligned: pd.DataFrame,
    cols: list[str],
    feature_set: str,
    covariate_mode: str,
) -> tuple[list[str], pd.DataFrame]:
    started = time.perf_counter()
    groups = train_aligned["plot_id"].to_numpy()
    splitter = GroupShuffleSplit(
        n_splits=STABILITY_REPEATS, test_size=STABILITY_TEST_SIZE, random_state=SEED
    )
    counts = pd.Series(0, index=cols, dtype=float)
    abs_coef_sum = pd.Series(0.0, index=cols, dtype=float)
    fit_rows = []
    for repeat, (fit_idx, _) in enumerate(
        splitter.split(train_aligned, train_aligned[TARGET], groups=groups)
    ):
        fit_part = train_aligned.iloc[fit_idx].copy()
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "elasticnet",
                    ElasticNetCV(
                        l1_ratio=STABILITY_L1_RATIOS,
                        alphas=STABILITY_ALPHAS,
                        cv=3,
                        max_iter=5000,
                        random_state=SEED + repeat,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipeline.fit(fit_part[cols], fit_part[TARGET].to_numpy(float))
        coefs = np.asarray(pipeline.named_steps["elasticnet"].coef_, dtype=float)
        selected = np.abs(coefs) > 1e-8
        counts += selected.astype(float)
        abs_coef_sum += np.abs(coefs)
        fit_rows.append(
            {
                "repeat": repeat,
                "feature_set": feature_set,
                "covariates": covariate_mode,
                "alpha": float(pipeline.named_steps["elasticnet"].alpha_),
                "l1_ratio": float(pipeline.named_steps["elasticnet"].l1_ratio_),
                "selected_features": int(selected.sum()),
            }
        )
    frequency = counts / STABILITY_REPEATS
    mean_abs_coef = abs_coef_sum / STABILITY_REPEATS
    selection = pd.DataFrame(
        {
            "feature_set": feature_set,
            "covariates": covariate_mode,
            "feature": cols,
            "selection_frequency": frequency.to_numpy(float),
            "mean_abs_elasticnet_coef": mean_abs_coef.to_numpy(float),
        }
    ).sort_values(["selection_frequency", "mean_abs_elasticnet_coef"], ascending=[False, False])
    selection["meets_stability_threshold"] = (
        selection["selection_frequency"] >= STABILITY_MIN_FREQUENCY
    )
    selected_features = selection.loc[selection["meets_stability_threshold"], "feature"].tolist()
    selection_mode = "threshold"
    if not selected_features:
        selection_mode = "fallback_top_k"
        selected_features = selection.head(min(STABILITY_FALLBACK_TOP_K, len(selection)))[
            "feature"
        ].tolist()
    fit_audit = pd.DataFrame(fit_rows)
    selection["mean_selected_features_per_repeat"] = (
        float(fit_audit["selected_features"].mean()) if not fit_audit.empty else np.nan
    )
    selection["mean_elasticnet_alpha"] = (
        float(fit_audit["alpha"].mean()) if not fit_audit.empty else np.nan
    )
    selection["mean_elasticnet_l1_ratio"] = (
        float(fit_audit["l1_ratio"].mean()) if not fit_audit.empty else np.nan
    )
    selection["selection_mode"] = selection_mode
    logging.info(
        "[PHASE] stability selection %s %s: selected %d/%d features in %.1fs",
        feature_set,
        covariate_mode,
        len(selected_features),
        len(cols),
        time.perf_counter() - started,
    )
    return selected_features, selection


def evaluate_stability_selected_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
    covariate_mode: str,
) -> tuple[dict, pd.DataFrame]:
    cols, train_aligned, test_aligned = align_train_test(train, test)
    train_aligned, test_aligned, cols = add_known_covariates(
        train_aligned, test_aligned, cols, covariate_mode
    )
    selected_cols, selection = select_stable_features(
        train_aligned, cols, feature_set, covariate_mode
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    fit_t0 = time.perf_counter()
    pipeline.fit(train_aligned[selected_cols], train_aligned[TARGET].to_numpy(float))
    fit_time = time.perf_counter() - fit_t0
    pred_t0 = time.perf_counter()
    pred = pipeline.predict(test_aligned[selected_cols])
    y_train = train_aligned[TARGET].to_numpy(float)
    pred = np.clip(pred, float(np.nanmin(y_train)), float(np.nanmax(y_train)))
    predict_time = time.perf_counter() - pred_t0
    y = test_aligned[TARGET].to_numpy(float)
    model_name = "ridge_stability_raw_severity"
    predictions = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    predictions["model"] = model_name
    predictions["feature_set"] = feature_set
    predictions["covariates"] = covariate_mode
    predictions["y_true"] = y
    predictions["y_pred"] = pred
    save_predictions(predictions, model_name, feature_set, covariate_mode)
    result = {
        "feature_set": feature_set,
        "model": model_name,
        "covariates": covariate_mode,
        "reduction": "elasticnet_stability",
        "n_train": len(train_aligned),
        "n_test": len(test_aligned),
        "n_features": len(cols),
        "n_reduced_features": len(selected_cols),
        "stability_min_frequency": STABILITY_MIN_FREQUENCY,
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred) if len(np.unique(y)) > 1 else math.nan,
        "spearman": safe_spearman(y, pred),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }
    selection["selected_for_final_model"] = selection["feature"].isin(selected_cols)
    return result, selection


def evaluate_xgboost(
    train: pd.DataFrame, test: pd.DataFrame, feature_set: str, covariate_mode: str
) -> dict:
    cols, train_aligned, test_aligned = align_train_test(train, test)
    train_aligned, test_aligned, cols = add_known_covariates(
        train_aligned, test_aligned, cols, covariate_mode
    )
    train_idx, eval_idx = split_train_eval_by_plot(train_aligned)
    fit_part = train_aligned.iloc[train_idx].copy()
    eval_part = train_aligned.iloc[eval_idx].copy()
    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(fit_part[cols])
    x_eval = imputer.transform(eval_part[cols])
    x_test = imputer.transform(test_aligned[cols])
    y_fit = fit_part[TARGET].to_numpy(float)
    y_eval = eval_part[TARGET].to_numpy(float)
    y_test = test_aligned[TARGET].to_numpy(float)
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=1200,
        learning_rate=0.015,
        max_depth=2,
        min_child_weight=12,
        subsample=0.70,
        colsample_bytree=0.55,
        reg_alpha=2.0,
        reg_lambda=25.0,
        gamma=1.0,
        early_stopping_rounds=50,
        random_state=SEED,
        tree_method="hist",
        n_jobs=4,
    )
    fit_t0 = time.perf_counter()
    model.fit(x_fit, y_fit, eval_set=[(x_fit, y_fit), (x_eval, y_eval)], verbose=False)
    fit_time = time.perf_counter() - fit_t0
    pred_t0 = time.perf_counter()
    eval_pred = np.clip(model.predict(x_eval), float(np.nanmin(y_fit)), float(np.nanmax(y_fit)))
    test_pred = np.clip(model.predict(x_test), float(np.nanmin(y_fit)), float(np.nanmax(y_fit)))
    predict_time = time.perf_counter() - pred_t0
    predictions = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    predictions["model"] = "xgboost_regularized_raw_severity"
    predictions["feature_set"] = feature_set
    predictions["covariates"] = covariate_mode
    predictions["y_true"] = y_test
    predictions["y_pred"] = test_pred
    save_predictions(predictions, "xgboost_regularized_raw_severity", feature_set, covariate_mode)
    return {
        "feature_set": feature_set,
        "model": "xgboost_regularized_raw_severity",
        "covariates": covariate_mode,
        "n_train_fit": len(fit_part),
        "n_train_eval": len(eval_part),
        "n_test": len(test_aligned),
        "n_features": len(cols),
        "best_iteration": getattr(model, "best_iteration", np.nan),
        "eval_rmse_2024": math.sqrt(mean_squared_error(y_eval, eval_pred)),
        "rmse": math.sqrt(mean_squared_error(y_test, test_pred)),
        "mae": mean_absolute_error(y_test, test_pred),
        "r2": r2_score(y_test, test_pred) if len(np.unique(y_test)) > 1 else math.nan,
        "spearman": safe_spearman(y_test, test_pred),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }


def evaluate_lagged_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
    covariate_mode: str,
    lag_policy: str,
) -> dict:
    cols, train_aligned, test_aligned = align_train_test(train, test)
    train_aligned, test_aligned, cols = add_lagged_known_covariates(
        train_aligned, test_aligned, cols, covariate_mode
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    fit_t0 = time.perf_counter()
    pipeline.fit(train_aligned[cols], train_aligned[TARGET].to_numpy(float))
    fit_time = time.perf_counter() - fit_t0
    pred_t0 = time.perf_counter()
    pred = pipeline.predict(test_aligned[cols])
    y_train = train_aligned[TARGET].to_numpy(float)
    pred = np.clip(pred, float(np.nanmin(y_train)), float(np.nanmax(y_train)))
    predict_time = time.perf_counter() - pred_t0
    y = test_aligned[TARGET].to_numpy(float)
    predictions = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    predictions["model"] = "ridge_raw_severity_lagged"
    predictions["feature_set"] = feature_set
    predictions["covariates"] = covariate_mode
    predictions["lag_policy"] = lag_policy
    predictions["y_true"] = y
    predictions["y_pred"] = pred
    save_predictions(
        predictions, f"ridge_raw_severity_lagged_{lag_policy}", feature_set, covariate_mode
    )
    return {
        "feature_set": feature_set,
        "model": "ridge_raw_severity_lagged",
        "covariates": covariate_mode,
        "lag_policy": lag_policy,
        "n_train": len(train_aligned),
        "n_test": len(test_aligned),
        "n_features": len(cols),
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred) if len(np.unique(y)) > 1 else math.nan,
        "spearman": safe_spearman(y, pred),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }


def delta_vs_nadir(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline_to_candidates = {
        "distribution_nadir": ["distribution_multiangular", "distribution_multiangular_shape"],
        "compact_anomaly_nadir": [
            "compact_anomaly_multiangular",
            "compact_anomaly_multiangular_shape",
        ],
    }
    for (model, covariates), group in results.groupby(["model", "covariates"], dropna=False):
        for baseline_name, candidate_names in baseline_to_candidates.items():
            baseline = group[group["feature_set"] == baseline_name]
            if baseline.empty:
                continue
            base = baseline.iloc[0]
            for _, row in group[group["feature_set"].isin(candidate_names)].iterrows():
                rows.append(
                    {
                        "model": model,
                        "covariates": covariates,
                        "baseline": baseline_name,
                        "feature_set": row["feature_set"],
                        "rmse_reduction_vs_baseline": base["rmse"] - row["rmse"],
                        "mae_reduction_vs_baseline": base["mae"] - row["mae"],
                        "delta_r2": row["r2"] - base["r2"],
                        "delta_spearman": row["spearman"] - base["spearman"],
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("rmse_reduction_vs_baseline", ascending=False)


def regression_metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else math.nan,
        "spearman": safe_spearman(y_true, y_pred),
    }


def paired_bootstrap_delta_ci(
    baseline_pred: pd.DataFrame,
    candidate_pred: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, float]:
    key_cols = ["plot_id", "predictor_week", "target_week"]
    merged = baseline_pred[key_cols + ["y_true", "y_pred"]].merge(
        candidate_pred[key_cols + ["y_true", "y_pred"]],
        on=key_cols + ["y_true"],
        suffixes=("_baseline", "_candidate"),
        how="inner",
    )
    if merged.empty:
        return {}

    plot_ids = np.asarray(sorted(merged["plot_id"].unique()))
    plot_to_indices = {
        plot_id: merged.index[merged["plot_id"] == plot_id].to_numpy() for plot_id in plot_ids
    }
    observed_base = regression_metric_values(
        merged["y_true"].to_numpy(float),
        merged["y_pred_baseline"].to_numpy(float),
    )
    observed_candidate = regression_metric_values(
        merged["y_true"].to_numpy(float),
        merged["y_pred_candidate"].to_numpy(float),
    )
    observed = {
        "rmse_reduction": observed_base["rmse"] - observed_candidate["rmse"],
        "mae_reduction": observed_base["mae"] - observed_candidate["mae"],
        "delta_r2": observed_candidate["r2"] - observed_base["r2"],
        "delta_spearman": observed_candidate["spearman"] - observed_base["spearman"],
    }
    samples = {metric: [] for metric in observed}
    for _ in range(PAIRED_BOOTSTRAP_ITERATIONS):
        sampled_plots = rng.choice(plot_ids, size=len(plot_ids), replace=True)
        sampled_indices = np.concatenate([plot_to_indices[plot_id] for plot_id in sampled_plots])
        sampled = merged.loc[sampled_indices]
        y_true = sampled["y_true"].to_numpy(float)
        base = regression_metric_values(y_true, sampled["y_pred_baseline"].to_numpy(float))
        cand = regression_metric_values(y_true, sampled["y_pred_candidate"].to_numpy(float))
        samples["rmse_reduction"].append(base["rmse"] - cand["rmse"])
        samples["mae_reduction"].append(base["mae"] - cand["mae"])
        samples["delta_r2"].append(cand["r2"] - base["r2"])
        samples["delta_spearman"].append(cand["spearman"] - base["spearman"])

    ci_low = 100 * PAIRED_BOOTSTRAP_ALPHA / 2
    ci_high = 100 * (1 - PAIRED_BOOTSTRAP_ALPHA / 2)
    out: dict[str, float] = {
        "n_test_rows": int(len(merged)),
        "n_plots": int(len(plot_ids)),
        "n_bootstrap": PAIRED_BOOTSTRAP_ITERATIONS,
    }
    for metric, values in samples.items():
        arr = np.asarray(values, dtype=float)
        out[f"{metric}_observed"] = observed[metric]
        out[f"{metric}_ci_low"] = float(np.nanpercentile(arr, ci_low))
        out[f"{metric}_ci_high"] = float(np.nanpercentile(arr, ci_high))
        out[f"{metric}_prob_gt_zero"] = float(np.nanmean(arr > 0))
    return out


def bootstrap_delta_vs_nadir(deltas: pd.DataFrame) -> pd.DataFrame:
    started = time.perf_counter()
    rows = []
    rng = np.random.default_rng(SEED)
    for _, delta in deltas.iterrows():
        model = str(delta["model"])
        covariates = str(delta["covariates"])
        baseline = str(delta["baseline"])
        feature_set = str(delta["feature_set"])
        baseline_path = prediction_path(model, baseline, covariates)
        candidate_path = prediction_path(model, feature_set, covariates)
        if not baseline_path.exists() or not candidate_path.exists():
            continue
        baseline_pred = pd.read_csv(baseline_path)
        candidate_pred = pd.read_csv(candidate_path)
        stats = paired_bootstrap_delta_ci(baseline_pred, candidate_pred, rng)
        if not stats:
            continue
        rows.append(
            {
                "model": model,
                "covariates": covariates,
                "baseline": baseline,
                "feature_set": feature_set,
                "resample_unit": "plot_id",
                **stats,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("rmse_reduction_observed", ascending=False)
    logging.info("[PHASE] paired bootstrap delta CI: %.1fs", time.perf_counter() - started)
    return out


def write_report(
    outputs: dict[str, Path],
    results: pd.DataFrame,
    deltas: pd.DataFrame,
    delta_ci: pd.DataFrame,
    audit: pd.DataFrame,
    log_path: Path,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best_cols = [
        "feature_set",
        "model",
        "covariates",
        "n_features",
        "n_reduced_features",
        "rmse",
        "mae",
        "r2",
        "spearman",
    ]
    report = [
        "## Results: Multiangular Image-Distribution Feature Family",
        "",
        "This experiment tests whether per-pixel distribution summaries from multiangular plot images improve external-year severity prediction over matched nadir-only distribution baselines. It includes both broad distribution features and a compact anomaly-style subset.",
        "",
        "### Best External Severity Results",
        "",
        markdown_table(results.sort_values("rmse")[best_cols].round(4), max_rows=20),
        "",
        "### Delta vs Distribution Nadir",
        "",
        "Positive RMSE reduction means the multiangular feature set improved over its matched nadir baseline under the same model and covariate mode.",
        "",
        markdown_table(deltas.round(4), max_rows=30),
        "",
        "### Paired Bootstrap CI For Delta vs Nadir",
        "",
        "Confidence intervals resample external-test `plot_id` values with replacement and calculate candidate-minus-baseline performance on the same sampled plots. For RMSE and MAE, positive values mean error reduction versus nadir.",
        "",
        markdown_table(delta_ci.round(4), max_rows=20),
        "",
        "### Input Audit",
        "",
        markdown_table(audit.round(3), max_rows=20),
        "",
        "**Interpretation**: Distribution features test the SugarViT-like idea that image heterogeneity and low-tail vegetation/symptom proxies may contain disease-relevant information beyond plot mean reflectance. Any paper claim should focus on paired improvement over the matched nadir-distribution baseline.",
        "",
        "### Reproducibility",
        "",
        "- Train year: `2024`",
        "- Test year: `2025`",
        f"- Ground filter: OSAVI > `{OSAVI_THRESHOLD}`",
        f"- Fine VZA bins: `{FINE_VZA_MIN}` to `{FINE_VZA_MAX}` degrees in 5-degree bins",
        f"- Minimum pixels per plot/week/VZA/bin: `{MIN_BIN_PIXELS}`",
        "- Broad distribution metrics: mean, median, p05, p10, p25, p75, p90, p95, IQR, CV, OSAVI low-tail fractions",
        "- Compact anomaly metrics: NIR/red-edge/red/OSAVI p10, p25, IQR, CV, OSAVI low-tail fractions, plus compact angular-shape summaries",
        f"- Dimensionality reduction: PCA fitted on 2024 only to explain `{PCA_EXPLAINED_VARIANCE:.0%}` variance; SelectKBest fitted on 2024 only with target `k={SELECT_K_FEATURES}` where available",
        f"- Stability selection: ElasticNetCV fitted on `{STABILITY_REPEATS}` grouped 2024 splits by plot; final features require frequency >= `{STABILITY_MIN_FREQUENCY}` or fallback top `{STABILITY_FALLBACK_TOP_K}` when no feature meets the threshold. Fallback-selected features are marked in `selection_mode` and should not be interpreted as stable.",
        f"- Paired CI: `{PAIRED_BOOTSTRAP_ITERATIONS}` bootstrap resamples of external-test `plot_id` values, percentile `{100 * (1 - PAIRED_BOOTSTRAP_ALPHA):.0f}%` CI",
        "- Models: Ridge raw/log severity, Ridge PCA, Ridge SelectK, Ridge stability-selected, and heavily regularized XGBoost raw severity",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
    ]
    for label, path in outputs.items():
        report.append(f"- {label}: `{path}`")
    outputs["report"].write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    total_t0 = time.perf_counter()
    log_path = setup_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    disease_t0 = time.perf_counter()
    disease_2024 = load_2024_disease_with_fallback()
    disease_2025, audit_2025 = load_2025_disease_with_fallback()
    log_phase("load disease scores", disease_t0)

    long_2024_path = RESULTS_DIR / "distribution_features_long_2024.csv"
    long_2025_path = RESULTS_DIR / "distribution_features_long_2025.csv"
    audit_path = RESULTS_DIR / "distribution_feature_input_audit.csv"
    if long_2024_path.exists() and long_2025_path.exists() and audit_path.exists():
        cache_t0 = time.perf_counter()
        logging.info("Reusing cached distribution feature tables from %s", RESULTS_DIR)
        long_2024 = pd.read_csv(long_2024_path)
        long_2025 = pd.read_csv(long_2025_path)
        full_audit = pd.read_csv(audit_path)
        log_phase("load cached distribution feature tables", cache_t0)
    else:
        long_2024, audit_2024 = build_distribution_long(2024)
        long_2025, audit_2025_features = build_distribution_long(2025)
        full_audit = pd.concat([audit_2024, audit_2025_features], ignore_index=True)
        feature_output_t0 = time.perf_counter()
        long_2024.to_csv(long_2024_path, index=False)
        long_2025.to_csv(long_2025_path, index=False)
        full_audit.to_csv(audit_path, index=False)
        log_phase("write distribution feature tables", feature_output_t0)

    features = build_feature_sets(long_2024, long_2025)

    rows = []
    stability_selection_rows = []
    model_t0 = time.perf_counter()
    for feature_set, (train_features, test_features) in features.items():
        train = build_model_table(train_features, disease_2024)
        test = build_model_table(test_features, disease_2025)
        if train.empty or test.empty:
            logging.warning("Skipping empty model table for %s", feature_set)
            continue
        for covariate_mode in ["spectral_only", "spectral_plus_week", "spectral_plus_week_horizon"]:
            rows.append(evaluate_ridge(train, test, feature_set, covariate_mode, use_log=False))
            rows.append(evaluate_ridge(train, test, feature_set, covariate_mode, use_log=True))
            rows.append(
                evaluate_reduced_ridge(train, test, feature_set, covariate_mode, reduction="pca95")
            )
            rows.append(
                evaluate_reduced_ridge(
                    train, test, feature_set, covariate_mode, reduction="selectk"
                )
            )
            rows.append(evaluate_xgboost(train, test, feature_set, covariate_mode))

        if feature_set.startswith("compact_anomaly"):
            for covariate_mode in ["spectral_only", "spectral_plus_week_horizon"]:
                result, selection = evaluate_stability_selected_ridge(
                    train, test, feature_set, covariate_mode
                )
                rows.append(result)
                stability_selection_rows.append(selection)

        train_lagged = add_lagged_disease_features(train, disease_2024, "before_predictor_week")
        test_lagged = add_lagged_disease_features(test, disease_2025, "before_predictor_week")
        for covariate_mode in ["lagged_only", "lagged_plus_predictor_week"]:
            rows.append(
                evaluate_lagged_ridge(
                    train_lagged,
                    test_lagged,
                    feature_set,
                    covariate_mode,
                    lag_policy="before_predictor_week",
                )
            )
    log_phase("fit/evaluate models", model_t0)

    results = pd.DataFrame(rows).sort_values(["rmse", "feature_set"])
    deltas = delta_vs_nadir(results)
    delta_ci = bootstrap_delta_vs_nadir(deltas)
    results_path = RESULTS_DIR / "distribution_feature_external_2024_train_2025_test.csv"
    deltas_path = RESULTS_DIR / "distribution_feature_delta_vs_nadir.csv"
    delta_ci_path = RESULTS_DIR / "distribution_feature_delta_vs_nadir_paired_bootstrap_ci.csv"
    stability_selection_path = RESULTS_DIR / "stability_selection_feature_frequencies.csv"
    results.to_csv(results_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    delta_ci.to_csv(delta_ci_path, index=False)
    if stability_selection_rows:
        pd.concat(stability_selection_rows, ignore_index=True).to_csv(
            stability_selection_path, index=False
        )
    else:
        pd.DataFrame().to_csv(stability_selection_path, index=False)

    audit_summary = full_audit.groupby(["year", "week"], as_index=False).agg(
        plots=("plot_id", "nunique"),
        original_rows=("original_rows", "sum"),
        rows_after_quality=("rows_after_quality", "sum"),
        rows_after_ground_filter=("rows_after_ground_filter", "sum"),
        rows_after_sampling=("rows_after_sampling", "sum"),
    )
    audit_summary_path = RESULTS_DIR / "distribution_feature_input_audit_summary.csv"
    audit_summary.to_csv(audit_summary_path, index=False)

    outputs = {
        "results": results_path,
        "delta_vs_nadir": deltas_path,
        "delta_vs_nadir_paired_bootstrap_ci": delta_ci_path,
        "stability_selection": stability_selection_path,
        "long_features_2024": long_2024_path,
        "long_features_2025": long_2025_path,
        "input_audit": audit_path,
        "input_audit_summary": audit_summary_path,
        "predictions": PREDICTIONS_DIR,
        "report": REPORTS_DIR / "multiangular_distribution_feature_family_summary.md",
    }
    write_report(outputs, results, deltas, delta_ci, audit_summary, log_path)
    logging.info("[PHASE] total: %.1fs", time.perf_counter() - total_t0)


if __name__ == "__main__":
    main()
