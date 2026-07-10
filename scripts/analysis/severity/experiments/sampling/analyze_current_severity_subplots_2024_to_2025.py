#!/usr/bin/env python3
"""Exploratory current-severity models using 20 spatial subplots per plot-week."""

from __future__ import annotations

import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity.analyze_cross_year_generalization_2024_to_2025 import (
    ALPHAS,
    SEED,
    TARGET,
)

DATA_ROOT = Path("/run/media/davidem/data/ONCERCO")
OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/subplots_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"
FEATURE_CACHE_DIR = RESULTS_DIR / "feature_cache"
DISEASE_2024_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2024.csv"
DISEASE_2025_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2025.csv"

N_SUBPLOT_X = 5
N_SUBPLOT_Y = 4
MAX_ROWS_PER_PLOT = 500_000
MIN_BIN_PIXELS = 100
OSAVI_THRESHOLD = 0.20
VZA_MIN = 10
VZA_MAX = 55
BANDS = {
    "band1": "blue",
    "band2": "green",
    "band3": "red",
    "band4": "red_edge",
    "band5": "nir",
}
META_COLS = {"year", "week", "plot_id", "parent_plot_id", "subplot_id", TARGET}

PLOT_DIRS = {
    2024: {
        0: DATA_ROOT
        / "data/processed/2024/20240603_week0/metashape/20241205_products_uav_data/output/extract/polygon_df",
        2: DATA_ROOT / "data/extracted/2024/week2/output/plots",
        3: DATA_ROOT
        / "data/processed/2024/20240624_week3/metashape/20241206_week3_products_uav_data/output/plots",
        4: DATA_ROOT / "data/extracted/2024/week4/output/plots",
        5: DATA_ROOT
        / "data/processed/2024/20240715_week5/metashape/20241207_week5_products_uav_data/output/plots",
        6: DATA_ROOT / "data/extracted/2024/week6/output/plots",
        7: DATA_ROOT / "data/extracted/2024/week7/output/plots",
        8: DATA_ROOT
        / "data/processed/2024/20240826_week8/metashape/20241029_products_uav_data/output/extract/polygon_df",
    },
    2025: {
        0: DATA_ROOT
        / "data/processed/2025/week0/metashape/20250822_products_uas_data/output/plots",
        3: DATA_ROOT
        / "data/processed/2025/week3/metashape/20250828_products_uas_data/output/plots",
        5: DATA_ROOT
        / "data/processed/2025/week5/metashape/20250829_products_uas_data/output/plots",
        7: DATA_ROOT / "data/extracted/2025/week7/output/plots",
    },
}


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_current_severity_subplots_2024_to_2025_{timestamp}.log"
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
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return math.nan
    value = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return float(value) if value is not None else math.nan


def ensure_osavi(frame: pl.DataFrame) -> pl.DataFrame:
    if "OSAVI" in frame.columns:
        return frame
    return frame.with_columns(
        (1.16 * (pl.col("band5") - pl.col("band3")) / (pl.col("band5") + pl.col("band3") + 0.16))
        .cast(pl.Float32)
        .alias("OSAVI")
    )


def assign_subplots(frame: pl.DataFrame) -> pl.DataFrame:
    bounds = frame.select(
        pl.col("Xw").min().alias("xmin"),
        pl.col("Xw").max().alias("xmax"),
        pl.col("Yw").min().alias("ymin"),
        pl.col("Yw").max().alias("ymax"),
    ).row(0, named=True)
    x_span = max(float(bounds["xmax"] - bounds["xmin"]), 1e-9)
    y_span = max(float(bounds["ymax"] - bounds["ymin"]), 1e-9)
    x_bin = (
        (((pl.col("Xw") - bounds["xmin"]) / x_span) * N_SUBPLOT_X)
        .floor()
        .clip(0, N_SUBPLOT_X - 1)
        .cast(pl.Int32)
    )
    y_bin = (
        (((pl.col("Yw") - bounds["ymin"]) / y_span) * N_SUBPLOT_Y)
        .floor()
        .clip(0, N_SUBPLOT_Y - 1)
        .cast(pl.Int32)
    )
    return frame.with_columns((y_bin * N_SUBPLOT_X + x_bin).alias("subplot_num"))


def assign_vza_bins(frame: pl.DataFrame) -> pl.DataFrame:
    low = (((pl.col("vza") - VZA_MIN) / 5).floor() * 5 + VZA_MIN).cast(pl.Int32)
    return (
        frame.filter((pl.col("vza") >= VZA_MIN) & (pl.col("vza") < VZA_MAX))
        .with_columns(low.alias("vza_low"))
        .with_columns(
            (
                pl.col("vza_low").cast(pl.Utf8)
                + pl.lit("_")
                + (pl.col("vza_low") + 5).cast(pl.Utf8)
            ).alias("vza_class"),
            (pl.col("vza_low") + 2.5).cast(pl.Float64).alias("vza_midpoint"),
        )
        .drop("vza_low")
    )


def sample_rows(frame: pl.DataFrame, max_rows: int = MAX_ROWS_PER_PLOT) -> pl.DataFrame:
    if frame.height <= max_rows:
        return frame
    return frame.sample(n=max_rows, seed=SEED)


def summarize_plot(path: Path, year: int, week: int) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    parent_plot_id = path.stem
    try:
        available = set(pl.scan_parquet(path).collect_schema().names())
        columns = ["Xw", "Yw", "band1", "band2", "band3", "band4", "band5", "vza"]
        if "OSAVI" in available:
            columns.append("OSAVI")
        frame = pl.read_parquet(path, columns=columns)
    except Exception as exc:
        logging.warning("Skipping unreadable parquet %s: %s", path, exc)
        return [], {
            "year": year,
            "week": week,
            "plot_id": parent_plot_id,
            "original_rows": 0,
            "rows_after_filter": 0,
            "rows_after_sample": 0,
            "feature_rows": 0,
            "seconds": time.perf_counter() - t0,
            "error": str(exc),
        }
    original_rows = frame.height
    frame = ensure_osavi(frame)
    quality = (
        pl.col("vza").is_finite()
        & (pl.col("vza") >= VZA_MIN)
        & (pl.col("vza") < VZA_MAX)
        & pl.col("OSAVI").is_finite()
        & (pl.col("OSAVI") > OSAVI_THRESHOLD)
    )
    for band in BANDS:
        quality = quality & pl.col(band).is_finite() & (pl.col(band) > 0)
    frame = frame.filter(quality)
    rows_after_filter = frame.height
    if frame.is_empty():
        return [], {
            "year": year,
            "week": week,
            "plot_id": parent_plot_id,
            "original_rows": original_rows,
            "rows_after_filter": rows_after_filter,
            "rows_after_sample": 0,
            "feature_rows": 0,
            "seconds": time.perf_counter() - t0,
        }
    frame = assign_subplots(assign_vza_bins(sample_rows(frame)))
    rows_after_sample = frame.height

    long = frame.unpivot(
        index=["subplot_num", "vza_class", "vza_midpoint"],
        on=list(BANDS),
        variable_name="band",
        value_name="reflectance",
    )
    summary = (
        long.group_by("subplot_num", "vza_class", "vza_midpoint", "band")
        .agg(
            pl.len().alias("n_pixels"),
            pl.col("reflectance").mean().alias("mean"),
            pl.col("reflectance").quantile(0.10).alias("p10"),
            pl.col("reflectance").quantile(0.25).alias("p25"),
            pl.col("reflectance").quantile(0.75).alias("p75"),
            pl.col("reflectance").std().alias("sd"),
        )
        .with_columns(
            (pl.col("p75") - pl.col("p25")).alias("iqr"),
            (pl.col("sd") / pl.col("mean")).alias("cv"),
        )
        .filter(pl.col("n_pixels") >= MIN_BIN_PIXELS)
    )
    osavi = (
        frame.group_by("subplot_num", "vza_class", "vza_midpoint")
        .agg(
            pl.len().alias("n_pixels"),
            pl.col("OSAVI").mean().alias("mean"),
            pl.col("OSAVI").quantile(0.10).alias("p10"),
            pl.col("OSAVI").quantile(0.25).alias("p25"),
            pl.col("OSAVI").quantile(0.75).alias("p75"),
            (pl.col("OSAVI") < 0.25).mean().alias("frac_lt_025"),
            (pl.col("OSAVI") < 0.35).mean().alias("frac_lt_035"),
        )
        .with_columns((pl.col("p75") - pl.col("p25")).alias("iqr"))
        .filter(pl.col("n_pixels") >= MIN_BIN_PIXELS)
    )

    rows = []
    for record in summary.to_dicts():
        subplot = int(record["subplot_num"])
        subplot_id = f"s{subplot:02d}"
        for metric in ["mean", "p10", "p25", "iqr", "cv"]:
            value = record[metric]
            if value is None or not np.isfinite(value):
                continue
            rows.append(
                {
                    "year": year,
                    "week": week,
                    "parent_plot_id": parent_plot_id,
                    "subplot_id": subplot_id,
                    "plot_id": f"{parent_plot_id}__{subplot_id}",
                    "band_name": BANDS[str(record["band"])],
                    "vza_class": record["vza_class"],
                    "vza_midpoint": record["vza_midpoint"],
                    "metric": metric,
                    "value": float(value),
                }
            )
    for record in osavi.to_dicts():
        subplot = int(record["subplot_num"])
        subplot_id = f"s{subplot:02d}"
        for metric in ["mean", "p10", "iqr", "frac_lt_025", "frac_lt_035"]:
            value = record[metric]
            if value is None or not np.isfinite(value):
                continue
            rows.append(
                {
                    "year": year,
                    "week": week,
                    "parent_plot_id": parent_plot_id,
                    "subplot_id": subplot_id,
                    "plot_id": f"{parent_plot_id}__{subplot_id}",
                    "band_name": "osavi",
                    "vza_class": record["vza_class"],
                    "vza_midpoint": record["vza_midpoint"],
                    "metric": f"osavi_{metric}",
                    "value": float(value),
                }
            )
    return rows, {
        "year": year,
        "week": week,
        "plot_id": parent_plot_id,
        "original_rows": original_rows,
        "rows_after_filter": rows_after_filter,
        "rows_after_sample": rows_after_sample,
        "feature_rows": len(rows),
        "seconds": time.perf_counter() - t0,
    }


def build_long_features(year: int) -> pd.DataFrame:
    cache = FEATURE_CACHE_DIR / f"subplot_distribution_features_long_{year}.csv"
    audit_cache = FEATURE_CACHE_DIR / f"subplot_distribution_feature_audit_{year}.csv"
    if cache.exists():
        logging.info("Using cached subplot features: %s", cache)
        return pd.read_csv(cache)
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    audit_rows = []
    for week, directory in PLOT_DIRS[year].items():
        if not directory.exists():
            logging.warning("Skipping missing %s week %s: %s", year, week, directory)
            continue
        files = sorted(directory.glob("plot_*.parquet"))
        logging.info("%s week %s: %d plot files from %s", year, week, len(files), directory)
        for path in files:
            rows, audit = summarize_plot(path, year, week)
            all_rows.extend(rows)
            audit_rows.append(audit)
    long = pd.DataFrame(all_rows)
    audit = pd.DataFrame(audit_rows)
    long.to_csv(cache, index=False)
    audit.to_csv(audit_cache, index=False)
    return long


def pivot_features(long: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    data = long.copy()
    if feature_set == "nadir":
        idx = (
            data.sort_values("vza_midpoint")
            .groupby(
                ["year", "week", "plot_id", "parent_plot_id", "subplot_id", "band_name", "metric"]
            )
            .head(1)
            .index
        )
        data = data.loc[idx].copy()
        data["feature"] = (
            "subplot_nadir__" + data["band_name"].map(clean_token) + "__" + data["metric"]
        )
    elif feature_set == "multiangular":
        data["feature"] = (
            "subplot_multiangular__"
            + data["band_name"].map(clean_token)
            + "__"
            + data["vza_class"].map(clean_token)
            + "__"
            + data["metric"]
        )
    else:
        raise ValueError(feature_set)
    pivot = data.pivot_table(
        index=["year", "week", "plot_id", "parent_plot_id", "subplot_id"],
        columns="feature",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def load_disease() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(DISEASE_2024_CLEAN), pd.read_csv(DISEASE_2025_CLEAN)


def model_table(features: pd.DataFrame, disease: pd.DataFrame) -> pd.DataFrame:
    target = disease[["plot_id", "week", "ds_plot"]].rename(
        columns={"plot_id": "parent_plot_id", "ds_plot": TARGET}
    )
    table = features.merge(target, on=["parent_plot_id", "week"], how="inner")
    return table.sort_values(["week", "parent_plot_id", "subplot_id"]).reset_index(drop=True)


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in META_COLS]


def align(train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    cols = sorted(set(feature_columns(train)).intersection(feature_columns(test)))
    train = train.copy()
    test = test.copy()
    train[cols] = train[cols].replace([np.inf, -np.inf], np.nan)
    test[cols] = test[cols].replace([np.inf, -np.inf], np.nan)
    cols = [c for c in cols if train[c].notna().mean() >= 0.5 and test[c].notna().mean() > 0]
    train["known__week"] = train["week"]
    test["known__week"] = test["week"]
    cols.append("known__week")
    keep = ["plot_id", "parent_plot_id", "subplot_id", "week", TARGET] + cols
    return cols, train[keep], test[keep]


def score(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred) if len(np.unique(y)) > 1 else math.nan,
        "spearman": safe_spearman(y, pred),
    }


def ridge_model(
    train: pd.DataFrame, test: pd.DataFrame, feature_set: str
) -> tuple[dict, pd.DataFrame]:
    cols, train, test = align(train, test)
    y = train[TARGET].to_numpy(float)
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    pipe.fit(train[cols], y)
    pred = np.clip(pipe.predict(test[cols]), float(np.nanmin(y)), float(np.nanmax(y)))
    out = test[["plot_id", "parent_plot_id", "subplot_id", "week", TARGET]].copy()
    out["model"] = "ridge"
    out["feature_set"] = feature_set
    out["y_pred"] = pred
    result = {
        "model": "ridge",
        "feature_set": feature_set,
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(cols),
        **score(test[TARGET].to_numpy(float), pred),
    }
    return result, out


def phenology_floor_ridge(
    train: pd.DataFrame, test: pd.DataFrame, feature_set: str
) -> tuple[dict, pd.DataFrame]:
    result, out = ridge_model(train, test, feature_set)
    zero_weeks = train.groupby("week")[TARGET].max().loc[lambda s: s <= 0].index.to_numpy()
    if zero_weeks.size:
        out.loc[out["week"].isin(zero_weeks), "y_pred"] = 0.0
    metrics = score(out[TARGET].to_numpy(float), out["y_pred"].to_numpy(float))
    result = {
        **result,
        **metrics,
        "model": "phenology_floor_ridge",
        "zero_weeks": ",".join(map(str, zero_weeks.tolist())),
    }
    out["model"] = "phenology_floor_ridge"
    return result, out


def hurdle_model(
    train: pd.DataFrame, test: pd.DataFrame, feature_set: str
) -> tuple[dict, pd.DataFrame]:
    cols, train, test = align(train, test)
    y = train[TARGET].to_numpy(float)
    present = (y > 0).astype(int)
    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegressionCV(
                    Cs=np.logspace(-2, 1, 8),
                    cv=3,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    classifier.fit(train[cols], present)
    prob = classifier.predict_proba(test[cols])[:, 1]
    positive = y > 0
    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    regressor.fit(train.loc[positive, cols], y[positive])
    pred = prob * regressor.predict(test[cols])
    zero_weeks = train.groupby("week")[TARGET].max().loc[lambda s: s <= 0].index.to_numpy()
    if zero_weeks.size:
        pred[np.isin(test["week"].to_numpy(), zero_weeks)] = 0.0
    pred = np.clip(pred, float(np.nanmin(y)), float(np.nanmax(y)))
    out = test[["plot_id", "parent_plot_id", "subplot_id", "week", TARGET]].copy()
    out["model"] = "hurdle_probability_times_severity"
    out["feature_set"] = feature_set
    out["y_pred"] = pred
    result = {
        "model": "hurdle_probability_times_severity",
        "feature_set": feature_set,
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(cols),
        **score(test[TARGET].to_numpy(float), pred),
    }
    return result, out


def xgboost_model(
    train: pd.DataFrame, test: pd.DataFrame, feature_set: str
) -> tuple[dict, pd.DataFrame]:
    cols, train, test = align(train, test)
    y_train = train[TARGET].to_numpy(float)
    x_train = train[cols].replace([np.inf, -np.inf], np.nan)
    x_test = test[cols].replace([np.inf, -np.inf], np.nan)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    fit_idx, eval_idx = next(
        splitter.split(x_train, y_train, groups=train["parent_plot_id"].to_numpy())
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=600,
        learning_rate=0.025,
        max_depth=2,
        min_child_weight=25,
        subsample=0.75,
        colsample_bytree=0.70,
        reg_alpha=3.0,
        reg_lambda=50.0,
        gamma=1.0,
        random_state=SEED,
        n_jobs=1,
        early_stopping_rounds=40,
    )
    model.fit(
        x_train.iloc[fit_idx],
        y_train[fit_idx],
        eval_set=[(x_train.iloc[eval_idx], y_train[eval_idx])],
        verbose=False,
    )
    pred = model.predict(x_test)
    zero_weeks = train.groupby("week")[TARGET].max().loc[lambda s: s <= 0].index.to_numpy()
    if zero_weeks.size:
        pred[np.isin(test["week"].to_numpy(), zero_weeks)] = 0.0
    pred = np.clip(pred, float(np.nanmin(y_train)), float(np.nanmax(y_train)))
    out = test[["plot_id", "parent_plot_id", "subplot_id", "week", TARGET]].copy()
    out["model"] = "xgboost_grouped"
    out["feature_set"] = feature_set
    out["y_pred"] = pred
    result = {
        "model": "xgboost_grouped",
        "feature_set": feature_set,
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(cols),
        "best_iteration": getattr(model, "best_iteration", math.nan),
        "zero_weeks": ",".join(map(str, zero_weeks.tolist())),
        **score(test[TARGET].to_numpy(float), pred),
    }
    return result, out


def xgboost_hurdle_model(
    train: pd.DataFrame, test: pd.DataFrame, feature_set: str
) -> tuple[dict, pd.DataFrame]:
    cols, train, test = align(train, test)
    y_train = train[TARGET].to_numpy(float)
    present = (y_train > 0).astype(int)
    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegressionCV(
                    Cs=np.logspace(-2, 1, 8),
                    cv=3,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    classifier.fit(train[cols], present)
    probability = classifier.predict_proba(test[cols])[:, 1]

    positive = y_train > 0
    x_positive = train.loc[positive, cols].replace([np.inf, -np.inf], np.nan)
    y_positive = y_train[positive]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    fit_idx, eval_idx = next(
        splitter.split(
            x_positive, y_positive, groups=train.loc[positive, "parent_plot_id"].to_numpy()
        )
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=500,
        learning_rate=0.035,
        max_depth=2,
        min_child_weight=10,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=1.0,
        reg_lambda=15.0,
        gamma=0.0,
        random_state=SEED,
        n_jobs=1,
        early_stopping_rounds=40,
    )
    model.fit(
        x_positive.iloc[fit_idx],
        y_positive[fit_idx],
        eval_set=[(x_positive.iloc[eval_idx], y_positive[eval_idx])],
        verbose=False,
    )
    magnitude = model.predict(test[cols].replace([np.inf, -np.inf], np.nan))
    pred = probability * magnitude
    zero_weeks = train.groupby("week")[TARGET].max().loc[lambda s: s <= 0].index.to_numpy()
    if zero_weeks.size:
        pred[np.isin(test["week"].to_numpy(), zero_weeks)] = 0.0
    pred = np.clip(pred, float(np.nanmin(y_train)), float(np.nanmax(y_train)))
    out = test[["plot_id", "parent_plot_id", "subplot_id", "week", TARGET]].copy()
    out["model"] = "xgboost_hurdle"
    out["feature_set"] = feature_set
    out["y_pred"] = pred
    result = {
        "model": "xgboost_hurdle",
        "feature_set": feature_set,
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(cols),
        "best_iteration": getattr(model, "best_iteration", math.nan),
        "zero_weeks": ",".join(map(str, zero_weeks.tolist())),
        **score(test[TARGET].to_numpy(float), pred),
    }
    return result, out


def week_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["model", "feature_set", "week"]):
        y = group[TARGET].to_numpy(float)
        pred = group["y_pred"].to_numpy(float)
        rows.append(
            {
                "model": keys[0],
                "feature_set": keys[1],
                "week": keys[2],
                "n_subplots": len(group),
                "n_parent_plots": group["parent_plot_id"].nunique(),
                "rmse": math.sqrt(mean_squared_error(y, pred)),
                "mae": mean_absolute_error(y, pred),
                "bias": float(np.mean(pred - y)),
                "mean_observed": float(np.mean(y)),
            }
        )
    return pd.DataFrame(rows)


def add_fixed_ensembles(predictions: pd.DataFrame) -> pd.DataFrame:
    index = ["plot_id", "parent_plot_id", "subplot_id", "week", TARGET]
    wide = predictions.pivot_table(
        index=index, columns=["model", "feature_set"], values="y_pred"
    ).reset_index()
    members = [
        ("phenology_floor_ridge", "multiangular"),
        ("hurdle_probability_times_severity", "nadir"),
    ]
    if not all(member in wide.columns for member in members):
        return predictions
    ensemble = wide.loc[:, [(column, "") for column in index]].copy()
    ensemble.columns = index
    ensemble["model"] = "ensemble_ridge_multiangular_hurdle_nadir"
    ensemble["feature_set"] = "fixed_50_50"
    ensemble["y_pred"] = 0.5 * wide[members[0]] + 0.5 * wide[members[1]]
    return pd.concat([predictions, ensemble], ignore_index=True)


def prediction_result_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["model", "feature_set"]):
        rows.append(
            {
                "model": keys[0],
                "feature_set": keys[1],
                "n_train": math.nan,
                "n_test": len(group),
                "n_features": math.nan,
                **score(group[TARGET].to_numpy(float), group["y_pred"].to_numpy(float)),
            }
        )
    return pd.DataFrame(rows)


def plot_level_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["model", "feature_set"]):
        plot_predictions = group.groupby(["week", "parent_plot_id"], as_index=False).agg(
            y_true=(TARGET, "first"),
            y_pred=("y_pred", "mean"),
            n_subplots=("subplot_id", "nunique"),
        )
        rows.append(
            {
                "model": keys[0],
                "feature_set": keys[1],
                "n_parent_plot_weeks": len(plot_predictions),
                "mean_subplots_per_plot_week": plot_predictions["n_subplots"].mean(),
                **score(
                    plot_predictions["y_true"].to_numpy(float),
                    plot_predictions["y_pred"].to_numpy(float),
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    rows = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.3f}" if np.isfinite(value) else "")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main() -> None:
    total = time.perf_counter()
    for directory in [RESULTS_DIR, REPORTS_DIR, FEATURE_CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()
    disease_2024, disease_2025 = load_disease()

    feature_t0 = time.perf_counter()
    long_2024 = build_long_features(2024)
    long_2025 = build_long_features(2025)
    log_phase("build/load subplot long features", feature_t0)

    results = []
    preds = []
    for feature_set in ["nadir", "multiangular"]:
        train_features = pivot_features(long_2024, feature_set)
        test_features = pivot_features(long_2025, feature_set)
        train = model_table(train_features, disease_2024)
        test = model_table(test_features, disease_2025)
        logging.info("%s model table: train=%d test=%d", feature_set, len(train), len(test))
        for fit in [
            ridge_model,
            phenology_floor_ridge,
            hurdle_model,
            xgboost_model,
            xgboost_hurdle_model,
        ]:
            result, pred = fit(train, test, feature_set)
            results.append(result)
            preds.append(pred)

    predictions = pd.concat(preds, ignore_index=True)
    predictions = add_fixed_ensembles(predictions)
    model_rows_from_predictions = prediction_result_rows(predictions)
    recorded_results = pd.DataFrame(results)
    ensemble_rows = model_rows_from_predictions[
        model_rows_from_predictions["model"].eq("ensemble_ridge_multiangular_hurdle_nadir")
    ]
    results_df = pd.concat([recorded_results, ensemble_rows], ignore_index=True).sort_values(
        ["rmse", "model"]
    )
    weeks = week_summary(predictions).sort_values(["model", "feature_set", "week"])
    plot_level = plot_level_summary(predictions).sort_values(["rmse", "model"])
    results_df.to_csv(RESULTS_DIR / "subplot_current_severity_model_comparison.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "subplot_current_severity_predictions.csv", index=False)
    weeks.to_csv(RESULTS_DIR / "subplot_current_severity_week_summary.csv", index=False)
    plot_level.to_csv(
        RESULTS_DIR / "subplot_current_severity_plot_level_model_comparison.csv", index=False
    )

    report = [
        "## Results: Current Severity With 20 Subplots Per Plot",
        "",
        "Each plot-week parquet was split into a 5 x 4 spatial grid. The same plot-level `ds_plot` target was assigned to all subplots from that plot-week, so this is an exploratory pseudo-replication test rather than an independent biological replicate design.",
        "",
        "### Model Comparison",
        "",
        markdown_table(results_df.round(4)),
        "",
        "### Week Summary",
        "",
        markdown_table(weeks.round(4)),
        "",
        "### Plot-Level Summary",
        "",
        "The disease target is plot-level, so these rows average the 20 subplot predictions back to one prediction per parent plot-week before scoring.",
        "",
        markdown_table(plot_level.round(4)),
        "",
        "### Reproducibility",
        "",
        f"- Subplots per plot-week: {N_SUBPLOT_X * N_SUBPLOT_Y}",
        f"- Ground filter: OSAVI > {OSAVI_THRESHOLD}",
        f"- Max rows sampled per plot parquet: {MAX_ROWS_PER_PLOT}",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
        f"- model comparison: `{RESULTS_DIR / 'subplot_current_severity_model_comparison.csv'}`",
        f"- predictions: `{RESULTS_DIR / 'subplot_current_severity_predictions.csv'}`",
        f"- week summary: `{RESULTS_DIR / 'subplot_current_severity_week_summary.csv'}`",
        f"- plot-level model comparison: `{RESULTS_DIR / 'subplot_current_severity_plot_level_model_comparison.csv'}`",
    ]
    report_path = REPORTS_DIR / "subplot_current_severity_2024_to_2025_summary.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    logging.info("Report: %s", report_path)
    log_phase("total", total)


if __name__ == "__main__":
    main()
