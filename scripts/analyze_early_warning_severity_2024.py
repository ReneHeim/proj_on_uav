"""Early-warning disease severity prediction from nadir vs multiangular reflectance.

This script rebuilds the target from observed 2024 DSDI disease severity tables.
It does not use treatment or inoculation status as the disease label.
"""

from __future__ import annotations

import logging
import math
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DISEASE_ROOT = ROOT / "outputs/backup_metadata/csv/data/processed/2024/rating"
POLYGON_PATH = Path("/run/media/davidem/Heim/2024_oncerco_plot_polygons.gpkg")
VZA_FEATURES = (
    ROOT
    / "outputs/result_01_reflectance_distributions/2024/ground_filtered/results/plot_week_angle_features_2024.parquet"
)
RAA_FEATURES = (
    ROOT
    / "outputs/result_01_raa_sun_geometry/2024/ground_filtered/results/plot_week_vza_raa_features_2024.parquet"
)

OUTPUT_ROOT = ROOT / "outputs/early_warning_severity_2024"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
DISEASE_OUT_DIR = ROOT / "outputs/disease"
LOGS_DIR = ROOT / "outputs/logs"

SEED = 42
N_SPLITS = 5
ALPHAS = np.logspace(-3, 3, 25)
TARGET = "future_ds_plot"
TARGET_LOG = "future_ds_plot_log1p"
WARNING_TARGET = "future_warning_ds_ge_5"
WARNING_THRESHOLD = 5.0
MIN_NON_NULL_FRACTION = 0.50


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_early_warning_severity_2024_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Log file: %s", log_path)
    return log_path


def log_phase(name: str, t0: float) -> None:
    logging.info("[PHASE] %s: %.1fs", name, time.time() - t0)


def natural_plot_sort_key(plot_id: str) -> int:
    match = re.search(r"(\d+)$", str(plot_id))
    return int(match.group(1)) if match else 10**9


def clean_token(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def markdown_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float) or isinstance(value, np.floating):
                values.append("" if pd.isna(value) else f"{value:.{float_digits}f}")
            else:
                values.append("" if pd.isna(value) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def read_polygons() -> pd.DataFrame:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required to read the 2024 polygon file") from exc

    polygons = gpd.read_file(POLYGON_PATH).drop(columns="geometry")
    polygons = polygons.reset_index(drop=True)
    polygons["plot_id"] = [f"plot_{i}" for i in range(len(polygons))]
    polygons["ifz_id"] = polygons["ifz_id"].astype(int)
    polygons["cult"] = polygons["cult"].astype(str).str.lower()
    polygons["trt"] = polygons["trt"].astype(str).str.lower()
    return polygons[["plot_id", "ifz_id", "cult", "trt", "ino"]]


def load_disease_scores(plot_map: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    paths = sorted(DISEASE_ROOT.glob("week*/*/*.csv"))
    if not paths:
        raise FileNotFoundError(f"No 2024 DSDI CSV files found below {DISEASE_ROOT}")

    rows = []
    read_times = []
    for path in paths:
        match = re.search(r"week(\d+)", str(path))
        if not match:
            continue
        week = int(match.group(1))
        date_match = re.search(r"(20\d{6})_oncerco_dsdi", str(path.parent))
        if date_match:
            parsed_date = pd.to_datetime(date_match.group(1), format="%Y%m%d").date().isoformat()
        else:
            parsed_date = None
        t_read = time.time()
        frame = pd.read_csv(path)
        read_times.append(time.time() - t_read)
        ds_cols = [c for c in frame.columns if re.fullmatch(r"ds_leaf\d+", c)]
        di_cols = [c for c in frame.columns if re.fullmatch(r"di_leaf\d+", c)]
        frame["year"] = 2024
        frame["week"] = week
        if parsed_date is not None:
            frame["date"] = parsed_date
        frame["ifz_id"] = pd.to_numeric(frame["ifz_id"], errors="coerce").astype("Int64")
        frame["ds_plot"] = pd.to_numeric(frame["ds_plot"], errors="coerce")
        frame["ds_leaf_mean"] = frame[ds_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        frame["ds_leaf_sd"] = frame[ds_cols].apply(pd.to_numeric, errors="coerce").std(axis=1)
        frame["di_leaf_mean"] = frame[di_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        rows.append(
            frame[
                [
                    "year",
                    "week",
                    "date",
                    "ifz_id",
                    "cult",
                    "trt",
                    "block",
                    "ino",
                    "ds_plot",
                    "ds_leaf_mean",
                    "ds_leaf_sd",
                    "di_leaf_mean",
                ]
            ]
        )

    disease = pd.concat(rows, ignore_index=True)
    disease["ifz_id"] = disease["ifz_id"].astype(int)
    disease["cult"] = disease["cult"].astype(str).str.lower()
    disease["trt"] = disease["trt"].astype(str).str.lower()
    disease = disease.merge(plot_map[["plot_id", "ifz_id"]], on="ifz_id", how="left")
    disease = disease[
        [
            "year",
            "week",
            "date",
            "plot_id",
            "ifz_id",
            "cult",
            "trt",
            "block",
            "ino",
            "ds_plot",
            "ds_leaf_mean",
            "ds_leaf_sd",
            "di_leaf_mean",
        ]
    ].sort_values(
        ["week", "plot_id"],
        key=lambda col: col.map(natural_plot_sort_key) if col.name == "plot_id" else col,
    )

    logging.info(
        "[PHASE] parquet/csv read summary: min=%.3fs median=%.3fs mean=%.3fs max=%.3fs",
        np.min(read_times),
        np.median(read_times),
        np.mean(read_times),
        np.max(read_times),
    )
    log_phase("load disease scores", t0)
    return disease


def validate_plot_mapping(
    vza_long: pd.DataFrame, plot_map: pd.DataFrame, disease: pd.DataFrame
) -> pd.DataFrame:
    t0 = time.time()
    reflectance_meta = (
        vza_long[["plot_id", "cult", "trt"]]
        .drop_duplicates()
        .sort_values("plot_id", key=lambda s: s.map(natural_plot_sort_key))
    )
    validation = reflectance_meta.merge(
        plot_map, on="plot_id", suffixes=("_reflectance", "_polygon"), how="left"
    )
    validation["cult_match"] = (
        validation["cult_reflectance"].str.lower() == validation["cult_polygon"].str.lower()
    )
    validation["trt_match"] = (
        validation["trt_reflectance"].str.lower() == validation["trt_polygon"].str.lower()
    )

    disease_meta = disease[disease["week"] == disease["week"].min()][
        ["plot_id", "cult", "trt"]
    ].drop_duplicates()
    validation = validation.merge(disease_meta, on="plot_id", suffixes=("", "_disease"), how="left")
    validation["cult_disease_match"] = (
        validation["cult_reflectance"].str.lower() == validation["cult"].str.lower()
    )
    validation["trt_disease_match"] = (
        validation["trt_reflectance"].str.lower() == validation["trt"].str.lower()
    )
    validation = validation.rename(columns={"cult": "cult_disease", "trt": "trt_disease"})

    checks = ["cult_match", "trt_match", "cult_disease_match", "trt_disease_match"]
    if not validation[checks].all().all():
        failed = validation.loc[~validation[checks].all(axis=1)]
        raise RuntimeError(f"Plot mapping validation failed:\n{failed.to_string(index=False)}")
    log_phase("validate plot mapping", t0)
    return validation


def load_long_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    read_times = []
    t_read = time.time()
    vza = pl.read_parquet(VZA_FEATURES).to_pandas()
    read_times.append(time.time() - t_read)
    t_read = time.time()
    raa = pl.read_parquet(RAA_FEATURES).to_pandas()
    read_times.append(time.time() - t_read)
    logging.info(
        "[PHASE] parquet read summary: min=%.3fs median=%.3fs mean=%.3fs max=%.3fs",
        np.min(read_times),
        np.median(read_times),
        np.mean(read_times),
        np.max(read_times),
    )
    log_phase("load long reflectance features", t0)
    return vza, raa


def pivot_features(frame: pd.DataFrame, feature_set: str, category_cols: list[str]) -> pd.DataFrame:
    data = frame.copy()
    parts = [data["band_name"].map(clean_token)]
    for col in category_cols:
        parts.append(data[col].map(clean_token))
    data["feature"] = feature_set + "__" + parts[0]
    for part in parts[1:]:
        data["feature"] = data["feature"] + "__" + part
    pivot = data.pivot_table(
        index=["year", "week", "plot_id", "cult", "trt"],
        columns="feature",
        values="reflectance",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def build_feature_sets(vza: pd.DataFrame, raa: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t0 = time.time()
    vza = vza[vza["year"] == 2024].copy()
    raa = raa[raa["year"] == 2024].copy()

    # Nadir: choose the closest-to-nadir available VZA bin for each plot/week/band.
    nadir_idx = (
        vza.sort_values("vza_midpoint")
        .groupby(["year", "week", "plot_id", "band_name"], as_index=False)
        .head(1)
        .index
    )
    nadir = vza.loc[nadir_idx].copy()

    feature_sets = {
        "nadir": pivot_features(nadir, "nadir", []),
        "multiangular_vza": pivot_features(vza, "vza", ["vza_class"]),
        "multiangular_vza_raa": pivot_features(raa, "vza_raa", ["vza_class", "raa_class"]),
        "multiangular_vza_phase": pivot_features(raa, "vza_phase", ["vza_class", "phase_class"]),
    }

    for name, frame in feature_sets.items():
        logging.info("  %s: %s rows, %s feature columns", name, frame.shape[0], frame.shape[1] - 5)
    log_phase("build feature sets", t0)
    return feature_sets


def next_target_week(predictor_week: int, disease_weeks: list[int]) -> int | None:
    future = [week for week in disease_weeks if week > predictor_week]
    return min(future) if future else None


def build_model_table(features: pd.DataFrame, disease: pd.DataFrame) -> pd.DataFrame:
    rows = []
    disease_weeks = sorted(disease["week"].unique().tolist())
    disease_target = disease[["plot_id", "week", "ds_plot"]].rename(
        columns={"week": "target_week", "ds_plot": TARGET}
    )
    for predictor_week in sorted(features["week"].unique().tolist()):
        target_week = next_target_week(int(predictor_week), disease_weeks)
        if target_week is None:
            continue
        sub = features[features["week"] == predictor_week].copy()
        sub["predictor_week"] = predictor_week
        sub["target_week"] = target_week
        target = disease_target[disease_target["target_week"] == target_week]
        merged = sub.merge(target, on=["plot_id", "target_week"], how="inner")
        merged[TARGET_LOG] = np.log1p(merged[TARGET])
        merged[WARNING_TARGET] = (merged[TARGET] >= WARNING_THRESHOLD).astype(int)
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "year",
        "week",
        "plot_id",
        "cult",
        "trt",
        "predictor_week",
        "target_week",
        TARGET,
        TARGET_LOG,
        WARNING_TARGET,
    }
    return [c for c in frame.columns if c not in excluded]


def filter_sparse_columns(train: pd.DataFrame, cols: list[str]) -> list[str]:
    keep = []
    for col in cols:
        if train[col].notna().mean() >= MIN_NON_NULL_FRACTION:
            keep.append(col)
    return keep


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return math.nan
    value = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return float(value) if value is not None else math.nan


def adjusted_r2(r2: float, n: int, p: int) -> float:
    if not np.isfinite(r2) or n <= p + 1:
        return math.nan
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def evaluate_feature_set(name: str, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    cols = feature_columns(table)
    groups = table["plot_id"].to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(N_SPLITS, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)

    fold_rows = []
    prediction_rows = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(table, table[TARGET], groups=groups)
    ):
        train = table.iloc[train_idx].copy()
        test = table.iloc[test_idx].copy()
        fold_cols = filter_sparse_columns(train, cols)
        if not fold_cols:
            continue

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=ALPHAS)),
            ]
        )
        x_train = train[fold_cols]
        y_train = train[TARGET].to_numpy(float)
        x_test = test[fold_cols]
        y_test = test[TARGET].to_numpy(float)

        fit_t0 = time.time()
        pipeline.fit(x_train, y_train)
        fit_time = time.time() - fit_t0
        pred_t0 = time.time()
        pred = pipeline.predict(x_test)
        predict_time = time.time() - pred_t0

        rmse = math.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred) if len(np.unique(y_test)) > 1 else math.nan
        rho = safe_spearman(y_test, pred)
        alpha = float(pipeline.named_steps["ridge"].alpha_)
        fold_rows.append(
            {
                "feature_set": name,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "n_plots_train": train["plot_id"].nunique(),
                "n_plots_test": test["plot_id"].nunique(),
                "n_features": len(fold_cols),
                "alpha": alpha,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "adjusted_r2": adjusted_r2(r2, len(test), len(fold_cols)),
                "spearman": rho,
                "fit_time_s": fit_time,
                "predict_time_s": predict_time,
            }
        )
        prediction_rows.extend(
            {
                "feature_set": name,
                "fold": fold,
                "plot_id": row.plot_id,
                "predictor_week": int(row.predictor_week),
                "target_week": int(row.target_week),
                "observed_ds_plot": float(obs),
                "predicted_ds_plot": float(yhat),
            }
            for row, obs, yhat in zip(test.itertuples(index=False), y_test, pred)
        )
    log_phase(f"fit/predict {name}", t0)
    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def evaluate_log_severity_feature_set(
    name: str, table: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    cols = feature_columns(table)
    groups = table["plot_id"].to_numpy()
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)

    fold_rows = []
    prediction_rows = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(table, table[TARGET_LOG], groups=groups)
    ):
        train = table.iloc[train_idx].copy()
        test = table.iloc[test_idx].copy()
        fold_cols = filter_sparse_columns(train, cols)
        if not fold_cols:
            continue

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=ALPHAS)),
            ]
        )
        x_train = train[fold_cols]
        y_train_log = train[TARGET_LOG].to_numpy(float)
        x_test = test[fold_cols]
        y_test = test[TARGET].to_numpy(float)

        fit_t0 = time.time()
        pipeline.fit(x_train, y_train_log)
        fit_time = time.time() - fit_t0
        pred_t0 = time.time()
        pred_log = pipeline.predict(x_test)
        pred = np.expm1(pred_log)
        pred = np.clip(pred, 0, None)
        predict_time = time.time() - pred_t0

        rmse = math.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred) if len(np.unique(y_test)) > 1 else math.nan
        rho = safe_spearman(y_test, pred)
        fold_rows.append(
            {
                "feature_set": name,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "n_plots_train": train["plot_id"].nunique(),
                "n_plots_test": test["plot_id"].nunique(),
                "n_features": len(fold_cols),
                "alpha": float(pipeline.named_steps["ridge"].alpha_),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "adjusted_r2": adjusted_r2(r2, len(test), len(fold_cols)),
                "spearman": rho,
                "fit_time_s": fit_time,
                "predict_time_s": predict_time,
            }
        )
        prediction_rows.extend(
            {
                "feature_set": name,
                "fold": fold,
                "plot_id": row.plot_id,
                "predictor_week": int(row.predictor_week),
                "target_week": int(row.target_week),
                "observed_ds_plot": float(obs),
                "predicted_ds_plot": float(yhat),
            }
            for row, obs, yhat in zip(test.itertuples(index=False), y_test, pred)
        )
    log_phase(f"fit/predict log severity {name}", t0)
    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def evaluate_warning_feature_set(
    name: str, table: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()
    cols = feature_columns(table)
    groups = table["plot_id"].to_numpy()
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)

    fold_rows = []
    prediction_rows = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(table, table[WARNING_TARGET], groups=groups)
    ):
        train = table.iloc[train_idx].copy()
        test = table.iloc[test_idx].copy()
        fold_cols = filter_sparse_columns(train, cols)
        if not fold_cols:
            continue

        y_train = train[WARNING_TARGET].to_numpy(int)
        y_test = test[WARNING_TARGET].to_numpy(int)
        if len(np.unique(y_train)) < 2:
            logging.info("    %s warning fold %s skipped: train has one class", name, fold)
            continue

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegressionCV(
                        Cs=10,
                        cv=3,
                        class_weight="balanced",
                        max_iter=5000,
                        scoring="average_precision",
                        random_state=SEED,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        fit_t0 = time.time()
        pipeline.fit(train[fold_cols], y_train)
        fit_time = time.time() - fit_t0
        pred_t0 = time.time()
        prob = pipeline.predict_proba(test[fold_cols])[:, 1]
        pred = (prob >= 0.5).astype(int)
        predict_time = time.time() - pred_t0

        has_two_test_classes = len(np.unique(y_test)) > 1
        fold_rows.append(
            {
                "feature_set": name,
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "n_positive_test": int(y_test.sum()),
                "n_features": len(fold_cols),
                "auroc": roc_auc_score(y_test, prob) if has_two_test_classes else math.nan,
                "auprc": (
                    average_precision_score(y_test, prob) if has_two_test_classes else math.nan
                ),
                "f1": f1_score(y_test, pred, zero_division=0),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "balanced_accuracy": (
                    balanced_accuracy_score(y_test, pred) if has_two_test_classes else math.nan
                ),
                "fit_time_s": fit_time,
                "predict_time_s": predict_time,
            }
        )
        prediction_rows.extend(
            {
                "feature_set": name,
                "fold": fold,
                "plot_id": row.plot_id,
                "predictor_week": int(row.predictor_week),
                "target_week": int(row.target_week),
                "observed_warning": int(obs),
                "predicted_warning": int(yhat),
                "warning_probability": float(p),
                "observed_ds_plot": float(row.future_ds_plot),
            }
            for row, obs, yhat, p in zip(test.itertuples(index=False), y_test, pred, prob)
        )
    log_phase(f"fit/predict binary warning {name}", t0)
    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows)


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    return (
        folds.groupby("feature_set", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            sd_mae=("mae", "std"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            mean_spearman=("spearman", "mean"),
            sd_spearman=("spearman", "std"),
            mean_features=("n_features", "mean"),
        )
        .sort_values("mean_rmse")
    )


def summarize_warning_folds(folds: pd.DataFrame) -> pd.DataFrame:
    return (
        folds.groupby("feature_set", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_auroc=("auroc", "mean"),
            sd_auroc=("auroc", "std"),
            mean_auprc=("auprc", "mean"),
            sd_auprc=("auprc", "std"),
            mean_f1=("f1", "mean"),
            sd_f1=("f1", "std"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_features=("n_features", "mean"),
        )
        .sort_values("mean_f1", ascending=False)
    )


def paired_warning_deltas(folds: pd.DataFrame, baseline: str = "nadir") -> pd.DataFrame:
    metrics = ["auroc", "auprc", "f1", "precision", "recall", "balanced_accuracy"]
    base = folds[folds["feature_set"] == baseline][["fold"] + metrics].rename(
        columns={m: f"{baseline}_{m}" for m in metrics}
    )
    rows = []
    for feature_set in sorted(set(folds["feature_set"]) - {baseline}):
        comp = folds[folds["feature_set"] == feature_set][["fold"] + metrics].rename(
            columns={m: f"{feature_set}_{m}" for m in metrics}
        )
        joined = base.merge(comp, on="fold", how="inner")
        row = {"baseline": baseline, "comparator": feature_set, "paired_folds": len(joined)}
        for metric in metrics:
            delta = joined[f"{feature_set}_{metric}"] - joined[f"{baseline}_{metric}"]
            row[f"delta_{metric}_mean"] = delta.mean()
            row[f"delta_{metric}_sd"] = delta.std()
            row[f"folds_improved_{metric}"] = int((delta > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(folds: pd.DataFrame, baseline: str = "nadir") -> pd.DataFrame:
    metrics = ["rmse", "mae", "r2", "spearman"]
    base = folds[folds["feature_set"] == baseline][["fold"] + metrics].rename(
        columns={m: f"{baseline}_{m}" for m in metrics}
    )
    rows = []
    for feature_set in sorted(set(folds["feature_set"]) - {baseline}):
        comp = folds[folds["feature_set"] == feature_set][["fold"] + metrics].rename(
            columns={m: f"{feature_set}_{m}" for m in metrics}
        )
        joined = base.merge(comp, on="fold", how="inner")
        row = {"baseline": baseline, "comparator": feature_set, "paired_folds": len(joined)}
        for metric in metrics:
            delta = joined[f"{feature_set}_{metric}"] - joined[f"{baseline}_{metric}"]
            row[f"delta_{metric}_mean"] = delta.mean()
            row[f"delta_{metric}_sd"] = delta.std()
            row[f"folds_improved_{metric}"] = (
                int((delta < 0).sum()) if metric in {"rmse", "mae"} else int((delta > 0).sum())
            )
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_from_predictions(group: pd.DataFrame) -> pd.Series:
    y_true = group["observed_ds_plot"].to_numpy(float)
    y_pred = group["predicted_ds_plot"].to_numpy(float)
    r2 = r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else math.nan
    return pd.Series(
        {
            "n_predictions": len(group),
            "n_plots": group["plot_id"].nunique(),
            "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2,
            "spearman": safe_spearman(y_true, y_pred),
        }
    )


def summarize_by_week_pair(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["feature_set", "predictor_week", "target_week"], as_index=False)
        .apply(metrics_from_predictions, include_groups=False)
        .reset_index(drop=True)
        .sort_values(["predictor_week", "target_week", "feature_set"])
    )


def paired_deltas_by_week_pair(pair_summary: pd.DataFrame, baseline: str = "nadir") -> pd.DataFrame:
    metrics = ["rmse", "mae", "r2", "spearman"]
    base = pair_summary[pair_summary["feature_set"] == baseline][
        ["predictor_week", "target_week"] + metrics
    ].rename(columns={m: f"{baseline}_{m}" for m in metrics})
    rows = []
    for feature_set in sorted(set(pair_summary["feature_set"]) - {baseline}):
        comp = pair_summary[pair_summary["feature_set"] == feature_set][
            ["predictor_week", "target_week"] + metrics
        ].rename(columns={m: f"{feature_set}_{m}" for m in metrics})
        joined = base.merge(comp, on=["predictor_week", "target_week"], how="inner")
        for _, row in joined.iterrows():
            out = {
                "baseline": baseline,
                "comparator": feature_set,
                "predictor_week": int(row["predictor_week"]),
                "target_week": int(row["target_week"]),
            }
            for metric in metrics:
                out[f"delta_{metric}"] = (
                    row[f"{feature_set}_{metric}"] - row[f"{baseline}_{metric}"]
                )
            rows.append(out)
    return pd.DataFrame(rows).sort_values(["predictor_week", "target_week", "comparator"])


def coefficient_table(name: str, table: pd.DataFrame) -> pd.DataFrame:
    cols = filter_sparse_columns(table, feature_columns(table))
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    pipeline.fit(table[cols], table[TARGET].to_numpy(float))
    coef = pipeline.named_steps["ridge"].coef_
    return (
        pd.DataFrame({"feature_set": name, "feature": cols, "coefficient": coef})
        .assign(abs_coefficient=lambda df: df["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
    )


def plot_predictions(predictions: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, (feature_set, sub) in zip(axes, predictions.groupby("feature_set")):
        ax.scatter(sub["observed_ds_plot"], sub["predicted_ds_plot"], s=34, alpha=0.75)
        lim_min = min(sub["observed_ds_plot"].min(), sub["predicted_ds_plot"].min())
        lim_max = max(sub["observed_ds_plot"].max(), sub["predicted_ds_plot"].max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color="#444444", linewidth=1)
        ax.set_title(feature_set.replace("_", " "))
        ax.grid(alpha=0.25)
    for ax in axes:
        ax.set_xlabel("Observed future ds_plot")
        ax.set_ylabel("Predicted future ds_plot")
    fig.suptitle("Early-warning disease severity prediction, 2024", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("mean_rmse")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(
        ordered["feature_set"],
        ordered["mean_rmse"],
        xerr=ordered["sd_rmse"],
        color="#486B53",
        alpha=0.85,
    )
    ax.set_xlabel("Cross-validated RMSE, lower is better")
    ax.set_ylabel("")
    ax.set_title("Nadir vs multiangular severity prediction")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    log_summary: pd.DataFrame,
    log_deltas: pd.DataFrame,
    warning_summary: pd.DataFrame,
    warning_deltas: pd.DataFrame,
    pair_summary: pd.DataFrame,
    pair_deltas: pd.DataFrame,
    mapping_validation: pd.DataFrame,
    disease_summary: pd.DataFrame,
    outputs: dict[str, Path],
    log_path: Path,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best = summary.sort_values("mean_rmse").iloc[0]
    lines = [
        "## Results: Early-Warning Disease Severity Prediction, 2024",
        "",
        "The target is observed future plot-level disease severity (`ds_plot`) from DSDI tables. Treatment and inoculation are metadata only, not labels.",
        "",
        "### Model comparison",
        "",
        markdown_table(summary.round(4)),
        "",
        "### Paired deltas versus nadir",
        "",
        markdown_table(deltas.round(4)),
        "",
        "### Log-transformed severity model",
        "",
        markdown_table(log_summary.round(4)),
        "",
        "### Log-transformed severity deltas versus nadir",
        "",
        markdown_table(log_deltas.round(4)),
        "",
        f"### Binary warning model (`ds_plot >= {WARNING_THRESHOLD:g}`)",
        "",
        markdown_table(warning_summary.round(4)),
        "",
        "### Binary warning deltas versus nadir",
        "",
        markdown_table(warning_deltas.round(4)),
        "",
        "### Disease progression",
        "",
        markdown_table(disease_summary.round(4)),
        "",
        "### Week-pair performance",
        "",
        markdown_table(pair_summary.round(4)),
        "",
        "### Week-pair deltas versus nadir",
        "",
        markdown_table(pair_deltas.round(4)),
        "",
        f"**Interpretation**: The best feature set by mean RMSE was `{best['feature_set']}`. Multiangular value should be judged by the paired delta table: negative `delta_rmse_mean` or `delta_mae_mean` means improvement over nadir; positive `delta_r2_mean` or `delta_spearman_mean` means improvement over nadir.",
        "",
        "### Reproducibility",
        "",
        "- Year: `2024`",
        f"- Targets: numeric future `ds_plot`, log-transformed `log1p(ds_plot)`, binary warning `ds_plot >= {WARNING_THRESHOLD:g}`",
        "- Horizon: next available DSDI disease week",
        "- Ground filter: existing `ground_filtered` reflectance outputs, OSAVI threshold from upstream analysis",
        "- Model: `SimpleImputer(median) -> StandardScaler -> RidgeCV`",
        f"- Ridge alphas: `{ALPHAS.tolist()}`",
        f"- CV: `GroupKFold(n_splits={N_SPLITS})`, grouped by `plot_id`",
        f"- Random seed: `{SEED}`",
        f"- Log: `{log_path}`",
        "",
        "### Mapping validation",
        "",
        f"- Validated plot mappings: `{len(mapping_validation)}`",
        "- Mapping was checked against cultivar and treatment in reflectance, polygon, and DSDI metadata.",
        "",
        "### Outputs",
        "",
    ]
    for label, path in outputs.items():
        lines.append(f"- {label}: `{path}`")
    outputs["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DISEASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()

    plot_map = read_polygons()
    disease = load_disease_scores(plot_map)
    vza, raa = load_long_features()
    mapping_validation = validate_plot_mapping(vza, plot_map, disease)
    feature_sets = build_feature_sets(vza, raa)

    disease_summary = (
        disease.groupby("week", as_index=False)
        .agg(
            date=("date", "first"),
            plots=("plot_id", "nunique"),
            mean_ds_plot=("ds_plot", "mean"),
            max_ds_plot=("ds_plot", "max"),
            mean_ds_leaf=("ds_leaf_mean", "mean"),
            mean_di_leaf=("di_leaf_mean", "mean"),
        )
        .sort_values("week")
    )

    all_folds = []
    all_predictions = []
    all_coefficients = []
    all_log_folds = []
    all_log_predictions = []
    all_warning_folds = []
    all_warning_predictions = []
    for name, features in feature_sets.items():
        model_table = build_model_table(features, disease)
        model_table = model_table.sort_values(
            ["predictor_week", "plot_id"],
            key=lambda col: col.map(natural_plot_sort_key) if col.name == "plot_id" else col,
        )
        folds, predictions = evaluate_feature_set(name, model_table)
        log_folds, log_predictions = evaluate_log_severity_feature_set(name, model_table)
        warning_folds, warning_predictions = evaluate_warning_feature_set(name, model_table)
        all_folds.append(folds)
        all_predictions.append(predictions)
        all_log_folds.append(log_folds)
        all_log_predictions.append(log_predictions)
        all_warning_folds.append(warning_folds)
        all_warning_predictions.append(warning_predictions)
        all_coefficients.append(coefficient_table(name, model_table).head(50))

    folds = pd.concat(all_folds, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    log_folds = pd.concat(all_log_folds, ignore_index=True)
    log_predictions = pd.concat(all_log_predictions, ignore_index=True)
    warning_folds = pd.concat(all_warning_folds, ignore_index=True)
    warning_predictions = pd.concat(all_warning_predictions, ignore_index=True)
    coefficients = pd.concat(all_coefficients, ignore_index=True)
    summary = summarize_folds(folds)
    deltas = paired_deltas(folds)
    log_summary = summarize_folds(log_folds)
    log_deltas = paired_deltas(log_folds)
    warning_summary = summarize_warning_folds(warning_folds)
    warning_deltas = paired_warning_deltas(warning_folds)
    pair_summary = summarize_by_week_pair(predictions)
    pair_deltas = paired_deltas_by_week_pair(pair_summary)

    outputs = {
        "clean_disease": DISEASE_OUT_DIR / "clean_disease_scores_2024.csv",
        "mapping_validation": RESULTS_DIR / "plot_ifz_mapping_validation_2024.csv",
        "disease_progression": RESULTS_DIR / "disease_progression_2024.csv",
        "fold_metrics": RESULTS_DIR / "early_warning_severity_by_fold_2024.csv",
        "summary": RESULTS_DIR / "early_warning_severity_summary_2024.csv",
        "paired_deltas": RESULTS_DIR / "early_warning_severity_paired_deltas_2024.csv",
        "week_pair_summary": RESULTS_DIR / "early_warning_severity_by_week_pair_2024.csv",
        "week_pair_deltas": RESULTS_DIR / "early_warning_severity_week_pair_deltas_2024.csv",
        "log_fold_metrics": RESULTS_DIR / "early_warning_log_severity_by_fold_2024.csv",
        "log_summary": RESULTS_DIR / "early_warning_log_severity_summary_2024.csv",
        "log_paired_deltas": RESULTS_DIR / "early_warning_log_severity_paired_deltas_2024.csv",
        "log_predictions": RESULTS_DIR / "early_warning_log_severity_predictions_2024.csv",
        "warning_fold_metrics": RESULTS_DIR / "early_warning_binary_warning_by_fold_2024.csv",
        "warning_summary": RESULTS_DIR / "early_warning_binary_warning_summary_2024.csv",
        "warning_paired_deltas": RESULTS_DIR
        / "early_warning_binary_warning_paired_deltas_2024.csv",
        "warning_predictions": RESULTS_DIR / "early_warning_binary_warning_predictions_2024.csv",
        "predictions": RESULTS_DIR / "early_warning_severity_predictions_2024.csv",
        "coefficients": RESULTS_DIR / "early_warning_severity_top_coefficients_2024.csv",
        "prediction_plot": FIGURES_DIR / "early_warning_observed_vs_predicted_2024.png",
        "summary_plot": FIGURES_DIR / "early_warning_feature_set_rmse_2024.png",
        "report": REPORTS_DIR / "early_warning_severity_2024_summary.md",
    }

    disease.to_csv(outputs["clean_disease"], index=False)
    mapping_validation.to_csv(outputs["mapping_validation"], index=False)
    disease_summary.to_csv(outputs["disease_progression"], index=False)
    folds.to_csv(outputs["fold_metrics"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    deltas.to_csv(outputs["paired_deltas"], index=False)
    pair_summary.to_csv(outputs["week_pair_summary"], index=False)
    pair_deltas.to_csv(outputs["week_pair_deltas"], index=False)
    log_folds.to_csv(outputs["log_fold_metrics"], index=False)
    log_summary.to_csv(outputs["log_summary"], index=False)
    log_deltas.to_csv(outputs["log_paired_deltas"], index=False)
    log_predictions.to_csv(outputs["log_predictions"], index=False)
    warning_folds.to_csv(outputs["warning_fold_metrics"], index=False)
    warning_summary.to_csv(outputs["warning_summary"], index=False)
    warning_deltas.to_csv(outputs["warning_paired_deltas"], index=False)
    warning_predictions.to_csv(outputs["warning_predictions"], index=False)
    predictions.to_csv(outputs["predictions"], index=False)
    coefficients.to_csv(outputs["coefficients"], index=False)
    plot_predictions(predictions, outputs["prediction_plot"])
    plot_summary(summary, outputs["summary_plot"])
    write_report(
        summary,
        deltas,
        log_summary,
        log_deltas,
        warning_summary,
        warning_deltas,
        pair_summary,
        pair_deltas,
        mapping_validation,
        disease_summary,
        outputs,
        log_path,
    )
    log_phase("total", start)


if __name__ == "__main__":
    main()
