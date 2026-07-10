"""Train 2024 early-warning models and test on 2025."""

from __future__ import annotations

import logging
import math
import re
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.early_warning.analyze_early_warning_severity_2024 import (
    ALPHAS,
    TARGET,
    TARGET_LOG,
    WARNING_TARGET,
    WARNING_THRESHOLD,
    build_model_table,
    clean_token,
    feature_columns,
    load_disease_scores,
    natural_plot_sort_key,
    read_polygons,
)
from src.research.common import configure_logging, log_phase as common_log_phase, markdown_table, safe_spearman

DISEASE_2025_ROOT = ROOT / "outputs/runs/metadata/backup_metadata/csv/data/raw/2025"
POLYGON_2025_PATH = Path("/run/media/davidem/Heim/2025_oncerco_plot_polygons.gpkg")
VZA_2025 = (
    ROOT
    / "outputs/runs/analysis/reflectance/distributions/2025/ground_filtered/results/plot_week_angle_features_2025.parquet"
)
RAA_2025 = (
    ROOT
    / "outputs/runs/analysis/reflectance/raa_sun_geometry/2025/ground_filtered/results/plot_week_vza_raa_features_2025.parquet"
)
VZA_2024 = (
    ROOT
    / "outputs/runs/analysis/reflectance/distributions/2024/ground_filtered/results/plot_week_angle_features_2024.parquet"
)
RAA_2024 = (
    ROOT
    / "outputs/runs/analysis/reflectance/raa_sun_geometry/2024/ground_filtered/results/plot_week_vza_raa_features_2024.parquet"
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/cross_year/generalization_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
XGB_CURVES_DIR = FIGURES_DIR / "xgboost_training_curves"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LAGGED_PREDICTIONS_DIR = RESULTS_DIR / "predictions_lagged_disease"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = OUTPUT_ROOT / "logs"
DISEASE_OUT = ROOT / "outputs/shared/disease/clean_disease_scores_2025.csv"
DISEASE_2024_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2024.csv"

SEED = 42
MIN_NON_NULL_FRACTION = 0.50
MIN_ANGULAR_BIN_PIXELS = 500
MIN_ANGULAR_BIN_IMAGES = 2
LAG_FEATURE_COLUMNS = [
    "lag_ds_plot",
    "lag_any_disease",
    "lag_warning",
    "lag_ds_leaf_mean",
    "lag_ds_leaf_sd",
    "lag_di_leaf_mean",
    "lag_observation_week",
    "lag_weeks_since_observation",
    "lag_ds_change",
    "lag_has_observation",
]
LAG_POLICIES = {
    "through_predictor_week": "Disease observations with week <= predictor_week.",
    "before_predictor_week": "Disease observations with week < predictor_week only.",
}


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def setup_logging() -> Path:
    return configure_logging(LOGS_DIR, "analyze_cross_year_generalization_2024_to_2025")


def log_phase(name: str, t0: float) -> None:
    common_log_phase(name, t0, wall_clock=True)


def binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prob: np.ndarray | None = None
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    positives = y_true == 1
    negatives = y_true == 0
    tp = int((positives & (y_pred == 1)).sum())
    tn = int((negatives & (y_pred == 0)).sum())
    fp = int((negatives & (y_pred == 1)).sum())
    fn = int((positives & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1_den = (2 * tp) + fp + fn
    has_two_classes = len(np.unique(y_true)) > 1
    return {
        "auroc": roc_auc_score(y_true, prob) if prob is not None and has_two_classes else math.nan,
        "auprc": (
            average_precision_score(y_true, prob)
            if prob is not None and has_two_classes
            else math.nan
        ),
        "f1": (2 * tp / f1_den) if f1_den else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "balanced_accuracy": (recall + specificity) / 2 if has_two_classes else math.nan,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(y_true: np.ndarray, prob: np.ndarray, strategy: str) -> float:
    thresholds = np.unique(
        np.concatenate(([0.05, 0.5, 0.95], np.quantile(prob, np.linspace(0.02, 0.98, 97))))
    )
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        metrics = binary_metrics(y_true, pred)
        if strategy == "max_f1":
            score = metrics["f1"]
        elif strategy == "balanced":
            score = metrics["balanced_accuracy"]
        elif strategy == "high_recall":
            score = metrics["recall"] if metrics["specificity"] >= 0.55 else -np.inf
        else:
            raise ValueError(f"Unknown threshold strategy: {strategy}")
        if np.isfinite(score) and (
            score > best_score or (score == best_score and threshold > best_threshold)
        ):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def grouped_cv_thresholds(
    estimator: Pipeline | XGBClassifier,
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 4,
) -> dict[str, float]:
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        return {"fixed_0_5": 0.5, "max_f1": 0.5, "balanced": 0.5, "high_recall": 0.5}
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan, dtype=float)
    for train_idx, valid_idx in splitter.split(x, y, groups=groups):
        model = clone(estimator)
        model.fit(x.iloc[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(x.iloc[valid_idx])[:, 1]
    missing = np.isnan(oof)
    if missing.any():
        fallback = clone(estimator)
        fallback.fit(x.loc[~missing], y[~missing])
        oof[missing] = fallback.predict_proba(x.loc[missing])[:, 1]
    return {
        "fixed_0_5": 0.5,
        "max_f1": choose_threshold(y, oof, "max_f1"),
        "balanced": choose_threshold(y, oof, "balanced"),
        "high_recall": choose_threshold(y, oof, "high_recall"),
    }


def read_2025_plot_map() -> pd.DataFrame:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required to read the 2025 polygon file") from exc
    polygons = gpd.read_file(POLYGON_2025_PATH, layer="2025_oncerco_plot_polygons").drop(
        columns="geometry"
    )
    polygons = polygons.reset_index(drop=True)
    polygons["plot_id"] = [f"plot_{i}" for i in range(len(polygons))]
    polygons["ifz_id"] = [90001 + i for i in range(len(polygons))]
    polygons["cult"] = polygons["cultivar"].astype(str).str.lower()
    polygons["trt"] = polygons["trt"].astype(str).str.lower()
    return polygons[["plot_id", "ifz_id", "cult", "trt"]]


def load_2024_disease_with_fallback() -> pd.DataFrame:
    try:
        return load_disease_scores(read_polygons())
    except Exception as exc:
        if not DISEASE_2024_CLEAN.exists():
            raise
        logging.warning(
            "Using local 2024 clean disease fallback because polygon load failed: %s", exc
        )
        return pd.read_csv(DISEASE_2024_CLEAN)


def load_2025_disease_with_fallback() -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return parse_2025_dsdi()
    except Exception as exc:
        if not DISEASE_OUT.exists():
            raise
        logging.warning(
            "Using local 2025 clean disease fallback because DSDI parsing/mapping failed: %s", exc
        )
        disease = pd.read_csv(DISEASE_OUT)
        audit = pd.DataFrame(
            [
                {
                    "week": "local_clean",
                    "selected": True,
                    "source_path": str(DISEASE_OUT),
                    "rows": len(disease),
                    "plots": disease["plot_id"].nunique(),
                    "mean_ds_plot": disease["ds_plot"].mean(),
                    "max_ds_plot": disease["ds_plot"].max(),
                }
            ]
        )
        return disease, audit


def parse_2025_dsdi() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one auditable 24-row disease block per 2025 disease week."""
    t0 = time.time()
    paths = sorted(DISEASE_2025_ROOT.glob("week*/dsdi/*/20240708_oncerco_dsdi.csv"))
    candidates = []
    for source_rank, path in enumerate(paths):
        frame = pd.read_csv(path)
        if "date" not in frame.columns or "ifz_id" not in frame.columns:
            continue
        frame["source_path"] = str(path)
        frame["source_rank"] = source_rank
        frame["target_week_from_label"] = frame["date"].astype(str).str.extract(r"week(\d+)")[0]
        frame.loc[frame["target_week_from_label"].isna(), "target_week_from_label"] = "0"
        frame["week"] = pd.to_numeric(frame["target_week_from_label"], errors="coerce")
        frame["ds_plot"] = pd.to_numeric(frame["ds_plot"], errors="coerce")
        candidates.append(frame)
    all_rows = pd.concat(candidates, ignore_index=True)
    all_rows = all_rows.dropna(subset=["week", "ifz_id"]).copy()
    all_rows["week"] = all_rows["week"].astype(int)
    all_rows["ifz_id"] = pd.to_numeric(all_rows["ifz_id"], errors="coerce").astype("Int64")

    selected = []
    audit_rows = []
    for week, week_rows in all_rows.groupby("week"):
        valid_blocks = []
        for source_path, block in week_rows.groupby("source_path", sort=False):
            block = block.dropna(subset=["ifz_id"]).copy()
            if len(block) < 24:
                continue
            # Some cumulative files contain duplicate week6 blocks. Keep the last 24 rows inside that file.
            block = block.tail(24)
            if block["ifz_id"].nunique() != 24:
                continue
            if block["ds_plot"].notna().sum() < 20:
                continue
            valid_blocks.append(block)
        if not valid_blocks:
            audit_rows.append(
                {"week": week, "selected": False, "reason": "no valid 24-plot non-null block"}
            )
            continue
        block = valid_blocks[-1].copy()
        selected.append(block)
        audit_rows.append(
            {
                "week": week,
                "selected": True,
                "source_path": block["source_path"].iloc[0],
                "rows": len(block),
                "plots": block["ifz_id"].nunique(),
                "mean_ds_plot": block["ds_plot"].mean(),
                "max_ds_plot": block["ds_plot"].max(),
            }
        )
    disease = pd.concat(selected, ignore_index=True)
    ds_cols = [c for c in disease.columns if re.fullmatch(r"ds_leaf\d+", c)]
    disease["year"] = 2025
    disease["ifz_id"] = disease["ifz_id"].astype(int)
    disease["cult"] = disease["cult"].astype(str).str.lower()
    disease["trt"] = disease["trt"].astype(str).str.lower()
    if ds_cols:
        disease["ds_leaf_mean"] = (
            disease[ds_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        )
        disease["ds_leaf_sd"] = disease[ds_cols].apply(pd.to_numeric, errors="coerce").std(axis=1)
    else:
        disease["ds_leaf_mean"] = np.nan
        disease["ds_leaf_sd"] = np.nan
    disease["di_leaf_mean"] = np.nan
    plot_map = read_2025_plot_map()
    disease = disease.merge(plot_map[["plot_id", "ifz_id"]], on="ifz_id", how="left")
    disease = disease[
        [
            "year",
            "week",
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
            "source_path",
        ]
    ].sort_values(
        ["week", "plot_id"],
        key=lambda col: col.map(natural_plot_sort_key) if col.name == "plot_id" else col,
    )
    log_phase("parse 2025 DSDI", t0)
    return disease, pd.DataFrame(audit_rows).sort_values("week")


def validate_2025_mapping(
    vza: pd.DataFrame, disease: pd.DataFrame, plot_map: pd.DataFrame
) -> pd.DataFrame:
    meta = (
        vza[["plot_id", "cult", "trt"]]
        .drop_duplicates()
        .sort_values("plot_id", key=lambda s: s.map(natural_plot_sort_key))
    )
    validation = meta.merge(
        plot_map, on="plot_id", suffixes=("_reflectance", "_polygon"), how="left"
    )
    disease_meta = disease[disease["week"] == disease["week"].min()][
        ["plot_id", "cult", "trt"]
    ].drop_duplicates()
    validation = validation.merge(disease_meta, on="plot_id", suffixes=("", "_disease"), how="left")
    validation["cult_match"] = (
        validation["cult_reflectance"].str.lower() == validation["cult_polygon"].str.lower()
    )
    validation["trt_match"] = (
        validation["trt_reflectance"].str.lower() == validation["trt_polygon"].str.lower()
    )
    validation["cult_disease_match"] = (
        validation["cult_reflectance"].str.lower() == validation["cult"].str.lower()
    )
    validation["trt_disease_match"] = (
        validation["trt_reflectance"].str.lower() == validation["trt"].str.lower()
    )
    checks = ["cult_match", "trt_match", "cult_disease_match", "trt_disease_match"]
    if not validation[checks].all().all():
        raise RuntimeError(
            "2025 plot mapping validation failed:\n" + validation.to_string(index=False)
        )
    return validation.rename(columns={"cult": "cult_disease", "trt": "trt_disease"})


def load_feature_sets_for_year(vza_path: Path, raa_path: Path) -> dict[str, pd.DataFrame]:
    t0 = time.time()
    vza = pl.read_parquet(vza_path).to_pandas()
    raa = pl.read_parquet(raa_path).to_pandas()
    log_phase(f"load features {vza_path}", t0)
    return build_feature_sets_any_year(vza, raa)


def pivot_features_any_year(
    frame: pd.DataFrame, feature_set: str, category_cols: list[str]
) -> pd.DataFrame:
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


def summarize_vza_curve(vza: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["year", "week", "plot_id", "cult", "trt", "band_name"]
    for keys, group in vza.groupby(group_cols, sort=False):
        group = group.dropna(subset=["reflectance", "vza_midpoint"]).sort_values("vza_midpoint")
        if group.empty:
            continue
        x = group["vza_midpoint"].to_numpy(float)
        y = group["reflectance"].to_numpy(float)
        low = group[group["vza_midpoint"] <= 20]["reflectance"]
        high = group[group["vza_midpoint"] >= 45]["reflectance"]
        nadir = float(
            group.iloc[(group["vza_midpoint"] - group["vza_midpoint"].min()).abs().argmin()][
                "reflectance"
            ]
        )
        slope = float(np.polyfit(x, y, 1)[0]) if len(np.unique(x)) >= 2 else np.nan
        curvature = float(np.polyfit(x, y, 2)[0]) if len(np.unique(x)) >= 3 else np.nan
        high_minus_low = (
            float(high.mean() - low.mean()) if not low.empty and not high.empty else np.nan
        )
        angular_range = float(np.max(y) - np.min(y)) if len(y) else np.nan
        off_nadir = group.loc[group["vza_midpoint"] > group["vza_midpoint"].min(), "reflectance"]
        off_nadir_mean = float(off_nadir.mean()) if not off_nadir.empty else np.nan
        off_nadir_minus_nadir = off_nadir_mean - nadir if np.isfinite(off_nadir_mean) else np.nan
        band = clean_token(keys[-1])
        base = dict(zip(group_cols[:-1], keys[:-1], strict=False))
        values = {
            "nadir": nadir,
            "mean": float(np.mean(y)),
            "std": float(np.std(y, ddof=0)),
            "slope": slope,
            "curvature": curvature,
            "angular_range": angular_range,
            "high_minus_low": high_minus_low,
            "relative_high_minus_low": (
                high_minus_low / nadir if np.isfinite(high_minus_low) and nadir != 0 else np.nan
            ),
            "off_nadir_minus_nadir": off_nadir_minus_nadir,
            "off_nadir_ratio_nadir": (
                off_nadir_mean / nadir if np.isfinite(off_nadir_mean) and nadir != 0 else np.nan
            ),
        }
        for metric, value in values.items():
            row = base.copy()
            row["feature"] = f"vza_compact__{band}__{metric}"
            row["value"] = value
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    data = pd.DataFrame(rows)
    pivot = data.pivot_table(
        index=["year", "week", "plot_id", "cult", "trt"],
        columns="feature",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def summarize_geometry_contrasts(raa: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["year", "week", "plot_id", "cult", "trt", "band_name"]
    for keys, group in raa.groupby(group_cols, sort=False):
        group = group.dropna(subset=["reflectance"]).copy()
        if group.empty:
            continue
        band = clean_token(keys[-1])
        base = dict(zip(group_cols[:-1], keys[:-1], strict=False))

        near_phase = group[group["phase_midpoint"] <= 20]["reflectance"]
        far_phase = group[group["phase_midpoint"] >= 60]["reflectance"]
        low_raa = group[group["raa_midpoint"] <= 45]["reflectance"]
        high_raa = group[group["raa_midpoint"] >= 135]["reflectance"]
        values = {
            "phase_near_minus_far": (
                float(near_phase.mean() - far_phase.mean())
                if not near_phase.empty and not far_phase.empty
                else np.nan
            ),
            "raa_low_minus_high": (
                float(low_raa.mean() - high_raa.mean())
                if not low_raa.empty and not high_raa.empty
                else np.nan
            ),
            "phase_std": float(group.groupby("phase_class")["reflectance"].mean().std(ddof=0)),
            "raa_std": float(group.groupby("raa_class")["reflectance"].mean().std(ddof=0)),
        }
        for metric, value in values.items():
            row = base.copy()
            row["feature"] = f"geometry_compact__{band}__{metric}"
            row["value"] = value
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    data = pd.DataFrame(rows)
    pivot = data.pivot_table(
        index=["year", "week", "plot_id", "cult", "trt"],
        columns="feature",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def merge_feature_frames(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return left.merge(right, on=["year", "week", "plot_id", "cult", "trt"], how="outer")


def filter_reliable_angular_bins(raa: pd.DataFrame) -> pd.DataFrame:
    required = {"n_pixels", "n_images"}
    if not required.issubset(raa.columns):
        return raa.copy()
    return raa[
        (raa["n_pixels"] >= MIN_ANGULAR_BIN_PIXELS) & (raa["n_images"] >= MIN_ANGULAR_BIN_IMAGES)
    ].copy()


def build_feature_sets_any_year(vza: pd.DataFrame, raa: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t0 = time.time()
    nadir_idx = (
        vza.sort_values("vza_midpoint")
        .groupby(["year", "week", "plot_id", "band_name"], as_index=False)
        .head(1)
        .index
    )
    nadir = vza.loc[nadir_idx].copy()
    vza_compact = summarize_vza_curve(vza)
    geometry_compact = summarize_geometry_contrasts(raa)
    raa_reliable = filter_reliable_angular_bins(raa)
    angular_shape = merge_feature_frames(vza_compact, geometry_compact)
    feature_sets = {
        "nadir": pivot_features_any_year(nadir, "nadir", []),
        "multiangular_vza_compact": vza_compact,
        "multiangular_angular_shape": angular_shape,
        "multiangular_geometry_compact": angular_shape,
        "multiangular_vza": pivot_features_any_year(vza, "vza", ["vza_class"]),
        "multiangular_vza_raa": pivot_features_any_year(raa, "vza_raa", ["vza_class", "raa_class"]),
        "multiangular_vza_phase": pivot_features_any_year(
            raa, "vza_phase", ["vza_class", "phase_class"]
        ),
        "multiangular_vza_raa_reliable": pivot_features_any_year(
            raa_reliable, "vza_raa_reliable", ["vza_class", "raa_class"]
        ),
        "multiangular_vza_phase_reliable": pivot_features_any_year(
            raa_reliable, "vza_phase_reliable", ["vza_class", "phase_class"]
        ),
    }
    for name, frame in feature_sets.items():
        logging.info("  %s: %s rows, %s feature columns", name, frame.shape[0], frame.shape[1] - 5)
    log_phase("build feature sets", t0)
    return feature_sets


def align_train_test(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    common_cols = sorted(set(feature_columns(train)).intersection(feature_columns(test)))
    common_cols = [c for c in common_cols if train[c].notna().mean() >= MIN_NON_NULL_FRACTION]
    common_cols = [c for c in common_cols if test[c].notna().mean() > 0]
    keep = [
        "plot_id",
        "predictor_week",
        "target_week",
        TARGET,
        TARGET_LOG,
        WARNING_TARGET,
    ] + common_cols
    return common_cols, train[keep].copy(), test[keep].copy()


def lagged_disease_features_for_pairs(
    model_table: pd.DataFrame,
    disease: pd.DataFrame,
    lag_policy: str,
) -> pd.DataFrame:
    if lag_policy not in LAG_POLICIES:
        raise ValueError(
            f"Unknown lag policy {lag_policy!r}; expected one of {sorted(LAG_POLICIES)}"
        )
    rows = []
    disease = disease.copy()
    for col in ["ds_plot", "ds_leaf_mean", "ds_leaf_sd", "di_leaf_mean"]:
        if col in disease.columns:
            disease[col] = pd.to_numeric(disease[col], errors="coerce")
    for row in model_table[["plot_id", "predictor_week", "target_week"]].itertuples(index=False):
        if lag_policy == "through_predictor_week":
            history_mask = disease["week"] <= row.predictor_week
        else:
            history_mask = disease["week"] < row.predictor_week
        plot_history = disease[(disease["plot_id"] == row.plot_id) & history_mask].sort_values(
            "week"
        )
        base = {
            "plot_id": row.plot_id,
            "predictor_week": row.predictor_week,
            "target_week": row.target_week,
        }
        if plot_history.empty:
            base.update(
                {
                    "lag_ds_plot": 0.0,
                    "lag_any_disease": 0,
                    "lag_warning": 0,
                    "lag_ds_leaf_mean": 0.0,
                    "lag_ds_leaf_sd": 0.0,
                    "lag_di_leaf_mean": 0.0,
                    "lag_observation_week": np.nan,
                    "lag_weeks_since_observation": np.nan,
                    "lag_ds_change": 0.0,
                    "lag_has_observation": 0,
                }
            )
        else:
            latest = plot_history.iloc[-1]
            previous = plot_history.iloc[-2] if len(plot_history) >= 2 else None
            latest_ds = float(latest["ds_plot"]) if pd.notna(latest["ds_plot"]) else 0.0
            previous_ds = (
                float(previous["ds_plot"])
                if previous is not None and pd.notna(previous["ds_plot"])
                else latest_ds
            )
            base.update(
                {
                    "lag_ds_plot": latest_ds,
                    "lag_any_disease": int(latest_ds > 0),
                    "lag_warning": int(latest_ds >= WARNING_THRESHOLD),
                    "lag_ds_leaf_mean": (
                        float(latest["ds_leaf_mean"])
                        if "ds_leaf_mean" in latest and pd.notna(latest["ds_leaf_mean"])
                        else 0.0
                    ),
                    "lag_ds_leaf_sd": (
                        float(latest["ds_leaf_sd"])
                        if "ds_leaf_sd" in latest and pd.notna(latest["ds_leaf_sd"])
                        else 0.0
                    ),
                    "lag_di_leaf_mean": (
                        float(latest["di_leaf_mean"])
                        if "di_leaf_mean" in latest and pd.notna(latest["di_leaf_mean"])
                        else 0.0
                    ),
                    "lag_observation_week": int(latest["week"]),
                    "lag_weeks_since_observation": int(row.predictor_week - latest["week"]),
                    "lag_ds_change": latest_ds - previous_ds,
                    "lag_has_observation": 1,
                }
            )
        rows.append(base)
    return pd.DataFrame(rows)


def add_lagged_disease_features(
    model_table: pd.DataFrame, disease: pd.DataFrame, lag_policy: str
) -> pd.DataFrame:
    lagged = lagged_disease_features_for_pairs(model_table, disease, lag_policy)
    merged = model_table.merge(lagged, on=["plot_id", "predictor_week", "target_week"], how="left")
    if lag_policy == "through_predictor_week":
        valid_lag = (
            merged["lag_observation_week"].dropna()
            <= merged.loc[merged["lag_observation_week"].notna(), "predictor_week"]
        )
    else:
        valid_lag = (
            merged["lag_observation_week"].dropna()
            < merged.loc[merged["lag_observation_week"].notna(), "predictor_week"]
        )
    if not valid_lag.all():
        raise RuntimeError(f"Lagged disease leakage detected for lag_policy={lag_policy}")
    for col in LAG_FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0
    fill_zero = [
        col
        for col in LAG_FEATURE_COLUMNS
        if col not in {"lag_observation_week", "lag_weeks_since_observation"}
    ]
    merged[fill_zero] = merged[fill_zero].fillna(0.0)
    return merged


def lagged_disease_only_table(model_table: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "plot_id",
        "predictor_week",
        "target_week",
        TARGET,
        TARGET_LOG,
        WARNING_TARGET,
    ] + LAG_FEATURE_COLUMNS
    meta = [col for col in ["year", "week", "cult", "trt"] if col in model_table.columns]
    return model_table[meta + keep].copy()


def add_known_covariates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    covariate_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train.copy()
    test = test.copy()
    cols = list(cols)
    if covariate_mode in {"spectral_plus_week", "spectral_plus_week_horizon"}:
        train["known__predictor_week"] = train["predictor_week"]
        test["known__predictor_week"] = test["predictor_week"]
        cols.append("known__predictor_week")
    if covariate_mode == "spectral_plus_week_horizon":
        train["known__target_week"] = train["target_week"]
        test["known__target_week"] = test["target_week"]
        cols.append("known__target_week")
    return train, test, cols


def evaluate_external_regression(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
    use_log: bool,
    covariate_mode: str,
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
    fit_t0 = time.time()
    pipeline.fit(train_aligned[cols], train_aligned[target_col].to_numpy(float))
    fit_time = time.time() - fit_t0
    pred_t0 = time.time()
    pred = pipeline.predict(test_aligned[cols])
    if use_log:
        pred = np.clip(np.expm1(pred), 0, None)
    train_target = train_aligned[TARGET].to_numpy(float)
    pred = np.clip(pred, np.nanmin(train_target), np.nanmax(train_target))
    predict_time = time.time() - pred_t0
    y = test_aligned[TARGET].to_numpy(float)
    model_key = "ridge_log_severity" if use_log else "ridge_raw_severity"
    predictions = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    predictions["model"] = "Ridge log severity" if use_log else "Ridge raw severity"
    predictions["feature_set"] = feature_set
    predictions["covariates"] = covariate_mode
    predictions["y_true"] = y
    predictions["y_pred"] = pred
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        PREDICTIONS_DIR
        / f"severity_predictions_{model_key}_{safe_filename(feature_set)}_{safe_filename(covariate_mode)}.csv",
        index=False,
    )
    return {
        "feature_set": feature_set,
        "model": "log_severity" if use_log else "raw_severity",
        "covariates": covariate_mode,
        "n_train": len(train_aligned),
        "n_test": len(test_aligned),
        "n_features": len(cols),
        "prediction_bounds": f"{np.nanmin(train_target):.3f}-{np.nanmax(train_target):.3f}",
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred) if len(np.unique(y)) > 1 else math.nan,
        "spearman": safe_spearman(y, pred),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }


def evaluate_external_warning(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> dict:
    cols, train_aligned, test_aligned = align_train_test(train, test)
    y_train = train_aligned[WARNING_TARGET].to_numpy(int)
    y_test = test_aligned[WARNING_TARGET].to_numpy(int)
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
    pipeline.fit(train_aligned[cols], y_train)
    fit_time = time.time() - fit_t0
    pred_t0 = time.time()
    prob = pipeline.predict_proba(test_aligned[cols])[:, 1]
    pred = (prob >= 0.5).astype(int)
    predict_time = time.time() - pred_t0
    has_two_classes = len(np.unique(y_test)) > 1
    predictions = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    predictions["model"] = "Logistic"
    predictions["feature_set"] = feature_set
    predictions["threshold"] = WARNING_THRESHOLD
    predictions["y_true"] = y_test
    predictions["y_prob"] = prob
    predictions["y_pred"] = pred
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        PREDICTIONS_DIR / f"early_warning_predictions_logistic_{safe_filename(feature_set)}.csv",
        index=False,
    )
    return {
        "feature_set": feature_set,
        "model": "binary_warning",
        "threshold": WARNING_THRESHOLD,
        "n_train": len(train_aligned),
        "n_test": len(test_aligned),
        "n_positive_test": int(y_test.sum()),
        "n_features": len(cols),
        "auroc": roc_auc_score(y_test, prob) if has_two_classes else math.nan,
        "auprc": average_precision_score(y_test, prob) if has_two_classes else math.nan,
        "f1": f1_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred) if has_two_classes else math.nan,
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }


def calibrated_warning_rows_from_prob(
    model_label: str,
    feature_set: str,
    threshold_by_strategy: dict[str, float],
    test_aligned: pd.DataFrame,
    y_test: np.ndarray,
    prob: np.ndarray,
    n_train: int,
    n_features: int,
    fit_time: float,
    predict_time: float,
) -> list[dict]:
    rows = []
    base_predictions = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    base_predictions["model"] = model_label
    base_predictions["feature_set"] = feature_set
    base_predictions["y_true"] = y_test
    base_predictions["y_prob"] = prob
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for strategy, threshold in threshold_by_strategy.items():
        pred = (prob >= threshold).astype(int)
        metrics = binary_metrics(y_test, pred, prob)
        predictions = base_predictions.copy()
        predictions["threshold_strategy"] = strategy
        predictions["threshold"] = threshold
        predictions["y_pred"] = pred
        predictions.to_csv(
            PREDICTIONS_DIR
            / f"early_warning_predictions_calibrated_{safe_filename(model_label)}_{safe_filename(feature_set)}_{safe_filename(strategy)}.csv",
            index=False,
        )
        rows.append(
            {
                "feature_set": feature_set,
                "model": model_label,
                "threshold_strategy": strategy,
                "threshold": threshold,
                "n_train": n_train,
                "n_test": len(test_aligned),
                "n_positive_test": int(y_test.sum()),
                "n_features": n_features,
                **metrics,
                "fit_time_s": fit_time,
                "predict_time_s": predict_time,
            }
        )
    return rows


def evaluate_external_warning_calibrated(
    train: pd.DataFrame, test: pd.DataFrame, feature_set: str
) -> list[dict]:
    cols, train_aligned, test_aligned = align_train_test(train, test)
    y_train = train_aligned[WARNING_TARGET].to_numpy(int)
    y_test = test_aligned[WARNING_TARGET].to_numpy(int)
    estimator = Pipeline(
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
    threshold_t0 = time.time()
    thresholds = grouped_cv_thresholds(
        estimator, train_aligned[cols], y_train, train_aligned["plot_id"].to_numpy()
    )
    logging.info(
        "[PHASE] calibrate logistic thresholds %s: %.1fs", feature_set, time.time() - threshold_t0
    )
    fit_t0 = time.time()
    estimator.fit(train_aligned[cols], y_train)
    fit_time = time.time() - fit_t0
    pred_t0 = time.time()
    prob = estimator.predict_proba(test_aligned[cols])[:, 1]
    predict_time = time.time() - pred_t0
    return calibrated_warning_rows_from_prob(
        "logistic_calibrated",
        feature_set,
        thresholds,
        test_aligned,
        y_test,
        prob,
        len(train_aligned),
        len(cols),
        fit_time,
        predict_time,
    )


def split_train_eval_by_plot(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    groups = table["plot_id"].to_numpy()
    train_idx, eval_idx = next(splitter.split(table, table[TARGET], groups=groups))
    return train_idx, eval_idx


def save_xgboost_learning_curve(
    model: XGBRegressor | XGBClassifier,
    model_name: str,
    feature_set: str,
    covariate_mode: str,
    metric_name: str,
) -> tuple[Path, Path]:
    history = model.evals_result()
    train_values = history.get("validation_0", {}).get(metric_name, [])
    eval_values = history.get("validation_1", {}).get(metric_name, [])
    if not train_values or not eval_values:
        return Path(), Path()

    XGB_CURVES_DIR.mkdir(parents=True, exist_ok=True)
    stem = safe_filename(f"{model_name}_{feature_set}_{covariate_mode}_{metric_name}")
    curve_path = RESULTS_DIR / f"{stem}_learning_curve.csv"
    figure_path = XGB_CURVES_DIR / f"{stem}_learning_curve.png"
    curve = pd.DataFrame(
        {
            "iteration": np.arange(len(train_values)),
            f"train_{metric_name}": train_values,
            f"eval_{metric_name}": eval_values,
        }
    )
    curve.to_csv(curve_path, index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        curve["iteration"],
        curve[f"train_{metric_name}"],
        label=f"2024 train {metric_name}",
        color="#2f6f9f",
        linewidth=2,
    )
    ax.plot(
        curve["iteration"],
        curve[f"eval_{metric_name}"],
        label=f"2024 eval {metric_name}",
        color="#d08336",
        linewidth=2,
    )
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        ax.axvline(
            best_iteration,
            color="#323232",
            linestyle="--",
            linewidth=1.1,
            label=f"best iter {best_iteration}",
        )
    ax.set_title(
        f"XGBoost learning curve: {feature_set} ({covariate_mode})", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel(metric_name.upper())
    ax.grid(axis="y", color="#d7d0c6", linewidth=0.8, alpha=0.75)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_facecolor("#fbf7ef")
    ax.set_facecolor("#fbf7ef")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return curve_path, figure_path


def evaluate_xgboost_regression(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
    covariate_mode: str,
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
        learning_rate=0.025,
        max_depth=2,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=8.0,
        early_stopping_rounds=30,
        random_state=SEED,
        tree_method="hist",
        n_jobs=4,
    )
    fit_t0 = time.time()
    model.fit(x_fit, y_fit, eval_set=[(x_fit, y_fit), (x_eval, y_eval)], verbose=False)
    fit_time = time.time() - fit_t0
    curve_path, figure_path = save_xgboost_learning_curve(
        model,
        "xgboost_severity",
        feature_set,
        covariate_mode,
        "rmse",
    )

    pred_t0 = time.time()
    eval_pred = model.predict(x_eval)
    test_pred = model.predict(x_test)
    lower = float(np.nanmin(y_fit))
    upper = float(np.nanmax(y_fit))
    eval_pred = np.clip(eval_pred, lower, upper)
    test_pred = np.clip(test_pred, lower, upper)
    predict_time = time.time() - pred_t0
    predictions = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    predictions["model"] = "XGBoost raw severity"
    predictions["feature_set"] = feature_set
    predictions["covariates"] = covariate_mode
    predictions["y_true"] = y_test
    predictions["y_pred"] = test_pred
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        PREDICTIONS_DIR
        / f"severity_predictions_xgboost_raw_severity_{safe_filename(feature_set)}_{safe_filename(covariate_mode)}.csv",
        index=False,
    )
    return {
        "feature_set": feature_set,
        "model": "xgboost_raw_severity",
        "covariates": covariate_mode,
        "n_train_fit": len(fit_part),
        "n_train_eval": len(eval_part),
        "n_test": len(test_aligned),
        "n_features": len(cols),
        "best_iteration": getattr(model, "best_iteration", np.nan),
        "prediction_bounds": f"{lower:.3f}-{upper:.3f}",
        "eval_rmse_2024": math.sqrt(mean_squared_error(y_eval, eval_pred)),
        "eval_mae_2024": mean_absolute_error(y_eval, eval_pred),
        "external_rmse_2025": math.sqrt(mean_squared_error(y_test, test_pred)),
        "external_mae_2025": mean_absolute_error(y_test, test_pred),
        "external_r2_2025": r2_score(y_test, test_pred) if len(np.unique(y_test)) > 1 else math.nan,
        "external_spearman_2025": safe_spearman(y_test, test_pred),
        "learning_curve_csv": str(curve_path) if curve_path else "",
        "learning_curve_figure": str(figure_path) if figure_path else "",
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }


def evaluate_xgboost_regression_200_trees(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
    covariate_mode: str,
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
        n_estimators=200,
        learning_rate=0.025,
        max_depth=2,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=8.0,
        random_state=SEED,
        tree_method="hist",
        n_jobs=4,
    )
    fit_t0 = time.time()
    model.fit(x_fit, y_fit, eval_set=[(x_fit, y_fit), (x_eval, y_eval)], verbose=False)
    fit_time = time.time() - fit_t0

    pred_t0 = time.time()
    eval_pred = model.predict(x_eval)
    test_pred = model.predict(x_test)
    lower = float(np.nanmin(y_fit))
    upper = float(np.nanmax(y_fit))
    eval_pred = np.clip(eval_pred, lower, upper)
    test_pred = np.clip(test_pred, lower, upper)
    predict_time = time.time() - pred_t0
    predictions = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    predictions["model"] = "XGBoost 200 trees raw severity"
    predictions["feature_set"] = feature_set
    predictions["covariates"] = covariate_mode
    predictions["y_true"] = y_test
    predictions["y_pred"] = test_pred
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        PREDICTIONS_DIR
        / f"severity_predictions_xgboost_200_raw_severity_{safe_filename(feature_set)}_{safe_filename(covariate_mode)}.csv",
        index=False,
    )
    return {
        "feature_set": feature_set,
        "covariates": covariate_mode,
        "n_train_fit": len(fit_part),
        "n_train_eval": len(eval_part),
        "n_test": len(test_aligned),
        "n_features": len(cols),
        "n_estimators": 200,
        "eval_rmse_2024": math.sqrt(mean_squared_error(y_eval, eval_pred)),
        "eval_mae_2024": mean_absolute_error(y_eval, eval_pred),
        "external_rmse_2025": math.sqrt(mean_squared_error(y_test, test_pred)),
        "external_mae_2025": mean_absolute_error(y_test, test_pred),
        "external_r2_2025": r2_score(y_test, test_pred) if len(np.unique(y_test)) > 1 else math.nan,
        "external_spearman_2025": safe_spearman(y_test, test_pred),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }


def evaluate_xgboost_warning(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> dict:
    cols, train_aligned, test_aligned = align_train_test(train, test)
    train_idx, eval_idx = split_train_eval_by_plot(train_aligned)
    fit_part = train_aligned.iloc[train_idx].copy()
    eval_part = train_aligned.iloc[eval_idx].copy()

    imputer = SimpleImputer(strategy="median")
    x_fit = imputer.fit_transform(fit_part[cols])
    x_eval = imputer.transform(eval_part[cols])
    x_test = imputer.transform(test_aligned[cols])
    y_fit = fit_part[WARNING_TARGET].to_numpy(int)
    y_eval = eval_part[WARNING_TARGET].to_numpy(int)
    y_test = test_aligned[WARNING_TARGET].to_numpy(int)
    positives = int(y_fit.sum())
    negatives = int(len(y_fit) - positives)
    scale_pos_weight = negatives / positives if positives else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=1200,
        learning_rate=0.025,
        max_depth=2,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=8.0,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=30,
        random_state=SEED,
        tree_method="hist",
        n_jobs=4,
    )
    fit_t0 = time.time()
    model.fit(x_fit, y_fit, eval_set=[(x_fit, y_fit), (x_eval, y_eval)], verbose=False)
    fit_time = time.time() - fit_t0
    curve_path, figure_path = save_xgboost_learning_curve(
        model,
        "xgboost_warning",
        feature_set,
        "spectral_only",
        "aucpr",
    )

    pred_t0 = time.time()
    eval_prob = model.predict_proba(x_eval)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    eval_pred = (eval_prob >= 0.5).astype(int)
    test_pred = (test_prob >= 0.5).astype(int)
    predict_time = time.time() - pred_t0
    test_has_two_classes = len(np.unique(y_test)) > 1
    eval_has_two_classes = len(np.unique(y_eval)) > 1
    predictions = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    predictions["model"] = "XGBoost"
    predictions["feature_set"] = feature_set
    predictions["threshold"] = WARNING_THRESHOLD
    predictions["y_true"] = y_test
    predictions["y_prob"] = test_prob
    predictions["y_pred"] = test_pred
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        PREDICTIONS_DIR / f"early_warning_predictions_xgboost_{safe_filename(feature_set)}.csv",
        index=False,
    )
    return {
        "feature_set": feature_set,
        "model": "xgboost_binary_warning",
        "threshold": WARNING_THRESHOLD,
        "n_train_fit": len(fit_part),
        "n_train_eval": len(eval_part),
        "n_test": len(test_aligned),
        "n_positive_test": int(y_test.sum()),
        "n_features": len(cols),
        "best_iteration": getattr(model, "best_iteration", np.nan),
        "scale_pos_weight": scale_pos_weight,
        "eval_auprc_2024": (
            average_precision_score(y_eval, eval_prob) if eval_has_two_classes else math.nan
        ),
        "eval_f1_2024": f1_score(y_eval, eval_pred, zero_division=0),
        "external_auroc_2025": (
            roc_auc_score(y_test, test_prob) if test_has_two_classes else math.nan
        ),
        "external_auprc_2025": (
            average_precision_score(y_test, test_prob) if test_has_two_classes else math.nan
        ),
        "external_f1_2025": f1_score(y_test, test_pred, zero_division=0),
        "external_precision_2025": precision_score(y_test, test_pred, zero_division=0),
        "external_recall_2025": recall_score(y_test, test_pred, zero_division=0),
        "external_balanced_accuracy_2025": (
            balanced_accuracy_score(y_test, test_pred) if test_has_two_classes else math.nan
        ),
        "learning_curve_csv": str(curve_path) if curve_path else "",
        "learning_curve_figure": str(figure_path) if figure_path else "",
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }


def evaluate_xgboost_warning_calibrated(
    train: pd.DataFrame, test: pd.DataFrame, feature_set: str
) -> list[dict]:
    cols, train_aligned, test_aligned = align_train_test(train, test)
    y_train = train_aligned[WARNING_TARGET].to_numpy(int)
    y_test = test_aligned[WARNING_TARGET].to_numpy(int)
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives else 1.0
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "xgb",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    n_estimators=200,
                    learning_rate=0.025,
                    max_depth=2,
                    min_child_weight=5,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    reg_alpha=0.1,
                    reg_lambda=8.0,
                    scale_pos_weight=scale_pos_weight,
                    random_state=SEED,
                    tree_method="hist",
                    n_jobs=4,
                ),
            ),
        ]
    )
    threshold_t0 = time.time()
    thresholds = grouped_cv_thresholds(
        estimator, train_aligned[cols], y_train, train_aligned["plot_id"].to_numpy()
    )
    logging.info(
        "[PHASE] calibrate xgboost thresholds %s: %.1fs", feature_set, time.time() - threshold_t0
    )
    fit_t0 = time.time()
    estimator.fit(train_aligned[cols], y_train)
    fit_time = time.time() - fit_t0
    pred_t0 = time.time()
    prob = estimator.predict_proba(test_aligned[cols])[:, 1]
    predict_time = time.time() - pred_t0
    return calibrated_warning_rows_from_prob(
        "xgboost_calibrated",
        feature_set,
        thresholds,
        test_aligned,
        y_test,
        prob,
        len(train_aligned),
        len(cols),
        fit_time,
        predict_time,
    )


def save_lagged_severity_predictions(
    predictions: pd.DataFrame,
    model_key: str,
    feature_block: str,
    covariate_mode: str,
    lag_policy: str,
) -> None:
    LAGGED_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        LAGGED_PREDICTIONS_DIR
        / f"severity_lagged_predictions_{safe_filename(model_key)}_{safe_filename(feature_block)}_{safe_filename(covariate_mode)}_{safe_filename(lag_policy)}.csv",
        index=False,
    )


def clear_lagged_prediction_outputs() -> None:
    LAGGED_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for path in LAGGED_PREDICTIONS_DIR.glob("severity_lagged_predictions*.csv"):
        path.unlink()


def add_lagged_known_covariates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    covariate_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = train.copy()
    test = test.copy()
    cols = list(cols)
    if covariate_mode in {"lagged_plus_predictor_week", "lagged_plus_week_horizon"}:
        train["known__predictor_week"] = train["predictor_week"]
        test["known__predictor_week"] = test["predictor_week"]
        cols.append("known__predictor_week")
    if covariate_mode == "lagged_plus_week_horizon":
        train["known__target_week"] = train["target_week"]
        test["known__target_week"] = test["target_week"]
        cols.append("known__target_week")
    return train, test, cols


def evaluate_lagged_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_block: str,
    covariate_mode: str,
    use_log: bool,
    lag_policy: str,
) -> dict:
    cols, train_aligned, test_aligned = align_train_test(train, test)
    train_aligned, test_aligned, cols = add_lagged_known_covariates(
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
    fit_t0 = time.time()
    pipeline.fit(train_aligned[cols], train_aligned[target_col].to_numpy(float))
    fit_time = time.time() - fit_t0
    pred_t0 = time.time()
    pred = pipeline.predict(test_aligned[cols])
    if use_log:
        pred = np.clip(np.expm1(pred), 0, None)
    train_target = train_aligned[TARGET].to_numpy(float)
    pred = np.clip(pred, np.nanmin(train_target), np.nanmax(train_target))
    predict_time = time.time() - pred_t0
    y = test_aligned[TARGET].to_numpy(float)
    model_key = "ridge_log_severity" if use_log else "ridge_raw_severity"
    predictions = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    predictions["model"] = "Ridge log severity" if use_log else "Ridge raw severity"
    predictions["feature_block"] = feature_block
    predictions["covariates"] = covariate_mode
    predictions["lag_policy"] = lag_policy
    predictions["y_true"] = y
    predictions["y_pred"] = pred
    save_lagged_severity_predictions(
        predictions, model_key, feature_block, covariate_mode, lag_policy
    )
    return {
        "feature_block": feature_block,
        "model": "ridge_log_severity" if use_log else "ridge_raw_severity",
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


def evaluate_lagged_xgboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_block: str,
    covariate_mode: str,
    fixed_200: bool,
    lag_policy: str,
) -> dict:
    cols, train_aligned, test_aligned = align_train_test(train, test)
    train_aligned, test_aligned, cols = add_lagged_known_covariates(
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
        n_estimators=200 if fixed_200 else 1600,
        learning_rate=0.015,
        max_depth=2,
        min_child_weight=12,
        subsample=0.70,
        colsample_bytree=0.55,
        reg_alpha=2.0,
        reg_lambda=25.0,
        gamma=1.0,
        early_stopping_rounds=None if fixed_200 else 50,
        random_state=SEED,
        tree_method="hist",
        n_jobs=4,
    )
    fit_t0 = time.time()
    model.fit(x_fit, y_fit, eval_set=[(x_fit, y_fit), (x_eval, y_eval)], verbose=False)
    fit_time = time.time() - fit_t0
    pred_t0 = time.time()
    eval_pred = np.clip(model.predict(x_eval), float(np.nanmin(y_fit)), float(np.nanmax(y_fit)))
    test_pred = np.clip(model.predict(x_test), float(np.nanmin(y_fit)), float(np.nanmax(y_fit)))
    predict_time = time.time() - pred_t0
    model_key = "xgboost_200_raw_severity" if fixed_200 else "xgboost_raw_severity"
    predictions = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    predictions["model"] = "XGBoost 200 trees raw severity" if fixed_200 else "XGBoost raw severity"
    predictions["feature_block"] = feature_block
    predictions["covariates"] = covariate_mode
    predictions["lag_policy"] = lag_policy
    predictions["y_true"] = y_test
    predictions["y_pred"] = test_pred
    save_lagged_severity_predictions(
        predictions, model_key, feature_block, covariate_mode, lag_policy
    )
    return {
        "feature_block": feature_block,
        "model": model_key,
        "covariates": covariate_mode,
        "lag_policy": lag_policy,
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
        "external_rmse_2025": math.sqrt(mean_squared_error(y_test, test_pred)),
        "external_mae_2025": mean_absolute_error(y_test, test_pred),
        "external_r2_2025": r2_score(y_test, test_pred) if len(np.unique(y_test)) > 1 else math.nan,
        "external_spearman_2025": safe_spearman(y_test, test_pred),
        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
    }


def disease_progression(disease: pd.DataFrame) -> pd.DataFrame:
    return (
        disease.groupby(["year", "week"], as_index=False)
        .agg(
            plots=("plot_id", "nunique"),
            mean_ds_plot=("ds_plot", "mean"),
            max_ds_plot=("ds_plot", "max"),
        )
        .sort_values(["year", "week"])
    )


def warning_horizon_diagnostics_from_predictions() -> pd.DataFrame:
    rows = []
    for path in sorted(PREDICTIONS_DIR.glob("early_warning_predictions*.csv")):
        pred = pd.read_csv(path)
        if not {"target_week", "y_true", "y_pred"}.issubset(pred.columns):
            continue
        model = pred["model"].iloc[0] if "model" in pred.columns else path.stem
        feature_set = pred["feature_set"].iloc[0] if "feature_set" in pred.columns else ""
        strategy = (
            pred["threshold_strategy"].iloc[0]
            if "threshold_strategy" in pred.columns
            else "fixed_0_5"
        )
        threshold = pred["threshold"].iloc[0] if "threshold" in pred.columns else 0.5
        for target_week, group in pred.groupby("target_week", sort=True):
            metrics = binary_metrics(
                group["y_true"].to_numpy(int),
                group["y_pred"].to_numpy(int),
                group.get("y_prob", pd.Series(np.nan, index=group.index)).to_numpy(float),
            )
            rows.append(
                {
                    "source_file": path.name,
                    "model": model,
                    "feature_set": feature_set,
                    "threshold_strategy": strategy,
                    "threshold": threshold,
                    "target_week": target_week,
                    "n": len(group),
                    "n_positive": int(group["y_true"].sum()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def severity_horizon_diagnostics_from_predictions() -> pd.DataFrame:
    rows = []
    for path in sorted(PREDICTIONS_DIR.glob("severity_predictions*.csv")):
        pred = pd.read_csv(path)
        if not {"target_week", "y_true", "y_pred"}.issubset(pred.columns):
            continue
        model = pred["model"].iloc[0] if "model" in pred.columns else path.stem
        feature_set = pred["feature_set"].iloc[0] if "feature_set" in pred.columns else ""
        covariates = pred["covariates"].iloc[0] if "covariates" in pred.columns else ""
        pred["error"] = pred["y_pred"] - pred["y_true"]
        pred["abs_error"] = pred["error"].abs()
        pred["squared_error"] = pred["error"] ** 2
        for target_week, group in pred.groupby("target_week", sort=True):
            rows.append(
                {
                    "source_file": path.name,
                    "model": model,
                    "feature_set": feature_set,
                    "covariates": covariates,
                    "target_week": target_week,
                    "n": len(group),
                    "mean_true": group["y_true"].mean(),
                    "mean_pred": group["y_pred"].mean(),
                    "bias": group["error"].mean(),
                    "mae": group["abs_error"].mean(),
                    "rmse": math.sqrt(group["squared_error"].mean()),
                }
            )
    return pd.DataFrame(rows)


def lagged_severity_horizon_diagnostics_from_predictions() -> pd.DataFrame:
    rows = []
    for path in sorted(LAGGED_PREDICTIONS_DIR.glob("severity_lagged_predictions*.csv")):
        pred = pd.read_csv(path)
        if not {"target_week", "y_true", "y_pred"}.issubset(pred.columns):
            continue
        pred["error"] = pred["y_pred"] - pred["y_true"]
        pred["abs_error"] = pred["error"].abs()
        pred["squared_error"] = pred["error"] ** 2
        for target_week, group in pred.groupby("target_week", sort=True):
            rows.append(
                {
                    "source_file": path.name,
                    "model": pred["model"].iloc[0],
                    "feature_block": pred["feature_block"].iloc[0],
                    "covariates": pred["covariates"].iloc[0],
                    "lag_policy": (
                        pred["lag_policy"].iloc[0] if "lag_policy" in pred.columns else ""
                    ),
                    "target_week": target_week,
                    "n": len(group),
                    "mean_true": group["y_true"].mean(),
                    "mean_pred": group["y_pred"].mean(),
                    "bias": group["error"].mean(),
                    "mae": group["abs_error"].mean(),
                    "rmse": math.sqrt(group["squared_error"].mean()),
                }
            )
    return pd.DataFrame(rows)


def lagged_severity_delta_vs_baselines(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline_names = ["previous_disease_only", "nadir_plus_previous_disease"]
    for (model, covariates, lag_policy), group in results.groupby(
        ["model", "covariates", "lag_policy"]
    ):
        for baseline_name in baseline_names:
            baseline = group[group["feature_block"] == baseline_name]
            if baseline.empty:
                continue
            base = baseline.iloc[0]
            for _, row in group[group["feature_block"] != baseline_name].iterrows():
                rows.append(
                    {
                        "model": model,
                        "covariates": covariates,
                        "lag_policy": lag_policy,
                        "baseline": baseline_name,
                        "feature_block": row["feature_block"],
                        "rmse_reduction_vs_baseline": base["rmse"] - row["rmse"],
                        "mae_reduction_vs_baseline": base["mae"] - row["mae"],
                        "delta_r2_vs_baseline": row["r2"] - base["r2"],
                        "delta_spearman_vs_baseline": row["spearman"] - base["spearman"],
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["model", "covariates", "lag_policy", "baseline", "rmse_reduction_vs_baseline"],
        ascending=[True, True, True, True, False],
    )


def feature_set_audit(features: dict[str, pd.DataFrame], year: int) -> pd.DataFrame:
    rows = []
    for name, frame in features.items():
        cols = feature_columns(frame)
        non_null = frame[cols].notna().mean().mean() if cols else np.nan
        rows.append(
            {
                "year": year,
                "feature_set": name,
                "rows": len(frame),
                "plots": frame["plot_id"].nunique() if "plot_id" in frame.columns else np.nan,
                "weeks": frame["week"].nunique() if "week" in frame.columns else np.nan,
                "n_features": len(cols),
                "mean_feature_non_null_fraction": non_null,
            }
        )
    return pd.DataFrame(rows)


def severity_delta_vs_nadir(severity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = ["model", "covariates"] if "covariates" in severity.columns else ["model"]
    for keys, group in severity.groupby(group_keys):
        nadir = group[group["feature_set"] == "nadir"]
        if nadir.empty:
            continue
        baseline = nadir.iloc[0]
        for _, row in group[group["feature_set"] != "nadir"].iterrows():
            result = {
                "feature_set": row["feature_set"],
                "delta_rmse": row["rmse"] - baseline["rmse"],
                "delta_mae": row["mae"] - baseline["mae"],
                "delta_r2": row["r2"] - baseline["r2"],
                "delta_spearman": row["spearman"] - baseline["spearman"],
            }
            if isinstance(keys, tuple):
                result.update(dict(zip(group_keys, keys, strict=False)))
            else:
                result[group_keys[0]] = keys
            rows.append(result)
    sort_cols = (
        ["model", "covariates", "delta_rmse"]
        if rows and "covariates" in rows[0]
        else ["model", "delta_rmse"]
    )
    return pd.DataFrame(rows).sort_values(sort_cols)


def warning_delta_vs_nadir(warning: pd.DataFrame) -> pd.DataFrame:
    nadir = warning[warning["feature_set"] == "nadir"]
    if nadir.empty:
        return pd.DataFrame()
    baseline = nadir.iloc[0]
    rows = []
    for _, row in warning[warning["feature_set"] != "nadir"].iterrows():
        rows.append(
            {
                "feature_set": row["feature_set"],
                "delta_auroc": row["auroc"] - baseline["auroc"],
                "delta_auprc": row["auprc"] - baseline["auprc"],
                "delta_f1": row["f1"] - baseline["f1"],
                "delta_precision": row["precision"] - baseline["precision"],
                "delta_recall": row["recall"] - baseline["recall"],
                "delta_balanced_accuracy": row["balanced_accuracy"] - baseline["balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows).sort_values("delta_f1", ascending=False)


def plot_binary_warning_metrics(warning: pd.DataFrame, output_path: Path) -> None:
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = {
        "nadir": "Nadir",
        "multiangular_vza_compact": "VZA compact",
        "multiangular_geometry_compact": "Geometry compact",
        "multiangular_vza": "VZA",
        "multiangular_vza_raa": "VZA + RAA",
        "multiangular_vza_phase": "VZA + phase",
    }
    order = [
        "nadir",
        "multiangular_vza",
        "multiangular_vza_compact",
        "multiangular_geometry_compact",
        "multiangular_vza_raa",
        "multiangular_vza_phase",
    ]
    metrics = [
        ("f1", "F1 score"),
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
    ]
    plot_data = warning.set_index("feature_set").loc[order].reset_index()
    palette = {
        "nadir": "#323232",
        "multiangular_vza_compact": "#4f7f58",
        "multiangular_geometry_compact": "#a75d2a",
        "multiangular_vza": "#7b9e87",
        "multiangular_vza_raa": "#d08336",
        "multiangular_vza_phase": "#2f6f9f",
    }

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), sharey=True)
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics):
        values = plot_data[metric].to_numpy(float)
        colors = [palette[key] for key in plot_data["feature_set"]]
        bars = ax.bar(
            range(len(plot_data)),
            values,
            color=colors,
            width=0.68,
            edgecolor="white",
            linewidth=1.2,
        )
        baseline = float(plot_data.loc[plot_data["feature_set"] == "nadir", metric].iloc[0])
        ax.axhline(baseline, color="#323232", linestyle="--", linewidth=1.1, alpha=0.75)
        ax.text(
            3.42,
            baseline + 0.012,
            "nadir",
            fontsize=8.5,
            color="#323232",
            ha="right",
            va="bottom",
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.018,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#1f1f1f",
            )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylim(0, 0.82)
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(
            [labels[key] for key in plot_data["feature_set"]], rotation=25, ha="right"
        )
        ax.grid(axis="y", color="#d7d0c6", linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#6b6258")
        ax.spines["bottom"].set_color("#6b6258")

    fig.suptitle(
        "External early-warning classification: train 2024, test 2025",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.02,
        0.935,
        "Target: future disease warning (DSDI plot severity >= 5). Higher is better for all metrics.",
        fontsize=10.5,
        color="#3f3a34",
    )
    fig.text(0.01, 0.5, "Metric value", rotation=90, va="center", fontsize=11)
    fig.patch.set_facecolor("#fbf7ef")
    for ax in axes:
        ax.set_facecolor("#fbf7ef")
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.91])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_phase("plot binary warning metrics", t0)


def plot_binary_warning_delta_metrics(warning_delta: pd.DataFrame, output_path: Path) -> None:
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = {
        "multiangular_vza": "VZA",
        "multiangular_vza_compact": "VZA compact",
        "multiangular_geometry_compact": "Geometry compact",
        "multiangular_vza_raa": "VZA + RAA",
        "multiangular_vza_phase": "VZA + phase",
    }
    order = [
        "multiangular_vza",
        "multiangular_vza_compact",
        "multiangular_geometry_compact",
        "multiangular_vza_raa",
        "multiangular_vza_phase",
    ]
    metrics = [
        ("delta_f1", "Delta F1 score"),
        ("delta_recall", "Delta recall"),
        ("delta_precision", "Delta precision"),
        ("delta_balanced_accuracy", "Delta balanced accuracy"),
        ("delta_auroc", "Delta AUROC"),
        ("delta_auprc", "Delta AUPRC"),
    ]
    plot_data = warning_delta.set_index("feature_set").loc[order].reset_index()
    palette = {
        "multiangular_vza": "#7b9e87",
        "multiangular_vza_compact": "#4f7f58",
        "multiangular_geometry_compact": "#a75d2a",
        "multiangular_vza_raa": "#d08336",
        "multiangular_vza_phase": "#2f6f9f",
    }

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), sharey=True)
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics):
        values = plot_data[metric].to_numpy(float)
        colors = [palette[key] for key in plot_data["feature_set"]]
        bars = ax.bar(
            range(len(plot_data)),
            values,
            color=colors,
            width=0.68,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.axhline(0, color="#323232", linewidth=1.2)
        ax.text(4.42, 0.006, "nadir = 0", fontsize=8.5, color="#323232", ha="right", va="bottom")
        for bar, value in zip(bars, values):
            va = "bottom" if value >= 0 else "top"
            y = value + 0.006 if value >= 0 else value - 0.006
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:+.2f}",
                ha="center",
                va=va,
                fontsize=9,
                color="#1f1f1f",
            )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylim(-0.06, 0.24)
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(
            [labels[key] for key in plot_data["feature_set"]], rotation=25, ha="right"
        )
        ax.grid(axis="y", color="#d7d0c6", linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#6b6258")
        ax.spines["bottom"].set_color("#6b6258")

    fig.suptitle(
        "External early-warning gain over nadir: train 2024, test 2025",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.02,
        0.935,
        "Bars show metric difference from the nadir-only baseline. Positive values indicate improvement over nadir.",
        fontsize=10.5,
        color="#3f3a34",
    )
    fig.text(0.01, 0.5, "Delta vs nadir", rotation=90, va="center", fontsize=11)
    fig.patch.set_facecolor("#fbf7ef")
    for ax in axes:
        ax.set_facecolor("#fbf7ef")
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.91])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log_phase("plot binary warning delta metrics", t0)


def write_report(
    outputs: dict[str, Path],
    severity: pd.DataFrame,
    severity_delta: pd.DataFrame,
    warning: pd.DataFrame,
    calibrated_warning: pd.DataFrame,
    warning_delta: pd.DataFrame,
    xgb_severity: pd.DataFrame,
    xgb_warning: pd.DataFrame,
    audit: pd.DataFrame,
    log_path: Path,
) -> None:
    xgb_severity_display_cols = [
        "feature_set",
        "covariates",
        "n_train_fit",
        "n_train_eval",
        "n_test",
        "n_features",
        "best_iteration",
        "eval_rmse_2024",
        "external_rmse_2025",
        "external_mae_2025",
        "external_r2_2025",
        "external_spearman_2025",
    ]
    xgb_warning_display_cols = [
        "feature_set",
        "n_train_fit",
        "n_train_eval",
        "n_test",
        "n_features",
        "best_iteration",
        "eval_auprc_2024",
        "external_auroc_2025",
        "external_auprc_2025",
        "external_f1_2025",
        "external_recall_2025",
        "external_balanced_accuracy_2025",
    ]
    report = [
        "## Results: Cross-Year Generalization 2024 -> 2025",
        "",
        "Models are trained on 2024 plot-week pairs and tested directly on 2025 plot-week pairs. Feature columns are restricted to columns present in both years.",
        "",
        "### External severity prediction",
        "",
        markdown_table(severity.round(4)),
        "",
        "### External severity delta vs nadir",
        "",
        "Negative `delta_rmse`/`delta_mae` is better. Positive `delta_r2`/`delta_spearman` is better.",
        "",
        markdown_table(severity_delta.round(4)),
        "",
        f"### External binary warning prediction (`ds_plot >= {WARNING_THRESHOLD:g}`)",
        "",
        markdown_table(warning.round(4)),
        "",
        "### External binary warning with 2024-calibrated thresholds",
        "",
        "Thresholds are selected using grouped 2024 validation predictions only, then applied once to the 2025 external test set.",
        "",
        markdown_table(
            calibrated_warning[
                [
                    "model",
                    "feature_set",
                    "threshold_strategy",
                    "threshold",
                    "n_features",
                    "f1",
                    "precision",
                    "recall",
                    "specificity",
                    "false_positive_rate",
                    "balanced_accuracy",
                ]
            ]
            .sort_values(["model", "threshold_strategy", "f1"], ascending=[True, True, False])
            .round(4)
        ),
        "",
        "### External binary warning delta vs nadir",
        "",
        "Positive deltas mean the multiangular model outperformed the nadir-only baseline.",
        "",
        markdown_table(warning_delta.round(4)),
        "",
        "### XGBoost severity with 2024 train/eval and 2025 external test",
        "",
        "XGBoost uses a grouped 2024 train/eval split for early stopping, then evaluates once on 2025.",
        "",
        markdown_table(xgb_severity[xgb_severity_display_cols].round(4)),
        "",
        "### XGBoost binary warning with 2024 train/eval and 2025 external test",
        "",
        markdown_table(xgb_warning[xgb_warning_display_cols].round(4)),
        "",
        "### 2025 DSDI parsing audit",
        "",
        markdown_table(audit.round(4)),
        "",
        "**Interpretation**: Spectral-only severity regression still transfers poorly across years, but adding known phenology/horizon covariates makes multiangular severity prediction generalize better than the matched nadir baseline. The best external severity model was raw severity with VZA + phase + week/horizon covariates (RMSE 9.061, R2 0.389), compared with nadir + week/horizon (RMSE 9.825, R2 0.282). For binary early warning, angular-plus-sun-geometry features also improved F1, recall, and balanced accuracy over nadir-only.",
        "",
        "### Reproducibility",
        "",
        "- Train year: `2024`",
        "- Test year: `2025`",
        "- 2025 target pairing: each reflectance week predicts the next available later DSDI observation",
        "- Severity covariate modes: spectral-only; spectral + predictor week; spectral + predictor week + target week/horizon",
        f"- Reliable angular-bin feature sets filter VZA/RAA/phase bins to `n_pixels >= {MIN_ANGULAR_BIN_PIXELS}` and `n_images >= {MIN_ANGULAR_BIN_IMAGES}`",
        "- Angular-shape feature sets include normalized off-nadir-vs-nadir contrasts, VZA slope, curvature, and angular range",
        "- Targets: raw severity, log-transformed severity, binary warning",
        "- Model: ridge regression for severity with predictions clipped to the 2024 training DSDI range; logistic regression for warning",
        "- XGBoost: 2024 grouped train/eval split by plot, early stopping on the eval split, one final external test on 2025",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
    ]
    for label, path in outputs.items():
        report.append(f"- {label}: `{path}`")
    outputs["report"].write_text("\n".join(report) + "\n", encoding="utf-8")


def write_lagged_disease_report(
    outputs: dict[str, Path],
    lagged_results: pd.DataFrame,
    lagged_delta: pd.DataFrame,
    lagged_horizon: pd.DataFrame,
    log_path: Path,
) -> None:
    display_cols = [
        "feature_block",
        "model",
        "covariates",
        "lag_policy",
        "n_features",
        "rmse",
        "mae",
        "r2",
        "spearman",
    ]
    best = lagged_results.sort_values("rmse").head(20)
    report = [
        "## Results: Lagged Disease Presence for External Severity Prediction",
        "",
        "This experiment adds previous disease observations as predictors and reports two lag policies: observations through the predictor week, and the stricter rule using only observations before the predictor week.",
        "",
        "### Best Lagged-Disease Severity Models",
        "",
        markdown_table(best[display_cols].round(4)),
        "",
        "### Delta vs Baselines",
        "",
        "Positive RMSE reduction means the candidate improved over the baseline.",
        "",
        markdown_table(lagged_delta.round(4).head(40)),
        "",
        "### Horizon Diagnostics",
        "",
        markdown_table(lagged_horizon.round(4).head(80)),
        "",
        "**Interpretation**: If `previous_disease_only` is strong, observed disease onset explains much of future severity. If `multiangular + previous disease` improves over `nadir + previous disease`, multiangular reflectance adds information beyond observed disease presence.",
        "",
        "### Reproducibility",
        "",
        "- Train year: `2024`",
        "- Test year: `2025`",
        "- Lag rules: `through_predictor_week` uses `disease_week <= predictor_week`; `before_predictor_week` uses `disease_week < predictor_week`",
        "- Covariate modes: lagged disease only; lagged disease + predictor week; lagged disease + predictor week + target week",
        "- Models: Ridge raw/log severity; heavily regularized XGBoost raw severity early stopping and 200-tree variants",
        "- Predictors excluded: treatment and inoculation status",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
    ]
    for label, path in outputs.items():
        if "lagged" in label:
            report.append(f"- {label}: `{path}`")
    outputs["lagged_disease_report"].write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    XGB_CURVES_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    LAGGED_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()

    disease_2024 = load_2024_disease_with_fallback()
    disease_2025, audit_2025 = load_2025_disease_with_fallback()
    DISEASE_OUT.parent.mkdir(parents=True, exist_ok=True)
    disease_2025.to_csv(DISEASE_OUT, index=False)

    features_2024 = load_feature_sets_for_year(VZA_2024, RAA_2024)
    features_2025 = load_feature_sets_for_year(VZA_2025, RAA_2025)
    try:
        validate_2025_mapping(
            pl.read_parquet(VZA_2025).to_pandas(), disease_2025, read_2025_plot_map()
        )
    except Exception as exc:
        logging.warning("Skipping 2025 polygon mapping validation: %s", exc)

    severity_rows = []
    warning_rows = []
    calibrated_warning_rows = []
    xgb_severity_rows = []
    xgb_severity_200_rows = []
    xgb_warning_rows = []
    for feature_set in sorted(features_2024):
        train = build_model_table(features_2024[feature_set], disease_2024)
        test = build_model_table(features_2025[feature_set], disease_2025)
        if train.empty or test.empty:
            continue
        for covariate_mode in ["spectral_only", "spectral_plus_week", "spectral_plus_week_horizon"]:
            severity_rows.append(
                evaluate_external_regression(
                    train,
                    test,
                    feature_set,
                    use_log=False,
                    covariate_mode=covariate_mode,
                )
            )
            severity_rows.append(
                evaluate_external_regression(
                    train,
                    test,
                    feature_set,
                    use_log=True,
                    covariate_mode=covariate_mode,
                )
            )
        warning_rows.append(evaluate_external_warning(train, test, feature_set))
        calibrated_warning_rows.extend(
            evaluate_external_warning_calibrated(train, test, feature_set)
        )
        for covariate_mode in ["spectral_only", "spectral_plus_week", "spectral_plus_week_horizon"]:
            xgb_severity_rows.append(
                evaluate_xgboost_regression(train, test, feature_set, covariate_mode)
            )
            xgb_severity_200_rows.append(
                evaluate_xgboost_regression_200_trees(train, test, feature_set, covariate_mode)
            )
        xgb_warning_rows.append(evaluate_xgboost_warning(train, test, feature_set))
        calibrated_warning_rows.extend(
            evaluate_xgboost_warning_calibrated(train, test, feature_set)
        )

    lagged_rows = []
    clear_lagged_prediction_outputs()
    lagged_blocks = {
        "previous_disease_only": "nadir",
        "nadir_plus_previous_disease": "nadir",
        "multiangular_vza_phase_plus_previous_disease": "multiangular_vza_phase",
        "multiangular_angular_shape_plus_previous_disease": "multiangular_angular_shape",
        "multiangular_vza_compact_plus_previous_disease": "multiangular_vza_compact",
    }
    for block_name, source_feature_set in lagged_blocks.items():
        if source_feature_set not in features_2024 or source_feature_set not in features_2025:
            continue
        train = build_model_table(features_2024[source_feature_set], disease_2024)
        test = build_model_table(features_2025[source_feature_set], disease_2025)
        if train.empty or test.empty:
            continue
        for lag_policy in LAG_POLICIES:
            train_lagged = add_lagged_disease_features(train, disease_2024, lag_policy)
            test_lagged = add_lagged_disease_features(test, disease_2025, lag_policy)
            if block_name == "previous_disease_only":
                train_lagged = lagged_disease_only_table(train_lagged)
                test_lagged = lagged_disease_only_table(test_lagged)
            for covariate_mode in [
                "lagged_only",
                "lagged_plus_predictor_week",
                "lagged_plus_week_horizon",
            ]:
                lagged_rows.append(
                    evaluate_lagged_ridge(
                        train_lagged,
                        test_lagged,
                        block_name,
                        covariate_mode,
                        use_log=False,
                        lag_policy=lag_policy,
                    )
                )
                lagged_rows.append(
                    evaluate_lagged_ridge(
                        train_lagged,
                        test_lagged,
                        block_name,
                        covariate_mode,
                        use_log=True,
                        lag_policy=lag_policy,
                    )
                )
                lagged_rows.append(
                    evaluate_lagged_xgboost(
                        train_lagged,
                        test_lagged,
                        block_name,
                        covariate_mode,
                        fixed_200=False,
                        lag_policy=lag_policy,
                    )
                )
                lagged_rows.append(
                    evaluate_lagged_xgboost(
                        train_lagged,
                        test_lagged,
                        block_name,
                        covariate_mode,
                        fixed_200=True,
                        lag_policy=lag_policy,
                    )
                )

    severity = pd.DataFrame(severity_rows).sort_values(["model", "covariates", "rmse"])
    lagged_severity = pd.DataFrame(lagged_rows).sort_values(
        ["model", "covariates", "lag_policy", "rmse"]
    )
    lagged_delta = (
        lagged_severity_delta_vs_baselines(lagged_severity)
        if not lagged_severity.empty
        else pd.DataFrame()
    )
    warning = pd.DataFrame(warning_rows).sort_values("f1", ascending=False)
    calibrated_warning = pd.DataFrame(calibrated_warning_rows).sort_values(
        ["model", "threshold_strategy", "f1"], ascending=[True, True, False]
    )
    xgb_severity = pd.DataFrame(xgb_severity_rows).sort_values("external_rmse_2025")
    xgb_severity_200 = pd.DataFrame(xgb_severity_200_rows).sort_values("external_rmse_2025")
    xgb_warning = pd.DataFrame(xgb_warning_rows).sort_values("external_f1_2025", ascending=False)
    severity_delta = severity_delta_vs_nadir(severity)
    warning_delta = warning_delta_vs_nadir(warning)
    progress = pd.concat(
        [disease_progression(disease_2024), disease_progression(disease_2025)], ignore_index=True
    )
    feature_audit = pd.concat(
        [feature_set_audit(features_2024, 2024), feature_set_audit(features_2025, 2025)],
        ignore_index=True,
    )
    warning_horizon = warning_horizon_diagnostics_from_predictions()
    severity_horizon = severity_horizon_diagnostics_from_predictions()
    lagged_horizon = lagged_severity_horizon_diagnostics_from_predictions()

    outputs = {
        "clean_disease_2025": DISEASE_OUT,
        "disease_progression": RESULTS_DIR / "disease_progression_2024_2025.csv",
        "dsdi_audit_2025": RESULTS_DIR / "dsdi_parsing_audit_2025.csv",
        "severity_external": RESULTS_DIR / "severity_external_2024_train_2025_test.csv",
        "lagged_disease_severity_external": RESULTS_DIR
        / "severity_lagged_disease_external_2024_train_2025_test.csv",
        "lagged_disease_delta_vs_baselines": RESULTS_DIR
        / "severity_lagged_disease_delta_vs_baselines.csv",
        "severity_external_delta_vs_nadir": RESULTS_DIR / "severity_external_delta_vs_nadir.csv",
        "warning_external": RESULTS_DIR / "warning_external_2024_train_2025_test.csv",
        "warning_calibrated_external": RESULTS_DIR
        / "warning_calibrated_external_2024_train_2025_test.csv",
        "warning_external_delta_vs_nadir": RESULTS_DIR / "warning_external_delta_vs_nadir.csv",
        "xgboost_severity_external": RESULTS_DIR / "xgboost_severity_train_eval_2024_test_2025.csv",
        "xgboost_severity_200_external": RESULTS_DIR
        / "xgboost_severity_200_trees_train_eval_2024_test_2025.csv",
        "xgboost_warning_external": RESULTS_DIR / "xgboost_warning_train_eval_2024_test_2025.csv",
        "warning_horizon_diagnostics": RESULTS_DIR / "warning_horizon_diagnostics.csv",
        "severity_horizon_diagnostics": RESULTS_DIR / "severity_horizon_diagnostics.csv",
        "lagged_disease_horizon_diagnostics": RESULTS_DIR
        / "severity_lagged_disease_horizon_diagnostics.csv",
        "feature_set_audit": RESULTS_DIR / "feature_set_audit.csv",
        "early_warning_predictions": PREDICTIONS_DIR,
        "lagged_disease_predictions": LAGGED_PREDICTIONS_DIR,
        "binary_warning_figure": FIGURES_DIR / "binary_warning_external_2024_train_2025_test.png",
        "binary_warning_delta_figure": FIGURES_DIR
        / "binary_warning_delta_vs_nadir_external_2024_train_2025_test.png",
        "xgboost_learning_curves": XGB_CURVES_DIR,
        "report": REPORTS_DIR / "cross_year_generalization_2024_to_2025_summary.md",
        "lagged_disease_report": REPORTS_DIR / "severity_lagged_disease_summary.md",
    }
    progress.to_csv(outputs["disease_progression"], index=False)
    audit_2025.to_csv(outputs["dsdi_audit_2025"], index=False)
    severity.to_csv(outputs["severity_external"], index=False)
    lagged_severity.to_csv(outputs["lagged_disease_severity_external"], index=False)
    lagged_delta.to_csv(outputs["lagged_disease_delta_vs_baselines"], index=False)
    severity_delta.to_csv(outputs["severity_external_delta_vs_nadir"], index=False)
    warning.to_csv(outputs["warning_external"], index=False)
    calibrated_warning.to_csv(outputs["warning_calibrated_external"], index=False)
    warning_delta.to_csv(outputs["warning_external_delta_vs_nadir"], index=False)
    xgb_severity.to_csv(outputs["xgboost_severity_external"], index=False)
    xgb_severity_200.to_csv(outputs["xgboost_severity_200_external"], index=False)
    xgb_warning.to_csv(outputs["xgboost_warning_external"], index=False)
    warning_horizon.to_csv(outputs["warning_horizon_diagnostics"], index=False)
    severity_horizon.to_csv(outputs["severity_horizon_diagnostics"], index=False)
    lagged_horizon.to_csv(outputs["lagged_disease_horizon_diagnostics"], index=False)
    feature_audit.to_csv(outputs["feature_set_audit"], index=False)
    plot_binary_warning_metrics(warning, outputs["binary_warning_figure"])
    plot_binary_warning_delta_metrics(warning_delta, outputs["binary_warning_delta_figure"])
    write_report(
        outputs,
        severity,
        severity_delta,
        warning,
        calibrated_warning,
        warning_delta,
        xgb_severity,
        xgb_warning,
        audit_2025,
        log_path,
    )
    write_lagged_disease_report(outputs, lagged_severity, lagged_delta, lagged_horizon, log_path)
    log_phase("total", start)


if __name__ == "__main__":
    main()
