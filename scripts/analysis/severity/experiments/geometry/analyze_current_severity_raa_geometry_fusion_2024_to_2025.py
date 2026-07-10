#!/usr/bin/env python3
"""Current severity prediction with RAA/phase angular geometry features.

This is an imaging-only follow-up to the current plot severity analysis. It
adds relative azimuth angle (RAA) and phase-angle binned reflectance features,
but it does not use treatment, cultivar, block, or other agronomic design
metadata as predictors.
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

ROOT = Path(__file__).resolve().parents[5]
MPLCONFIG_DIR = ROOT / "outputs/.matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity import (
    analyze_current_plot_severity_2024_to_2025 as current_severity,
)
from scripts.analysis.severity import (
    debug_multiangular_rmse_bottleneck as residual_pipeline,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    clean_token,
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/raa_geometry_fusion_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

VZA_2024 = (
    ROOT
    / "outputs/runs/analysis/reflectance/distributions/2024/ground_filtered/results/plot_week_angle_features_2024.parquet"
)
VZA_2025 = (
    ROOT
    / "outputs/runs/analysis/reflectance/distributions/2025/ground_filtered/results/plot_week_angle_features_2025.parquet"
)
RAA_2024 = (
    ROOT
    / "outputs/runs/analysis/reflectance/raa_sun_geometry/2024/ground_filtered/results/plot_week_vza_raa_features_2024.parquet"
)
RAA_2025 = (
    ROOT
    / "outputs/runs/analysis/reflectance/raa_sun_geometry/2025/ground_filtered/results/plot_week_vza_raa_features_2025.parquet"
)

DISEASE_2024_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2024.csv"
DISEASE_2025_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2025.csv"
CURRENT_RESULTS_DIR = ROOT / "outputs/runs/analysis/severity/current/2024_to_2025/results"
CURVE_RESULTS_DIR = ROOT / "outputs/runs/analysis/severity/current/curve_embeddings_2024_to_2025/results"

COVARIATES = "spectral_plus_week"
META_COLS = ["year", "week", "plot_id", "cult", "trt"]
MIN_ANGULAR_BIN_PIXELS = 500
MIN_ANGULAR_BIN_IMAGES = 2

TARGET = current_severity.TARGET
TARGET_LOG = current_severity.TARGET_LOG
WARNING_TARGET = current_severity.WARNING_TARGET
WARNING_THRESHOLD = current_severity.WARNING_THRESHOLD
SEED = current_severity.SEED


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        LOGS_DIR / f"analyze_current_severity_raa_geometry_fusion_2024_to_2025_{timestamp}.log"
    )
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


def configure_reused_pipeline_paths() -> None:
    residual_pipeline.ROOT = ROOT
    residual_pipeline.COVARIATES = COVARIATES
    residual_pipeline.OUTPUT_ROOT = OUTPUT_ROOT
    residual_pipeline.RESULTS_DIR = RESULTS_DIR
    residual_pipeline.REPORTS_DIR = REPORTS_DIR
    residual_pipeline.FIGURES_DIR = FIGURES_DIR
    residual_pipeline.PREDICTIONS_DIR = PREDICTIONS_DIR
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "raa_geometry_manifest.json"


def read_parquet_with_timing(path: Path) -> pd.DataFrame:
    started = time.perf_counter()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pl.read_parquet(path).to_pandas()
    logging.info("Read %s rows x %s cols from %s", frame.shape[0], frame.shape[1], path)
    log_phase(f"read parquet {path.name}", started)
    return frame


def read_inputs() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    vza_2024 = read_parquet_with_timing(VZA_2024)
    vza_2025 = read_parquet_with_timing(VZA_2025)
    raa_2024 = read_parquet_with_timing(RAA_2024)
    raa_2025 = read_parquet_with_timing(RAA_2025)
    started = time.perf_counter()
    disease_2024 = pd.read_csv(DISEASE_2024_CLEAN)
    disease_2025 = pd.read_csv(DISEASE_2025_CLEAN)
    log_phase("read clean disease CSVs", started)
    return vza_2024, vza_2025, raa_2024, raa_2025, disease_2024, disease_2025


def pivot_reflectance_features(
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
        index=META_COLS,
        columns="feature",
        values="reflectance",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def summarize_vza_curve(vza: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = META_COLS + ["band_name"]
    for keys, group in vza.groupby(group_cols, sort=False):
        group = group.dropna(subset=["reflectance", "vza_midpoint"]).sort_values("vza_midpoint")
        if group.empty:
            continue
        x = group["vza_midpoint"].to_numpy(float)
        y = group["reflectance"].to_numpy(float)
        low = group[group["vza_midpoint"] <= 20]["reflectance"]
        high = group[group["vza_midpoint"] >= 45]["reflectance"]
        nadir_idx = (group["vza_midpoint"] - group["vza_midpoint"].min()).abs().argmin()
        nadir = float(group.iloc[nadir_idx]["reflectance"])
        off_nadir = group.loc[group["vza_midpoint"] > group["vza_midpoint"].min(), "reflectance"]
        off_nadir_mean = float(off_nadir.mean()) if not off_nadir.empty else np.nan
        high_minus_low = (
            float(high.mean() - low.mean()) if not low.empty and not high.empty else np.nan
        )
        values = {
            "nadir": nadir,
            "mean": float(np.mean(y)),
            "std": float(np.std(y, ddof=0)),
            "slope": float(np.polyfit(x, y, 1)[0]) if len(np.unique(x)) >= 2 else np.nan,
            "curvature": float(np.polyfit(x, y, 2)[0]) if len(np.unique(x)) >= 3 else np.nan,
            "angular_range": float(np.max(y) - np.min(y)),
            "high_minus_low": high_minus_low,
            "relative_high_minus_low": (
                high_minus_low / nadir if np.isfinite(high_minus_low) and nadir != 0 else np.nan
            ),
            "off_nadir_minus_nadir": (
                off_nadir_mean - nadir if np.isfinite(off_nadir_mean) else np.nan
            ),
            "off_nadir_ratio_nadir": (
                off_nadir_mean / nadir if np.isfinite(off_nadir_mean) and nadir != 0 else np.nan
            ),
        }
        base = dict(zip(group_cols[:-1], keys[:-1], strict=False))
        band = clean_token(keys[-1])
        for metric, value in values.items():
            row = base.copy()
            row["feature"] = f"vza_compact__{band}__{metric}"
            row["value"] = value
            rows.append(row)
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
    return pivot


def summarize_geometry_contrasts(raa: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = META_COLS + ["band_name"]
    for keys, group in raa.groupby(group_cols, sort=False):
        group = group.dropna(subset=["reflectance"]).copy()
        if group.empty:
            continue
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
        base = dict(zip(group_cols[:-1], keys[:-1], strict=False))
        band = clean_token(keys[-1])
        for metric, value in values.items():
            row = base.copy()
            row["feature"] = f"geometry_compact__{band}__{metric}"
            row["value"] = value
            rows.append(row)
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
    return pivot


def merge_feature_frames(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return left.merge(right, on=META_COLS, how="outer")


def merge_many_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=META_COLS)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merge_feature_frames(merged, frame)
    return merged


def filter_reliable_angular_bins(raa: pd.DataFrame) -> pd.DataFrame:
    required = {"n_pixels", "n_images"}
    if not required.issubset(raa.columns):
        return raa.copy()
    return raa[
        (raa["n_pixels"] >= MIN_ANGULAR_BIN_PIXELS) & (raa["n_images"] >= MIN_ANGULAR_BIN_IMAGES)
    ].copy()


def build_geometry_feature_sets(
    vza_2024: pd.DataFrame,
    vza_2025: pd.DataFrame,
    raa_2024: pd.DataFrame,
    raa_2025: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    started = time.perf_counter()
    features: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for year_label, vza, raa in [
        ("2024", vza_2024, raa_2024),
        ("2025", vza_2025, raa_2025),
    ]:
        logging.info(
            "%s support: VZA rows=%d, RAA rows=%d, plot-weeks=%d",
            year_label,
            len(vza),
            len(raa),
            raa[["plot_id", "week"]].drop_duplicates().shape[0],
        )

    def one_year(vza: pd.DataFrame, raa: pd.DataFrame) -> dict[str, pd.DataFrame]:
        nadir_idx = (
            vza.sort_values("vza_midpoint")
            .groupby(["year", "week", "plot_id", "band_name"], as_index=False)
            .head(1)
            .index
        )
        nadir = vza.loc[nadir_idx].copy()
        vza_compact = summarize_vza_curve(vza)
        geometry_compact = summarize_geometry_contrasts(raa)
        angular_shape = merge_feature_frames(vza_compact, geometry_compact)
        raa_reliable = filter_reliable_angular_bins(raa)
        vza_features = pivot_reflectance_features(vza, "vza", ["vza_class"])
        raa_features = pivot_reflectance_features(raa, "vza_raa", ["vza_class", "raa_class"])
        phase_features = pivot_reflectance_features(raa, "vza_phase", ["vza_class", "phase_class"])
        raa_reliable_features = pivot_reflectance_features(
            raa_reliable, "vza_raa_reliable", ["vza_class", "raa_class"]
        )
        phase_reliable_features = pivot_reflectance_features(
            raa_reliable, "vza_phase_reliable", ["vza_class", "phase_class"]
        )
        return {
            "nadir_from_vza": pivot_reflectance_features(nadir, "nadir", []),
            "multiangular_vza": vza_features,
            "multiangular_vza_compact": vza_compact,
            "multiangular_geometry_compact": geometry_compact,
            "multiangular_angular_shape": angular_shape,
            "multiangular_vza_raa": raa_features,
            "multiangular_vza_phase": phase_features,
            "multiangular_vza_raa_reliable": raa_reliable_features,
            "multiangular_vza_phase_reliable": phase_reliable_features,
            "multiangular_vza_raa_phase": merge_many_feature_frames(
                [vza_features, raa_features, phase_features]
            ),
            "multiangular_vza_raa_phase_reliable": merge_many_feature_frames(
                [vza_features, raa_reliable_features, phase_reliable_features]
            ),
        }

    train_sets = one_year(vza_2024, raa_2024)
    test_sets = one_year(vza_2025, raa_2025)
    for name in train_sets:
        features[name] = (train_sets[name], test_sets[name])
        n_train_features = len([c for c in train_sets[name].columns if c not in META_COLS])
        n_test_features = len([c for c in test_sets[name].columns if c not in META_COLS])
        logging.info(
            "%s: train=%s rows/%s features, test=%s rows/%s features",
            name,
            len(train_sets[name]),
            n_train_features,
            len(test_sets[name]),
            n_test_features,
        )
    log_phase("build VZA/RAA feature sets", started)
    return features


def evaluate_geometry_feature_sets(
    features: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    results: list[dict[str, object]] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    selections: list[pd.DataFrame] = []
    model_started = time.perf_counter()

    feature_order = [
        "nadir_from_vza",
        "multiangular_vza",
        "multiangular_vza_compact",
        "multiangular_geometry_compact",
        "multiangular_angular_shape",
        "multiangular_vza_raa",
        "multiangular_vza_phase",
        "multiangular_vza_raa_reliable",
        "multiangular_vza_phase_reliable",
        "multiangular_vza_raa_phase",
        "multiangular_vza_raa_phase_reliable",
    ]
    for feature_set in feature_order:
        train_features, test_features = features[feature_set]
        train = current_severity.build_current_model_table(train_features, disease_2024)
        test = current_severity.build_current_model_table(test_features, disease_2025)
        logging.info("%s current model table: train=%d test=%d", feature_set, len(train), len(test))

        result, pred, selection = current_severity.current_hurdle_stability_topk_model(
            train, test, feature_set, top_k=30, log_positive=False
        )
        results.append(result)
        predictions[(result["model"], feature_set)] = pred
        selections.append(selection)

        result, pred, selection = current_severity.current_hurdle_stability_topk_model(
            train, test, feature_set, top_k=50, log_positive=False
        )
        results.append(result)
        predictions[(result["model"], feature_set)] = pred
        selections.append(selection)

        result, pred = residual_pipeline.fit_hurdle_model(train, test, feature_set)
        results.append(result)
        predictions[(result["model"], feature_set)] = pred

    log_phase("fit current-severity geometry models", model_started)
    results_df = pd.DataFrame(results).sort_values(["rmse", "model", "feature_set"])
    selections_df = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    return results_df, predictions, selections_df


def score_prediction_frame(
    predictions: pd.DataFrame,
    model: str,
    feature_set: str,
    source: str,
    n_features: float = math.nan,
) -> dict[str, object]:
    y = predictions["y_true"].to_numpy(float)
    pred = predictions["y_pred"].to_numpy(float)
    return {
        "model": model,
        "feature_set": feature_set,
        "source": source,
        "n_test": len(predictions),
        "n_features": n_features,
        "rmse": math.sqrt(float(np.mean((y - pred) ** 2))),
        "mae": float(np.mean(np.abs(y - pred))),
        "r2": current_severity.r2_score(y, pred) if len(np.unique(y)) > 1 else math.nan,
        "spearman": residual_pipeline.safe_spearman(y, pred),
    }


def context_prediction_paths() -> dict[tuple[str, str], Path]:
    return {
        (
            "current_hurdle_top20_raw_positive",
            "compact_anomaly_nadir",
        ): CURRENT_RESULTS_DIR
        / "predictions/severity_predictions_current_hurdle_top20_raw_positive_compact_anomaly_nadir_spectral_plus_week.csv",
        (
            "current_hurdle_stability_top50_raw_positive",
            "compact_anomaly_multiangular",
        ): CURRENT_RESULTS_DIR
        / "predictions/severity_predictions_current_hurdle_stability_top50_raw_positive_compact_anomaly_multiangular_spectral_plus_week.csv",
        (
            "hurdle_probability_times_severity",
            "compact_anomaly_multiangular",
        ): CURRENT_RESULTS_DIR
        / "predictions/severity_predictions_hurdle_probability_times_severity_compact_anomaly_multiangular_spectral_plus_week.csv",
        (
            "current_hurdle_stability_top30_raw_positive",
            "curve_embedding_fpca_only",
        ): CURVE_RESULTS_DIR
        / "predictions/severity_predictions_current_hurdle_stability_top30_raw_positive_curve_embedding_fpca_only_spectral_plus_week.csv",
    }


def load_context_predictions() -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for key, path in context_prediction_paths().items():
        if path.exists():
            out[key] = pd.read_csv(path)
            logging.info("Loaded context prediction %s from %s", key, path)
        else:
            logging.warning("Missing context prediction %s at %s", key, path)
    return out


def make_week_gate_prediction(
    early: pd.DataFrame,
    late: pd.DataFrame,
    model: str,
    feature_set: str,
) -> pd.DataFrame:
    key_cols = ["plot_id", "predictor_week", "target_week", "y_true"]
    merged = early[key_cols + ["y_pred"]].merge(
        late[key_cols + ["y_pred"]],
        on=key_cols,
        suffixes=("_early", "_late"),
        how="inner",
    )
    week = merged["target_week"].to_numpy(int)
    pred = np.where(
        week <= 0, 0.0, np.where(week <= 3, merged["y_pred_early"], merged["y_pred_late"])
    )
    out = merged[key_cols].copy()
    out["model"] = model
    out["feature_set"] = feature_set
    out["covariates"] = COVARIATES
    out["y_pred"] = pred
    return out[
        [
            "plot_id",
            "predictor_week",
            "target_week",
            "model",
            "feature_set",
            "covariates",
            "y_true",
            "y_pred",
        ]
    ]


def build_week_gate_fusions(
    context_predictions: dict[tuple[str, str], pd.DataFrame],
    geometry_predictions: dict[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    fusion_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    curve_key = ("current_hurdle_stability_top30_raw_positive", "curve_embedding_fpca_only")
    compact_all_key = ("hurdle_probability_times_severity", "compact_anomaly_multiangular")
    compact_top50_key = (
        "current_hurdle_stability_top50_raw_positive",
        "compact_anomaly_multiangular",
    )
    if curve_key not in context_predictions:
        return pd.DataFrame(), {}

    fixed_late_keys = [
        (compact_all_key, "week_gate_curve_w3_compact_all_w5"),
        (compact_top50_key, "week_gate_curve_w3_compact_top50_w5"),
    ]
    for late_key, feature_set in fixed_late_keys:
        if late_key not in context_predictions:
            continue
        pred = make_week_gate_prediction(
            context_predictions[curve_key],
            context_predictions[late_key],
            "reflectance_week_gate",
            feature_set,
        )
        fusion_predictions[("reflectance_week_gate", feature_set)] = pred
        rows.append(
            score_prediction_frame(pred, "reflectance_week_gate", feature_set, "context_fusion")
        )

    for late_key, late_pred in geometry_predictions.items():
        model, feature = late_key
        if feature == "nadir_from_vza":
            continue
        feature_set = f"week_gate_curve_w3_{feature}_w5_{residual_pipeline.safe_filename(model)}"
        pred = make_week_gate_prediction(
            context_predictions[curve_key],
            late_pred,
            "geometry_week_gate",
            feature_set,
        )
        fusion_predictions[("geometry_week_gate", feature_set)] = pred
        rows.append(
            score_prediction_frame(pred, "geometry_week_gate", feature_set, "geometry_fusion")
        )
    fusion_df = pd.DataFrame(rows).sort_values(["rmse", "feature_set"]) if rows else pd.DataFrame()
    return fusion_df, fusion_predictions


def paired_delta_vs_baseline(
    all_predictions: dict[tuple[str, str], pd.DataFrame],
    baseline_key: tuple[str, str],
) -> pd.DataFrame:
    if baseline_key not in all_predictions:
        return pd.DataFrame()
    baseline = all_predictions[baseline_key]
    rng = np.random.default_rng(SEED)
    rows = []
    for key, candidate in all_predictions.items():
        if key == baseline_key:
            continue
        model, feature_set = key
        stats = residual_pipeline.paired_bootstrap_delta_ci(baseline, candidate, rng)
        rows.append(
            {
                "model": model,
                "feature_set": feature_set,
                "baseline_model": baseline_key[0],
                "baseline_feature_set": baseline_key[1],
                "resample_unit": "plot_id",
                **stats,
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_reduction_observed", ascending=False)


def write_prediction_files(predictions: dict[tuple[str, str], pd.DataFrame]) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    for (model, feature_set), frame in predictions.items():
        path = residual_pipeline.prediction_path(model, feature_set, COVARIATES)
        frame.to_csv(path, index=False)


def selected_feature_family(feature: str) -> str:
    if feature.startswith("known__"):
        return "timing"
    if feature.startswith("vza_raa_reliable"):
        return "raa_reliable"
    if feature.startswith("vza_phase_reliable"):
        return "phase_reliable"
    if feature.startswith("vza_raa"):
        return "raa"
    if feature.startswith("vza_phase"):
        return "phase"
    if feature.startswith("vza__"):
        return "vza"
    if feature.startswith("vza_compact"):
        return "vza_compact"
    if feature.startswith("geometry_compact"):
        return "geometry_compact"
    return "other"


def selected_feature_family_counts(selections: pd.DataFrame) -> pd.DataFrame:
    if selections.empty:
        return pd.DataFrame()
    selected = selections[
        selections["selected_for_final_model"]
        & selections["feature_set"].str.contains("vza_raa_phase", regex=False, na=False)
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["family"] = selected["feature"].map(selected_feature_family)
    return (
        selected.groupby(["model", "feature_set", "role", "family"], as_index=False)
        .size()
        .rename(columns={"size": "n_selected"})
        .sort_values(["feature_set", "model", "role", "family"])
    )


def build_report(
    geometry_results: pd.DataFrame,
    context_scores: pd.DataFrame,
    fusion_results: pd.DataFrame,
    delta: pd.DataFrame,
    week_summary: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    report_path = REPORTS_DIR / "current_severity_raa_geometry_fusion_summary.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    top_cols = ["model", "feature_set", "source", "rmse", "mae", "r2", "spearman"]
    geometry_show = geometry_results.copy()
    geometry_show["source"] = "raa_geometry_direct"
    combined_show = pd.concat(
        [
            context_scores[top_cols],
            geometry_show[top_cols],
            (
                fusion_results[top_cols]
                if not fusion_results.empty
                else pd.DataFrame(columns=top_cols)
            ),
        ],
        ignore_index=True,
        sort=False,
    ).sort_values("rmse")

    lines = [
        "## Results: Current Severity with RAA/Phase Geometry",
        "",
        "This experiment tests whether relative azimuth angle (RAA) and phase-angle binned reflectance help the same-week plot-level severity model trained on 2024 and tested on 2025.",
        "",
        "### Best Imaging-Only Candidates",
        "",
        markdown_table(combined_show[top_cols].round(4), max_rows=25),
        "",
        "### Direct RAA/Phase Models",
        "",
        markdown_table(
            geometry_results[
                ["model", "feature_set", "n_test", "n_features", "rmse", "mae", "r2", "spearman"]
            ].round(4),
            max_rows=30,
        ),
        "",
        "### Week-Gated Fusion Candidates",
        "",
        "The fixed fusion uses the curve FPCA model for week 3 and a late-disease model for week 5; week 0 is set to zero because 2024 has no current disease at week 0.",
        "",
        markdown_table(fusion_results[top_cols].round(4), max_rows=30),
        "",
        "### Paired Delta Versus Existing Nadir Baseline",
        "",
        "Positive RMSE reduction means the candidate improves over the existing compact nadir current-severity baseline on matched 2025 plot-week rows.",
        "",
        markdown_table(delta.round(4), max_rows=30),
        "",
        "### Target-Week Diagnostics",
        "",
        markdown_table(week_summary.round(4), max_rows=60),
        "",
        "**Interpretation**: RAA/phase is treated as imaging geometry, not as treatment/cultivar metadata. Any model here remains a current-severity model because predictor and target weeks are identical.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Target: same-week plot-level `ds_plot` stored in the shared severity target column.",
        "- Inputs: VZA reflectance bins, VZA+RAA reflectance bins, VZA+phase reflectance bins, and compact angular geometry summaries.",
        "- Excluded predictors: treatment, cultivar, block, inoculation/design metadata, and disease history.",
        "- Covariate: `known__predictor_week` only.",
        "- Feature selection: grouped 2024-only ElasticNet stability ranking for top-k hurdle models.",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
    ]
    lines.extend([f"- {label}: `{path}`" for label, path in paths.items()])
    persist_report(report_path, lines)
    return report_path


def main() -> None:
    total_started = time.perf_counter()
    configure_reused_pipeline_paths()
    for directory in [RESULTS_DIR, REPORTS_DIR, PREDICTIONS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()

    vza_2024, vza_2025, raa_2024, raa_2025, disease_2024, disease_2025 = read_inputs()
    features = build_geometry_feature_sets(vza_2024, vza_2025, raa_2024, raa_2025)
    geometry_results, geometry_predictions, selections = evaluate_geometry_feature_sets(
        features, disease_2024, disease_2025
    )

    context_predictions = load_context_predictions()
    context_rows = [
        score_prediction_frame(pred, model, feature_set, "existing_context")
        for (model, feature_set), pred in context_predictions.items()
    ]
    context_scores = pd.DataFrame(context_rows).sort_values("rmse")

    fusion_results, fusion_predictions = build_week_gate_fusions(
        context_predictions, geometry_predictions
    )
    write_prediction_files(fusion_predictions)

    all_predictions = {
        **context_predictions,
        **geometry_predictions,
        **fusion_predictions,
    }
    delta = paired_delta_vs_baseline(
        all_predictions,
        ("current_hurdle_top20_raw_positive", "compact_anomaly_nadir"),
    )
    week_summary = current_severity.prediction_week_summary(all_predictions)

    paths = {
        "geometry_model_comparison": RESULTS_DIR / "raa_geometry_model_comparison.csv",
        "context_scores": RESULTS_DIR / "raa_geometry_context_scores.csv",
        "week_gate_fusions": RESULTS_DIR / "raa_geometry_week_gate_fusion_comparison.csv",
        "paired_delta_vs_nadir": RESULTS_DIR / "raa_geometry_delta_vs_existing_nadir.csv",
        "target_week_summary": RESULTS_DIR / "raa_geometry_target_week_summary.csv",
        "selected_features": RESULTS_DIR
        / "raa_geometry_stability_selection_feature_frequencies.csv",
        "selected_family_counts": RESULTS_DIR / "raa_phase_augmented_selected_family_counts.csv",
        "fusion_predictions": PREDICTIONS_DIR,
    }
    geometry_results.to_csv(paths["geometry_model_comparison"], index=False)
    context_scores.to_csv(paths["context_scores"], index=False)
    fusion_results.to_csv(paths["week_gate_fusions"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    week_summary.to_csv(paths["target_week_summary"], index=False)
    selections.to_csv(paths["selected_features"], index=False)
    selected_feature_family_counts(selections).to_csv(paths["selected_family_counts"], index=False)

    report_path = build_report(
        geometry_results,
        context_scores,
        fusion_results,
        delta,
        week_summary,
        paths,
        log_path,
    )
    logging.info("Report: %s", report_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
