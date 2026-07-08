#!/usr/bin/env python3
"""Current-severity subplot models using reflectance-curve embeddings over VZA."""

from __future__ import annotations

import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity import (
    analyze_current_severity_subplots_2024_to_2025 as subplot_pipeline,
)
from scripts.analysis.severity.analyze_cross_year_generalization_2024_to_2025 import (
    SEED,
    TARGET,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    markdown_table,
)

INPUT_ROOT = ROOT / "outputs/current_severity_subplots_2024_to_2025/results"
FEATURE_CACHE_DIR = INPUT_ROOT / "feature_cache"
BASELINE_RESULTS_DIR = INPUT_ROOT
OUTPUT_ROOT = ROOT / "outputs/current_severity_subplots_curve_embeddings_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
LOGS_DIR = ROOT / "outputs/logs"
DISEASE_2024_CLEAN = ROOT / "outputs/disease/clean_disease_scores_2024.csv"
DISEASE_2025_CLEAN = ROOT / "outputs/disease/clean_disease_scores_2025.csv"

ANGLE_GRID = np.array([12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5])
CURVE_BANDS = {"red", "red_edge", "nir", "osavi"}
REFLECTANCE_METRICS = {"mean", "p10", "p25", "iqr", "cv"}
OSAVI_METRICS = {"osavi_mean", "osavi_p10", "osavi_iqr", "osavi_frac_lt_025", "osavi_frac_lt_035"}
N_FPCA_COMPONENTS = 3
META_COLS = ["year", "week", "plot_id", "parent_plot_id", "subplot_id"]


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        LOGS_DIR
        / f"analyze_current_severity_subplots_curve_embeddings_2024_to_2025_{timestamp}.log"
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


def clean_token(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    long_2024 = pd.read_csv(FEATURE_CACHE_DIR / "subplot_distribution_features_long_2024.csv")
    long_2025 = pd.read_csv(FEATURE_CACHE_DIR / "subplot_distribution_features_long_2025.csv")
    disease_2024 = pd.read_csv(DISEASE_2024_CLEAN)
    disease_2025 = pd.read_csv(DISEASE_2025_CLEAN)
    log_phase("read cached subplot long features and disease scores", started)
    return long_2024, long_2025, disease_2024, disease_2025


def filter_curve_rows(long: pd.DataFrame) -> pd.DataFrame:
    data = long.copy()
    data["band_token"] = data["band_name"].map(clean_token)
    data["metric_token"] = data["metric"].map(clean_token)
    keep = data["band_token"].isin(CURVE_BANDS) & (
        data["metric_token"].isin(REFLECTANCE_METRICS) | data["metric_token"].isin(OSAVI_METRICS)
    )
    keep &= (data["band_token"] != "osavi") | data["metric_token"].isin(OSAVI_METRICS)
    keep &= (data["band_token"] == "osavi") | data["metric_token"].isin(REFLECTANCE_METRICS)
    data = data.loc[keep].copy()
    data["curve_group"] = data["band_token"] + "__" + data["metric_token"]
    return data


def curve_pivots(
    long_2024: pd.DataFrame, long_2025: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frames = []
    for data in [filter_curve_rows(long_2024), filter_curve_rows(long_2025)]:
        pivot = data.pivot_table(
            index=META_COLS + ["curve_group"],
            columns="vza_midpoint",
            values="value",
            aggfunc="mean",
        ).reset_index()
        pivot.columns.name = None
        angle_cols = [
            col for col in pivot.columns if isinstance(col, float) or isinstance(col, int)
        ]
        pivot = pivot.rename(columns={angle: f"angle_{float(angle):04.1f}" for angle in angle_cols})
        frames.append(pivot)
    groups = sorted(set(frames[0]["curve_group"]).intersection(frames[1]["curve_group"]))
    return frames[0], frames[1], groups


def shape_features(values: np.ndarray, angles: np.ndarray) -> dict[str, np.ndarray]:
    centered_x = angles - float(np.mean(angles))
    denom = float(np.sum(centered_x**2))
    gradient = np.gradient(values, angles, axis=1)
    curvature = np.gradient(gradient, angles, axis=1)
    first = values[:, 0]
    last = values[:, -1]
    return {
        "auc": np.trapezoid(values, x=angles, axis=1),
        "range": np.max(values, axis=1) - np.min(values, axis=1),
        "std": np.std(values, axis=1),
        "offnadir_minus_nadir": last - first,
        "relative_offnadir_minus_nadir": np.divide(
            last - first,
            first,
            out=np.full_like(first, np.nan, dtype=float),
            where=np.isfinite(first) & (first != 0),
        ),
        "linear_slope": np.sum((values - values.mean(axis=1, keepdims=True)) * centered_x, axis=1)
        / denom,
        "mean_abs_derivative": np.mean(np.abs(gradient), axis=1),
        "mean_abs_curvature": np.mean(np.abs(curvature), axis=1),
    }


def fit_group_embedding(
    train_group: pd.DataFrame,
    test_group: pd.DataFrame,
    group: str,
    angle_cols: list[str],
    angles: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train_meta = train_group[META_COLS].reset_index(drop=True)
    test_meta = test_group[META_COLS].reset_index(drop=True)
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(train_group[angle_cols])
    test_values = imputer.transform(test_group[angle_cols])

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)
    n_components = min(N_FPCA_COMPONENTS, train_scaled.shape[1], train_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=SEED)
    train_scores = pca.fit_transform(train_scaled)
    test_scores = pca.transform(test_scaled)

    spline = SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation="continue")
    basis = spline.fit_transform(angles.reshape(-1, 1))
    pinv = np.linalg.pinv(basis)
    train_coef = train_values @ pinv.T
    test_coef = test_values @ pinv.T

    prefix = f"subplot_curve__{group}"
    train_features = train_meta.copy()
    test_features = test_meta.copy()
    for idx in range(train_coef.shape[1]):
        train_features[f"{prefix}__spline_coef_{idx:02d}"] = train_coef[:, idx]
        test_features[f"{prefix}__spline_coef_{idx:02d}"] = test_coef[:, idx]
    for idx in range(n_components):
        train_features[f"{prefix}__fpca_{idx + 1}"] = train_scores[:, idx]
        test_features[f"{prefix}__fpca_{idx + 1}"] = test_scores[:, idx]
    for name, values in shape_features(train_values, angles).items():
        train_features[f"{prefix}__{name}"] = values
    for name, values in shape_features(test_values, angles).items():
        test_features[f"{prefix}__{name}"] = values

    audit = {
        "curve_group": group,
        "n_train_curves": len(train_group),
        "n_test_curves": len(test_group),
        "n_angles": len(angle_cols),
        "n_spline_coefficients": train_coef.shape[1],
        "n_fpca_components": n_components,
        "fpca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }
    return train_features, test_features, audit


def build_curve_features(
    long_2024: pd.DataFrame, long_2025: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    train_pivot, test_pivot, groups = curve_pivots(long_2024, long_2025)
    train_frames = []
    test_frames = []
    audit_rows = []
    angle_cols = [f"angle_{angle:04.1f}" for angle in ANGLE_GRID]
    for group in groups:
        train_group = train_pivot.loc[train_pivot["curve_group"].eq(group)].copy()
        test_group = test_pivot.loc[test_pivot["curve_group"].eq(group)].copy()
        available = [
            col for col in angle_cols if col in train_group.columns and col in test_group.columns
        ]
        if len(available) < 5:
            continue
        angles = np.array([float(col.replace("angle_", "")) for col in available])
        train_features, test_features, audit = fit_group_embedding(
            train_group, test_group, group, available, angles
        )
        train_frames.append(train_features)
        test_frames.append(test_features)
        audit_rows.append(audit)
    if not train_frames:
        raise RuntimeError("No subplot curve features were created.")

    def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        merged = frames[0]
        for frame in frames[1:]:
            feature_cols = [col for col in frame.columns if col not in META_COLS]
            merged = merged.merge(frame[META_COLS + feature_cols], on=META_COLS, how="outer")
        return merged

    train = merge_frames(train_frames)
    test = merge_frames(test_frames)
    audit = pd.DataFrame(audit_rows)
    log_phase("build subplot curve embedding features", started)
    logging.info(
        "subplot curve features: train=%d test=%d features=%d",
        len(train),
        len(test),
        len(train.columns) - len(META_COLS),
    )
    return train, test, audit


def feature_variants(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    feature_cols = [col for col in train.columns if col not in META_COLS]
    variants = {
        "subplot_curve_all": feature_cols,
        "subplot_curve_fpca_shape": [col for col in feature_cols if "__spline_coef_" not in col],
        "subplot_curve_fpca_only": [col for col in feature_cols if "__fpca_" in col],
        "subplot_curve_shape_only": [
            col for col in feature_cols if "__fpca_" not in col and "__spline_coef_" not in col
        ],
    }
    return {
        name: (train[META_COLS + cols].copy(), test[META_COLS + cols].copy())
        for name, cols in variants.items()
    }


def plot_level_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["model", "feature_set"]):
        plot_predictions = group.groupby(["week", "parent_plot_id"], as_index=False).agg(
            y_true=(TARGET, "first"),
            y_pred=("y_pred", "mean"),
            n_subplots=("subplot_id", "nunique"),
        )
        y = plot_predictions["y_true"].to_numpy(float)
        pred = plot_predictions["y_pred"].to_numpy(float)
        rows.append(
            {
                "model": keys[0],
                "feature_set": keys[1],
                "n_parent_plot_weeks": len(plot_predictions),
                "mean_subplots_per_plot_week": plot_predictions["n_subplots"].mean(),
                **subplot_pipeline.score(y, pred),
            }
        )
    return pd.DataFrame(rows)


def baseline_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    model_path = BASELINE_RESULTS_DIR / "subplot_current_severity_model_comparison.csv"
    plot_path = BASELINE_RESULTS_DIR / "subplot_current_severity_plot_level_model_comparison.csv"
    model = pd.read_csv(model_path) if model_path.exists() else pd.DataFrame()
    plot = pd.read_csv(plot_path) if plot_path.exists() else pd.DataFrame()
    if not model.empty:
        model["source"] = "existing_subplot_analysis"
    if not plot.empty:
        plot["source"] = "existing_subplot_analysis"
    return model, plot


def write_report(
    results: pd.DataFrame,
    plot_results: pd.DataFrame,
    context: pd.DataFrame,
    plot_context: pd.DataFrame,
    audit: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "subplot_curve_embedding_current_severity_summary.md"
    lines = [
        "## Results: Current Severity With 20-Subplot Curve Embeddings",
        "",
        "Each subplot gets VZA-curve embeddings from band/metric reflectance curves. The disease target is still parent plot-level, so plot-level scoring averages subplot predictions back to each parent plot-week.",
        "",
        "### Subplot-Level Model Comparison",
        "",
        markdown_table(results.round(4), max_rows=30),
        "",
        "### Plot-Level Model Comparison",
        "",
        markdown_table(plot_results.round(4), max_rows=30),
        "",
        "### Existing Subplot Baselines",
        "",
        markdown_table(context.round(4), max_rows=12),
        "",
        "### Existing Plot-Level Baselines From Subplot Predictions",
        "",
        markdown_table(plot_context.round(4), max_rows=12),
        "",
        "### Curve Embedding Audit",
        "",
        markdown_table(audit.round(4), max_rows=30),
        "",
        "### Reproducibility",
        "",
        "- Target: same-week parent plot-level `ds_plot` assigned to subplots.",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Curve domain: VZA midpoints 12.5 to 52.5 degrees.",
        "- Embeddings: B-spline coefficients, FPCA scores fit on 2024 subplots only, and derivative/shape descriptors.",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
    ]
    lines.extend([f"- {label}: `{path}`" for label, path in paths.items()])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    total = time.perf_counter()
    for directory in [RESULTS_DIR, REPORTS_DIR, PREDICTIONS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()
    long_2024, long_2025, disease_2024, disease_2025 = read_inputs()
    train_curve, test_curve, audit = build_curve_features(long_2024, long_2025)
    variants = feature_variants(train_curve, test_curve)

    results = []
    preds = []
    model_t0 = time.perf_counter()
    for feature_set, (train_features, test_features) in variants.items():
        train = subplot_pipeline.model_table(train_features, disease_2024)
        test = subplot_pipeline.model_table(test_features, disease_2025)
        logging.info("%s model table: train=%d test=%d", feature_set, len(train), len(test))
        for fit in [
            subplot_pipeline.phenology_floor_ridge,
            subplot_pipeline.hurdle_model,
            subplot_pipeline.xgboost_hurdle_model,
        ]:
            result, pred = fit(train, test, feature_set)
            results.append(result)
            preds.append(pred)
    log_phase("fit subplot curve models", model_t0)

    results_df = pd.DataFrame(results).sort_values(["rmse", "model"])
    predictions = pd.concat(preds, ignore_index=True)
    plot_results = plot_level_summary(predictions).sort_values(["rmse", "model"])
    baseline_model, baseline_plot = baseline_rows()
    paths = {
        "model_comparison": RESULTS_DIR / "subplot_curve_embedding_model_comparison.csv",
        "plot_level_model_comparison": RESULTS_DIR
        / "subplot_curve_embedding_plot_level_model_comparison.csv",
        "predictions": RESULTS_DIR / "subplot_curve_embedding_predictions.csv",
        "curve_embedding_audit": RESULTS_DIR / "subplot_curve_embedding_audit.csv",
        "context_model_comparison": RESULTS_DIR
        / "subplot_curve_embedding_with_context_baselines.csv",
        "context_plot_level_comparison": RESULTS_DIR
        / "subplot_curve_embedding_plot_level_with_context_baselines.csv",
    }
    context = pd.concat([results_df, baseline_model], ignore_index=True, sort=False).sort_values(
        ["rmse", "model"]
    )
    plot_context = pd.concat(
        [plot_results, baseline_plot], ignore_index=True, sort=False
    ).sort_values(["rmse", "model"])
    results_df.to_csv(paths["model_comparison"], index=False)
    plot_results.to_csv(paths["plot_level_model_comparison"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    audit.to_csv(paths["curve_embedding_audit"], index=False)
    context.to_csv(paths["context_model_comparison"], index=False)
    plot_context.to_csv(paths["context_plot_level_comparison"], index=False)
    report = write_report(
        results_df, plot_results, baseline_model, baseline_plot, audit, paths, log_path
    )
    logging.info("Report: %s", report)
    log_phase("total", total)


if __name__ == "__main__":
    main()
