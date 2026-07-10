#!/usr/bin/env python3
"""Current severity from healthy-normalized angular reflectance signatures.

Hypothesis: disease changes canopy/leaf-angle structure, and that structure is
visible as a deviation from the healthy angular reflectance curve. This script
builds a fixed imaging-only feature set:

1. For each plot-week, band, and reflectance-distribution metric, form the VZA
   reflectance curve.
2. Normalize the curve so brightness/level is suppressed.
3. Estimate a healthy 2024 angular baseline from disease-free plot-weeks.
4. Express each curve as a residual from that healthy baseline.
5. Fit one hurdle severity model trained on 2024 and tested on 2025.

No treatment, cultivar, block, inoculation, or disease-history predictors are
used.
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
from sklearn.impute import SimpleImputer

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
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_curve_embeddings_2024_to_2025 import (
    ANGLE_GRID,
    CURVE_BANDS,
    INPUT_RESULTS_DIR,
    OSAVI_METRICS,
    REFLECTANCE_METRICS,
    clean_band_name,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    clean_token,
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/healthy_angular_signature_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

DISEASE_2024_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2024.csv"
DISEASE_2025_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2025.csv"
CURRENT_RESULTS_DIR = ROOT / "outputs/runs/analysis/severity/current/2024_to_2025/results"
CURVE_RESULTS_DIR = ROOT / "outputs/runs/analysis/severity/current/curve_embeddings_2024_to_2025/results"

FEATURE_SET = "healthy_angular_residual_signature"
COVARIATES = "spectral_plus_week"
META_COLS = ["year", "week", "plot_id", "cult", "trt"]
MIN_GROUP_ANGLES = 5
MIN_HEALTHY_BASELINE_ROWS = 4
EPS = 1e-4

TARGET = current_severity.TARGET
TARGET_LOG = current_severity.TARGET_LOG
WARNING_TARGET = current_severity.WARNING_TARGET
WARNING_THRESHOLD = current_severity.WARNING_THRESHOLD


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        LOGS_DIR
        / f"analyze_current_severity_healthy_angular_signature_2024_to_2025_{timestamp}.log"
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
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "healthy_angular_signature_manifest.json"


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    long_2024 = pd.read_csv(INPUT_RESULTS_DIR / "distribution_features_long_2024.csv")
    long_2025 = pd.read_csv(INPUT_RESULTS_DIR / "distribution_features_long_2025.csv")
    disease_2024 = pd.read_csv(DISEASE_2024_CLEAN)
    disease_2025 = pd.read_csv(DISEASE_2025_CLEAN)
    log_phase("read cached long features and disease scores", started)
    return long_2024, long_2025, disease_2024, disease_2025


def filter_curve_rows(long: pd.DataFrame) -> pd.DataFrame:
    data = long.copy()
    data["band_token"] = data["band_name"].map(clean_band_name)
    data["metric_token"] = data["metric"].map(clean_token)
    keep = data["band_token"].isin(CURVE_BANDS) & (
        data["metric_token"].isin(REFLECTANCE_METRICS) | data["metric_token"].isin(OSAVI_METRICS)
    )
    keep &= (data["band_token"] != "osavi") | data["metric_token"].isin(OSAVI_METRICS)
    keep &= (data["band_token"] == "osavi") | data["metric_token"].isin(REFLECTANCE_METRICS)
    data = data.loc[keep].copy()
    data["curve_group"] = data["band_token"] + "__" + data["metric_token"]
    return data


def pivot_curves(long: pd.DataFrame) -> pd.DataFrame:
    data = filter_curve_rows(long)
    pivot = data.pivot_table(
        index=META_COLS + ["curve_group"],
        columns="vza_midpoint",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    angle_cols = [col for col in pivot.columns if isinstance(col, int | float)]
    rename = {angle: f"angle_{float(angle):04.1f}" for angle in angle_cols}
    return pivot.rename(columns=rename)


def angle_columns_for_group(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[list[str], np.ndarray]:
    cols = []
    for angle in ANGLE_GRID:
        col = f"angle_{angle:04.1f}"
        if col in train.columns and col in test.columns:
            cols.append(col)
    angles = np.asarray([float(col.replace("angle_", "")) for col in cols], dtype=float)
    return cols, angles


def curve_transforms(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = values[:, [0]]
    scale = np.maximum(np.abs(first), EPS)
    relative = (values - first) / scale
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    std = np.where(std > EPS, std, 1.0)
    shape = (values - mean) / std
    return relative, shape


def healthy_baselines(
    transformed_train: np.ndarray,
    weeks: np.ndarray,
    severity: np.ndarray,
) -> dict[object, np.ndarray]:
    healthy_mask = severity <= 0
    if healthy_mask.sum() < MIN_HEALTHY_BASELINE_ROWS:
        healthy_mask = np.ones(len(severity), dtype=bool)
    global_baseline = np.nanmedian(transformed_train[healthy_mask], axis=0)
    baselines: dict[object, np.ndarray] = {"__global__": global_baseline}
    for week in sorted(np.unique(weeks)):
        week_mask = healthy_mask & (weeks == week)
        if week_mask.sum() >= MIN_HEALTHY_BASELINE_ROWS:
            baselines[int(week)] = np.nanmedian(transformed_train[week_mask], axis=0)
    return baselines


def residualize_to_healthy(
    transformed: np.ndarray,
    weeks: np.ndarray,
    baselines: dict[object, np.ndarray],
) -> np.ndarray:
    fallback = baselines["__global__"]
    row_baselines = np.vstack([baselines.get(int(week), fallback) for week in weeks])
    return transformed - row_baselines


def residual_summary_features(
    residuals: np.ndarray,
    angles: np.ndarray,
    prefix: str,
) -> dict[str, np.ndarray]:
    low = residuals[:, angles <= 25]
    high = residuals[:, angles >= 45]
    centered = angles - angles.mean()
    denom = float(np.sum(centered**2))
    slope = np.sum((residuals - residuals.mean(axis=1, keepdims=True)) * centered, axis=1) / denom
    quadratic = np.column_stack([np.ones_like(angles), centered, centered**2])
    coef = np.linalg.pinv(quadratic) @ residuals.T
    curvature = coef[2, :]
    return {
        f"{prefix}__l2": np.sqrt(np.mean(residuals**2, axis=1)),
        f"{prefix}__max_abs": np.max(np.abs(residuals), axis=1),
        f"{prefix}__signed_auc": np.trapezoid(residuals, x=angles, axis=1),
        f"{prefix}__high_minus_low": high.mean(axis=1) - low.mean(axis=1),
        f"{prefix}__slope": slope,
        f"{prefix}__curvature": curvature,
        f"{prefix}__roughness": np.mean(np.abs(np.diff(residuals, axis=1)), axis=1),
    }


def correlation_projection(
    train_residuals: np.ndarray,
    test_residuals: np.ndarray,
    severity: np.ndarray,
    prefix: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    mean = np.nanmean(train_residuals, axis=0)
    std = np.nanstd(train_residuals, axis=0)
    std = np.where(std > EPS, std, 1.0)
    train_z = (train_residuals - mean) / std
    test_z = (test_residuals - mean) / std
    y = np.log1p(severity)
    y_std = float(np.std(y))
    weights = []
    for idx in range(train_z.shape[1]):
        x = train_z[:, idx]
        if np.std(x) <= EPS or y_std <= EPS:
            weights.append(0.0)
        else:
            value = float(np.corrcoef(x, y)[0, 1])
            weights.append(value if np.isfinite(value) else 0.0)
    weights_arr = np.asarray(weights, dtype=float)
    norm = float(np.linalg.norm(weights_arr))
    if norm <= EPS:
        weights_arr = np.ones_like(weights_arr) / math.sqrt(len(weights_arr))
    else:
        weights_arr = weights_arr / norm
    train_projection = train_z @ weights_arr
    test_projection = test_z @ weights_arr
    audit = {
        f"{prefix}__signature_weight_l1": float(np.sum(np.abs(weights_arr))),
        f"{prefix}__signature_weight_max_abs": float(np.max(np.abs(weights_arr))),
    }
    return (
        {f"{prefix}__severity_signature_projection": train_projection},
        {f"{prefix}__severity_signature_projection": test_projection},
        audit,
    )


def build_group_signature_features(
    train_group: pd.DataFrame,
    test_group: pd.DataFrame,
    angle_cols: list[str],
    angles: np.ndarray,
    disease_2024: pd.DataFrame,
    group: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    disease_target = disease_2024[["plot_id", "week", "ds_plot"]].rename(
        columns={"ds_plot": TARGET}
    )
    train_group = train_group.merge(disease_target, on=["plot_id", "week"], how="left")
    train_group[TARGET] = train_group[TARGET].fillna(0.0)
    train_meta = train_group[META_COLS].reset_index(drop=True)
    test_meta = test_group[META_COLS].reset_index(drop=True)

    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(train_group[angle_cols])
    test_values = imputer.transform(test_group[angle_cols])
    train_relative, train_shape = curve_transforms(train_values)
    test_relative, test_shape = curve_transforms(test_values)

    weeks_train = train_group["week"].to_numpy(int)
    weeks_test = test_group["week"].to_numpy(int)
    severity = train_group[TARGET].to_numpy(float)
    relative_baselines = healthy_baselines(train_relative, weeks_train, severity)
    shape_baselines = healthy_baselines(train_shape, weeks_train, severity)
    train_relative_residual = residualize_to_healthy(
        train_relative, weeks_train, relative_baselines
    )
    test_relative_residual = residualize_to_healthy(test_relative, weeks_test, relative_baselines)
    train_shape_residual = residualize_to_healthy(train_shape, weeks_train, shape_baselines)
    test_shape_residual = residualize_to_healthy(test_shape, weeks_test, shape_baselines)

    token = clean_token(group)
    train_features: dict[str, np.ndarray] = {}
    test_features: dict[str, np.ndarray] = {}
    for transform_name, train_residual, test_residual in [
        ("relative", train_relative_residual, test_relative_residual),
        ("shape", train_shape_residual, test_shape_residual),
    ]:
        prefix = f"ads__{token}__{transform_name}"
        train_features.update(residual_summary_features(train_residual, angles, prefix))
        test_features.update(residual_summary_features(test_residual, angles, prefix))
        train_proj, test_proj, _ = correlation_projection(
            train_residual, test_residual, severity, prefix
        )
        train_features.update(train_proj)
        test_features.update(test_proj)

    audit = {
        "curve_group": group,
        "n_train_curves": len(train_group),
        "n_test_curves": len(test_group),
        "n_angles": len(angle_cols),
        "n_train_healthy_zero_severity": int((severity <= 0).sum()),
        "n_week_specific_relative_baselines": len(relative_baselines) - 1,
        "n_week_specific_shape_baselines": len(shape_baselines) - 1,
        "n_features": len(train_features),
    }
    train_out = pd.concat([train_meta, pd.DataFrame(train_features)], axis=1)
    test_out = pd.concat([test_meta, pd.DataFrame(test_features)], axis=1)
    return train_out, test_out, audit


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0]
    for frame in frames[1:]:
        feature_cols = [col for col in frame.columns if col not in META_COLS]
        merged = merged.merge(frame[META_COLS + feature_cols], on=META_COLS, how="outer")
    return merged


def build_signature_feature_set(
    long_2024: pd.DataFrame,
    long_2025: pd.DataFrame,
    disease_2024: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    train_pivot = pivot_curves(long_2024)
    test_pivot = pivot_curves(long_2025)
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []

    common_groups = sorted(set(train_pivot["curve_group"]).intersection(test_pivot["curve_group"]))
    for group in common_groups:
        train_group = train_pivot[train_pivot["curve_group"].eq(group)].copy()
        test_group = test_pivot[test_pivot["curve_group"].eq(group)].copy()
        angle_cols, angles = angle_columns_for_group(train_group, test_group)
        if len(angle_cols) < MIN_GROUP_ANGLES:
            continue
        train_features, test_features, audit = build_group_signature_features(
            train_group, test_group, angle_cols, angles, disease_2024, group
        )
        train_frames.append(train_features)
        test_frames.append(test_features)
        audit_rows.append(audit)

    if not train_frames:
        raise RuntimeError("No healthy angular signature features were created.")

    train = merge_feature_frames(train_frames)
    test = merge_feature_frames(test_frames)
    audit = pd.DataFrame(audit_rows).sort_values("curve_group")
    n_features = len([col for col in train.columns if col not in META_COLS])
    logging.info(
        "healthy angular signature: train rows=%d test rows=%d feature columns=%d groups=%d",
        len(train),
        len(test),
        n_features,
        len(audit),
    )
    log_phase("build healthy angular signature features", started)
    return train, test, audit


def score_prediction_frame(
    predictions: pd.DataFrame,
    model: str,
    feature_set: str,
    source: str,
) -> dict[str, object]:
    y = predictions["y_true"].to_numpy(float)
    pred = predictions["y_pred"].to_numpy(float)
    return {
        "model": model,
        "feature_set": feature_set,
        "source": source,
        "n_test": len(predictions),
        "rmse": math.sqrt(float(np.mean((y - pred) ** 2))),
        "mae": float(np.mean(np.abs(y - pred))),
        "r2": current_severity.r2_score(y, pred) if len(np.unique(y)) > 1 else math.nan,
        "spearman": residual_pipeline.safe_spearman(y, pred),
    }


def load_context_predictions() -> dict[tuple[str, str], pd.DataFrame]:
    paths = {
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
            "current_hurdle_stability_top30_raw_positive",
            "curve_embedding_fpca_only",
        ): CURVE_RESULTS_DIR
        / "predictions/severity_predictions_current_hurdle_stability_top30_raw_positive_curve_embedding_fpca_only_spectral_plus_week.csv",
    }
    out = {}
    for key, path in paths.items():
        if path.exists():
            out[key] = pd.read_csv(path)
        else:
            logging.warning("Missing context prediction: %s", path)
    return out


def paired_delta_vs_nadir(predictions: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    baseline_key = ("current_hurdle_top20_raw_positive", "compact_anomaly_nadir")
    if baseline_key not in predictions:
        return pd.DataFrame()
    rng = np.random.default_rng(current_severity.SEED)
    rows = []
    baseline = predictions[baseline_key]
    for key, candidate in predictions.items():
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


def apply_zero_week_floor(
    train: pd.DataFrame,
    raw_result: dict[str, object],
    raw_predictions: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    zero_weeks = (
        train.groupby("target_week")[TARGET].max().loc[lambda values: values <= 0].index.to_numpy()
    )
    model = "healthy_angular_signature_hurdle_zero_week_floor"
    predictions = raw_predictions.copy()
    predictions["model"] = model
    if zero_weeks.size:
        predictions.loc[predictions["target_week"].isin(zero_weeks), "y_pred"] = 0.0
    residual_pipeline.save_predictions(predictions, model, FEATURE_SET, COVARIATES)
    result = residual_pipeline.score_predictions(
        predictions,
        int(raw_result["n_train"]),
        int(raw_result["n_features"]),
        model,
        FEATURE_SET,
    )
    result["source"] = "healthy_angular_signature"
    result["fit_time_s"] = raw_result.get("fit_time_s", math.nan)
    result["zero_target_weeks_from_2024"] = ",".join(map(str, zero_weeks.tolist()))
    return result, predictions


def build_report(
    comparison: pd.DataFrame,
    delta: pd.DataFrame,
    week_summary: pd.DataFrame,
    audit: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "healthy_angular_signature_current_severity_summary.md"
    lines = [
        "## Results: Healthy Angular Residual Signature",
        "",
        "This is a fixed hypothesis-driven method: each VZA reflectance curve is normalized, residualized against a healthy 2024 angular baseline, summarized into disease-shape descriptors, and fitted with one hurdle Ridge model.",
        "",
        "### Model Comparison",
        "",
        markdown_table(
            comparison[
                ["model", "feature_set", "source", "n_test", "rmse", "mae", "r2", "spearman"]
            ]
            .round(4)
            .sort_values("rmse"),
            max_rows=20,
        ),
        "",
        "### Paired Delta Versus Existing Nadir",
        "",
        "Positive RMSE reduction means the candidate improves over the existing compact nadir current-severity baseline on matched 2025 plot-week rows.",
        "",
        markdown_table(delta.round(4), max_rows=20),
        "",
        "### Target-Week Diagnostics",
        "",
        markdown_table(week_summary.round(4), max_rows=30),
        "",
        "### Feature Construction Audit",
        "",
        markdown_table(audit.round(4), max_rows=30),
        "",
        "**Interpretation**: This is still a current-severity analysis, not early warning. The method is more defensible than model shopping because the feature transform is fixed by the canopy-angle hypothesis before seeing the 2025 labels.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Target: same-week plot-level `ds_plot`.",
        "- Inputs: VZA curves from cached multiangular reflectance distribution features.",
        "- Healthy baseline: 2024 plot-weeks with same-week `ds_plot == 0`; week-specific baseline when at least four healthy rows are available, otherwise global healthy baseline.",
        "- Excluded predictors: treatment, cultivar, block, inoculation/design metadata, and disease history.",
        "- Model: one hurdle model using logistic disease-presence probability times Ridge positive-severity prediction, with a current-severity zero-week floor when 2024 has no disease for that week.",
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

    long_2024, long_2025, disease_2024, disease_2025 = read_inputs()
    train_features, test_features, audit = build_signature_feature_set(
        long_2024, long_2025, disease_2024
    )
    train = current_severity.build_current_model_table(train_features, disease_2024)
    test = current_severity.build_current_model_table(test_features, disease_2025)
    logging.info("%s current model table: train=%d test=%d", FEATURE_SET, len(train), len(test))

    fit_started = time.perf_counter()
    raw_result, raw_predictions = residual_pipeline.fit_hurdle_model(train, test, FEATURE_SET)
    result, method_predictions = apply_zero_week_floor(train, raw_result, raw_predictions)
    log_phase("fit healthy angular signature model", fit_started)

    context_predictions = load_context_predictions()
    all_predictions = {**context_predictions, (result["model"], FEATURE_SET): method_predictions}
    context_rows = [
        score_prediction_frame(pred, model, feature_set, "existing_context")
        for (model, feature_set), pred in context_predictions.items()
    ]
    comparison = pd.DataFrame([*context_rows, result]).sort_values("rmse")
    delta = paired_delta_vs_nadir(all_predictions)
    week_summary = current_severity.prediction_week_summary(all_predictions)

    paths = {
        "model_comparison": RESULTS_DIR / "healthy_angular_signature_model_comparison.csv",
        "paired_delta_vs_nadir": RESULTS_DIR / "healthy_angular_signature_delta_vs_nadir.csv",
        "target_week_summary": RESULTS_DIR / "healthy_angular_signature_target_week_summary.csv",
        "feature_audit": RESULTS_DIR / "healthy_angular_signature_feature_audit.csv",
        "predictions": residual_pipeline.prediction_path(result["model"], FEATURE_SET, COVARIATES),
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    week_summary.to_csv(paths["target_week_summary"], index=False)
    audit.to_csv(paths["feature_audit"], index=False)
    report_path = build_report(comparison, delta, week_summary, audit, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
