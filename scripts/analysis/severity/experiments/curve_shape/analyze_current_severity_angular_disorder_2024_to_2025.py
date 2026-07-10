#!/usr/bin/env python3
"""Current severity from cross-band angular disorder features.

Hypothesis: disease changes canopy structure in a way that disrupts coherent
multi-band angular reflectance behavior. Healthy sugar beet should preserve a
fairly ordered angular-spectral response; diseased canopy should show:

* cross-band angular decoupling, especially NIR/red-edge/red shape mismatch;
* widening lower-quantile gaps and angular heterogeneity from patchy damage.

The analysis is imaging-only: no cultivar, treatment, block, inoculation,
RAA, or disease-history predictors are used.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    INPUT_RESULTS_DIR,
    clean_band_name,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    clean_token,
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/angular_disorder_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

DISEASE_2024_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2024.csv"
DISEASE_2025_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2025.csv"
CURRENT_RESULTS_DIR = ROOT / "outputs/runs/analysis/severity/current/2024_to_2025/results"
FUNCTIONAL_RESULTS_DIR = (
    ROOT / "outputs/runs/analysis/severity/current/magnitude_shape_functional_2024_to_2025/results"
)

COVARIATES = "spectral_plus_week"
META_COLS = ["year", "week", "plot_id", "cult", "trt"]
TARGET = current_severity.TARGET
SEED = current_severity.SEED
EPS = 1e-4

SPECTRAL_BANDS = ["red", "red_edge", "nir"]
DECOUPLING_METRICS = ["mean", "p10", "p25"]
HETEROGENEITY_BANDS = ["red", "red_edge", "nir", "osavi"]


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_current_severity_angular_disorder_2024_to_2025_{timestamp}.log"
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
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "angular_disorder_manifest.json"


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    long_2024 = pd.read_csv(INPUT_RESULTS_DIR / "distribution_features_long_2024.csv")
    long_2025 = pd.read_csv(INPUT_RESULTS_DIR / "distribution_features_long_2025.csv")
    disease_2024 = pd.read_csv(DISEASE_2024_CLEAN)
    disease_2025 = pd.read_csv(DISEASE_2025_CLEAN)
    log_phase("read cached long features and disease scores", started)
    return long_2024, long_2025, disease_2024, disease_2025


def metric_token(row: pd.Series) -> str:
    band = clean_band_name(row["band_name"])
    metric = clean_token(row["metric"])
    if band == "osavi":
        return metric
    return metric


def filter_rows(long: pd.DataFrame) -> pd.DataFrame:
    data = long.copy()
    data["band_token"] = data["band_name"].map(clean_band_name)
    data["metric_token"] = data.apply(metric_token, axis=1)
    keep_reflectance = data["band_token"].isin(SPECTRAL_BANDS) & data["metric_token"].isin(
        ["mean", "p10", "p25", "iqr", "cv"]
    )
    keep_osavi = data["band_token"].eq("osavi") & data["metric_token"].isin(
        ["osavi_mean", "osavi_p10", "osavi_iqr", "osavi_frac_lt_025", "osavi_frac_lt_035"]
    )
    return data.loc[keep_reflectance | keep_osavi].copy()


def pivot_curves(long: pd.DataFrame) -> pd.DataFrame:
    data = filter_rows(long)
    pivot = data.pivot_table(
        index=META_COLS + ["band_token", "metric_token"],
        columns="vza_midpoint",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    angle_cols = [col for col in pivot.columns if isinstance(col, (int, float))]
    return pivot.rename(columns={angle: f"angle_{float(angle):04.1f}" for angle in angle_cols})


def available_angle_cols(row: pd.Series) -> list[str]:
    cols = [f"angle_{angle:04.1f}" for angle in ANGLE_GRID if f"angle_{angle:04.1f}" in row.index]
    return [col for col in cols if pd.notna(row[col])]


def row_curve(row: pd.Series) -> np.ndarray | None:
    cols = available_angle_cols(row)
    if len(cols) < 5:
        return None
    return row[cols].to_numpy(float)


def shape_curve(values: np.ndarray) -> np.ndarray:
    shift = max(EPS, EPS - float(np.nanmin(values)))
    log_values = np.log(np.clip(values + shift, EPS, None))
    return log_values - log_values[0]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or float(np.std(x)) <= EPS or float(np.std(y)) <= EPS:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else 0.0


def linear_slope(values: np.ndarray) -> float:
    x = ANGLE_GRID[: len(values)]
    x = x - float(np.mean(x))
    denom = float(np.sum(x**2))
    return float(np.sum((values - values.mean()) * x) / denom) if denom > 0 else 0.0


def curve_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    low = values[: max(1, min(3, len(values)))]
    high = values[-max(1, min(3, len(values))) :]
    return {
        f"{prefix}__mean": float(np.mean(values)),
        f"{prefix}__std": float(np.std(values)),
        f"{prefix}__range": float(np.max(values) - np.min(values)),
        f"{prefix}__high_minus_low": float(np.mean(high) - np.mean(low)),
        f"{prefix}__slope": linear_slope(values),
        f"{prefix}__max_abs": float(np.max(np.abs(values))),
    }


def curves_for_plot_week(pivot: pd.DataFrame) -> dict[tuple, dict[tuple[str, str], np.ndarray]]:
    out: dict[tuple, dict[tuple[str, str], np.ndarray]] = {}
    for _, row in pivot.iterrows():
        key = tuple(row[col] for col in META_COLS)
        values = row_curve(row)
        if values is None:
            continue
        out.setdefault(key, {})[(str(row["band_token"]), str(row["metric_token"]))] = values
    return out


def cross_band_decoupling_features(curves: dict[tuple[str, str], np.ndarray]) -> dict[str, float]:
    features: dict[str, float] = {}
    pairs = [("nir", "red_edge"), ("nir", "red"), ("red_edge", "red")]
    for metric in DECOUPLING_METRICS:
        shapes = {}
        for band in SPECTRAL_BANDS:
            curve = curves.get((band, metric))
            if curve is not None:
                shapes[band] = shape_curve(curve)
        for left, right in pairs:
            if left not in shapes or right not in shapes:
                continue
            diff = shapes[left] - shapes[right]
            prefix = f"decouple__{metric}__{left}_vs_{right}"
            features[f"{prefix}__corr"] = safe_corr(shapes[left], shapes[right])
            features[f"{prefix}__one_minus_corr"] = 1.0 - features[f"{prefix}__corr"]
            features[f"{prefix}__rmse_shape_diff"] = float(np.sqrt(np.mean(diff**2)))
            features[f"{prefix}__max_abs_shape_diff"] = float(np.max(np.abs(diff)))
            features[f"{prefix}__high_angle_shape_diff"] = float(np.mean(diff[-3:]))

    mean_curves = {band: curves.get((band, "mean")) for band in SPECTRAL_BANDS}
    if all(curve is not None for curve in mean_curves.values()):
        nir = mean_curves["nir"]
        red_edge = mean_curves["red_edge"]
        red = mean_curves["red"]
        nir_red_edge = nir - red_edge
        red_edge_red = red_edge - red
        for name, diff in [
            ("nir_minus_red_edge", nir_red_edge),
            ("red_edge_minus_red", red_edge_red),
        ]:
            features.update(curve_summary(diff, f"order__{name}"))
            features[f"order__{name}__min_margin"] = float(np.min(diff))
            features[f"order__{name}__fraction_violation"] = float(np.mean(diff <= 0))
    return features


def lower_quantile_heterogeneity_features(
    curves: dict[tuple[str, str], np.ndarray]
) -> dict[str, float]:
    features: dict[str, float] = {}
    for band in ["red", "red_edge", "nir"]:
        mean = curves.get((band, "mean"))
        if mean is None:
            continue
        for quantile in ["p10", "p25"]:
            q = curves.get((band, quantile))
            if q is None:
                continue
            gap = mean - q
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_gap = np.divide(gap, np.maximum(np.abs(mean), EPS))
            features.update(curve_summary(gap, f"weak_tail__{band}__mean_minus_{quantile}"))
            features.update(
                curve_summary(rel_gap, f"weak_tail__{band}__relative_mean_minus_{quantile}")
            )
        for metric in ["iqr", "cv"]:
            hetero = curves.get((band, metric))
            if hetero is not None:
                features.update(curve_summary(hetero, f"heterogeneity__{band}__{metric}"))

    for metric in ["osavi_iqr", "osavi_frac_lt_025", "osavi_frac_lt_035"]:
        curve = curves.get(("osavi", metric))
        if curve is not None:
            features.update(curve_summary(curve, f"heterogeneity__osavi__{metric}"))
    mean_osavi = curves.get(("osavi", "osavi_mean"))
    p10_osavi = curves.get(("osavi", "osavi_p10"))
    if mean_osavi is not None and p10_osavi is not None:
        features.update(curve_summary(mean_osavi - p10_osavi, "weak_tail__osavi__mean_minus_p10"))
    return features


def build_feature_sets_from_long(
    long_2024: pd.DataFrame, long_2025: pd.DataFrame
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    started = time.perf_counter()
    pivot_2024 = pivot_curves(long_2024)
    pivot_2025 = pivot_curves(long_2025)
    curves_2024 = curves_for_plot_week(pivot_2024)
    curves_2025 = curves_for_plot_week(pivot_2025)

    def build_frames(
        curve_map: dict[tuple, dict[tuple[str, str], np.ndarray]]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        decoupling_rows = []
        heterogeneity_rows = []
        combined_rows = []
        for key, curves in curve_map.items():
            meta = dict(zip(META_COLS, key, strict=False))
            decoupling = cross_band_decoupling_features(curves)
            heterogeneity = lower_quantile_heterogeneity_features(curves)
            decoupling_rows.append({**meta, **decoupling})
            heterogeneity_rows.append({**meta, **heterogeneity})
            combined_rows.append({**meta, **decoupling, **heterogeneity})
        return (
            pd.DataFrame(decoupling_rows),
            pd.DataFrame(heterogeneity_rows),
            pd.DataFrame(combined_rows),
        )

    train_dec, train_het, train_combined = build_frames(curves_2024)
    test_dec, test_het, test_combined = build_frames(curves_2025)
    feature_sets = {
        "angular_disorder_cross_band_decoupling": (train_dec, test_dec),
        "angular_disorder_lower_quantile_heterogeneity": (train_het, test_het),
        "angular_disorder_combined": (train_combined, test_combined),
    }
    audit_rows = []
    for name, (train, test) in feature_sets.items():
        n_train_features = len([col for col in train.columns if col not in META_COLS])
        n_test_features = len([col for col in test.columns if col not in META_COLS])
        audit_rows.append(
            {
                "feature_set": name,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_features": n_train_features,
                "test_features": n_test_features,
            }
        )
        logging.info(
            "%s: train=%d rows/%d features test=%d rows/%d features",
            name,
            len(train),
            n_train_features,
            len(test),
            n_test_features,
        )
    log_phase("build angular disorder feature sets", started)
    return feature_sets, pd.DataFrame(audit_rows)


def apply_zero_week_floor(
    train: pd.DataFrame,
    result: dict[str, object],
    predictions: pd.DataFrame,
    feature_set: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    zero_weeks = (
        train.groupby("target_week")[TARGET].max().loc[lambda values: values <= 0].index.to_numpy()
    )
    model = f"{result['model']}_zero_week_floor"
    floor_pred = predictions.copy()
    floor_pred["model"] = model
    if zero_weeks.size:
        floor_pred.loc[floor_pred["target_week"].isin(zero_weeks), "y_pred"] = 0.0
    residual_pipeline.save_predictions(floor_pred, model, feature_set, COVARIATES)
    floor_result = residual_pipeline.score_predictions(
        floor_pred,
        int(result["n_train"]),
        int(result["n_features"]),
        model,
        feature_set,
    )
    for key in [
        "fit_time_s",
        "classifier_features",
        "regressor_features",
        "feature_selection_strategy",
        "train_in_sample_rmse",
        "train_grouped_oof_rmse",
        "train_oof_minus_in_sample_rmse",
    ]:
        floor_result[key] = result.get(key, math.nan)
    floor_result["external_minus_oof_rmse"] = (
        floor_result["rmse"] - float(result.get("train_grouped_oof_rmse", math.nan))
        if pd.notna(result.get("train_grouped_oof_rmse", math.nan))
        else math.nan
    )
    floor_result["source"] = "angular_disorder"
    floor_result["zero_target_weeks_from_2024"] = ",".join(map(str, zero_weeks.tolist()))
    return floor_result, floor_pred


def evaluate_feature_sets(
    feature_sets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    results = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    selections = []
    started = time.perf_counter()
    for feature_set, (train_features, test_features) in feature_sets.items():
        train = current_severity.build_current_model_table(train_features, disease_2024)
        test = current_severity.build_current_model_table(test_features, disease_2025)
        logging.info("%s current model table: train=%d test=%d", feature_set, len(train), len(test))
        result, pred, selection = current_severity.current_hurdle_stability_topk_model(
            train,
            test,
            feature_set,
            top_k=50,
            log_positive=False,
        )
        result["source"] = "angular_disorder"
        results.append(result)
        predictions[(result["model"], feature_set)] = pred
        selections.append(selection)
        floor_result, floor_pred = apply_zero_week_floor(train, result, pred, feature_set)
        results.append(floor_result)
        predictions[(floor_result["model"], feature_set)] = floor_pred
    log_phase("fit angular disorder models", started)
    selection_df = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    return pd.DataFrame(results).sort_values("rmse"), predictions, selection_df


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
            "current_hurdle_stability_top50_raw_positive_zero_week_floor",
            "sparse_functional_magnitude_shape_top50",
        ): FUNCTIONAL_RESULTS_DIR
        / "predictions/severity_predictions_current_hurdle_stability_top50_raw_positive_zero_week_floor_sparse_functional_magnitude_shape_top50_spectral_plus_week.csv",
    }
    out = {}
    for key, path in paths.items():
        if path.exists():
            out[key] = pd.read_csv(path)
        else:
            logging.warning("Missing context prediction: %s", path)
    return out


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
        "n_features": math.nan,
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred) if len(np.unique(y)) > 1 else math.nan,
        "spearman": residual_pipeline.safe_spearman(y, pred),
    }


def paired_delta_vs_nadir(predictions: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    baseline_key = ("current_hurdle_top20_raw_positive", "compact_anomaly_nadir")
    if baseline_key not in predictions:
        return pd.DataFrame()
    rng = np.random.default_rng(SEED)
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


def build_report(
    comparison: pd.DataFrame,
    delta: pd.DataFrame,
    week_summary: pd.DataFrame,
    audit: pd.DataFrame,
    selections: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "angular_disorder_current_severity_summary.md"
    display_cols = [
        "model",
        "feature_set",
        "source",
        "n_test",
        "n_features",
        "rmse",
        "mae",
        "r2",
        "spearman",
    ]
    selected = (
        selections[selections["selected_for_final_model"]].copy()
        if not selections.empty
        else pd.DataFrame()
    )
    selected_summary = (
        selected.groupby(["feature_set", "role"], as_index=False)
        .size()
        .rename(columns={"size": "n_selected"})
        if not selected.empty
        else pd.DataFrame()
    )
    lines = [
        "## Results: Angular Disorder Current Severity Model",
        "",
        "This analysis tests whether disease disrupts coherent cross-band angular reflectance behavior and increases lower-quantile/heterogeneity angular signals.",
        "",
        "### Model Comparison",
        "",
        markdown_table(comparison[display_cols].round(4).sort_values("rmse"), max_rows=20),
        "",
        "### Paired Delta Versus Existing Nadir",
        "",
        markdown_table(delta.round(4), max_rows=20),
        "",
        "### Target-Week Diagnostics",
        "",
        markdown_table(week_summary.round(4), max_rows=50),
        "",
        "### Feature Support",
        "",
        markdown_table(audit.round(4), max_rows=10),
        "",
        "### Selected Feature Count",
        "",
        markdown_table(selected_summary, max_rows=20),
        "",
        "**Interpretation**: This is a current-severity analysis. The features are intended to be biologically interpretable angular disorder descriptors rather than generic model-capacity additions.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Target: same-week plot-level `ds_plot`.",
        "- Inputs: VZA-binned reflectance distribution curves.",
        "- Feature sets: cross-band angular decoupling, lower-quantile/heterogeneity angular curves, and their combination.",
        "- Excluded predictors: treatment, cultivar, block, inoculation/design metadata, RAA, and disease history.",
        "- Model: grouped 2024-only top-50 stability-selected hurdle Ridge, plus a current-severity zero-week floor row.",
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
    feature_sets, audit = build_feature_sets_from_long(long_2024, long_2025)
    disorder_results, disorder_predictions, selections = evaluate_feature_sets(
        feature_sets, disease_2024, disease_2025
    )
    context_predictions = load_context_predictions()
    context_rows = [
        score_prediction_frame(pred, model, feature_set, "existing_context")
        for (model, feature_set), pred in context_predictions.items()
    ]
    comparison = pd.concat(
        [pd.DataFrame(context_rows), disorder_results],
        ignore_index=True,
        sort=False,
    ).sort_values("rmse")
    all_predictions = {**context_predictions, **disorder_predictions}
    delta = paired_delta_vs_nadir(all_predictions)
    week_summary = current_severity.prediction_week_summary(all_predictions)
    paths = {
        "model_comparison": RESULTS_DIR / "angular_disorder_model_comparison.csv",
        "paired_delta_vs_nadir": RESULTS_DIR / "angular_disorder_delta_vs_nadir.csv",
        "target_week_summary": RESULTS_DIR / "angular_disorder_target_week_summary.csv",
        "feature_support": RESULTS_DIR / "angular_disorder_feature_support.csv",
        "selected_features": RESULTS_DIR / "angular_disorder_selected_features.csv",
        "predictions": PREDICTIONS_DIR,
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    week_summary.to_csv(paths["target_week_summary"], index=False)
    audit.to_csv(paths["feature_support"], index=False)
    selections.to_csv(paths["selected_features"], index=False)
    report_path = build_report(comparison, delta, week_summary, audit, selections, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
