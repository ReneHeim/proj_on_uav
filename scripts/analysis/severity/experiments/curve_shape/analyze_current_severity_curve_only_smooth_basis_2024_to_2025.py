#!/usr/bin/env python3
"""Current severity from smooth curve-only VZA basis coefficients.

This is a constrained functional model: each plot-week angular reflectance
function is compressed into low-dimensional smooth polynomial or B-spline
coefficients before the same grouped stability-selected hurdle Ridge model.
"""

from __future__ import annotations

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
from sklearn.preprocessing import SplineTransformer

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
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_curve_only_functional_2024_to_2025 import (
    angle_columns,
    make_curve_sources,
    merge_feature_frames,
)
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_magnitude_shape_functional_2024_to_2025 import (
    COVARIATES,
    META_COLS,
    MIN_GROUP_ANGLES,
    load_context_predictions,
    paired_delta_vs_nadir,
    read_inputs,
    score_prediction_frame,
    shifted_log_values,
)
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_sparse_functional_discriminant_shape_2024_to_2025 import (
    RAA_2024,
    RAA_2025,
    read_raa,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    clean_token,
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/curve_only_smooth_basis_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

TARGET = current_severity.TARGET
SEED = current_severity.SEED


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        LOGS_DIR / f"analyze_current_severity_curve_only_smooth_basis_2024_to_2025_{timestamp}.log"
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
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "curve_only_smooth_basis_manifest.json"


def normalized_angles(angles: np.ndarray) -> np.ndarray:
    amin = float(np.min(angles))
    span = float(np.max(angles) - amin)
    if span <= 0:
        return np.zeros_like(angles)
    return (angles - amin) / span


def polynomial_coefficients(values: np.ndarray, angles: np.ndarray, degree: int) -> np.ndarray:
    x = normalized_angles(angles)
    design = np.vander(x, N=degree + 1, increasing=True)
    pinv = np.linalg.pinv(design)
    return values @ pinv.T


def spline_coefficients(
    values: np.ndarray,
    angles: np.ndarray,
    n_knots: int,
    degree: int,
) -> tuple[np.ndarray, int]:
    x = normalized_angles(angles).reshape(-1, 1)
    spline = SplineTransformer(
        n_knots=n_knots,
        degree=degree,
        include_bias=False,
        extrapolation="continue",
    )
    basis = spline.fit_transform(x)
    pinv = np.linalg.pinv(basis)
    return values @ pinv.T, basis.shape[1]


def derivative_features(values: np.ndarray, angles: np.ndarray) -> dict[str, np.ndarray]:
    gradient = np.gradient(values, angles, axis=1)
    curvature = np.gradient(gradient, angles, axis=1)
    low = angles <= 25
    high = angles >= 45
    return {
        "deriv_mean": gradient.mean(axis=1),
        "deriv_abs_mean": np.abs(gradient).mean(axis=1),
        "deriv_high_minus_low": gradient[:, high].mean(axis=1) - gradient[:, low].mean(axis=1),
        "curv_abs_mean": np.abs(curvature).mean(axis=1),
        "curv_high_minus_low": curvature[:, high].mean(axis=1) - curvature[:, low].mean(axis=1),
    }


def basis_features_for_group(
    train_group: pd.DataFrame,
    test_group: pd.DataFrame,
    source: str,
    group: str,
    cols: list[str],
    angles: np.ndarray,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(train_group[cols])
    test_values = imputer.transform(test_group[cols])
    train_log, test_log, shift = shifted_log_values(train_values, test_values)
    prefix = f"smoothcurve__{clean_token(source)}__{clean_token(group)}"
    train_out = train_group[META_COLS].reset_index(drop=True).copy()
    test_out = test_group[META_COLS].reset_index(drop=True).copy()

    if variant == "poly3":
        train_coef = polynomial_coefficients(train_log, angles, degree=3)
        test_coef = polynomial_coefficients(test_log, angles, degree=3)
        basis_name = "poly3"
    elif variant == "poly4":
        train_coef = polynomial_coefficients(train_log, angles, degree=4)
        test_coef = polynomial_coefficients(test_log, angles, degree=4)
        basis_name = "poly4"
    elif variant in {"spline4", "spline4_deriv"}:
        train_coef, n_basis = spline_coefficients(train_log, angles, n_knots=4, degree=3)
        test_coef, _ = spline_coefficients(test_log, angles, n_knots=4, degree=3)
        basis_name = f"spline4_{n_basis}basis"
    elif variant == "spline5":
        train_coef, n_basis = spline_coefficients(train_log, angles, n_knots=5, degree=3)
        test_coef, _ = spline_coefficients(test_log, angles, n_knots=5, degree=3)
        basis_name = f"spline5_{n_basis}basis"
    else:
        raise ValueError(f"Unknown variant: {variant}")

    for idx in range(train_coef.shape[1]):
        name = f"{prefix}__{basis_name}__coef_{idx:02d}"
        train_out[name] = train_coef[:, idx]
        test_out[name] = test_coef[:, idx]

    n_derivative_features = 0
    if variant == "spline4_deriv":
        train_deriv = derivative_features(train_log, angles)
        test_deriv = derivative_features(test_log, angles)
        n_derivative_features = len(train_deriv)
        for name, values in train_deriv.items():
            train_out[f"{prefix}__{basis_name}__{name}"] = values
        for name, values in test_deriv.items():
            test_out[f"{prefix}__{basis_name}__{name}"] = values

    audit = {
        "source": source,
        "curve_group": group,
        "variant": variant,
        "n_train_curves": len(train_group),
        "n_test_curves": len(test_group),
        "n_angles": len(cols),
        "n_basis_coefficients": train_coef.shape[1],
        "n_derivative_features": n_derivative_features,
        "n_features": train_out.shape[1] - len(META_COLS),
        "shift_for_log": shift,
    }
    return train_out, test_out, audit


def build_smooth_basis_features(
    train_sources: dict[str, pd.DataFrame],
    test_sources: dict[str, pd.DataFrame],
    selected_sources: list[str],
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for source in selected_sources:
        train_pivot = train_sources[source]
        test_pivot = test_sources[source]
        common_groups = sorted(
            set(train_pivot["curve_group"]).intersection(test_pivot["curve_group"])
        )
        for group in common_groups:
            train_group = train_pivot[train_pivot["curve_group"].eq(group)].copy()
            test_group = test_pivot[test_pivot["curve_group"].eq(group)].copy()
            cols, angles = angle_columns(train_group, test_group)
            if len(cols) < MIN_GROUP_ANGLES:
                continue
            train_out, test_out, audit = basis_features_for_group(
                train_group, test_group, source, group, cols, angles, variant
            )
            train_frames.append(train_out)
            test_frames.append(test_out)
            audits.append(audit)
    train = merge_feature_frames(train_frames)
    test = merge_feature_frames(test_frames)
    audit = pd.DataFrame(audits)
    logging.info(
        "%s/%s smooth curve features: train=%d rows/%d features test=%d rows/%d features",
        "+".join(selected_sources),
        variant,
        len(train),
        train.shape[1] - len(META_COLS),
        len(test),
        test.shape[1] - len(META_COLS),
    )
    log_phase(f"build smooth basis {variant}", started)
    return train, test, audit


def evaluate_feature_set(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
    feature_set: str,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    train = current_severity.build_current_model_table(train_features, disease_2024)
    test = current_severity.build_current_model_table(test_features, disease_2025)
    results: list[dict[str, object]] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    selections: list[pd.DataFrame] = []
    max_features = train_features.shape[1] - len(META_COLS)
    for top_k in [20, min(50, max_features)]:
        result, pred, selection = current_severity.current_hurdle_stability_topk_model(
            train,
            test,
            feature_set,
            top_k=top_k,
            log_positive=False,
        )
        result["source"] = "curve_only_smooth_basis"
        results.append(result)
        predictions[(result["model"], feature_set)] = pred
        selections.append(selection)
    return results, predictions, pd.concat(selections, ignore_index=True)


def write_report(
    comparison: pd.DataFrame,
    delta: pd.DataFrame,
    audit: pd.DataFrame,
    selection: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "curve_only_smooth_basis_current_severity_summary.md"
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
        "train_in_sample_rmse",
        "train_grouped_oof_rmse",
        "external_minus_oof_rmse",
    ]
    display = comparison.copy()
    for col in display_cols:
        if col not in display.columns:
            display[col] = math.nan
    selected_summary = (
        selection[selection["selected_for_final_model"]]
        .groupby(["model", "feature_set", "role"], dropna=False)
        .size()
        .reset_index(name="n_selected_features")
        if not selection.empty
        else pd.DataFrame()
    )
    lines = [
        "## Results: Curve-Only Smooth Basis Current Severity",
        "",
        "This analysis compresses each angular reflectance function into smooth polynomial or spline coefficients before severity modeling.",
        "",
        "### Model Comparison",
        "",
        markdown_table(display[display_cols].round(4).sort_values("rmse"), max_rows=25),
        "",
        "### Paired Delta Versus Existing Nadir",
        "",
        "Positive RMSE reduction means the smooth curve-basis candidate improves over the compact nadir current-severity baseline on matched 2025 plot-week rows.",
        "",
        markdown_table(delta.round(4), max_rows=25),
        "",
        "### Feature Construction Audit",
        "",
        markdown_table(audit.round(4), max_rows=40),
        "",
        "### Selected Feature Counts",
        "",
        markdown_table(selected_summary, max_rows=30),
        "",
        "**Interpretation**: This is still a curve-only representation, but the independent VZA samples are replaced by smooth basis coefficients. A good result with fewer features would be more defensible as a functional model than isolated selected angle bins.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Target: same-week plot-level `ds_plot`.",
        "- Inputs: VZA log-reflectance curves, compressed into polynomial or B-spline coefficients.",
        "- Models: grouped 2024 stability-selected hurdle Ridge with top-20 and top-50 feature caps.",
        "- Excluded predictors: compact engineered summaries, treatment, cultivar, block, inoculation/design metadata, disease history, RAA in the primary variants, and residual correction from another model.",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
    ]
    lines.extend([f"- {label}: `{path}`" for label, path in paths.items()])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    total_started = time.perf_counter()
    configure_reused_pipeline_paths()
    for directory in [RESULTS_DIR, REPORTS_DIR, PREDICTIONS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()

    long_2024, long_2025, disease_2024, disease_2025 = read_inputs()
    raa_2024 = read_raa(RAA_2024)
    raa_2025 = read_raa(RAA_2025)
    train_sources, test_sources = make_curve_sources(long_2024, long_2025, raa_2024, raa_2025)

    variants = {
        "smooth_curve_poly3_vza_log": (["vza"], "poly3"),
        "smooth_curve_poly4_vza_log": (["vza"], "poly4"),
        "smooth_curve_spline4_vza_log": (["vza"], "spline4"),
        "smooth_curve_spline5_vza_log": (["vza"], "spline5"),
        "smooth_curve_spline4_deriv_vza_log": (["vza"], "spline4_deriv"),
        "smooth_curve_spline4_vza_raa_log": (["vza", "vza_raa_reliable"], "spline4"),
    }
    result_rows: list[dict[str, object]] = []
    prediction_map: dict[tuple[str, str], pd.DataFrame] = {}
    audits: list[pd.DataFrame] = []
    selections: list[pd.DataFrame] = []
    for feature_set, (sources, variant) in variants.items():
        train_features, test_features, audit = build_smooth_basis_features(
            train_sources,
            test_sources,
            sources,
            variant,
        )
        rows, preds, selection = evaluate_feature_set(
            train_features,
            test_features,
            disease_2024,
            disease_2025,
            feature_set,
        )
        result_rows.extend(rows)
        prediction_map.update(preds)
        audits.append(audit.assign(feature_set=feature_set))
        selections.append(selection)

    context_predictions = load_context_predictions()
    context_rows = [
        score_prediction_frame(pred, model, feature_set, "existing_context")
        for (model, feature_set), pred in context_predictions.items()
    ]
    comparison = pd.concat(
        [pd.DataFrame(context_rows), pd.DataFrame(result_rows)],
        ignore_index=True,
        sort=False,
    ).sort_values("rmse")
    all_predictions = {**context_predictions, **prediction_map}
    delta = paired_delta_vs_nadir(all_predictions)
    audit_df = pd.concat(audits, ignore_index=True)
    selection_df = pd.concat(selections, ignore_index=True)

    paths = {
        "model_comparison": RESULTS_DIR / "curve_only_smooth_basis_model_comparison.csv",
        "paired_delta_vs_nadir": RESULTS_DIR / "curve_only_smooth_basis_delta_vs_nadir.csv",
        "feature_audit": RESULTS_DIR / "curve_only_smooth_basis_feature_audit.csv",
        "selected_features": RESULTS_DIR / "curve_only_smooth_basis_selected_features.csv",
        "predictions": PREDICTIONS_DIR,
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    audit_df.to_csv(paths["feature_audit"], index=False)
    selection_df.to_csv(paths["selected_features"], index=False)
    report_path = write_report(comparison, delta, audit_df, selection_df, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
