#!/usr/bin/env python3
"""Current severity from curve-only VZA functions with supervised PLS.

This tests whether Partial Least Squares is a better model than Ridge for
sampled angular reflectance curves.  The input representation is the same clean
curve-only log-reflectance function used in the Ridge experiment; only the
functional model changes.
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

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
    build_sampled_curve_features,
    make_curve_sources,
)
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_magnitude_shape_functional_2024_to_2025 import (
    COVARIATES,
    load_context_predictions,
    paired_delta_vs_nadir,
    read_inputs,
    score_prediction_frame,
)
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_sparse_functional_discriminant_shape_2024_to_2025 import (
    RAA_2024,
    RAA_2025,
    read_raa,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/curve_only_pls_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

TARGET = current_severity.TARGET
SEED = current_severity.SEED
COMPONENT_GRID = [2, 4, 6, 8, 10, 12, 16, 20]


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_current_severity_curve_only_pls_2024_to_2025_{timestamp}.log"
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
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "curve_only_pls_manifest.json"


def regression_scores(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": math.sqrt(mean_squared_error(y_true, pred)),
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred) if len(np.unique(y_true)) > 1 else math.nan,
        "spearman": residual_pipeline.safe_spearman(y_true, pred),
    }


def effective_components(requested: int, n_samples: int, n_features: int) -> int:
    return max(1, min(requested, n_samples - 1, n_features))


def fit_pls_scores(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train_x))
    x_test = scaler.transform(imputer.transform(test_x))
    n_eff = effective_components(n_components, x_train.shape[0], x_train.shape[1])
    pls = PLSRegression(n_components=n_eff, scale=False)
    pls.fit(x_train, y)
    return pls.transform(x_train), pls.transform(x_test)


def fit_hurdle_pls(
    train_aligned: pd.DataFrame,
    test_aligned: pd.DataFrame,
    cols: list[str],
    n_components: int,
) -> np.ndarray:
    y_train = train_aligned[TARGET].to_numpy(float)
    y_present = (y_train > 0).astype(int)
    if len(np.unique(y_present)) < 2:
        disease_prob = np.full(len(test_aligned), float(y_present.mean()))
    else:
        train_scores, test_scores = fit_pls_scores(
            train_aligned[cols],
            test_aligned[cols],
            y_present.astype(float),
            n_components,
        )
        classifier = LogisticRegressionCV(
            Cs=np.logspace(-2, 1, 8),
            cv=3,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=SEED,
        )
        classifier.fit(train_scores, y_present)
        disease_prob = classifier.predict_proba(test_scores)[:, 1]

    positive_mask = y_train > 0
    if positive_mask.sum() < 5:
        severity_pred = np.full(len(test_aligned), float(np.nanmean(y_train)))
    else:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        x_pos = scaler.fit_transform(imputer.fit_transform(train_aligned.loc[positive_mask, cols]))
        x_test = scaler.transform(imputer.transform(test_aligned[cols]))
        n_eff = effective_components(n_components, x_pos.shape[0], x_pos.shape[1])
        regressor = PLSRegression(n_components=n_eff, scale=False)
        regressor.fit(x_pos, y_train[positive_mask])
        severity_pred = regressor.predict(x_test).ravel()
    return residual_pipeline.clip_predictions(disease_prob * severity_pred, y_train)


def grouped_oof_predictions(
    train_aligned: pd.DataFrame,
    cols: list[str],
    n_components: int,
) -> np.ndarray:
    y = train_aligned[TARGET].to_numpy(float)
    groups = train_aligned["plot_id"].to_numpy()
    pred = np.zeros(len(train_aligned), dtype=float)
    n_splits = min(5, len(np.unique(groups)))
    for fit_idx, eval_idx in GroupKFold(n_splits=n_splits).split(train_aligned, y, groups):
        pred[eval_idx] = fit_hurdle_pls(
            train_aligned.iloc[fit_idx],
            train_aligned.iloc[eval_idx],
            cols,
            n_components,
        )
    return pred


def evaluate_pls_variant(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
    feature_set: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    train = current_severity.build_current_model_table(train_features, disease_2024)
    test = current_severity.build_current_model_table(test_features, disease_2025)
    cols, train_aligned, test_aligned = residual_pipeline.prepare_aligned(train, test)
    y_train = train_aligned[TARGET].to_numpy(float)

    tuning_rows: list[dict[str, object]] = []
    best_n = COMPONENT_GRID[0]
    best_rmse = math.inf
    for n_components in COMPONENT_GRID:
        pred = grouped_oof_predictions(train_aligned, cols, n_components)
        scores = regression_scores(y_train, pred)
        tuning_rows.append(
            {
                "feature_set": feature_set,
                "n_components": n_components,
                "oof_rmse": scores["rmse"],
                "oof_mae": scores["mae"],
                "oof_r2": scores["r2"],
                "oof_spearman": scores["spearman"],
            }
        )
        if scores["rmse"] < best_rmse:
            best_rmse = scores["rmse"]
            best_n = n_components

    external_pred = fit_hurdle_pls(train_aligned, test_aligned, cols, best_n)
    in_sample_pred = fit_hurdle_pls(train_aligned, train_aligned, cols, best_n)
    oof_pred = grouped_oof_predictions(train_aligned, cols, best_n)
    model = "curve_only_hurdle_pls"
    predictions = residual_pipeline.prediction_frame(
        test_aligned, external_pred, model, feature_set
    )
    residual_pipeline.save_predictions(predictions, model, feature_set, COVARIATES)
    result = residual_pipeline.score_predictions(
        predictions, len(train_aligned), len(cols), model, feature_set
    )
    in_sample = regression_scores(y_train, in_sample_pred)
    oof = regression_scores(y_train, oof_pred)
    result["source"] = "curve_only_pls"
    result["selected_n_components"] = best_n
    result["train_in_sample_rmse"] = in_sample["rmse"]
    result["train_grouped_oof_rmse"] = oof["rmse"]
    result["train_oof_minus_in_sample_rmse"] = oof["rmse"] - in_sample["rmse"]
    result["external_minus_oof_rmse"] = result["rmse"] - oof["rmse"]
    result["fit_time_s"] = time.perf_counter() - started
    return result, predictions, pd.DataFrame(tuning_rows)


def build_report(
    comparison: pd.DataFrame,
    delta: pd.DataFrame,
    tuning: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "curve_only_pls_current_severity_summary.md"
    display_cols = [
        "model",
        "feature_set",
        "source",
        "n_test",
        "n_features",
        "selected_n_components",
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
    lines = [
        "## Results: Curve-Only PLS Current Severity",
        "",
        "This analysis tests supervised PLS as the model for sampled angular reflectance curves.",
        "",
        "### Model Comparison",
        "",
        markdown_table(display[display_cols].round(4).sort_values("rmse"), max_rows=20),
        "",
        "### Paired Delta Versus Existing Nadir",
        "",
        "Positive RMSE reduction means the candidate improves over the compact nadir current-severity baseline on matched 2025 plot-week rows.",
        "",
        markdown_table(delta.round(4), max_rows=20),
        "",
        "### Component Tuning",
        "",
        markdown_table(tuning.round(4), max_rows=40),
        "",
        "**Interpretation**: PLS is useful only if grouped 2024 OOF selects a component count that transfers to 2025 without a large external-minus-OOF gap.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Target: same-week plot-level `ds_plot`.",
        "- Inputs: curve-only sampled VZA log reflectance, plus an optional reliable VZA-by-RAA log-curve variant.",
        "- Component selection: grouped 2024 OOF over n_components = [2, 4, 6, 8, 10, 12, 16, 20].",
        "- Excluded predictors: compact engineered summaries, treatment, cultivar, block, inoculation/design metadata, disease history, and residual correction.",
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
    raa_2024 = read_raa(RAA_2024)
    raa_2025 = read_raa(RAA_2025)
    train_sources, test_sources = make_curve_sources(long_2024, long_2025, raa_2024, raa_2025)

    variants = {
        "curve_only_pls_vza_log": (["vza"], ("log",)),
        "curve_only_pls_vza_raa_log": (["vza", "vza_raa_reliable"], ("log",)),
    }
    result_rows: list[dict[str, object]] = []
    prediction_map: dict[tuple[str, str], pd.DataFrame] = {}
    tuning_rows: list[pd.DataFrame] = []
    for feature_set, (sources, transforms) in variants.items():
        train_features, test_features, _ = build_sampled_curve_features(
            train_sources, test_sources, sources, transforms
        )
        result, predictions, tuning = evaluate_pls_variant(
            train_features,
            test_features,
            disease_2024,
            disease_2025,
            feature_set,
        )
        result_rows.append(result)
        prediction_map[(result["model"], feature_set)] = predictions
        tuning_rows.append(tuning)

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
    tuning = pd.concat(tuning_rows, ignore_index=True)

    paths = {
        "model_comparison": RESULTS_DIR / "curve_only_pls_model_comparison.csv",
        "paired_delta_vs_nadir": RESULTS_DIR / "curve_only_pls_delta_vs_nadir.csv",
        "component_tuning": RESULTS_DIR / "curve_only_pls_component_tuning.csv",
        "predictions": PREDICTIONS_DIR,
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    tuning.to_csv(paths["component_tuning"], index=False)
    report_path = build_report(comparison, delta, tuning, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
