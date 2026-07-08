#!/usr/bin/env python3
"""Future severity residual model with VZA+RAA reflectance features.

This script adds a matched VZA+RAA row for the slide figure that already uses
the frozen-style RidgeCV plus residual XGBoost severity pipeline. It keeps the
comparison imaging-only: no cultivar, treatment, block, or disease-history
predictors are used.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
MPLCONFIG_DIR = ROOT / "outputs/.matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.early_warning.analyze_early_warning_severity_2024 import (  # noqa: E402
    build_model_table,
)
from scripts.analysis.severity import (  # noqa: E402
    debug_multiangular_rmse_bottleneck as residual_pipeline,
)
from scripts.analysis.severity.analyze_current_severity_raa_geometry_fusion_2024_to_2025 import (  # noqa: E402
    build_geometry_feature_sets,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (  # noqa: E402
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/future_severity_raa_geometry_residual_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/logs"

VZA_2024 = (
    ROOT
    / "outputs/result_01_reflectance_distributions/2024/ground_filtered/results/plot_week_angle_features_2024.parquet"
)
VZA_2025 = (
    ROOT
    / "outputs/result_01_reflectance_distributions/2025/ground_filtered/results/plot_week_angle_features_2025.parquet"
)
RAA_2024 = (
    ROOT
    / "outputs/result_01_raa_sun_geometry/2024/ground_filtered/results/plot_week_vza_raa_features_2024.parquet"
)
RAA_2025 = (
    ROOT
    / "outputs/result_01_raa_sun_geometry/2025/ground_filtered/results/plot_week_vza_raa_features_2025.parquet"
)
DISEASE_2024 = ROOT / "outputs/disease/clean_disease_scores_2024.csv"
DISEASE_2025 = ROOT / "outputs/disease/clean_disease_scores_2025.csv"

FEATURE_SET = "multiangular_vza_raa"
SELECTED_MODEL = "residual_reliability_filtered_xgboost"
ALL_FEATURES_MODEL = "residual_all_features_xgboost"
COVARIATES = "spectral_plus_week_horizon"


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_future_severity_raa_residual_comparison_{timestamp}.log"
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


def configure_residual_pipeline_paths() -> None:
    residual_pipeline.ROOT = ROOT
    residual_pipeline.COVARIATES = COVARIATES
    residual_pipeline.OUTPUT_ROOT = OUTPUT_ROOT
    residual_pipeline.RESULTS_DIR = RESULTS_DIR
    residual_pipeline.REPORTS_DIR = REPORTS_DIR
    residual_pipeline.FIGURES_DIR = FIGURES_DIR
    residual_pipeline.PREDICTIONS_DIR = PREDICTIONS_DIR
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "future_raa_residual_manifest.json"


def read_parquet(path: Path) -> pd.DataFrame:
    started = time.perf_counter()
    frame = pl.read_parquet(path).to_pandas()
    logging.info("Read %s rows x %s cols from %s", frame.shape[0], frame.shape[1], path)
    log_phase(f"read parquet {path.name}", started)
    return frame


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    vza_2024 = read_parquet(VZA_2024)
    vza_2025 = read_parquet(VZA_2025)
    raa_2024 = read_parquet(RAA_2024)
    raa_2025 = read_parquet(RAA_2025)
    log_phase("read VZA/RAA inputs", started)
    return vza_2024, vza_2025, raa_2024, raa_2025


def fit_vza_raa_residual_models(
    vza_2024: pd.DataFrame,
    vza_2025: pd.DataFrame,
    raa_2024: pd.DataFrame,
    raa_2025: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    started = time.perf_counter()
    disease_2024 = pd.read_csv(DISEASE_2024)
    disease_2025 = pd.read_csv(DISEASE_2025)
    log_phase("read clean disease CSVs", started)

    feature_started = time.perf_counter()
    features = build_geometry_feature_sets(vza_2024, vza_2025, raa_2024, raa_2025)
    train_features, test_features = features[FEATURE_SET]
    log_phase("build VZA+RAA feature table", feature_started)

    table_started = time.perf_counter()
    train = build_model_table(train_features, disease_2024)
    test = build_model_table(test_features, disease_2025)
    logging.info("%s future model table: train=%d test=%d", FEATURE_SET, len(train), len(test))
    log_phase("build future-severity model tables", table_started)

    selected_started = time.perf_counter()
    selected_result, selected_predictions, selected_tuning = (
        residual_pipeline.fit_residual_reliability_filtered_xgboost(train, test, FEATURE_SET)
    )
    selected_tuning["model"] = selected_result["model"]
    selected_tuning["feature_set"] = FEATURE_SET
    selected_tuning["n_features"] = selected_result["n_features"]
    log_phase("fit selected residual reliability-filtered XGBoost", selected_started)

    all_started = time.perf_counter()
    cols, train_aligned, test_aligned = residual_pipeline.prepare_aligned(train, test)
    all_result, all_predictions, all_tuning = (
        residual_pipeline.fit_tuned_xgboost_residual_with_cols(
            train_aligned,
            test_aligned,
            cols,
            ALL_FEATURES_MODEL,
            FEATURE_SET,
        )
    )
    all_tuning["model"] = all_result["model"]
    all_tuning["feature_set"] = FEATURE_SET
    all_tuning["n_features"] = all_result["n_features"]
    log_phase("fit all-feature residual XGBoost", all_started)

    results = pd.DataFrame([selected_result, all_result])
    predictions = {
        selected_result["model"]: selected_predictions,
        all_result["model"]: all_predictions,
    }
    tuning = pd.concat([selected_tuning, all_tuning], ignore_index=True)
    return results, predictions, tuning


def write_report(
    results: pd.DataFrame,
    predictions_paths: dict[str, Path],
    tuning_path: Path,
    result_path: Path,
    log_path: Path,
) -> Path:
    started = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    selected_row = results[results["model"] == SELECTED_MODEL].iloc[0]
    all_row = results[results["model"] == ALL_FEATURES_MODEL].iloc[0]
    report = [
        "## Results: Future Severity VZA+RAA Residual Comparison",
        "",
        "This run fits selected and all-feature frozen-style future-severity residual pipelines on VZA+RAA reflectance bins so the slide bar plot can compare VZA-only and VZA+RAA under the same model family.",
        "",
        markdown_table(results.round(4), max_rows=10),
        "",
        f"**Interpretation**: The selected VZA+RAA residual model reached RMSE `{selected_row['rmse']:.3f}` on 2025 future severity. The all-VZA+RAA residual model reached RMSE `{all_row['rmse']:.3f}`. This is an imaging-only geometry extension; cultivar, treatment, block, and disease-history predictors are excluded.",
        "",
        "**Outputs**:",
        f"- `{result_path.relative_to(ROOT)}`",
        *[f"- `{path.relative_to(ROOT)}`" for path in predictions_paths.values()],
        f"- `{tuning_path.relative_to(ROOT)}`",
        "",
        "**Reproducibility**:",
        "- Train year: `2024`",
        "- Test year: `2025`",
        "- Target: next observed disease-severity week after each predictor week",
        f"- Feature set: `{FEATURE_SET}`",
        f"- Selected model: `{SELECTED_MODEL}`",
        f"- All-feature model: `{ALL_FEATURES_MODEL}`",
        f"- Covariates: `{COVARIATES}`",
        "- Pipeline: RidgeCV base model plus XGBoost residual correction with grouped 2024 OOF residuals; the selected row also applies stability selection and reliability filtering, while the all-feature row uses all aligned VZA+RAA reflectance features plus timing covariates",
        f"- Log: `{log_path.relative_to(ROOT)}`",
    ]
    path = REPORTS_DIR / "future_severity_raa_residual_comparison_summary.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    logging.info("Wrote report: %s", path)
    log_phase("write markdown report", started)
    return path


def main() -> None:
    total_started = time.perf_counter()
    log_path = setup_logging()
    configure_residual_pipeline_paths()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    vza_2024, vza_2025, raa_2024, raa_2025 = read_inputs()
    results, predictions, tuning = fit_vza_raa_residual_models(
        vza_2024, vza_2025, raa_2024, raa_2025
    )

    write_started = time.perf_counter()
    result_path = RESULTS_DIR / "future_severity_vza_raa_residual_model_comparison.csv"
    tuning_path = RESULTS_DIR / "future_severity_vza_raa_residual_xgboost_tuning_audit.csv"
    prediction_paths = {
        model: PREDICTIONS_DIR
        / f"severity_predictions_{model}_multiangular_vza_raa_spectral_plus_week_horizon.csv"
        for model in predictions
    }
    results.to_csv(result_path, index=False)
    for model, frame in predictions.items():
        frame.to_csv(prediction_paths[model], index=False)
    tuning.to_csv(tuning_path, index=False)
    logging.info("Wrote result: %s", result_path)
    for path in prediction_paths.values():
        logging.info("Wrote predictions: %s", path)
    logging.info("Wrote tuning audit: %s", tuning_path)
    log_phase("write result tables", write_started)

    write_report(results, prediction_paths, tuning_path, result_path, log_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
