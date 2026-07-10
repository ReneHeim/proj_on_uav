#!/usr/bin/env python3
"""Main compact multiangular tests after removing the 10-15 degree VZA bin."""

from __future__ import annotations

from src.research.common import write_report as persist_report

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MPLCONFIG_DIR = ROOT / "outputs/.matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity import analyze_current_plot_severity_2024_to_2025 as current_severity
from scripts.analysis.severity import debug_multiangular_rmse_bottleneck as future_severity
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import markdown_table

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/main_compact_multiangular_offnadir_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

FEATURE_SET = "compact_anomaly_multiangular_offnadir_no_10_15"
CURRENT_CONTEXT = ROOT / "outputs/runs/analysis/severity/current/2024_to_2025/results/current_severity_model_comparison.csv"
FUTURE_CONTEXT = (
    ROOT
    / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/model_bottleneck_debug/results/candidate_model_comparison.csv"
)
DISEASE_2024_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2024.csv"
DISEASE_2025_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2025.csv"


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_main_compact_multiangular_offnadir_2024_to_2025_{timestamp}.log"
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


def configure_future_pipeline(covariates: str) -> None:
    future_severity.ROOT = ROOT
    future_severity.INPUT_RESULTS_DIR = ROOT / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/results"
    future_severity.FEATURE_SETS = ["compact_anomaly_nadir", "compact_anomaly_multiangular"]
    future_severity.COVARIATES = covariates
    future_severity.OUTPUT_ROOT = OUTPUT_ROOT
    future_severity.RESULTS_DIR = RESULTS_DIR
    future_severity.REPORTS_DIR = REPORTS_DIR
    future_severity.FIGURES_DIR = FIGURES_DIR
    future_severity.PREDICTIONS_DIR = PREDICTIONS_DIR
    future_severity.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "main_compact_offnadir_manifest.json"


def drop_10_15_multiangular(frame: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        col
        for col in frame.columns
        if isinstance(col, str) and col.startswith("dist_multiangular__") and "__10_15__" in col
    ]
    logging.info("Dropping %d compact multiangular 10-15 degree columns.", len(drop_cols))
    return frame.drop(columns=drop_cols)


def load_offnadir_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = future_severity.load_cached_features()
    train, test = features["compact_anomaly_multiangular"]
    return drop_10_15_multiangular(train), drop_10_15_multiangular(test)


def context_rows(path: Path, model_names: list[str], feature_sets: list[str]) -> pd.DataFrame:
    if not path.exists():
        logging.warning("Missing context table: %s", path)
        return pd.DataFrame()
    table = pd.read_csv(path)
    out = table[table["model"].isin(model_names) & table["feature_set"].isin(feature_sets)].copy()
    out["source"] = "existing_context"
    return out


def run_current(train_features: pd.DataFrame, test_features: pd.DataFrame) -> pd.DataFrame:
    configure_future_pipeline("spectral_plus_week")
    disease_2024, disease_2025 = current_severity.load_clean_disease_scores()
    train = current_severity.build_current_model_table(train_features, disease_2024)
    test = current_severity.build_current_model_table(test_features, disease_2025)
    result, _pred, selection = current_severity.current_hurdle_stability_topk_model(
        train, test, FEATURE_SET, top_k=50, log_positive=False
    )
    result["source"] = "offnadir_no_10_15"
    selection.to_csv(RESULTS_DIR / "current_compact_offnadir_selected_features.csv", index=False)
    context = context_rows(
        CURRENT_CONTEXT,
        ["current_hurdle_stability_top50_raw_positive", "current_hurdle_top20_raw_positive"],
        ["compact_anomaly_multiangular", "compact_anomaly_nadir"],
    )
    return pd.concat([context, pd.DataFrame([result])], ignore_index=True, sort=False).sort_values(
        "rmse"
    )


def run_future(train_features: pd.DataFrame, test_features: pd.DataFrame) -> pd.DataFrame:
    configure_future_pipeline("spectral_plus_week_horizon")
    disease_2024 = pd.read_csv(DISEASE_2024_CLEAN)
    disease_2025 = pd.read_csv(DISEASE_2025_CLEAN)
    train = future_severity.build_model_table(train_features, disease_2024)
    test = future_severity.build_model_table(test_features, disease_2025)
    result, _pred, tuning = future_severity.fit_residual_reliability_filtered_xgboost(
        train, test, FEATURE_SET
    )
    result["source"] = "offnadir_no_10_15"
    tuning.to_csv(RESULTS_DIR / "future_compact_offnadir_xgboost_tuning.csv", index=False)
    context = context_rows(
        FUTURE_CONTEXT,
        ["residual_reliability_filtered_xgboost"],
        ["compact_anomaly_multiangular", "compact_anomaly_nadir"],
    )
    return pd.concat([context, pd.DataFrame([result])], ignore_index=True, sort=False).sort_values(
        "rmse"
    )


def build_report(current_results: pd.DataFrame, future_results: pd.DataFrame, paths: dict[str, Path], log_path: Path) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "main_compact_multiangular_offnadir_summary.md"
    display_cols = [
        "model",
        "feature_set",
        "source",
        "covariates",
        "n_train",
        "n_test",
        "n_features",
        "rmse",
        "mae",
        "r2",
        "spearman",
        "train_grouped_oof_rmse",
        "external_minus_oof_rmse",
        "n_stability_features_before_reliability",
        "n_features_removed_by_reliability",
        "xgboost_config",
        "base_model",
    ]

    def display(table: pd.DataFrame) -> pd.DataFrame:
        out = table.copy()
        for col in display_cols:
            if col not in out.columns:
                out[col] = pd.NA
        return out[display_cols].round(4).sort_values("rmse")

    lines = [
        "## Results: Main Compact Multiangular Without 10-15 Degree Bin",
        "",
        "This analysis reruns the main compact multiangular tests after removing every compact feature from the 10-15 degree VZA bin.",
        "",
        "### Current Same-Week Severity",
        "",
        markdown_table(display(current_results), max_rows=12),
        "",
        "### Future Severity",
        "",
        markdown_table(display(future_results), max_rows=12),
        "",
        "**Interpretation**: These are targeted main-model checks. If the off-nadir compact model stays close to the original compact multiangular model, the compact result does not depend on the near-nadir 10-15 degree bin.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Removed compact predictors: columns matching `dist_multiangular__*__10_15__*`.",
        "- Current model: `current_hurdle_stability_top50_raw_positive` with `spectral_plus_week`.",
        "- Future model: `residual_reliability_filtered_xgboost` with `spectral_plus_week_horizon`.",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
    ]
    lines.extend([f"- {label}: `{path}`" for label, path in paths.items()])
    persist_report(report_path, lines)
    return report_path


def main() -> None:
    total = time.perf_counter()
    for directory in [RESULTS_DIR, REPORTS_DIR, PREDICTIONS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()
    configure_future_pipeline("spectral_plus_week_horizon")
    train_features, test_features = load_offnadir_features()
    logging.info("Off-nadir compact features: train=%s test=%s", train_features.shape, test_features.shape)

    t0 = time.perf_counter()
    current_results = run_current(train_features, test_features)
    log_phase("current compact off-nadir main model", t0)

    t0 = time.perf_counter()
    future_results = run_future(train_features, test_features)
    log_phase("future compact off-nadir main model", t0)

    paths = {
        "current_results": RESULTS_DIR / "current_compact_multiangular_offnadir_model_comparison.csv",
        "future_results": RESULTS_DIR / "future_compact_multiangular_offnadir_model_comparison.csv",
        "current_selected_features": RESULTS_DIR / "current_compact_offnadir_selected_features.csv",
        "future_xgboost_tuning": RESULTS_DIR / "future_compact_offnadir_xgboost_tuning.csv",
        "predictions": PREDICTIONS_DIR,
    }
    current_results.to_csv(paths["current_results"], index=False)
    future_results.to_csv(paths["future_results"], index=False)
    report_path = build_report(current_results, future_results, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total)


if __name__ == "__main__":
    main()
