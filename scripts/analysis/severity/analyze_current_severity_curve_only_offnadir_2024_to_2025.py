#!/usr/bin/env python3
"""Current severity from curve-only VZA features after removing the 10-15 degree bin."""

from __future__ import annotations

import logging
import math
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
from scripts.analysis.severity import debug_multiangular_rmse_bottleneck as residual_pipeline
from scripts.analysis.severity.analyze_current_severity_curve_only_functional_2024_to_2025 import (
    build_sampled_curve_features,
    make_curve_sources,
)
from scripts.analysis.severity.analyze_current_severity_magnitude_shape_functional_2024_to_2025 import (
    COVARIATES,
    paired_delta_vs_nadir,
    read_inputs,
    score_prediction_frame,
)
from scripts.analysis.severity.analyze_current_severity_sparse_functional_discriminant_shape_2024_to_2025 import (
    RAA_2024,
    RAA_2025,
    read_raa,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import markdown_table

OUTPUT_ROOT = ROOT / "outputs/current_severity_curve_only_offnadir_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/logs"

FEATURE_SET = "curve_only_vza_log_offnadir_no_10_15"
CURRENT_RESULTS = ROOT / "outputs/current_severity_2024_to_2025/results"
CURVE_ONLY_RESULTS = ROOT / "outputs/current_severity_curve_only_functional_2024_to_2025/results"


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_current_severity_curve_only_offnadir_2024_to_2025_{timestamp}.log"
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


def configure_paths() -> None:
    residual_pipeline.ROOT = ROOT
    residual_pipeline.COVARIATES = COVARIATES
    residual_pipeline.OUTPUT_ROOT = OUTPUT_ROOT
    residual_pipeline.RESULTS_DIR = RESULTS_DIR
    residual_pipeline.REPORTS_DIR = REPORTS_DIR
    residual_pipeline.FIGURES_DIR = FIGURES_DIR
    residual_pipeline.PREDICTIONS_DIR = PREDICTIONS_DIR
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "curve_only_offnadir_manifest.json"


def remove_10_15_bin(frame: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in frame.columns if isinstance(col, str) and col.endswith("__vza_12.5")]
    logging.info("Dropping %d curve features from 10-15 degree bin.", len(drop_cols))
    return frame.drop(columns=drop_cols)


def load_context_predictions() -> dict[tuple[str, str], pd.DataFrame]:
    paths = {
        (
            "current_hurdle_top20_raw_positive",
            "compact_anomaly_nadir",
        ): CURRENT_RESULTS
        / "predictions/severity_predictions_current_hurdle_top20_raw_positive_compact_anomaly_nadir_spectral_plus_week.csv",
        (
            "current_hurdle_stability_top50_raw_positive",
            "compact_anomaly_multiangular",
        ): CURRENT_RESULTS
        / "predictions/severity_predictions_current_hurdle_stability_top50_raw_positive_compact_anomaly_multiangular_spectral_plus_week.csv",
        (
            "current_hurdle_stability_top50_raw_positive",
            "curve_only_vza_log",
        ): CURVE_ONLY_RESULTS
        / "predictions/severity_predictions_current_hurdle_stability_top50_raw_positive_curve_only_vza_log_spectral_plus_week.csv",
    }
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for key, path in paths.items():
        if path.exists():
            out[key] = pd.read_csv(path)
        else:
            logging.warning("Missing context prediction: %s", path)
    return out


def evaluate_offnadir(train_features: pd.DataFrame, test_features: pd.DataFrame, disease_2024: pd.DataFrame, disease_2025: pd.DataFrame):
    train = current_severity.build_current_model_table(train_features, disease_2024)
    test = current_severity.build_current_model_table(test_features, disease_2025)
    results = []
    predictions = {}
    selections = []
    for top_k in [20, 50]:
        result, pred, selection = current_severity.current_hurdle_stability_topk_model(
            train, test, FEATURE_SET, top_k=top_k, log_positive=False
        )
        result["source"] = "curve_only_offnadir"
        results.append(result)
        predictions[(result["model"], FEATURE_SET)] = pred
        selections.append(selection)
    return results, predictions, pd.concat(selections, ignore_index=True)


def write_report(comparison: pd.DataFrame, delta: pd.DataFrame, selection: pd.DataFrame, paths: dict[str, Path], log_path: Path) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "curve_only_offnadir_current_severity_summary.md"
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
    )
    lines = [
        "## Results: Curve-Only Off-Nadir Current Severity",
        "",
        "This analysis retrains the selected VZA log-curve model after removing all 10-15 degree curve predictors.",
        "",
        "### Model Comparison",
        "",
        markdown_table(display[display_cols].round(4).sort_values("rmse"), max_rows=20),
        "",
        "### Paired Delta Versus Existing Nadir",
        "",
        markdown_table(delta.round(4), max_rows=20),
        "",
        "### Selected Feature Counts",
        "",
        markdown_table(selected_summary, max_rows=20),
        "",
        "**Interpretation**: If the off-nadir-only model remains close to the full curve model, the angular claim does not depend on the near-nadir bin. If it degrades strongly, near-nadir magnitude carries a major part of the prediction signal.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Removed predictors: all sampled VZA curve features ending in `__vza_12.5`, representing the 10-15 degree bin.",
        "- Models: grouped 2024 stability-selected hurdle Ridge with top-20 and top-50 feature caps.",
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
    configure_paths()
    for directory in [RESULTS_DIR, REPORTS_DIR, PREDICTIONS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()

    long_2024, long_2025, disease_2024, disease_2025 = read_inputs()
    raa_2024 = read_raa(RAA_2024)
    raa_2025 = read_raa(RAA_2025)
    train_sources, test_sources = make_curve_sources(long_2024, long_2025, raa_2024, raa_2025)
    train_features, test_features, audit = build_sampled_curve_features(
        train_sources, test_sources, ["vza"], ("log",)
    )
    train_features = remove_10_15_bin(train_features)
    test_features = remove_10_15_bin(test_features)
    audit.to_csv(RESULTS_DIR / "curve_only_offnadir_feature_audit.csv", index=False)

    rows, new_predictions, selection = evaluate_offnadir(
        train_features, test_features, disease_2024, disease_2025
    )
    context_predictions = load_context_predictions()
    context_rows = [
        score_prediction_frame(pred, model, feature_set, "existing_context")
        for (model, feature_set), pred in context_predictions.items()
    ]
    comparison = pd.concat([pd.DataFrame(context_rows), pd.DataFrame(rows)], ignore_index=True, sort=False).sort_values("rmse")
    delta = paired_delta_vs_nadir({**context_predictions, **new_predictions})
    paths = {
        "model_comparison": RESULTS_DIR / "curve_only_offnadir_model_comparison.csv",
        "paired_delta_vs_nadir": RESULTS_DIR / "curve_only_offnadir_delta_vs_nadir.csv",
        "selected_features": RESULTS_DIR / "curve_only_offnadir_selected_features.csv",
        "feature_audit": RESULTS_DIR / "curve_only_offnadir_feature_audit.csv",
        "predictions": PREDICTIONS_DIR,
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    selection.to_csv(paths["selected_features"], index=False)
    report_path = write_report(comparison, delta, selection, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total)


if __name__ == "__main__":
    main()
