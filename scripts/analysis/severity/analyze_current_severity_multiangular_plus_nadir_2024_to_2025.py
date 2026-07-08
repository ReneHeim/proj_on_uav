#!/usr/bin/env python3
"""Current severity test for compact multiangular features plus explicit nadir features."""

from __future__ import annotations

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
from scripts.analysis.severity import debug_multiangular_rmse_bottleneck as residual_pipeline
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import markdown_table

OUTPUT_ROOT = ROOT / "outputs/current_severity_multiangular_plus_nadir_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/logs"

FEATURE_SET = "compact_anomaly_multiangular_plus_nadir"
CONTEXT_RESULTS = ROOT / "outputs/current_severity_2024_to_2025/results/current_severity_model_comparison.csv"


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_current_severity_multiangular_plus_nadir_2024_to_2025_{timestamp}.log"
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
    residual_pipeline.INPUT_RESULTS_DIR = ROOT / "outputs/multiangular_distribution_feature_family/results"
    residual_pipeline.COVARIATES = "spectral_plus_week"
    residual_pipeline.OUTPUT_ROOT = OUTPUT_ROOT
    residual_pipeline.RESULTS_DIR = RESULTS_DIR
    residual_pipeline.REPORTS_DIR = REPORTS_DIR
    residual_pipeline.FIGURES_DIR = FIGURES_DIR
    residual_pipeline.PREDICTIONS_DIR = PREDICTIONS_DIR
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "multiangular_plus_nadir_manifest.json"


def combine_multiangular_plus_nadir(features: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = ["year", "week", "plot_id", "cult", "trt"]
    train_multi, test_multi = features["compact_anomaly_multiangular"]
    train_nadir, test_nadir = features["compact_anomaly_nadir"]

    def merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        right_features = [col for col in right.columns if col not in meta]
        merged = left.merge(right[meta + right_features], on=meta, how="inner", validate="one_to_one")
        if len(merged) != len(left):
            raise RuntimeError(f"Merge lost rows: {len(left)} -> {len(merged)}")
        return merged

    return merge(train_multi, train_nadir), merge(test_multi, test_nadir)


def load_context_rows() -> pd.DataFrame:
    if not CONTEXT_RESULTS.exists():
        return pd.DataFrame()
    context = pd.read_csv(CONTEXT_RESULTS)
    keep = context[
        (
            context["model"].isin(
                [
                    "current_hurdle_stability_top50_raw_positive",
                    "current_hurdle_stability_top30_raw_positive",
                    "hurdle_probability_times_severity",
                    "current_hurdle_top20_raw_positive",
                ]
            )
        )
        & context["feature_set"].isin(["compact_anomaly_multiangular", "compact_anomaly_nadir"])
    ].copy()
    keep["source"] = "existing_context"
    return keep


def write_report(results: pd.DataFrame, paths: dict[str, Path], log_path: Path) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "current_severity_multiangular_plus_nadir_summary.md"
    display_cols = [
        "model",
        "feature_set",
        "source",
        "n_train",
        "n_test",
        "n_features",
        "rmse",
        "mae",
        "r2",
        "spearman",
        "train_in_sample_rmse",
        "train_grouped_oof_rmse",
        "external_minus_oof_rmse",
        "classifier_features",
        "regressor_features",
        "feature_selection_strategy",
    ]
    display = results.copy()
    for col in display_cols:
        if col not in display.columns:
            display[col] = pd.NA
    lines = [
        "## Results: Compact Multiangular Plus Explicit Nadir Current Severity",
        "",
        "This analysis tests whether appending the separate compact nadir feature block to compact multiangular features improves current same-week severity prediction.",
        "",
        "### Model Comparison",
        "",
        markdown_table(display[display_cols].round(4).sort_values("rmse"), max_rows=20),
        "",
        "**Interpretation**: The compact multiangular matrix already contains the near-nadir 10-15 degree VZA bin. This experiment tests whether adding the separate nadir aggregation block provides extra stable information or mostly redundant features.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Target: same-week plot-level `ds_plot`.",
        "- New feature set: `compact_anomaly_multiangular_plus_nadir`.",
        "- Models: same current hurdle stability top-30/top-50 raw-positive models used in the current-severity comparison.",
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

    t0 = time.perf_counter()
    disease_2024, disease_2025 = current_severity.load_clean_disease_scores()
    log_phase("load disease scores", t0)

    t0 = time.perf_counter()
    features = residual_pipeline.load_cached_features()
    train_features, test_features = combine_multiangular_plus_nadir(features)
    logging.info(
        "%s features: train=%s test=%s",
        FEATURE_SET,
        train_features.shape,
        test_features.shape,
    )
    log_phase("load and combine feature sets", t0)

    train = current_severity.build_current_model_table(train_features, disease_2024)
    test = current_severity.build_current_model_table(test_features, disease_2025)
    results = []
    selections = []
    predictions = {}

    t0 = time.perf_counter()
    for top_k in [30, 50]:
        result, pred, selection = current_severity.current_hurdle_stability_topk_model(
            train, test, FEATURE_SET, top_k=top_k, log_positive=False
        )
        result["source"] = "multiangular_plus_nadir"
        results.append(result)
        selections.append(selection)
        predictions[(result["model"], FEATURE_SET)] = pred
    log_phase("fit hurdle stability models", t0)

    context = load_context_rows()
    result_frame = pd.DataFrame(results)
    combined = pd.concat([context, result_frame], ignore_index=True, sort=False).sort_values("rmse")
    paths = {
        "model_comparison": RESULTS_DIR / "current_severity_multiangular_plus_nadir_model_comparison.csv",
        "selected_features": RESULTS_DIR / "current_severity_multiangular_plus_nadir_selected_features.csv",
        "predictions": PREDICTIONS_DIR,
    }
    combined.to_csv(paths["model_comparison"], index=False)
    pd.concat(selections, ignore_index=True).to_csv(paths["selected_features"], index=False)
    report_path = write_report(combined, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total)


if __name__ == "__main__":
    main()
