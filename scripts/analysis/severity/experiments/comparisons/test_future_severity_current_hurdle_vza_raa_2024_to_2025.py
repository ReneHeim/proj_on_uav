#!/usr/bin/env python3
"""Test current-severity VZA+RAA hurdle models on future severity.

This is a targeted architecture-transfer check: keep the direct current-severity
hurdle model family, but replace the same-week target with the future-severity
target used by the early-warning severity analysis.
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
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
MPLCONFIG_DIR = ROOT / "outputs/.matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.early_warning.analyze_early_warning_severity_2024 import (  # noqa: E402
    build_model_table,
)
from scripts.analysis.severity import (  # noqa: E402
    analyze_current_plot_severity_2024_to_2025 as current_severity,
)
from scripts.analysis.severity import debug_multiangular_rmse_bottleneck as residual_pipeline  # noqa: E402
from scripts.analysis.severity.experiments.geometry.analyze_current_severity_raa_geometry_fusion_2024_to_2025 import (  # noqa: E402
    build_geometry_feature_sets,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (  # noqa: E402
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/future/current_hurdle_vza_raa_2024_to_2025"
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
DISEASE_2024 = ROOT / "outputs/shared/disease/clean_disease_scores_2024.csv"
DISEASE_2025 = ROOT / "outputs/shared/disease/clean_disease_scores_2025.csv"

COVARIATES = "spectral_plus_week_horizon"
FEATURE_SETS = ["nadir_from_vza", "multiangular_vza", "multiangular_vza_raa"]
TOP_K_VALUES = [30, 50]

REFERENCE_PREDICTIONS = {
    "future_vza_only_selected_42": ROOT
    / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/model_bottleneck_debug/results/predictions/severity_predictions_exploratory_residual_xgboost_forced_top_042_compact_features_compact_anomaly_multiangular_spectral_plus_week_horizon.csv",
    "future_vza_raa_correction": ROOT
    / "outputs/runs/analysis/severity/future/vza_raa_feature_selection_improvement/results/vza_raa_oof_selected_blend_predictions_2025.csv",
    "future_nadir_residual_reference": ROOT
    / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/model_bottleneck_debug/results/predictions/severity_predictions_residual_reliability_filtered_xgboost_compact_anomaly_nadir_spectral_plus_week_horizon.csv",
}
PAIRWISE_COMPARISONS = [
    (
        "direct_vza_raa_top50",
        "future_vza_raa_correction",
        "Direct VZA+RAA top-50 vs existing VZA+RAA correction",
    ),
    (
        "direct_vza_raa_top50",
        "future_vza_only_selected_42",
        "Direct VZA+RAA top-50 vs existing selected VZA-only",
    ),
    (
        "direct_vza_raa_top50",
        "future_nadir_residual_reference",
        "Direct VZA+RAA top-50 vs existing nadir residual reference",
    ),
    (
        "direct_vza_raa_top30",
        "future_vza_raa_correction",
        "Direct VZA+RAA top-30 vs existing VZA+RAA correction",
    ),
]


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"test_future_severity_current_hurdle_vza_raa_{timestamp}.log"
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


def configure_reused_paths() -> None:
    residual_pipeline.ROOT = ROOT
    residual_pipeline.COVARIATES = COVARIATES
    residual_pipeline.OUTPUT_ROOT = OUTPUT_ROOT
    residual_pipeline.RESULTS_DIR = RESULTS_DIR
    residual_pipeline.REPORTS_DIR = REPORTS_DIR
    residual_pipeline.FIGURES_DIR = FIGURES_DIR
    residual_pipeline.PREDICTIONS_DIR = PREDICTIONS_DIR
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "future_current_hurdle_manifest.json"


def read_parquet(path: Path) -> pd.DataFrame:
    started = time.perf_counter()
    frame = pl.read_parquet(path).to_pandas()
    logging.info("Read %s rows x %s cols from %s", frame.shape[0], frame.shape[1], path)
    log_phase(f"read parquet {path.name}", started)
    return frame


def score_reference_predictions() -> pd.DataFrame:
    rows = []
    for label, path in REFERENCE_PREDICTIONS.items():
        if not path.exists():
            logging.warning("Missing reference prediction file: %s", path)
            continue
        pred = pd.read_csv(path)
        score = residual_pipeline.score_predictions(
            pred,
            n_train=166,
            n_features=float("nan"),
            model=label,
            feature_set="reference_existing_future_model",
        )
        score["source"] = "existing_future_reference"
        rows.append(score)
    return pd.DataFrame(rows)


def rmse_from_arrays(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def paired_rmse_delta(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Positive values mean the first prediction frame has lower RMSE."""
    key_cols = ["plot_id", "predictor_week", "target_week"]
    joined = first[key_cols + ["y_true", "y_pred"]].merge(
        second[key_cols + ["y_pred"]].rename(columns={"y_pred": "y_pred_second"}),
        on=key_cols,
        how="inner",
    )
    if len(joined) != len(first):
        raise ValueError(f"Paired {len(joined)} of {len(first)} rows")
    y = joined["y_true"].to_numpy(float)
    first_pred = joined["y_pred"].to_numpy(float)
    second_pred = joined["y_pred_second"].to_numpy(float)
    observed = rmse_from_arrays(y, second_pred) - rmse_from_arrays(y, first_pred)
    groups = [group for _, group in joined.groupby("plot_id", sort=False)]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample = rng.integers(0, len(groups), size=len(groups))
        y_sample = np.concatenate([groups[group_idx]["y_true"].to_numpy(float) for group_idx in sample])
        first_sample = np.concatenate(
            [groups[group_idx]["y_pred"].to_numpy(float) for group_idx in sample]
        )
        second_sample = np.concatenate(
            [groups[group_idx]["y_pred_second"].to_numpy(float) for group_idx in sample]
        )
        boot[idx] = rmse_from_arrays(y_sample, second_sample) - rmse_from_arrays(
            y_sample, first_sample
        )
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "rmse_delta_first_better": observed,
        "rmse_delta_ci_low": float(low),
        "rmse_delta_ci_high": float(high),
        "rmse_delta_prob_gt_zero": float(np.mean(boot > 0)),
    }


def run_hurdle_transfer() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], Path]]:
    started = time.perf_counter()
    vza_2024 = read_parquet(VZA_2024)
    vza_2025 = read_parquet(VZA_2025)
    raa_2024 = read_parquet(RAA_2024)
    raa_2025 = read_parquet(RAA_2025)
    disease_2024 = pd.read_csv(DISEASE_2024)
    disease_2025 = pd.read_csv(DISEASE_2025)
    log_phase("read all inputs", started)

    feature_started = time.perf_counter()
    features = build_geometry_feature_sets(vza_2024, vza_2025, raa_2024, raa_2025)
    log_phase("build VZA/RAA feature sets", feature_started)

    rows = []
    selections = []
    prediction_paths: dict[tuple[str, str], Path] = {}
    for feature_set in FEATURE_SETS:
        table_started = time.perf_counter()
        train_features, test_features = features[feature_set]
        train = build_model_table(train_features, disease_2024)
        test = build_model_table(test_features, disease_2025)
        logging.info(
            "%s future target table: train=%d rows, test=%d rows",
            feature_set,
            len(train),
            len(test),
        )
        log_phase(f"build future table {feature_set}", table_started)

        for top_k in TOP_K_VALUES:
            fit_started = time.perf_counter()
            result, predictions, selection = current_severity.current_hurdle_stability_topk_model(
                train,
                test,
                feature_set,
                top_k=top_k,
                log_positive=False,
            )
            result["source"] = "current_hurdle_architecture_future_target"
            result["architecture_note"] = "direct current-severity hurdle model transferred to future target"
            rows.append(result)
            selections.append(selection)
            prediction_paths[(result["model"], feature_set)] = (
                PREDICTIONS_DIR
                / f"severity_predictions_{residual_pipeline.safe_filename(result['model'])}_{residual_pipeline.safe_filename(feature_set)}_{COVARIATES}.csv"
            )
            logging.info(
                "%s/%s top%s RMSE %.3f MAE %.3f R2 %.3f",
                result["model"],
                feature_set,
                top_k,
                result["rmse"],
                result["mae"],
                result["r2"],
            )
            log_phase(f"fit current hurdle top{top_k} on future {feature_set}", fit_started)

    results = pd.DataFrame(rows)
    selections_df = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    return results, selections_df, prediction_paths


def pairwise_comparison_table() -> pd.DataFrame:
    prediction_frames = {
        "direct_vza_raa_top50": pd.read_csv(
            PREDICTIONS_DIR
            / "severity_predictions_current_hurdle_stability_top50_raw_positive_multiangular_vza_raa_spectral_plus_week_horizon.csv"
        ),
        "direct_vza_raa_top30": pd.read_csv(
            PREDICTIONS_DIR
            / "severity_predictions_current_hurdle_stability_top30_raw_positive_multiangular_vza_raa_spectral_plus_week_horizon.csv"
        ),
        **{
            name: pd.read_csv(path)
            for name, path in REFERENCE_PREDICTIONS.items()
            if path.exists()
        },
    }
    rows = []
    for first, second, label in PAIRWISE_COMPARISONS:
        if first not in prediction_frames or second not in prediction_frames:
            logging.warning("Skipping missing pairwise comparison: %s vs %s", first, second)
            continue
        metrics = paired_rmse_delta(prediction_frames[first], prediction_frames[second])
        rows.append({"comparison": label, "first": first, "second": second, **metrics})
    return pd.DataFrame(rows)


def write_report(
    results: pd.DataFrame,
    references: pd.DataFrame,
    selection_path: Path,
    result_path: Path,
    reference_path: Path,
    pairwise_path: Path,
    pairwise: pd.DataFrame,
    log_path: Path,
) -> Path:
    started = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    best = results.sort_values("rmse").iloc[0]
    best_reference = references.sort_values("rmse").iloc[0] if not references.empty else None
    comparison_text = ""
    if best_reference is not None:
        comparison_text = (
            f" The best existing future reference in this comparison is "
            f"`{best_reference['model']}` with RMSE `{best_reference['rmse']:.3f}`."
        )
    report = [
        "## Results: Future Severity With Current VZA+RAA Hurdle Architecture",
        "",
        "This test reuses the direct current-severity hurdle architecture on the future-severity target. It is intended to check whether the current `VZA + RAA` model family can replace the future `VZA + RAA correction` blend.",
        "",
        "### Transferred Hurdle Models",
        "",
        markdown_table(results.sort_values("rmse").round(4), max_rows=20),
        "",
        "### Existing Future References",
        "",
        markdown_table(references.sort_values("rmse").round(4), max_rows=20)
        if not references.empty
        else "No reference predictions found.",
        "",
        "### Pairwise RMSE Deltas",
        "",
        markdown_table(pairwise.round(4), max_rows=20) if not pairwise.empty else "No pairwise comparisons.",
        "",
        f"**Interpretation**: The best transferred direct hurdle model is `{best['model']}` on `{best['feature_set']}` with RMSE `{best['rmse']:.3f}`.{comparison_text}",
        "",
        "**Outputs**:",
        f"- `{result_path.relative_to(ROOT)}`",
        f"- `{reference_path.relative_to(ROOT)}`",
        f"- `{pairwise_path.relative_to(ROOT)}`",
        f"- `{selection_path.relative_to(ROOT)}`",
        "",
        "**Reproducibility**:",
        "- Train year: `2024`",
        "- Test year: `2025`",
        "- Target: next observed disease-severity week after each predictor week",
        "- Architecture: current-severity hurdle model with grouped ElasticNet feature selection for classifier and positive-severity regressor",
        f"- Feature sets: `{', '.join(FEATURE_SETS)}`",
        f"- Top-k values: `{TOP_K_VALUES}`",
        f"- Covariates: `{COVARIATES}`",
        f"- Log: `{log_path.relative_to(ROOT)}`",
    ]
    path = REPORTS_DIR / "future_severity_current_hurdle_vza_raa_summary.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    logging.info("Wrote report: %s", path)
    log_phase("write markdown report", started)
    return path


def main() -> None:
    total_started = time.perf_counter()
    log_path = setup_logging()
    configure_reused_paths()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results, selections, _ = run_hurdle_transfer()
    references = score_reference_predictions()
    pairwise = pairwise_comparison_table()

    write_started = time.perf_counter()
    result_path = RESULTS_DIR / "future_severity_current_hurdle_vza_raa_model_comparison.csv"
    reference_path = RESULTS_DIR / "future_severity_current_hurdle_existing_reference_scores.csv"
    pairwise_path = RESULTS_DIR / "future_severity_current_hurdle_pairwise_rmse_delta.csv"
    selection_path = RESULTS_DIR / "future_severity_current_hurdle_feature_selection.csv"
    results.to_csv(result_path, index=False)
    references.to_csv(reference_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    selections.to_csv(selection_path, index=False)
    log_phase("write CSV outputs", write_started)

    write_report(
        results,
        references,
        selection_path,
        result_path,
        reference_path,
        pairwise_path,
        pairwise,
        log_path,
    )
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
