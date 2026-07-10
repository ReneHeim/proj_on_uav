"""Explore whether adding more compact multiangular features helps severity RMSE.

This is a diagnostic only. It does not change the frozen model. The script
starts from the full compact anomaly multiangular candidate pool, applies the
existing 2024 stability and train/test reliability diagnostics to every feature,
then force-fits larger ranked top-k feature sets with the same two-stage
Ridge + residual XGBoost architecture.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity.analyze_cross_year_generalization_2024_to_2025 import (  # noqa: E402
    TARGET,
    build_model_table,
    load_2024_disease_with_fallback,
    load_2025_disease_with_fallback,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (  # noqa: E402
    STABILITY_MIN_FREQUENCY,
)
from scripts.analysis.severity.debug_multiangular_rmse_bottleneck import (  # noqa: E402
    COVARIATES,
    fit_tuned_xgboost_residual_with_cols,
    load_cached_features,
    prepare_aligned,
    score_predictions,
    select_stable_features,
    stable_distribution_shift,
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/model_bottleneck_debug"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

FEATURE_SET = "compact_anomaly_multiangular"
K_VALUES = list(range(1, 145))


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"test_extra_compact_features_residual_pipeline_{timestamp}.log"
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


def ensure_dirs() -> None:
    for path in [RESULTS_DIR, FIGURES_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_tables():
    t0 = time.perf_counter()
    features = load_cached_features()
    disease_2024 = load_2024_disease_with_fallback()
    disease_2025, _audit = load_2025_disease_with_fallback()
    train = build_model_table(features[FEATURE_SET][0], disease_2024)
    test = build_model_table(features[FEATURE_SET][1], disease_2025)
    log_phase("load compact multiangular features and severity targets", t0)
    return train, test


def classify_candidates(train, test) -> tuple[pl.DataFrame, list[str], list[str], object, object]:
    t0 = time.perf_counter()
    cols, train_aligned, test_aligned = prepare_aligned(train, test)
    selected_cols, selection = select_stable_features(train_aligned, cols)
    reliability = stable_distribution_shift(train_aligned, test_aligned, cols)

    selection_pl = pl.from_pandas(selection)
    reliability_pl = pl.from_pandas(reliability)
    candidates = (
        pl.DataFrame({"feature": cols})
        .join(
            selection_pl.select(["feature", "selection_frequency", "mean_abs_elasticnet_coef"]),
            on="feature",
            how="left",
        )
        .join(
            reliability_pl.select(
                [
                    "feature",
                    "train_non_null",
                    "test_non_null",
                    "standardized_mean_difference",
                    "passes_reliability_filter",
                ]
            ),
            on="feature",
            how="left",
        )
        .with_columns(
            [
                pl.col("feature").str.starts_with("known__").alias("is_timing_covariate"),
                (pl.col("selection_frequency") >= STABILITY_MIN_FREQUENCY)
                .fill_null(False)
                .alias("passes_stability_filter"),
                pl.col("feature").is_in(selected_cols).alias("selected_by_current_stability_step"),
            ]
        )
        .with_columns(
            (
                pl.col("passes_stability_filter")
                & pl.col("passes_reliability_filter").fill_null(False)
            ).alias("accepted_by_frozen_rules")
        )
        .with_columns(
            pl.when(pl.col("is_timing_covariate"))
            .then(pl.lit("timing_covariate"))
            .when(pl.col("accepted_by_frozen_rules"))
            .then(pl.lit("accepted"))
            .when(~pl.col("passes_stability_filter"))
            .then(pl.lit("discarded_by_stability"))
            .when(~pl.col("passes_reliability_filter").fill_null(False))
            .then(pl.lit("discarded_by_reliability"))
            .otherwise(pl.lit("discarded_other"))
            .alias("decision")
        )
        .with_columns(
            pl.when(pl.col("accepted_by_frozen_rules") & ~pl.col("is_timing_covariate"))
            .then(pl.lit(0))
            .when(~pl.col("is_timing_covariate"))
            .then(pl.lit(1))
            .otherwise(pl.lit(2))
            .alias("forced_inclusion_priority")
        )
        .sort(
            ["forced_inclusion_priority", "selection_frequency", "mean_abs_elasticnet_coef"],
            descending=[False, True, True],
        )
        .with_columns(
            pl.when(~pl.col("is_timing_covariate"))
            .then(pl.int_range(1, pl.len() + 1).over("is_timing_covariate"))
            .otherwise(None)
            .alias("forced_inclusion_rank")
        )
    )
    timing = candidates.filter(pl.col("is_timing_covariate")).get_column("feature").to_list()
    ranked_reflectance = (
        candidates.filter(~pl.col("is_timing_covariate"))
        .sort("forced_inclusion_rank")
        .get_column("feature")
        .to_list()
    )
    log_phase("classify full compact candidate pool", t0)
    return candidates, timing, ranked_reflectance, train_aligned, test_aligned


def evaluate_forced_topk(
    timing: list[str], ranked_reflectance: list[str], train_aligned, test_aligned
) -> pl.DataFrame:
    t0 = time.perf_counter()
    rows: list[dict[str, object]] = []
    max_k = len(ranked_reflectance)
    for k in sorted({value for value in K_VALUES if value <= max_k} | {max_k}):
        cols = timing + ranked_reflectance[:k]
        model_name = f"exploratory_residual_xgboost_forced_top_{k:03d}_compact_features"
        fit_started = time.perf_counter()
        result, predictions, tuning = fit_tuned_xgboost_residual_with_cols(
            train_aligned,
            test_aligned,
            cols,
            model_name,
            FEATURE_SET,
        )
        base_result = score_predictions(
            predictions.assign(y_pred=predictions["base_pred"]),
            len(train_aligned),
            len(cols),
            f"exploratory_ridge_base_forced_top_{k:03d}",
            FEATURE_SET,
        )
        rows.append(
            {
                "feature_set": FEATURE_SET,
                "experiment_status": "exploratory_not_frozen",
                "k_reflectance_features_forced": k,
                "n_timing_features": len(timing),
                "n_total_features": len(cols),
                "base_ridge_rmse": base_result["rmse"],
                "base_ridge_mae": base_result["mae"],
                "base_ridge_r2": base_result["r2"],
                "base_ridge_spearman": base_result["spearman"],
                "residual_gain_rmse": base_result["rmse"] - result["rmse"],
                "xgboost_config": result.get("xgboost_config"),
                "best_iteration": result.get("best_iteration"),
                "eval_rmse_2024": result.get("eval_rmse_2024"),
                "fit_time_s": time.perf_counter() - fit_started,
                **result,
            }
        )
        tuning_out = pl.from_pandas(tuning)
        tuning_out.write_csv(RESULTS_DIR / f"exploratory_extra_compact_tuning_top_{k:03d}.csv")
        logging.info(
            "Forced top-%s: final RMSE %.3f, base RMSE %.3f",
            k,
            result["rmse"],
            base_result["rmse"],
        )
    log_phase("evaluate forced larger top-k compact feature sets", t0)
    return pl.DataFrame(rows)


def summarize_candidates(candidates: pl.DataFrame) -> pl.DataFrame:
    return (
        candidates.group_by("decision")
        .agg(
            [
                pl.len().alias("n_features"),
                pl.col("selection_frequency").mean().alias("mean_selection_frequency"),
                pl.col("passes_reliability_filter").sum().alias("n_passing_reliability"),
            ]
        )
        .sort("n_features", descending=True)
    )


def plot_results(results: pl.DataFrame) -> Path:
    t0 = time.perf_counter()
    data = results.sort("k_reflectance_features_forced")
    x = data.get_column("k_reflectance_features_forced").to_numpy()
    y = data.get_column("rmse").to_numpy()
    y_base = data.get_column("base_ridge_rmse").to_numpy()
    best_idx = int(np.nanargmin(y))

    nadir_rmse = 9.320
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(x, y_base, color="#9a9a9a", lw=1.6, alpha=0.85, label="Ridge base")
    ax.plot(x, y, color="#1f6f5b", lw=2.2, alpha=0.95, label="Ridge + residual XGBoost")
    ax.scatter(x, y, s=9, color="#1f6f5b", alpha=0.75, zorder=3)
    ax.scatter(x, y_base, s=8, color="#9a9a9a", alpha=0.48, zorder=2)
    ax.axvspan(1, 42, color="#1f6f5b", alpha=0.06, label="Selected feature region")
    ax.axvline(42, color="#b5453c", lw=1.5, ls="--", label="Selected count = 42")
    ax.axhline(
        nadir_rmse,
        color="#c46d37",
        lw=1.5,
        ls=":",
        label=f"Nadir residual reference = {nadir_rmse:.2f}",
    )
    ax.scatter([x[best_idx]], [y[best_idx]], s=95, color="#1f6f5b", edgecolor="white", zorder=6)
    ax.set_title("Exploratory forced feature inclusion: severity RMSE", fontsize=12)
    ax.set_xlabel("Forced compact multiangular reflectance features")
    ax.set_ylabel("External-year 2025 RMSE")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(0, 146)
    ax.set_ylim(max(7.7, float(np.nanmin(y)) - 0.35), min(12.8, float(np.nanmax(y)) + 0.4))
    ax.legend(frameon=False, fontsize=8.2, loc="upper right")
    fig.tight_layout()
    path = FIGURES_DIR / "exploratory_extra_compact_features_rmse_curve.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    log_phase("plot exploratory extra-feature curve", t0)
    return path


def markdown_table(table: pl.DataFrame) -> str:
    headers = table.columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table.rows():
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.3f}"
    return str(value)


def write_report(
    candidates: pl.DataFrame,
    summary: pl.DataFrame,
    results: pl.DataFrame,
    figure_path: Path,
    log_path: Path,
) -> Path:
    t0 = time.perf_counter()
    compact_results = (
        results.select(
            [
                "k_reflectance_features_forced",
                "n_total_features",
                "base_ridge_rmse",
                "rmse",
                "mae",
                "r2",
                "spearman",
                "residual_gain_rmse",
            ]
        )
        .sort("k_reflectance_features_forced")
        .with_columns(
            [
                pl.col("base_ridge_rmse").round(3),
                pl.col("rmse").round(3),
                pl.col("mae").round(3),
                pl.col("r2").round(3),
                pl.col("spearman").round(3),
                pl.col("residual_gain_rmse").round(3),
            ]
        )
    )
    accepted = candidates.filter(pl.col("decision") == "accepted").height
    discarded_stability = candidates.filter(pl.col("decision") == "discarded_by_stability").height
    discarded_reliability = candidates.filter(
        pl.col("decision") == "discarded_by_reliability"
    ).height
    best = results.sort("rmse").row(0, named=True)
    report = f"""## Results: Extra Compact Feature Inclusion Diagnostic

This is an exploratory diagnostic and does not change the frozen severity pipeline.

### Candidate acceptance

{markdown_table(summary.with_columns([pl.col("mean_selection_frequency").round(3)]))}

Frozen-rule interpretation:

- Accepted compact reflectance features: `{accepted}`
- Discarded by stability: `{discarded_stability}`
- Discarded by reliability after passing stability: `{discarded_reliability}`

### Forced larger top-k residual models

{markdown_table(compact_results)}

**Interpretation**: The frozen pipeline stops at 41 compact reflectance features because those are the features that pass both the 2024 stability rule and the reliability screen. Forcing additional ranked features into the same two-stage model is exploratory. The best forced setting in this run was top-{int(best['k_reflectance_features_forced'])} with RMSE {float(best['rmse']):.3f}; this should not replace the frozen model unless the selection rule is rebuilt without using 2025 outcomes.

**Outputs**:

- `outputs/runs/analysis/severity/future/compact_distribution_feature_family/model_bottleneck_debug/results/exploratory_extra_compact_feature_acceptance.csv`
- `outputs/runs/analysis/severity/future/compact_distribution_feature_family/model_bottleneck_debug/results/exploratory_extra_compact_feature_acceptance_summary.csv`
- `outputs/runs/analysis/severity/future/compact_distribution_feature_family/model_bottleneck_debug/results/exploratory_extra_compact_forced_topk_results.csv`
- `{figure_path.relative_to(ROOT)}`
- `{log_path.relative_to(ROOT)}`

**Reproducibility**:

- Feature set: `{FEATURE_SET}`.
- Covariates: `{COVARIATES}`.
- Architecture: Ridge base plus residual XGBoost from the existing frozen/debug implementation.
- Feature ranking: 2024-only stability-selection frequency and coefficient magnitude.
- Reliability diagnostic: existing train/test non-null and standardized mean difference screen.
"""
    path = REPORTS_DIR / "exploratory_extra_compact_feature_inclusion_summary.md"
    path.write_text(report, encoding="utf-8")
    log_phase("write exploratory report", t0)
    return path


def main() -> None:
    ensure_dirs()
    log_path = setup_logging()
    started = time.perf_counter()
    train, test = load_tables()
    candidates, timing, ranked_reflectance, train_aligned, test_aligned = classify_candidates(
        train, test
    )
    summary = summarize_candidates(candidates)
    candidates_path = RESULTS_DIR / "exploratory_extra_compact_feature_acceptance.csv"
    summary_path = RESULTS_DIR / "exploratory_extra_compact_feature_acceptance_summary.csv"
    candidates.write_csv(candidates_path)
    summary.write_csv(summary_path)
    results = evaluate_forced_topk(timing, ranked_reflectance, train_aligned, test_aligned)
    results_path = RESULTS_DIR / "exploratory_extra_compact_forced_topk_results.csv"
    results.write_csv(results_path)
    figure_path = plot_results(results)
    report_path = write_report(candidates, summary, results, figure_path, log_path)
    logging.info("Candidate acceptance: %s", candidates_path)
    logging.info("Acceptance summary: %s", summary_path)
    logging.info("Forced top-k results: %s", results_path)
    logging.info("Figure: %s", figure_path)
    logging.info("Report: %s", report_path)
    log_phase("total runtime", started)


if __name__ == "__main__":
    main()
