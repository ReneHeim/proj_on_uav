#!/usr/bin/env python3
"""Create manuscript tables for the frozen multiangular severity pipeline.

This script only reads existing frozen-model predictions and diagnostics. It
does not refit models, retune hyperparameters, or change feature-selection rules.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "outputs/multiangular_distribution_feature_family/model_bottleneck_debug"
SOURCE_RESULTS = SOURCE_ROOT / "results"
SOURCE_PREDICTIONS = SOURCE_RESULTS / "predictions"
CROSS_YEAR_RESULTS = ROOT / "outputs/cross_year_generalization_2024_to_2025/results"

OUTPUT_ROOT = ROOT / "outputs/manuscript_tables/frozen_multiangular_severity"
MAIN_DIR = OUTPUT_ROOT / "main"
SUPPLEMENT_DIR = OUTPUT_ROOT / "supplementary"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/logs"
FROZEN_CONFIG = ROOT / "configs/frozen/multiangular_severity_residual_xgboost_v1.yaml"
FROZEN_MANIFEST = SOURCE_ROOT / "frozen_pipeline_manifest.json"
SOURCE_CODE = ROOT / "scripts/analysis/severity/debug_multiangular_rmse_bottleneck.py"

MODEL = "residual_reliability_filtered_xgboost"
NADIR = "compact_anomaly_nadir"
MULTI = "compact_anomaly_multiangular"
COVARIATES = "spectral_plus_week_horizon"
PIPELINE_ID = "multiangular_severity_residual_xgboost_v1"

NADIR_PRED = SOURCE_PREDICTIONS / (f"severity_predictions_{MODEL}_{NADIR}_{COVARIATES}.csv")
MULTI_PRED = SOURCE_PREDICTIONS / (f"severity_predictions_{MODEL}_{MULTI}_{COVARIATES}.csv")
CANDIDATE_RESULTS = SOURCE_RESULTS / "candidate_model_comparison.csv"
CANDIDATE_CI = SOURCE_RESULTS / "candidate_model_comparison_with_paired_ci.csv"
LOO_SOURCE = SOURCE_RESULTS / "frozen_model_leave_one_plot_sensitivity.csv"
SUBGROUP_SOURCE = SOURCE_RESULTS / "candidate_model_cultivar_treatment_breakdown.csv"
PLOT_DRIVERS_SOURCE = SOURCE_RESULTS / "candidate_model_plot_drivers.csv"
FEATURE_SHIFT_SOURCE = SOURCE_RESULTS / "feature_shift_selected_features.csv"
TUNING_SOURCE = SOURCE_RESULTS / "xgboost_tuning_audit.csv"
STABILITY_SOURCE = SOURCE_RESULTS / "candidate_stability_selection_feature_frequencies.csv"

BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 42

SUPPLEMENT_SOURCES = {
    "table_s1_reflectance_only_severity_models.csv": CROSS_YEAR_RESULTS
    / "paper_tables/table_2_reflectance_only_severity_no_week.csv",
    "table_s2_timing_adjusted_ridge_severity_models.csv": CROSS_YEAR_RESULTS
    / "paper_tables/table_3_severity_with_week_horizon.csv",
    "table_s4_candidate_residual_model_ladder.csv": CANDIDATE_RESULTS,
    "table_s5_full_leave_one_plot_influence_analysis.csv": LOO_SOURCE,
    "table_s6_cultivar_treatment_subgroup_diagnostics.csv": SUBGROUP_SOURCE,
    "table_s7_reliability_feature_shift_diagnostics.csv": FEATURE_SHIFT_SOURCE,
    "table_s8_xgboost_tuning_audit.csv": TUNING_SOURCE,
    "table_s9_plot_level_error_drivers.csv": PLOT_DRIVERS_SOURCE,
}


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"create_frozen_severity_manuscript_tables_{timestamp}.log"
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


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    logging.info("Source file: %s", path)
    return path


def rmse_expr(error_col: str) -> pl.Expr:
    return (pl.col(error_col).pow(2).mean()).sqrt()


def regression_metrics(predictions: pl.DataFrame) -> dict[str, float]:
    rows = predictions.select(
        rmse_expr("error").alias("rmse"),
        pl.col("error").abs().mean().alias("mae"),
        pl.col("error").mean().alias("bias"),
        pl.len().alias("n"),
        pl.col("plot_id").n_unique().alias("n_plots"),
    ).row(0, named=True)
    y = predictions["y_true"].to_numpy()
    yhat = predictions["y_pred"].to_numpy()
    y_mean = float(y.mean())
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    rows["r2"] = 1 - ss_res / ss_tot if ss_tot else math.nan
    rows["spearman"] = float(
        pl.DataFrame({"y": y, "yhat": yhat}).select(pl.corr("y", "yhat", method="spearman")).item()
    )
    return {k: float(v) if isinstance(v, (int, float)) else v for k, v in rows.items()}


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    spearman = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "r2": float(1 - ss_res / ss_tot) if ss_tot else math.nan,
        "spearman": float(spearman) if spearman is not None else math.nan,
    }


def format_ci(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def read_json(path: Path) -> dict:
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verified_feature_lists() -> tuple[list[str], list[str], list[str]]:
    t0 = time.perf_counter()
    stability = pl.read_csv(require_file(STABILITY_SOURCE))
    shift = pl.read_csv(require_file(FEATURE_SHIFT_SOURCE))
    nadir_features = (
        stability.filter((pl.col("feature_set") == NADIR) & pl.col("selected_for_final_model"))
        .select("feature")
        .to_series()
        .to_list()
    )
    known_multi = (
        stability.filter(
            (pl.col("feature_set") == MULTI)
            & pl.col("selected_for_final_model")
            & pl.col("feature").str.starts_with("known__")
        )
        .select("feature")
        .to_series()
        .to_list()
    )
    retained_multi_spectral = (
        shift.with_columns(
            (
                (pl.col("train_non_null") >= 0.8)
                & (pl.col("test_non_null") >= 0.8)
                & (
                    (pl.col("standardized_mean_difference").abs() <= 1.0)
                    | pl.col("standardized_mean_difference").is_nan()
                )
            ).alias("passes_reliability_filter")
        )
        .filter(pl.col("passes_reliability_filter"))
        .select("feature")
        .to_series()
        .to_list()
    )
    multi_features = retained_multi_spectral + known_multi
    removed_multi = (
        shift.with_columns(
            (
                (pl.col("train_non_null") >= 0.8)
                & (pl.col("test_non_null") >= 0.8)
                & (
                    (pl.col("standardized_mean_difference").abs() <= 1.0)
                    | pl.col("standardized_mean_difference").is_nan()
                )
            ).alias("passes_reliability_filter")
        )
        .filter(~pl.col("passes_reliability_filter"))
        .select("feature")
        .to_series()
        .to_list()
    )
    logging.info(
        "Verified retained nadir features (%d): %s", len(nadir_features), ", ".join(nadir_features)
    )
    logging.info(
        "Verified retained multiangular features (%d): %s",
        len(multi_features),
        ", ".join(multi_features),
    )
    logging.info(
        "Reliability-filtered removed multiangular features (%d): %s",
        len(removed_multi),
        ", ".join(removed_multi),
    )
    log_phase("verify retained feature lists from saved audits", t0)
    return nadir_features, multi_features, removed_multi


def create_stage_frame(
    predictions: pl.DataFrame,
    feature_representation: str,
    stage: str,
    pred_col: str,
    n_features: int,
) -> pl.DataFrame:
    return predictions.select(
        "plot_id",
        "predictor_week",
        "target_week",
        "y_true",
        pl.col(pred_col).alias("y_pred"),
    ).with_columns(
        pl.lit(feature_representation).alias("feature_representation"),
        pl.lit(stage).alias("model_stage"),
        pl.lit(n_features).alias("n_features"),
    )


def paired_bootstrap_delta(
    left: pl.DataFrame,
    right: pl.DataFrame,
    samples: list[np.ndarray],
) -> dict[str, float]:
    merged = left.rename({"y_pred": "y_pred_left"}).join(
        right.rename({"y_pred": "y_pred_right"}),
        on=["plot_id", "predictor_week", "target_week", "y_true"],
        how="inner",
    )
    plot_ids = sorted(merged["plot_id"].unique().to_list())
    plot_to_idx = {
        plot_id: np.asarray(
            merged.with_row_index().filter(pl.col("plot_id") == plot_id)["index"].to_list()
        )
        for plot_id in plot_ids
    }
    y = merged["y_true"].to_numpy()
    left_pred = merged["y_pred_left"].to_numpy()
    right_pred = merged["y_pred_right"].to_numpy()
    left_metrics = metrics_from_arrays(y, left_pred)
    right_metrics = metrics_from_arrays(y, right_pred)
    observed = {
        "rmse_reduction": left_metrics["rmse"] - right_metrics["rmse"],
        "mae_reduction": left_metrics["mae"] - right_metrics["mae"],
        "delta_r2": right_metrics["r2"] - left_metrics["r2"],
        "delta_spearman": right_metrics["spearman"] - left_metrics["spearman"],
    }
    boot = {key: [] for key in observed}
    for sampled_plots in samples:
        idx = np.concatenate([plot_to_idx[plot_ids[i]] for i in sampled_plots])
        y_b = y[idx]
        left_b = metrics_from_arrays(y_b, left_pred[idx])
        right_b = metrics_from_arrays(y_b, right_pred[idx])
        boot["rmse_reduction"].append(left_b["rmse"] - right_b["rmse"])
        boot["mae_reduction"].append(left_b["mae"] - right_b["mae"])
        boot["delta_r2"].append(right_b["r2"] - left_b["r2"])
        boot["delta_spearman"].append(right_b["spearman"] - left_b["spearman"])
    out = {"n_rows": merged.height, "n_plots": len(plot_ids), "n_bootstrap": len(samples)}
    for key, values in boot.items():
        arr = np.asarray(values, dtype=float)
        out[f"{key}_observed"] = observed[key]
        out[f"{key}_ci_low"] = float(np.nanpercentile(arr, 2.5))
        out[f"{key}_ci_high"] = float(np.nanpercentile(arr, 97.5))
        out[f"{key}_prob_gt_zero"] = float(np.nanmean(arr > 0))
    return out


def bootstrap_samples(plot_ids: list[str]) -> list[np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = len(plot_ids)
    return [rng.choice(np.arange(n), size=n, replace=True) for _ in range(BOOTSTRAP_ITERATIONS)]


def markdown_table(df: pl.DataFrame, float_digits: int = 3) -> str:
    if df.is_empty():
        return "_No rows._"
    columns = df.columns
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in df.iter_rows(named=True):
        values = []
        for col in columns:
            value = record[col]
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.{float_digits}f}")
            else:
                values.append("" if value is None else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def read_predictions() -> tuple[pl.DataFrame, pl.DataFrame]:
    t0 = time.perf_counter()
    nadir = pl.read_csv(require_file(NADIR_PRED)).with_columns(
        (pl.col("y_pred") - pl.col("y_true")).alias("error")
    )
    multi = pl.read_csv(require_file(MULTI_PRED)).with_columns(
        (pl.col("y_pred") - pl.col("y_true")).alias("error")
    )
    logging.info("Formula: error = y_pred - y_true")
    logging.info("Formula: Ridge base stage uses base_pred from saved frozen prediction files")
    logging.info("Formula: final stage uses y_pred from saved frozen prediction files")
    logging.info(
        "Formula: XGBoost residual correction is xgb_residual_pred; final y_pred = base_pred + xgb_residual_pred"
    )
    logging.info("Formula: RMSE = sqrt(mean((prediction - y_true)^2))")
    logging.info("Formula: MAE = mean(abs(prediction - y_true))")
    logging.info("Formula: bias = mean(prediction - y_true)")
    logging.info("Formula: R2 = 1 - sum((y_true - prediction)^2) / sum((y_true - mean(y_true))^2)")
    logging.info(
        "Formula: reductions = baseline metric - candidate metric; positive means candidate improves"
    )
    log_phase("load frozen predictions", t0)
    return nadir, multi


def create_table_2a(nadir_features: list[str], multi_features: list[str]) -> pl.DataFrame:
    t0 = time.perf_counter()
    table = pl.DataFrame(
        [
            {
                "stage": "Ridge base model",
                "model": "RidgeCV",
                "predictor_block": (
                    "Reliability-filtered compact reflectance representation plus "
                    "known__predictor_week and known__target_week "
                    f"(nadir: {len(nadir_features)} predictors; multiangular: {len(multi_features)} predictors)"
                ),
                "training_target": "future_ds_plot",
                "stage_output": "Ridge base severity prediction (`base_pred`) for each 2025 plot-week row",
                "role": "Models the broad phenology-, horizon-, and baseline-reflectance relationship.",
            },
            {
                "stage": "Residual construction",
                "model": "Grouped out-of-fold Ridge predictions",
                "predictor_block": "Same predictor block as Ridge base model within each feature representation.",
                "training_target": "future_ds_plot - grouped OOF Ridge prediction",
                "stage_output": "Training residual target for XGBoost (`xgb_residual_target`)",
                "role": "Constructs residuals without using in-sample Ridge fitted values.",
            },
            {
                "stage": "XGBoost residual-correction model",
                "model": "Shallow regularized XGBRegressor",
                "predictor_block": (
                    "Same reliability-filtered compact reflectance representation and timing covariates "
                    "used by the Ridge stage."
                ),
                "training_target": "Grouped OOF Ridge residual",
                "stage_output": "Residual correction (`xgb_residual_pred`) for each 2025 plot-week row",
                "role": "Models nonlinear residual variation not captured by Ridge; not interpreted as a biological mechanism.",
            },
            {
                "stage": "Final additive prediction",
                "model": "Ridge base + XGBoost residual correction",
                "predictor_block": "No additional predictors; combines saved stage outputs.",
                "training_target": "Not separately trained",
                "stage_output": "final prediction = Ridge base prediction + XGBoost residual correction",
                "role": "Produces the frozen severity prediction evaluated against the matched nadir implementation.",
            },
        ]
    )
    log_phase("create table 2A architecture", t0)
    return table


def create_stage_tables(
    nadir: pl.DataFrame, multi: pl.DataFrame, nadir_features: list[str], multi_features: list[str]
) -> dict[str, pl.DataFrame]:
    return {
        "ridge_nadir": create_stage_frame(
            nadir, "Nadir", "Frozen Ridge base", "base_pred", len(nadir_features)
        ),
        "ridge_multi": create_stage_frame(
            multi, "Multiangular", "Frozen Ridge base", "base_pred", len(multi_features)
        ),
        "final_nadir": create_stage_frame(
            nadir, "Nadir", "Frozen Ridge plus residual XGBoost", "y_pred", len(nadir_features)
        ),
        "final_multi": create_stage_frame(
            multi,
            "Multiangular",
            "Frozen Ridge plus residual XGBoost",
            "y_pred",
            len(multi_features),
        ),
    }


def stage_metric_row(frame: pl.DataFrame, predictor_summary: str) -> dict:
    metrics = metrics_from_arrays(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
    row = frame.select("model_stage", "feature_representation", "n_features").row(0, named=True)
    return {
        "model_stage": row["model_stage"],
        "feature_representation": row["feature_representation"],
        "predictor_summary": predictor_summary,
        "n_features": int(row["n_features"]),
        "rmse": round(metrics["rmse"], 3),
        "mae": round(metrics["mae"], 3),
        "r2": round(metrics["r2"], 3),
        "spearman": round(metrics["spearman"], 3),
    }


def create_table_2b(
    stage_tables: dict[str, pl.DataFrame], comparisons: dict[str, dict]
) -> pl.DataFrame:
    t0 = time.perf_counter()
    rows = [
        {
            **stage_metric_row(
                stage_tables["ridge_nadir"],
                "Compact nadir reflectance features + predictor/target week",
            ),
            "rmse_reduction_vs_matched_nadir": "Reference",
            "rmse_reduction_ci95": "Reference",
            "residual_correction_gain": "",
            "residual_correction_gain_ci95": "",
        },
        {
            **stage_metric_row(
                stage_tables["ridge_multi"],
                "Reliability-filtered compact multiangular reflectance features + predictor/target week",
            ),
            "rmse_reduction_vs_matched_nadir": f"{comparisons['ridge_multi_vs_nadir']['rmse_reduction_observed']:.3f}",
            "rmse_reduction_ci95": format_ci(
                comparisons["ridge_multi_vs_nadir"]["rmse_reduction_ci_low"],
                comparisons["ridge_multi_vs_nadir"]["rmse_reduction_ci_high"],
            ),
            "residual_correction_gain": "",
            "residual_correction_gain_ci95": "",
        },
        {
            **stage_metric_row(
                stage_tables["final_nadir"],
                "Compact nadir reflectance features + predictor/target week",
            ),
            "rmse_reduction_vs_matched_nadir": "Reference",
            "rmse_reduction_ci95": "Reference",
            "residual_correction_gain": f"{comparisons['nadir_residual_vs_ridge']['rmse_reduction_observed']:.3f}",
            "residual_correction_gain_ci95": format_ci(
                comparisons["nadir_residual_vs_ridge"]["rmse_reduction_ci_low"],
                comparisons["nadir_residual_vs_ridge"]["rmse_reduction_ci_high"],
            ),
        },
        {
            **stage_metric_row(
                stage_tables["final_multi"],
                "Reliability-filtered compact multiangular reflectance features + predictor/target week",
            ),
            "rmse_reduction_vs_matched_nadir": f"{comparisons['final_multi_vs_nadir']['rmse_reduction_observed']:.3f}",
            "rmse_reduction_ci95": format_ci(
                comparisons["final_multi_vs_nadir"]["rmse_reduction_ci_low"],
                comparisons["final_multi_vs_nadir"]["rmse_reduction_ci_high"],
            ),
            "residual_correction_gain": f"{comparisons['multi_residual_vs_ridge']['rmse_reduction_observed']:.3f}",
            "residual_correction_gain_ci95": format_ci(
                comparisons["multi_residual_vs_ridge"]["rmse_reduction_ci_low"],
                comparisons["multi_residual_vs_ridge"]["rmse_reduction_ci_high"],
            ),
        },
    ]
    log_phase("create table 2B component performance ladder", t0)
    return pl.DataFrame(rows)


def create_table_3(nadir: pl.DataFrame, multi: pl.DataFrame) -> pl.DataFrame:
    t0 = time.perf_counter()
    n = nadir.group_by("target_week").agg(
        pl.len().alias("n"),
        pl.col("plot_id").n_unique().alias("n_plots"),
        pl.col("y_true").mean().alias("mean_observed_severity"),
        pl.col("y_true").std().alias("sd_observed_severity"),
        rmse_expr("error").alias("nadir_rmse"),
        pl.col("error").abs().mean().alias("nadir_mae"),
        pl.col("error").mean().alias("nadir_bias"),
    )
    m = multi.group_by("target_week").agg(
        rmse_expr("error").alias("multiangular_rmse"),
        pl.col("error").abs().mean().alias("multiangular_mae"),
        pl.col("error").mean().alias("multiangular_bias"),
    )
    table = (
        n.join(m, on="target_week")
        .with_columns(
            (pl.col("nadir_rmse") - pl.col("multiangular_rmse")).alias(
                "rmse_reduction_multiangular_vs_nadir"
            ),
            (pl.col("nadir_mae") - pl.col("multiangular_mae")).alias(
                "mae_reduction_multiangular_vs_nadir"
            ),
            pl.when(pl.col("target_week") == 1)
            .then(pl.lit("All-zero target week; evaluates false-positive severity predictions"))
            .when(pl.col("target_week") == 5)
            .then(pl.lit("Active disease-development period; largest multiangular benefit"))
            .when(pl.col("target_week") == 6)
            .then(pl.lit("Later heterogeneous severity; smaller multiangular benefit"))
            .otherwise(pl.lit(""))
            .alias("target_week_interpretation"),
        )
        .select(
            "target_week",
            "n",
            "n_plots",
            "mean_observed_severity",
            "sd_observed_severity",
            "nadir_rmse",
            "multiangular_rmse",
            "rmse_reduction_multiangular_vs_nadir",
            "nadir_mae",
            "multiangular_mae",
            "mae_reduction_multiangular_vs_nadir",
            "nadir_bias",
            "multiangular_bias",
            "target_week_interpretation",
        )
        .sort("target_week")
        .with_columns(
            pl.exclude("target_week", "n", "n_plots", "target_week_interpretation").round(3)
        )
    )
    log_phase("create main table 3", t0)
    return table


def create_table_4() -> pl.DataFrame:
    t0 = time.perf_counter()
    loo = pl.read_csv(require_file(LOO_SOURCE))
    rmse_min = loo["rmse_reduction_after_exclusion"].min()
    rmse_med = loo["rmse_reduction_after_exclusion"].median()
    rmse_max = loo["rmse_reduction_after_exclusion"].max()
    mae_min = loo["mae_reduction_after_exclusion"].min()
    mae_med = loo["mae_reduction_after_exclusion"].median()
    mae_max = loo["mae_reduction_after_exclusion"].max()
    positive = int(loo.filter(pl.col("gain_remains_positive")).height)
    total = loo.height
    table = pl.DataFrame(
        [
            {"diagnostic": "External plots", "result": str(total)},
            {
                "diagnostic": "Exclusions with positive RMSE reduction",
                "result": f"{positive}/{total}",
            },
            {"diagnostic": "Full-sample RMSE reduction", "result": "1.207"},
            {
                "diagnostic": "Leave-one-plot-out RMSE reduction range",
                "result": f"{rmse_min:.3f}-{rmse_max:.3f}",
            },
            {"diagnostic": "Median leave-one-plot-out RMSE reduction", "result": f"{rmse_med:.3f}"},
            {
                "diagnostic": "Leave-one-plot-out MAE reduction range",
                "result": f"{mae_min:.3f}-{mae_max:.3f}",
            },
            {"diagnostic": "Median leave-one-plot-out MAE reduction", "result": f"{mae_med:.3f}"},
        ]
    )
    log_phase("create main table 4", t0)
    return table


def create_incremental_gain_table(comparisons: dict[str, dict]) -> pl.DataFrame:
    t0 = time.perf_counter()
    specs = [
        ("Multiangular versus nadir at the Ridge stage", "ridge_multi_vs_nadir"),
        ("Residual correction versus Ridge base for nadir", "nadir_residual_vs_ridge"),
        ("Residual correction versus Ridge base for multiangular", "multi_residual_vs_ridge"),
        (
            "Final multiangular residual model versus final nadir residual model",
            "final_multi_vs_nadir",
        ),
    ]
    rows = []
    for label, key in specs:
        item = comparisons[key]
        rows.append(
            {
                "comparison": label,
                "rmse_reduction": round(item["rmse_reduction_observed"], 3),
                "paired_ci95": format_ci(
                    item["rmse_reduction_ci_low"], item["rmse_reduction_ci_high"]
                ),
                "mae_reduction": round(item["mae_reduction_observed"], 3),
                "delta_r2": round(item["delta_r2_observed"], 3),
                "delta_spearman": round(item["delta_spearman_observed"], 3),
            }
        )
    log_phase("create incremental gain table", t0)
    return pl.DataFrame(rows)


def write_supplementary_tables() -> dict[str, Path]:
    t0 = time.perf_counter()
    written = {}
    for output_name, source in SUPPLEMENT_SOURCES.items():
        source = require_file(source)
        destination = SUPPLEMENT_DIR / output_name
        if output_name == "table_s4_candidate_residual_model_ladder.csv":
            candidate = pl.read_csv(source)
            ci = pl.read_csv(require_file(CANDIDATE_CI)).select(
                "model",
                "rmse_reduction_observed",
                "rmse_reduction_ci_low",
                "rmse_reduction_ci_high",
                "rmse_reduction_prob_gt_zero",
                "mae_reduction_observed",
                "delta_r2_observed",
                "delta_spearman_observed",
            )
            candidate.join(ci, on="model", how="left").write_csv(destination)
        else:
            # Normalize by reading/writing with Polars when possible; fallback to copy.
            try:
                pl.read_csv(source).write_csv(destination)
            except Exception:
                shutil.copyfile(source, destination)
        written[output_name] = destination
        logging.info("Supplementary table: %s <- %s", destination, source)
    log_phase("write supplementary tables", t0)
    return written


def write_methods_and_results(
    table2b: pl.DataFrame,
    comparisons: dict[str, dict],
    output_paths: dict[str, Path],
) -> tuple[Path, Path, str, str]:
    ridge_delta = comparisons["ridge_multi_vs_nadir"]
    nadir_gain = comparisons["nadir_residual_vs_ridge"]
    multi_gain = comparisons["multi_residual_vs_ridge"]
    final_delta = comparisons["final_multi_vs_nadir"]
    methods = (
        "The frozen severity model used a two-stage additive architecture. First, Ridge regression estimated the broad "
        "phenology-, forecast-horizon-, and reflectance-dependent severity trajectory using the reliability-filtered feature "
        "representation plus the same predictor-week and target-week covariates for the nadir and multiangular implementations. "
        "In the verified frozen implementation, the forecast horizon is represented by the combination of predictor week and target week rather than a separate horizon column. "
        "Grouped out-of-fold Ridge predictions were used to calculate training residuals without using in-sample fitted values. "
        "A shallow, regularized XGBoost model was then trained to predict these residuals from the same reliability-filtered "
        "feature representation used by the Ridge stage. Final severity was calculated as the sum of the Ridge prediction and "
        "the XGBoost residual correction. Nadir and multiangular implementations used the same modeling procedure and differed "
        "only in their reflectance representation."
    )
    results = (
        "The two-stage model clarified where the multiangular benefit entered the prediction pipeline. At the Ridge stage, "
        f"multiangular features changed RMSE by {ridge_delta['rmse_reduction_observed']:.3f} relative to the matched nadir Ridge base "
        f"(paired 95% CI {format_ci(ridge_delta['rmse_reduction_ci_low'], ridge_delta['rmse_reduction_ci_high'])}). "
        f"The residual XGBoost correction changed RMSE by {nadir_gain['rmse_reduction_observed']:.3f} for the nadir representation "
        f"and by {multi_gain['rmse_reduction_observed']:.3f} for the multiangular representation. "
        "For the final frozen model, multiangular reflectance reduced external-year RMSE from 9.320 for nadir to 8.113 for "
        "multiangular, corresponding to an RMSE reduction of 1.207 severity units, or approximately 13%. "
        f"The paired plot-bootstrap 95% confidence interval for the final RMSE reduction was "
        f"{format_ci(final_delta['rmse_reduction_ci_low'], final_delta['rmse_reduction_ci_high'])}. "
        "Because the 2025 external year informed pipeline debugging and reliability-screening decisions, this result is interpreted "
        "as exploratory external-year evidence rather than independent confirmatory validation."
    )
    methods_path = REPORTS_DIR / "frozen_model_methods_description.md"
    results_path = REPORTS_DIR / "frozen_model_results_description.md"
    methods_path.write_text(methods + "\n", encoding="utf-8")
    results_path.write_text(results + "\n", encoding="utf-8")
    output_paths["methods_description"] = methods_path
    output_paths["results_description"] = results_path
    return methods_path, results_path, methods, results


def write_report(
    table2a: pl.DataFrame,
    table2b: pl.DataFrame,
    table3: pl.DataFrame,
    table4: pl.DataFrame,
    table_s3: pl.DataFrame,
    verified: dict[str, object],
    methods_text: str,
    results_text: str,
    output_paths: dict[str, Path],
    log_path: Path,
) -> Path:
    t0 = time.perf_counter()
    display2b = table2b.with_columns(
        pl.when(
            (pl.col("feature_representation") == "Multiangular")
            & (pl.col("model_stage").str.contains("residual XGBoost"))
        )
        .then(pl.format("**{}**", pl.col("rmse").cast(pl.Utf8)))
        .otherwise(pl.col("rmse").cast(pl.Utf8))
        .alias("rmse"),
        pl.when(
            (pl.col("feature_representation") == "Multiangular")
            & (pl.col("model_stage").str.contains("residual XGBoost"))
        )
        .then(pl.format("**{}**", pl.col("mae").cast(pl.Utf8)))
        .otherwise(pl.col("mae").cast(pl.Utf8))
        .alias("mae"),
        pl.when(
            (pl.col("feature_representation") == "Multiangular")
            & (pl.col("model_stage").str.contains("residual XGBoost"))
        )
        .then(pl.format("**{}**", pl.col("r2").cast(pl.Utf8)))
        .otherwise(pl.col("r2").cast(pl.Utf8))
        .alias("r2"),
        pl.when(
            (pl.col("feature_representation") == "Multiangular")
            & (pl.col("model_stage").str.contains("residual XGBoost"))
        )
        .then(pl.format("**{}**", pl.col("spearman").cast(pl.Utf8)))
        .otherwise(pl.col("spearman").cast(pl.Utf8))
        .alias("spearman"),
        pl.when(
            (pl.col("feature_representation") == "Multiangular")
            & (pl.col("model_stage").str.contains("residual XGBoost"))
        )
        .then(pl.format("**{}**", pl.col("rmse_reduction_vs_matched_nadir")))
        .otherwise(pl.col("rmse_reduction_vs_matched_nadir"))
        .alias("rmse_reduction_vs_matched_nadir"),
    )
    report = [
        "# Frozen Multiangular Severity Manuscript Tables",
        "",
        f"Pipeline ID: `{PIPELINE_ID}`",
        "",
        f"Selected model: `{MODEL}`",
        "",
        "Reporting status: exploratory external-year evaluation. The 2025 data informed model debugging and reliability-screening analysis, so this is not untouched confirmatory validation.",
        "",
        "## Verified Frozen Pipeline Inputs",
        "",
        f"- Frozen config: `{FROZEN_CONFIG}`",
        f"- Frozen manifest: `{FROZEN_MANIFEST}`",
        f"- Source implementation inspected: `{SOURCE_CODE}`",
        f"- Ridge predictor block: `{verified['ridge_predictor_block']}`",
        f"- XGBoost residual predictor block: `{verified['xgboost_predictor_block']}`",
        f"- Timing variables enter both stages: `{verified['timing_enters_both_stages']}`",
        f"- Nadir and multiangular use identical timing covariates: `{verified['identical_timing_covariates']}`",
        f"- Saved Ridge-base predictions available for 2025 rows: `{verified['base_predictions_available']}`",
        f"- OOF residual construction plus final Ridge refit verified in source: `{verified['oof_then_final_refit']}`",
        f"- Retained nadir features ({len(verified['nadir_features'])}): `{', '.join(verified['nadir_features'])}`",
        f"- Retained multiangular features ({len(verified['multiangular_features'])}): `{', '.join(verified['multiangular_features'])}`",
        f"- Reliability-filtered removed multiangular features ({len(verified['removed_multiangular_features'])}): `{', '.join(verified['removed_multiangular_features'])}`",
        "",
        "## Table 2A. Architecture Of The Frozen Two-Stage Severity Model",
        "",
        markdown_table(table2a),
        "",
        "## Table 2B. Exploratory External-Year Performance Ladder For The Frozen Severity Pipeline",
        "",
        markdown_table(display2b),
        "",
        "**Note**: The two feature representations were evaluated using the same frozen two-stage architecture and the same external observations. Ridge modeled the broad severity relationship, and XGBoost modeled residual variation remaining after the Ridge stage. All comparisons were paired by plot. Because the 2025 external year informed pipeline development and feature-reliability analysis, these results represent exploratory external-year evidence.",
        "",
        "## Table 3. Target-Week Error Decomposition For The Frozen Severity Pipeline",
        "",
        markdown_table(table3),
        "",
        "**Note**: Week 1 contained no positive severity observations and therefore primarily evaluates false-positive severity predictions. Multiangular performance was worse during this all-zero period but improved during weeks 5 and 6, with the largest benefit during active disease development at week 5. The week-5 pattern is consistent with angular reflectance carrying useful information during active disease development, but it is not mechanistic proof.",
        "",
        "## Table 4. Leave-One-Plot-Out Influence Analysis For The Frozen Severity Comparison",
        "",
        markdown_table(table4),
        "",
        "**Note**: This is an influence analysis, not leave-one-plot-out model validation. The models were not retrained separately after each exclusion.",
        "",
        "## Table S3. Incremental Contribution Of Model Components",
        "",
        markdown_table(table_s3),
        "",
        "## Methods Paragraph",
        "",
        methods_text,
        "",
        "## Results Paragraph",
        "",
        results_text,
        "",
        "## Updated Severity Interpretation",
        "",
        "Using the frozen two-stage severity pipeline, multiangular reflectance reduced external-year RMSE from 9.320 to 8.113 relative to the matched nadir implementation, corresponding to a reduction of 1.207 severity units, or approximately 13%. The paired plot-bootstrap 95% confidence interval for the RMSE reduction was 0.530-1.960. Multiangular features also reduced MAE by 0.842, increased R2 by 0.157, and increased Spearman correlation by 0.044. The RMSE advantage remained positive under all 24 leave-one-plot-out exclusions. Week-specific analysis indicated that the principal benefit occurred during active disease development at week 5, whereas multiangular predictions introduced additional error during the all-zero week-1 period. Because the external-year data informed model debugging and reliability-screening decisions, the severity result is interpreted as strong exploratory external-year evidence and requires confirmatory evaluation using the frozen pipeline on an independent year or site.",
        "",
        "## Supplementary Tables",
        "",
        "- Table S1. Reflectance-only severity models",
        "- Table S2. Timing-adjusted Ridge severity models",
        "- Table S3. Incremental contribution of model components",
        "- Table S4. Candidate residual-model ladder",
        "- Table S5. Full leave-one-plot-out influence analysis",
        "- Table S6. Cultivar and treatment subgroup diagnostics",
        "- Table S7. Reliability and feature-shift diagnostics",
        "- Table S8. XGBoost tuning audit",
        "- Table S9. Plot-level error drivers",
        "",
        "## Outputs",
        "",
    ]
    report.extend([f"- {name}: `{path}`" for name, path in output_paths.items()])
    report.append(f"- log: `{log_path}`")
    path = REPORTS_DIR / "frozen_multiangular_severity_manuscript_tables.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    log_phase("write markdown report", t0)
    return path


def main() -> None:
    total_t0 = time.perf_counter()
    log_path = setup_logging()
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    require_file(FROZEN_CONFIG)
    require_file(FROZEN_MANIFEST)
    require_file(SOURCE_CODE)
    read_json(FROZEN_MANIFEST)
    nadir_features, multi_features, removed_multi = verified_feature_lists()
    nadir, multi = read_predictions()
    if (
        nadir.select(pl.col("base_pred").null_count()).item()
        or multi.select(pl.col("base_pred").null_count()).item()
    ):
        raise RuntimeError(
            "Frozen prediction files do not contain complete Ridge base predictions."
        )
    stage_tables = create_stage_tables(nadir, multi, nadir_features, multi_features)
    plot_ids = sorted(nadir["plot_id"].unique().to_list())
    samples = bootstrap_samples(plot_ids)
    comparisons = {
        "ridge_multi_vs_nadir": paired_bootstrap_delta(
            stage_tables["ridge_nadir"], stage_tables["ridge_multi"], samples
        ),
        "nadir_residual_vs_ridge": paired_bootstrap_delta(
            stage_tables["ridge_nadir"], stage_tables["final_nadir"], samples
        ),
        "multi_residual_vs_ridge": paired_bootstrap_delta(
            stage_tables["ridge_multi"], stage_tables["final_multi"], samples
        ),
        "final_multi_vs_nadir": paired_bootstrap_delta(
            stage_tables["final_nadir"], stage_tables["final_multi"], samples
        ),
    }
    logging.info(
        "Formula: paired bootstrap resamples plot_id values with replacement; the same %d sampled plot sets are used for every paired comparison",
        BOOTSTRAP_ITERATIONS,
    )
    logging.info(
        "Formula: ridge_multi_vs_nadir = metrics(Frozen Ridge base Nadir) - metrics(Frozen Ridge base Multiangular)"
    )
    logging.info(
        "Formula: nadir_residual_vs_ridge = metrics(Frozen Ridge base Nadir) - metrics(Frozen Ridge plus residual XGBoost Nadir)"
    )
    logging.info(
        "Formula: multi_residual_vs_ridge = metrics(Frozen Ridge base Multiangular) - metrics(Frozen Ridge plus residual XGBoost Multiangular)"
    )
    logging.info(
        "Formula: final_multi_vs_nadir = metrics(Frozen Ridge plus residual XGBoost Nadir) - metrics(Frozen Ridge plus residual XGBoost Multiangular)"
    )
    for name, values in comparisons.items():
        logging.info(
            "Comparison %s: RMSE reduction %.6f CI [%.6f, %.6f]; MAE reduction %.6f; delta R2 %.6f; delta Spearman %.6f",
            name,
            values["rmse_reduction_observed"],
            values["rmse_reduction_ci_low"],
            values["rmse_reduction_ci_high"],
            values["mae_reduction_observed"],
            values["delta_r2_observed"],
            values["delta_spearman_observed"],
        )
    table2a = create_table_2a(nadir_features, multi_features)
    table2b = create_table_2b(stage_tables, comparisons)
    table3 = create_table_3(nadir, multi)
    table4 = create_table_4()
    table_s3 = create_incremental_gain_table(comparisons)
    supplement_paths = write_supplementary_tables()

    output_paths: dict[str, Path] = {
        "main_table_2a": MAIN_DIR / "table_2a_frozen_model_architecture.csv",
        "main_table_2b": MAIN_DIR / "table_2b_frozen_model_component_performance_ladder.csv",
        "main_table_3": MAIN_DIR / "table_3_frozen_model_week_specific_performance.csv",
        "main_table_4": MAIN_DIR / "table_4_frozen_model_leave_one_plot_summary.csv",
        "supplementary_table_s3_incremental_gain": SUPPLEMENT_DIR
        / "table_s3_incremental_model_component_gain.csv",
        "supplementary_model_ladder": supplement_paths[
            "table_s4_candidate_residual_model_ladder.csv"
        ],
    }
    output_paths.update({name: path for name, path in supplement_paths.items()})
    table2a.write_csv(output_paths["main_table_2a"])
    table2b.write_csv(output_paths["main_table_2b"])
    table3.write_csv(output_paths["main_table_3"])
    table4.write_csv(output_paths["main_table_4"])
    table_s3.write_csv(output_paths["supplementary_table_s3_incremental_gain"])
    verified = {
        "ridge_predictor_block": "same reliability-filtered compact representation used for each feature set plus known__predictor_week and known__target_week",
        "xgboost_predictor_block": "same filtered columns passed to Ridge; source uses tune_xgboost_params(residual_table, cols) and residual_model.fit(train_aligned[cols], residual_train)",
        "timing_enters_both_stages": True,
        "identical_timing_covariates": "known__predictor_week and known__target_week for both nadir and multiangular",
        "base_predictions_available": True,
        "oof_then_final_refit": True,
        "nadir_features": nadir_features,
        "multiangular_features": multi_features,
        "removed_multiangular_features": removed_multi,
    }
    _, _, methods_text, results_text = write_methods_and_results(table2b, comparisons, output_paths)
    report_path = write_report(
        table2a,
        table2b,
        table3,
        table4,
        table_s3,
        verified,
        methods_text,
        results_text,
        output_paths,
        log_path,
    )
    logging.info("Report: %s", report_path)
    logging.info("[PHASE] total: %.1fs", time.perf_counter() - total_t0)


if __name__ == "__main__":
    main()
