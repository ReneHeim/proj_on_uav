"""Quantify Aluco/Capone influence on severity targets and model errors.

The analysis uses saved prediction files only; it does not refit models. It
compares observed severity, nadir errors, selected multiangular errors, and
multiangular improvement by cultivar, treatment, and target week.
"""

from __future__ import annotations

from src.research.common import write_report as persist_report

import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[5]
RESULT_ROOT = ROOT / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/model_bottleneck_debug"
RESULTS_DIR = RESULT_ROOT / "results"
REPORTS_DIR = RESULT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

NADIR_PREDICTIONS = (
    RESULTS_DIR
    / "predictions/severity_predictions_residual_reliability_filtered_xgboost_compact_anomaly_nadir_spectral_plus_week_horizon.csv"
)
SELECTED_MULTI_PREDICTIONS = (
    RESULTS_DIR
    / "predictions/severity_predictions_exploratory_residual_xgboost_forced_top_042_compact_features_compact_anomaly_multiangular_spectral_plus_week_horizon.csv"
)
META_SOURCE = RESULTS_DIR / "residual_debug_by_week_plot.csv"


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_cultivar_influence_severity_{timestamp}.log"
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


def rmse_expr(error_col: str) -> pl.Expr:
    return (pl.col(error_col).pow(2).mean().sqrt()).alias(error_col.replace("err_", "rmse_"))


def read_inputs() -> pl.DataFrame:
    t0 = time.perf_counter()
    meta = (
        pl.read_csv(META_SOURCE)
        .select(["plot_id", "cult", "trt"])
        .unique()
        .with_columns(
            [
                pl.col("cult").str.to_lowercase(),
                pl.col("trt").str.to_lowercase(),
            ]
        )
    )
    nadir = (
        pl.read_csv(NADIR_PREDICTIONS)
        .select(["plot_id", "predictor_week", "target_week", "y_true", "y_pred"])
        .rename({"y_pred": "y_pred_nadir"})
    )
    multi = (
        pl.read_csv(SELECTED_MULTI_PREDICTIONS)
        .select(
            [
                "plot_id",
                "predictor_week",
                "target_week",
                "y_true",
                "y_pred",
                "base_pred",
                "xgb_residual_pred",
            ]
        )
        .rename(
            {
                "y_pred": "y_pred_multiangular",
                "base_pred": "y_pred_ridge_base_multiangular",
                "xgb_residual_pred": "xgb_residual_correction_multiangular",
            }
        )
    )
    joined = (
        nadir.join(multi, on=["plot_id", "predictor_week", "target_week", "y_true"], how="inner")
        .join(meta, on="plot_id", how="left")
        .with_columns(
            [
                (pl.col("y_pred_nadir") - pl.col("y_true")).alias("err_nadir"),
                (pl.col("y_pred_multiangular") - pl.col("y_true")).alias("err_multiangular"),
                (pl.col("y_pred_ridge_base_multiangular") - pl.col("y_true")).alias(
                    "err_ridge_base_multiangular"
                ),
            ]
        )
        .with_columns(
            [
                pl.col("err_nadir").abs().alias("abs_err_nadir"),
                pl.col("err_multiangular").abs().alias("abs_err_multiangular"),
                pl.col("err_ridge_base_multiangular")
                .abs()
                .alias("abs_err_ridge_base_multiangular"),
                (pl.col("err_nadir").abs() - pl.col("err_multiangular").abs()).alias(
                    "abs_error_reduction_multi_vs_nadir"
                ),
                (pl.col("err_nadir").pow(2) - pl.col("err_multiangular").pow(2)).alias(
                    "sq_error_reduction_multi_vs_nadir"
                ),
                (
                    pl.col("err_ridge_base_multiangular").abs() - pl.col("err_multiangular").abs()
                ).alias("abs_error_reduction_residual_vs_ridge"),
                (
                    pl.col("err_ridge_base_multiangular").pow(2) - pl.col("err_multiangular").pow(2)
                ).alias("sq_error_reduction_residual_vs_ridge"),
            ]
        )
    )
    log_phase("read and join prediction/meta files", t0)
    return joined


def grouped_metrics(df: pl.DataFrame, group_cols: list[str]) -> pl.DataFrame:
    return (
        df.group_by(group_cols)
        .agg(
            [
                pl.len().alias("n_rows"),
                pl.col("plot_id").n_unique().alias("n_plots"),
                pl.col("y_true").mean().alias("mean_observed_severity"),
                pl.col("y_true").std().alias("sd_observed_severity"),
                rmse_expr("err_nadir"),
                rmse_expr("err_multiangular"),
                rmse_expr("err_ridge_base_multiangular"),
                pl.col("abs_err_nadir").mean().alias("mae_nadir"),
                pl.col("abs_err_multiangular").mean().alias("mae_multiangular"),
                pl.col("abs_err_ridge_base_multiangular")
                .mean()
                .alias("mae_ridge_base_multiangular"),
                pl.col("err_nadir").mean().alias("bias_nadir"),
                pl.col("err_multiangular").mean().alias("bias_multiangular"),
                pl.col("abs_error_reduction_multi_vs_nadir")
                .mean()
                .alias("mean_abs_error_reduction_multi_vs_nadir"),
                pl.col("sq_error_reduction_multi_vs_nadir")
                .mean()
                .alias("mean_sq_error_reduction_multi_vs_nadir"),
                pl.col("abs_error_reduction_residual_vs_ridge")
                .mean()
                .alias("mean_abs_error_reduction_residual_vs_ridge"),
            ]
        )
        .with_columns(
            [
                (pl.col("rmse_nadir") - pl.col("rmse_multiangular")).alias(
                    "rmse_reduction_multi_vs_nadir"
                ),
                (pl.col("mae_nadir") - pl.col("mae_multiangular")).alias(
                    "mae_reduction_multi_vs_nadir"
                ),
                (pl.col("rmse_ridge_base_multiangular") - pl.col("rmse_multiangular")).alias(
                    "rmse_reduction_residual_vs_ridge"
                ),
            ]
        )
        .sort(group_cols)
    )


def design_matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.ones((df.height, 1), dtype=float)
    arrays = [np.ones(df.height, dtype=float)]
    for col in cols:
        values = df.get_column(col).cast(pl.Utf8).to_list()
        levels = sorted(set(values))
        for level in levels[1:]:
            arrays.append(
                np.asarray([1.0 if value == level else 0.0 for value in values], dtype=float)
            )
    return np.column_stack(arrays)


def linear_r2(y: np.ndarray, x: np.ndarray) -> float:
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    yv = y[valid]
    xv = x[valid]
    if yv.size < 3 or np.nanvar(yv) <= 0:
        return math.nan
    coef, *_ = np.linalg.lstsq(xv, yv, rcond=None)
    pred = xv @ coef
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - np.mean(yv)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan


def influence_metrics(df: pl.DataFrame) -> pl.DataFrame:
    t0 = time.perf_counter()
    outcomes = [
        "y_true",
        "abs_err_nadir",
        "abs_err_multiangular",
        "abs_error_reduction_multi_vs_nadir",
        "sq_error_reduction_multi_vs_nadir",
        "xgb_residual_correction_multiangular",
    ]
    base_cols = ["target_week", "trt"]
    full_cols = ["target_week", "trt", "cult"]
    rows = []
    x_cult = design_matrix(df, ["cult"])
    x_base = design_matrix(df, base_cols)
    x_full = design_matrix(df, full_cols)
    for outcome in outcomes:
        y = df.get_column(outcome).to_numpy().astype(float)
        rows.append(
            {
                "outcome": outcome,
                "cultivar_only_r2_eta2": linear_r2(y, x_cult),
                "week_trt_r2": linear_r2(y, x_base),
                "week_trt_cultivar_r2": linear_r2(y, x_full),
                "incremental_r2_from_cultivar_after_week_trt": linear_r2(y, x_full)
                - linear_r2(y, x_base),
            }
        )
    out = pl.DataFrame(rows)
    log_phase("calculate cultivar influence metrics", t0)
    return out


def write_outputs(df: pl.DataFrame, log_path: Path) -> None:
    t0 = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joined_path = RESULTS_DIR / "cultivar_influence_selected42_prediction_rows.csv"
    by_cult_path = RESULTS_DIR / "cultivar_influence_by_cultivar.csv"
    by_cult_week_path = RESULTS_DIR / "cultivar_influence_by_cultivar_week.csv"
    by_cult_trt_week_path = RESULTS_DIR / "cultivar_influence_by_cultivar_treatment_week.csv"
    influence_path = RESULTS_DIR / "cultivar_influence_incremental_r2.csv"

    by_cult = grouped_metrics(df, ["cult"])
    by_cult_week = grouped_metrics(df, ["cult", "target_week"])
    by_cult_trt_week = grouped_metrics(df, ["cult", "trt", "target_week"])
    influence = influence_metrics(df)

    df.write_csv(joined_path)
    by_cult.write_csv(by_cult_path)
    by_cult_week.write_csv(by_cult_week_path)
    by_cult_trt_week.write_csv(by_cult_trt_week_path)
    influence.write_csv(influence_path)

    report_path = REPORTS_DIR / "cultivar_influence_selected42_severity_summary.md"
    report = build_report(
        by_cult,
        by_cult_week,
        influence,
        joined_path,
        by_cult_path,
        by_cult_week_path,
        by_cult_trt_week_path,
        influence_path,
        log_path,
    )
    persist_report(report_path, report)
    logging.info("Report: %s", report_path)
    log_phase("write outputs", t0)


def markdown_table(df: pl.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    headers = df.columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.rows():
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


def build_report(
    by_cult: pl.DataFrame,
    by_cult_week: pl.DataFrame,
    influence: pl.DataFrame,
    joined_path: Path,
    by_cult_path: Path,
    by_cult_week_path: Path,
    by_cult_trt_week_path: Path,
    influence_path: Path,
    log_path: Path,
) -> str:
    table_cols = [
        "cult",
        "n_rows",
        "n_plots",
        "mean_observed_severity",
        "rmse_nadir",
        "rmse_multiangular",
        "rmse_reduction_multi_vs_nadir",
        "mae_reduction_multi_vs_nadir",
        "bias_multiangular",
    ]
    week_cols = [
        "cult",
        "target_week",
        "mean_observed_severity",
        "rmse_nadir",
        "rmse_multiangular",
        "rmse_reduction_multi_vs_nadir",
        "mean_abs_error_reduction_multi_vs_nadir",
    ]
    influence_cols = [
        "outcome",
        "cultivar_only_r2_eta2",
        "week_trt_r2",
        "week_trt_cultivar_r2",
        "incremental_r2_from_cultivar_after_week_trt",
    ]
    return f"""## Results: Cultivar Influence on Selected Severity Model

### Overall cultivar diagnostics

{markdown_table(by_cult.select(table_cols).with_columns(pl.all().exclude(["cult", "n_rows", "n_plots"]).round(3)))}

### Cultivar by target week

{markdown_table(by_cult_week.select(week_cols).with_columns(pl.all().exclude(["cult", "target_week"]).round(3)))}

### Influence metrics

{markdown_table(influence.select(influence_cols).with_columns(pl.all().exclude("outcome").round(3)))}

**Interpretation**: Cultivar affects the data mainly through disease severity level and subgroup-specific model error. The incremental R2 rows estimate how much extra variance cultivar explains after target week and treatment are already included; larger values mean a stronger cultivar-specific effect that is not just week/treatment imbalance.

**Outputs**:

- Joined prediction rows: `{joined_path.relative_to(ROOT)}`
- By cultivar: `{by_cult_path.relative_to(ROOT)}`
- By cultivar/week: `{by_cult_week_path.relative_to(ROOT)}`
- By cultivar/treatment/week: `{by_cult_trt_week_path.relative_to(ROOT)}`
- Influence R2 metrics: `{influence_path.relative_to(ROOT)}`
- Log: `{log_path.relative_to(ROOT)}`

**Reproducibility**:

- Nadir prediction file: `{NADIR_PREDICTIONS.relative_to(ROOT)}`
- Selected multiangular prediction file: `{SELECTED_MULTI_PREDICTIONS.relative_to(ROOT)}`
- Metadata source: `{META_SOURCE.relative_to(ROOT)}`
- No model fitting or retuning was performed.
"""


def main() -> None:
    log_path = setup_logging()
    started = time.perf_counter()
    df = read_inputs()
    write_outputs(df, log_path)
    log_phase("total runtime", started)


if __name__ == "__main__":
    main()
