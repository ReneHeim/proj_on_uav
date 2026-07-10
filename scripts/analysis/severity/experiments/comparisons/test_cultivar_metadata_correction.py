"""Test whether cultivar metadata can correct selected severity predictions.

This is a post-hoc diagnostic on saved predictions. It does not retrain the
reflectance model. It fits small residual correction models on prediction rows
to estimate whether cultivar contributes useful calibration information beyond
week, treatment, and the model prediction itself.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[5]
RESULT_ROOT = ROOT / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/model_bottleneck_debug"
RESULTS_DIR = RESULT_ROOT / "results"
REPORTS_DIR = RESULT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"
INPUT_ROWS = RESULTS_DIR / "cultivar_influence_selected42_prediction_rows.csv"
ALPHAS = np.logspace(-3, 4, 20)


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"test_cultivar_metadata_correction_{timestamp}.log"
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


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred) if len(np.unique(y)) > 1 else math.nan,
    }


def make_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )


def oof_predictions(
    df: pl.DataFrame, numeric_cols: list[str], categorical_cols: list[str], name: str
) -> pl.DataFrame:
    t0 = time.perf_counter()
    import pandas as pd

    pdf = df.to_pandas()
    y = pdf["y_true"].to_numpy(float)
    groups = pdf["plot_id"].to_numpy()
    unique_groups = np.unique(groups)
    splitter = LeaveOneGroupOut() if len(unique_groups) <= 24 else GroupKFold(n_splits=5)
    pred = np.full(len(pdf), np.nan, dtype=float)
    for train_idx, test_idx in splitter.split(pdf, y, groups=groups):
        model = make_pipeline(numeric_cols, categorical_cols)
        model.fit(pdf.iloc[train_idx], y[train_idx])
        pred[test_idx] = model.predict(pdf.iloc[test_idx])
    pred = np.clip(pred, 0.0, float(np.nanmax(y)))
    metrics = regression_metrics(y, pred)
    out = df.select(
        ["plot_id", "cult", "trt", "predictor_week", "target_week", "y_true", "y_pred_multiangular"]
    ).with_columns(
        [
            pl.Series(name="y_pred_corrected", values=pred),
            pl.lit(name).alias("metadata_model"),
        ]
    )
    log_phase(f"OOF metadata correction {name}", t0)
    return out.with_columns(
        [
            pl.lit(metrics["rmse"]).alias("overall_rmse"),
            pl.lit(metrics["mae"]).alias("overall_mae"),
            pl.lit(metrics["r2"]).alias("overall_r2"),
        ]
    )


def coefficient_table(
    df: pl.DataFrame, numeric_cols: list[str], categorical_cols: list[str], name: str
) -> pl.DataFrame:
    import pandas as pd

    pdf = df.to_pandas()
    model = make_pipeline(numeric_cols, categorical_cols)
    model.fit(pdf, pdf["y_true"].to_numpy(float))
    preprocess = model.named_steps["preprocess"]
    feature_names = list(preprocess.get_feature_names_out())
    coefs = model.named_steps["ridge"].coef_
    return pl.DataFrame(
        {
            "metadata_model": name,
            "feature": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
            "ridge_alpha": float(model.named_steps["ridge"].alpha_),
        }
    ).sort("abs_coefficient", descending=True)


def grouped_summary(predictions: pl.DataFrame) -> pl.DataFrame:
    return (
        predictions.with_columns(
            [
                (pl.col("y_pred_corrected") - pl.col("y_true")).alias("err_corrected"),
                (pl.col("y_pred_multiangular") - pl.col("y_true")).alias("err_original"),
            ]
        )
        .with_columns(
            [
                pl.col("err_corrected").abs().alias("abs_err_corrected"),
                pl.col("err_original").abs().alias("abs_err_original"),
            ]
        )
        .group_by(["metadata_model", "cult", "trt"])
        .agg(
            [
                pl.len().alias("n"),
                pl.col("plot_id").n_unique().alias("n_plots"),
                pl.col("abs_err_original").mean().alias("mae_original"),
                pl.col("abs_err_corrected").mean().alias("mae_corrected"),
                (pl.col("err_original").pow(2).mean().sqrt()).alias("rmse_original"),
                (pl.col("err_corrected").pow(2).mean().sqrt()).alias("rmse_corrected"),
            ]
        )
        .with_columns(
            [
                (pl.col("rmse_original") - pl.col("rmse_corrected")).alias(
                    "rmse_reduction_after_metadata_correction"
                ),
                (pl.col("mae_original") - pl.col("mae_corrected")).alias(
                    "mae_reduction_after_metadata_correction"
                ),
            ]
        )
        .sort(["metadata_model", "cult", "trt"])
    )


def main() -> None:
    log_path = setup_logging()
    started = time.perf_counter()
    df = pl.read_csv(INPUT_ROWS).with_columns(
        [
            pl.col("cult").str.to_lowercase(),
            pl.col("trt").str.to_lowercase(),
            pl.col("target_week").cast(pl.Utf8).alias("target_week_cat"),
            pl.col("predictor_week").cast(pl.Utf8).alias("predictor_week_cat"),
        ]
    )
    baseline_metrics = regression_metrics(
        df.get_column("y_true").to_numpy(), df.get_column("y_pred_multiangular").to_numpy()
    )
    specs = {
        "prediction_only": (["y_pred_multiangular"], []),
        "prediction_cultivar": (["y_pred_multiangular"], ["cult"]),
        "prediction_week_only": (["y_pred_multiangular"], ["target_week_cat"]),
        "prediction_week_trt": (["y_pred_multiangular"], ["target_week_cat", "trt"]),
        "prediction_week_cultivar": (["y_pred_multiangular"], ["target_week_cat", "cult"]),
        "prediction_week_trt_cultivar": (
            ["y_pred_multiangular"],
            ["target_week_cat", "trt", "cult"],
        ),
        "prediction_week_trt_cultivar_interactions": (
            ["y_pred_multiangular"],
            ["target_week_cat", "trt", "cult"],
        ),
    }
    predictions = []
    coefs = []
    for name, (numeric_cols, categorical_cols) in specs.items():
        predictions.append(oof_predictions(df, numeric_cols, categorical_cols, name))
        coefs.append(coefficient_table(df, numeric_cols, categorical_cols, name))
    pred_all = pl.concat(predictions)
    coef_all = pl.concat(coefs)
    summary = (
        pred_all.group_by("metadata_model")
        .agg(
            [
                pl.first("overall_rmse").alias("corrected_rmse"),
                pl.first("overall_mae").alias("corrected_mae"),
                pl.first("overall_r2").alias("corrected_r2"),
            ]
        )
        .with_columns(
            [
                pl.lit(baseline_metrics["rmse"]).alias("original_rmse"),
                pl.lit(baseline_metrics["mae"]).alias("original_mae"),
                pl.lit(baseline_metrics["r2"]).alias("original_r2"),
            ]
        )
        .with_columns(
            [
                (pl.col("original_rmse") - pl.col("corrected_rmse")).alias(
                    "rmse_reduction_from_metadata_correction"
                ),
                (pl.col("original_mae") - pl.col("corrected_mae")).alias(
                    "mae_reduction_from_metadata_correction"
                ),
                (pl.col("corrected_r2") - pl.col("original_r2")).alias(
                    "delta_r2_from_metadata_correction"
                ),
            ]
        )
        .sort("corrected_rmse")
    )
    by_group = grouped_summary(pred_all)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "cultivar_metadata_correction_summary.csv"
    by_group_path = RESULTS_DIR / "cultivar_metadata_correction_by_group.csv"
    coef_path = RESULTS_DIR / "cultivar_metadata_correction_coefficients.csv"
    predictions_path = RESULTS_DIR / "cultivar_metadata_correction_oof_predictions.csv"
    summary.write_csv(summary_path)
    by_group.write_csv(by_group_path)
    coef_all.write_csv(coef_path)
    pred_all.write_csv(predictions_path)

    report_path = REPORTS_DIR / "cultivar_metadata_correction_summary.md"
    report_path.write_text(
        build_report(
            summary,
            by_group,
            coef_all,
            summary_path,
            by_group_path,
            coef_path,
            predictions_path,
            log_path,
        ),
        encoding="utf-8",
    )
    logging.info("Report: %s", report_path)
    log_phase("total runtime", started)


def markdown_table(df: pl.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
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
    summary: pl.DataFrame,
    by_group: pl.DataFrame,
    coefs: pl.DataFrame,
    summary_path: Path,
    by_group_path: Path,
    coef_path: Path,
    predictions_path: Path,
    log_path: Path,
) -> str:
    summary_view = summary.with_columns(pl.all().exclude("metadata_model").round(3))
    coef_view = (
        coefs.filter(pl.col("feature").str.contains("cult"))
        .select(["metadata_model", "feature", "coefficient", "abs_coefficient", "ridge_alpha"])
        .with_columns(pl.all().exclude(["metadata_model", "feature"]).round(3))
    )
    return f"""## Results: Does Cultivar Metadata Help?

### Out-of-fold correction performance

{markdown_table(summary_view)}

### Cultivar coefficients in fitted correction models

{markdown_table(coef_view)}

### Group-level correction effects

{markdown_table(by_group.with_columns(pl.all().exclude(["metadata_model", "cult", "trt"]).round(3)), max_rows=12)}

**Interpretation**: These models test whether cultivar metadata can recalibrate saved selected-model predictions under grouped out-of-fold evaluation by plot. If the cultivar model lowers RMSE compared with the original prediction, cultivar carries practical calibration information. If it does not, the cultivar effect is visible descriptively but not useful enough to improve held-out plot predictions.

**Outputs**:

- Summary: `{summary_path.relative_to(ROOT)}`
- Group diagnostics: `{by_group_path.relative_to(ROOT)}`
- Coefficients: `{coef_path.relative_to(ROOT)}`
- OOF predictions: `{predictions_path.relative_to(ROOT)}`
- Log: `{log_path.relative_to(ROOT)}`
"""


if __name__ == "__main__":
    main()
