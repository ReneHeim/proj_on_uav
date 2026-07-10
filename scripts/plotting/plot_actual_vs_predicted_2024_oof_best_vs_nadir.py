#!/usr/bin/env python3
"""Plot 2024 grouped-OOF observed versus predicted severity.

This mirrors the 2025 held-out observed-vs-predicted figure, but evaluates the
same model families on 2024 using grouped out-of-fold predictions by plot.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
MPLCONFIG_DIR = ROOT / "outputs/.matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.early_warning.analyze_early_warning_severity_2024 import (  # noqa: E402
    build_model_table as build_future_model_table,
)
from scripts.analysis.severity import (  # noqa: E402
    analyze_current_plot_severity_2024_to_2025 as current_severity,
)
from scripts.analysis.severity import debug_multiangular_rmse_bottleneck as residual_pipeline  # noqa: E402
from scripts.analysis.severity.experiments.geometry.analyze_current_severity_raa_geometry_fusion_2024_to_2025 import (  # noqa: E402
    RAA_2024,
    RAA_2025,
    VZA_2024,
    VZA_2025,
    build_geometry_feature_sets,
)

OUTPUT_ROOT = ROOT / "outputs/deliverables/presentation/severity_actual_vs_predicted_2024_oof"
FIGURES_DIR = OUTPUT_ROOT / "figures"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

DISEASE_2024 = ROOT / "outputs/shared/disease/clean_disease_scores_2024.csv"
COMPACT_FEATURE_DIR = ROOT / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/results"

TARGET = current_severity.TARGET
SEED = current_severity.SEED

PALETTE = {
    "navy": "#0B132B",
    "coral": "#FF6B6B",
    "gold": "#F6C85F",
    "grey": "#6b7280",
    "light_grey": "#e5e7eb",
}
CULTIVAR_COLORS = {"aluco": PALETTE["navy"], "capone": PALETTE["coral"]}
CULTIVAR_LABELS = {"aluco": "Aluco", "capone": "Capone"}

PANEL_ORDER = [
    ("Current-week severity", "Selected multiangular"),
    ("Current-week severity", "Standard nadir"),
    ("Future severity", "Selected multiangular"),
    ("Future severity", "Standard nadir"),
]


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"plot_actual_vs_predicted_2024_oof_best_vs_nadir_{timestamp}.log"
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


def configure_residual_paths(covariates: str) -> None:
    residual_pipeline.ROOT = ROOT
    residual_pipeline.INPUT_RESULTS_DIR = COMPACT_FEATURE_DIR
    residual_pipeline.OUTPUT_ROOT = OUTPUT_ROOT
    residual_pipeline.RESULTS_DIR = RESULTS_DIR
    residual_pipeline.REPORTS_DIR = REPORTS_DIR
    residual_pipeline.FIGURES_DIR = FIGURES_DIR
    residual_pipeline.PREDICTIONS_DIR = RESULTS_DIR / "intermediate_predictions"
    residual_pipeline.COVARIATES = covariates


def read_parquet(path: Path) -> pd.DataFrame:
    started = time.perf_counter()
    frame = pl.read_parquet(path).to_pandas()
    logging.info("Read %s rows x %s cols from %s", frame.shape[0], frame.shape[1], path)
    log_phase(f"read parquet {path.name}", started)
    return frame


def load_geometry_features() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    started = time.perf_counter()
    features = build_geometry_feature_sets(
        read_parquet(VZA_2024),
        read_parquet(VZA_2025),
        read_parquet(RAA_2024),
        read_parquet(RAA_2025),
    )
    log_phase("load and build geometry feature sets", started)
    return features


def load_compact_nadir_2024() -> pd.DataFrame:
    started = time.perf_counter()
    configure_residual_paths("spectral_plus_week")
    residual_pipeline.FEATURE_SETS = ["compact_anomaly_nadir"]
    features = residual_pipeline.load_cached_features()
    log_phase("load compact nadir features", started)
    return features["compact_anomaly_nadir"][0]


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(float(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum((y_true - y_pred) ** 2) / denom) if denom else math.nan
    bias = float(np.mean(y_pred - y_true))
    return {"rmse": rmse, "mae": mae, "r2": r2, "bias": bias}


def select_hurdle_columns(
    train_aligned: pd.DataFrame,
    cols: list[str],
    *,
    top_k: int,
    selection: str,
    log_positive: bool,
) -> tuple[list[str], list[str], pd.DataFrame]:
    y_train = train_aligned[TARGET].to_numpy(float)
    groups = train_aligned["plot_id"].to_numpy()
    force_cols = {"known__predictor_week"}.intersection(cols)
    y_present = (y_train > 0).astype(float)

    if selection == "stability":
        classifier_cols, classifier_ranking = current_severity.stable_ranked_columns(
            train_aligned,
            cols,
            y_present,
            groups,
            top_k,
            force_cols,
            role="classifier",
        )
    elif selection == "ranked":
        classifier_cols = current_severity.ranked_top_columns(
            train_aligned, cols, y_present, top_k, force_cols
        )
        classifier_ranking = pd.DataFrame(
            {
                "role": "classifier",
                "feature": classifier_cols,
                "rank": np.arange(1, len(classifier_cols) + 1),
                "selected_for_final_model": True,
                "selection_strategy": f"correlation_top{top_k}",
            }
        )
    else:
        raise ValueError(f"Unknown selection strategy: {selection}")

    positive_mask = y_train > 0
    if positive_mask.sum() >= 5:
        reg_target = np.log1p(y_train[positive_mask]) if log_positive else y_train[positive_mask]
        if selection == "stability":
            regressor_cols, regressor_ranking = current_severity.stable_ranked_columns(
                train_aligned.loc[positive_mask],
                cols,
                reg_target,
                train_aligned.loc[positive_mask, "plot_id"].to_numpy(),
                top_k,
                force_cols,
                role="positive_severity_regressor",
            )
        else:
            regressor_cols = current_severity.ranked_top_columns(
                train_aligned.loc[positive_mask], cols, reg_target, top_k, force_cols
            )
            regressor_ranking = pd.DataFrame(
                {
                    "role": "positive_severity_regressor",
                    "feature": regressor_cols,
                    "rank": np.arange(1, len(regressor_cols) + 1),
                    "selected_for_final_model": True,
                    "selection_strategy": f"correlation_top{top_k}",
                }
            )
    else:
        regressor_cols = classifier_cols
        regressor_ranking = classifier_ranking.copy()
        regressor_ranking["role"] = "positive_severity_regressor"

    selection_table = pd.concat([classifier_ranking, regressor_ranking], ignore_index=True)
    return classifier_cols, regressor_cols, selection_table


def grouped_oof_prediction_frame(
    table: pd.DataFrame,
    *,
    application: str,
    model_label: str,
    model: str,
    feature_set: str,
    covariates: str,
    top_k: int,
    selection: str,
    log_positive: bool = False,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    configure_residual_paths(covariates)
    cultivar_map = None
    if "cult" in table.columns:
        cultivar_map = table[["plot_id", "predictor_week", "target_week", "cult"]].drop_duplicates()
    cols, train_aligned, _ = residual_pipeline.prepare_aligned(table, table)
    classifier_cols, regressor_cols, selection_table = select_hurdle_columns(
        train_aligned,
        cols,
        top_k=top_k,
        selection=selection,
        log_positive=log_positive,
    )
    fit_started = time.perf_counter()
    y_pred = current_severity.grouped_oof_hurdle_predictions(
        train_aligned, classifier_cols, regressor_cols, log_positive
    )
    log_phase(f"grouped OOF fit/predict {application} {model_label}", fit_started)

    out_cols = ["plot_id", "predictor_week", "target_week", TARGET]
    predictions = train_aligned[out_cols].copy().rename(columns={TARGET: "y_true"})
    if cultivar_map is not None:
        predictions = predictions.merge(
            cultivar_map, on=["plot_id", "predictor_week", "target_week"], how="left"
        )
    if "cult" not in predictions.columns:
        predictions["cult"] = "unknown"
    predictions["cult"] = predictions["cult"].fillna("unknown").astype(str).str.lower()
    predictions["model"] = model
    predictions["model_label"] = model_label
    predictions["feature_set"] = feature_set
    predictions["application"] = application
    predictions["covariates"] = covariates
    predictions["year"] = 2024
    predictions["horizon"] = predictions["target_week"] - predictions["predictor_week"]
    predictions["y_pred"] = y_pred
    predictions = predictions[
        [
            "year",
            "application",
            "model_label",
            "plot_id",
            "cult",
            "predictor_week",
            "target_week",
            "horizon",
            "model",
            "feature_set",
            "covariates",
            "y_true",
            "y_pred",
        ]
    ]

    y_true = predictions["y_true"].to_numpy(float)
    metrics = {
        "year": 2024,
        "application": application,
        "model_label": model_label,
        "model": model,
        "feature_set": feature_set,
        "covariates": covariates,
        "selection": selection,
        "top_k": top_k,
        "n": len(predictions),
        "n_plots": predictions["plot_id"].nunique(),
        "horizons_present": ",".join(
            str(value) for value in sorted(predictions["horizon"].dropna().unique())
        ),
        "n_features": len(sorted(set(classifier_cols + regressor_cols))),
        "classifier_features": len(classifier_cols),
        "regressor_features": len(regressor_cols),
        **regression_scores(y_true, y_pred),
    }

    selection_table = selection_table.copy()
    selection_table["year"] = 2024
    selection_table["application"] = application
    selection_table["model_label"] = model_label
    selection_table["model"] = model
    selection_table["feature_set"] = feature_set
    selection_table["covariates"] = covariates
    selection_table["top_k"] = top_k
    return predictions, metrics, selection_table


def build_2024_oof_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    disease_2024 = pd.read_csv(DISEASE_2024)
    geometry_features = load_geometry_features()
    compact_nadir = load_compact_nadir_2024()

    jobs = []

    current_vza_raa = current_severity.build_current_model_table(
        geometry_features["multiangular_vza_raa"][0], disease_2024
    )
    jobs.append(
        grouped_oof_prediction_frame(
            current_vza_raa,
            application="Current-week severity",
            model_label="Selected multiangular",
            model="current_hurdle_stability_top30_raw_positive",
            feature_set="multiangular_vza_raa",
            covariates="spectral_plus_week",
            top_k=30,
            selection="stability",
        )
    )

    current_nadir = current_severity.build_current_model_table(compact_nadir, disease_2024)
    jobs.append(
        grouped_oof_prediction_frame(
            current_nadir,
            application="Current-week severity",
            model_label="Standard nadir",
            model="current_hurdle_top20_raw_positive",
            feature_set="compact_anomaly_nadir",
            covariates="spectral_plus_week",
            top_k=20,
            selection="ranked",
        )
    )

    future_vza_raa = build_future_model_table(
        geometry_features["multiangular_vza_raa"][0], disease_2024
    )
    jobs.append(
        grouped_oof_prediction_frame(
            future_vza_raa,
            application="Future severity",
            model_label="Selected multiangular",
            model="current_hurdle_stability_top50_raw_positive",
            feature_set="multiangular_vza_raa",
            covariates="spectral_plus_week_horizon",
            top_k=50,
            selection="stability",
        )
    )

    future_nadir = build_future_model_table(geometry_features["nadir_from_vza"][0], disease_2024)
    jobs.append(
        grouped_oof_prediction_frame(
            future_nadir,
            application="Future severity",
            model_label="Standard nadir",
            model="current_hurdle_stability_top50_raw_positive",
            feature_set="nadir_from_vza",
            covariates="spectral_plus_week_horizon",
            top_k=50,
            selection="stability",
        )
    )

    predictions = pd.concat([item[0] for item in jobs], ignore_index=True)
    metrics = pd.DataFrame([item[1] for item in jobs])
    selections = pd.concat([item[2] for item in jobs], ignore_index=True)
    metrics["order"] = metrics.apply(
        lambda row: PANEL_ORDER.index((row["application"], row["model_label"])), axis=1
    )
    metrics = metrics.sort_values("order").drop(columns="order")
    log_phase("build all 2024 OOF predictions", started)
    return predictions, metrics, selections


def jitter(values: np.ndarray, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return values + rng.normal(0.0, scale, size=values.shape)


def scatter_by_cultivar(ax: plt.Axes, subset: pd.DataFrame, axis_limit: float, seed: int) -> None:
    for offset, cultivar in enumerate(["aluco", "capone"]):
        cult_data = subset[subset["cult"].astype(str).str.lower() == cultivar]
        if cult_data.empty:
            continue
        y_true = cult_data["y_true"].to_numpy(float)
        y_pred = cult_data["y_pred"].to_numpy(float)
        ax.scatter(
            jitter(y_true, 0.10 if axis_limit <= 15 else 0.35, seed=seed + offset),
            jitter(y_pred, 0.10 if axis_limit <= 15 else 0.35, seed=seed + 100 + offset),
            s=38,
            color=CULTIVAR_COLORS[cultivar],
            alpha=0.76,
            edgecolor="white",
            linewidth=0.45,
            label=CULTIVAR_LABELS[cultivar],
        )


def axis_limit_for(predictions: pd.DataFrame, application: str) -> float:
    app_data = predictions[predictions["application"] == application]
    max_axis = max(float(app_data["y_true"].max()), float(app_data["y_pred"].max()))
    return max(10.0, math.ceil(max_axis / 5.0) * 5.0)


def draw_panel(
    ax: plt.Axes,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    application: str,
    model_label: str,
    *,
    axis_limit: float,
    seed: int,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    subset = predictions[
        (predictions["application"] == application) & (predictions["model_label"] == model_label)
    ]
    metric = metrics[
        (metrics["application"] == application) & (metrics["model_label"] == model_label)
    ].iloc[0]

    scatter_by_cultivar(ax, subset, axis_limit, seed)
    ax.plot(
        [0, axis_limit],
        [0, axis_limit],
        color=PALETTE["grey"],
        linewidth=1.1,
        linestyle=(0, (4, 4)),
    )
    ax.set_xlim(-0.5, axis_limit + 0.5)
    ax.set_ylim(-0.5, axis_limit + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=PALETTE["light_grey"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"{application}\n{model_label}", fontsize=13, fontweight="bold", color=PALETTE["navy"])
    ax.text(
        0.04,
        0.96,
        f"RMSE {metric['rmse']:.2f}\nR2 {metric['r2']:.2f}\nMAE {metric['mae']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=PALETTE["navy"],
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": PALETTE["light_grey"],
            "alpha": 0.92,
        },
    )
    if show_xlabel:
        ax.set_xlabel("Observed severity", fontsize=11, color=PALETTE["navy"])
    if show_ylabel:
        ax.set_ylabel("Predicted severity", fontsize=11, color=PALETTE["navy"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["navy"])
    ax.spines["bottom"].set_color(PALETTE["navy"])


def save_figure(fig: plt.Figure, output_prefix: Path) -> list[Path]:
    paths = [output_prefix.with_suffix(ext) for ext in [".png", ".pdf", ".svg"]]
    for path in paths:
        fig.savefig(path, dpi=300)
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    return paths


def write_combined_plot(predictions: pd.DataFrame, metrics: pd.DataFrame) -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), sharex=False, sharey=False)
    fig.patch.set_facecolor("white")
    applications = ["Current-week severity", "Future severity"]
    models = ["Selected multiangular", "Standard nadir"]
    for row_idx, application in enumerate(applications):
        axis_limit = axis_limit_for(predictions, application)
        for col_idx, model_label in enumerate(models):
            draw_panel(
                axes[row_idx, col_idx],
                predictions,
                metrics,
                application,
                model_label,
                axis_limit=axis_limit,
                seed=100 + row_idx * 10 + col_idx,
                show_xlabel=row_idx == len(applications) - 1,
                show_ylabel=col_idx == 0,
            )
    fig.suptitle(
        "2024 Observed vs Predicted Severity",
        fontsize=22,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.985,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.02,
        "Points are real 2024 grouped out-of-fold plot-week predictions, colored by cultivar; jitter only separates overlaps.",
        ha="center",
        fontsize=9.5,
        color=PALETTE["grey"],
    )
    fig.tight_layout(rect=[0.03, 0.045, 1, 0.91])
    paths = save_figure(fig, FIGURES_DIR / "actual_vs_predicted_2024_oof_best_vs_nadir")
    log_phase("write combined observed-predicted plot", started)
    return paths


def write_application_plot(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    application: str,
    stem: str,
) -> list[Path]:
    started = time.perf_counter()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), sharex=False, sharey=False)
    fig.patch.set_facecolor("white")
    axis_limit = axis_limit_for(predictions, application)
    for col_idx, model_label in enumerate(["Selected multiangular", "Standard nadir"]):
        draw_panel(
            axes[col_idx],
            predictions,
            metrics,
            application,
            model_label,
            axis_limit=axis_limit,
            seed=300 + col_idx,
            show_xlabel=True,
            show_ylabel=col_idx == 0,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=2, frameon=False)
    fig.suptitle(
        f"2024 {application}: Observed vs Predicted",
        fontsize=19,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.99,
    )
    fig.text(
        0.5,
        0.02,
        "Grouped out-of-fold plot-week predictions, colored by cultivar.",
        ha="center",
        fontsize=9.5,
        color=PALETTE["grey"],
    )
    fig.tight_layout(rect=[0.03, 0.06, 1, 0.84])
    paths = save_figure(fig, FIGURES_DIR / stem)
    log_phase(f"write {stem}", started)
    return paths


def markdown_table(metrics: pd.DataFrame) -> str:
    display = metrics[
        [
            "application",
            "model_label",
            "rmse",
            "mae",
            "r2",
            "bias",
            "horizons_present",
            "n",
            "n_plots",
            "n_features",
            "model",
            "feature_set",
        ]
    ].copy()
    for col in ["rmse", "mae", "r2", "bias"]:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_outputs(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    selections: pd.DataFrame,
    figure_paths: list[Path],
    log_path: Path,
) -> Path:
    started = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = RESULTS_DIR / "actual_vs_predicted_2024_oof_predictions.csv"
    metrics_path = RESULTS_DIR / "actual_vs_predicted_2024_oof_metrics.csv"
    selection_path = RESULTS_DIR / "actual_vs_predicted_2024_oof_feature_selection.csv"
    predictions.to_csv(prediction_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    selections.to_csv(selection_path, index=False)
    report = f"""## Results: 2024 Observed vs Predicted Severity

{markdown_table(metrics)}

**Interpretation**: This is the 2024 analogue of the held-out 2025 observed-vs-predicted figure. The model fitting is grouped out-of-fold by plot, so each plotted row is predicted by a model that did not fit on that plot. Feature selection is fixed from the full 2024 table to mirror the external 2025 recipe, so this is not a fully nested feature-selection estimate. Current-week severity uses same-week targets; future severity uses the original future-target construction with all available future horizons.

**Outputs**:
- `{prediction_path.relative_to(ROOT)}`
- `{metrics_path.relative_to(ROOT)}`
- `{selection_path.relative_to(ROOT)}`
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in figure_paths)}

**Reproducibility**:
- Evaluation year: `2024`
- Split: `GroupKFold`, grouped by `plot_id`; feature selection fixed before OOF fitting
- Current-week target: `target_week - predictor_week = 0`
- Future-severity target: next observed disease-severity week after each predictor week; no strict next-week filter
- Current selected multiangular: `multiangular_vza_raa`, `current_hurdle_stability_top30_raw_positive`
- Current standard nadir: `compact_anomaly_nadir`, `current_hurdle_top20_raw_positive`
- Future selected multiangular: `multiangular_vza_raa`, `current_hurdle_stability_top50_raw_positive`
- Future standard nadir: `nadir_from_vza`, `current_hurdle_stability_top50_raw_positive`
- Log: `{log_path.relative_to(ROOT)}`
"""
    report_path = REPORTS_DIR / "actual_vs_predicted_2024_oof_summary.md"
    report_path.write_text(report, encoding="utf-8")
    logging.info("Wrote report: %s", report_path)
    log_phase("write outputs", started)
    return report_path


def main() -> None:
    total_started = time.perf_counter()
    log_path = setup_logging()
    predictions, metrics, selections = build_2024_oof_predictions()
    figure_paths = [
        *write_combined_plot(predictions, metrics),
        *write_application_plot(
            predictions,
            metrics,
            "Current-week severity",
            "actual_vs_predicted_2024_oof_current_week_best_vs_nadir_by_cultivar",
        ),
        *write_application_plot(
            predictions,
            metrics,
            "Future severity",
            "actual_vs_predicted_2024_oof_future_best_vs_nadir_by_cultivar",
        ),
    ]
    write_outputs(predictions, metrics, selections, figure_paths, log_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
