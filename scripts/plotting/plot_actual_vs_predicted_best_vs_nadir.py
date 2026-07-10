#!/usr/bin/env python3
"""Plot observed versus predicted severity for best multiangular and nadir baselines."""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
MPLCONFIG_DIR = ROOT / "outputs/.matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.research.common import (
    artifact_path,
    configure_logging,
    log_phase,
    markdown_table as render_markdown_table,
    regression_metrics,
    save_figure as persist_figure,
    write_report as persist_report,
)

OUTPUT_ROOT = ROOT / "outputs/deliverables/presentation/severity_actual_vs_predicted_2025"
FIGURES_DIR = OUTPUT_ROOT / "figures"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = OUTPUT_ROOT / "logs"

PALETTE = {
    "navy": "#0B132B",
    "coral": "#FF6B6B",
    "gold": "#F6C85F",
    "grey": "#6b7280",
    "light_grey": "#e5e7eb",
}
CULTIVAR_COLORS = {"aluco": PALETTE["navy"], "capone": PALETTE["coral"]}
CULTIVAR_LABELS = {"aluco": "Aluco", "capone": "Capone"}
CULTIVAR_SOURCE = artifact_path("severity.cultivar_metadata_2025")

PANEL_SPECS = [
    {
        "application": "Current-week severity",
        "model_label": "Selected multiangular",
        "prediction_artifact": "severity.current.selected_multiangular_predictions",
        "horizon": 0,
    },
    {
        "application": "Current-week severity",
        "model_label": "Standard nadir",
        "prediction_artifact": "severity.current.standard_nadir_predictions",
        "horizon": 0,
    },
    {
        "application": "Future severity",
        "model_label": "Selected multiangular",
        "prediction_artifact": "severity.future.selected_multiangular_predictions",
        "horizon": None,
    },
    {
        "application": "Future severity",
        "model_label": "Standard nadir",
        "prediction_artifact": "severity.future.standard_nadir_predictions",
        "horizon": None,
    },
]


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = regression_metrics(y_true, y_pred)
    return {key: metrics[key] for key in ("rmse", "mae", "r2", "bias")}


def load_cultivar_map() -> pl.DataFrame:
    if not CULTIVAR_SOURCE.exists():
        raise FileNotFoundError(CULTIVAR_SOURCE)
    return (
        pl.scan_csv(CULTIVAR_SOURCE)
        .select(["plot_id", "cult"])
        .unique()
        .collect()
    )


def load_panel_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    started = time.perf_counter()
    cultivar_map = load_cultivar_map()
    frames = []
    metrics = []
    for spec in PANEL_SPECS:
        path = artifact_path(spec["prediction_artifact"])
        if not path.exists():
            raise FileNotFoundError(path)
        frame = (
            pl.read_csv(path)
            .with_columns((pl.col("target_week") - pl.col("predictor_week")).alias("horizon"))
            .join(cultivar_map, on="plot_id", how="left")
            .with_columns(
            pl.lit(spec["application"]).alias("application"),
            pl.lit(spec["model_label"]).alias("model_label"),
            pl.lit(str(path.relative_to(ROOT))).alias("prediction_file"),
            )
        )
        if spec["horizon"] is not None:
            frame = frame.filter(pl.col("horizon") == spec["horizon"])
        if frame.height == 0:
            raise ValueError(f"No rows left after horizon={spec['horizon']} filter for {path}")
        if frame.get_column("cult").null_count():
            missing = frame.filter(pl.col("cult").is_null()).select("plot_id").unique()
            raise ValueError(f"Missing cultivar metadata for {missing}")
        y = frame.get_column("y_true").to_numpy().astype(float)
        pred = frame.get_column("y_pred").to_numpy().astype(float)
        row = {
            "application": spec["application"],
            "model_label": spec["model_label"],
            "horizon_filter": "all" if spec["horizon"] is None else str(spec["horizon"]),
            "horizons_present": ",".join(
                str(value)
                for value in frame.select("horizon").unique().sort("horizon").get_column("horizon")
            ),
            "n": frame.height,
            "n_plots": frame.select(pl.col("plot_id").n_unique()).item(),
            **score(y, pred),
            "prediction_file": str(path.relative_to(ROOT)),
        }
        frames.append(frame)
        metrics.append(row)
    predictions = pl.concat(frames, how="vertical")
    metric_frame = pl.DataFrame(metrics)
    log_phase("load prediction files", started)
    return predictions, metric_frame


def jitter(values: np.ndarray, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return values + rng.normal(0.0, scale, size=values.shape)


def scatter_by_cultivar(ax: plt.Axes, subset: pl.DataFrame, axis_limit: float, seed: int) -> None:
    for offset, cultivar in enumerate(["aluco", "capone"]):
        cult_data = subset.filter(pl.col("cult") == cultivar)
        if cult_data.is_empty():
            continue
        y_true = cult_data.get_column("y_true").to_numpy().astype(float)
        y_pred = cult_data.get_column("y_pred").to_numpy().astype(float)
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


def draw_panel(
    ax: plt.Axes,
    predictions: pl.DataFrame,
    metrics: pl.DataFrame,
    application: str,
    model_label: str,
    *,
    axis_limit: float,
    seed: int,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    subset = predictions.filter(
        (pl.col("application") == application) & (pl.col("model_label") == model_label)
    )
    metric = metrics.filter(
        (pl.col("application") == application) & (pl.col("model_label") == model_label)
    ).row(0, named=True)

    scatter_by_cultivar(ax, subset, axis_limit, seed)
    ax.plot([0, axis_limit], [0, axis_limit], color=PALETTE["grey"], linewidth=1.1, linestyle=(0, (4, 4)))
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
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PALETTE["light_grey"], "alpha": 0.92},
    )
    if show_xlabel:
        ax.set_xlabel("Observed severity", fontsize=11, color=PALETTE["navy"])
    if show_ylabel:
        ax.set_ylabel("Predicted severity", fontsize=11, color=PALETTE["navy"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["navy"])
    ax.spines["bottom"].set_color(PALETTE["navy"])


def axis_limit_for(predictions: pl.DataFrame, application: str) -> float:
    app_data = predictions.filter(pl.col("application") == application)
    max_axis = float(max(app_data.get_column("y_true").max(), app_data.get_column("y_pred").max()))
    return max(10.0, math.ceil(max_axis / 5.0) * 5.0)


def save_figure(fig: plt.Figure, output_prefix: Path) -> list[Path]:
    paths = persist_figure(fig, output_prefix)
    for path in paths:
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    return paths


def write_combined_plot(predictions: pl.DataFrame, metrics: pl.DataFrame) -> list[Path]:
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

    fig.suptitle("Observed vs Predicted Severity", fontsize=22, fontweight="bold", color=PALETTE["navy"], y=0.985)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.02,
        "Points are real 2025 held-out plot-week predictions, colored by cultivar; small deterministic jitter only separates overlapping points.",
        ha="center",
        fontsize=9.5,
        color=PALETTE["grey"],
    )
    fig.tight_layout(rect=[0.03, 0.045, 1, 0.91])
    paths = save_figure(fig, FIGURES_DIR / "actual_vs_predicted_best_vs_nadir")
    log_phase("write observed-predicted plot", started)
    return paths


def write_application_plot(predictions: pl.DataFrame, metrics: pl.DataFrame, application: str, stem: str) -> list[Path]:
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
    fig.suptitle(f"{application}: Observed vs Predicted Severity", fontsize=19, fontweight="bold", color=PALETTE["navy"], y=0.99)
    fig.text(
        0.5,
        0.02,
        "Held-out 2025 plot-week predictions, colored by cultivar.",
        ha="center",
        fontsize=9.5,
        color=PALETTE["grey"],
    )
    fig.tight_layout(rect=[0.03, 0.06, 1, 0.84])
    paths = save_figure(fig, FIGURES_DIR / stem)
    log_phase(f"write {stem}", started)
    return paths


def markdown_table(metrics: pl.DataFrame) -> str:
    display = metrics.select(
        [
            "application",
            "model_label",
            pl.col("rmse").round(3),
            pl.col("mae").round(3),
            pl.col("r2").round(3),
            pl.col("bias").round(3),
            "horizon_filter",
            "horizons_present",
            "n",
            "n_plots",
            "prediction_file",
        ]
    )
    return render_markdown_table(display, float_digits=3)


def write_outputs(predictions: pl.DataFrame, metrics: pl.DataFrame, figure_paths: list[Path], log_path: Path) -> Path:
    started = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = RESULTS_DIR / "actual_vs_predicted_best_vs_nadir_predictions.csv"
    metrics_path = RESULTS_DIR / "actual_vs_predicted_best_vs_nadir_metrics.csv"
    predictions.write_csv(prediction_path)
    metrics.write_csv(metrics_path)
    report = f"""## Results: Observed vs Predicted Severity

{markdown_table(metrics)}

**Interpretation**: These figures compare the selected multiangular model against the standard nadir baseline for current-week and future-severity prediction. No synthetic observations are added; jitter is visual only. Cultivar colors are joined from the 2025 feature metadata (`aluco`, `capone`).

**Outputs**:
- `{prediction_path.relative_to(ROOT)}`
- `{metrics_path.relative_to(ROOT)}`
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in figure_paths)}

**Reproducibility**:
- Train year: `2024`
- Test year: `2025`
- Unit shown: held-out plot-week prediction rows
- Current-week filter: `target_week - predictor_week = 0`
- Future-severity filter: no horizon filter; uses the saved experiment rows as-is.
- Subplot data were not used in the main figure because the saved 20-subplot experiment assigns the same plot-level target to all subplots and is pseudo-replication.
- Cultivar metadata: `{CULTIVAR_SOURCE.relative_to(ROOT)}`
- Log: `{log_path.relative_to(ROOT)}`
"""
    report_path = REPORTS_DIR / "actual_vs_predicted_best_vs_nadir_summary.md"
    persist_report(report_path, report)
    log_phase("write outputs", started)
    return report_path


def main() -> None:
    total_started = time.perf_counter()
    log_path = configure_logging(LOGS_DIR, "plot_actual_vs_predicted_best_vs_nadir")
    predictions, metrics = load_panel_data()
    figure_paths = [
        *write_combined_plot(predictions, metrics),
        *write_application_plot(
            predictions,
            metrics,
            "Current-week severity",
            "actual_vs_predicted_current_week_best_vs_nadir_by_cultivar",
        ),
        *write_application_plot(
            predictions,
            metrics,
            "Future severity",
            "actual_vs_predicted_future_best_vs_nadir_by_cultivar",
        ),
    ]
    write_outputs(predictions, metrics, figure_paths, log_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
