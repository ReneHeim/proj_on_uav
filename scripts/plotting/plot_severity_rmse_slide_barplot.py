"""Create a slide-ready RMSE bar plot for the compact severity models."""

from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MPLCONFIG_DIR = ROOT / "outputs/.matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from scipy.stats import spearmanr

OUTPUT_ROOT = ROOT / "outputs/multiangular_distribution_feature_family/model_bottleneck_debug"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/logs"

SELECTED_RESULT = RESULTS_DIR / "selected_42_feature_severity_result.csv"
CANDIDATE_COMPARISON = RESULTS_DIR / "candidate_model_comparison.csv"
FORCED_TOPK = RESULTS_DIR / "exploratory_extra_compact_forced_topk_results.csv"
VZA_RAA_BLEND_SUMMARY = (
    ROOT
    / "outputs/future_severity_vza_raa_feature_selection_improvement/results/vza_raa_oof_selected_blend_summary.csv"
)
VZA_RAA_BLEND_PREDICTIONS = (
    ROOT
    / "outputs/future_severity_vza_raa_feature_selection_improvement/results/vza_raa_oof_selected_blend_predictions_2025.csv"
)


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"plot_severity_rmse_slide_barplot_{timestamp}.log"
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


def score_prediction_frame(predictions: pl.DataFrame, pred_col: str) -> dict[str, float]:
    y = predictions.get_column("y_true").to_numpy().astype(float)
    pred = predictions.get_column(pred_col).to_numpy().astype(float)
    rmse = math.sqrt(float(np.mean((y - pred) ** 2)))
    mae = float(np.mean(np.abs(y - pred)))
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
    spearman = float(spearmanr(y, pred, nan_policy="omit").correlation)
    return {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman}


def load_plot_data() -> pl.DataFrame:
    started = time.perf_counter()
    selected = pl.read_csv(SELECTED_RESULT).row(0, named=True)
    candidates = pl.read_csv(CANDIDATE_COMPARISON)
    forced = pl.read_csv(FORCED_TOPK)
    blend_summary = pl.read_csv(VZA_RAA_BLEND_SUMMARY).row(0, named=True)
    blend_predictions = pl.read_csv(VZA_RAA_BLEND_PREDICTIONS)
    blend_scores = score_prediction_frame(blend_predictions, "y_pred")

    nadir = candidates.filter(
        (pl.col("model") == "residual_reliability_filtered_xgboost")
        & (pl.col("feature_set") == "compact_anomaly_nadir")
        & (pl.col("covariates") == "spectral_plus_week_horizon")
    ).row(0, named=True)

    all_features = (
        forced.sort("k_reflectance_features_forced", descending=True).head(1).row(0, named=True)
    )

    rows = [
        {
            "label": "Nadir only",
            "model_detail": "Residual pipeline, compact nadir",
            "rmse": float(nadir["rmse"]),
            "mae": float(nadir["mae"]),
            "r2": float(nadir["r2"]),
            "spearman": float(nadir["spearman"]),
            "reflectance_features": int(nadir["n_features"]) - 2,
            "total_features": int(nadir["n_features"]),
            "bar_note": f"{int(nadir['n_features']) - 2} features",
            "status": "reference",
        },
        {
            "label": "All VZA features",
            "model_detail": "Residual pipeline, all compact VZA-only candidates",
            "rmse": float(all_features["rmse"]),
            "mae": float(all_features["mae"]),
            "r2": float(all_features["r2"]),
            "spearman": float(all_features["spearman"]),
            "reflectance_features": int(all_features["k_reflectance_features_forced"]),
            "total_features": int(all_features["n_total_features"]),
            "bar_note": f"{int(all_features['k_reflectance_features_forced'])} features",
            "status": "exploratory",
        },
        {
            "label": "VZA only",
            "model_detail": "Residual pipeline, selected compact VZA-only subset",
            "rmse": float(selected["rmse"]),
            "mae": float(selected["mae"]),
            "r2": float(selected["r2"]),
            "spearman": float(selected["spearman"]),
            "reflectance_features": int(selected["selected_reflectance_features"]),
            "total_features": int(selected["n_total_features"]),
            "bar_note": f"{int(selected['selected_reflectance_features'])} features",
            "status": "selected",
        },
        {
            "label": "VZA + RAA correction",
            "model_detail": "OOF-selected blend: VZA-only plus shrinked all-VZA+RAA correction",
            "rmse": blend_scores["rmse"],
            "mae": blend_scores["mae"],
            "r2": blend_scores["r2"],
            "spearman": blend_scores["spearman"],
            "reflectance_features": int(selected["selected_reflectance_features"]),
            "total_features": int(selected["n_total_features"]),
            "bar_note": f"{float(blend_summary['alpha_on_raa']):.0%} RAA",
            "status": f"oof_blend_alpha_raa_{float(blend_summary['alpha_on_raa']):.2f}",
        },
    ]
    out = pl.DataFrame(rows)
    log_phase("load RMSE source tables", started)
    return out


def write_plot(data: pl.DataFrame) -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = data.sort("rmse")
    labels = data.get_column("label").to_list()
    rmse = data.get_column("rmse").to_list()
    bar_notes = data.get_column("bar_note").to_list()
    palette = {
        "All VZA features": "#FF6B6B",
        "Nadir only": "#F6C85F",
        "VZA only": "#0B132B",
        "VZA + RAA correction": "#00A6A6",
    }
    colors = [palette[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10.0, 6.25))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(labels, rmse, color=colors, width=0.62, edgecolor="#202020", linewidth=0.8)
    nadir_rmse = float(data.filter(pl.col("label") == "Nadir only")["rmse"][0])
    ax.axhline(nadir_rmse, color="#0B132B", linewidth=1.1, linestyle=(0, (4, 4)), alpha=0.55)

    for idx, (bar, value, note) in enumerate(zip(bars, rmse, bar_notes, strict=True)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.12,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color="#111827",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(value - 0.72, 0.35),
            note,
            ha="center",
            va="center",
            fontsize=10.5,
            color=(
                "#0B132B"
                if labels[idx] not in {"VZA only", "VZA + RAA correction", "All VZA features"}
                else "white"
            ),
            fontweight="bold",
        )

    fig.suptitle(
        "Severity Prediction RMSE",
        fontsize=24,
        fontweight="bold",
        color="#0B132B",
        y=0.965,
    )
    ax.text(
        0.0,
        0.985,
        "Lower RMSE is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color="#6b7280",
    )
    ax.set_ylabel("RMSE severity units", fontsize=12, color="#0B132B")
    ax.set_ylim(0, max(rmse) + 1.6)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#0B132B")
    ax.spines["bottom"].set_color("#0B132B")

    fig.tight_layout(rect=[0, 0, 1, 0.945])
    paths = [
        FIGURES_DIR / "severity_rmse_nadir_all_selected_barplot.png",
        FIGURES_DIR / "severity_rmse_nadir_all_selected_barplot.pdf",
        FIGURES_DIR / "severity_rmse_nadir_all_selected_barplot.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=300)
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    log_phase("write RMSE bar plot", started)
    return paths


def markdown_table(data: pl.DataFrame) -> str:
    display = data.select(
        [
            "label",
            pl.col("rmse").round(3),
            pl.col("mae").round(3),
            pl.col("r2").round(3),
            pl.col("spearman").round(3),
            "reflectance_features",
            "total_features",
            "bar_note",
            "status",
        ]
    )
    headers = display.columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.iter_rows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(data: pl.DataFrame, figure_paths: list[Path], log_path: Path) -> Path:
    started = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    nadir_rmse = float(data.filter(pl.col("label") == "Nadir only")["rmse"][0])
    selected_rmse = float(data.filter(pl.col("label") == "VZA only")["rmse"][0])
    vza_raa_rmse = float(data.filter(pl.col("label") == "VZA + RAA correction")["rmse"][0])
    report = f"""## Results: Severity RMSE Slide Bar Plot

{markdown_table(data)}

**Interpretation**: The OOF-selected VZA+RAA correction reduced external-year RMSE by {selected_rmse - vza_raa_rmse:.3f} severity units relative to the selected VZA-only residual model, and by {nadir_rmse - vza_raa_rmse:.3f} severity units relative to nadir-only.

**Outputs**:
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in figure_paths)}

**Reproducibility**:

- Source selected result: `{SELECTED_RESULT.relative_to(ROOT)}`
- Source candidate comparison: `{CANDIDATE_COMPARISON.relative_to(ROOT)}`
- Source all-feature top-k run: `{FORCED_TOPK.relative_to(ROOT)}`
- Source VZA+RAA blend summary: `{VZA_RAA_BLEND_SUMMARY.relative_to(ROOT)}`
- Source VZA+RAA blend predictions: `{VZA_RAA_BLEND_PREDICTIONS.relative_to(ROOT)}`
- Log: `{log_path.relative_to(ROOT)}`
"""
    path = REPORTS_DIR / "severity_rmse_slide_barplot_summary.md"
    path.write_text(report, encoding="utf-8")
    logging.info("Wrote report: %s", path)
    log_phase("write RMSE bar plot report", started)
    return path


def main() -> None:
    log_path = setup_logging()
    data = load_plot_data()
    paths = write_plot(data)
    write_report(data, paths, log_path)


if __name__ == "__main__":
    main()
