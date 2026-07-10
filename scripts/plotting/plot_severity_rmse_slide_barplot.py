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

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/model_bottleneck_debug"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

SELECTED_RESULT = RESULTS_DIR / "selected_42_feature_severity_result.csv"
CANDIDATE_COMPARISON = RESULTS_DIR / "candidate_model_comparison.csv"
FORCED_TOPK = RESULTS_DIR / "exploratory_extra_compact_forced_topk_results.csv"
DIRECT_VZA_RAA_HURDLE_RESULTS = (
    ROOT
    / "outputs/runs/analysis/severity/future/current_hurdle_vza_raa_2024_to_2025/results/future_severity_current_hurdle_vza_raa_model_comparison.csv"
)
DIRECT_VZA_RAA_HURDLE_PREDICTIONS = (
    ROOT
    / "outputs/runs/analysis/severity/future/current_hurdle_vza_raa_2024_to_2025/results/predictions/severity_predictions_current_hurdle_stability_top50_raw_positive_multiangular_vza_raa_spectral_plus_week_horizon.csv"
)
PREDICTION_FILES = {
    "Nadir only": ROOT
    / "outputs/runs/analysis/severity/future/current_hurdle_vza_raa_2024_to_2025/results/predictions/severity_predictions_current_hurdle_stability_top50_raw_positive_nadir_from_vza_spectral_plus_week_horizon.csv",
    "All VZA features": RESULTS_DIR
    / "predictions/severity_predictions_exploratory_residual_xgboost_forced_top_144_compact_features_compact_anomaly_multiangular_spectral_plus_week_horizon.csv",
    "VZA only": RESULTS_DIR
    / "predictions/severity_predictions_exploratory_residual_xgboost_forced_top_042_compact_features_compact_anomaly_multiangular_spectral_plus_week_horizon.csv",
    "VZA+RAA": DIRECT_VZA_RAA_HURDLE_PREDICTIONS,
}


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


def paired_rmse_reduction_ci(
    model_prediction_path: Path,
    baseline_prediction_path: Path,
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    keys = ["plot_id", "predictor_week", "target_week"]
    model = pl.read_csv(model_prediction_path).select(keys + ["y_true", "y_pred"])
    baseline = pl.read_csv(baseline_prediction_path).select(keys + ["y_pred"])
    joined = model.join(baseline.rename({"y_pred": "y_pred_baseline"}), on=keys, how="inner")
    if joined.height != model.height:
        raise ValueError(
            f"Could not pair all rows for {model_prediction_path}; "
            f"paired {joined.height} of {model.height}"
        )

    y = joined.get_column("y_true").to_numpy().astype(float)
    pred = joined.get_column("y_pred").to_numpy().astype(float)
    baseline_pred = joined.get_column("y_pred_baseline").to_numpy().astype(float)
    observed = float(
        np.sqrt(np.mean((y - baseline_pred) ** 2)) - np.sqrt(np.mean((y - pred) ** 2))
    )

    groups = [
        group
        for _, group in joined.group_by("plot_id", maintain_order=True)
    ]
    rng = np.random.default_rng(seed)
    boot_reductions = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        y_sample = np.concatenate(
            [groups[group_idx].get_column("y_true").to_numpy().astype(float) for group_idx in sampled]
        )
        model_sample = np.concatenate(
            [groups[group_idx].get_column("y_pred").to_numpy().astype(float) for group_idx in sampled]
        )
        baseline_sample = np.concatenate(
            [
                groups[group_idx].get_column("y_pred_baseline").to_numpy().astype(float)
                for group_idx in sampled
            ]
        )
        boot_reductions[idx] = np.sqrt(np.mean((y_sample - baseline_sample) ** 2)) - np.sqrt(
            np.mean((y_sample - model_sample) ** 2)
        )
    lower, upper = np.quantile(boot_reductions, [0.025, 0.975])
    prob_gt_zero = float(np.mean(boot_reductions > 0.0))
    return observed, float(lower), float(upper), prob_gt_zero


def bootstrap_rmse_ci(
    prediction_path: Path,
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float]:
    predictions = pl.read_csv(prediction_path).select(["plot_id", "y_true", "y_pred"])
    y = predictions.get_column("y_true").to_numpy().astype(float)
    pred = predictions.get_column("y_pred").to_numpy().astype(float)
    observed = math.sqrt(float(np.mean((y - pred) ** 2)))

    groups = [group for _, group in predictions.group_by("plot_id", maintain_order=True)]
    rng = np.random.default_rng(seed)
    boot_rmse = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        y_sample = np.concatenate(
            [groups[group_idx].get_column("y_true").to_numpy().astype(float) for group_idx in sampled]
        )
        pred_sample = np.concatenate(
            [groups[group_idx].get_column("y_pred").to_numpy().astype(float) for group_idx in sampled]
        )
        boot_rmse[idx] = math.sqrt(float(np.mean((y_sample - pred_sample) ** 2)))
    lower, upper = np.quantile(boot_rmse, [0.025, 0.975])
    return observed, float(lower), float(upper)


def add_rmse_uncertainty(data: pl.DataFrame) -> pl.DataFrame:
    baseline_prediction_file = PREDICTION_FILES["Nadir only"]
    rows = []
    for row in data.iter_rows(named=True):
        prediction_file = PREDICTION_FILES[row["label"]]
        rmse, rmse_ci_low, rmse_ci_high = bootstrap_rmse_ci(prediction_file)
        row["rmse"] = rmse
        row["rmse_ci_low"] = rmse_ci_low
        row["rmse_ci_high"] = rmse_ci_high
        row["rmse_err_low"] = max(rmse - rmse_ci_low, 0.0)
        row["rmse_err_high"] = max(rmse_ci_high - rmse, 0.0)
        if row["label"] == "Nadir only":
            reduction = 0.0
            ci_low = 0.0
            ci_high = 0.0
            prob_gt_zero = 0.0
        else:
            reduction, ci_low, ci_high, prob_gt_zero = paired_rmse_reduction_ci(
                prediction_file,
                baseline_prediction_file,
            )
        row["rmse_reduction_vs_nadir"] = reduction
        row["rmse_reduction_ci_low"] = ci_low
        row["rmse_reduction_ci_high"] = ci_high
        row["rmse_reduction_prob_gt_zero"] = prob_gt_zero
        row["prediction_file"] = str(prediction_file.relative_to(ROOT))
        rows.append(row)
        logging.info(
            "Paired RMSE reduction CI vs nadir for %s from %s: %.3f [%.3f, %.3f]",
            row["label"],
            prediction_file,
            reduction,
            ci_low,
            ci_high,
        )
    return pl.DataFrame(rows)


def load_plot_data() -> pl.DataFrame:
    started = time.perf_counter()
    selected = pl.read_csv(SELECTED_RESULT).row(0, named=True)
    candidates = pl.read_csv(CANDIDATE_COMPARISON)
    forced = pl.read_csv(FORCED_TOPK)
    direct_vza_raa = (
        pl.read_csv(DIRECT_VZA_RAA_HURDLE_RESULTS)
        .filter(
            (pl.col("model") == "current_hurdle_stability_top50_raw_positive")
            & (pl.col("feature_set") == "multiangular_vza_raa")
        )
        .sort("rmse")
    )
    direct_predictions = pl.read_csv(DIRECT_VZA_RAA_HURDLE_PREDICTIONS)
    direct_scores = score_prediction_frame(direct_predictions, "y_pred")
    direct_n_features = 84
    if direct_vza_raa.height:
        direct_row = direct_vza_raa.row(0, named=True)
        direct_n_features = int(direct_row["n_features"])

    nadir_predictions = pl.read_csv(PREDICTION_FILES["Nadir only"])
    nadir_scores = score_prediction_frame(nadir_predictions, "y_pred")

    all_features = (
        forced.sort("k_reflectance_features_forced", descending=True).head(1).row(0, named=True)
    )

    rows = [
        {
            "label": "Nadir only",
            "model_detail": "Matched direct hurdle model, nadir-only from VZA table",
            "rmse": nadir_scores["rmse"],
            "mae": nadir_scores["mae"],
            "r2": nadir_scores["r2"],
            "spearman": nadir_scores["spearman"],
            "reflectance_features": 5,
            "total_features": 7,
            "bar_note": "5 features",
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
            "label": "VZA+RAA",
            "model_detail": "Direct transferred hurdle model with VZA plus relative-azimuth features",
            "rmse": direct_scores["rmse"],
            "mae": direct_scores["mae"],
            "r2": direct_scores["r2"],
            "spearman": direct_scores["spearman"],
            "reflectance_features": direct_n_features,
            "total_features": direct_n_features,
            "bar_note": f"{direct_n_features} features",
            "status": "current_hurdle_top50_transferred",
        },
    ]
    out = add_rmse_uncertainty(pl.DataFrame(rows))
    log_phase("load RMSE source tables", started)
    return out


def write_plot(data: pl.DataFrame) -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = data.sort("rmse")
    labels = data.get_column("label").to_list()
    rmse = data.get_column("rmse").to_list()
    bar_notes = data.get_column("bar_note").to_list()
    yerr = np.vstack(
        [
            data.get_column("rmse_err_low").to_numpy(),
            data.get_column("rmse_err_high").to_numpy(),
        ]
    )
    palette = {
        "All VZA features": "#FF6B6B",
        "Nadir only": "#F6C85F",
        "VZA only": "#0B132B",
        "VZA+RAA": "#00A6A6",
    }
    colors = [palette[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10.0, 6.25))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(labels, rmse, color=colors, width=0.62, edgecolor="#202020", linewidth=0.8)
    ax.errorbar(
        labels,
        rmse,
        yerr=yerr,
        fmt="none",
        ecolor="#0B132B",
        elinewidth=1.1,
        capsize=4,
        capthick=1.1,
        zorder=5,
    )
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
                if labels[idx] not in {"VZA only", "VZA+RAA", "All VZA features"}
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
    y_top = float(np.max(np.asarray(rmse, dtype=float) + yerr[1]))
    ax.set_ylim(0, y_top + 1.1)
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
            pl.col("rmse_ci_low").round(3),
            pl.col("rmse_ci_high").round(3),
            pl.col("rmse_reduction_vs_nadir").round(3),
            pl.col("rmse_reduction_ci_low").round(3),
            pl.col("rmse_reduction_ci_high").round(3),
            pl.col("rmse_reduction_prob_gt_zero").round(3),
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
    vza_raa_rmse = float(data.filter(pl.col("label") == "VZA+RAA")["rmse"][0])
    report = f"""## Results: Severity RMSE Slide Bar Plot

{markdown_table(data)}

**Interpretation**: The direct transferred VZA+RAA hurdle model reduced external-year RMSE by {selected_rmse - vza_raa_rmse:.3f} severity units relative to the selected VZA-only residual model, and by {nadir_rmse - vza_raa_rmse:.3f} severity units relative to nadir-only.

**Outputs**:
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in figure_paths)}

**Reproducibility**:

- Source selected result: `{SELECTED_RESULT.relative_to(ROOT)}`
- Source candidate comparison: `{CANDIDATE_COMPARISON.relative_to(ROOT)}`
- Source all-feature top-k run: `{FORCED_TOPK.relative_to(ROOT)}`
- Source matched direct nadir predictions: `{PREDICTION_FILES["Nadir only"].relative_to(ROOT)}`
- Source direct VZA+RAA hurdle result: `{DIRECT_VZA_RAA_HURDLE_RESULTS.relative_to(ROOT)}`
- Source direct VZA+RAA hurdle predictions: `{DIRECT_VZA_RAA_HURDLE_PREDICTIONS.relative_to(ROOT)}`
- Error bars: absolute plot-level bootstrap 95% CI for each model RMSE; seed=42, n_bootstrap=5000. Paired improvement columns are still computed versus nadir.
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
