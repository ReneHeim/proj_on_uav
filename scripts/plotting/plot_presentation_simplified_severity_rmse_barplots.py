#!/usr/bin/env python3
"""Create simplified presentation RMSE bar plots for current and future severity."""

from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime
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

LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"
REPORTS_DIR = ROOT / "outputs/archive/legacy_unscoped/reports"

CURRENT_OUTPUT = ROOT / "outputs/runs/analysis/severity/current/2024_to_2025"
FUTURE_OUTPUT = ROOT / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/model_bottleneck_debug"

PALETTE = {
    "navy": "#0B132B",
    "coral": "#FF6B6B",
    "gold": "#F6C85F",
    "grey": "#6b7280",
    "light_grey": "#d1d5db",
}

CURRENT_ROWS = [
    {
        "label": "Multiangular selected",
        "rmse": 3.2362847483912085,
        "mae": 1.766263,
        "r2": 0.871226,
        "spearman": 0.695584,
        "features": 52,
        "prediction_file": ROOT
        / "outputs/runs/analysis/severity/current/raa_geometry_fusion_2024_to_2025/results/predictions/severity_predictions_current_hurdle_stability_top30_raw_positive_multiangular_vza_raa_spectral_plus_week.csv",
        "source_model": "current_hurdle_stability_top30_raw_positive / multiangular_vza_raa",
    },
    {
        "label": "All multiangular features",
        "rmse": 4.018790260539746,
        "mae": 2.133,
        "r2": 0.801,
        "spearman": 0.723,
        "features": 206,
        "prediction_file": ROOT
        / "outputs/runs/analysis/severity/current/raa_geometry_fusion_2024_to_2025/results/predictions/severity_predictions_hurdle_probability_times_severity_multiangular_vza_raa_spectral_plus_week.csv",
        "source_model": "hurdle_probability_times_severity / multiangular_vza_raa",
    },
    {
        "label": "Standard nadir",
        "rmse": 4.6233779977445,
        "mae": 2.725,
        "r2": 0.737,
        "spearman": 0.734,
        "features": 17,
        "prediction_file": ROOT
        / "outputs/runs/analysis/severity/current/2024_to_2025/results/predictions/severity_predictions_current_hurdle_top20_raw_positive_compact_anomaly_nadir_spectral_plus_week.csv",
        "source_model": "current_hurdle_top20_raw_positive / compact_anomaly_nadir",
    },
]

FUTURE_ROWS = [
    {
        "label": "Multiangular selected",
        "rmse": 7.611959181890869,
        "mae": 5.121267,
        "r2": 0.568995,
        "spearman": 0.785311,
        "features": 84,
        "prediction_file": ROOT
        / "outputs/runs/analysis/severity/future/current_hurdle_vza_raa_2024_to_2025/results/predictions/severity_predictions_current_hurdle_stability_top50_raw_positive_multiangular_vza_raa_spectral_plus_week_horizon.csv",
        "source_model": "current_hurdle_stability_top50_raw_positive / multiangular_vza_raa",
    },
    {
        "label": "All multiangular features",
        "rmse": 10.496457,
        "mae": 7.563041,
        "r2": 0.180451,
        "spearman": 0.750562,
        "features": 207,
        "prediction_file": ROOT
        / "outputs/runs/analysis/severity/future/raa_geometry_residual_2024_to_2025/results/predictions/severity_predictions_residual_all_features_xgboost_multiangular_vza_raa_spectral_plus_week_horizon.csv",
        "source_model": "residual_all_features_xgboost / multiangular_vza_raa",
    },
    {
        "label": "Standard nadir",
        "rmse": 9.487493197742946,
        "mae": 5.519308,
        "r2": 0.330435,
        "spearman": 0.759988,
        "features": 5,
        "prediction_file": ROOT
        / "outputs/runs/analysis/severity/future/current_hurdle_vza_raa_2024_to_2025/results/predictions/severity_predictions_current_hurdle_stability_top50_raw_positive_nadir_from_vza_spectral_plus_week_horizon.csv",
        "source_model": "current_hurdle_stability_top50_raw_positive / nadir_from_vza",
    },
]


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"plot_presentation_simplified_severity_rmse_barplots_{timestamp}.log"
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


def score_prediction_file(path: Path) -> dict[str, float]:
    frame = pl.read_csv(path)
    y = frame.get_column("y_true").to_numpy().astype(float)
    pred = frame.get_column("y_pred").to_numpy().astype(float)
    rmse = math.sqrt(float(np.mean((y - pred) ** 2)))
    mae = float(np.mean(np.abs(y - pred)))
    r2 = 1.0 - float(np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
    return {"rmse": rmse, "mae": mae, "r2": r2}


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
            f"Could not pair all rows for {model_prediction_path}; paired {joined.height}"
        )
    y = joined.get_column("y_true").to_numpy().astype(float)
    pred = joined.get_column("y_pred").to_numpy().astype(float)
    baseline_pred = joined.get_column("y_pred_baseline").to_numpy().astype(float)
    observed = float(
        np.sqrt(np.mean((y - baseline_pred) ** 2)) - np.sqrt(np.mean((y - pred) ** 2))
    )
    groups = [group for _, group in joined.group_by("plot_id", maintain_order=True)]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        y_sample = np.concatenate(
            [groups[group_idx].get_column("y_true").to_numpy().astype(float) for group_idx in sampled]
        )
        pred_sample = np.concatenate(
            [groups[group_idx].get_column("y_pred").to_numpy().astype(float) for group_idx in sampled]
        )
        baseline_sample = np.concatenate(
            [
                groups[group_idx].get_column("y_pred_baseline").to_numpy().astype(float)
                for group_idx in sampled
            ]
        )
        boot[idx] = np.sqrt(np.mean((y_sample - baseline_sample) ** 2)) - np.sqrt(
            np.mean((y_sample - pred_sample) ** 2)
        )
    low, high = np.quantile(boot, [0.025, 0.975])
    return observed, float(low), float(high), float(np.mean(boot > 0.0))


def bootstrap_rmse_ci(
    prediction_path: Path,
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> tuple[float, float, float]:
    frame = pl.read_csv(prediction_path).select(["plot_id", "y_true", "y_pred"])
    y = frame.get_column("y_true").to_numpy().astype(float)
    pred = frame.get_column("y_pred").to_numpy().astype(float)
    observed = math.sqrt(float(np.mean((y - pred) ** 2)))

    groups = [group for _, group in frame.group_by("plot_id", maintain_order=True)]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sampled = rng.integers(0, len(groups), size=len(groups))
        y_sample = np.concatenate(
            [groups[group_idx].get_column("y_true").to_numpy().astype(float) for group_idx in sampled]
        )
        pred_sample = np.concatenate(
            [groups[group_idx].get_column("y_pred").to_numpy().astype(float) for group_idx in sampled]
        )
        boot[idx] = math.sqrt(float(np.mean((y_sample - pred_sample) ** 2)))
    low, high = np.quantile(boot, [0.025, 0.975])
    return observed, float(low), float(high)


def build_rows(rows: list[dict[str, object]]) -> pl.DataFrame:
    baseline = next(row for row in rows if row["label"] == "Standard nadir")
    built = []
    for row in rows:
        scores = score_prediction_file(row["prediction_file"])
        row = row.copy()
        row["rmse"] = scores["rmse"]
        row["mae"] = scores["mae"]
        row["r2"] = scores["r2"]
        observed_rmse, rmse_ci_low, rmse_ci_high = bootstrap_rmse_ci(row["prediction_file"])
        row["rmse"] = observed_rmse
        row["rmse_ci_low"] = rmse_ci_low
        row["rmse_ci_high"] = rmse_ci_high
        row["rmse_err_low"] = max(observed_rmse - rmse_ci_low, 0.0)
        row["rmse_err_high"] = max(rmse_ci_high - observed_rmse, 0.0)
        if row["label"] == "Standard nadir":
            reduction, ci_low, ci_high, prob = 0.0, 0.0, 0.0, 0.0
        else:
            reduction, ci_low, ci_high, prob = paired_rmse_reduction_ci(
                row["prediction_file"],
                baseline["prediction_file"],
            )
        row["rmse_reduction_vs_nadir"] = reduction
        row["rmse_reduction_ci_low"] = ci_low
        row["rmse_reduction_ci_high"] = ci_high
        row["rmse_reduction_prob_gt_zero"] = prob
        row["prediction_file"] = str(row["prediction_file"].relative_to(ROOT))
        built.append(row)
    return pl.DataFrame(built)


def write_plot(data: pl.DataFrame, title: str, output_prefix: Path, y_limit_extra: float) -> list[Path]:
    started = time.perf_counter()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    data = data.sort("rmse")
    labels = data.get_column("label").to_list()
    rmse = data.get_column("rmse").to_list()
    features = data.get_column("features").to_list()
    yerr = np.vstack(
        [data.get_column("rmse_err_low").to_numpy(), data.get_column("rmse_err_high").to_numpy()]
    )
    colors_by_label = {
        "Multiangular selected": PALETTE["navy"],
        "All multiangular features": PALETTE["coral"],
        "Standard nadir": PALETTE["gold"],
    }
    colors = [colors_by_label[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10.0, 6.25))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.bar(labels, rmse, color=colors, width=0.62, edgecolor=PALETTE["navy"], linewidth=0.8)
    ax.errorbar(
        labels,
        rmse,
        yerr=yerr,
        fmt="none",
        ecolor=PALETTE["navy"],
        elinewidth=1.1,
        capsize=4,
        capthick=1.1,
        zorder=5,
    )
    nadir_rmse = float(data.filter(pl.col("label") == "Standard nadir")["rmse"][0])
    ax.axhline(nadir_rmse, color=PALETTE["grey"], linewidth=1.1, linestyle=(0, (4, 4)), alpha=0.7)

    for idx, (bar, value, n_features) in enumerate(zip(bars, rmse, features, strict=True)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.12,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color=PALETTE["navy"],
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(value - 0.62, 0.35),
            f"{n_features} features",
            ha="center",
            va="center",
            fontsize=10.5,
            color="white" if labels[idx] != "Standard nadir" else PALETTE["navy"],
            fontweight="bold",
        )

    fig.suptitle(title, fontsize=24, fontweight="bold", color=PALETTE["navy"], y=0.965)
    ax.text(
        0.0,
        0.985,
        "Lower RMSE is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color=PALETTE["grey"],
    )
    ax.set_ylabel("RMSE severity units", fontsize=12, color=PALETTE["navy"])
    y_top = float(np.max(np.asarray(rmse, dtype=float) + yerr[1]))
    ax.set_ylim(0, y_top + y_limit_extra)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", color=PALETTE["light_grey"], linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["navy"])
    ax.spines["bottom"].set_color(PALETTE["navy"])
    fig.tight_layout(rect=[0, 0, 1, 0.945])

    paths = [output_prefix.with_suffix(ext) for ext in [".png", ".pdf", ".svg"]]
    for path in paths:
        fig.savefig(path, dpi=300)
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    log_phase(f"write {output_prefix.name}", started)
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
            "features",
            "source_model",
            "prediction_file",
        ]
    )
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for row in display.iter_rows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    current: pl.DataFrame,
    future: pl.DataFrame,
    current_paths: list[Path],
    future_paths: list[Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = f"""## Results: Simplified Presentation Severity RMSE Bar Plots

### Current severity

{markdown_table(current)}

### Future severity

{markdown_table(future)}

**Interpretation**: These are presentation-label versions of the severity RMSE plots. The selected multiangular bar uses the VZA+RAA model family, the all-feature bar uses all VZA+RAA/multiangular features where available, and the nadir bar is labelled as the standard nadir baseline.

**Outputs**:
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in current_paths + future_paths)}

**Reproducibility**:

- Train year: `2024`
- Test year: `2025`
- Error bars: absolute plot-level bootstrap 95% CI for each model RMSE; seed=42, n_bootstrap=5000. Paired improvement columns are still computed versus standard nadir.
- Log: `{log_path.relative_to(ROOT)}`
"""
    path = REPORTS_DIR / "presentation_simplified_severity_rmse_barplots_summary.md"
    path.write_text(report, encoding="utf-8")
    logging.info("Wrote report: %s", path)
    return path


def main() -> None:
    total_started = time.perf_counter()
    log_path = setup_logging()
    current = build_rows(CURRENT_ROWS)
    future = build_rows(FUTURE_ROWS)
    current_paths = write_plot(
        current,
        "Current Severity Prediction RMSE",
        CURRENT_OUTPUT / "figures/current_severity_rmse_presentation_labels",
        y_limit_extra=1.05,
    )
    future_paths = write_plot(
        future,
        "Severity Prediction RMSE",
        FUTURE_OUTPUT / "figures/severity_rmse_presentation_labels",
        y_limit_extra=1.6,
    )
    write_report(current, future, current_paths, future_paths, log_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
