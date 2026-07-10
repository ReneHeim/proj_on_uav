"""Create a slide-ready RMSE bar plot for current severity prediction."""

from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MPLCONFIG_DIR = ROOT / "outputs/.matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

MODEL_COMPARISON = RESULTS_DIR / "current_severity_model_comparison.csv"
RAA_MODEL_COMPARISON = (
    ROOT
    / "outputs/runs/analysis/severity/current/raa_geometry_fusion_2024_to_2025/results/raa_geometry_model_comparison.csv"
)
PREDICTION_FILES = {
    "VZA only": RESULTS_DIR
    / "predictions/severity_predictions_current_hurdle_stability_top50_raw_positive_compact_anomaly_multiangular_spectral_plus_week.csv",
    "VZA+RAA": ROOT
    / "outputs/runs/analysis/severity/current/raa_geometry_fusion_2024_to_2025/results/predictions/severity_predictions_current_hurdle_stability_top30_raw_positive_multiangular_vza_raa_spectral_plus_week.csv",
    "All VZA features": RESULTS_DIR
    / "predictions/severity_predictions_hurdle_probability_times_severity_compact_anomaly_multiangular_spectral_plus_week.csv",
    "All VZA+RAA": ROOT
    / "outputs/runs/analysis/severity/current/raa_geometry_fusion_2024_to_2025/results/predictions/severity_predictions_hurdle_probability_times_severity_multiangular_vza_raa_spectral_plus_week.csv",
    "Nadir only": RESULTS_DIR
    / "predictions/severity_predictions_current_hurdle_top20_raw_positive_compact_anomaly_nadir_spectral_plus_week.csv",
}

PALETTE = {
    "navy": "#0B132B",
    "teal": "#00A6A6",
    "coral": "#FF6B6B",
    "gold": "#F6C85F",
    "background": "#F7F9FC",
    "grey": "#5C677D",
    "light_grey": "#E7ECF3",
}


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"plot_current_severity_rmse_slide_barplot_{timestamp}.log"
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


def row_by_model(data: pl.DataFrame, model: str, feature_set: str) -> dict[str, object]:
    rows = data.filter((pl.col("model") == model) & (pl.col("feature_set") == feature_set))
    if rows.height != 1:
        raise ValueError(f"Expected one row for {model} / {feature_set}, found {rows.height}")
    return rows.row(0, named=True)


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
    current = pl.read_csv(MODEL_COMPARISON)
    raa_current = pl.read_csv(RAA_MODEL_COMPARISON)

    selected = row_by_model(
        current,
        "current_hurdle_stability_top50_raw_positive",
        "compact_anomaly_multiangular",
    )
    all_features = row_by_model(
        current,
        "hurdle_probability_times_severity",
        "compact_anomaly_multiangular",
    )
    nadir = row_by_model(current, "current_hurdle_top20_raw_positive", "compact_anomaly_nadir")
    vza_raa = row_by_model(
        raa_current,
        "current_hurdle_stability_top30_raw_positive",
        "multiangular_vza_raa",
    )
    all_vza_raa = row_by_model(
        raa_current,
        "hurdle_probability_times_severity",
        "multiangular_vza_raa",
    )

    rows = [
        {
            "label": "VZA only",
            "model_detail": "Selected compact VZA-only current-severity model",
            "feature_set": selected["feature_set"],
            "rmse": float(selected["rmse"]),
            "mae": float(selected["mae"]),
            "r2": float(selected["r2"]),
            "spearman": float(selected["spearman"]),
            "total_features": int(float(selected["n_features"])),
            "status": "selected",
        },
        {
            "label": "VZA+RAA",
            "model_detail": "Selected VZA plus relative-azimuth current-severity model",
            "feature_set": vza_raa["feature_set"],
            "rmse": float(vza_raa["rmse"]),
            "mae": float(vza_raa["mae"]),
            "r2": float(vza_raa["r2"]),
            "spearman": float(vza_raa["spearman"]),
            "total_features": int(float(vza_raa["n_features"])),
            "status": "geometry_extension",
        },
        {
            "label": "All VZA features",
            "model_detail": "All compact multiangular current-severity features",
            "feature_set": all_features["feature_set"],
            "rmse": float(all_features["rmse"]),
            "mae": float(all_features["mae"]),
            "r2": float(all_features["r2"]),
            "spearman": float(all_features["spearman"]),
            "total_features": int(all_features["n_features"]),
            "status": "unselected",
        },
        {
            "label": "All VZA+RAA",
            "model_detail": "All VZA plus relative-azimuth current-severity features",
            "feature_set": all_vza_raa["feature_set"],
            "rmse": float(all_vza_raa["rmse"]),
            "mae": float(all_vza_raa["mae"]),
            "r2": float(all_vza_raa["r2"]),
            "spearman": float(all_vza_raa["spearman"]),
            "total_features": int(float(all_vza_raa["n_features"])),
            "status": "unselected_geometry",
        },
        {
            "label": "Nadir only",
            "model_detail": "Compact nadir current-severity baseline",
            "feature_set": nadir["feature_set"],
            "rmse": float(nadir["rmse"]),
            "mae": float(nadir["mae"]),
            "r2": float(nadir["r2"]),
            "spearman": float(nadir["spearman"]),
            "total_features": int(nadir["n_features"]),
            "status": "reference",
        },
    ]
    out = add_rmse_uncertainty(pl.DataFrame(rows))
    log_phase("load current severity RMSE source tables", started)
    return out


def write_plot(data: pl.DataFrame) -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = data.sort("rmse")
    labels = data.get_column("label").to_list()
    rmse = data.get_column("rmse").to_list()
    features = data.get_column("total_features").to_list()
    yerr = np.vstack(
        [
            data.get_column("rmse_err_low").to_numpy(),
            data.get_column("rmse_err_high").to_numpy(),
        ]
    )
    colors_by_label = {
        "VZA only": PALETTE["navy"],
        "VZA+RAA": PALETTE["coral"],
        "All VZA features": PALETTE["coral"],
        "All VZA+RAA": PALETTE["teal"],
        "Nadir only": PALETTE["gold"],
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
    nadir_rmse = float(data.filter(pl.col("label") == "Nadir only")["rmse"][0])
    ax.axhline(nadir_rmse, color=PALETTE["teal"], linewidth=1.2, linestyle=(0, (4, 4)), alpha=0.65)

    for idx, (bar, value, n_features) in enumerate(zip(bars, rmse, features, strict=True)):
        label = labels[idx]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.10,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
            color=PALETTE["navy"],
        )
        inner_color = (
            "white"
            if label in {"VZA only", "VZA+RAA", "All VZA features", "All VZA+RAA"}
            else PALETTE["navy"]
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(value - 0.45, 0.25),
            f"{n_features} features",
            ha="center",
            va="center",
            fontsize=10.5,
            color=inner_color,
            fontweight="bold",
        )

    fig.suptitle(
        "Current Severity Prediction RMSE",
        fontsize=24,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.955,
    )
    ax.text(
        0.015,
        0.965,
        "Lower is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        color=PALETTE["grey"],
    )
    ax.set_ylabel("RMSE severity units", fontsize=12, color=PALETTE["navy"])
    y_top = float(np.max(np.asarray(rmse, dtype=float) + yerr[1]))
    ax.set_ylim(0, y_top + 0.75)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", color=PALETTE["light_grey"], linewidth=0.9, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["navy"])
    ax.spines["bottom"].set_color(PALETTE["navy"])

    fig.tight_layout(rect=[0, 0, 1, 0.945])
    paths = [
        FIGURES_DIR / "current_severity_rmse_nadir_all_selected_barplot.png",
        FIGURES_DIR / "current_severity_rmse_nadir_all_selected_barplot.pdf",
        FIGURES_DIR / "current_severity_rmse_nadir_all_selected_barplot.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=300)
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    log_phase("write current severity RMSE bar plot", started)
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
            "total_features",
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
    all_vza_raa_rmse = float(data.filter(pl.col("label") == "All VZA+RAA")["rmse"][0])
    report = f"""## Results: Current Severity RMSE Slide Bar Plot

{markdown_table(data)}

**Interpretation**: The selected compact VZA-only model reduced 2025 current-severity RMSE by {nadir_rmse - selected_rmse:.3f} severity units relative to the nadir-only reference. The selected VZA+RAA extension changed RMSE by {vza_raa_rmse - selected_rmse:+.3f} severity units relative to VZA-only, while the all-VZA+RAA model changed RMSE by {all_vza_raa_rmse - vza_raa_rmse:+.3f} relative to selected VZA+RAA.

**Outputs**:
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in figure_paths)}

**Reproducibility**:

- Source current-severity model comparison: `{MODEL_COMPARISON.relative_to(ROOT)}`
- Source VZA+RAA current-severity comparison: `{RAA_MODEL_COMPARISON.relative_to(ROOT)}`
- Target: same-week plot-level severity, trained on 2024 and validated on 2025.
- Error bars: absolute plot-level bootstrap 95% CI for each model RMSE; seed=42, n_bootstrap=5000. Paired improvement columns are still computed versus nadir.
- Figure format: 4:3 slide figure, palette Navy `{PALETTE["navy"]}`, Teal `{PALETTE["teal"]}`, Coral `{PALETTE["coral"]}`, Gold `{PALETTE["gold"]}`.
- Log: `{log_path.relative_to(ROOT)}`
"""
    path = REPORTS_DIR / "current_severity_rmse_slide_barplot_summary.md"
    path.write_text(report, encoding="utf-8")
    logging.info("Wrote report: %s", path)
    log_phase("write current severity RMSE bar plot report", started)
    return path


def main() -> None:
    log_path = setup_logging()
    data = load_plot_data()
    figure_paths = write_plot(data)
    write_report(data, figure_paths, log_path)


if __name__ == "__main__":
    main()
