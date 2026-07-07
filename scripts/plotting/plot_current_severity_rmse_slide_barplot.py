"""Create a slide-ready RMSE bar plot for current severity prediction."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/current_severity_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/logs"

MODEL_COMPARISON = RESULTS_DIR / "current_severity_model_comparison.csv"

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


def load_plot_data() -> pl.DataFrame:
    started = time.perf_counter()
    current = pl.read_csv(MODEL_COMPARISON)

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

    rows = [
        {
            "label": "Selected multiangular",
            "model_detail": "Selected compact multiangular current-severity model",
            "feature_set": selected["feature_set"],
            "rmse": float(selected["rmse"]),
            "mae": float(selected["mae"]),
            "r2": float(selected["r2"]),
            "spearman": float(selected["spearman"]),
            "total_features": int(float(selected["n_features"])),
            "status": "selected",
        },
        {
            "label": "All features",
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
    out = pl.DataFrame(rows)
    log_phase("load current severity RMSE source tables", started)
    return out


def write_plot(data: pl.DataFrame) -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = data.sort("rmse")
    labels = data.get_column("label").to_list()
    rmse = data.get_column("rmse").to_list()
    features = data.get_column("total_features").to_list()
    colors_by_label = {
        "Selected multiangular": PALETTE["navy"],
        "All features": PALETTE["coral"],
        "Nadir only": PALETTE["gold"],
    }
    colors = [colors_by_label[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10.0, 6.25))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(labels, rmse, color=colors, width=0.62, edgecolor=PALETTE["navy"], linewidth=0.8)
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
            "white" if label in {"Selected multiangular", "All features"} else PALETTE["navy"]
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
    ax.set_ylim(0, max(rmse) + 1.05)
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
    selected_rmse = float(data.filter(pl.col("label") == "Selected multiangular")["rmse"][0])
    report = f"""## Results: Current Severity RMSE Slide Bar Plot

{markdown_table(data)}

**Interpretation**: The selected compact multiangular model reduced 2025 current-severity RMSE by {nadir_rmse - selected_rmse:.3f} severity units relative to the nadir-only reference. The all-feature compact multiangular model performs worse than the selected model, so the slide emphasizes both multiangular information and feature selection.

**Outputs**:
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in figure_paths)}

**Reproducibility**:

- Source current-severity model comparison: `{MODEL_COMPARISON.relative_to(ROOT)}`
- Target: same-week plot-level severity, trained on 2024 and validated on 2025.
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
