#!/usr/bin/env python3
"""Generate nadir-vs-multiangular AUROC comparison bar chart."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

SUMMARY_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "outputs" / "results" / "model_comparison_summary.csv"
)
FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not SUMMARY_PATH.exists():
        print(f"Summary file not found: {SUMMARY_PATH}")
        sys.exit(1)

    df = pl.read_csv(SUMMARY_PATH)
    # ensure consistent ordering
    order = ["M1", "M2", "M3", "M4", "M5"]
    df = df.with_columns(pl.col("feature_set").cast(pl.Categorical))
    df = df.filter(pl.col("feature_set").is_in(order))
    df_dict = {row["feature_set"]: row for row in df.iter_rows(named=True)}

    ordered = {k: df_dict[k] for k in order if k in df_dict}

    labels = list(ordered.keys())
    means = [ordered[k]["AUROC_mean"] for k in labels]
    stds = [ordered[k]["AUROC_std"] for k in labels]

    color_map = {
        "M1": "#3498db",
        "M2": "#3498db",
        "M3": "#e74c3c",
        "M4": "#e74c3c",
        "M5": "#e74c3c",
    }
    colors = [color_map.get(l, "#999999") for l in labels]

    display_labels = {
        "M1": "M1\nNadir bands",
        "M2": "M2\nNadir indices",
        "M3": "M3\nVZA",
        "M4": "M4\nVZA+RAA",
        "M5": "M5\nAngular\ncontrast",
    }
    tick_labels = [display_labels.get(l, l) for l in labels]

    # group bar positions with space between groups
    x_positions = [0.0, 1.0, 2.5, 3.5, 4.5]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        x_positions,
        means,
        yerr=stds,
        color=colors,
        capsize=6,
        width=0.7,
        edgecolor="white",
        linewidth=0.8,
    )

    # add value labels on top of bars
    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # nadir reference line
    nadir_means = [ordered[k]["AUROC_mean"] for k in labels if k in ("M1", "M2")]
    if nadir_means:
        ref = np.mean(nadir_means)
        ax.axhline(y=ref, color="#3498db", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(4.9, ref + 0.01, f"nadir mean\n({ref:.3f})", fontsize=8,
                color="#3498db", ha="right", va="bottom")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_ylabel("Mean AUROC (5-fold CV)", fontsize=12)
    ax.set_title("Nadir vs Multiangular Feature Sets for Disease Prediction",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.0)

    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498db", label="Nadir (M1-M2)"),
        Patch(facecolor="#e74c3c", label="Multiangular (M3-M5)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path = FIGURES_DIR / "nadir_vs_multiangular_auroc.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
