#!/usr/bin/env python3
"""Plot sun-view geometry heatmaps relative to the near-nadir reflectance reference."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAME = Path(__file__).stem
REPORT_DIR = ROOT / "outputs/reports"
LOG_DIR = ROOT / "outputs/logs"
BANDS = {
    "band1": "Blue",
    "band2": "Green",
    "band3": "Red",
    "band4": "Red edge",
    "band5": "NIR",
}
REFERENCE_VZA_MAX = 10.0
MIN_PLOTS_FOR_HEATMAP = 4
MAX_ABS_PERCENT_CHANGE_FOR_HEATMAP = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=[2024, 2025], default=2024)
    parser.add_argument(
        "--filter-state", choices=["ground_filtered", "unfiltered"], default="ground_filtered"
    )
    return parser.parse_args()


def configure_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{SCRIPT_NAME}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return path


def phase(name: str, started: float) -> None:
    logging.info("[PHASE] %s: %.2fs", name, time.perf_counter() - started)


def paths_for_run(year: int, filter_state: str) -> tuple[Path, Path, Path, Path]:
    out_dir = ROOT / f"outputs/result_01_raa_sun_geometry/{year}/{filter_state}"
    input_path = out_dir / f"results/plot_week_vza_raa_features_{year}.parquet"
    return input_path, out_dir / "figures/main", out_dir / "results", out_dir


def load_features(input_path: Path) -> pl.DataFrame:
    started = time.perf_counter()
    read_started = time.perf_counter()
    df = pl.read_parquet(input_path)
    read_seconds = time.perf_counter() - read_started
    logging.info(
        "[IO] parquet reads: n=1 min=%.3fs median=%.3fs max=%.3fs mean=%.3fs",
        read_seconds,
        read_seconds,
        read_seconds,
        read_seconds,
    )
    phase("load_features", started)
    return df


def build_nadir_contrast(df: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    started = time.perf_counter()
    reference_label = "0-10"
    reference = (
        df.filter(pl.col("vza_midpoint") < REFERENCE_VZA_MAX)
        .with_columns((pl.col("reflectance") * pl.col("n_pixels")).alias("weighted_reflectance"))
        .group_by("plot_id", "week", "band")
        .agg(
            (pl.col("weighted_reflectance").sum() / pl.col("n_pixels").sum()).alias(
                "reference_reflectance"
            ),
            pl.col("n_pixels").sum().alias("reference_pixels"),
        )
    )
    if reference.is_empty():
        raise RuntimeError(
            f"No VZA observations below {REFERENCE_VZA_MAX} degrees are available for nadir reference."
        )
    contrast = df.join(reference, on=["plot_id", "week", "band"], how="inner").with_columns(
        (pl.col("reflectance") - pl.col("reference_reflectance")).alias("absolute_contrast"),
        (
            (pl.col("reflectance") - pl.col("reference_reflectance"))
            / pl.col("reference_reflectance")
        ).alias("relative_contrast"),
    )
    summary = (
        contrast.group_by(
            "year",
            "week",
            "band",
            "band_name",
            "vza_class",
            "vza_midpoint",
            "raa_class",
            "raa_midpoint",
        )
        .agg(
            pl.col("plot_id").n_unique().alias("plots"),
            pl.col("n_pixels").sum().alias("pixels"),
            pl.col("absolute_contrast").median().alias("median_absolute_contrast"),
            pl.col("absolute_contrast").mean().alias("mean_absolute_contrast"),
            pl.col("absolute_contrast").quantile(0.25).alias("q25_absolute_contrast"),
            pl.col("absolute_contrast").quantile(0.75).alias("q75_absolute_contrast"),
            (pl.col("relative_contrast").median() * 100.0).alias("median_percent_change"),
        )
        .sort("band", "week", "vza_midpoint", "raa_midpoint")
    )
    phase("build_nadir_contrast", started)
    return summary, reference_label


def build_phase_contrast(df: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    started = time.perf_counter()
    reference_label = "0-10"
    reference = (
        df.filter(pl.col("vza_midpoint") < REFERENCE_VZA_MAX)
        .with_columns((pl.col("reflectance") * pl.col("n_pixels")).alias("weighted_reflectance"))
        .group_by("plot_id", "week", "band")
        .agg(
            (pl.col("weighted_reflectance").sum() / pl.col("n_pixels").sum()).alias(
                "reference_reflectance"
            )
        )
    )
    if reference.is_empty():
        raise RuntimeError(
            f"No VZA observations below {REFERENCE_VZA_MAX} degrees are available for nadir reference."
        )
    contrast = df.join(reference, on=["plot_id", "week", "band"], how="inner").with_columns(
        (pl.col("reflectance") - pl.col("reference_reflectance")).alias("absolute_contrast"),
        (
            (pl.col("reflectance") - pl.col("reference_reflectance"))
            / pl.col("reference_reflectance")
        ).alias("relative_contrast"),
    )
    summary = (
        contrast.group_by(
            "year", "week", "band", "band_name", "phase_class", "raa_class", "raa_midpoint"
        )
        .agg(
            pl.col("phase_midpoint").mean().alias("phase_midpoint"),
            pl.col("plot_id").n_unique().alias("plots"),
            pl.col("n_pixels").sum().alias("pixels"),
            pl.col("absolute_contrast").median().alias("median_absolute_contrast"),
            pl.col("absolute_contrast").mean().alias("mean_absolute_contrast"),
            pl.col("absolute_contrast").quantile(0.25).alias("q25_absolute_contrast"),
            pl.col("absolute_contrast").quantile(0.75).alias("q75_absolute_contrast"),
            (pl.col("relative_contrast").median() * 100.0).alias("median_percent_change"),
        )
        .sort("band", "week", "phase_midpoint", "raa_midpoint")
    )
    phase("build_phase_contrast", started)
    return summary, reference_label


def heatmap_matrix(
    data: pl.DataFrame, value_column: str
) -> tuple[np.ndarray, list[str], list[str]]:
    if data.is_empty():
        return np.empty((0, 0)), [], []
    matrix = (
        data.pivot(
            on="raa_class",
            index="vza_class",
            values=value_column,
            aggregate_function="mean",
        )
        .with_columns(
            pl.col("vza_class").str.split("-").list.first().cast(pl.Int64).alias("vza_low")
        )
        .sort("vza_low")
    )
    raa_cols = sorted(
        [col for col in matrix.columns if col not in {"vza_class", "vza_low"}],
        key=lambda value: int(value.split("-")[0]),
    )
    return matrix.select(raa_cols).to_numpy(), matrix["vza_class"].to_list(), raa_cols


def phase_heatmap_matrix(
    data: pl.DataFrame, value_column: str
) -> tuple[np.ndarray, list[str], list[str]]:
    if data.is_empty():
        return np.empty((0, 0)), [], []
    matrix = (
        data.pivot(
            on="raa_class",
            index="phase_class",
            values=value_column,
            aggregate_function="mean",
        )
        .with_columns(
            pl.col("phase_class").str.split("-").list.first().cast(pl.Int64).alias("phase_low")
        )
        .sort("phase_low")
    )
    raa_cols = sorted(
        [col for col in matrix.columns if col not in {"phase_class", "phase_low"}],
        key=lambda value: int(value.split("-")[0]),
    )
    return matrix.select(raa_cols).to_numpy(), matrix["phase_class"].to_list(), raa_cols


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in {
        ".png": {"dpi": 450},
        ".pdf": {},
        ".svg": {},
    }.items():
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_heatmap(
    summary: pl.DataFrame,
    reference_class: str,
    fig_dir: Path,
    year: int,
    value_column: str = "median_percent_change",
    colorbar_label: str | None = None,
    stem_prefix: str = "raa_vza_nadir_percent_change_heatmap",
    vmax: float | None = None,
    title_metric: str = "percent reflectance change",
    suffix: str = "_trimmed",
    title_note: str = ", extremes removed",
) -> list[Path]:
    started = time.perf_counter()
    weeks = sorted(summary["week"].unique().to_list())
    bands = list(BANDS)
    if vmax is None:
        vmax = MAX_ABS_PERCENT_CHANGE_FOR_HEATMAP
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axes = plt.subplots(
        len(bands),
        len(weeks),
        figsize=(2.35 * len(weeks), 10.8),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    last_image = None
    for row_index, band in enumerate(bands):
        for col_index, week in enumerate(weeks):
            ax = axes[row_index, col_index]
            cell = summary.filter((pl.col("band") == band) & (pl.col("week") == week))
            values, vza_labels, raa_labels = heatmap_matrix(cell, value_column)
            if values.size == 0:
                ax.axis("off")
                continue
            last_image = ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=norm)
            if row_index == 0:
                ax.set_title(f"Week {week}", fontsize=9, fontweight="bold")
            if col_index == 0:
                ax.set_ylabel(f"{BANDS[band]}\nVZA class (deg)", fontsize=8.5, fontweight="bold")
                ax.set_yticks(range(len(vza_labels)), vza_labels, fontsize=6.5)
            else:
                ax.set_yticks(range(len(vza_labels)), [])
            if row_index == len(bands) - 1:
                ax.set_xlabel("RAA to sun (deg)", fontsize=8)
                ax.set_xticks(
                    range(len(raa_labels)), raa_labels, rotation=45, ha="right", fontsize=6.2
                )
            else:
                ax.set_xticks(range(len(raa_labels)), [])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes,
            shrink=0.58,
            label=colorbar_label or f"Median reflectance change vs {reference_class} VZA (%)",
        )
    fig.suptitle(
        f"{year}: {title_metric} over VZA x RAA{title_note}",
        fontsize=12,
        fontweight="bold",
    )
    outputs = save_figure(fig, fig_dir / f"{stem_prefix}_{year}{suffix}")
    phase("plot_heatmap", started)
    return outputs


def plot_phase_heatmap(
    summary: pl.DataFrame,
    reference_class: str,
    fig_dir: Path,
    year: int,
    value_column: str = "median_percent_change",
    colorbar_label: str | None = None,
    stem_prefix: str = "raa_phase_nadir_percent_change_heatmap",
    vmax: float | None = None,
    title_metric: str = "percent reflectance change",
    suffix: str = "_trimmed",
    title_note: str = ", extremes removed",
) -> list[Path]:
    started = time.perf_counter()
    weeks = sorted(summary["week"].unique().to_list())
    bands = list(BANDS)
    if vmax is None:
        vmax = MAX_ABS_PERCENT_CHANGE_FOR_HEATMAP
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axes = plt.subplots(
        len(bands),
        len(weeks),
        figsize=(2.35 * len(weeks), 10.8),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    last_image = None
    for row_index, band in enumerate(bands):
        for col_index, week in enumerate(weeks):
            ax = axes[row_index, col_index]
            cell = summary.filter((pl.col("band") == band) & (pl.col("week") == week))
            values, phase_labels, raa_labels = phase_heatmap_matrix(cell, value_column)
            if values.size == 0:
                ax.axis("off")
                continue
            last_image = ax.imshow(values, aspect="auto", cmap="RdBu_r", norm=norm)
            if row_index == 0:
                ax.set_title(f"Week {week}", fontsize=9, fontweight="bold")
            if col_index == 0:
                ax.set_ylabel(f"{BANDS[band]}\nPhase angle (deg)", fontsize=8.5, fontweight="bold")
                ax.set_yticks(range(len(phase_labels)), phase_labels, fontsize=6.0)
            else:
                ax.set_yticks(range(len(phase_labels)), [])
            if row_index == len(bands) - 1:
                ax.set_xlabel("RAA to sun (deg)", fontsize=8)
                ax.set_xticks(
                    range(len(raa_labels)), raa_labels, rotation=45, ha="right", fontsize=6.2
                )
            else:
                ax.set_xticks(range(len(raa_labels)), [])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=axes,
            shrink=0.58,
            label=colorbar_label or f"Median reflectance change vs {reference_class} VZA (%)",
        )
    fig.suptitle(
        f"{year}: {title_metric} over phase angle x RAA{title_note}",
        fontsize=12,
        fontweight="bold",
    )
    outputs = save_figure(fig, fig_dir / f"{stem_prefix}_{year}{suffix}")
    phase("plot_phase_heatmap", started)
    return outputs


def trim_for_heatmap(summary: pl.DataFrame) -> pl.DataFrame:
    return summary.filter(
        (pl.col("plots") >= MIN_PLOTS_FOR_HEATMAP)
        & (pl.col("median_percent_change").abs() <= MAX_ABS_PERCENT_CHANGE_FOR_HEATMAP)
    )


def markdown_table(top: pl.DataFrame, geometry_column: str) -> list[str]:
    table_lines = [
        f"| week | band_name | {geometry_column} | raa_class | plots | median_percent_change |",
        f"|---:|---|---|---|---:|---:|",
    ]
    for row in top.to_dicts():
        table_lines.append(
            "| {week} | {band_name} | {geometry} | {raa_class} | {plots} | {contrast:.2f} |".format(
                week=row["week"],
                band_name=row["band_name"],
                geometry=row[geometry_column],
                raa_class=row["raa_class"],
                plots=row["plots"],
                contrast=row["median_percent_change"],
            )
        )
    return table_lines


def write_report(
    year: int,
    filter_state: str,
    input_path: Path,
    summary: pl.DataFrame,
    phase_summary: pl.DataFrame,
    reference_class: str,
    outputs: list[Path],
    phase_outputs: list[Path],
    untrimmed_outputs: list[Path],
    untrimmed_phase_outputs: list[Path],
    absolute_outputs: list[Path],
    absolute_phase_outputs: list[Path],
    summary_path: Path,
    phase_summary_path: Path,
    untrimmed_summary_path: Path,
    untrimmed_phase_summary_path: Path,
    log_path: Path,
) -> None:
    started = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    top = (
        summary.filter(pl.col("plots") >= 4)
        .with_columns(pl.col("median_percent_change").abs().alias("abs_median_percent_change"))
        .sort("abs_median_percent_change", descending=True)
        .head(12)
        .select("week", "band_name", "vza_class", "raa_class", "plots", "median_percent_change")
    )
    phase_top = (
        phase_summary.filter(pl.col("plots") >= 4)
        .with_columns(pl.col("median_percent_change").abs().alias("abs_median_percent_change"))
        .sort("abs_median_percent_change", descending=True)
        .head(12)
        .select("week", "band_name", "phase_class", "raa_class", "plots", "median_percent_change")
    )
    lines = [
        f"## Results: {year} VZA x RAA Nadir-Reference Contrast",
        "",
        *markdown_table(top, "vza_class"),
        "",
        f"## Results: {year} Phase Angle x RAA Nadir-Reference Contrast",
        "",
        *markdown_table(phase_top, "phase_class"),
        "",
        (
            f"**Interpretation**: The heatmaps show median plot-level percent reflectance change relative to the "
            f"`{reference_class}` near-nadir VZA reference from the same plot, week, and band. The phase-angle version "
            "summarizes the combined sun-view geometry more directly than VZA alone; positive cells indicate brighter "
            "views than the near-nadir reference. RAA and phase angle are based on the corrected sun geometry feature table, "
            "not the stale stored solar columns from the original extraction bug."
        ),
        "",
        "**Outputs**:",
        *[f"- `{path}`" for path in outputs],
        *[f"- `{path}`" for path in phase_outputs],
        *[f"- `{path}`" for path in untrimmed_outputs],
        *[f"- `{path}`" for path in untrimmed_phase_outputs],
        *[f"- `{path}`" for path in absolute_outputs],
        *[f"- `{path}`" for path in absolute_phase_outputs],
        f"- `{summary_path}`",
        f"- `{phase_summary_path}`",
        f"- `{untrimmed_summary_path}`",
        f"- `{untrimmed_phase_summary_path}`",
        f"- `{log_path}`",
        "",
        "**Config**:",
        f"- Input: `{input_path}`",
        f"- Year: `{year}`",
        f"- Filter state: `{filter_state}`",
        "- Ground filter: existing source table, generated with OSAVI > 0.2 when filter state is `ground_filtered`",
        f"- Reference: per-plot/week/band pixel-weighted mean reflectance for `0 <= VZA < {REFERENCE_VZA_MAX:g}`",
        "- Aggregation: median percent change across plot-level VZA x RAA and phase-angle x RAA cells",
        "- Random seed: not used",
    ]
    report_path = REPORT_DIR / f"{SCRIPT_NAME}_{year}_{filter_state}_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    if year == 2024 and filter_state == "ground_filtered":
        legacy_path = REPORT_DIR / f"{SCRIPT_NAME}_summary.md"
        legacy_path.write_text("\n".join(lines), encoding="utf-8")
    phase("write_report", started)


def main() -> None:
    args = parse_args()
    input_path, fig_dir, results_dir, _ = paths_for_run(args.year, args.filter_state)
    log_path = configure_logging()
    total_started = time.perf_counter()
    logging.info(
        "Starting %s for year=%s filter_state=%s", SCRIPT_NAME, args.year, args.filter_state
    )
    df = load_features(input_path)
    summary, reference_class = build_nadir_contrast(df)
    phase_summary, _ = build_phase_contrast(df)
    plot_summary = trim_for_heatmap(summary)
    plot_phase_summary = trim_for_heatmap(phase_summary)
    results_dir.mkdir(parents=True, exist_ok=True)
    untrimmed_summary_path = results_dir / f"raa_vza_nadir_percent_change_summary_{args.year}.csv"
    untrimmed_phase_summary_path = (
        results_dir / f"raa_phase_nadir_percent_change_summary_{args.year}.csv"
    )
    summary_path = results_dir / f"raa_vza_nadir_percent_change_summary_{args.year}_trimmed.csv"
    phase_summary_path = (
        results_dir / f"raa_phase_nadir_percent_change_summary_{args.year}_trimmed.csv"
    )
    summary.write_csv(untrimmed_summary_path)
    phase_summary.write_csv(untrimmed_phase_summary_path)
    plot_summary.write_csv(summary_path)
    plot_phase_summary.write_csv(phase_summary_path)
    absolute_vmax = float(
        max(
            summary["median_absolute_contrast"].abs().quantile(0.98),
            phase_summary["median_absolute_contrast"].abs().quantile(0.98),
            1e-6,
        )
    )
    absolute_outputs = plot_heatmap(
        summary,
        reference_class,
        fig_dir,
        args.year,
        value_column="median_absolute_contrast",
        colorbar_label=f"Median reflectance difference vs {reference_class} VZA",
        stem_prefix="raa_vza_nadir_contrast_heatmap",
        vmax=absolute_vmax,
        title_metric="median reflectance difference",
        suffix="",
        title_note="",
    )
    absolute_phase_outputs = plot_phase_heatmap(
        phase_summary,
        reference_class,
        fig_dir,
        args.year,
        value_column="median_absolute_contrast",
        colorbar_label=f"Median reflectance difference vs {reference_class} VZA",
        stem_prefix="raa_phase_nadir_contrast_heatmap",
        vmax=absolute_vmax,
        title_metric="median reflectance difference",
        suffix="",
        title_note="",
    )
    untrimmed_outputs = plot_heatmap(
        summary, reference_class, fig_dir, args.year, suffix="", title_note=""
    )
    untrimmed_phase_outputs = plot_phase_heatmap(
        phase_summary, reference_class, fig_dir, args.year, suffix="", title_note=""
    )
    outputs = plot_heatmap(plot_summary, reference_class, fig_dir, args.year)
    phase_outputs = plot_phase_heatmap(plot_phase_summary, reference_class, fig_dir, args.year)
    write_report(
        args.year,
        args.filter_state,
        input_path,
        plot_summary,
        plot_phase_summary,
        reference_class,
        outputs,
        phase_outputs,
        untrimmed_outputs,
        untrimmed_phase_outputs,
        absolute_outputs,
        absolute_phase_outputs,
        summary_path,
        phase_summary_path,
        untrimmed_summary_path,
        untrimmed_phase_summary_path,
        log_path,
    )
    phase("total", total_started)


if __name__ == "__main__":
    main()
