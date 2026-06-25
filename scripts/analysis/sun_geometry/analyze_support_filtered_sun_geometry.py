#!/usr/bin/env python3
"""Support-filtered sun-view geometry analysis for defensible noise removal."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAME = Path(__file__).stem
YEARS = [2024, 2025]
BANDS = {
    "band1": "Blue",
    "band2": "Green",
    "band3": "Red",
    "band4": "Red edge",
    "band5": "NIR",
}
MAIN_BANDS = ["band4", "band5"]
MIN_CELL_PLOTS = 15
MAX_WEEK = 6
REFERENCE_VZA_MAX = 10.0
MAX_ABS_PERCENT_FOR_DISPLAY = 30.0
SEED = 42

OUT_DIR = ROOT / "outputs/result_01_raa_sun_geometry/support_filtered"
FIG_DIR = OUT_DIR / "figures"
RESULTS_DIR = OUT_DIR / "results"
REPORT_DIR = ROOT / "outputs/reports"
LOG_DIR = ROOT / "outputs/logs"

VZA_ONLY_FORMULA = "reflectance ~ C(vza_class) + C(week) + C(cult) + C(trt)"
PHASE_FORMULA = "reflectance ~ C(vza_class) + mean_phase_angle + C(week) + C(cult) + C(trt)"
RAA_FORMULA = (
    "reflectance ~ C(vza_class) + C(raa_class) + C(vza_class):C(raa_class) "
    "+ C(week) + C(cult) + C(trt)"
)


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


def log_phase(name: str, started: float) -> None:
    logging.info("[PHASE] %s: %.2fs", name, time.perf_counter() - started)


def input_path(year: int) -> Path:
    return (
        ROOT
        / f"outputs/result_01_raa_sun_geometry/{year}/ground_filtered/results/plot_week_vza_raa_features_{year}.parquet"
    )


def load_all_features() -> pl.DataFrame:
    started = time.perf_counter()
    frames = []
    read_times = []
    for year in YEARS:
        path = input_path(year)
        read_started = time.perf_counter()
        frames.append(pl.read_parquet(path))
        read_times.append(time.perf_counter() - read_started)
    logging.info(
        "[IO] parquet reads: n=%d min=%.3fs median=%.3fs max=%.3fs mean=%.3fs",
        len(read_times),
        min(read_times),
        float(np.median(read_times)),
        max(read_times),
        float(np.mean(read_times)),
    )
    data = pl.concat(frames, how="diagonal_relaxed")
    logging.info("Loaded %d plot-cell-band rows", data.height)
    log_phase("load_features", started)
    return data


def add_support(data: pl.DataFrame) -> pl.DataFrame:
    started = time.perf_counter()
    support = data.group_by("year", "week", "band", "vza_class", "raa_class").agg(
        pl.col("plot_id").n_unique().alias("cell_plots"),
        pl.col("n_pixels").sum().alias("cell_pixels"),
        pl.col("n_images").sum().alias("cell_images"),
    )
    result = data.join(support, on=["year", "week", "band", "vza_class", "raa_class"], how="left")
    log_phase("add_support", started)
    return result


def filter_variant(data: pl.DataFrame, variant: str) -> pl.DataFrame:
    filtered = data.filter(
        (pl.col("week") <= MAX_WEEK)
        & (pl.col("cell_plots") >= MIN_CELL_PLOTS)
        & pl.col("reflectance").is_finite()
        & pl.col("mean_phase_angle").is_finite()
    )
    if variant == "main_no_week3":
        filtered = filtered.filter(pl.col("week") != 3)
    elif variant != "main_with_week3":
        raise ValueError(f"Unknown variant: {variant}")
    return filtered


def build_phase_contrast(data: pl.DataFrame, variant: str) -> pl.DataFrame:
    started = time.perf_counter()
    reference = (
        data.filter(pl.col("vza_midpoint") < REFERENCE_VZA_MAX)
        .group_by("year", "plot_id", "week", "band")
        .agg(
            ((pl.col("reflectance") * pl.col("n_pixels")).sum() / pl.col("n_pixels").sum()).alias(
                "reference_reflectance"
            ),
            pl.col("cell_plots").max().alias("reference_cell_plots"),
        )
        .filter(pl.col("reference_cell_plots") >= MIN_CELL_PLOTS)
    )
    contrast = data.join(
        reference, on=["year", "plot_id", "week", "band"], how="inner"
    ).with_columns(
        (pl.col("reflectance") - pl.col("reference_reflectance")).alias("absolute_contrast"),
        (
            (pl.col("reflectance") - pl.col("reference_reflectance"))
            / pl.col("reference_reflectance")
            * 100.0
        ).alias("percent_change"),
        pl.lit(variant).alias("variant"),
    )
    summary = (
        contrast.group_by(
            "variant", "year", "week", "band", "band_name", "phase_class", "phase_midpoint"
        )
        .agg(
            pl.col("plot_id").n_unique().alias("plots"),
            pl.len().alias("observations"),
            pl.col("percent_change").median().alias("median_percent_change"),
            pl.col("percent_change").mean().alias("mean_percent_change"),
            (pl.col("percent_change").std() / pl.len().sqrt()).alias("se_percent_change"),
            pl.col("percent_change").quantile(0.25).alias("q25_percent_change"),
            pl.col("percent_change").quantile(0.75).alias("q75_percent_change"),
        )
        .sort("variant", "year", "band", "week", "phase_midpoint")
    )
    log_phase(f"phase_contrast_{variant}", started)
    return summary


def fit_models(data: pl.DataFrame, variant: str) -> pl.DataFrame:
    started = time.perf_counter()
    rows = []
    for year in YEARS:
        for band, band_name in BANDS.items():
            frame = data.filter((pl.col("year") == year) & (pl.col("band") == band)).to_pandas()
            if frame.empty or frame["plot_id"].nunique() < 2:
                continue
            fit_started = time.perf_counter()
            base = smf.ols(VZA_ONLY_FORMULA, frame).fit(
                cov_type="cluster", cov_kwds={"groups": frame["plot_id"]}
            )
            phase_model = smf.ols(PHASE_FORMULA, frame).fit(
                cov_type="cluster", cov_kwds={"groups": frame["plot_id"]}
            )
            raa_model = smf.ols(RAA_FORMULA, frame).fit(
                cov_type="cluster", cov_kwds={"groups": frame["plot_id"]}
            )
            logging.info(
                "[ML] fit %s %s %s: %.2fs",
                variant,
                year,
                band_name,
                time.perf_counter() - fit_started,
            )
            rows.append(
                {
                    "variant": variant,
                    "year": year,
                    "band": band,
                    "band_name": band_name,
                    "nobs": int(base.nobs),
                    "plots": int(frame["plot_id"].nunique()),
                    "weeks": ",".join(map(str, sorted(frame["week"].unique()))),
                    "vza_r2": float(base.rsquared),
                    "phase_r2": float(phase_model.rsquared),
                    "raa_r2": float(raa_model.rsquared),
                    "delta_r2_phase_vs_vza": float(phase_model.rsquared - base.rsquared),
                    "delta_r2_raa_vs_vza": float(raa_model.rsquared - base.rsquared),
                    "vza_adj_r2": float(base.rsquared_adj),
                    "phase_adj_r2": float(phase_model.rsquared_adj),
                    "raa_adj_r2": float(raa_model.rsquared_adj),
                    "delta_adj_r2_phase_vs_vza": float(
                        phase_model.rsquared_adj - base.rsquared_adj
                    ),
                    "delta_adj_r2_raa_vs_vza": float(raa_model.rsquared_adj - base.rsquared_adj),
                    "delta_aic_phase_vs_vza": float(phase_model.aic - base.aic),
                    "delta_aic_raa_vs_vza": float(raa_model.aic - base.aic),
                    "delta_bic_phase_vs_vza": float(phase_model.bic - base.bic),
                    "delta_bic_raa_vs_vza": float(raa_model.bic - base.bic),
                    "phase_angle_estimate": float(
                        phase_model.params.get("mean_phase_angle", np.nan)
                    ),
                    "phase_angle_p": float(phase_model.pvalues.get("mean_phase_angle", np.nan)),
                }
            )
    result = pl.DataFrame(rows)
    log_phase(f"model_fitting_{variant}", started)
    return result


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in {
        ".png": {"dpi": 350},
        ".pdf": {},
        ".svg": {},
    }.items():
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_phase_curves(summary: pl.DataFrame, variant: str) -> list[Path]:
    started = time.perf_counter()
    data = summary.filter(pl.col("band").is_in(MAIN_BANDS))
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True, sharex=True)
    for row_idx, band in enumerate(MAIN_BANDS):
        for col_idx, year in enumerate(YEARS):
            ax = axes[row_idx, col_idx]
            panel = data.filter((pl.col("band") == band) & (pl.col("year") == year))
            for week in sorted(panel["week"].unique().to_list()):
                curve = panel.filter(pl.col("week") == week).sort("phase_midpoint")
                x = curve["phase_midpoint"].to_numpy()
                y = curve["median_percent_change"].to_numpy()
                se = curve["se_percent_change"].fill_null(0).to_numpy()
                ax.errorbar(
                    x,
                    y,
                    yerr=se,
                    marker="o",
                    linewidth=1.3,
                    markersize=3.2,
                    capsize=2,
                    label=f"W{week}",
                )
            ax.axhline(0, color="#333333", linewidth=0.8)
            ax.set_title(f"{year} {BANDS[band]}", loc="left", fontsize=10, fontweight="bold")
            ax.set_xlabel("Phase angle bin midpoint (deg)")
            ax.set_ylabel("% change vs 0-10 VZA")
            ax.grid(axis="y", color="#E5E1D8", linewidth=0.6)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(labels)), frameon=False)
    fig.suptitle(
        f"Support-filtered phase-angle response ({variant.replace('_', ' ')})",
        fontsize=13,
        fontweight="bold",
    )
    outputs = save_figure(fig, FIG_DIR / f"phase_angle_percent_change_{variant}")
    log_phase(f"plot_phase_curves_{variant}", started)
    return outputs


def write_report(
    model_tables: list[pl.DataFrame],
    phase_tables: list[pl.DataFrame],
    figure_outputs: list[Path],
    log_path: Path,
) -> None:
    started = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    models = pl.concat(model_tables, how="diagonal_relaxed")
    main = models.filter(pl.col("band").is_in(MAIN_BANDS)).sort("variant", "year", "band")
    lines = [
        "## Results: Support-Filtered Sun-View Geometry",
        "",
        "| Variant | Year | Band | Weeks | Plots | VZA R2 | Phase R2 | Delta adj R2 phase | Phase coef | Phase p | RAA R2 | Delta adj R2 RAA |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main.iter_rows(named=True):
        lines.append(
            f"| {row['variant']} | {row['year']} | {row['band_name']} | {row['weeks']} | {row['plots']} | "
            f"{row['vza_r2']:.4f} | {row['phase_r2']:.4f} | {row['delta_adj_r2_phase_vs_vza']:.4f} | "
            f"{row['phase_angle_estimate']:.6f} | {row['phase_angle_p']:.3g} | "
            f"{row['raa_r2']:.4f} | {row['delta_adj_r2_raa_vs_vza']:.4f} |"
        )
    lines.extend(
        [
            "",
            "**Interpretation**: This is an objective noise-removal analysis. It keeps only cells with at least "
            f"`{MIN_CELL_PLOTS}` plots and only weeks `0-{MAX_WEEK}`. The `main_no_week3` variant tests whether the "
            "phase-angle signal holds after removing the low-support week 3. A consistent negative phase coefficient "
            "supports the backscatter/low-phase reflectance assumption without manually changing values.",
            "",
            "**Outputs**:",
            *[f"- `{path}`" for path in figure_outputs],
            f"- `{RESULTS_DIR / 'support_filtered_model_comparison.csv'}`",
            f"- `{RESULTS_DIR / 'support_filtered_phase_percent_change.csv'}`",
            f"- `{log_path}`",
            "",
            "**Config**:",
            f"- Cell support threshold: `cell_plots >= {MIN_CELL_PLOTS}`",
            f"- Main biological window: `week <= {MAX_WEEK}`",
            "- Variants: `main_with_week3`, `main_no_week3`",
            f"- Reference: per-plot/week/band weighted mean for `0 <= VZA < {REFERENCE_VZA_MAX:g}`",
            "- Inputs: corrected `plot_week_vza_raa_features_{year}.parquet` tables from ground-filtered RAA analysis",
            f"- Random seed: `{SEED}`",
        ]
    )
    report_path = REPORT_DIR / f"{SCRIPT_NAME}_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log_phase("write_report", started)


def main() -> None:
    log_path = configure_logging()
    total_started = time.perf_counter()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = add_support(load_all_features())
    model_tables = []
    phase_tables = []
    figure_outputs = []
    for variant in ["main_with_week3", "main_no_week3"]:
        variant_data = filter_variant(data, variant)
        logging.info(
            "%s retained %d rows, %d plot-year-week records",
            variant,
            variant_data.height,
            variant_data.select("year", "plot_id", "week").unique().height,
        )
        phase_summary = build_phase_contrast(variant_data, variant)
        models = fit_models(variant_data, variant)
        phase_tables.append(phase_summary)
        model_tables.append(models)
        figure_outputs.extend(plot_phase_curves(phase_summary, variant))
    all_models = pl.concat(model_tables, how="diagonal_relaxed")
    all_phase = pl.concat(phase_tables, how="diagonal_relaxed")
    all_models.write_csv(RESULTS_DIR / "support_filtered_model_comparison.csv")
    all_phase.write_csv(RESULTS_DIR / "support_filtered_phase_percent_change.csv")
    write_report(model_tables, phase_tables, figure_outputs, log_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
