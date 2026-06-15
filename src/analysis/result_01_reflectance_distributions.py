#!/usr/bin/env python3
"""Create preliminary Result 1 tables and figures for the 2024 season."""

from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs/ARCHIVED/features/M3_multiangular_vza.parquet"
DEFAULT_OUTPUT = ROOT / "outputs/result_01_reflectance_distributions"

BANDS = {
    "band1": "Blue",
    "band2": "Green",
    "band3": "Red",
    "band4": "Red edge",
    "band5": "NIR",
}
ANGLES = ["0-15", "15-25", "25-35", "35-45", "45-60"]
ANGLE_MIDPOINTS = {"0-15": 7.5, "15-25": 20.0, "25-35": 30.0, "35-45": 40.0, "45-60": 52.5}
WEEK_COLORS = {
    0: "#B8E0D2",
    2: "#8DD3C7",
    3: "#66C2A4",
    4: "#4EB3D3",
    5: "#3288BD",
    6: "#5E4FA2",
    7: "#3B4CC0",
    8: "#253494",
}
CULTIVAR_STYLES = {"aluco": ("#0072B2", "-"), "capone": ("#D55E00", "--")}
SEED = 42
MAX_PLOT_SAMPLE = 500_000
WEEK_PLOT_DIRS = {
    0: Path("/run/media/davidem/Heim/2024/20240603_week0/metashape/20241205_products_uav_data/output/extract/polygon_df"),
    3: Path("/run/media/davidem/Heim/2024/20240624_week3/metashape/20241206_week3_products_uav_data/output/plots"),
    5: Path("/run/media/davidem/Heim/2024/20240715_week5/metashape/20241207_week5_products_uav_data/output/plots"),
    6: Path("/run/media/davidem/Heim/2024/recovered_weeks/week6/output/plots"),
    8: Path("/run/media/davidem/Heim/2024/20240826_week8/metashape/20241029_products_uav_data/output/extract/polygon_df"),
}
POLYGON_PATH = Path("/run/media/davidem/Heim/2024_oncerco_plot_polygons.gpkg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fine-vza-bins",
        action="store_true",
        help="Aggregate from plot-level parquets into 5-degree VZA bins instead of using broad cached bins.",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> Path:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"result_01_preliminary_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return path


def log_phase(name: str, started: float) -> None:
    logging.info("[PHASE] %s: %.2fs", name, time.perf_counter() - started)


def load_long(input_path: Path) -> pl.DataFrame:
    started = time.perf_counter()
    read_started = time.perf_counter()
    wide = pl.read_parquet(input_path).filter(pl.col("year") == 2024)
    read_seconds = time.perf_counter() - read_started
    logging.info(
        "[IO] parquet reads: n=1 min=%.3fs median=%.3fs max=%.3fs mean=%.3fs",
        read_seconds,
        read_seconds,
        read_seconds,
        read_seconds,
    )
    value_columns = [column for column in wide.columns if re.fullmatch(r"band[1-5]_vza\d+_\d+", column)]
    long = (
        wide.unpivot(
            index=["plot_id", "week", "cult", "trt", "year"],
            on=value_columns,
            variable_name="feature",
            value_name="reflectance",
        )
        .with_columns(
            pl.col("feature").str.extract(r"^(band[1-5])", 1).alias("band"),
            pl.col("feature").str.extract(r"vza(\d+)_(\d+)", 1).alias("vza_low"),
            pl.col("feature").str.extract(r"vza(\d+)_(\d+)", 2).alias("vza_high"),
        )
        .with_columns(
            (pl.col("vza_low") + pl.lit("-") + pl.col("vza_high")).alias("vza_class"),
            pl.col("reflectance").cast(pl.Float64),
        )
        .drop("feature", "vza_low", "vza_high")
        .filter(pl.col("reflectance").is_finite())
        .with_columns(
            pl.col("band").replace_strict(BANDS).alias("band_name"),
            pl.col("vza_class").replace_strict(ANGLE_MIDPOINTS).cast(pl.Float64).alias("vza_midpoint"),
        )
    )
    logging.info("Loaded %d finite angle observations", long.height)
    log_phase("load_and_reshape", started)
    return long


def load_polygon_meta() -> dict[str, dict[str, object]]:
    import geopandas as gpd

    gdf = gpd.read_file(POLYGON_PATH)
    gdf["disease_label"] = gdf["ino"].astype(int)
    gdf["plot_id"] = "plot_" + (gdf["ifz_id"] - 90001).astype(str)
    return {row["plot_id"]: row for row in gdf[["plot_id", "cult", "trt", "disease_label"]].to_dict("records")}


def assign_fine_vza_bins(frame: pl.DataFrame) -> pl.DataFrame:
    vza_low = (((pl.col("vza") - 10) / 5).floor() * 5 + 10).cast(pl.Int64)
    return (
        frame.filter((pl.col("vza") >= 10) & (pl.col("vza") < 60))
        .with_columns(vza_low.alias("vza_low"))
        .with_columns(
            (pl.col("vza_low").cast(pl.Utf8) + pl.lit("-") + (pl.col("vza_low") + 5).cast(pl.Utf8)).alias(
                "vza_class"
            ),
            (pl.col("vza_low") + 2.5).cast(pl.Float64).alias("vza_midpoint"),
        )
        .drop("vza_low")
    )


def load_fine_vza_from_plot_parquets() -> pl.DataFrame:
    started = time.perf_counter()
    meta = load_polygon_meta()
    rows = []
    read_seconds = []
    band_columns = list(BANDS)
    for week, directory in WEEK_PLOT_DIRS.items():
        files = sorted(directory.glob("plot_*.parquet"))
        logging.info("Week %s: reading %d plot parquets from %s", week, len(files), directory)
        for path in files:
            plot_id = path.stem
            if plot_id not in meta:
                continue
            read_started = time.perf_counter()
            frame = pl.read_parquet(path)
            read_seconds.append(time.perf_counter() - read_started)
            if frame.height > MAX_PLOT_SAMPLE:
                frame = frame.sample(n=MAX_PLOT_SAMPLE, seed=SEED)
            mask = pl.col("vza").is_not_nan() & (pl.col("vza") >= 10) & (pl.col("vza") < 60)
            for band in band_columns:
                mask = mask & pl.col(band).is_not_nan() & (pl.col(band) > 0)
            frame = assign_fine_vza_bins(frame.filter(mask))
            if frame.is_empty():
                continue
            summary = frame.group_by("vza_class", "vza_midpoint").agg([pl.col(band).mean().alias(band) for band in band_columns])
            for record in summary.to_dicts():
                for band in band_columns:
                    rows.append(
                        {
                            "plot_id": plot_id,
                            "week": week,
                            "cult": meta[plot_id]["cult"],
                            "trt": meta[plot_id]["trt"],
                            "year": 2024,
                            "band": band,
                            "band_name": BANDS[band],
                            "vza_class": record["vza_class"],
                            "vza_midpoint": record["vza_midpoint"],
                            "reflectance": record[band],
                        }
                    )
    if not rows:
        raise RuntimeError("No fine-bin plot-level observations were built.")
    if read_seconds:
        logging.info(
            "[IO] parquet reads: n=%d min=%.3fs median=%.3fs max=%.3fs mean=%.3fs",
            len(read_seconds),
            min(read_seconds),
            float(np.median(read_seconds)),
            max(read_seconds),
            float(np.mean(read_seconds)),
        )
    long = pl.DataFrame(rows).filter(pl.col("reflectance").is_finite())
    coverage = long.group_by("week", "vza_class").agg(pl.col("plot_id").n_unique().alias("plots"))
    sparse = coverage.filter(pl.col("plots") < 10)
    if sparse.height:
        logging.warning("Sparse fine VZA bins detected:\n%s", sparse.sort("week", "vza_class"))
    logging.info("Loaded %d fine-bin angle observations", long.height)
    log_phase("load_fine_vza_from_plot_parquets", started)
    return long


def bootstrap_median(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan
    samples = rng.choice(values, size=(2000, values.size), replace=True)
    return tuple(np.quantile(np.median(samples, axis=1), [0.025, 0.975]))


def build_summaries(long: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    started = time.perf_counter()
    reference_class = long.sort("vza_midpoint")["vza_class"][0]
    logging.info("Using %s as the matched angular reference class", reference_class)
    summary = long.group_by("week", "band", "band_name", "vza_class", "vza_midpoint").agg(
        pl.len().alias("observations"),
        pl.col("plot_id").n_unique().alias("plots"),
        pl.col("reflectance").mean().alias("mean_reflectance"),
        pl.col("reflectance").median().alias("median_reflectance"),
        pl.col("reflectance").std().alias("sd_reflectance"),
        pl.col("reflectance").quantile(0.05).alias("q05"),
        pl.col("reflectance").quantile(0.25).alias("q25"),
        pl.col("reflectance").quantile(0.75).alias("q75"),
        pl.col("reflectance").quantile(0.95).alias("q95"),
    ).sort("band", "week", "vza_midpoint")

    nadir = long.filter(pl.col("vza_class") == reference_class).select(
        "plot_id", "week", "band", pl.col("reflectance").alias("nadir_reflectance")
    )
    matched = (
        long.join(nadir, on=["plot_id", "week", "band"], how="inner")
        .filter(pl.col("vza_class") != reference_class)
        .with_columns(
            (pl.col("reflectance") - pl.col("nadir_reflectance")).alias("absolute_contrast"),
            ((pl.col("reflectance") - pl.col("nadir_reflectance")) / pl.col("nadir_reflectance")).alias(
                "relative_contrast"
            ),
        )
    )

    rng = np.random.default_rng(SEED)
    rows = []
    for keys, group in matched.group_by("week", "band", "band_name", "vza_class", "vza_midpoint"):
        values = group["absolute_contrast"].to_numpy()
        ci_low, ci_high = bootstrap_median(values, rng)
        rows.append(
            {
                "week": keys[0],
                "band": keys[1],
                "band_name": keys[2],
                "vza_class": keys[3],
                "vza_midpoint": keys[4],
                "matched_plots": group["plot_id"].n_unique(),
                "median_absolute_contrast": float(np.median(values)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "median_relative_contrast": float(np.median(group["relative_contrast"].to_numpy())),
            }
        )
    contrasts = pl.DataFrame(rows).sort("band", "week", "vza_midpoint")
    log_phase("summaries_and_bootstrap", started)
    return summary, matched, contrasts


def fit_models(long: pl.DataFrame) -> pl.DataFrame:
    started = time.perf_counter()
    rows = []
    for band, band_name in BANDS.items():
        frame = long.filter(pl.col("band") == band).to_pandas()
        angle_order = (
            long.select("vza_class", "vza_midpoint")
            .unique()
            .sort("vza_midpoint")["vza_class"]
            .to_list()
        )
        frame["vza_class"] = pd.Categorical(frame["vza_class"], categories=angle_order, ordered=True)
        fit_started = time.perf_counter()
        formula = "reflectance ~ C(vza_class) * C(week) + C(cult) + C(trt)"
        try:
            model = smf.ols(formula, frame).fit(cov_type="cluster", cov_kwds={"groups": frame["plot_id"]})
            conf = model.conf_int()
            for term, estimate in model.params.items():
                rows.append(
                    {
                        "band": band,
                        "band_name": band_name,
                        "term": term,
                        "estimate": float(estimate),
                        "std_error": float(model.bse[term]),
                        "ci_low": float(conf.loc[term, 0]),
                        "ci_high": float(conf.loc[term, 1]),
                        "p_value": float(model.pvalues[term]),
                        "n_observations": int(model.nobs),
                        "r_squared": float(model.rsquared),
                        "error": None,
                    }
                )
        except Exception as exc:
            logging.exception("Model failed for %s", band_name)
            rows.append(
                {
                    "band": band,
                    "band_name": band_name,
                    "term": None,
                    "estimate": None,
                    "std_error": None,
                    "ci_low": None,
                    "ci_high": None,
                    "p_value": None,
                    "n_observations": frame.shape[0],
                    "r_squared": None,
                    "error": str(exc),
                }
            )
        logging.info("[ML] fit %s clustered OLS: %.3fs", band_name, time.perf_counter() - fit_started)
    log_phase("clustered_models", started)
    return pl.DataFrame(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E3E6E8", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=8)


def save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def week_color(week: int) -> str:
    """Return a stable color for planned and future recovered weeks."""
    if week in WEEK_COLORS:
        return WEEK_COLORS[week]
    cmap = plt.get_cmap("viridis")
    return matplotlib.colors.to_hex(cmap((week % 12) / 11))


def plot_angular_fingerprints(summary: pl.DataFrame, output_dir: Path) -> None:
    started = time.perf_counter()
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.5), constrained_layout=True)
    for index, (band, band_name) in enumerate(BANDS.items()):
        axis = axes.flat[index]
        for week in sorted(summary["week"].unique().to_list()):
            data = summary.filter((pl.col("band") == band) & (pl.col("week") == week)).sort("vza_midpoint")
            x = data["vza_midpoint"].to_numpy()
            color = week_color(week)
            axis.plot(
                x,
                data["median_reflectance"].to_numpy(),
                marker="o",
                markersize=3.5,
                linewidth=1.7,
                color=color,
                label=f"Week {week}",
            )
            axis.fill_between(x, data["q25"].to_numpy(), data["q75"].to_numpy(), color=color, alpha=0.13)
        axis.set_title(band_name, loc="left", fontsize=10, fontweight="bold")
        axis.set_xlabel("View zenith angle (degrees)", fontsize=9)
        axis.set_ylabel("Reflectance", fontsize=9)
        axis.text(-0.12, 1.04, f"({chr(97 + index)})", transform=axis.transAxes, fontweight="bold", fontsize=9)
        style_axis(axis)
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc="center", frameon=False, title="2024 acquisition")
    save_figure(figure, output_dir / "figures/main/angular_reflectance_curves_preliminary")
    log_phase("angular_fingerprint_figure", started)


def plot_matched_contrasts(contrasts: pl.DataFrame, output_dir: Path) -> None:
    started = time.perf_counter()
    weeks = sorted(contrasts["week"].unique().to_list())
    angle_order = contrasts.select("vza_class", "vza_midpoint").unique().sort("vza_midpoint")["vza_class"].to_list()
    height = max(5.9, len(BANDS) * len(angle_order) * 0.22)
    figure, axes = plt.subplots(1, len(weeks), figsize=(12.7, height), sharey=True, constrained_layout=True)
    labels = [f"{band_name}  {angle}" for band_name in BANDS.values() for angle in angle_order]
    y = np.arange(len(labels))
    for axis, week in zip(axes, weeks):
        values, lows, highs = [], [], []
        for band in BANDS:
            for angle in angle_order:
                row = contrasts.filter(
                    (pl.col("week") == week) & (pl.col("band") == band) & (pl.col("vza_class") == angle)
                )
                values.append(row["median_absolute_contrast"][0] if row.height else np.nan)
                lows.append(row["ci_low"][0] if row.height else np.nan)
                highs.append(row["ci_high"][0] if row.height else np.nan)
        values = np.asarray(values)
        axis.errorbar(
            values,
            y,
            xerr=np.vstack((values - np.asarray(lows), np.asarray(highs) - values)),
            fmt="o",
            color="#176B6B",
            ecolor="#76A9A9",
            markersize=3.5,
            capsize=2,
            linewidth=1,
        )
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_title(f"Week {week}", fontsize=10, fontweight="bold")
        axis.set_xlabel("Off-nadir minus nadir", fontsize=9)
        style_axis(axis)
    axes[0].set_yticks(y, labels, fontsize=7)
    axes[0].invert_yaxis()
    save_figure(figure, output_dir / "figures/main/matched_off_nadir_effects_preliminary")
    log_phase("matched_contrast_figure", started)


def plot_cultivar_curves(long: pl.DataFrame, output_dir: Path) -> None:
    started = time.perf_counter()
    selected_bands = list(BANDS)
    weeks = sorted(long["week"].unique().to_list())
    figure, axes = plt.subplots(len(selected_bands), len(weeks), figsize=(12, 10.5), sharex=True, constrained_layout=True)
    for row_index, band in enumerate(selected_bands):
        for column_index, week in enumerate(weeks):
            axis = axes[row_index, column_index]
            for cultivar in sorted(long["cult"].unique().to_list()):
                data = (
                    long.filter(
                        (pl.col("band") == band) & (pl.col("week") == week) & (pl.col("cult") == cultivar)
                    )
                    .group_by("vza_class", "vza_midpoint")
                    .agg(
                        pl.col("reflectance").mean().alias("mean"),
                        pl.col("reflectance").std().alias("sd"),
                        pl.len().alias("n"),
                    )
                    .with_columns((pl.col("sd") / pl.col("n").sqrt()).alias("se"))
                    .sort("vza_midpoint")
                )
                color, linestyle = CULTIVAR_STYLES[cultivar]
                x = data["vza_midpoint"].to_numpy()
                mean = data["mean"].to_numpy()
                se = data["se"].to_numpy()
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    label=cultivar.capitalize(),
                )
                axis.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.1)
            if row_index == 0:
                axis.set_title(f"Week {week}", fontsize=10, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(f"{BANDS[band]}\nreflectance", fontsize=9)
            if row_index == len(selected_bands) - 1:
                axis.set_xlabel("VZA (degrees)", fontsize=9)
            style_axis(axis)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    save_figure(figure, output_dir / "figures/main/angular_reflectance_by_cultivar_preliminary")
    log_phase("cultivar_figure", started)


def write_report(
    output_dir: Path,
    input_path: Path,
    long: pl.DataFrame,
    contrasts: pl.DataFrame,
    models: pl.DataFrame,
    log_path: Path,
) -> None:
    started = time.perf_counter()
    weeks = sorted(long["week"].unique().to_list())
    reference_class = long.sort("vza_midpoint")["vza_class"][0]
    pending_weeks = [week for week in [2, 4, 6, 7] if week not in weeks]
    pending_text = ", ".join(map(str, pending_weeks)) if pending_weeks else "none"
    strongest = (
        contrasts.with_columns(pl.col("median_absolute_contrast").abs().alias("magnitude"))
        .sort("magnitude", descending=True)
        .head(10)
    )
    significant = models.filter(pl.col("p_value").is_not_null() & (pl.col("p_value") < 0.05)).height
    lines = [
        "# Preliminary Result 1: Reflectance Across Viewing Angles",
        "",
        f"> This analysis uses validated 2024 weeks {', '.join(map(str, weeks))}. It will be rerun after pending weeks are extracted and validated: {pending_text}.",
        "",
        "## Dataset",
        "",
        f"- Plot-week records: **{long.select('plot_id', 'week').unique().height}**",
        f"- Unique plots: **{long['plot_id'].n_unique()}**",
        f"- Weeks: **{', '.join(map(str, weeks))}**",
        f"- Cultivars: **{', '.join(sorted(long['cult'].unique().to_list()))}**",
        "- Analytical unit: plot-level mean reflectance within a VZA class.",
        "",
        "## Largest Matched Angular Contrasts",
        "",
        "| Week | Band | VZA class | Median contrast | Bootstrap 95% CI | Matched plots |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in strongest.iter_rows(named=True):
        lines.append(
            f"| {row['week']} | {row['band_name']} | {row['vza_class']} | "
            f"{row['median_absolute_contrast']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{row['matched_plots']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The preliminary figures test whether canopy reflectance changes systematically with viewing angle and whether the shape of this angular response changes over the season. Matched contrasts compare each higher-angle observation with the {reference_class} reference value from the same plot and week. Cultivar-specific curves expose canopy-architecture differences that should be controlled before disease effects are tested.",
            "",
            "These results establish an angular signal but do not yet show that the signal is caused by disease or that it improves disease prediction. Disease severity measurements will be linked in a later result, after the complete 2024 week series has been recovered.",
            "",
            "## Model Summary",
            "",
            f"The band-specific clustered models contain **{significant}** coefficients with p < 0.05. Final reporting should emphasize effect sizes and confidence intervals after all recoverable weeks are included.",
            "",
            "## Reproducibility",
            "",
            f"- Input: `{input_path}`",
            f"- Log: `{log_path}`",
            f"- Random seed: `{SEED}`",
            "- Confidence intervals for matched contrasts: 2,000 bootstrap resamples of the median.",
            "- Model: `reflectance ~ VZA class * week + cultivar + treatment`, with cluster-robust standard errors by plot.",
            f"- Pending update: weeks {pending_text}.",
            "",
        ]
    )
    report_path = output_dir / "reports/reflectance_distributions_preliminary_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    log_phase("report", started)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(args.output_dir)
    total_started = time.perf_counter()
    logging.info("Starting preliminary Result 1 analysis")

    long = load_fine_vza_from_plot_parquets() if args.fine_vza_bins else load_long(args.input)
    summary, matched, contrasts = build_summaries(long)
    models = fit_models(long)

    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    long.write_parquet(results_dir / "plot_week_angle_features_preliminary.parquet", compression="zstd")
    matched.write_parquet(results_dir / "matched_plot_contrasts_preliminary.parquet", compression="zstd")
    summary.write_csv(results_dir / "reflectance_by_vza_summary_preliminary.csv")
    contrasts.write_csv(results_dir / "matched_angular_contrasts_preliminary.csv")
    models.write_csv(results_dir / "angle_week_cultivar_models_preliminary.csv")

    plot_angular_fingerprints(summary, args.output_dir)
    plot_matched_contrasts(contrasts, args.output_dir)
    plot_cultivar_curves(long, args.output_dir)
    write_report(args.output_dir, args.input, long, contrasts, models, log_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
