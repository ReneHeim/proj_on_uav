#!/usr/bin/env python3
"""Create Result 1 tables and figures for recovered ONCERCO seasons."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
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
from matplotlib.transforms import blended_transform_factory

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
FINE_VZA_MIN = 10
FINE_VZA_MAX = 55
YEAR_PLOT_DIRS = {
    2024: {
        0: Path(
            "/run/media/davidem/Heim/2024/20240603_week0/metashape/20241205_products_uav_data/output/extract/polygon_df"
        ),
        2: Path("/run/media/davidem/Heim/2024/recovered_weeks/week2/output/plots"),
        3: Path(
            "/run/media/davidem/Heim/2024/20240624_week3/metashape/20241206_week3_products_uav_data/output/plots"
        ),
        4: Path("/run/media/davidem/Heim/2024/recovered_weeks/week4/output/plots"),
        5: Path(
            "/run/media/davidem/Heim/2024/20240715_week5/metashape/20241207_week5_products_uav_data/output/plots"
        ),
        6: Path("/run/media/davidem/Heim/2024/recovered_weeks/week6/output/plots"),
        7: Path("/run/media/davidem/Heim/2024/recovered_weeks/week7/output/plots"),
        8: Path(
            "/run/media/davidem/Heim/2024/20240826_week8/metashape/20241029_products_uav_data/output/extract/polygon_df"
        ),
    },
    2025: {
        0: Path("/run/media/davidem/Heim/2025/recovered_weeks/week0/output/plots"),
        3: Path("/run/media/davidem/Heim/2025/recovered_weeks/week3/output/plots"),
        5: Path("/run/media/davidem/Heim/2025/recovered_weeks/week5/output/plots"),
        7: Path("/run/media/davidem/Heim/2025/recovered_weeks/week7/output/plots"),
    },
}
POLYGON_PATHS = {
    2024: Path("/run/media/davidem/Heim/2024_oncerco_plot_polygons.gpkg"),
    2025: Path("/run/media/davidem/Heim/2025_oncerco_plot_polygons.gpkg"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=sorted(YEAR_PLOT_DIRS), default=2024)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fine-vza-bins",
        action="store_true",
        help="Aggregate from plot-level parquets into 5-degree VZA bins instead of using broad cached bins.",
    )
    parser.add_argument(
        "--ground-filter",
        action="store_true",
        help="Exclude likely ground/background pixels before aggregation using OSAVI and optional ExcessGreen thresholds.",
    )
    parser.add_argument(
        "--osavi-threshold",
        type=float,
        default=0.2,
        help="Vegetation threshold used when --ground-filter is set; keeps rows with OSAVI above this value.",
    )
    parser.add_argument(
        "--excess-green-threshold",
        type=float,
        default=None,
        help="Optional second threshold when --ground-filter is set; keeps rows with ExcessGreen above this value.",
    )
    return parser.parse_args()


def configure_logging(output_dir: Path, year: int) -> Path:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"result_01_{year}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return path


def log_phase(name: str, started: float) -> None:
    logging.info("[PHASE] %s: %.2fs", name, time.perf_counter() - started)


def season_output_dir(base_output_dir: Path, year: int, ground_filter: bool = False) -> Path:
    filter_name = "ground_filtered" if ground_filter else "unfiltered"
    return base_output_dir / str(year) / filter_name


def load_long(input_path: Path, year: int) -> pl.DataFrame:
    started = time.perf_counter()
    read_started = time.perf_counter()
    wide = pl.read_parquet(input_path).filter(pl.col("year") == year)
    read_seconds = time.perf_counter() - read_started
    logging.info(
        "[IO] parquet reads: n=1 min=%.3fs median=%.3fs max=%.3fs mean=%.3fs",
        read_seconds,
        read_seconds,
        read_seconds,
        read_seconds,
    )
    value_columns = [
        column for column in wide.columns if re.fullmatch(r"band[1-5]_vza\d+_\d+", column)
    ]
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
            pl.col("vza_class")
            .replace_strict(ANGLE_MIDPOINTS)
            .cast(pl.Float64)
            .alias("vza_midpoint"),
        )
    )
    logging.info("Loaded %d finite angle observations", long.height)
    log_phase("load_and_reshape", started)
    return long


def load_polygon_meta(year: int) -> dict[str, dict[str, object]]:
    import geopandas as gpd

    if year == 2024:
        gdf = gpd.read_file(POLYGON_PATHS[year])
        gdf["disease_label"] = gdf["ino"].astype(int)
        gdf["plot_id"] = "plot_" + (gdf["ifz_id"] - 90001).astype(str)
        columns = ["plot_id", "cult", "trt", "disease_label"]
    else:
        gdf = gpd.read_file(POLYGON_PATHS[year], layer="20250331_plots")
        gdf = gdf.reset_index(drop=True)
        gdf["plot_id"] = "plot_" + gdf.index.astype(str)
        gdf["cult"] = gdf["cultivar"]
        gdf["disease_label"] = None
        columns = ["plot_id", "cult", "trt", "disease_label"]
    return {row["plot_id"]: row for row in gdf[columns].to_dict("records")}


def assign_fine_vza_bins(frame: pl.DataFrame) -> pl.DataFrame:
    vza_low = (((pl.col("vza") - FINE_VZA_MIN) / 5).floor() * 5 + FINE_VZA_MIN).cast(pl.Int64)
    return (
        frame.filter((pl.col("vza") >= FINE_VZA_MIN) & (pl.col("vza") < FINE_VZA_MAX))
        .with_columns(vza_low.alias("vza_low"))
        .with_columns(
            (
                pl.col("vza_low").cast(pl.Utf8)
                + pl.lit("-")
                + (pl.col("vza_low") + 5).cast(pl.Utf8)
            ).alias("vza_class"),
            (pl.col("vza_low") + 2.5).cast(pl.Float64).alias("vza_midpoint"),
        )
        .drop("vza_low")
    )


def ground_filter_expressions(
    osavi_threshold: float,
    excess_green_threshold: float | None,
) -> pl.Expr:
    mask = pl.col("OSAVI").is_finite() & (pl.col("OSAVI") > osavi_threshold)
    if excess_green_threshold is not None:
        mask = (
            mask
            & pl.col("ExcessGreen").is_finite()
            & (pl.col("ExcessGreen") > excess_green_threshold)
        )
    return mask


def ensure_indices(frame: pl.DataFrame) -> pl.DataFrame:
    missing = [column for column in ["OSAVI", "ExcessGreen"] if column not in frame.columns]
    if not missing:
        return frame
    expressions = []
    if "OSAVI" in missing:
        expressions.append(
            (
                1.16
                * (pl.col("band5") - pl.col("band3"))
                / (pl.col("band5") + pl.col("band3") + 0.16)
            ).alias("OSAVI")
        )
    if "ExcessGreen" in missing:
        expressions.append(
            (2 * pl.col("band2") - pl.col("band3") - pl.col("band1")).alias("ExcessGreen")
        )
    return frame.with_columns(expressions)


def load_fine_vza_from_plot_parquets(
    year: int,
    week_plot_dirs: dict[int, Path],
    ground_filter: bool = False,
    osavi_threshold: float = 0.1,
    excess_green_threshold: float | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    started = time.perf_counter()
    meta = load_polygon_meta(year)
    rows = []
    filter_rows = []
    read_seconds = []
    band_columns = list(BANDS)
    for week, directory in week_plot_dirs.items():
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
            sampled_rows = frame.height
            mask = (
                pl.col("vza").is_not_nan()
                & (pl.col("vza") >= FINE_VZA_MIN)
                & (pl.col("vza") < FINE_VZA_MAX)
            )
            for band in band_columns:
                mask = mask & pl.col(band).is_not_nan() & (pl.col(band) > 0)
            frame = frame.filter(mask)
            basic_rows = frame.height
            if ground_filter:
                frame = ensure_indices(frame).filter(
                    ground_filter_expressions(osavi_threshold, excess_green_threshold)
                )
            filter_rows.append(
                {
                    "year": year,
                    "week": week,
                    "plot_id": plot_id,
                    "sampled_rows": sampled_rows,
                    "rows_after_basic_quality": basic_rows,
                    "rows_after_ground_filter": frame.height,
                    "rows_removed_by_ground_filter": basic_rows - frame.height,
                    "ground_filter_enabled": ground_filter,
                    "osavi_threshold": osavi_threshold if ground_filter else None,
                    "excess_green_threshold": excess_green_threshold if ground_filter else None,
                }
            )
            frame = assign_fine_vza_bins(frame)
            if frame.is_empty():
                continue
            summary = frame.group_by("vza_class", "vza_midpoint").agg(
                [pl.col(band).mean().alias(band) for band in band_columns]
            )
            for record in summary.to_dicts():
                for band in band_columns:
                    rows.append(
                        {
                            "plot_id": plot_id,
                            "week": week,
                            "cult": meta[plot_id]["cult"],
                            "trt": meta[plot_id]["trt"],
                            "year": year,
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
    filter_stats = pl.DataFrame(filter_rows)
    if ground_filter and filter_stats.height:
        total_basic = filter_stats["rows_after_basic_quality"].sum()
        total_after = filter_stats["rows_after_ground_filter"].sum()
        logging.info(
            "[FILTER] ground filter OSAVI > %.3f%s: %d -> %d rows (removed %.2f%%)",
            osavi_threshold,
            (
                ""
                if excess_green_threshold is None
                else f", ExcessGreen > {excess_green_threshold:.3f}"
            ),
            total_basic,
            total_after,
            ((total_basic - total_after) / total_basic * 100) if total_basic else 0,
        )
    return long, filter_stats


def bootstrap_median(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan
    samples = rng.choice(values, size=(2000, values.size), replace=True)
    return tuple(np.quantile(np.median(samples, axis=1), [0.025, 0.975]))


def paired_cohens_d(values: np.ndarray) -> float:
    """Cohen's dz for paired differences against the reference VZA class."""
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    sd = np.std(values, ddof=1)
    if sd == 0:
        return np.nan
    return float(np.mean(values) / sd)


def build_summaries(long: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    started = time.perf_counter()
    reference_class = long.sort("vza_midpoint")["vza_class"][0]
    logging.info("Using %s as the matched angular reference class", reference_class)
    summary = (
        long.group_by("week", "band", "band_name", "vza_class", "vza_midpoint")
        .agg(
            pl.len().alias("observations"),
            pl.col("plot_id").n_unique().alias("plots"),
            pl.col("reflectance").mean().alias("mean_reflectance"),
            pl.col("reflectance").median().alias("median_reflectance"),
            pl.col("reflectance").std().alias("sd_reflectance"),
            pl.col("reflectance").quantile(0.05).alias("q05"),
            pl.col("reflectance").quantile(0.25).alias("q25"),
            pl.col("reflectance").quantile(0.75).alias("q75"),
            pl.col("reflectance").quantile(0.95).alias("q95"),
        )
        .sort("band", "week", "vza_midpoint")
    )

    nadir = long.filter(pl.col("vza_class") == reference_class).select(
        "plot_id", "week", "band", pl.col("reflectance").alias("nadir_reflectance")
    )
    matched = (
        long.join(nadir, on=["plot_id", "week", "band"], how="inner")
        .filter(pl.col("vza_class") != reference_class)
        .with_columns(
            (pl.col("reflectance") - pl.col("nadir_reflectance")).alias("absolute_contrast"),
            (
                (pl.col("reflectance") - pl.col("nadir_reflectance")) / pl.col("nadir_reflectance")
            ).alias("relative_contrast"),
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
                "mean_absolute_contrast": float(np.mean(values)),
                "median_absolute_contrast": float(np.median(values)),
                "sd_absolute_contrast": (
                    float(np.std(values, ddof=1)) if values.size > 1 else np.nan
                ),
                "cohens_dz": paired_cohens_d(values),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "median_relative_contrast": float(np.median(group["relative_contrast"].to_numpy())),
            }
        )
    contrasts = pl.DataFrame(rows).sort("band", "week", "vza_midpoint")
    log_phase("summaries_and_bootstrap", started)
    return summary, matched, contrasts


def build_temporal_changes(long: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    started = time.perf_counter()
    weeks = sorted(long["week"].unique().to_list())
    rows = []
    for earlier, later in zip(weeks[:-1], weeks[1:]):
        earlier_frame = long.filter(pl.col("week") == earlier).select(
            "plot_id",
            "band",
            "band_name",
            "cult",
            "trt",
            "vza_class",
            "vza_midpoint",
            pl.col("reflectance").alias("earlier_reflectance"),
        )
        later_frame = long.filter(pl.col("week") == later).select(
            "plot_id",
            "band",
            "vza_class",
            pl.col("reflectance").alias("later_reflectance"),
        )
        joined = earlier_frame.join(later_frame, on=["plot_id", "band", "vza_class"], how="inner")
        if joined.is_empty():
            continue
        rows.append(
            joined.with_columns(
                pl.lit(earlier).alias("week_start"),
                pl.lit(later).alias("week_end"),
                (pl.col("later_reflectance") - pl.col("earlier_reflectance")).alias(
                    "temporal_change"
                ),
                (
                    (pl.col("later_reflectance") - pl.col("earlier_reflectance"))
                    / pl.col("earlier_reflectance")
                ).alias("relative_temporal_change"),
            )
        )
    if not rows:
        empty = pl.DataFrame()
        log_phase("temporal_changes", started)
        return empty, empty
    temporal = pl.concat(rows).filter(pl.col("temporal_change").is_finite())
    temporal_summary = (
        temporal.group_by(
            "week_start", "week_end", "band", "band_name", "vza_class", "vza_midpoint"
        )
        .agg(
            pl.len().alias("matched_plot_angle_observations"),
            pl.col("plot_id").n_unique().alias("matched_plots"),
            pl.col("temporal_change").mean().alias("mean_temporal_change"),
            pl.col("temporal_change").median().alias("median_temporal_change"),
            pl.col("temporal_change").std().alias("sd_temporal_change"),
            pl.col("temporal_change").quantile(0.25).alias("q25_temporal_change"),
            pl.col("temporal_change").quantile(0.75).alias("q75_temporal_change"),
            pl.col("relative_temporal_change").median().alias("median_relative_temporal_change"),
        )
        .sort("band", "week_start", "vza_midpoint")
    )
    log_phase("temporal_changes", started)
    return temporal, temporal_summary


def build_angular_contrast_changes(matched: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    started = time.perf_counter()
    weeks = sorted(matched["week"].unique().to_list())
    rows = []
    for earlier, later in zip(weeks[:-1], weeks[1:]):
        earlier_frame = matched.filter(pl.col("week") == earlier).select(
            "plot_id",
            "band",
            "band_name",
            "cult",
            "trt",
            "vza_class",
            "vza_midpoint",
            pl.col("absolute_contrast").alias("earlier_absolute_contrast"),
            pl.col("relative_contrast").alias("earlier_relative_contrast"),
        )
        later_frame = matched.filter(pl.col("week") == later).select(
            "plot_id",
            "band",
            "vza_class",
            pl.col("absolute_contrast").alias("later_absolute_contrast"),
            pl.col("relative_contrast").alias("later_relative_contrast"),
        )
        joined = earlier_frame.join(later_frame, on=["plot_id", "band", "vza_class"], how="inner")
        if joined.is_empty():
            continue
        rows.append(
            joined.with_columns(
                pl.lit(earlier).alias("week_start"),
                pl.lit(later).alias("week_end"),
                (pl.col("later_absolute_contrast") - pl.col("earlier_absolute_contrast")).alias(
                    "angular_contrast_change"
                ),
                (pl.col("later_relative_contrast") - pl.col("earlier_relative_contrast")).alias(
                    "relative_angular_contrast_change"
                ),
            )
        )
    if not rows:
        empty = pl.DataFrame()
        log_phase("angular_contrast_changes", started)
        return empty, empty
    changes = pl.concat(rows).filter(pl.col("angular_contrast_change").is_finite())
    summary = (
        changes.group_by("week_start", "week_end", "band", "band_name", "vza_class", "vza_midpoint")
        .agg(
            pl.len().alias("matched_plot_angle_observations"),
            pl.col("plot_id").n_unique().alias("matched_plots"),
            pl.col("angular_contrast_change").mean().alias("mean_angular_contrast_change"),
            pl.col("angular_contrast_change").median().alias("median_angular_contrast_change"),
            pl.col("angular_contrast_change").std().alias("sd_angular_contrast_change"),
            pl.col("angular_contrast_change").quantile(0.25).alias("q25_angular_contrast_change"),
            pl.col("angular_contrast_change").quantile(0.75).alias("q75_angular_contrast_change"),
            pl.col("relative_angular_contrast_change")
            .median()
            .alias("median_relative_angular_contrast_change"),
        )
        .sort("band", "week_start", "vza_midpoint")
    )
    log_phase("angular_contrast_changes", started)
    return changes, summary


def build_cultivar_angular_comparison(long: pl.DataFrame, matched: pl.DataFrame) -> pl.DataFrame:
    started = time.perf_counter()
    reflectance = long.group_by(
        "week", "band", "band_name", "vza_class", "vza_midpoint", "cult"
    ).agg(
        pl.len().alias("observations"),
        pl.col("plot_id").n_unique().alias("plots"),
        pl.col("reflectance").mean().alias("mean_reflectance"),
        pl.col("reflectance").median().alias("median_reflectance"),
        pl.col("reflectance").quantile(0.25).alias("q25_reflectance"),
        pl.col("reflectance").quantile(0.75).alias("q75_reflectance"),
    )
    contrast = matched.group_by("week", "band", "vza_class", "cult").agg(
        pl.col("plot_id").n_unique().alias("matched_plots"),
        pl.col("absolute_contrast").median().alias("median_angular_contrast"),
        pl.col("relative_contrast").median().alias("median_relative_angular_contrast"),
    )
    joined = reflectance.join(contrast, on=["week", "band", "vza_class", "cult"], how="left")
    cultivars = sorted(joined["cult"].unique().to_list())
    if len(cultivars) == 2:
        pivot = joined.select(
            "week",
            "band",
            "vza_class",
            "cult",
            "median_reflectance",
            "median_angular_contrast",
        ).pivot(
            index=["week", "band", "vza_class"],
            on="cult",
            values=["median_reflectance", "median_angular_contrast"],
        )
        first, second = cultivars
        pivot = pivot.with_columns(
            (pl.col(f"median_reflectance_{second}") - pl.col(f"median_reflectance_{first}")).alias(
                f"median_reflectance_difference_{second}_minus_{first}"
            ),
            (
                pl.col(f"median_angular_contrast_{second}")
                - pl.col(f"median_angular_contrast_{first}")
            ).alias(f"median_angular_contrast_difference_{second}_minus_{first}"),
        )
        joined = joined.join(
            pivot.select(
                "week",
                "band",
                "vza_class",
                f"median_reflectance_difference_{second}_minus_{first}",
                f"median_angular_contrast_difference_{second}_minus_{first}",
            ),
            on=["week", "band", "vza_class"],
            how="left",
        )
    log_phase("cultivar_angular_comparison", started)
    return joined.sort("band", "week", "vza_midpoint", "cult")


def build_missingness_diagnostics(long: pl.DataFrame, year: int) -> pl.DataFrame:
    started = time.perf_counter()
    plot_meta = pl.DataFrame(list(load_polygon_meta(year).values())).select(
        "plot_id", "cult", "trt"
    )
    weeks = pl.DataFrame({"week": sorted(long["week"].unique().to_list())})
    angle_classes = long.select("vza_class", "vza_midpoint").unique().sort("vza_midpoint")
    expected = weeks.join(plot_meta, how="cross").join(angle_classes, how="cross")
    present = (
        long.select("week", "plot_id", "vza_class")
        .unique()
        .with_columns(pl.lit(1).alias("present"))
    )
    diagnostic = (
        expected.join(present, on=["week", "plot_id", "vza_class"], how="left")
        .with_columns(pl.col("present").fill_null(0).cast(pl.Int8))
        .group_by("week", "vza_class", "vza_midpoint", "cult", "trt")
        .agg(
            pl.len().alias("expected_plots"),
            pl.col("present").sum().alias("present_plots"),
            (pl.len() - pl.col("present").sum()).alias("missing_plots"),
        )
        .with_columns(
            (pl.col("missing_plots") / pl.col("expected_plots")).alias("missing_fraction")
        )
        .sort("week", "vza_midpoint", "cult", "trt")
    )
    log_phase("missingness_diagnostics", started)
    return diagnostic


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
        frame["vza_class"] = pd.Categorical(
            frame["vza_class"], categories=angle_order, ordered=True
        )
        fit_started = time.perf_counter()
        formula = "reflectance ~ C(vza_class) * C(week) + C(cult) + C(trt)"
        try:
            model = smf.ols(formula, frame).fit(
                cov_type="cluster", cov_kwds={"groups": frame["plot_id"]}
            )
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
        logging.info(
            "[ML] fit %s clustered OLS: %.3fs", band_name, time.perf_counter() - fit_started
        )
    log_phase("clustered_models", started)
    return pl.DataFrame(rows)


def fit_cultivar_angle_models(long: pl.DataFrame) -> pl.DataFrame:
    started = time.perf_counter()
    rows = []
    angle_order = (
        long.select("vza_class", "vza_midpoint")
        .unique()
        .sort("vza_midpoint")["vza_class"]
        .to_list()
    )
    formulas = [
        ("full_cultivar_angle_week", "reflectance ~ C(cult) * C(vza_class) * C(week) + C(trt)"),
        (
            "reduced_cultivar_interactions",
            "reflectance ~ C(cult) * C(vza_class) + C(cult) * C(week) + C(vza_class) * C(week) + C(trt)",
        ),
    ]
    for band, band_name in BANDS.items():
        frame = long.filter(pl.col("band") == band).to_pandas()
        frame["vza_class"] = pd.Categorical(
            frame["vza_class"], categories=angle_order, ordered=True
        )
        fitted = False
        for model_type, formula in formulas:
            fit_started = time.perf_counter()
            try:
                model = smf.ols(formula, frame).fit(
                    cov_type="cluster", cov_kwds={"groups": frame["plot_id"]}
                )
                conf = model.conf_int()
                for term, estimate in model.params.items():
                    rows.append(
                        {
                            "band": band,
                            "band_name": band_name,
                            "model_type": model_type,
                            "term": term,
                            "estimate": float(estimate),
                            "std_error": float(model.bse[term]),
                            "ci_low": float(conf.loc[term, 0]),
                            "ci_high": float(conf.loc[term, 1]),
                            "p_value": float(model.pvalues[term]),
                            "n_observations": int(model.nobs),
                            "r_squared": float(model.rsquared),
                            "formula": formula,
                            "error": None,
                        }
                    )
                fitted = True
                logging.info(
                    "[ML] fit %s %s clustered OLS: %.3fs",
                    band_name,
                    model_type,
                    time.perf_counter() - fit_started,
                )
                break
            except Exception as exc:
                logging.exception("Cultivar model %s failed for %s", model_type, band_name)
                rows.append(
                    {
                        "band": band,
                        "band_name": band_name,
                        "model_type": model_type,
                        "term": None,
                        "estimate": None,
                        "std_error": None,
                        "ci_low": None,
                        "ci_high": None,
                        "p_value": None,
                        "n_observations": frame.shape[0],
                        "r_squared": None,
                        "formula": formula,
                        "error": str(exc),
                    }
                )
        if not fitted:
            logging.error("No cultivar interaction model could be fit for %s", band_name)
    log_phase("cultivar_angle_models", started)
    return pl.DataFrame(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E3E6E8", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=8)


def save_figure(figure: plt.Figure, stem: Path, aliases: list[Path] | None = None) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    for alias in aliases or []:
        alias.parent.mkdir(parents=True, exist_ok=True)
        for suffix in [".pdf", ".png", ".svg", ".tiff"]:
            shutil.copyfile(stem.with_suffix(suffix), alias.with_suffix(suffix))
    plt.close(figure)


def week_color(week: int) -> str:
    """Return a stable color for planned and future recovered weeks."""
    if week in WEEK_COLORS:
        return WEEK_COLORS[week]
    cmap = plt.get_cmap("viridis")
    return matplotlib.colors.to_hex(cmap((week % 12) / 11))


def angle_color_map(angle_order: list[str]) -> dict[str, str]:
    colors = plt.get_cmap("GnBu")(np.linspace(0.88, 0.38, len(angle_order)))
    return {angle: matplotlib.colors.to_hex(color) for angle, color in zip(angle_order, colors)}


def plot_reflectance_distributions_by_vza(long: pl.DataFrame, output_dir: Path) -> None:
    started = time.perf_counter()
    angle_order = (
        long.select("vza_class", "vza_midpoint")
        .unique()
        .sort("vza_midpoint")["vza_class"]
        .to_list()
    )
    colors = angle_color_map(angle_order)
    figure, axes = plt.subplots(
        1, len(BANDS), figsize=(16, 4.8), sharex=True, constrained_layout=True
    )
    for index, (axis, (band, band_name)) in enumerate(zip(axes, BANDS.items())):
        data = long.filter(pl.col("band") == band)
        values = [
            data.filter(pl.col("vza_class") == angle)["reflectance"].to_numpy()
            for angle in angle_order
        ]
        box = axis.boxplot(
            values,
            positions=np.arange(len(angle_order)),
            widths=0.58,
            patch_artist=True,
            showfliers=False,
        )
        for patch, angle in zip(box["boxes"], angle_order):
            patch.set(facecolor=colors[angle], edgecolor="#34495E", linewidth=0.8, alpha=0.72)
        for item in box["medians"]:
            item.set(color="#111111", linewidth=1.2)
        rng = np.random.default_rng(SEED + index)
        for pos, angle, angle_values in zip(np.arange(len(angle_order)), angle_order, values):
            if angle_values.size:
                jitter = rng.normal(0, 0.045, size=angle_values.size)
                axis.scatter(
                    np.full(angle_values.size, pos) + jitter,
                    angle_values,
                    color=colors[angle],
                    edgecolor="#263238",
                    linewidth=0.2,
                    s=9,
                    alpha=0.45,
                )
        axis.set_title(band_name, loc="left", fontsize=10, fontweight="bold")
        axis.set_xticks(np.arange(len(angle_order)), angle_order, rotation=45, ha="right")
        axis.set_ylabel("Plot-level reflectance", fontsize=9)
        axis.text(
            -0.16,
            1.04,
            f"({chr(97 + index)})",
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9,
        )
        style_axis(axis)
    figure.supxlabel("View zenith angle class (degrees)", fontsize=10)
    save_figure(figure, output_dir / "figures/main/reflectance_distributions_by_vza")
    log_phase("reflectance_distribution_vza_figure", started)


def plot_seasonal_distribution_atlas(long: pl.DataFrame, output_dir: Path, year: int) -> None:
    started = time.perf_counter()
    weeks = sorted(long["week"].unique().to_list())
    angle_order = (
        long.select("vza_class", "vza_midpoint")
        .unique()
        .sort("vza_midpoint")["vza_class"]
        .to_list()
    )
    colors = angle_color_map(angle_order)
    figure, axes = plt.subplots(
        len(BANDS),
        len(weeks),
        figsize=(2.45 * len(weeks), 11.8),
        sharex=True,
        constrained_layout=True,
    )
    for row_index, (band, band_name) in enumerate(BANDS.items()):
        band_data = long.filter(pl.col("band") == band)
        y_min = float(band_data["reflectance"].quantile(0.02))
        y_max = float(band_data["reflectance"].quantile(0.98))
        pad = max((y_max - y_min) * 0.08, 0.002)
        for col_index, week in enumerate(weeks):
            axis = axes[row_index, col_index]
            data = band_data.filter(pl.col("week") == week)
            values = [
                data.filter(pl.col("vza_class") == angle)["reflectance"].to_numpy()
                for angle in angle_order
            ]
            box = axis.boxplot(
                values,
                positions=np.arange(len(angle_order)),
                widths=0.55,
                patch_artist=True,
                showfliers=False,
            )
            for patch, angle in zip(box["boxes"], angle_order):
                patch.set(facecolor=colors[angle], edgecolor="#34495E", linewidth=0.65, alpha=0.7)
            for median in box["medians"]:
                median.set(color="#111111", linewidth=1.0)
            axis.set_ylim(y_min - pad, y_max + pad)
            if row_index == 0:
                axis.set_title(f"Week {week}", fontsize=9.5, fontweight="bold")
            if col_index == 0:
                axis.set_ylabel(f"{band_name}\nreflectance", fontsize=8.5)
            if row_index == len(BANDS) - 1:
                axis.set_xticks(
                    np.arange(len(angle_order)), angle_order, rotation=45, ha="right", fontsize=7
                )
            else:
                axis.set_xticks(np.arange(len(angle_order)), [])
            style_axis(axis)
    save_figure(figure, output_dir / f"figures/main/reflectance_distributions_by_week_{year}")
    log_phase("seasonal_distribution_atlas_figure", started)


def plot_angular_fingerprints(summary: pl.DataFrame, output_dir: Path, year: int) -> None:
    started = time.perf_counter()
    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.4), constrained_layout=True)
    for index, (band, band_name) in enumerate(BANDS.items()):
        axis = axes.flat[index]
        for week in sorted(summary["week"].unique().to_list()):
            data = summary.filter((pl.col("band") == band) & (pl.col("week") == week)).sort(
                "vza_midpoint"
            )
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
            axis.fill_between(
                x, data["q25"].to_numpy(), data["q75"].to_numpy(), color=color, alpha=0.13
            )
        axis.set_title(band_name, loc="left", fontsize=10, fontweight="bold")
        axis.set_xlabel("View zenith angle (degrees)", fontsize=9)
        axis.set_ylabel("Reflectance", fontsize=9)
        axis.text(
            -0.12,
            1.04,
            f"({chr(97 + index)})",
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9,
        )
        style_axis(axis)
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc="center", frameon=False, title=f"{year} acquisition")
    save_figure(
        figure,
        output_dir / "figures/main/angular_reflectance_curves_by_week",
        aliases=[
            output_dir / "figures/main/angular_reflectance_curves",
            output_dir / "figures/main/angular_reflectance_curves_preliminary",
        ],
    )
    log_phase("angular_fingerprint_figure", started)


def plot_matched_contrasts(contrasts: pl.DataFrame, output_dir: Path, reference_class: str) -> None:
    started = time.perf_counter()
    weeks = sorted(contrasts["week"].unique().to_list())
    angle_order = (
        contrasts.select("vza_class", "vza_midpoint")
        .unique()
        .sort("vza_midpoint")["vza_class"]
        .to_list()
    )
    height = max(8.4, len(BANDS) * len(angle_order) * 0.32)
    figure, axes = plt.subplots(
        1,
        len(weeks),
        figsize=(3.15 * len(weeks) + 2.2, height),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(weeks) == 1:
        axes = [axes]
    labels = [angle for _ in BANDS for angle in angle_order]
    y = np.arange(len(labels))
    lower = float(contrasts["ci_low"].min())
    upper = float(contrasts["ci_high"].max())
    span = max(upper - lower, 0.01)
    x_min = min(lower - 0.08 * span, -0.005)
    x_max = max(upper + 0.08 * span, 0.005)
    group_size = len(angle_order)
    band_centers = [index * group_size + (group_size - 1) / 2 for index in range(len(BANDS))]
    separators = [index * group_size - 0.5 for index in range(1, len(BANDS))]
    for axis, week in zip(axes, weeks):
        values, lows, highs = [], [], []
        for band in BANDS:
            for angle in angle_order:
                row = contrasts.filter(
                    (pl.col("week") == week)
                    & (pl.col("band") == band)
                    & (pl.col("vza_class") == angle)
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
            markersize=4.2,
            capsize=2,
            elinewidth=0.9,
            linewidth=0.9,
        )
        axis.axvline(0, color="#222222", linewidth=1.25, zorder=0)
        for separator in separators:
            axis.axhline(separator, color="#C8CDD1", linewidth=0.8, zorder=0)
        axis.set_title(f"Week {week}", fontsize=11, fontweight="bold")
        axis.set_xlim(x_min, x_max)
        style_axis(axis)
        axis.grid(axis="x", color="#ECEFF1", linewidth=0.5)
    axes[0].set_yticks(y, labels, fontsize=8)
    transform = blended_transform_factory(axes[0].transAxes, axes[0].transData)
    for center, band_name in zip(band_centers, BANDS.values()):
        axes[0].text(
            -0.23,
            center,
            band_name,
            transform=transform,
            ha="right",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[0].invert_yaxis()
    figure.suptitle("Matched angular reflectance effect by week", fontsize=13, fontweight="bold")
    figure.supxlabel(f"Reflectance difference vs {reference_class} degree VZA", fontsize=11)
    save_figure(
        figure,
        output_dir / "figures/main/matched_off_nadir_effects",
        aliases=[output_dir / "figures/main/matched_off_nadir_effects_preliminary"],
    )
    log_phase("matched_contrast_figure", started)


def plot_seasonal_angular_contrast_change(contrast_summary: pl.DataFrame, output_dir: Path) -> None:
    started = time.perf_counter()
    if contrast_summary.is_empty():
        logging.warning(
            "Skipping seasonal contrast change figure because no contrast changes were built"
        )
        return
    angle_order = (
        contrast_summary.select("vza_class", "vza_midpoint")
        .unique()
        .sort("vza_midpoint")["vza_class"]
        .to_list()
    )
    colors = angle_color_map(angle_order)
    figure, axes = plt.subplots(2, 3, figsize=(14.4, 8.2), constrained_layout=True)
    for index, (band, band_name) in enumerate(BANDS.items()):
        axis = axes.flat[index]
        data = contrast_summary.filter(pl.col("band") == band).sort("week_end", "vza_midpoint")
        for angle in angle_order:
            angle_data = data.filter(pl.col("vza_class") == angle)
            if angle_data.is_empty():
                continue
            x = angle_data["week_end"].to_numpy()
            y = angle_data["median_angular_contrast_change"].to_numpy()
            q25 = angle_data["q25_angular_contrast_change"].to_numpy()
            q75 = angle_data["q75_angular_contrast_change"].to_numpy()
            axis.plot(
                x, y, marker="o", markersize=3.5, linewidth=1.45, color=colors[angle], label=angle
            )
            axis.fill_between(x, q25, q75, color=colors[angle], alpha=0.12)
        axis.axhline(0, color="#222222", linewidth=0.9)
        axis.set_title(band_name, loc="left", fontsize=10, fontweight="bold")
        axis.set_xlabel("Later week in consecutive pair", fontsize=9)
        axis.set_ylabel("Change in off-nadir contrast", fontsize=9)
        axis.text(
            -0.12,
            1.04,
            f"({chr(97 + index)})",
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=9,
        )
        style_axis(axis)
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc="center", frameon=False, title="VZA class")
    save_figure(figure, output_dir / "figures/main/seasonal_change_in_angular_contrast")
    log_phase("seasonal_contrast_change_figure", started)


def plot_cultivar_curves(long: pl.DataFrame, output_dir: Path, year: int) -> None:
    started = time.perf_counter()
    selected_bands = list(BANDS)
    weeks = sorted(long["week"].unique().to_list())
    figure, axes = plt.subplots(
        len(selected_bands),
        len(weeks),
        figsize=(3.15 * len(weeks), 13.5),
        sharex=True,
        constrained_layout=True,
    )
    for row_index, band in enumerate(selected_bands):
        for column_index, week in enumerate(weeks):
            axis = axes[row_index, column_index]
            for cultivar in sorted(long["cult"].unique().to_list()):
                data = (
                    long.filter(
                        (pl.col("band") == band)
                        & (pl.col("week") == week)
                        & (pl.col("cult") == cultivar)
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
    figure.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03)
    )
    save_figure(
        figure,
        output_dir / f"figures/main/angular_reflectance_curves_by_cultivar_{year}",
        aliases=[
            output_dir / "figures/main/angular_reflectance_by_cultivar_preliminary",
            output_dir / f"figures/supplementary/reflectance_distributions_by_cultivar_week_{year}",
        ],
    )
    log_phase("cultivar_figure", started)


def write_report(
    output_dir: Path,
    input_path: Path,
    year: int,
    week_plot_dirs: dict[int, Path],
    ground_filter: bool,
    osavi_threshold: float,
    excess_green_threshold: float | None,
    filter_stats: pl.DataFrame,
    long: pl.DataFrame,
    contrasts: pl.DataFrame,
    temporal_summary: pl.DataFrame,
    contrast_change_summary: pl.DataFrame,
    cultivar_comparison: pl.DataFrame,
    missingness: pl.DataFrame,
    models: pl.DataFrame,
    cultivar_models: pl.DataFrame,
    log_path: Path,
    fine_vza_bins: bool = False,
) -> None:
    started = time.perf_counter()
    weeks = sorted(long["week"].unique().to_list())
    reference_class = long.sort("vza_midpoint")["vza_class"][0]
    pending_weeks = [week for week in sorted(week_plot_dirs) if week not in weeks]
    pending_text = ", ".join(map(str, pending_weeks)) if pending_weeks else "none"
    strongest = (
        contrasts.filter(pl.col("matched_plots") >= 10)
        .with_columns(pl.col("median_absolute_contrast").abs().alias("magnitude"))
        .sort("magnitude", descending=True)
        .head(10)
    )
    sparse = contrasts.filter(pl.col("matched_plots") < 10).sort(
        "matched_plots", "week", "vza_midpoint"
    )
    significant = models.filter(pl.col("p_value").is_not_null() & (pl.col("p_value") < 0.05)).height
    cultivar_significant = cultivar_models.filter(
        pl.col("p_value").is_not_null() & (pl.col("p_value") < 0.05)
    ).height
    temporal_headline = (
        temporal_summary.with_columns(pl.col("median_temporal_change").abs().alias("magnitude"))
        .sort("magnitude", descending=True)
        .head(5)
        if not temporal_summary.is_empty()
        else temporal_summary
    )
    missing_headline = missingness.sort("missing_fraction", descending=True).head(5)
    filter_lines = []
    if ground_filter and filter_stats.height:
        total_basic = filter_stats["rows_after_basic_quality"].sum()
        total_after = filter_stats["rows_after_ground_filter"].sum()
        removed = total_basic - total_after
        removed_pct = (removed / total_basic * 100) if total_basic else 0
        rule = f"`OSAVI > {osavi_threshold}`"
        if excess_green_threshold is not None:
            rule += f" and `ExcessGreen > {excess_green_threshold}`"
        filter_lines = [
            "",
            "## Ground/Background Filter",
            "",
            f"- Rule: {rule}",
            f"- Rows after basic VZA/band quality filter: **{total_basic:,}**",
            f"- Rows retained after ground filter: **{total_after:,}**",
            f"- Rows removed by ground filter: **{removed:,}** ({removed_pct:.1f}%)",
            f"- Retention table: `{output_dir / f'results/ground_filter_retention_{year}.csv'}`",
        ]
    lines = [
        "# Result 1: Reflectance Across Viewing Angles",
        "",
        f"> This analysis uses validated {year} weeks {', '.join(map(str, weeks))}. Pending recoverable weeks: {pending_text}.",
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
        "| Week | Band | VZA class | Median contrast | Bootstrap 95% CI | Cohen's dz | Matched plots |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in strongest.iter_rows(named=True):
        lines.append(
            f"| {row['week']} | {row['band_name']} | {row['vza_class']} | "
            f"{row['median_absolute_contrast']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{row['cohens_dz']:.2f} | "
            f"{row['matched_plots']} |"
        )
    if sparse.height:
        sparse_descriptions = [
            f"Week {row['week']} {row['band_name']} {row['vza_class']} (n={row['matched_plots']})"
            for row in sparse.iter_rows(named=True)
        ]
        lines.extend(
            [
                "",
                "**Coverage caveat**: Headline contrasts require at least 10 matched plots. "
                f"Sparse estimates retained only in the full CSV: {', '.join(sparse_descriptions)}.",
            ]
        )
    lines.extend(
        [
            "",
            "## Largest Matched Temporal Changes",
            "",
            "| Week pair | Band | VZA class | Median change | Matched plots |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in temporal_headline.iter_rows(named=True):
        lines.append(
            f"| {row['week_start']} to {row['week_end']} | {row['band_name']} | {row['vza_class']} | "
            f"{row['median_temporal_change']:.4f} | {row['matched_plots']} |"
        )
    lines.extend(
        [
            "",
            "## Missing Plot Support Checks",
            "",
            "| Week | VZA class | Cultivar | Treatment | Missing plots | Missing fraction |",
            "|---:|---|---|---|---:|---:|",
        ]
    )
    for row in missing_headline.iter_rows(named=True):
        lines.append(
            f"| {row['week']} | {row['vza_class']} | {row['cult']} | {row['trt']} | "
            f"{row['missing_plots']} | {row['missing_fraction']:.3f} |"
        )
    lines.extend(filter_lines)
    if fine_vza_bins:
        input_lines = ["- Inputs: plot-level parquet directories:"] + [
            f"  - Week {week}: `{directory}`" for week, directory in sorted(week_plot_dirs.items())
        ]
    else:
        input_lines = [f"- Input: `{input_path}`"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"The preliminary figures test whether canopy reflectance changes systematically with viewing angle and whether the shape of this angular response changes over the season. Matched contrasts compare each higher-angle observation with the {reference_class} reference value from the same plot and week. Cultivar-specific curves expose canopy-architecture differences that should be controlled before disease effects are tested.",
            "",
            f"These {year} results establish an angular signal but do not yet show that the signal is caused by disease or that it improves disease prediction. Disease severity measurements should be linked in a later result before making disease claims.",
            "",
            "## Model Summary",
            "",
            f"The band-specific clustered models contain **{significant}** coefficients with p < 0.05. Final reporting should emphasize effect sizes and confidence intervals after all recoverable weeks are included.",
            f"The cultivar interaction model table contains **{cultivar_significant}** coefficients with p < 0.05. Cultivar terms are interpreted as canopy-architecture controls, not disease effects.",
            "",
            "## Outputs",
            "",
            f"- Reflectance summary: `{output_dir / 'results/reflectance_by_vza_summary.csv'}`",
            f"- Matched angular contrasts: `{output_dir / 'results/matched_angular_contrasts.csv'}`",
            f"- Temporal reflectance changes: `{output_dir / f'results/temporal_reflectance_changes_{year}.csv'}`",
            f"- Angular contrast changes: `{output_dir / f'results/angular_contrast_changes_{year}.csv'}`",
            f"- Cultivar comparison: `{output_dir / f'results/cultivar_angular_comparison_{year}.csv'}`",
            f"- Cultivar angle model: `{output_dir / f'results/cultivar_angle_model_{year}.csv'}`",
            f"- Missingness diagnostics: `{output_dir / f'results/missing_plot_support_by_week_angle_cultivar_{year}.csv'}`",
            f"- Main figures: `{output_dir / 'figures/main'}`",
            "",
            "## Reproducibility",
            "",
            *input_lines,
            f"- Log: `{log_path}`",
            f"- Random seed: `{SEED}`",
            "- Confidence intervals for matched contrasts: 2,000 bootstrap resamples of the median.",
            f"- Ground/background filter: {'enabled' if ground_filter else 'disabled'}.",
            f"- OSAVI threshold: {osavi_threshold if ground_filter else 'not applied'}.",
            f"- ExcessGreen threshold: {excess_green_threshold if ground_filter and excess_green_threshold is not None else 'not applied'}.",
            "- Model: `reflectance ~ VZA class * week + cultivar + treatment`, with cluster-robust standard errors by plot.",
            "- Cultivar model: full `cultivar * VZA class * week + treatment` model when stable, otherwise the reduced cultivar interaction model.",
            f"- VZA boundaries: {', '.join(sorted(long['vza_class'].unique().to_list(), key=lambda x: float(x.split('-')[0])))}.",
            f"- Cultivar rows in comparison table: {cultivar_comparison.height}.",
            f"- Angular contrast change rows: {contrast_change_summary.height}.",
            f"- Pending update: weeks {pending_text}.",
            "",
        ]
    )
    report_path = output_dir / f"reports/reflectance_distributions_summary_{year}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    shutil.copyfile(report_path, output_dir / "reports/reflectance_distributions_summary.md")
    shutil.copyfile(
        report_path, output_dir / "reports/reflectance_distributions_preliminary_summary.md"
    )
    log_phase("report", started)


def write_captions(output_dir: Path, long: pl.DataFrame, year: int) -> None:
    started = time.perf_counter()
    weeks = ", ".join(map(str, sorted(long["week"].unique().to_list())))
    angle_classes = ", ".join(
        sorted(long["vza_class"].unique().to_list(), key=lambda value: float(value.split("-")[0]))
    )
    captions = [
        "# Result 1 Figure Captions",
        "",
        "## Figure 1. Seasonal angular distribution atlas",
        "",
        f"Plot-level reflectance distributions for validated {year} weeks {weeks}. The analytical unit is the plot-week-angle median or mean reflectance, not individual pixels. VZA classes are {angle_classes}. Boxes summarize plot-level observations within each week, band, and VZA class; sparse classes should be interpreted with the support table.",
        "",
        "## Figure 2. Angular reflectance fingerprints",
        "",
        "Band-specific angular reflectance curves across VZA-bin midpoints. Lines show weekly medians and shaded ribbons show the interquartile range across plot-level observations. Lines are not interpolated beyond observed bins.",
        "",
        "## Figure 3. Matched off-nadir effect plot",
        "",
        "Matched off-nadir minus reference-VZA reflectance contrasts from the same plot and week. Points show median paired contrasts and intervals show bootstrap 95% confidence intervals over matched plots. The vertical reference line marks no difference from the reference VZA class.",
        "",
        "## Figure 4. Cultivar angular response",
        "",
        "Cultivar-specific angular reflectance curves by week and band. Lines show cultivar means across plot-level observations; ribbons show approximate 95% intervals from standard errors. Cultivar differences are interpreted as canopy and trial-structure diagnostics, not disease effects.",
        "",
        "## Figure 5. Seasonal change in angular contrast",
        "",
        "Changes in matched off-nadir contrast between consecutive validated weeks. Lines show median changes in angular contrast for each VZA class, with interquartile ribbons across matched plots.",
        "",
    ]
    path = output_dir / f"reports/figure_captions_{year}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(captions), encoding="utf-8")
    shutil.copyfile(path, output_dir / "reports/figure_captions.md")
    log_phase("captions", started)


def main() -> None:
    args = parse_args()
    output_dir = season_output_dir(args.output_dir, args.year, args.ground_filter)
    output_dir.mkdir(parents=True, exist_ok=True)
    week_plot_dirs = YEAR_PLOT_DIRS[args.year]
    log_path = configure_logging(output_dir, args.year)
    total_started = time.perf_counter()
    logging.info("Starting Result 1 analysis for %s", args.year)

    if args.fine_vza_bins:
        long, filter_stats = load_fine_vza_from_plot_parquets(
            args.year,
            week_plot_dirs,
            ground_filter=args.ground_filter,
            osavi_threshold=args.osavi_threshold,
            excess_green_threshold=args.excess_green_threshold,
        )
    else:
        if args.ground_filter:
            raise ValueError(
                "--ground-filter requires --fine-vza-bins so filtering is applied before aggregation."
            )
        long = load_long(args.input, args.year)
        filter_stats = pl.DataFrame()
    reference_class = long.sort("vza_midpoint")["vza_class"][0]
    summary, matched, contrasts = build_summaries(long)
    temporal, temporal_summary = build_temporal_changes(long)
    contrast_changes, contrast_change_summary = build_angular_contrast_changes(matched)
    cultivar_comparison = build_cultivar_angular_comparison(long, matched)
    missingness = build_missingness_diagnostics(long, args.year)
    models = fit_models(long)
    cultivar_models = fit_cultivar_angle_models(long)

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    long.write_parquet(
        results_dir / f"plot_week_angle_features_{args.year}.parquet", compression="zstd"
    )
    matched.write_parquet(
        results_dir / f"matched_plot_contrasts_{args.year}.parquet", compression="zstd"
    )
    temporal.write_parquet(
        results_dir / f"matched_temporal_changes_{args.year}.parquet", compression="zstd"
    )
    contrast_changes.write_parquet(
        results_dir / f"matched_angular_contrast_changes_{args.year}.parquet", compression="zstd"
    )
    summary.write_csv(results_dir / "reflectance_by_vza_summary.csv")
    contrasts.write_csv(results_dir / "matched_angular_contrasts.csv")
    temporal_summary.write_csv(results_dir / f"temporal_reflectance_changes_{args.year}.csv")
    contrast_change_summary.write_csv(results_dir / f"angular_contrast_changes_{args.year}.csv")
    cultivar_comparison.write_csv(results_dir / f"cultivar_angular_comparison_{args.year}.csv")
    models.write_csv(results_dir / "angle_week_cultivar_models_preliminary.csv")
    cultivar_models.write_csv(results_dir / f"cultivar_angle_model_{args.year}.csv")
    missingness.write_csv(
        results_dir / f"missing_plot_support_by_week_angle_cultivar_{args.year}.csv"
    )
    if filter_stats.height:
        filter_stats.write_csv(results_dir / f"ground_filter_retention_{args.year}.csv")
    shutil.copyfile(
        results_dir / f"plot_week_angle_features_{args.year}.parquet",
        results_dir / "plot_week_angle_features_preliminary.parquet",
    )
    shutil.copyfile(
        results_dir / f"matched_plot_contrasts_{args.year}.parquet",
        results_dir / "matched_plot_contrasts_preliminary.parquet",
    )
    shutil.copyfile(
        results_dir / "reflectance_by_vza_summary.csv",
        results_dir / "reflectance_by_vza_summary_preliminary.csv",
    )
    shutil.copyfile(
        results_dir / "matched_angular_contrasts.csv",
        results_dir / "matched_angular_contrasts_preliminary.csv",
    )

    plot_reflectance_distributions_by_vza(long, output_dir)
    plot_seasonal_distribution_atlas(long, output_dir, args.year)
    plot_angular_fingerprints(summary, output_dir, args.year)
    plot_matched_contrasts(contrasts, output_dir, reference_class)
    plot_seasonal_angular_contrast_change(contrast_change_summary, output_dir)
    plot_cultivar_curves(long, output_dir, args.year)
    write_report(
        output_dir,
        args.input,
        args.year,
        week_plot_dirs,
        args.ground_filter,
        args.osavi_threshold,
        args.excess_green_threshold,
        filter_stats,
        long,
        contrasts,
        temporal_summary,
        contrast_change_summary,
        cultivar_comparison,
        missingness,
        models,
        cultivar_models,
        log_path,
        fine_vza_bins=args.fine_vza_bins,
    )
    write_captions(output_dir, long, args.year)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
