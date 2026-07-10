#!/usr/bin/env python3
"""Analyze RAA and sun-relative geometry effects for Result 1."""

from __future__ import annotations

import argparse
import logging
import math
import re
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
import pysolar.solar
import pytz

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from matplotlib.colors import Normalize

from src.analysis.result_01_reflectance_distributions import (
    BANDS,
    FINE_VZA_MAX,
    FINE_VZA_MIN,
    MAX_PLOT_SAMPLE,
    SEED,
    WEEK_COLORS,
    YEAR_PLOT_DIRS,
    ensure_indices,
    ground_filter_expressions,
    load_polygon_meta,
    style_axis,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/runs/analysis/reflectance/raa_sun_geometry"
RAA_EDGES = [0, 45, 90, 135, 180]
PHASE_STEP = 10
MIN_HEADLINE_PLOTS = 10
EXPORT_DPI = 300
FIELD_LAT = 51.5648
FIELD_LON = 9.9177
SUN_TIME_ZONE = "Europe/Berlin"
WEEK_ACQUISITION_DATETIMES = {
    2024: {
        0: "2024-06-03 12:00:00",
        2: "2024-06-22 12:00:00",
        3: "2024-06-24 12:00:00",
        4: "2024-07-08 12:00:00",
        5: "2024-07-15 12:00:00",
        6: "2024-07-23 12:00:00",
        7: "2024-07-30 12:00:00",
        8: "2024-08-26 12:00:00",
    },
    2025: {
        0: "2025-06-03 12:00:00",
        3: "2025-06-24 12:00:00",
        5: "2025-07-15 12:00:00",
        7: "2025-07-30 12:00:00",
    },
}
VZA_RAA_MODEL_FORMULA = (
    "reflectance ~ C(vza_class) + C(raa_class) + C(vza_class):C(raa_class) "
    "+ C(week) + C(cult) + C(trt)"
)
VZA_ONLY_MODEL_FORMULA = "reflectance ~ C(vza_class) + C(week) + C(cult) + C(trt)"
PHASE_MODEL_FORMULA = "reflectance ~ C(vza_class) + mean_phase_angle + C(week) + C(cult) + C(trt)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=sorted(YEAR_PLOT_DIRS), default=2024)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ground-filter",
        action="store_true",
        help="Exclude likely ground/background pixels before aggregation using OSAVI and optional ExcessGreen thresholds.",
    )
    parser.add_argument("--osavi-threshold", type=float, default=0.2)
    parser.add_argument("--excess-green-threshold", type=float, default=None)
    return parser.parse_args()


def output_dir(base_output_dir: Path, year: int, ground_filter: bool) -> Path:
    return base_output_dir / str(year) / ("ground_filtered" if ground_filter else "unfiltered")


def configure_logging(out_dir: Path, year: int) -> Path:
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"result_01_raa_{year}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return path


def log_phase(name: str, started: float) -> None:
    logging.info("[PHASE] %s: %.2fs", name, time.perf_counter() - started)


def corrected_sun_angles(year: int, week: int) -> tuple[float, float]:
    timestamp = WEEK_ACQUISITION_DATETIMES.get(year, {}).get(week)
    if timestamp is None:
        raise ValueError(f"No acquisition timestamp configured for year={year}, week={week}")
    tz = pytz.timezone(SUN_TIME_ZONE)
    dt = tz.localize(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))
    elevation = float(pysolar.solar.get_altitude(FIELD_LAT, FIELD_LON, dt))
    azimuth = float(pysolar.solar.get_azimuth(FIELD_LAT, FIELD_LON, dt))
    if azimuth < 0:
        azimuth += 360.0
    return elevation, azimuth


def raa_signed_expr() -> pl.Expr:
    return (
        (pl.col("saa").cast(pl.Float64) - pl.col("vaa").cast(pl.Float64) + 180.0) % 360.0
    ) - 180.0


def raa_abs_expr() -> pl.Expr:
    return raa_signed_expr().abs()


def phase_angle_expr() -> pl.Expr:
    sza_rad = (90.0 - pl.col("sunelev").cast(pl.Float64)) * math.pi / 180.0
    vza_rad = pl.col("vza").cast(pl.Float64) * math.pi / 180.0
    raa_rad = pl.col("raa_abs").cast(pl.Float64) * math.pi / 180.0
    cos_phase = sza_rad.cos() * vza_rad.cos() + sza_rad.sin() * vza_rad.sin() * raa_rad.cos()
    return cos_phase.clip(-1.0, 1.0).arccos() * 180.0 / math.pi


def assign_geometry_bins(frame: pl.DataFrame) -> pl.DataFrame:
    vza_low = (((pl.col("vza") - FINE_VZA_MIN) / 5).floor() * 5 + FINE_VZA_MIN).cast(pl.Int64)
    raa_low = ((pl.col("raa_abs") / 45).floor() * 45).clip(0, 135).cast(pl.Int64)
    phase_low = (
        ((pl.col("phase_angle") / PHASE_STEP).floor() * PHASE_STEP).clip(0, 170).cast(pl.Int64)
    )
    return (
        frame.with_columns(
            vza_low.alias("vza_low"),
            raa_low.alias("raa_low"),
            phase_low.alias("phase_low"),
        )
        .with_columns(
            (
                pl.col("vza_low").cast(pl.Utf8)
                + pl.lit("-")
                + (pl.col("vza_low") + 5).cast(pl.Utf8)
            ).alias("vza_class"),
            (pl.col("vza_low") + 2.5).cast(pl.Float64).alias("vza_midpoint"),
            (
                pl.col("raa_low").cast(pl.Utf8)
                + pl.lit("-")
                + (pl.col("raa_low") + 45).cast(pl.Utf8)
            ).alias("raa_class"),
            (pl.col("raa_low") + 22.5).cast(pl.Float64).alias("raa_midpoint"),
            (
                pl.col("phase_low").cast(pl.Utf8)
                + pl.lit("-")
                + (pl.col("phase_low") + PHASE_STEP).cast(pl.Utf8)
            ).alias("phase_class"),
            (pl.col("phase_low") + PHASE_STEP / 2).cast(pl.Float64).alias("phase_midpoint"),
        )
        .drop("vza_low", "raa_low", "phase_low")
    )


def required_columns() -> set[str]:
    return set(BANDS) | {"vza", "vaa", "saa", "sunelev", "path"}


def load_geometry_summary(
    year: int,
    week_plot_dirs: dict[int, Path],
    ground_filter: bool,
    osavi_threshold: float,
    excess_green_threshold: float | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    started = time.perf_counter()
    meta = load_polygon_meta(year)
    rows = []
    filter_rows = []
    read_seconds = []
    missing_schema = []
    for week, directory in sorted(week_plot_dirs.items()):
        files = sorted(directory.glob("plot_*.parquet"))
        logging.info("Week %s: reading %d plot parquets from %s", week, len(files), directory)
        for path in files:
            plot_id = path.stem
            if plot_id not in meta:
                continue
            read_started = time.perf_counter()
            frame = pl.read_parquet(path)
            read_seconds.append(time.perf_counter() - read_started)
            missing = sorted(required_columns() - set(frame.columns))
            if missing:
                missing_schema.append(
                    {"week": week, "path": str(path), "missing_columns": ",".join(missing)}
                )
                continue
            if frame.height > MAX_PLOT_SAMPLE:
                frame = frame.sample(n=MAX_PLOT_SAMPLE, seed=SEED)
            sampled_rows = frame.height
            mask = (
                pl.col("vza").is_finite()
                & pl.col("vaa").is_finite()
                & pl.col("saa").is_finite()
                & pl.col("sunelev").is_finite()
                & (pl.col("vza") >= FINE_VZA_MIN)
                & (pl.col("vza") < FINE_VZA_MAX)
            )
            for band in BANDS:
                mask = mask & pl.col(band).is_finite() & (pl.col(band) > 0)
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
            if frame.is_empty():
                continue
            frame = (
                frame.select(list(BANDS) + ["vza", "vaa", "saa", "sunelev", "path"])
                .with_columns(
                    (90.0 - pl.col("sunelev").cast(pl.Float64)).alias("sza"),
                    raa_signed_expr().alias("raa_signed"),
                )
                .with_columns(pl.col("raa_signed").abs().alias("raa_abs"))
                .filter((pl.col("raa_abs") >= 0) & (pl.col("raa_abs") <= 180))
                .with_columns(phase_angle_expr().alias("phase_angle"))
            )
            frame = assign_geometry_bins(frame)
            summary = frame.group_by("vza_class", "vza_midpoint", "raa_class", "raa_midpoint").agg(
                pl.len().alias("n_pixels"),
                pl.col("path").n_unique().alias("n_images"),
                pl.col("vza").mean().alias("mean_vza"),
                pl.col("raa_abs").mean().alias("mean_raa"),
                pl.col("sza").mean().alias("mean_sza"),
                pl.col("phase_angle").mean().alias("mean_phase_angle"),
                pl.col("phase_class").mode().first().alias("phase_class"),
                pl.col("phase_midpoint").mean().alias("phase_midpoint"),
                *[pl.col(band).mean().alias(band) for band in BANDS],
            )
            for record in summary.iter_rows(named=True):
                for band, band_name in BANDS.items():
                    rows.append(
                        {
                            "year": year,
                            "week": week,
                            "plot_id": plot_id,
                            "cult": meta[plot_id]["cult"],
                            "trt": meta[plot_id]["trt"],
                            "band": band,
                            "band_name": band_name,
                            "vza_class": record["vza_class"],
                            "vza_midpoint": record["vza_midpoint"],
                            "raa_class": record["raa_class"],
                            "raa_midpoint": record["raa_midpoint"],
                            "phase_class": record["phase_class"],
                            "phase_midpoint": record["phase_midpoint"],
                            "reflectance": record[band],
                            "n_pixels": record["n_pixels"],
                            "n_images": record["n_images"],
                            "mean_vza": record["mean_vza"],
                            "mean_raa": record["mean_raa"],
                            "mean_sza": record["mean_sza"],
                            "mean_phase_angle": record["mean_phase_angle"],
                        }
                    )
    if missing_schema:
        raise RuntimeError(
            f"Plot parquet files are missing required geometry columns: {missing_schema[:5]}"
        )
    if not rows:
        raise RuntimeError("No RAA geometry observations were built.")
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
    logging.info("Loaded %d plot-level RAA observations", long.height)
    log_phase("load_and_aggregate_raa_geometry", started)
    return long, filter_stats


def bootstrap_median(values: np.ndarray, seed: int = SEED) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(2000, values.size), replace=True)
    return tuple(np.quantile(np.median(samples, axis=1), [0.025, 0.975]))


def paired_cohens_d(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    sd = np.std(values, ddof=1)
    return np.nan if sd == 0 else float(np.mean(values) / sd)


def save_readable_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ["pdf", "png", "svg", "tiff"]:
        kwargs = {"bbox_inches": "tight"}
        if extension in {"png", "tiff"}:
            kwargs["dpi"] = EXPORT_DPI
        figure.savefig(stem.with_suffix(f".{extension}"), **kwargs)
    plt.close(figure)


def build_summary_tables(long: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    started = time.perf_counter()
    summary = (
        long.group_by(
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
            pl.len().alias("observations"),
            pl.col("plot_id").n_unique().alias("plots"),
            pl.col("n_pixels").sum().alias("pixels"),
            pl.col("n_images").sum().alias("images"),
            pl.col("reflectance").mean().alias("mean_reflectance"),
            pl.col("reflectance").median().alias("median_reflectance"),
            pl.col("reflectance").quantile(0.25).alias("q25"),
            pl.col("reflectance").quantile(0.75).alias("q75"),
            pl.col("mean_sza").mean().alias("mean_sza"),
            pl.col("mean_phase_angle").mean().alias("mean_phase_angle"),
        )
        .sort("band", "week", "vza_midpoint", "raa_midpoint")
    )
    support = summary.select(
        "year",
        "week",
        "band",
        "band_name",
        "vza_class",
        "vza_midpoint",
        "raa_class",
        "raa_midpoint",
        "plots",
        "pixels",
        "images",
    )
    phase = (
        long.group_by("year", "week", "band", "band_name", "phase_class", "phase_midpoint")
        .agg(
            pl.len().alias("observations"),
            pl.col("plot_id").n_unique().alias("plots"),
            pl.col("reflectance").mean().alias("mean_reflectance"),
            pl.col("reflectance").median().alias("median_reflectance"),
            pl.col("reflectance").quantile(0.25).alias("q25"),
            pl.col("reflectance").quantile(0.75).alias("q75"),
        )
        .sort("band", "week", "phase_midpoint")
    )
    log_phase("summary_tables", started)
    return summary, support, phase


def best_supported_raa_reference(support: pl.DataFrame) -> str:
    return (
        support.group_by("raa_class", "raa_midpoint")
        .agg(pl.col("plots").sum().alias("plot_support"))
        .sort(["plot_support", "raa_midpoint"], descending=[True, False])["raa_class"][0]
    )


def build_matched_raa_contrasts(long: pl.DataFrame, reference_class: str) -> pl.DataFrame:
    started = time.perf_counter()
    reference = long.filter(pl.col("raa_class") == reference_class).select(
        "plot_id",
        "week",
        "band",
        "vza_class",
        pl.col("reflectance").alias("reference_reflectance"),
    )
    matched = (
        long.join(reference, on=["plot_id", "week", "band", "vza_class"], how="inner")
        .filter(pl.col("raa_class") != reference_class)
        .with_columns(
            (pl.col("reflectance") - pl.col("reference_reflectance")).alias("absolute_contrast"),
            (
                (pl.col("reflectance") - pl.col("reference_reflectance"))
                / pl.col("reference_reflectance")
            ).alias("relative_contrast"),
        )
    )
    rows = []
    for key, group in matched.group_by(
        "week", "band", "band_name", "vza_class", "vza_midpoint", "raa_class", "raa_midpoint"
    ):
        week, band, band_name, vza_class, vza_midpoint, raa_class, raa_midpoint = key
        values = group["absolute_contrast"].to_numpy()
        ci_low, ci_high = bootstrap_median(values)
        rows.append(
            {
                "week": week,
                "band": band,
                "band_name": band_name,
                "vza_class": vza_class,
                "vza_midpoint": vza_midpoint,
                "reference_raa_class": reference_class,
                "raa_class": raa_class,
                "raa_midpoint": raa_midpoint,
                "matched_plots": group["plot_id"].n_unique(),
                "median_absolute_contrast": float(np.nanmedian(values)),
                "mean_absolute_contrast": float(np.nanmean(values)),
                "median_relative_contrast": float(
                    np.nanmedian(group["relative_contrast"].to_numpy())
                ),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "cohens_dz": paired_cohens_d(values),
            }
        )
    result = (
        pl.DataFrame(rows).sort("band", "week", "vza_midpoint", "raa_midpoint")
        if rows
        else pl.DataFrame()
    )
    log_phase("matched_raa_contrasts", started)
    return result


def safe_ols(formula: str, frame, cluster_groups):
    model = smf.ols(formula, frame).fit(cov_type="cluster", cov_kwds={"groups": cluster_groups})
    return model


def classify_model_term(term: str) -> str:
    if "vza_class" in term and "raa_class" in term:
        return "vza_raa_interaction"
    if "raa_class" in term:
        return "raa_main"
    if "vza_class" in term:
        return "vza_main"
    if term == "mean_phase_angle":
        return "phase_angle"
    if "C(week)" in term:
        return "week_control"
    if "C(cult)" in term:
        return "cultivar_control"
    if "C(trt)" in term:
        return "treatment_control"
    if term == "Intercept":
        return "intercept"
    return "other"


def model_term_level(term: str, variable: str) -> str | None:
    match = re.search(rf"C\({variable}\)\[T\.([^\]]+)\]", term)
    return match.group(1) if match else None


def wald_test_for_raa_terms(model, terms: list[str]) -> tuple[float, float, int, str]:
    if not terms:
        return np.nan, np.nan, 0, ""
    constraint = np.zeros((len(terms), len(model.params)))
    index = list(model.params.index)
    for row_idx, term in enumerate(terms):
        constraint[row_idx, index.index(term)] = 1.0
    captured_warning = ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        test = model.wald_test(constraint, scalar=True)
        if caught:
            captured_warning = "; ".join(str(item.message) for item in caught)
    return float(test.statistic), float(test.pvalue), len(terms), captured_warning


def extract_model_terms(model, band: str, band_name: str) -> list[dict[str, object]]:
    conf = model.conf_int()
    rows = []
    for term in model.params.index:
        term_type = classify_model_term(term)
        if term_type not in {"raa_main", "vza_raa_interaction", "phase_angle"}:
            continue
        rows.append(
            {
                "band": band,
                "band_name": band_name,
                "term": term,
                "term_type": term_type,
                "vza_class": model_term_level(term, "vza_class"),
                "raa_class": model_term_level(term, "raa_class"),
                "estimate": float(model.params[term]),
                "std_error": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "ci_low": float(conf.loc[term, 0]),
                "ci_high": float(conf.loc[term, 1]),
                "abs_estimate": abs(float(model.params[term])),
            }
        )
    return rows


def build_model_comparison(long: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    started = time.perf_counter()
    rows = []
    term_rows = []
    for band, band_name in BANDS.items():
        frame = long.filter(pl.col("band") == band).to_pandas()
        if frame["plot_id"].nunique() < 2:
            continue
        try:
            base_model = safe_ols(VZA_ONLY_MODEL_FORMULA, frame, frame["plot_id"])
            raa_model = safe_ols(VZA_RAA_MODEL_FORMULA, frame, frame["plot_id"])
            phase_model = safe_ols(PHASE_MODEL_FORMULA, frame, frame["plot_id"])
            raa_terms = [idx for idx in raa_model.params.index if "raa_class" in idx]
            min_raa_p = min((float(raa_model.pvalues[term]) for term in raa_terms), default=np.nan)
            wald_stat, wald_p, wald_constraints, wald_warning = wald_test_for_raa_terms(
                raa_model, raa_terms
            )
            term_rows.extend(extract_model_terms(raa_model, band, band_name))
            term_rows.extend(extract_model_terms(phase_model, band, band_name))
            rows.append(
                {
                    "band": band,
                    "band_name": band_name,
                    "nobs": int(base_model.nobs),
                    "vza_only_r2": float(base_model.rsquared),
                    "vza_raa_r2": float(raa_model.rsquared),
                    "phase_r2": float(phase_model.rsquared),
                    "delta_r2_raa_vs_vza": float(raa_model.rsquared - base_model.rsquared),
                    "delta_r2_phase_vs_vza": float(phase_model.rsquared - base_model.rsquared),
                    "vza_only_adj_r2": float(base_model.rsquared_adj),
                    "vza_raa_adj_r2": float(raa_model.rsquared_adj),
                    "phase_adj_r2": float(phase_model.rsquared_adj),
                    "delta_adj_r2_raa_vs_vza": float(
                        raa_model.rsquared_adj - base_model.rsquared_adj
                    ),
                    "delta_adj_r2_phase_vs_vza": float(
                        phase_model.rsquared_adj - base_model.rsquared_adj
                    ),
                    "vza_only_aic": float(base_model.aic),
                    "vza_raa_aic": float(raa_model.aic),
                    "phase_aic": float(phase_model.aic),
                    "delta_aic_raa_vs_vza": float(raa_model.aic - base_model.aic),
                    "delta_aic_phase_vs_vza": float(phase_model.aic - base_model.aic),
                    "vza_only_bic": float(base_model.bic),
                    "vza_raa_bic": float(raa_model.bic),
                    "phase_bic": float(phase_model.bic),
                    "delta_bic_raa_vs_vza": float(raa_model.bic - base_model.bic),
                    "delta_bic_phase_vs_vza": float(phase_model.bic - base_model.bic),
                    "min_raa_term_p": min_raa_p,
                    "raa_joint_wald_stat": wald_stat,
                    "raa_joint_wald_p": wald_p,
                    "raa_joint_wald_constraints": wald_constraints,
                    "raa_joint_wald_warning": wald_warning,
                    "phase_angle_estimate": float(
                        phase_model.params.get("mean_phase_angle", np.nan)
                    ),
                    "phase_angle_p": float(phase_model.pvalues.get("mean_phase_angle", np.nan)),
                }
            )
            logging.info("[ML] fit %s RAA model comparison", band_name)
        except Exception as exc:  # statsmodels can fail on singular sparse cells.
            logging.warning("Skipping %s model comparison: %s", band_name, exc)
    result = pl.DataFrame(rows) if rows else pl.DataFrame()
    terms = pl.DataFrame(term_rows) if term_rows else pl.DataFrame()
    log_phase("model_comparison", started)
    return result, terms


def heatmap_matrix(
    data: pl.DataFrame, value_column: str
) -> tuple[np.ndarray, list[str], list[str]]:
    matrix = data.pivot(
        on="raa_class",
        index="vza_class",
        values=value_column,
        aggregate_function="mean",
    ).sort("vza_class")
    raa_cols = sorted(
        [col for col in matrix.columns if col != "vza_class"], key=lambda x: int(x.split("-")[0])
    )
    return matrix.select(raa_cols).to_numpy(), matrix["vza_class"].to_list(), raa_cols


def plot_raa_heatmap(summary: pl.DataFrame, out_dir: Path, year: int) -> None:
    started = time.perf_counter()
    weeks = sorted(summary["week"].unique().to_list())
    figure, axes = plt.subplots(
        len(BANDS), len(weeks), figsize=(1.85 * len(weeks) + 1.0, 9.6), constrained_layout=True
    )
    last_image = None
    for row_index, (band, band_name) in enumerate(BANDS.items()):
        band_data = summary.filter(pl.col("band") == band)
        norm = Normalize(
            vmin=float(band_data["median_reflectance"].min()),
            vmax=float(band_data["median_reflectance"].max()),
        )
        for col_index, week in enumerate(weeks):
            axis = axes[row_index, col_index]
            values, vza_labels, raa_labels = heatmap_matrix(
                band_data.filter(pl.col("week") == week), "median_reflectance"
            )
            last_image = axis.imshow(values, aspect="auto", cmap="YlGnBu", norm=norm)
            if row_index == 0:
                axis.set_title(f"W{week}", fontsize=8.5, fontweight="bold")
            if col_index == 0:
                axis.set_ylabel(band_name, fontsize=8.5, fontweight="bold")
                axis.set_yticks(range(len(vza_labels)), vza_labels, fontsize=6.5)
            else:
                axis.set_yticks([])
            if row_index == len(BANDS) - 1:
                axis.set_xticks(
                    range(len(raa_labels)), raa_labels, rotation=45, ha="right", fontsize=6.5
                )
            else:
                axis.set_xticks([])
            axis.tick_params(length=0)
            for spine in axis.spines.values():
                spine.set_visible(False)
    figure.colorbar(last_image, ax=axes, shrink=0.55, label="Median reflectance")
    figure.suptitle(f"{year}: median reflectance over VZA x RAA", fontsize=11, fontweight="bold")
    save_readable_figure(figure, out_dir / f"figures/main/raa_vza_reflectance_heatmap_{year}")
    log_phase("raa_heatmaps", started)


def plot_raa_curves(summary: pl.DataFrame, out_dir: Path, year: int) -> None:
    started = time.perf_counter()
    weeks = sorted(summary["week"].unique().to_list())
    vza_order = (
        summary.select("vza_class", "vza_midpoint")
        .unique()
        .sort("vza_midpoint")["vza_class"]
        .to_list()
    )
    palette = plt.cm.viridis(np.linspace(0.15, 0.90, len(vza_order)))
    colors = {vza: palette[index] for index, vza in enumerate(vza_order)}
    figure, axes = plt.subplots(
        len(BANDS), len(weeks), figsize=(2.35 * len(weeks) + 1.1, 11.2), constrained_layout=True
    )
    for row_idx, (band, band_name) in enumerate(BANDS.items()):
        for col_idx, week in enumerate(weeks):
            axis = axes[row_idx, col_idx]
            for vza in vza_order:
                data = summary.filter(
                    (pl.col("band") == band)
                    & (pl.col("week") == week)
                    & (pl.col("vza_class") == vza)
                ).sort("raa_midpoint")
                if data.is_empty():
                    continue
                x = data["raa_midpoint"].to_numpy()
                y = data["mean_reflectance"].to_numpy()
                axis.plot(
                    x, y, marker="o", markersize=2.2, linewidth=1.05, label=vza, color=colors[vza]
                )
                axis.fill_between(
                    x, data["q25"].to_numpy(), data["q75"].to_numpy(), color=colors[vza], alpha=0.08
                )
            if row_idx == 0:
                axis.set_title(f"Week {week}", fontsize=9, fontweight="bold")
            if col_idx == 0:
                axis.set_ylabel(f"{band_name}\nreflectance", fontsize=8)
            if row_idx == len(BANDS) - 1:
                axis.set_xlabel("RAA midpoint", fontsize=8)
            style_axis(axis)
            axis.tick_params(axis="both", labelsize=6.5)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(5, len(labels)),
        frameon=False,
        title="VZA class",
    )
    save_readable_figure(figure, out_dir / f"figures/main/raa_curves_within_vza_{year}")
    log_phase("raa_curves", started)


def plot_phase_curves(phase: pl.DataFrame, out_dir: Path, year: int) -> None:
    started = time.perf_counter()
    weeks = sorted(phase["week"].unique().to_list())
    figure, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), constrained_layout=True)
    for idx, (band, band_name) in enumerate(BANDS.items()):
        axis = axes.flat[idx]
        for week in weeks:
            data = phase.filter((pl.col("band") == band) & (pl.col("week") == week)).sort(
                "phase_midpoint"
            )
            axis.plot(
                data["phase_midpoint"].to_numpy(),
                data["median_reflectance"].to_numpy(),
                marker="o",
                markersize=3,
                linewidth=1.35,
                color=WEEK_COLORS.get(week, "#4D4D4D"),
                label=f"Week {week}",
            )
            axis.fill_between(
                data["phase_midpoint"].to_numpy(),
                data["q25"].to_numpy(),
                data["q75"].to_numpy(),
                color=WEEK_COLORS.get(week, "#4D4D4D"),
                alpha=0.10,
            )
        axis.set_title(band_name, loc="left", fontsize=10, fontweight="bold")
        axis.set_xlabel("Phase angle midpoint (degrees)")
        axis.set_ylabel("Reflectance")
        style_axis(axis)
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc="center", frameon=False)
    save_readable_figure(figure, out_dir / f"figures/main/phase_angle_reflectance_curves_{year}")
    log_phase("phase_curves", started)


def plot_support_heatmap(support: pl.DataFrame, out_dir: Path, year: int) -> None:
    started = time.perf_counter()
    weeks = sorted(support["week"].unique().to_list())
    figure, axes = plt.subplots(
        1, len(weeks), figsize=(1.9 * len(weeks) + 1.0, 3.2), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    support_any_band = (
        support.group_by("week", "vza_class", "vza_midpoint", "raa_class", "raa_midpoint")
        .agg(pl.col("plots").min().alias("plots"))
        .sort("week", "vza_midpoint", "raa_midpoint")
    )
    for axis, week in zip(axes, weeks):
        data = support_any_band.filter(pl.col("week") == week)
        values, vza_labels, raa_cols = heatmap_matrix(data, "plots")
        image = axis.imshow(
            values, aspect="auto", cmap="Greys", vmin=0, vmax=max(1, float(support["plots"].max()))
        )
        axis.set_title(f"Week {week}", fontsize=9, fontweight="bold")
        axis.set_xticks(range(len(raa_cols)), raa_cols, rotation=45, ha="right", fontsize=6.5)
        axis.set_yticks(range(len(vza_labels)), vza_labels, fontsize=6.5)
        axis.set_xlabel("RAA class")
        axis.set_ylabel("VZA class")
    figure.colorbar(image, ax=axes, shrink=0.82, label="Plots")
    save_readable_figure(figure, out_dir / f"figures/main/raa_support_heatmap_{year}")
    log_phase("support_heatmap", started)


def plot_matched_contrasts(contrasts: pl.DataFrame, out_dir: Path, year: int) -> None:
    started = time.perf_counter()
    if contrasts.is_empty():
        logging.warning("Skipping matched RAA contrast figure because no contrasts were built")
        return
    data = contrasts.filter(pl.col("matched_plots") >= MIN_HEADLINE_PLOTS)
    if data.is_empty():
        data = contrasts
    top = (
        data.with_columns(pl.col("median_absolute_contrast").abs().alias("magnitude"))
        .sort("magnitude", descending=True)
        .group_by("band", maintain_order=True)
        .head(12)
        .sort("band", "magnitude", descending=[False, True])
    )
    figure, axes = plt.subplots(2, 3, figsize=(11.8, 8.0), constrained_layout=True)
    for idx, (band, band_name) in enumerate(BANDS.items()):
        axis = axes.flat[idx]
        band_data = top.filter(pl.col("band") == band).sort("magnitude")
        labels = [
            f"W{row['week']} {row['vza_class']} {row['raa_class']}"
            for row in band_data.iter_rows(named=True)
        ]
        y = np.arange(len(labels))
        values = band_data["median_absolute_contrast"].to_numpy()
        lows = band_data["ci_low"].to_numpy()
        highs = band_data["ci_high"].to_numpy()
        if len(values):
            axis.errorbar(
                values,
                y,
                xerr=np.vstack((values - lows, highs - values)),
                fmt="o",
                markersize=3.0,
                color="#176B6B",
                ecolor="#76A9A9",
                elinewidth=0.8,
                capsize=1.8,
            )
        axis.axvline(0, color="#222222", linewidth=0.9)
        axis.set_title(band_name, loc="left", fontsize=10, fontweight="bold")
        axis.set_yticks(y, labels, fontsize=6.2)
        axis.set_xlabel("Reflectance difference")
        style_axis(axis)
    axes.flat[5].axis("off")
    figure.suptitle(f"{year}: strongest matched RAA contrasts", fontsize=11, fontweight="bold")
    save_readable_figure(figure, out_dir / f"figures/main/matched_raa_contrasts_{year}")
    log_phase("matched_contrast_figure", started)


def write_report(
    out_dir: Path,
    year: int,
    ground_filter: bool,
    osavi_threshold: float,
    excess_green_threshold: float | None,
    filter_stats: pl.DataFrame,
    long: pl.DataFrame,
    support: pl.DataFrame,
    contrasts: pl.DataFrame,
    models: pl.DataFrame,
    model_terms: pl.DataFrame,
    log_path: Path,
) -> None:
    started = time.perf_counter()
    weeks = ", ".join(map(str, sorted(long["week"].unique().to_list())))
    sparse = support.filter(pl.col("plots") < MIN_HEADLINE_PLOTS).height
    best = (
        contrasts.filter(pl.col("matched_plots") >= MIN_HEADLINE_PLOTS)
        .with_columns(pl.col("median_absolute_contrast").abs().alias("magnitude"))
        .sort("magnitude", descending=True)
        .head(8)
        if not contrasts.is_empty()
        else contrasts
    )
    top_terms = (
        model_terms.filter(pl.col("term_type") == "vza_raa_interaction")
        .sort("abs_estimate", descending=True)
        .head(10)
        if not model_terms.is_empty()
        else model_terms
    )
    filter_lines = []
    if ground_filter and filter_stats.height:
        total_basic = filter_stats["rows_after_basic_quality"].sum()
        total_after = filter_stats["rows_after_ground_filter"].sum()
        removed = total_basic - total_after
        rule = f"`OSAVI > {osavi_threshold}`"
        if excess_green_threshold is not None:
            rule += f" and `ExcessGreen > {excess_green_threshold}`"
        filter_lines = [
            "",
            "## Ground/Background Filter",
            "",
            f"- Rule: {rule}",
            f"- Rows after basic geometry/band quality filter: **{total_basic:,}**",
            f"- Rows retained after ground filter: **{total_after:,}**",
            f"- Rows removed by ground filter: **{removed:,}** ({(removed / total_basic * 100) if total_basic else 0:.1f}%)",
            f"- Retention table: `{out_dir / f'results/ground_filter_retention_{year}.csv'}`",
        ]
    lines = [
        "# Result 1 Support: RAA and Sun-Relative Geometry",
        "",
        f"Validated {year} weeks: **{weeks}**.",
        "",
        "## Interpretation",
        "",
        "This diagnostic tests whether reflectance varies with relative azimuth to the sun after holding VZA approximately constant. Strong RAA effects, especially in red edge and NIR, indicate that the VZA-only Result 1 summaries hide sun-view and canopy-structure anisotropy.",
        "",
        "## Dataset",
        "",
        f"- Plot-week records: **{long.select('plot_id', 'week').unique().height}**",
        f"- Plot-level RAA observations: **{long.height:,}**",
        f"- Sparse support cells with fewer than {MIN_HEADLINE_PLOTS} plots: **{sparse}**",
        "- RAA formula: `abs(((saa - vaa + 180) % 360) - 180)`.",
        "- Phase angle is derived from existing `sunelev`, `vza`, and RAA.",
        "",
        "## Largest Matched RAA Contrasts",
        "",
        "| Week | Band | VZA | Reference RAA | RAA | Median contrast | 95% CI | Matched plots |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]
    for row in best.iter_rows(named=True):
        lines.append(
            f"| {row['week']} | {row['band_name']} | {row['vza_class']} | {row['reference_raa_class']} | "
            f"{row['raa_class']} | {row['median_absolute_contrast']:.4f} | "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['matched_plots']} |"
        )
    if not models.is_empty():
        lines.extend(
            [
                "",
                "## Model Comparison and Interpretability",
                "",
                "The VZA-only model is the baseline. The RAA model adds categorical RAA and VZA x RAA interactions; the phase model adds continuous sun-view phase angle. Negative Delta AIC means better AIC than VZA-only. Delta BIC can penalize the full categorical RAA model because it has many interaction terms.",
                "",
                "| Band | R2 VZA | R2 RAA | Delta R2 RAA | Delta adj R2 RAA | Delta AIC RAA | Delta BIC RAA | R2 phase | Delta R2 phase | phase angle coef | phase p |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in models.iter_rows(named=True):
            lines.append(
                f"| {row['band_name']} | {row['vza_only_r2']:.4f} | {row['vza_raa_r2']:.4f} | "
                f"{row['delta_r2_raa_vs_vza']:.4f} | {row['delta_adj_r2_raa_vs_vza']:.4f} | "
                f"{row['delta_aic_raa_vs_vza']:.2f} | {row['delta_bic_raa_vs_vza']:.2f} | "
                f"{row['phase_r2']:.4f} | {row['delta_r2_phase_vs_vza']:.4f} | "
                f"{row['phase_angle_estimate']:.6f} | {row['phase_angle_p']:.4g} |"
            )
        lines.extend(
            [
                "",
                "For interpretation, prioritize the matched RAA contrasts because they compare the same plot, week, band, and VZA bin against a common RAA reference. The coefficient table is useful for model diagnostics, but coefficients are reference-coded relative to the model baseline categories.",
            ]
        )
    if not top_terms.is_empty():
        lines.extend(
            [
                "",
                "## Largest Reference-Coded VZA x RAA Model Terms",
                "",
                "| Band | VZA term | RAA term | Estimate | 95% CI | p |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for row in top_terms.iter_rows(named=True):
            lines.append(
                f"| {row['band_name']} | {row['vza_class']} | {row['raa_class']} | "
                f"{row['estimate']:.5f} | [{row['ci_low']:.5f}, {row['ci_high']:.5f}] | {row['p_value']:.4g} |"
            )
    lines.extend(filter_lines)
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- RAA summary: `{out_dir / f'results/raa_reflectance_summary_{year}.csv'}`",
            f"- Matched RAA contrasts: `{out_dir / f'results/matched_raa_contrasts_{year}.csv'}`",
            f"- Model comparison: `{out_dir / f'results/raa_vza_model_comparison_{year}.csv'}`",
            f"- Reference-coded model terms: `{out_dir / f'results/raa_model_terms_{year}.csv'}`",
            f"- Largest RAA interaction terms: `{out_dir / f'results/raa_top_interactions_{year}.csv'}`",
            f"- Support table: `{out_dir / f'results/raa_support_by_week_vza_{year}.csv'}`",
            f"- Phase summary: `{out_dir / f'results/phase_angle_reflectance_summary_{year}.csv'}`",
            f"- Main figures: `{out_dir / 'figures/main'}`",
            "",
            "## Reproducibility",
            "",
            f"- Log: `{log_path}`",
            f"- Random seed: `{SEED}`",
            f"- Ground/background filter: {'enabled' if ground_filter else 'disabled'}.",
            f"- OSAVI threshold: {osavi_threshold if ground_filter else 'not applied'}.",
            f"- ExcessGreen threshold: {excess_green_threshold if ground_filter and excess_green_threshold is not None else 'not applied'}.",
            f"- VZA range: `{FINE_VZA_MIN} <= vza < {FINE_VZA_MAX}`.",
            f"- RAA classes: `{', '.join(f'{lo}-{hi}' for lo, hi in zip(RAA_EDGES[:-1], RAA_EDGES[1:]))}`.",
            "- Sampling: plot parquets above `MAX_PLOT_SAMPLE` are sampled with seed 42 before filtering.",
            "",
        ]
    )
    report = out_dir / f"reports/raa_sun_geometry_summary_{year}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    shutil.copyfile(report, out_dir / "reports/raa_sun_geometry_summary.md")
    log_phase("report", started)


def write_outputs(
    out_dir: Path,
    year: int,
    ground_filter: bool,
    filter_stats: pl.DataFrame,
    long: pl.DataFrame,
    summary: pl.DataFrame,
    support: pl.DataFrame,
    phase: pl.DataFrame,
    contrasts: pl.DataFrame,
    models: pl.DataFrame,
    model_terms: pl.DataFrame,
) -> None:
    started = time.perf_counter()
    results = out_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    retention_path = results / f"ground_filter_retention_{year}.csv"
    if not ground_filter and retention_path.exists():
        retention_path.unlink()
    long.write_parquet(results / f"plot_week_vza_raa_features_{year}.parquet", compression="zstd")
    summary.write_csv(results / f"raa_reflectance_summary_{year}.csv")
    support.write_csv(results / f"raa_support_by_week_vza_{year}.csv")
    phase.write_csv(results / f"phase_angle_reflectance_summary_{year}.csv")
    contrasts.write_csv(results / f"matched_raa_contrasts_{year}.csv")
    models.write_csv(results / f"raa_vza_model_comparison_{year}.csv")
    if model_terms.is_empty():
        (results / f"raa_model_terms_{year}.csv").write_text("", encoding="utf-8")
        (results / f"raa_top_interactions_{year}.csv").write_text("", encoding="utf-8")
    else:
        model_terms.write_csv(results / f"raa_model_terms_{year}.csv")
        (
            model_terms.filter(pl.col("term_type") == "vza_raa_interaction")
            .sort("abs_estimate", descending=True)
            .write_csv(results / f"raa_top_interactions_{year}.csv")
        )
    if ground_filter and filter_stats.height:
        filter_stats.write_csv(retention_path)
    log_phase("write_outputs", started)


def main() -> None:
    args = parse_args()
    out_dir = output_dir(args.output_dir, args.year, args.ground_filter)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(out_dir, args.year)
    total_started = time.perf_counter()
    logging.info("Starting RAA/sun-geometry Result 1 support analysis for %s", args.year)

    long, filter_stats = load_geometry_summary(
        args.year,
        YEAR_PLOT_DIRS[args.year],
        args.ground_filter,
        args.osavi_threshold,
        args.excess_green_threshold,
    )
    summary, support, phase = build_summary_tables(long)
    reference_raa = best_supported_raa_reference(support)
    logging.info("Using %s as matched RAA reference class", reference_raa)
    contrasts = build_matched_raa_contrasts(long, reference_raa)
    models, model_terms = build_model_comparison(long)
    write_outputs(
        out_dir,
        args.year,
        args.ground_filter,
        filter_stats,
        long,
        summary,
        support,
        phase,
        contrasts,
        models,
        model_terms,
    )
    plot_raa_heatmap(summary, out_dir, args.year)
    plot_raa_curves(summary, out_dir, args.year)
    plot_phase_curves(phase, out_dir, args.year)
    plot_support_heatmap(support, out_dir, args.year)
    plot_matched_contrasts(contrasts, out_dir, args.year)
    write_report(
        out_dir,
        args.year,
        args.ground_filter,
        args.osavi_threshold,
        args.excess_green_threshold,
        filter_stats,
        long,
        support,
        contrasts,
        models,
        model_terms,
        log_path,
    )
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
