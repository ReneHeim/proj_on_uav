#!/usr/bin/env python3
"""EDA: test whether LAI instrument canopy variables explain VZA reflectance contrasts."""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/proj_on_uav_matplotlib")

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
LAI_RAW_ROOT = Path("/run/media/davidem/heim_data/Backup/proj_on_cerco/data/processed/2024")
POLYGON_PATH = Path("/run/media/davidem/Heim/2024_oncerco_plot_polygons.gpkg")
CONTRAST_SOURCE = (
    ROOT
    / "outputs/result_01_reflectance_distributions/2024/ground_filtered/results/matched_plot_contrasts_preliminary.parquet"
)
OUT_ROOT = ROOT / "outputs/backup_metadata/eda_lai_canopy_vza"
REPORTS_ROOT = ROOT / "outputs/reports"
LOG_ROOT = ROOT / "outputs/logs"

BANDS_FOR_PAPER = ["Red edge", "NIR"]
ANGLE_RINGS = [7, 23, 38, 53, 68]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lai-raw-root", type=Path, default=LAI_RAW_ROOT)
    parser.add_argument("--polygon-path", type=Path, default=POLYGON_PATH)
    parser.add_argument("--contrast-source", type=Path, default=CONTRAST_SOURCE)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def configure_logging() -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / f"analyze_lai_canopy_vza_reflectance_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return path


def phase(name: str, t0: float) -> None:
    logging.info("[PHASE] %s: %.1fs", name, time.perf_counter() - t0)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def normalize_key(key: str) -> str:
    return key.strip().lower().replace("#", "").replace(".", "").replace(" ", "_")


def parse_numeric_values(parts: list[str]) -> list[float]:
    values = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values


def parse_lai_txt(path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "source_path": str(path),
        "file_name": path.name,
        "file_base": path.stem,
    }
    match = re.search(r"week(\d+)", str(path))
    row["week"] = int(match.group(1)) if match else np.nan
    plot_match = re.match(r"(\d+)-(\d+)", path.stem)
    if plot_match:
        row["ifz_id"] = int(plot_match.group(1))
        row["rep"] = int(plot_match.group(2))
        row["plot_id"] = f"plot_{90024 - int(plot_match.group(1))}"

    for raw_line in path.read_text(errors="replace").splitlines():
        if not raw_line or raw_line.startswith("###"):
            continue
        parts = raw_line.strip().split()
        if not parts:
            continue
        key = normalize_key(parts[0])
        if key == "date" and len(parts) >= 3:
            row["measurement_datetime"] = f"{parts[1]} {parts[2]}"
            continue
        if key in {
            "lai",
            "sel",
            "acf",
            "difn",
            "mta",
            "sem",
            "smp",
            "gpslat",
            "gpslong",
            "gpsalt",
            "gpshdop",
            "gpsnum",
        }:
            values = parse_numeric_values(parts[1:])
            row[key] = values[0] if values else np.nan
            continue
        if key in {"avgtrans", "acfs", "cntct", "stddev", "dists", "gaps", "angles", "mask"}:
            values = parse_numeric_values(parts[1:])
            for idx, value in enumerate(values[: len(ANGLE_RINGS)]):
                suffix = ANGLE_RINGS[idx]
                prefix = "cntct" if key == "cntct" else key
                row[f"{prefix}_{suffix}"] = value
    return row


def load_lai_raw(root: Path, polygon_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.perf_counter()
    files = sorted(root.glob("*week*/lai/*.TXT"))
    read_times = []
    rows = []
    for path in files:
        ft0 = time.perf_counter()
        rows.append(parse_lai_txt(path))
        read_times.append(time.perf_counter() - ft0)
    raw = pd.DataFrame(rows)
    logging.info("[I/O] read LAI TXT files n=%d root=%s", len(files), root)
    if read_times:
        arr = np.asarray(read_times)
        logging.info(
            "[I/O] LAI TXT read seconds min=%.4f median=%.4f mean=%.4f max=%.4f",
            arr.min(),
            np.median(arr),
            arr.mean(),
            arr.max(),
        )

    polygons = gpd.read_file(polygon_path)[["ifz_id", "cult", "trt"]].copy()
    polygons["ifz_id"] = polygons["ifz_id"].astype(int)
    raw = raw.merge(polygons, on="ifz_id", how="left", validate="many_to_one")
    raw["measurement_datetime"] = pd.to_datetime(
        raw["measurement_datetime"], format="%Y%m%d %H:%M:%S", errors="coerce"
    )
    phase("load_lai_raw", t0)
    return raw, polygons


def aggregate_lai(raw: pd.DataFrame) -> pd.DataFrame:
    t0 = time.perf_counter()
    numeric_cols = [
        col
        for col in raw.columns
        if col
        in {
            "lai",
            "sel",
            "acf",
            "difn",
            "mta",
            "sem",
            "smp",
            *[f"avgtrans_{a}" for a in ANGLE_RINGS],
            *[f"acfs_{a}" for a in ANGLE_RINGS],
            *[f"cntct_{a}" for a in ANGLE_RINGS],
            *[f"stddev_{a}" for a in ANGLE_RINGS],
            *[f"dists_{a}" for a in ANGLE_RINGS],
            *[f"gaps_{a}" for a in ANGLE_RINGS],
        }
    ]
    for col in numeric_cols:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    agg = (
        raw.groupby(["plot_id", "ifz_id", "week", "cult", "trt"], as_index=False)
        .agg(
            **{f"{col}_mean": (col, "mean") for col in numeric_cols},
            lai_sd_reps=("lai", "std"),
            n_lai_reps=("lai", "size"),
            n_lai_valid=("lai", "count"),
        )
        .rename(columns={"lai_mean": "lai", "difn_mean": "difn", "acf_mean": "acf"})
    )
    phase("aggregate_lai", t0)
    return agg


def nearest_ring(row: pd.Series, prefix: str) -> float:
    ring = min(ANGLE_RINGS, key=lambda angle: abs(angle - float(row["vza_midpoint"])))
    return row.get(f"{prefix}_{ring}_mean", np.nan)


def build_join(contrast_source: Path, lai_plot_week: pd.DataFrame) -> pd.DataFrame:
    t0 = time.perf_counter()
    contrasts = pl.read_parquet(contrast_source).to_pandas()
    logging.info(
        "[I/O] read contrasts rows=%d cols=%d path=%s",
        contrasts.shape[0],
        contrasts.shape[1],
        contrast_source,
    )
    contrasts["week"] = contrasts["week"].astype(int)
    lai_plot_week["week"] = lai_plot_week["week"].astype(int)
    joined = contrasts.merge(
        lai_plot_week,
        on=["plot_id", "week", "cult", "trt"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_lai"),
    )
    for prefix in ["gaps", "avgtrans", "cntct", "acfs"]:
        joined[f"{prefix}_nearest_vza"] = joined.apply(
            lambda row: nearest_ring(row, prefix), axis=1
        )
    joined["canopy_closure"] = 1.0 - joined["difn"]
    joined["lai_z"] = (joined["lai"] - joined["lai"].mean()) / joined["lai"].std(ddof=0)
    joined["gap_z"] = (joined["gaps_nearest_vza"] - joined["gaps_nearest_vza"].mean()) / joined[
        "gaps_nearest_vza"
    ].std(ddof=0)
    joined["trans_z"] = (
        joined["avgtrans_nearest_vza"] - joined["avgtrans_nearest_vza"].mean()
    ) / joined["avgtrans_nearest_vza"].std(ddof=0)
    phase("build_join", t0)
    return joined


def fit_models(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.perf_counter()
    formulas = {
        "M0_vza_controls": "relative_contrast ~ C(week) + C(cult) + C(trt) + C(vza_class)",
        "M1_add_lai": "relative_contrast ~ C(week) + C(cult) + C(trt) + C(vza_class) + lai_z",
        "M2_add_difn_acf": "relative_contrast ~ C(week) + C(cult) + C(trt) + C(vza_class) + lai_z + difn + acf",
        "M3_add_angle_gap": "relative_contrast ~ C(week) + C(cult) + C(trt) + C(vza_class) + lai_z + difn + acf + gap_z",
        "M4_gap_lai_interactions": (
            "relative_contrast ~ C(week) + C(cult) + C(trt) + C(vza_class)"
            " + lai_z + difn + acf + gap_z + C(vza_class):lai_z + C(vza_class):gap_z"
        ),
    }
    model_rows = []
    term_rows = []
    for band in sorted(joined["band_name"].dropna().unique()):
        band_df = joined[joined["band_name"] == band].dropna(
            subset=["relative_contrast", "lai_z", "difn", "acf", "gap_z", "vza_class", "plot_id"]
        )
        if band_df["plot_id"].nunique() < 8:
            continue
        for model_name, formula in formulas.items():
            fit_t0 = time.perf_counter()
            model = smf.ols(formula, data=band_df).fit(
                cov_type="cluster", cov_kwds={"groups": band_df["plot_id"]}
            )
            logging.info(
                "[ML] fit band=%s model=%s time=%.2fs",
                band,
                model_name,
                time.perf_counter() - fit_t0,
            )
            pred_t0 = time.perf_counter()
            _ = model.predict(band_df.head(10))
            logging.info(
                "[ML] predict band=%s model=%s time=%.4fs",
                band,
                model_name,
                time.perf_counter() - pred_t0,
            )
            model_rows.append(
                {
                    "band_name": band,
                    "model": model_name,
                    "n_rows": int(model.nobs),
                    "n_plots": int(band_df["plot_id"].nunique()),
                    "adj_r2": model.rsquared_adj,
                    "aic": model.aic,
                    "bic": model.bic,
                }
            )
            if model_name == "M3_add_angle_gap":
                for term in ["lai_z", "difn", "acf", "gap_z"]:
                    term_rows.append(
                        {
                            "band_name": band,
                            "term": term,
                            "coef": model.params.get(term, np.nan),
                            "p_cluster": model.pvalues.get(term, np.nan),
                            "conf_low": (
                                model.conf_int().loc[term, 0]
                                if term in model.params.index
                                else np.nan
                            ),
                            "conf_high": (
                                model.conf_int().loc[term, 1]
                                if term in model.params.index
                                else np.nan
                            ),
                        }
                    )
    phase("fit_models", t0)
    return pd.DataFrame(model_rows), pd.DataFrame(term_rows)


def correlation_summary(joined: pd.DataFrame) -> pd.DataFrame:
    t0 = time.perf_counter()
    rows = []
    variables = [
        "lai",
        "difn",
        "acf",
        "gaps_nearest_vza",
        "avgtrans_nearest_vza",
        "cntct_nearest_vza",
    ]
    targets = ["relative_contrast", "absolute_contrast", "reflectance"]
    for band, band_df in joined.groupby("band_name"):
        for var in variables:
            for target in targets:
                clean = band_df[[var, target, "plot_id"]].dropna()
                if len(clean) < 20:
                    continue
                rows.append(
                    {
                        "band_name": band,
                        "variable": var,
                        "target": target,
                        "n_rows": len(clean),
                        "n_plots": clean["plot_id"].nunique(),
                        "pearson_r": clean[var].corr(clean[target], method="pearson"),
                        "spearman_r": clean[var].corr(clean[target], method="spearman"),
                    }
                )
    phase("correlation_summary", t0)
    return pd.DataFrame(rows)


def model_deltas(model_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band, band_df in model_summary.groupby("band_name"):
        lookup = band_df.set_index("model")
        base = lookup.loc["M0_vza_controls"]
        for model in [
            "M1_add_lai",
            "M2_add_difn_acf",
            "M3_add_angle_gap",
            "M4_gap_lai_interactions",
        ]:
            if model not in lookup.index:
                continue
            row = lookup.loc[model]
            rows.append(
                {
                    "band_name": band,
                    "model": model,
                    "delta_adj_r2_vs_M0": row["adj_r2"] - base["adj_r2"],
                    "delta_aic_vs_M0": row["aic"] - base["aic"],
                    "delta_bic_vs_M0": row["bic"] - base["bic"],
                }
            )
    return pd.DataFrame(rows)


def summarize_gap_group_curves(joined: pd.DataFrame) -> pd.DataFrame:
    """Summarize VZA contrast curves for low- and high-gap canopies."""
    data = joined[
        (joined["band_name"].isin(BANDS_FOR_PAPER)) & (joined["week"].isin([5, 6]))
    ].dropna(subset=["relative_contrast", "gaps_nearest_vza", "vza_midpoint"])
    rows: list[dict[str, object]] = []
    for (band, vza_class), group in data.groupby(["band_name", "vza_class"]):
        low_cut = group["gaps_nearest_vza"].quantile(0.33)
        high_cut = group["gaps_nearest_vza"].quantile(0.67)
        for label, subset in [
            ("closed canopy\nlow gap", group[group["gaps_nearest_vza"] <= low_cut]),
            ("open canopy\nhigh gap", group[group["gaps_nearest_vza"] >= high_cut]),
        ]:
            if subset.empty:
                continue
            rows.append(
                {
                    "band_name": band,
                    "vza_class": vza_class,
                    "vza_midpoint": float(subset["vza_midpoint"].median()),
                    "gap_group": label,
                    "median_relative_contrast": float(subset["relative_contrast"].median()),
                    "q25_relative_contrast": float(subset["relative_contrast"].quantile(0.25)),
                    "q75_relative_contrast": float(subset["relative_contrast"].quantile(0.75)),
                    "mean_gap_fraction": float(subset["gaps_nearest_vza"].mean()),
                    "n_rows": int(len(subset)),
                    "n_plots": int(subset["plot_id"].nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(["band_name", "gap_group", "vza_midpoint"])


def summarize_gap_tercile_points(
    joined: pd.DataFrame, weeks: tuple[int, ...] = (5, 6)
) -> pd.DataFrame:
    """Summarize closed/mid/open canopy mean contrast points per VZA bin."""
    data = joined[
        (joined["band_name"].isin(BANDS_FOR_PAPER)) & (joined["week"].isin(weeks))
    ].dropna(subset=["relative_contrast", "gaps_nearest_vza", "vza_midpoint"])
    rows: list[dict[str, object]] = []
    labels = ["closed canopy", "mid canopy", "open canopy"]
    for (band, vza_class), group in data.groupby(["band_name", "vza_class"]):
        try:
            group = group.copy()
            group["gap_group"] = pd.qcut(
                group["gaps_nearest_vza"], q=3, labels=labels, duplicates="drop"
            )
        except ValueError:
            continue
        for gap_group, subset in group.groupby("gap_group", observed=True):
            if subset.empty:
                continue
            rows.append(
                {
                    "band_name": band,
                    "vza_class": vza_class,
                    "vza_midpoint": float(subset["vza_midpoint"].median()),
                    "gap_group": str(gap_group),
                    "mean_relative_contrast": float(subset["relative_contrast"].mean()),
                    "se_relative_contrast": float(
                        subset["relative_contrast"].std(ddof=1) / np.sqrt(len(subset))
                        if len(subset) > 1
                        else 0.0
                    ),
                    "median_relative_contrast": float(subset["relative_contrast"].median()),
                    "mean_gap_fraction": float(subset["gaps_nearest_vza"].mean()),
                    "n_rows": int(len(subset)),
                    "n_plots": int(subset["plot_id"].nunique()),
                    "weeks": ",".join(str(week) for week in sorted(subset["week"].unique())),
                }
            )
    return pd.DataFrame(rows).sort_values(["band_name", "gap_group", "vza_midpoint"])


def gap_tercile_observations(joined: pd.DataFrame, weeks: tuple[int, ...]) -> pd.DataFrame:
    """Return row-level closed/open canopy observations for boxplots."""
    data = joined[
        (joined["band_name"].isin(BANDS_FOR_PAPER)) & (joined["week"].isin(weeks))
    ].dropna(subset=["relative_contrast", "gaps_nearest_vza", "vza_midpoint"])
    rows = []
    labels = ["closed canopy", "mid canopy", "open canopy"]
    for (band, vza_class), group in data.groupby(["band_name", "vza_class"]):
        try:
            group = group.copy()
            group["gap_group"] = pd.qcut(
                group["gaps_nearest_vza"], q=3, labels=labels, duplicates="drop"
            )
        except ValueError:
            continue
        rows.append(group[group["gap_group"].astype(str).isin(["closed canopy", "open canopy"])])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_stable_canopy_vza_points(joined: pd.DataFrame) -> pd.DataFrame:
    """Summarize contrast using plot-week canopy groups and wider VZA zones.

    The per-VZA tercile plot is useful for exploration, but it can make unstable
    groups at high VZA. This summary assigns canopy openness once per plot-week
    using DIFN, then pools neighboring VZA bins into broader zones.
    """
    data = joined[
        (joined["band_name"].isin(BANDS_FOR_PAPER)) & (joined["week"].isin([5, 6]))
    ].dropna(subset=["relative_contrast", "difn", "vza_midpoint"])
    if data.empty:
        return pd.DataFrame()

    data = data.copy()
    plot_week = data[["plot_id", "week", "difn"]].drop_duplicates()
    labels = ["closed canopy", "mid canopy", "open canopy"]
    plot_week["canopy_group"] = pd.qcut(plot_week["difn"], q=3, labels=labels, duplicates="drop")
    data = data.merge(
        plot_week[["plot_id", "week", "canopy_group"]], on=["plot_id", "week"], how="left"
    )
    data["vza_zone"] = pd.cut(
        data["vza_midpoint"],
        bins=[0, 30, 45, 60],
        labels=["low VZA\n15-30 deg", "mid VZA\n30-45 deg", "high VZA\n45-55 deg"],
        include_lowest=True,
        right=True,
    )
    zone_midpoints = {
        "low VZA\n15-30 deg": 22.5,
        "mid VZA\n30-45 deg": 37.5,
        "high VZA\n45-55 deg": 50.0,
    }

    rows: list[dict[str, object]] = []
    for (band, vza_zone, canopy_group), subset in data.groupby(
        ["band_name", "vza_zone", "canopy_group"], observed=True
    ):
        if subset.empty:
            continue
        rows.append(
            {
                "band_name": band,
                "vza_zone": str(vza_zone),
                "vza_midpoint": zone_midpoints[str(vza_zone)],
                "canopy_group": str(canopy_group),
                "mean_relative_contrast": float(subset["relative_contrast"].mean()),
                "se_relative_contrast": float(
                    subset["relative_contrast"].std(ddof=1) / np.sqrt(len(subset))
                    if len(subset) > 1
                    else 0.0
                ),
                "median_relative_contrast": float(subset["relative_contrast"].median()),
                "q25_relative_contrast": float(subset["relative_contrast"].quantile(0.25)),
                "q75_relative_contrast": float(subset["relative_contrast"].quantile(0.75)),
                "mean_difn": float(subset["difn"].mean()),
                "n_rows": int(len(subset)),
                "n_plot_weeks": int(subset[["plot_id", "week"]].drop_duplicates().shape[0]),
                "n_plots": int(subset["plot_id"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["band_name", "canopy_group", "vza_midpoint"])


def save_figures(joined: pd.DataFrame, model_delta: pd.DataFrame, output_root: Path) -> list[Path]:
    t0 = time.perf_counter()
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    canopy = joined[["week", "plot_id", "cult", "trt", "lai", "difn", "acf"]].drop_duplicates()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for treatment, style in [("trt", "-"), ("no_trt", "--")]:
        for cultivar, color in [("aluco", "#0072B2"), ("capone", "#D55E00")]:
            data = canopy[(canopy["trt"] == treatment) & (canopy["cult"] == cultivar)]
            summary = data.groupby("week", as_index=False).agg(
                lai=("lai", "mean"), difn=("difn", "mean")
            )
            label = f"{cultivar} {treatment}"
            axes[0].plot(
                summary["week"],
                summary["lai"],
                linestyle=style,
                color=color,
                marker="o",
                label=label,
            )
            axes[1].plot(
                summary["week"],
                summary["difn"],
                linestyle=style,
                color=color,
                marker="o",
                label=label,
            )
    axes[0].set_title("LAI progression")
    axes[0].set_ylabel("LAI")
    axes[1].set_title("Diffuse non-interceptance / openness")
    axes[1].set_ylabel("DIFN")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#E3E6E8")
        axis.set_xlabel("Week")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path = fig_dir / "lai_difn_progression_by_cultivar_treatment_2024.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    filtered = model_delta[model_delta["band_name"].isin(BANDS_FOR_PAPER)].copy()
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    x_labels = ["M1_add_lai", "M2_add_difn_acf", "M3_add_angle_gap", "M4_gap_lai_interactions"]
    x = np.arange(len(x_labels))
    width = 0.36
    for idx, band in enumerate(BANDS_FOR_PAPER):
        data = filtered[filtered["band_name"] == band].set_index("model").reindex(x_labels)
        axis.bar(x + (idx - 0.5) * width, data["delta_adj_r2_vs_M0"], width=width, label=band)
    axis.axhline(0, color="#4D4D4D", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(["+LAI", "+DIFN/ACF", "+angle gap", "+interactions"], rotation=0)
    axis.set_ylabel("Delta adjusted R2 vs VZA-only baseline")
    axis.set_title("Do canopy variables explain angular reflectance contrast?")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E3E6E8")
    axis.legend(frameon=False)
    fig.tight_layout()
    path = fig_dir / "canopy_variable_model_gain_nir_rededge_2024.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    gap_curves = summarize_gap_group_curves(joined)
    if not gap_curves.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
        group_styles = {
            "closed canopy\nlow gap": {
                "color": "#176B6B",
                "label": "closed canopy, low gap",
                "marker": "o",
            },
            "open canopy\nhigh gap": {
                "color": "#D55E00",
                "label": "open canopy, high gap",
                "marker": "s",
            },
        }
        for axis, band in zip(axes, BANDS_FOR_PAPER, strict=True):
            band_df = gap_curves[gap_curves["band_name"] == band]
            for group_name, style in group_styles.items():
                group_df = band_df[band_df["gap_group"] == group_name].sort_values("vza_midpoint")
                if group_df.empty:
                    continue
                x_values = group_df["vza_midpoint"].to_numpy(dtype=float)
                median = group_df["median_relative_contrast"].to_numpy(dtype=float)
                q25 = group_df["q25_relative_contrast"].to_numpy(dtype=float)
                q75 = group_df["q75_relative_contrast"].to_numpy(dtype=float)
                axis.plot(
                    x_values,
                    median,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=2.5,
                    label=style["label"],
                )
                axis.fill_between(x_values, q25, q75, color=style["color"], alpha=0.16, linewidth=0)
            axis.axhline(0, color="#4D4D4D", linewidth=0.9)
            axis.set_title(f"{band}: off-nadir contrast by canopy gap")
            axis.set_xlabel("View zenith angle midpoint (deg)")
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#E3E6E8")
            axis.set_axisbelow(True)
        axes[0].set_ylabel("Relative contrast vs nadir\n(0 = same as nadir)")
        axes[1].legend(frameon=False, loc="upper left")
        fig.suptitle("Does canopy openness change the VZA reflectance response? Weeks 5-6", y=1.03)
        fig.text(
            0.5,
            -0.02,
            "Low/high gap groups are defined within each VZA bin using the LAI instrument gap fraction nearest that viewing angle. "
            "Lines show medians; shaded bands show IQR.",
            ha="center",
            fontsize=9,
            color="#4D4D4D",
        )
        fig.tight_layout()
        path = fig_dir / "vza_contrast_low_vs_high_canopy_gap_weeks5_6_2024.png"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        pivot = gap_curves.pivot_table(
            index=["band_name", "vza_class", "vza_midpoint"],
            columns="gap_group",
            values="median_relative_contrast",
            aggfunc="first",
        ).reset_index()
        closed_col = "closed canopy\nlow gap"
        open_col = "open canopy\nhigh gap"
        if closed_col in pivot.columns and open_col in pivot.columns:
            pivot["open_minus_closed"] = pivot[open_col] - pivot[closed_col]
            fig, axis = plt.subplots(figsize=(8.8, 5.2))
            for band, color, marker in [("Red edge", "#0072B2", "o"), ("NIR", "#D55E00", "s")]:
                band_df = pivot[pivot["band_name"] == band].sort_values("vza_midpoint")
                if band_df.empty:
                    continue
                axis.plot(
                    band_df["vza_midpoint"],
                    band_df["open_minus_closed"],
                    marker=marker,
                    linewidth=2.6,
                    color=color,
                    label=band,
                )
            axis.axhline(0, color="#4D4D4D", linewidth=1.0)
            axis.fill_between(
                [pivot["vza_midpoint"].min(), pivot["vza_midpoint"].max()], 0, 0.0, color="none"
            )
            axis.set_title("Effect of canopy gaps on VZA reflectance contrast, weeks 5-6")
            axis.set_xlabel("View zenith angle midpoint (deg)")
            axis.set_ylabel("Open canopy minus closed canopy\nrelative contrast vs nadir")
            axis.text(
                0.02,
                0.96,
                "Above zero: open/gappy canopy has stronger off-nadir reflectance contrast\n"
                "Below zero: closed canopy has stronger off-nadir contrast",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
                color="#4D4D4D",
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#E3E6E8")
            axis.legend(frameon=False, loc="upper right")
            fig.tight_layout()
            path = fig_dir / "vza_contrast_open_minus_closed_gap_weeks5_6_2024.png"
            fig.savefig(path, dpi=240, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

    gap_tercile_configs = [
        ((5, 6), "weeks 5-6", "weeks5_6"),
        ((3, 5, 6), "weeks 3, 5, and 6", "weeks3_5_6"),
        ((3, 4, 5, 6), "weeks 3, 4, 5, and 6", "weeks3_4_5_6"),
    ]
    for weeks, week_label, week_slug in gap_tercile_configs:
        gap_terciles = summarize_gap_tercile_points(joined, weeks=weeks)
        if gap_terciles.empty:
            continue
        style = {
            "closed canopy": {"color": "#176B6B", "marker": "o"},
            "open canopy": {"color": "#D55E00", "marker": "s"},
        }
        for band in BANDS_FOR_PAPER:
            data = gap_terciles[gap_terciles["band_name"] == band]
            if data.empty:
                continue
            fig, axis = plt.subplots(figsize=(8.2, 5.2))
            for gap_group in ["closed canopy", "open canopy"]:
                group = data[data["gap_group"] == gap_group].sort_values("vza_midpoint")
                if group.empty:
                    continue
                axis.errorbar(
                    group["vza_midpoint"],
                    group["mean_relative_contrast"],
                    yerr=group["se_relative_contrast"],
                    color=style[gap_group]["color"],
                    marker=style[gap_group]["marker"],
                    linewidth=2.2,
                    markersize=7,
                    capsize=3,
                    label=gap_group,
                )
            axis.axhline(0, color="#4D4D4D", linewidth=0.9)
            axis.set_title(f"{band}: averaged VZA contrast by canopy openness, {week_label}")
            axis.set_xlabel("View zenith angle midpoint (deg)")
            axis.set_ylabel("Mean relative contrast vs nadir\n(error bars = SE)")
            axis.text(
                0.02,
                0.97,
                "Each VZA bin is split by LAI-instrument gap fraction nearest that VZA.\n"
                f"Dataset: {week_label}; only lower and upper terciles are shown.",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
                color="#4D4D4D",
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#E3E6E8")
            axis.legend(frameon=False, loc="upper right")
            fig.tight_layout()
            path = (
                fig_dir
                / f"vza_relative_contrast_gap_tercile_means_{band.lower().replace(' ', '_')}_{week_slug}_2024.png"
            )
            fig.savefig(path, dpi=240, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

    box_weeks = (3, 4, 5, 6)
    box_data = gap_tercile_observations(joined, weeks=box_weeks)
    if not box_data.empty:
        style = {
            "closed canopy": {"color": "#176B6B", "offset": -0.75},
            "open canopy": {"color": "#D55E00", "offset": 0.75},
        }
        for band in BANDS_FOR_PAPER:
            data = box_data[box_data["band_name"] == band]
            if data.empty:
                continue
            vza_midpoints = sorted(data["vza_midpoint"].dropna().unique())
            fig, axis = plt.subplots(figsize=(9.8, 5.8))
            legend_handles = []
            legend_labels = []
            for gap_group, spec in style.items():
                values = []
                positions = []
                for midpoint in vza_midpoints:
                    subset = data[
                        (data["gap_group"].astype(str) == gap_group)
                        & (data["vza_midpoint"] == midpoint)
                    ]["relative_contrast"].dropna()
                    if subset.empty:
                        continue
                    values.append(subset.to_numpy(dtype=float))
                    positions.append(midpoint + spec["offset"])
                if not values:
                    continue
                box = axis.boxplot(
                    values,
                    positions=positions,
                    widths=1.2,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    medianprops={"color": "#111111", "linewidth": 1.4},
                    whiskerprops={"color": spec["color"], "linewidth": 1.2},
                    capprops={"color": spec["color"], "linewidth": 1.2},
                    boxprops={"edgecolor": spec["color"], "linewidth": 1.2},
                )
                for patch in box["boxes"]:
                    patch.set_facecolor(spec["color"])
                    patch.set_alpha(0.32)
                legend_handles.append(box["boxes"][0])
                legend_labels.append(gap_group)
            axis.axhline(0, color="#4D4D4D", linewidth=0.9)
            axis.set_xticks(vza_midpoints)
            axis.set_xticklabels([f"{value:g}" for value in vza_midpoints])
            axis.set_title(f"{band}: VZA contrast distribution by canopy openness, weeks 3-6")
            axis.set_xlabel("View zenith angle midpoint (deg)")
            axis.set_ylabel("Relative contrast vs nadir")
            axis.text(
                0.02,
                0.97,
                "Boxes show row-level plot-week observations after equal-VZA pixel sampling.\n"
                "Closed/open groups are the lower/upper gap-fraction terciles within each VZA bin.",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
                color="#4D4D4D",
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#E3E6E8")
            if legend_handles:
                axis.legend(legend_handles, legend_labels, frameon=False, loc="upper right")
            fig.tight_layout()
            path = (
                fig_dir
                / f"vza_relative_contrast_gap_tercile_boxes_{band.lower().replace(' ', '_')}_weeks3_4_5_6_2024.png"
            )
            fig.savefig(path, dpi=240, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

    stable_canopy = summarize_stable_canopy_vza_points(joined)
    if not stable_canopy.empty:
        style = {
            "closed canopy": {"color": "#176B6B", "marker": "o"},
            "open canopy": {"color": "#D55E00", "marker": "s"},
        }
        for band in BANDS_FOR_PAPER:
            data = stable_canopy[stable_canopy["band_name"] == band]
            if data.empty:
                continue
            fig, axis = plt.subplots(figsize=(8.5, 5.2))
            for canopy_group in ["closed canopy", "open canopy"]:
                group = data[data["canopy_group"] == canopy_group].sort_values("vza_midpoint")
                if group.empty:
                    continue
                axis.errorbar(
                    group["vza_midpoint"],
                    group["median_relative_contrast"],
                    yerr=[
                        group["median_relative_contrast"] - group["q25_relative_contrast"],
                        group["q75_relative_contrast"] - group["median_relative_contrast"],
                    ],
                    color=style[canopy_group]["color"],
                    marker=style[canopy_group]["marker"],
                    linewidth=2.4,
                    markersize=7,
                    capsize=3,
                    label=f"{canopy_group} (median, IQR)",
                )
            axis.axhline(0, color="#4D4D4D", linewidth=0.9)
            axis.set_xticks([22.5, 37.5, 50.0])
            axis.set_xticklabels(["low\n15-30 deg", "mid\n30-45 deg", "high\n45-55 deg"])
            axis.set_title(f"{band}: VZA contrast by stable canopy openness groups")
            axis.set_xlabel("Pooled view zenith angle zone")
            axis.set_ylabel("Median relative contrast vs nadir\n(error bars = IQR)")
            axis.text(
                0.02,
                0.97,
                "Canopy groups are assigned once per plot-week from DIFN terciles.\n"
                "VZA bins are pooled to avoid unstable high-angle small samples.\n"
                "Sample size per canopy group: n=48 rows at low/mid VZA; n=28-32 at high VZA.",
                transform=axis.transAxes,
                va="top",
                fontsize=9,
                color="#4D4D4D",
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#E3E6E8")
            axis.legend(
                frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8
            )
            fig.tight_layout(rect=[0, 0.08, 1, 1])
            path = (
                fig_dir
                / f"vza_relative_contrast_stable_canopy_groups_{band.lower().replace(' ', '_')}_weeks5_6_2024.png"
            )
            fig.savefig(path, dpi=240, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

    phase("save_figures", t0)
    return paths


def write_report(
    output_root: Path,
    log_path: Path,
    raw_path: Path,
    lai_path: Path,
    joined_path: Path,
    model_path: Path,
    delta_path: Path,
    term_path: Path,
    corr_path: Path,
    figure_paths: list[Path],
    model_delta: pd.DataFrame,
    term_summary: pd.DataFrame,
) -> Path:
    t0 = time.perf_counter()
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_ROOT / "analyze_lai_canopy_vza_reflectance_summary.md"
    paper_delta = model_delta[model_delta["band_name"].isin(BANDS_FOR_PAPER)].copy()
    for col in ["delta_adj_r2_vs_M0", "delta_aic_vs_M0", "delta_bic_vs_M0"]:
        paper_delta[col] = paper_delta[col].round(4)
    paper_terms = term_summary[term_summary["band_name"].isin(BANDS_FOR_PAPER)].copy()
    for col in ["coef", "p_cluster", "conf_low", "conf_high"]:
        paper_terms[col] = paper_terms[col].round(4)

    lines = [
        "# LAI Canopy Variables x VZA Reflectance EDA",
        "",
        "## Results: model gain over VZA baseline",
        "",
        markdown_table(paper_delta),
        "",
        "## Results: canopy terms in M3",
        "",
        markdown_table(paper_terms),
        "",
        "## Interpretation",
        "",
        (
            "The LAI instrument data are useful because the raw TXT files provide independent canopy structure variables "
            "(`LAI`, `DIFN`, `ACF`, and angle-ring `GAPS`/`AVGTRANS`). In this EDA, the useful signal is not a direct "
            "side-leaf-orientation measurement; it is canopy density/openness that can help explain why VZA-binned "
            "reflectance differs from nadir."
        ),
        "",
        (
            "If adding `LAI`, `DIFN/ACF`, or nearest-VZA gap fraction improves adjusted R2 and lowers AIC/BIC, then these "
            "variables help explain the VZA reflectance distribution. If only the interaction model improves weakly or "
            "BIC worsens, the canopy variables are explanatory context but not strong standalone evidence."
        ),
        "",
        "## Outputs",
        "",
        f"- Raw parsed LAI instrument table: `{raw_path}`",
        f"- Plot-week LAI canopy table: `{lai_path}`",
        f"- Joined canopy-reflectance table: `{joined_path}`",
        f"- Model comparison: `{model_path}`",
        f"- Model deltas: `{delta_path}`",
        f"- Canopy model terms: `{term_path}`",
        f"- Correlations: `{corr_path}`",
    ]
    lines.extend([f"- Figure: `{path}`" for path in figure_paths])
    lines.extend(
        [
            f"- Log: `{log_path}`",
            "",
            "## Reproducibility",
            "",
            f"- LAI raw root: `{LAI_RAW_ROOT}`",
            f"- Reflectance source: `{CONTRAST_SOURCE}`",
            f"- Polygon metadata: `{POLYGON_PATH}`",
            "- Response variable for models: `relative_contrast` versus nadir",
            "- Baseline model: `relative_contrast ~ C(week) + C(cult) + C(trt) + C(vza_class)`",
            "- Canopy angle-ring matching: each reflectance VZA midpoint is matched to the nearest LAI instrument ring in `[7, 23, 38, 53, 68]` degrees.",
            "- Random seed: not used; deterministic EDA.",
        ]
    )
    report_path.write_text("\n".join(lines))
    phase("write_report", t0)
    return report_path


def main() -> None:
    args = parse_args()
    log_path = configure_logging()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "results").mkdir(parents=True, exist_ok=True)

    raw, _ = load_lai_raw(args.lai_raw_root, args.polygon_path)
    lai_plot_week = aggregate_lai(raw)
    joined = build_join(args.contrast_source, lai_plot_week)
    corr = correlation_summary(joined)
    model_summary, term_summary = fit_models(joined)
    delta = model_deltas(model_summary)
    figure_paths = save_figures(joined, delta, args.output_root)

    raw_path = args.output_root / "results/lai_instrument_raw_parsed_2024.csv"
    lai_path = args.output_root / "results/lai_canopy_plot_week_2024.csv"
    joined_path = args.output_root / "results/lai_canopy_vza_reflectance_join_2024.csv"
    corr_path = args.output_root / "results/lai_canopy_vza_correlations_2024.csv"
    model_path = args.output_root / "results/lai_canopy_vza_model_comparison_2024.csv"
    delta_path = args.output_root / "results/lai_canopy_vza_model_deltas_2024.csv"
    term_path = args.output_root / "results/lai_canopy_vza_model_terms_2024.csv"

    raw.to_csv(raw_path, index=False)
    lai_plot_week.to_csv(lai_path, index=False)
    joined.to_csv(joined_path, index=False)
    corr.to_csv(corr_path, index=False)
    model_summary.to_csv(model_path, index=False)
    delta.to_csv(delta_path, index=False)
    term_summary.to_csv(term_path, index=False)

    logging.info("[I/O] wrote %s", raw_path)
    logging.info("[I/O] wrote %s", lai_path)
    logging.info("[I/O] wrote %s", joined_path)
    logging.info("[I/O] wrote %s", model_path)

    report_path = write_report(
        args.output_root,
        log_path,
        raw_path,
        lai_path,
        joined_path,
        model_path,
        delta_path,
        term_path,
        corr_path,
        figure_paths,
        delta,
        term_summary,
    )
    logging.info("[I/O] wrote %s", report_path)


if __name__ == "__main__":
    main()
