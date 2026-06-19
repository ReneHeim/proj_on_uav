#!/usr/bin/env python3
"""Quantify angular reflectance curve shape by year, week, cultivar, and band."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
INPUT_TEMPLATE = ROOT / "outputs/result_01_reflectance_distributions/{year}/ground_filtered/results/plot_week_angle_features_{year}.parquet"
OUT_ROOT = ROOT / "outputs/result_03_vza_curve_shape_metrics"
LOG_ROOT = ROOT / "outputs/logs"
REPORTS_ROOT = ROOT / "outputs/reports"
YEARS = [2024, 2025]
SEED = 42


def configure_logging() -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / f"analyze_vza_curve_shape_metrics_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return path


def phase(name: str, started: float) -> None:
    logging.info("[PHASE] %s: %.1fs", name, time.perf_counter() - started)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def load_inputs() -> pd.DataFrame:
    t0 = time.perf_counter()
    frames = []
    read_times = []
    for year in YEARS:
        path = Path(str(INPUT_TEMPLATE).format(year=year))
        ft0 = time.perf_counter()
        frame = pl.read_parquet(path)
        read_times.append(time.perf_counter() - ft0)
        logging.info("[I/O] read rows=%d path=%s", frame.height, path)
        frames.append(frame)
    if read_times:
        arr = np.asarray(read_times)
        logging.info(
            "[I/O] parquet read seconds min=%.3f median=%.3f mean=%.3f max=%.3f",
            arr.min(),
            np.median(arr),
            arr.mean(),
            arr.max(),
        )
    data = pl.concat(frames, how="vertical").to_pandas()
    phase("load_inputs", t0)
    return data


def summarize_curves(data: pd.DataFrame) -> pd.DataFrame:
    t0 = time.perf_counter()
    summary = (
        data.groupby(["year", "week", "cult", "band", "band_name", "vza_class", "vza_midpoint"], as_index=False)
        .agg(
            mean_reflectance=("reflectance", "mean"),
            sd_reflectance=("reflectance", "std"),
            n_observations=("reflectance", "size"),
            n_plots=("plot_id", "nunique"),
        )
        .sort_values(["year", "week", "cult", "band", "vza_midpoint"])
    )
    summary["se_reflectance"] = summary["sd_reflectance"] / np.sqrt(summary["n_observations"])
    summary["se_reflectance"] = summary["se_reflectance"].fillna(0.0)
    phase("summarize_curves", t0)
    return summary


def curve_metrics(curve: pd.DataFrame) -> dict[str, float | int | str]:
    curve = curve.sort_values("vza_midpoint")
    x = curve["vza_midpoint"].to_numpy(dtype=float)
    y = curve["mean_reflectance"].to_numpy(dtype=float)
    n_bins = len(curve)
    result: dict[str, float | int | str] = {
        "n_vza_bins": n_bins,
        "n_observations": int(curve["n_observations"].sum()),
        "n_plots_min": int(curve["n_plots"].min()),
        "n_plots_max": int(curve["n_plots"].max()),
        "mean_reflectance": float(np.mean(y)),
        "curve_amplitude": float(np.max(y) - np.min(y)),
        "peak_vza": float(x[np.argmax(y)]),
        "trough_vza": float(x[np.argmin(y)]),
    }
    low = curve[curve["vza_midpoint"] <= 25]["mean_reflectance"]
    high = curve[curve["vza_midpoint"] >= 45]["mean_reflectance"]
    mid = curve[(curve["vza_midpoint"] >= 30) & (curve["vza_midpoint"] <= 40)]["mean_reflectance"]
    result["high_minus_low_reflectance"] = float(high.mean() - low.mean()) if len(low) and len(high) else np.nan
    result["high_minus_mid_reflectance"] = float(high.mean() - mid.mean()) if len(mid) and len(high) else np.nan

    if n_bins >= 3:
        x_centered = x - x.mean()
        linear = np.polyfit(x_centered, y, deg=1)
        quadratic = np.polyfit(x_centered, y, deg=2)
        y_hat = np.polyval(quadratic, x_centered)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        result["linear_slope_per_degree"] = float(linear[0])
        result["quadratic_curvature"] = float(quadratic[0])
        result["quadratic_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        result["spearman_vza_reflectance"] = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    else:
        result["linear_slope_per_degree"] = np.nan
        result["quadratic_curvature"] = np.nan
        result["quadratic_r2"] = np.nan
        result["spearman_vza_reflectance"] = np.nan
    return result


def build_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    t0 = time.perf_counter()
    rows = []
    group_cols = ["year", "week", "cult", "band", "band_name"]
    for keys, group in summary.groupby(group_cols):
        row = dict(zip(group_cols, keys, strict=True))
        row.update(curve_metrics(group))
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values(["year", "week", "cult", "band"])
    phase("build_metrics", t0)
    return metrics


def write_report(log_path: Path, summary_path: Path, metrics_path: Path, metrics: pd.DataFrame) -> Path:
    t0 = time.perf_counter()
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_ROOT / "vza_curve_shape_metrics_summary.md"
    focus = metrics[metrics["band_name"].isin(["Red edge", "NIR"])].copy()
    focus["curve_amplitude"] = focus["curve_amplitude"].round(4)
    focus["high_minus_low_reflectance"] = focus["high_minus_low_reflectance"].round(4)
    focus["linear_slope_per_degree"] = focus["linear_slope_per_degree"].round(5)
    focus["quadratic_curvature"] = focus["quadratic_curvature"].round(6)
    focus = focus[
        [
            "year",
            "week",
            "cult",
            "band_name",
            "curve_amplitude",
            "high_minus_low_reflectance",
            "linear_slope_per_degree",
            "quadratic_curvature",
            "peak_vza",
            "n_vza_bins",
            "n_plots_min",
        ]
    ]
    lines = [
        "## Results: VZA Curve Shape Metrics",
        "",
        markdown_table(focus.to_string(index=False).splitlines() and focus.head(40).astype(str)),
        "",
        "**Interpretation**: These metrics quantify whether the VZA reflectance curve is flat, increasing, peaked, or U-shaped by week and cultivar. Larger `curve_amplitude` and `high_minus_low_reflectance` indicate stronger angular structure; `quadratic_curvature` captures bending of the curve rather than only a linear trend.",
        "",
        "## Outputs",
        "",
        f"- Curve summaries: `{summary_path}`",
        f"- Shape metrics: `{metrics_path}`",
        f"- Log: `{log_path}`",
        "",
        "## Reproducibility",
        "",
        f"- Inputs: `{INPUT_TEMPLATE}` for years {YEARS}",
        f"- Random seed: {SEED}",
        "- Aggregation: plot-week-angle reflectance means from Result 1 ground-filtered outputs.",
        "- Metrics: OLS polynomial coefficients over VZA-bin mean reflectance; high VZA is >=45 deg and low VZA is <=25 deg.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    phase("write_report", t0)
    return report_path


def main() -> None:
    log_path = configure_logging()
    total = time.perf_counter()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    summary = summarize_curves(data)
    metrics = build_metrics(summary)
    summary_path = OUT_ROOT / "vza_curve_shape_summary_by_year_week_cultivar_band.csv"
    metrics_path = OUT_ROOT / "vza_curve_shape_metrics_by_year_week_cultivar_band.csv"
    summary.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    logging.info("[I/O] wrote %s", summary_path)
    logging.info("[I/O] wrote %s", metrics_path)
    report_path = write_report(log_path, summary_path, metrics_path, metrics)
    logging.info("[I/O] wrote %s", report_path)
    phase("total", total)


if __name__ == "__main__":
    main()
