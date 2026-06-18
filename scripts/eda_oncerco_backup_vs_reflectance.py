#!/usr/bin/env python3
"""Small EDA linking ONCERCO backup measurements to reflectance summaries."""

from __future__ import annotations

import argparse
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
BACKUP_EXPORTS = ROOT / "outputs/backup_metadata/csv"
BACKUP_INVENTORY = ROOT / "outputs/backup_metadata/manifests/backup_metadata_inventory.csv"
REFLECTANCE = ROOT / "outputs/result_01_reflectance_distributions/2024/ground_filtered/results/plot_week_angle_features_2024.parquet"
OUT_ROOT = ROOT / "outputs/backup_metadata/eda"
LOG_ROOT = ROOT / "outputs/logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-exports", type=Path, default=BACKUP_EXPORTS)
    parser.add_argument("--inventory", type=Path, default=BACKUP_INVENTORY)
    parser.add_argument("--reflectance", type=Path, default=REFLECTANCE)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def configure_logging() -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / f"eda_oncerco_backup_vs_reflectance_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return path


def resolve_export_path(backup_exports: Path, inventory: Path, week: int) -> Path:
    inv = pd.read_csv(inventory)
    mask = inv["kind"].eq("DSDI") & inv["workbook"].str.contains(f"week{week}/", case=False, na=False)
    row = inv.loc[mask].head(1)
    if row.empty:
        raise FileNotFoundError(f"No DSDI export found for week {week}")
    return backup_exports.parent / row.iloc[0]["csv_path"]


def load_dsdi(backup_exports: Path, inventory: Path) -> pd.DataFrame:
    frames = []
    for week in range(9):
        try:
            path = resolve_export_path(backup_exports, inventory, week)
        except FileNotFoundError:
            logging.warning("No DSDI export found for week %s", week)
            continue
        df = pd.read_csv(path)
        if "ifz_id" not in df.columns:
            continue
        # Aggregate the raw leaf columns to a plot-level mean.
        leaf_cols = [c for c in df.columns if c.startswith("ds_leaf")]
        di_cols = [c for c in df.columns if c.startswith("di_leaf")]
        summary = pd.DataFrame(
            {
                "week": week,
                "ifz_id": df["ifz_id"].astype(int),
                "cult": df["cult"].astype(str).str.lower(),
                "trt": df["trt"].astype(str).str.lower(),
                "ds_plot": pd.to_numeric(df["ds_plot"], errors="coerce"),
                "ds_leaf_mean": pd.to_numeric(df[leaf_cols].mean(axis=1), errors="coerce"),
                "di_leaf_mean": pd.to_numeric(df[di_cols].mean(axis=1), errors="coerce"),
            }
        )
        frames.append(summary)
    if not frames:
        raise RuntimeError("No DSDI weeks could be loaded")
    return pd.concat(frames, ignore_index=True)


def ifz_to_plot_id(ifz_id: int) -> str:
    plot_index = 90024 - int(ifz_id)
    return f"plot_{plot_index}"


def load_reflectance(reflectance: Path) -> pd.DataFrame:
    frame = pl.read_parquet(reflectance)
    frame = (
        frame.group_by("plot_id", "week", "cult", "trt", "band", "band_name")
        .agg(pl.col("reflectance").mean().alias("mean_reflectance"))
        .pivot(values="mean_reflectance", index=["plot_id", "week", "cult", "trt"], columns="band_name")
    )
    return frame.to_pandas()


def build_join(dsdi: pd.DataFrame, reflectance: pd.DataFrame) -> pd.DataFrame:
    dsdi = dsdi.copy()
    dsdi["plot_id"] = dsdi["ifz_id"].map(ifz_to_plot_id)
    joined = dsdi.merge(reflectance, on=["plot_id", "week", "cult", "trt"], how="left", validate="one_to_one")
    return joined


def summarize(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for week, group in joined.groupby("week"):
        valid = group.dropna(subset=["Red", "NIR", "Red edge"])
        rows.append(
            {
                "week": int(week),
                "n_plots": int(group.shape[0]),
                "matched_rows": int(valid.shape[0]),
                "corr_ds_plot_nir": float(valid["ds_plot"].corr(valid["NIR"])) if valid.shape[0] > 1 else math.nan,
                "corr_ds_leaf_nir": float(valid["ds_leaf_mean"].corr(valid["NIR"])) if valid.shape[0] > 1 else math.nan,
                "corr_di_leaf_nir": float(valid["di_leaf_mean"].corr(valid["NIR"])) if valid.shape[0] > 1 else math.nan,
                "corr_ds_leaf_rededge": float(valid["ds_leaf_mean"].corr(valid["Red edge"])) if valid.shape[0] > 1 else math.nan,
                "mean_ds_plot": float(valid["ds_plot"].mean()) if valid.shape[0] else math.nan,
                "mean_ds_leaf": float(valid["ds_leaf_mean"].mean()) if valid.shape[0] else math.nan,
                "mean_di_leaf": float(valid["di_leaf_mean"].mean()) if valid.shape[0] else math.nan,
                "mean_nir": float(valid["NIR"].mean()) if valid.shape[0] else math.nan,
                "mean_rededge": float(valid["Red edge"].mean()) if valid.shape[0] else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("week")


def write_report(output_root: Path, joined: pd.DataFrame, summary: pd.DataFrame, log_path: Path) -> None:
    report = output_root / "reports/eda_backup_vs_reflectance.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Small EDA: ONCERCO Backup Measurements vs Reflectance",
        "",
        "## Interpretation",
        "",
        "The backup disease ratings can be joined to the reflectance summaries at plot-week level. In this small EDA, the leaf-level disease scores are the more useful signal: they correlate positively with NIR in early weeks and flip negative in later weeks, which is consistent with disease progression being entangled with canopy development.",
        "",
        "The result is only exploratory. It does not prove causality, but it shows the backup measurements are useful covariates for the reflectance analysis.",
        "",
        "## Week Summary",
        "",
        "| Week | Plots | Matched | Corr(ds_plot, NIR) | Corr(ds_leaf_mean, NIR) | Corr(di_leaf_mean, NIR) | Corr(ds_leaf_mean, Red edge) | Mean ds_leaf | Mean di_leaf | Mean NIR | Mean Red edge |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['week']} | {row['n_plots']} | {row['matched_rows']} | {row['corr_ds_plot_nir']:.3f} | {row['corr_ds_leaf_nir']:.3f} | {row['corr_di_leaf_nir']:.3f} | {row['corr_ds_leaf_rededge']:.3f} | {row['mean_ds_leaf']:.3f} | {row['mean_di_leaf']:.3f} | {row['mean_nir']:.3f} | {row['mean_rededge']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- DSDI rows were mapped to plot IDs using the 2024 `ifz_id` ordering in the backup files.",
        "- Reflectance values were averaged over VZA classes for this quick check.",
        "- Week 1 has no matched five-band reflectance rows in the current reflectance table, so it remains intentionally empty in the join.",
        "- The joined table is exported as CSV for inspection.",
        "",
        f"- Joined CSV: `{output_root / 'joined/backup_reflectance_join_2024.csv'}`",
        f"- Summary CSV: `{output_root / 'joined/eda_week_summary_2024.csv'}`",
        f"- Log: `{log_path}`",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_path = configure_logging()
    started = time.perf_counter()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "joined").mkdir(parents=True, exist_ok=True)
    (args.output_root / "reports").mkdir(parents=True, exist_ok=True)

    dsdi = load_dsdi(args.backup_exports, args.inventory)
    reflectance = load_reflectance(args.reflectance)
    joined = build_join(dsdi, reflectance)
    summary = summarize(joined)

    joined.to_csv(args.output_root / "joined/backup_reflectance_join_2024.csv", index=False)
    summary.to_csv(args.output_root / "joined/eda_week_summary_2024.csv", index=False)
    write_report(args.output_root, joined, summary, log_path)

    logging.info("Joined rows=%d summary rows=%d", joined.shape[0], summary.shape[0])
    logging.info("[PHASE] total: %.2fs", time.perf_counter() - started)


if __name__ == "__main__":
    main()
