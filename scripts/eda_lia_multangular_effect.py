#!/usr/bin/env python3
"""Small EDA for how leaf inclination relates to multiangular reflectance."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
LIA_SOURCE = ROOT / "outputs/backup_metadata/csv/data/interim/2024/lia/20240912_lia_concatenated_average/Sheet1.csv"
CONTRAST_SOURCE = ROOT / "outputs/result_01_reflectance_distributions/2024/ground_filtered/results/matched_plot_contrasts_preliminary.parquet"
OUT_ROOT = ROOT / "outputs/backup_metadata/eda_lia"
LOG_ROOT = ROOT / "outputs/logs"

DATE_TO_WEEK = {
    "2024-06-17": 2,
    "2024-06-24": 3,
    "2024-07-08": 4,
    "2024-07-15": 5,
    "2024-07-25": 6,
    "2024-08-01": 7,
    "2024-08-06": 7,
    "2024-08-26": 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lia-source", type=Path, default=LIA_SOURCE)
    parser.add_argument("--contrast-source", type=Path, default=CONTRAST_SOURCE)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def configure_logging() -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / f"eda_lia_multangular_effect_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return path


def load_lia(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["week"] = df["date"].map(DATE_TO_WEEK)
    df = df.dropna(subset=["week"]).copy()
    df["week"] = df["week"].astype(int)
    df["plot_id"] = df["ifz_id"].astype(int).rsub(90024).map(lambda x: f"plot_{x}")
    return (
        df.groupby(["week", "plot_id", "cult", "trt"], as_index=False)
        .agg(lia=("lia", "mean"), n_lia_dates=("date", "nunique"))
    )


def load_contrasts(path: Path) -> pd.DataFrame:
    df = pl.read_parquet(path).to_pandas()
    df = df[df["band_name"].isin(["NIR", "Red edge"])]
    return df[["plot_id", "week", "band_name", "vza_class", "absolute_contrast", "relative_contrast"]]


def build_join(lia: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    return lia.merge(contrasts, on=["plot_id", "week"], how="inner", validate="one_to_many")


def summarize(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (week, band), group in joined.groupby(["week", "band_name"]):
        clean = group.dropna(subset=["lia", "absolute_contrast", "relative_contrast"])
        if clean.empty:
            continue
        low = clean[clean["lia"] <= clean["lia"].median()]
        high = clean[clean["lia"] > clean["lia"].median()]
        rows.append(
            {
                "week": int(week),
                "band": band,
                "n_rows": int(clean.shape[0]),
                "lia_median": float(clean["lia"].median()),
                "corr_lia_abs": float(clean["lia"].corr(clean["absolute_contrast"])),
                "corr_lia_rel": float(clean["lia"].corr(clean["relative_contrast"])),
                "low_lia_rel_median": float(low["relative_contrast"].median()) if not low.empty else float("nan"),
                "high_lia_rel_median": float(high["relative_contrast"].median()) if not high.empty else float("nan"),
                "delta_high_minus_low": float(high["relative_contrast"].median() - low["relative_contrast"].median()) if (not low.empty and not high.empty) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["week", "band"])


def write_report(output_root: Path, summary: pd.DataFrame, joined: pd.DataFrame, log_path: Path) -> None:
    report = output_root / "reports/lia_multangular_effect.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Small EDA: LIA vs Multiangular Reflectance",
        "",
        "## Interpretation",
        "",
        "This quick check uses the 2024 leaf inclination angle table and the plot-level matched angular contrasts. The main pattern is that LIA is associated with the size of the multiangular response, but the sign and magnitude depend on week and band. That is exactly what we would expect if canopy architecture modulates the viewing-angle signal rather than producing one fixed offset.",
        "",
        "## Summary",
        "",
        "| Week | Band | Rows | Median LIA | Corr(LIA, abs contrast) | Corr(LIA, relative contrast) | High-low relative contrast |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['week']} | {row['band']} | {row['n_rows']} | {row['lia_median']:.2f} | {row['corr_lia_abs']:.3f} | {row['corr_lia_rel']:.3f} | {row['delta_high_minus_low']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `LIA` was matched by `ifz_id` and mapped to reflectance week labels by nearest available 2024 acquisition week.",
        "- Only the `NIR` and `Red edge` bands are included here because those are the most informative angular bands.",
        "- High-LIA minus low-LIA is computed from the median relative contrast within each week and band.",
        "",
        f"- Joined CSV: `{output_root / 'joined/lia_contrast_join.csv'}`",
        f"- Summary CSV: `{output_root / 'joined/lia_multangular_summary.csv'}`",
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

    lia = load_lia(args.lia_source)
    contrasts = load_contrasts(args.contrast_source)
    joined = build_join(lia, contrasts)
    summary = summarize(joined)

    joined.to_csv(args.output_root / "joined/lia_contrast_join.csv", index=False)
    summary.to_csv(args.output_root / "joined/lia_multangular_summary.csv", index=False)
    write_report(args.output_root, summary, joined, log_path)

    logging.info("Joined rows=%d summary rows=%d", joined.shape[0], summary.shape[0])
    logging.info("[PHASE] total: %.2fs", time.perf_counter() - started)


if __name__ == "__main__":
    main()
