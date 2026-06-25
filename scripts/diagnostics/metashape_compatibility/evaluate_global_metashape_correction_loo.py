#!/usr/bin/env python3
"""Leave-one-week-out test for global Metashape compatibility factors."""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def setup_logging() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("outputs/logs") / f"evaluate_global_metashape_correction_loo_{timestamp}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path)],
        force=True,
    )
    return path


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["ratio_reference_over_custom"] = float(row["ratio_reference_over_custom"])
    return rows


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def train_factors(rows: list[dict], heldout_week: str, mode: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["mode"] != mode or row["week"] == heldout_week:
            continue
        ratio = row["ratio_reference_over_custom"]
        if ratio > 0:
            grouped[row["band"]].append(ratio)
    return {band: median(values) for band, values in grouped.items() if values}


def evaluate(rows: list[dict], modes: list[str]) -> tuple[list[dict], list[dict]]:
    weeks = sorted({row["week"] for row in rows})
    detail = []
    summary = []
    for mode in modes:
        for heldout_week in weeks:
            factors = train_factors(rows, heldout_week, mode)
            for row in rows:
                if row["mode"] != mode or row["week"] != heldout_week:
                    continue
                factor = factors.get(row["band"])
                if factor is None:
                    continue
                corrected_ratio = row["ratio_reference_over_custom"] / factor
                detail.append(
                    {
                        "mode": mode,
                        "heldout_week": heldout_week,
                        "capture_id": row["capture_id"],
                        "band": row["band"],
                        "train_factor": factor,
                        "raw_ratio_reference_over_custom": row["ratio_reference_over_custom"],
                        "corrected_ratio_reference_over_custom": corrected_ratio,
                        "abs_percent_error_after_correction": abs(corrected_ratio - 1.0) * 100.0,
                    }
                )
            for band in sorted({row["band"] for row in detail if row["mode"] == mode and row["heldout_week"] == heldout_week}):
                subset = [
                    row
                    for row in detail
                    if row["mode"] == mode and row["heldout_week"] == heldout_week and row["band"] == band
                ]
                errors = [float(row["abs_percent_error_after_correction"]) for row in subset]
                corrected = [float(row["corrected_ratio_reference_over_custom"]) for row in subset]
                summary.append(
                    {
                        "mode": mode,
                        "heldout_week": heldout_week,
                        "band": band,
                        "n_captures": len(subset),
                        "train_factor": factors.get(band, float("nan")),
                        "corrected_ratio_median": median(corrected),
                        "median_abs_percent_error_after_correction": median(errors),
                        "passes_20pct": median(errors) <= 20.0,
                    }
                )
    return detail, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: list[dict], outputs: list[Path], args: argparse.Namespace, log_path: Path) -> None:
    mode_errors: dict[str, list[float]] = defaultdict(list)
    mode_passes: dict[str, int] = defaultdict(int)
    mode_counts: dict[str, int] = defaultdict(int)
    for row in summary:
        mode = row["mode"]
        mode_errors[mode].append(float(row["median_abs_percent_error_after_correction"]))
        mode_counts[mode] += 1
        mode_passes[mode] += int(row["passes_20pct"] in {True, "True"})
    lines = [
        "| mode | median_abs_percent_error | passing_week_band_checks |",
        "| --- | --- | --- |",
    ]
    for mode in sorted(mode_errors):
        lines.append(
            f"| {mode} | {median(mode_errors[mode]):.3f} | {mode_passes[mode]}/{mode_counts[mode]} |"
        )
    text = f"""## Results: Leave-One-Week-Out Global Metashape Correction

{chr(10).join(lines)}

**Interpretation**: This tests whether per-band correction factors trained on available Metashape weeks transfer to a held-out week. A passing result requires corrected reference/custom ratios close to 1.0 without using the held-out week to estimate factors.

**Outputs**
{chr(10).join(f"- `{output}`" for output in outputs + [log_path])}

**Reproducibility**
- input_detail_csv: `{args.input_detail_csv}`
- modes: `{','.join(args.modes)}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-detail-csv",
        type=Path,
        default=Path("outputs/results/micasense_radiometry_mode_transferability_detail.csv"),
    )
    parser.add_argument("--modes", nargs="+", default=["micasense_dls", "micasense_panel", "panel_dls_tie"])
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("outputs/results/global_metashape_correction_loo"),
    )
    args = parser.parse_args()
    log_path = setup_logging()
    t0 = time.perf_counter()
    rows = read_rows(args.input_detail_csv)
    logging.info("[PHASE] read input detail: %.1fs", time.perf_counter() - t0)
    t0 = time.perf_counter()
    detail, summary = evaluate(rows, args.modes)
    logging.info("[PHASE] evaluate LOO corrections: %.1fs", time.perf_counter() - t0)
    detail_csv = args.out_prefix.with_name(args.out_prefix.name + "_detail.csv")
    summary_csv = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    report_path = Path("outputs/reports") / f"{args.out_prefix.name}_summary.md"
    write_csv(detail_csv, detail)
    write_csv(summary_csv, summary)
    write_report(report_path, summary, [detail_csv, summary_csv, report_path], args, log_path)
    logging.info("[DONE] outputs=%s", [detail_csv, summary_csv, report_path, log_path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
