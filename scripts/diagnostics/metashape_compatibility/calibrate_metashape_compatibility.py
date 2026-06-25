#!/usr/bin/env python3
"""Estimate and validate Metashape-compatible correction factors for RedEdge-P stacks.

The custom MicaSense pipeline writes uint16 reflectance stacks where
32767 = 1.0 reflectance. Metashape reference stacks are float reflectance.
This script compares matched capture IDs by band distribution, estimates
one frozen multiplicative correction per wavelength-ordered band, and writes
QA tables for paper/project traceability.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio

BAND_NAMES = ("Blue", "Green", "Red", "Red edge", "NIR")
REFLECTANCE_SCALE = 32767.0


def setup_logging(script_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("outputs/logs") / f"{script_name}_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )
    return log_path


def phase(name: str, start: float) -> float:
    elapsed = time.perf_counter() - start
    logging.info("[PHASE] %s: %.1fs", name, elapsed)
    return time.perf_counter()


def capture_id(path: Path) -> str | None:
    match = re.search(r"IMG_(\d+)_6\.tif$", path.name)
    return match.group(1) if match else None


def matched_files(reference_dir: Path, custom_dir: Path) -> list[tuple[str, Path, Path]]:
    references = {
        cid: path
        for path in reference_dir.glob("IMG_*_6.tif")
        if (cid := capture_id(path)) is not None
    }
    customs = {
        cid: path
        for path in custom_dir.glob("IMG_*_6.tif")
        if (cid := capture_id(path)) is not None
    }
    common = sorted(set(references) & set(customs))
    return [(cid, references[cid], customs[cid]) for cid in common]


def read_reflectance(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
        dtype = src.dtypes[0]
    if dtype == "uint16" or float(np.nanmax(data)) > 4.0:
        data = data / REFLECTANCE_SCALE
    return data[:5]


def central_mask(shape: tuple[int, int], trim_fraction: float) -> np.ndarray:
    height, width = shape
    y0 = int(height * trim_fraction)
    y1 = int(height * (1.0 - trim_fraction))
    x0 = int(width * trim_fraction)
    x1 = int(width * (1.0 - trim_fraction))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def valid_values(
    stack: np.ndarray,
    band_index: int,
    trim_fraction: float,
    min_reflectance: float,
    max_reflectance: float,
    vegetation_ndvi_min: float | None,
) -> np.ndarray:
    band = stack[band_index]
    mask = central_mask(band.shape, trim_fraction)
    mask &= np.isfinite(band) & (band > min_reflectance) & (band < max_reflectance)
    if vegetation_ndvi_min is not None:
        red = stack[2]
        nir = stack[4]
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
        mask &= np.isfinite(ndvi) & (ndvi >= vegetation_ndvi_min)
    return band[mask]


def band_stats(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "p99": np.nan,
        }
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def compare_pair(
    cid: str,
    reference_path: Path,
    custom_path: Path,
    trim_fraction: float,
    min_reflectance: float,
    max_reflectance: float,
    vegetation_ndvi_min: float | None,
) -> list[dict[str, float | int | str]]:
    reference = read_reflectance(reference_path)
    custom = read_reflectance(custom_path)
    rows = []
    for band_index, band_name in enumerate(BAND_NAMES):
        ref_values = valid_values(
            reference,
            band_index,
            trim_fraction,
            min_reflectance,
            max_reflectance,
            vegetation_ndvi_min,
        )
        custom_values = valid_values(
            custom,
            band_index,
            trim_fraction,
            min_reflectance,
            max_reflectance,
            vegetation_ndvi_min,
        )
        ref_stats = band_stats(ref_values)
        custom_stats = band_stats(custom_values)
        median_ratio = (
            float(ref_stats["median"]) / float(custom_stats["median"])
            if custom_stats["n"] and float(custom_stats["median"]) > 0
            else np.nan
        )
        mean_ratio = (
            float(ref_stats["mean"]) / float(custom_stats["mean"])
            if custom_stats["n"] and float(custom_stats["mean"]) > 0
            else np.nan
        )
        rows.append(
            {
                "capture_id": cid,
                "band": band_name,
                "reference_file": str(reference_path),
                "custom_file": str(custom_path),
                "reference_n": ref_stats["n"],
                "custom_n": custom_stats["n"],
                "reference_mean": ref_stats["mean"],
                "custom_mean": custom_stats["mean"],
                "reference_median": ref_stats["median"],
                "custom_median": custom_stats["median"],
                "reference_p10": ref_stats["p10"],
                "custom_p10": custom_stats["p10"],
                "reference_p90": ref_stats["p90"],
                "custom_p90": custom_stats["p90"],
                "reference_p99": ref_stats["p99"],
                "custom_p99": custom_stats["p99"],
                "median_ratio_reference_over_custom": median_ratio,
                "mean_ratio_reference_over_custom": mean_ratio,
                "median_abs_error_after_identity": (
                    abs(float(ref_stats["median"]) - float(custom_stats["median"]))
                    if ref_stats["n"] and custom_stats["n"]
                    else np.nan
                ),
            }
        )
    return rows


def aggregate_factors(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    factors = {}
    for band_name in BAND_NAMES:
        ratios = np.array(
            [
                float(row["median_ratio_reference_over_custom"])
                for row in rows
                if row["band"] == band_name
                and np.isfinite(float(row["median_ratio_reference_over_custom"]))
            ],
            dtype="float64",
        )
        if ratios.size == 0:
            raise RuntimeError(f"no finite ratios available for band {band_name}")
        factors[band_name] = float(np.median(ratios))
    return factors


def summarize_rows(
    rows: list[dict[str, float | int | str]], factors: dict[str, float]
) -> list[dict]:
    summary = []
    for band_name in BAND_NAMES:
        band_rows = [row for row in rows if row["band"] == band_name]
        ratios = np.array(
            [
                float(row["median_ratio_reference_over_custom"])
                for row in band_rows
                if np.isfinite(float(row["median_ratio_reference_over_custom"]))
            ],
            dtype="float64",
        )
        corrected_errors = []
        for row in band_rows:
            ref = float(row["reference_median"])
            custom = float(row["custom_median"])
            if np.isfinite(ref) and np.isfinite(custom):
                corrected_errors.append(abs(ref - custom * factors[band_name]))
        corrected_errors_arr = np.array(corrected_errors, dtype="float64")
        summary.append(
            {
                "band": band_name,
                "n_captures": len(band_rows),
                "factor": factors[band_name],
                "ratio_median": float(np.median(ratios)),
                "ratio_iqr": float(np.percentile(ratios, 75) - np.percentile(ratios, 25)),
                "corrected_median_abs_error": float(np.median(corrected_errors_arr)),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    summary_rows: list[dict],
    output_paths: list[Path],
    args: argparse.Namespace,
    log_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = markdown_table(
        summary_rows,
        ["band", "n_captures", "factor", "ratio_iqr", "corrected_median_abs_error"],
    )
    interpretation = (
        "The frozen factors quantify the band-specific transformation needed to make "
        "the custom MicaSense-panel output distribution comparable to the existing "
        "Metashape reflectance exports. These factors should be frozen and applied "
        "unchanged to missing weeks, not re-estimated week by week."
    )
    outputs = "\n".join(f"- `{path}`" for path in output_paths + [log_path])
    text = f"""## Results: Metashape Compatibility Calibration

{table}

**Interpretation**: {interpretation}

**Outputs**
{outputs}

**Reproducibility**
- reference_dir: `{args.reference_dir}`
- custom_dir: `{args.custom_dir}`
- trim_fraction: `{args.trim_fraction}`
- min_reflectance: `{args.min_reflectance}`
- max_reflectance: `{args.max_reflectance}`
- vegetation_ndvi_min: `{args.vegetation_ndvi_min}`
- correction_method: median of per-capture reference/custom median ratios
- random_seed: none
"""
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--custom-dir", type=Path, required=True)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("outputs/results/metashape_compatibility"),
        help="Output prefix without extension.",
    )
    parser.add_argument(
        "--write-correction-json",
        type=Path,
        default=Path("configs/frozen/metashape_compat_rededgep_2025.json"),
    )
    parser.add_argument("--trim-fraction", type=float, default=0.05)
    parser.add_argument("--min-reflectance", type=float, default=1e-6)
    parser.add_argument("--max-reflectance", type=float, default=1.5)
    parser.add_argument(
        "--vegetation-ndvi-min",
        type=float,
        default=None,
        help="Optional NDVI threshold applied independently to each stack.",
    )
    args = parser.parse_args()

    script_name = Path(__file__).stem
    log_path = setup_logging(script_name)
    t0 = time.perf_counter()
    logging.info("[START] reference_dir=%s custom_dir=%s", args.reference_dir, args.custom_dir)

    pairs = matched_files(args.reference_dir, args.custom_dir)
    t0 = phase("match files", t0)
    if not pairs:
        raise FileNotFoundError("no matching IMG_*_6.tif files found")
    logging.info("[MATCH] matched captures=%s", len(pairs))

    rows: list[dict[str, float | int | str]] = []
    read_times = []
    for cid, reference_path, custom_path in pairs:
        read_t0 = time.perf_counter()
        rows.extend(
            compare_pair(
                cid,
                reference_path,
                custom_path,
                args.trim_fraction,
                args.min_reflectance,
                args.max_reflectance,
                args.vegetation_ndvi_min,
            )
        )
        read_times.append(time.perf_counter() - read_t0)
    if read_times:
        logging.info(
            "[PHASE] raster read/compare per file pair: min=%.2fs median=%.2fs mean=%.2fs max=%.2fs",
            float(np.min(read_times)),
            float(np.median(read_times)),
            float(np.mean(read_times)),
            float(np.max(read_times)),
        )
    t0 = phase("compare rasters", t0)

    factors = aggregate_factors(rows)
    summary_rows = summarize_rows(rows, factors)
    t0 = phase("aggregate factors", t0)

    detail_csv = args.out_prefix.with_name(args.out_prefix.name + "_detail.csv")
    summary_csv = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    report_md = Path("outputs/reports") / f"{args.out_prefix.name}_summary.md"
    write_csv(detail_csv, rows)
    write_csv(summary_csv, summary_rows)
    t0 = phase("write CSV outputs", t0)

    correction_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "scripts/calibrate_metashape_compatibility.py",
        "reference_dir": str(args.reference_dir),
        "custom_dir": str(args.custom_dir),
        "matched_capture_count": len(pairs),
        "band_order": list(BAND_NAMES),
        "reflectance_scale": "32767 = 1.0 reflectance for custom uint16 stacks",
        "correction_method": "median per-band reference/custom median ratio",
        "trim_fraction": args.trim_fraction,
        "min_reflectance": args.min_reflectance,
        "max_reflectance": args.max_reflectance,
        "vegetation_ndvi_min": args.vegetation_ndvi_min,
        "band_correction_factors": factors,
    }
    args.write_correction_json.parent.mkdir(parents=True, exist_ok=True)
    args.write_correction_json.write_text(json.dumps(correction_payload, indent=2))
    t0 = phase("write correction JSON", t0)

    write_report(
        report_md,
        summary_rows,
        [detail_csv, summary_csv, args.write_correction_json],
        args,
        log_path,
    )
    phase("write markdown report", t0)
    logging.info(
        "[DONE] outputs=%s",
        [str(detail_csv), str(summary_csv), str(args.write_correction_json), str(report_md)],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
