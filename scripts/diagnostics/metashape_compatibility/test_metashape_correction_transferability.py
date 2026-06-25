#!/usr/bin/env python3
"""Test whether frozen Metashape-compatibility factors transfer across weeks.

This script does not require Metashape outputs for missing weeks. It uses
available Metashape weeks only as an evaluation set: week-specific raw captures
are processed through the same MicaSense panel radiometry path and compared to
their matched Metashape reference distributions before and after applying a
frozen correction JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from skimage.transform import ProjectiveTransform

np.mat = np.asmatrix
sys.path.insert(0, "/home/davidem/miniforge3/lib/python3.13/site-packages")
from micasense.capture import Capture  # noqa: E402

BAND_NAMES = ("Blue", "Green", "Red", "Red edge", "NIR")
METASHAPE_ORDER = (0, 1, 2, 4, 3)
REFLECTANCE_SCALE = 32767.0
DEFAULT_WEEKS = ("week0", "week3", "week5", "week7")


def setup_logging() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("outputs/logs") / f"test_metashape_correction_transferability_{timestamp}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path)],
        force=True,
    )
    return path


def phase(name: str, t0: float) -> float:
    logging.info("[PHASE] %s: %.1fs", name, time.perf_counter() - t0)
    return time.perf_counter()


def capture_id(path: Path) -> str | None:
    match = re.search(r"IMG_(\d+)_([16])\.tif$", path.name)
    return match.group(1) if match else None


def capture_files(seed: Path) -> list[str]:
    return [
        str(seed.with_name(seed.name.replace("_1.tif", f"_{index}.tif"))) for index in range(1, 7)
    ]


def find_reference_dir(processed_root: Path, week: str) -> Path:
    candidates = sorted(
        processed_root.glob(f"{week}/metashape/*/orthophotos"),
        key=lambda path: str(path),
    )
    candidates = [path for path in candidates if any(path.glob("IMG_*_6.tif"))]
    if not candidates:
        raise FileNotFoundError(f"no Metashape orthophoto directory for {week}")
    # Prefer the normally dated product folder when duplicate/copy folders exist.
    candidates = [path for path in candidates if " copy" not in str(path)] or candidates
    return candidates[0]


def raw_seed_map(raw_root: Path, week: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted((raw_root / week / "rededgep").glob("**/IMG_*_1.tif")):
        cid = capture_id(path)
        if cid is None:
            continue
        mapping.setdefault(cid, path)
    return mapping


def reference_map(reference_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(reference_dir.glob("IMG_*_6.tif")):
        cid = capture_id(path)
        if cid is None:
            continue
        mapping.setdefault(cid, path)
    return mapping


def sample_ids(common_ids: list[str], count: int) -> list[str]:
    if count <= 0 or count >= len(common_ids):
        return common_ids
    indexes = np.linspace(0, len(common_ids) - 1, count, dtype=int)
    return [common_ids[index] for index in sorted(set(indexes))]


def load_factors(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    factors = payload.get("band_correction_factors", payload)
    return {band: float(factors[band]) for band in BAND_NAMES}


def panel_irradiance_for_seed(
    seed: Path, cache: dict[Path, tuple[list[float], dict]]
) -> list[float]:
    set_dir = next(parent for parent in seed.parents if parent.name.endswith("SET"))
    if set_dir in cache:
        return cache[set_dir][0]
    panel_seed = set_dir / "000" / "IMG_0000_1.tif"
    if not panel_seed.exists():
        raise FileNotFoundError(f"panel seed missing: {panel_seed}")
    cap = Capture.from_filelist(capture_files(panel_seed))
    detected = int(cap.detect_panels())
    if detected < 5:
        raise RuntimeError(f"panel detection failed for {panel_seed}: {detected}/6 bands")
    irradiance = [float(value) for value in cap.panel_irradiance()]
    cache[set_dir] = (
        irradiance,
        {
            "panel_seed": str(panel_seed),
            "panel_detected_bands": detected,
            "panel_irradiance": irradiance,
        },
    )
    logging.info("[PANEL] %s detected=%s irradiance=%s", set_dir.name, detected, irradiance)
    return irradiance


def compute_custom_stack(seed: Path, irradiance: list[float]) -> np.ndarray:
    cap = Capture.from_filelist(capture_files(seed))
    warps = cap.get_warp_matrices(ref_index=5)
    warp_objs = [
        (
            ProjectiveTransform(matrix=np.array(warp))
            if not isinstance(warp, ProjectiveTransform)
            else warp
        )
        for warp in warps
    ]
    cap.radiometric_pan_sharpened_aligned_capture(
        warp_matrices=warp_objs,
        irradiance_list=irradiance,
        img_type="reflectance",
    )
    aligned = cap._Capture__aligned_radiometric_pan_sharpened_capture[1]
    return np.moveaxis(aligned[:, :, METASHAPE_ORDER], 2, 0).astype("float32", copy=False)


def read_reference(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
        dtype = src.dtypes[0]
    if dtype == "uint16" or float(np.nanmax(data)) > 4.0:
        data = data / REFLECTANCE_SCALE
    return data[:5]


def central_valid_values(
    stack: np.ndarray,
    band_index: int,
    trim_fraction: float,
    min_reflectance: float,
    max_reflectance: float,
) -> np.ndarray:
    band = stack[band_index]
    height, width = band.shape
    y0 = int(height * trim_fraction)
    y1 = int(height * (1 - trim_fraction))
    x0 = int(width * trim_fraction)
    x1 = int(width * (1 - trim_fraction))
    core = band[y0:y1, x0:x1]
    mask = np.isfinite(core) & (core > min_reflectance) & (core < max_reflectance)
    return core[mask]


def median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else float("nan")


def compare_capture(
    week: str,
    capture: str,
    reference_path: Path,
    seed: Path,
    factors: dict[str, float],
    trim_fraction: float,
    min_reflectance: float,
    max_reflectance: float,
    panel_cache: dict[Path, tuple[list[float], dict]],
) -> list[dict]:
    t0 = time.perf_counter()
    reference = read_reference(reference_path)
    irradiance = panel_irradiance_for_seed(seed, panel_cache)
    custom = compute_custom_stack(seed, irradiance)
    logging.info("[CAPTURE] %s %s processed in %.1fs", week, capture, time.perf_counter() - t0)
    rows = []
    for band_index, band in enumerate(BAND_NAMES):
        ref_median = median_or_nan(
            central_valid_values(
                reference, band_index, trim_fraction, min_reflectance, max_reflectance
            )
        )
        custom_median = median_or_nan(
            central_valid_values(
                custom, band_index, trim_fraction, min_reflectance, max_reflectance
            )
        )
        corrected_median = custom_median * factors[band]
        raw_ratio = ref_median / custom_median if custom_median > 0 else float("nan")
        corrected_ratio = ref_median / corrected_median if corrected_median > 0 else float("nan")
        rows.append(
            {
                "week": week,
                "capture_id": capture,
                "band": band,
                "reference_file": str(reference_path),
                "raw_seed": str(seed),
                "reference_median": ref_median,
                "custom_median_uncorrected": custom_median,
                "frozen_factor": factors[band],
                "custom_median_corrected": corrected_median,
                "raw_ratio_reference_over_custom": raw_ratio,
                "corrected_ratio_reference_over_custom": corrected_ratio,
                "abs_log_corrected_ratio": (
                    abs(float(np.log(corrected_ratio))) if corrected_ratio > 0 else float("nan")
                ),
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for week in sorted({row["week"] for row in rows}):
        for band in BAND_NAMES:
            subset = [row for row in rows if row["week"] == week and row["band"] == band]
            raw_ratios = np.array(
                [row["raw_ratio_reference_over_custom"] for row in subset], dtype="float64"
            )
            corrected = np.array(
                [row["corrected_ratio_reference_over_custom"] for row in subset], dtype="float64"
            )
            corrected = corrected[np.isfinite(corrected)]
            raw_ratios = raw_ratios[np.isfinite(raw_ratios)]
            summary.append(
                {
                    "week": week,
                    "band": band,
                    "n_captures": len(subset),
                    "raw_ratio_median": float(np.median(raw_ratios)),
                    "corrected_ratio_median": float(np.median(corrected)),
                    "corrected_ratio_iqr": float(
                        np.percentile(corrected, 75) - np.percentile(corrected, 25)
                    ),
                    "median_abs_percent_error_after_correction": float(
                        np.median(np.abs(corrected - 1.0)) * 100.0
                    ),
                    "passes_20pct": bool(np.median(np.abs(corrected - 1.0)) <= 0.20),
                }
            )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict]) -> str:
    columns = [
        "week",
        "band",
        "n_captures",
        "raw_ratio_median",
        "corrected_ratio_median",
        "median_abs_percent_error_after_correction",
        "passes_20pct",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        vals = []
        for col in columns:
            value = row[col]
            vals.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    summary: list[dict],
    outputs: list[Path],
    args: argparse.Namespace,
    log_path: Path,
) -> None:
    failing = [f"{row['week']} {row['band']}" for row in summary if not bool(row["passes_20pct"])]
    interpretation = (
        "Frozen factors are considered transferable only where the corrected "
        "Metashape/custom median ratio remains near 1.0. Bands or weeks failing "
        "the 20% threshold indicate week-specific calibration drift or a mismatch "
        "between the custom and Metashape radiometric assumptions."
    )
    text = f"""## Results: Metashape Correction Transferability

{markdown_table(summary)}

**Interpretation**: {interpretation}

**Failing week-band checks (>20% median residual error)**: {', '.join(failing) if failing else 'none'}

**Outputs**
{chr(10).join(f"- `{output}`" for output in outputs + [log_path])}

**Reproducibility**
- raw_root: `{args.raw_root}`
- processed_root: `{args.processed_root}`
- correction_json: `{args.correction_json}`
- weeks: `{','.join(args.weeks)}`
- samples_per_week: `{args.samples_per_week}`
- trim_fraction: `{args.trim_fraction}`
- min_reflectance: `{args.min_reflectance}`
- max_reflectance: `{args.max_reflectance}`
- correction frozen before this test: yes
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/data/ONCERCO/data/raw/2025"))
    parser.add_argument(
        "--processed-root", type=Path, default=Path("/mnt/data/ONCERCO/data/processed/2025")
    )
    parser.add_argument(
        "--correction-json",
        type=Path,
        default=Path("configs/frozen/metashape_compat_rededgep_2025.json"),
    )
    parser.add_argument("--weeks", nargs="+", default=list(DEFAULT_WEEKS))
    parser.add_argument("--samples-per-week", type=int, default=4)
    parser.add_argument("--trim-fraction", type=float, default=0.05)
    parser.add_argument("--min-reflectance", type=float, default=1e-6)
    parser.add_argument("--max-reflectance", type=float, default=1.5)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("outputs/results/metashape_correction_transferability"),
    )
    args = parser.parse_args()

    log_path = setup_logging()
    t0 = time.perf_counter()
    factors = load_factors(args.correction_json)
    t0 = phase("load correction factors", t0)

    rows: list[dict] = []
    panel_cache: dict[Path, tuple[list[float], dict]] = {}
    for week in args.weeks:
        reference_dir = find_reference_dir(args.processed_root, week)
        refs = reference_map(reference_dir)
        raws = raw_seed_map(args.raw_root, week)
        common = sorted(set(refs) & set(raws))
        selected = sample_ids(common, args.samples_per_week)
        logging.info(
            "[WEEK] %s reference_dir=%s common=%s selected=%s",
            week,
            reference_dir,
            len(common),
            selected,
        )
        for cid in selected:
            try:
                rows.extend(
                    compare_capture(
                        week,
                        cid,
                        refs[cid],
                        raws[cid],
                        factors,
                        args.trim_fraction,
                        args.min_reflectance,
                        args.max_reflectance,
                        panel_cache,
                    )
                )
            except Exception as exc:
                logging.exception("[FAIL] %s IMG_%s: %s", week, cid, exc)
    t0 = phase("process sampled captures", t0)
    if not rows:
        raise RuntimeError("no successful sampled captures")

    summary = summarize(rows)
    detail_csv = args.out_prefix.with_name(args.out_prefix.name + "_detail.csv")
    summary_csv = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    report_path = Path("outputs/reports") / f"{args.out_prefix.name}_summary.md"
    write_csv(detail_csv, rows)
    write_csv(summary_csv, summary)
    t0 = phase("write CSV outputs", t0)
    write_report(report_path, summary, [detail_csv, summary_csv], args, log_path)
    phase("write markdown report", t0)
    logging.info("[DONE] outputs=%s", [detail_csv, summary_csv, report_path, log_path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
