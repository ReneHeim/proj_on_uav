#!/usr/bin/env python3
"""Compare MicaSense radiometry modes against existing Metashape references.

This is an evaluation script, not a calibration script for missing weeks. It uses
available Metashape products only to measure which raw-to-reflectance strategy is
most transferable across weeks.
"""

from __future__ import annotations

import argparse
import csv
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
RADIOMETRY_MODES = ("micasense_panel", "micasense_dls", "panel_dls_tie")


def setup_logging() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("outputs/logs") / f"evaluate_micasense_radiometry_modes_{timestamp}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path)],
        force=True,
    )
    return path


def log_phase(name: str, started: float) -> float:
    logging.info("[PHASE] %s: %.1fs", name, time.perf_counter() - started)
    return time.perf_counter()


def capture_id(path: Path) -> str | None:
    match = re.search(r"IMG_(\d+)_([16])\.tif$", path.name)
    return match.group(1) if match else None


def capture_files(seed: Path) -> list[str]:
    return [str(seed.with_name(seed.name.replace("_1.tif", f"_{index}.tif"))) for index in range(1, 7)]


def find_reference_dir(processed_root: Path, week: str) -> Path:
    candidates = sorted(processed_root.glob(f"{week}/metashape/*/orthophotos"))
    candidates = [path for path in candidates if " copy" not in str(path) and any(path.glob("IMG_*_6.tif"))]
    if not candidates:
        raise FileNotFoundError(f"no Metashape orthophoto directory for {week}")
    return candidates[0]


def map_raw_seeds(raw_root: Path, week: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted((raw_root / week / "rededgep").glob("**/IMG_*_1.tif")):
        cid = capture_id(path)
        if cid is not None:
            out.setdefault(cid, path)
    return out


def map_references(reference_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(reference_dir.glob("IMG_*_6.tif")):
        cid = capture_id(path)
        if cid is not None:
            out.setdefault(cid, path)
    return out


def sample_ids(common_ids: list[str], count: int) -> list[str]:
    if count <= 0 or count >= len(common_ids):
        return common_ids
    indexes = np.linspace(0, len(common_ids) - 1, count, dtype=int)
    return [common_ids[index] for index in sorted(set(indexes))]


def read_reference(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
        dtype = src.dtypes[0]
    if dtype == "uint16" or float(np.nanmax(data)) > 4.0:
        data = data / REFLECTANCE_SCALE
    return data[:5]


def set_dir_for_seed(seed: Path) -> Path:
    return next(parent for parent in seed.parents if parent.name.endswith("SET"))


def panel_state_for_seed(seed: Path, cache: dict[Path, dict]) -> dict:
    set_dir = set_dir_for_seed(seed)
    if set_dir in cache:
        return cache[set_dir]
    panel_seed = set_dir / "000" / "IMG_0000_1.tif"
    if not panel_seed.exists():
        raise FileNotFoundError(f"panel seed missing: {panel_seed}")
    cap = Capture.from_filelist(capture_files(panel_seed))
    detected = int(cap.detect_panels())
    if detected < 5:
        raise RuntimeError(f"panel detection failed for {panel_seed}: {detected}/6 bands")
    panel_irradiance = [float(value) for value in cap.panel_irradiance()]
    panel_dls = [float(value) for value in cap.dls_irradiance()]
    scale = [
        panel_value / dls_value if dls_value > 0 else 1.0
        for panel_value, dls_value in zip(panel_irradiance, panel_dls)
    ]
    state = {
        "set_dir": str(set_dir),
        "panel_seed": str(panel_seed),
        "panel_detected_bands": detected,
        "panel_irradiance": panel_irradiance,
        "panel_dls_irradiance": panel_dls,
        "panel_dls_to_panel_scale": scale,
    }
    cache[set_dir] = state
    logging.info("[PANEL] %s detected=%s scale=%s", set_dir.name, detected, scale)
    return state


def irradiance_for_mode(cap: Capture, panel_state: dict, mode: str) -> list[float]:
    if mode == "micasense_panel":
        return list(panel_state["panel_irradiance"])
    dls = [float(value) for value in cap.dls_irradiance()]
    if mode == "micasense_dls":
        return dls
    if mode == "panel_dls_tie":
        return [
            dls_value * scale
            for dls_value, scale in zip(dls, panel_state["panel_dls_to_panel_scale"])
        ]
    raise ValueError(f"unsupported mode: {mode}")


def compute_custom_stack(seed: Path, panel_state: dict, mode: str) -> np.ndarray:
    cap = Capture.from_filelist(capture_files(seed))
    warps = cap.get_warp_matrices(ref_index=5)
    warp_objs = [
        ProjectiveTransform(matrix=np.array(warp)) if not isinstance(warp, ProjectiveTransform) else warp
        for warp in warps
    ]
    irradiance = irradiance_for_mode(cap, panel_state, mode)
    cap.radiometric_pan_sharpened_aligned_capture(
        warp_matrices=warp_objs,
        irradiance_list=irradiance,
        img_type="reflectance",
    )
    aligned = cap._Capture__aligned_radiometric_pan_sharpened_capture[1]
    return np.moveaxis(aligned[:, :, METASHAPE_ORDER], 2, 0).astype("float32", copy=False)


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
    y1 = int(height * (1.0 - trim_fraction))
    x0 = int(width * trim_fraction)
    x1 = int(width * (1.0 - trim_fraction))
    core = band[y0:y1, x0:x1]
    mask = np.isfinite(core) & (core > min_reflectance) & (core < max_reflectance)
    return core[mask]


def median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if values.size else float("nan")


def compare_capture(
    week: str,
    cid: str,
    reference_path: Path,
    seed: Path,
    modes: tuple[str, ...],
    trim_fraction: float,
    min_reflectance: float,
    max_reflectance: float,
    panel_cache: dict[Path, dict],
) -> list[dict]:
    reference = read_reference(reference_path)
    ref_medians = [
        median_or_nan(central_valid_values(reference, i, trim_fraction, min_reflectance, max_reflectance))
        for i in range(len(BAND_NAMES))
    ]
    panel_state = panel_state_for_seed(seed, panel_cache)
    rows = []
    for mode in modes:
        started = time.perf_counter()
        custom = compute_custom_stack(seed, panel_state, mode)
        logging.info("[CAPTURE] %s IMG_%s mode=%s %.1fs", week, cid, mode, time.perf_counter() - started)
        for band_index, band in enumerate(BAND_NAMES):
            custom_median = median_or_nan(
                central_valid_values(custom, band_index, trim_fraction, min_reflectance, max_reflectance)
            )
            ratio = ref_medians[band_index] / custom_median if custom_median > 0 else float("nan")
            rows.append(
                {
                    "week": week,
                    "capture_id": cid,
                    "mode": mode,
                    "band": band,
                    "reference_file": str(reference_path),
                    "raw_seed": str(seed),
                    "reference_median": ref_medians[band_index],
                    "custom_median": custom_median,
                    "ratio_reference_over_custom": ratio,
                    "abs_percent_error": abs(ratio - 1.0) * 100.0 if np.isfinite(ratio) else float("nan"),
                }
            )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for mode in RADIOMETRY_MODES:
        for week in sorted({row["week"] for row in rows}):
            for band in BAND_NAMES:
                subset = [row for row in rows if row["mode"] == mode and row["week"] == week and row["band"] == band]
                ratios = np.array([row["ratio_reference_over_custom"] for row in subset], dtype="float64")
                ratios = ratios[np.isfinite(ratios)]
                if ratios.size == 0:
                    continue
                summary.append(
                    {
                        "mode": mode,
                        "week": week,
                        "band": band,
                        "n_captures": len(subset),
                        "ratio_median": float(np.median(ratios)),
                        "ratio_iqr": float(np.percentile(ratios, 75) - np.percentile(ratios, 25)),
                        "median_abs_percent_error": float(np.median(np.abs(ratios - 1.0)) * 100.0),
                        "passes_20pct": bool(np.median(np.abs(ratios - 1.0)) <= 0.20),
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
    cols = ["mode", "week", "band", "n_captures", "ratio_median", "median_abs_percent_error", "passes_20pct"]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        values = []
        for col in cols:
            value = row[col]
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, summary: list[dict], outputs: list[Path], args: argparse.Namespace, log_path: Path) -> None:
    by_mode: dict[str, list[float]] = {}
    for row in summary:
        by_mode.setdefault(row["mode"], []).append(float(row["median_abs_percent_error"]))
    mode_scores = {
        mode: float(np.median(np.array(errors, dtype="float64")))
        for mode, errors in by_mode.items()
        if errors
    }
    best_mode = min(mode_scores, key=mode_scores.get) if mode_scores else "none"
    text = f"""## Results: MicaSense Radiometry Mode Transferability

{markdown_table(summary)}

**Interpretation**: The best tested mode by median week-band error was `{best_mode}`. A mode is considered Metashape-compatible only where the reference/custom median ratio remains near 1.0 across weeks and bands; failures indicate unresolved differences in panel/DLS handling, illumination normalization, or Metashape processing assumptions.

**Mode-level median absolute percent error**
{chr(10).join(f"- `{mode}`: {score:.1f}%" for mode, score in sorted(mode_scores.items()))}

**Outputs**
{chr(10).join(f"- `{output}`" for output in outputs + [log_path])}

**Reproducibility**
- raw_root: `{args.raw_root}`
- processed_root: `{args.processed_root}`
- weeks: `{','.join(args.weeks)}`
- modes: `{','.join(args.modes)}`
- samples_per_week: `{args.samples_per_week}`
- trim_fraction: `{args.trim_fraction}`
- min_reflectance: `{args.min_reflectance}`
- max_reflectance: `{args.max_reflectance}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/data/ONCERCO/data/raw/2025"))
    parser.add_argument("--processed-root", type=Path, default=Path("/mnt/data/ONCERCO/data/processed/2025"))
    parser.add_argument("--weeks", nargs="+", default=list(DEFAULT_WEEKS))
    parser.add_argument("--modes", nargs="+", choices=RADIOMETRY_MODES, default=list(RADIOMETRY_MODES))
    parser.add_argument("--samples-per-week", type=int, default=3)
    parser.add_argument("--trim-fraction", type=float, default=0.05)
    parser.add_argument("--min-reflectance", type=float, default=1e-6)
    parser.add_argument("--max-reflectance", type=float, default=1.5)
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("outputs/results/micasense_radiometry_mode_transferability"),
    )
    args = parser.parse_args()

    log_path = setup_logging()
    started = time.perf_counter()
    rows: list[dict] = []
    panel_cache: dict[Path, dict] = {}
    modes = tuple(args.modes)
    for week in args.weeks:
        reference_dir = find_reference_dir(args.processed_root, week)
        refs = map_references(reference_dir)
        raws = map_raw_seeds(args.raw_root, week)
        common = sorted(set(refs) & set(raws))
        selected = sample_ids(common, args.samples_per_week)
        logging.info("[WEEK] %s reference_dir=%s common=%s selected=%s", week, reference_dir, len(common), selected)
        for cid in selected:
            try:
                rows.extend(
                    compare_capture(
                        week,
                        cid,
                        refs[cid],
                        raws[cid],
                        modes,
                        args.trim_fraction,
                        args.min_reflectance,
                        args.max_reflectance,
                        panel_cache,
                    )
                )
            except Exception as exc:
                logging.exception("[FAIL] %s IMG_%s: %s", week, cid, exc)
    started = log_phase("process sampled captures", started)
    if not rows:
        raise RuntimeError("no successful captures")
    summary = summarize(rows)
    detail_csv = args.out_prefix.with_name(args.out_prefix.name + "_detail.csv")
    summary_csv = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    report_path = Path("outputs/reports") / f"{args.out_prefix.name}_summary.md"
    write_csv(detail_csv, rows)
    write_csv(summary_csv, summary)
    started = log_phase("write CSV outputs", started)
    write_report(report_path, summary, [detail_csv, summary_csv], args, log_path)
    log_phase("write markdown report", started)
    logging.info("[DONE] outputs=%s", [detail_csv, summary_csv, report_path, log_path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
