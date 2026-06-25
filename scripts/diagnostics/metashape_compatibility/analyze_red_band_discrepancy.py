#!/usr/bin/env python3
"""Focused red-band discrepancy analysis between Metashape and custom products."""

from __future__ import annotations

import argparse
import csv
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

BAND_NAMES = ("Blue", "Green", "Red", "Red edge", "NIR")
REFLECTANCE_SCALE = 32767.0
DEFAULT_PAIRS = (
    (
        "week3",
        "IMG_0002_6",
        "/mnt/data/ONCERCO/data/processed/2025/week3/metashape/20250828_products_uas_data/orthophotos/IMG_0002_6.tif",
        "/tmp/week3_gpu_metashape_compat_0002_0005/0000SET/preprocessed_stacks/IMG_0002_6.tif",
    ),
    (
        "week3",
        "IMG_0004_6",
        "/mnt/data/ONCERCO/data/processed/2025/week3/metashape/20250828_products_uas_data/orthophotos/IMG_0004_6.tif",
        "/tmp/week3_gpu_metashape_compat_0002_0005/0000SET/preprocessed_stacks/IMG_0004_6.tif",
    ),
)


def setup_logging() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("outputs/logs") / f"analyze_red_band_discrepancy_{timestamp}.log"
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


def read_stack(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
        profile = {
            "count": src.count,
            "height": src.height,
            "width": src.width,
            "dtype": src.dtypes[0],
            "descriptions": src.descriptions,
            "crs": src.crs.to_string() if src.crs else None,
            "transform": tuple(src.transform),
        }
    if profile["dtype"] == "uint16" or float(np.nanmax(data)) > 4.0:
        data = data / REFLECTANCE_SCALE
    return data[:5], profile


def crop_core(stack: np.ndarray, crop_fraction: float) -> np.ndarray:
    _, height, width = stack.shape
    margin_y = int(height * (1.0 - crop_fraction) / 2.0)
    margin_x = int(width * (1.0 - crop_fraction) / 2.0)
    return stack[:, margin_y : height - margin_y, margin_x : width - margin_x]


def valid_mask(stack: np.ndarray, min_reflectance: float, max_reflectance: float) -> np.ndarray:
    mask = np.all(np.isfinite(stack), axis=0)
    mask &= np.all((stack > min_reflectance) & (stack < max_reflectance), axis=0)
    return mask


def vegetation_mask(stack: np.ndarray, base_mask: np.ndarray, ndre_min: float) -> np.ndarray:
    red_edge = stack[3]
    nir = stack[4]
    ndre = (nir - red_edge) / np.maximum(nir + red_edge, 1e-6)
    return base_mask & np.isfinite(ndre) & (ndre >= ndre_min)


def soil_like_mask(
    stack: np.ndarray,
    base_mask: np.ndarray,
    ndvi_max: float,
    ndre_max: float,
    nir_rededge_ratio_max: float,
) -> np.ndarray:
    red = stack[2]
    red_edge = stack[3]
    nir = stack[4]
    ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    ndre = (nir - red_edge) / np.maximum(nir + red_edge, 1e-6)
    nir_rededge = nir / np.maximum(red_edge, 1e-6)
    return (
        base_mask
        & np.isfinite(ndvi)
        & np.isfinite(ndre)
        & np.isfinite(nir_rededge)
        & (ndvi <= ndvi_max)
        & (ndre <= ndre_max)
        & (nir_rededge <= nir_rededge_ratio_max)
    )


def summarize_values(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {name: math.nan for name in ("mean", "median", "p10", "p25", "p75", "p90")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
    }


def metric_arrays(stack: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    blue, green, red, red_edge, nir = [stack[index][mask] for index in range(5)]
    return {
        "Blue": blue,
        "Green": green,
        "Red": red,
        "Red edge": red_edge,
        "NIR": nir,
        "Red/Green": red / np.maximum(green, 1e-6),
        "RedEdge/Red": red_edge / np.maximum(red, 1e-6),
        "NIR/RedEdge": nir / np.maximum(red_edge, 1e-6),
        "NDVI": (nir - red) / np.maximum(nir + red, 1e-6),
        "NDRE": (nir - red_edge) / np.maximum(nir + red_edge, 1e-6),
    }


def analyze_pair(
    week: str,
    capture: str,
    reference_path: Path,
    custom_path: Path,
    crop_fraction: float,
    ndre_min: float,
    min_reflectance: float,
    max_reflectance: float,
    ndvi_soil_max: float,
    ndre_soil_max: float,
    nir_rededge_soil_max: float,
) -> tuple[list[dict], dict, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    ref, ref_profile = read_stack(reference_path)
    custom, custom_profile = read_stack(custom_path)
    ref_crop = crop_core(ref, crop_fraction)
    custom_crop = crop_core(custom, crop_fraction)
    ref_base = valid_mask(ref_crop, min_reflectance, max_reflectance)
    custom_base = valid_mask(custom_crop, min_reflectance, max_reflectance)
    ref_veg = vegetation_mask(ref_crop, ref_base, ndre_min)
    custom_veg = vegetation_mask(custom_crop, custom_base, ndre_min)
    ref_soil = soil_like_mask(ref_crop, ref_base, ndvi_soil_max, ndre_soil_max, nir_rededge_soil_max)
    custom_soil = soil_like_mask(
        custom_crop,
        custom_base,
        ndvi_soil_max,
        ndre_soil_max,
        nir_rededge_soil_max,
    )
    rows = []
    for mask_name, ref_mask, custom_mask in (
        ("valid_core", ref_base, custom_base),
        ("ndre_vegetation_core", ref_veg, custom_veg),
        ("soil_like_core", ref_soil, custom_soil),
    ):
        ref_metrics = metric_arrays(ref_crop, ref_mask)
        custom_metrics = metric_arrays(custom_crop, custom_mask)
        for metric in ref_metrics:
            ref_summary = summarize_values(ref_metrics[metric])
            custom_summary = summarize_values(custom_metrics[metric])
            rows.append(
                {
                    "week": week,
                    "capture": capture,
                    "mask": mask_name,
                    "metric": metric,
                    "reference_file": str(reference_path),
                    "custom_file": str(custom_path),
                    "reference_pixels": int(ref_mask.sum()),
                    "custom_pixels": int(custom_mask.sum()),
                    "reference_median": ref_summary["median"],
                    "custom_median": custom_summary["median"],
                    "custom_over_reference_median": custom_summary["median"] / ref_summary["median"]
                    if ref_summary["median"] and np.isfinite(ref_summary["median"])
                    else math.nan,
                    "reference_p25": ref_summary["p25"],
                    "reference_p75": ref_summary["p75"],
                    "custom_p25": custom_summary["p25"],
                    "custom_p75": custom_summary["p75"],
                }
            )
    meta = {
        "reference_profile": ref_profile,
        "custom_profile": custom_profile,
        "reference_valid_pixels": int(ref_base.sum()),
        "custom_valid_pixels": int(custom_base.sum()),
        "reference_veg_pixels": int(ref_veg.sum()),
        "custom_veg_pixels": int(custom_veg.sum()),
        "reference_soil_like_pixels": int(ref_soil.sum()),
        "custom_soil_like_pixels": int(custom_soil.sum()),
    }
    return rows, meta, (ref_crop, custom_crop, ref_veg, custom_veg)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_pair(path: Path, week: str, capture: str, arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    ref, custom, ref_mask, custom_mask = arrays
    metrics = [
        ("Red", ref[2][ref_mask], custom[2][custom_mask]),
        ("Green", ref[1][ref_mask], custom[1][custom_mask]),
        ("Red/Green", ref[2][ref_mask] / np.maximum(ref[1][ref_mask], 1e-6), custom[2][custom_mask] / np.maximum(custom[1][custom_mask], 1e-6)),
        ("NDVI", (ref[4][ref_mask] - ref[2][ref_mask]) / np.maximum(ref[4][ref_mask] + ref[2][ref_mask], 1e-6), (custom[4][custom_mask] - custom[2][custom_mask]) / np.maximum(custom[4][custom_mask] + custom[2][custom_mask], 1e-6)),
        ("NDRE", (ref[4][ref_mask] - ref[3][ref_mask]) / np.maximum(ref[4][ref_mask] + ref[3][ref_mask], 1e-6), (custom[4][custom_mask] - custom[3][custom_mask]) / np.maximum(custom[4][custom_mask] + custom[3][custom_mask], 1e-6)),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(17, 3.6), constrained_layout=True)
    for ax, (label, ref_values, custom_values) in zip(axes, metrics):
        ref_values = ref_values[np.isfinite(ref_values)]
        custom_values = custom_values[np.isfinite(custom_values)]
        ref_values = ref_values[:: max(1, ref_values.size // 50000)]
        custom_values = custom_values[:: max(1, custom_values.size // 50000)]
        ax.hist(ref_values, bins=80, alpha=0.55, density=True, label="Metashape", color="#355C7D")
        ax.hist(custom_values, bins=80, alpha=0.55, density=True, label="Custom", color="#C06C84")
        ax.axvline(np.median(ref_values), color="#355C7D", lw=2)
        ax.axvline(np.median(custom_values), color="#C06C84", lw=2)
        ax.set_title(label)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{week} {capture}: vegetation-core red-band diagnostic", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict]) -> str:
    columns = [
        "week",
        "capture",
        "mask",
        "metric",
        "reference_median",
        "custom_median",
        "custom_over_reference_median",
    ]
    selected = [row for row in rows if row["metric"] in {"Red", "Green", "Red/Green", "NDVI", "NDRE"}]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in selected:
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, rows: list[dict], figures: list[Path], args: argparse.Namespace, log_path: Path) -> None:
    text = f"""## Results: Red Band Discrepancy Diagnostic

{markdown_table(rows)}

**Interpretation**: In vegetation-like central crops, the custom pipeline keeps some NIR/red-edge structure but strongly inflates red reflectance relative to Metashape. This depresses NDVI and RedEdge/Red even when NDRE is closer, indicating a red-band radiometric/product mismatch rather than a simple global scale error.

**Figures**
{chr(10).join(f"- `{figure}`" for figure in figures)}

**Outputs**
- `{args.out_prefix}_summary.csv`
- `{args.out_prefix}_profiles.json`
- `{path}`
- `{log_path}`

**Reproducibility**
- crop_fraction: `{args.crop_fraction}`
- ndre_min: `{args.ndre_min}`
- ndvi_soil_max: `{args.ndvi_soil_max}`
- ndre_soil_max: `{args.ndre_soil_max}`
- nir_rededge_soil_max: `{args.nir_rededge_soil_max}`
- min_reflectance: `{args.min_reflectance}`
- max_reflectance: `{args.max_reflectance}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-fraction", type=float, default=0.60)
    parser.add_argument("--ndre-min", type=float, default=0.10)
    parser.add_argument("--ndvi-soil-max", type=float, default=0.45)
    parser.add_argument("--ndre-soil-max", type=float, default=0.18)
    parser.add_argument("--nir-rededge-soil-max", type=float, default=1.7)
    parser.add_argument("--min-reflectance", type=float, default=1e-6)
    parser.add_argument("--max-reflectance", type=float, default=1.5)
    parser.add_argument("--out-prefix", type=Path, default=Path("outputs/results/red_band_discrepancy"))
    args = parser.parse_args()
    log_path = setup_logging()
    t0 = time.perf_counter()
    all_rows = []
    profiles = {}
    figures = []
    for week, capture, ref_path, custom_path in DEFAULT_PAIRS:
        ref_path = Path(ref_path)
        custom_path = Path(custom_path)
        if not ref_path.exists() or not custom_path.exists():
            logging.warning("[SKIP] missing pair %s %s", ref_path, custom_path)
            continue
        rows, meta, arrays = analyze_pair(
            week,
            capture,
            ref_path,
            custom_path,
            args.crop_fraction,
            args.ndre_min,
            args.min_reflectance,
            args.max_reflectance,
            args.ndvi_soil_max,
            args.ndre_soil_max,
            args.nir_rededge_soil_max,
        )
        all_rows.extend(rows)
        profiles[f"{week}_{capture}"] = meta
        fig_path = Path("outputs/figures") / f"red_band_discrepancy_{week}_{capture}.png"
        plot_pair(fig_path, week, capture, arrays)
        figures.append(fig_path)
    t0 = phase("analyze pairs", t0)
    if not all_rows:
        raise RuntimeError("no pairs analyzed")
    csv_path = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    write_csv(csv_path, all_rows)
    import json

    profile_path = args.out_prefix.with_name(args.out_prefix.name + "_profiles.json")
    profile_path.write_text(json.dumps(profiles, indent=2))
    t0 = phase("write CSV/profile outputs", t0)
    report_path = Path("outputs/reports") / f"{args.out_prefix.name}_summary.md"
    write_report(report_path, all_rows, figures, args, log_path)
    phase("write markdown report", t0)
    logging.info("[DONE] outputs=%s", [csv_path, profile_path, report_path, *figures, log_path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
