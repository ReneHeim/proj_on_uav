#!/usr/bin/env python3
"""Run a small MicaSense alignment sample following the official tutorial path."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

np.mat = np.asmatrix  # NumPy 2 compatibility for MicaSense imageprocessing.

import micasense.capture as capture_mod
import micasense.imageutils as imageutils
import rasterio
from rasterio.transform import Affine
from skimage.filters import sobel
from skimage.registration import phase_cross_correlation


MS_BAND_NAMES = ("Blue", "Green", "Red", "NIR", "Red edge")
MS_BAND_INDEXES = (0, 1, 2, 3, 4)


def capture_files(raw_dir: Path, capture_id: str) -> list[str]:
    return [str(raw_dir / f"IMG_{capture_id}_{band}.tif") for band in range(1, 7)]


def stretch(array: np.ndarray) -> np.ndarray:
    array = array.astype("float32")
    mask = np.isfinite(array) & (array > 0)
    if mask.sum() < 100:
        return np.zeros_like(array, dtype="float32")
    low, high = np.percentile(array[mask], [2, 98])
    if high <= low:
        return np.zeros_like(array, dtype="float32")
    return np.clip((array - low) / (high - low), 0, 1)


def write_stack(path: Path, aligned: np.ndarray) -> None:
    data = np.moveaxis(aligned[:, :, MS_BAND_INDEXES], 2, 0)
    data = np.nan_to_num(data, nan=0.0, posinf=2.0, neginf=0.0)
    data = np.clip(data, 0.0, 2.0)
    data = np.rint(data * 32767.0).astype("uint16")
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": "uint16",
        "compress": "deflate",
        "tiled": True,
        "transform": Affine.identity(),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for index, name in enumerate(MS_BAND_NAMES, start=1):
            dst.set_band_description(index, name)
        dst.update_tags(
            software="MicaSense official alignment tutorial path",
            reflectance_scale="32767 = 1.0 reflectance",
            pansharpened="false",
        )


def write_previews(stack_dir: Path, preview_dir: Path) -> list[dict[str, float | str]]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(stack_dir.glob("*.tif"))
    for mode, indexes, title in [
        ("true_rgb", (2, 1, 0), "True color: Red/Green/Blue"),
        ("false_color_nir", (3, 2, 1), "False color: NIR/Red/Green"),
    ]:
        fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
        for axis, path in zip(axes.ravel(), files):
            with rasterio.open(path) as src:
                stack = src.read()
            rgb = np.dstack([stretch(stack[index]) for index in indexes])
            axis.imshow(rgb)
            axis.set_title(path.stem, fontsize=9)
            axis.axis("off")
        fig.suptitle(title, fontsize=14)
        fig.savefig(preview_dir / f"{mode}_contact_sheet.png", dpi=160)
        plt.close(fig)

    metrics: list[dict[str, float | str]] = []
    for path in files:
        with rasterio.open(path) as src:
            stack = src.read()
            descriptions = src.descriptions
        reference = sobel(stretch(stack[2]))
        for index, name in enumerate(descriptions):
            if index == 2:
                continue
            moving = sobel(stretch(stack[index]))
            shift, _error, _phase = phase_cross_correlation(
                reference, moving, upsample_factor=10
            )
            metrics.append(
                {
                    "file": path.name,
                    "band": name,
                    "shift_y_vs_red_px": float(shift[0]),
                    "shift_x_vs_red_px": float(shift[1]),
                }
            )

    central = files[len(files) // 2]
    with rasterio.open(central) as src:
        stack = src.read()
    red_edges = stretch(sobel(stretch(stack[2])))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for axis, (band_index, name) in zip(
        axes.ravel(), [(0, "Blue"), (1, "Green"), (3, "NIR"), (4, "Red edge")]
    ):
        moving_edges = stretch(sobel(stretch(stack[band_index])))
        axis.imshow(np.dstack([red_edges, moving_edges, np.zeros_like(red_edges)]))
        axis.set_title(f"{central.stem}: red edges=red, {name} edges=green")
        axis.axis("off")
    fig.savefig(preview_dir / f"{central.stem}_edge_overlay_vs_red.png", dpi=180)
    plt.close(fig)

    with (preview_dir / "band_shift_metrics_vs_red.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=155)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--align-id", default="0160")
    args = parser.parse_args()

    stack_dir = args.out_root / "official_aligned_reflectance_stacks"
    preview_dir = args.out_root / "qa_previews"
    stack_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    align_cap = capture_mod.Capture.from_filelist(capture_files(args.raw_dir, args.align_id))
    warp_mode = cv2.MOTION_HOMOGRAPHY
    match_index = 1
    warp_matrices, _alignment_pairs = imageutils.align_capture(
        align_cap,
        ref_index=match_index,
        max_iterations=10,
        warp_mode=warp_mode,
        pyramid_levels=0,
    )
    cropped_dimensions, _edges = imageutils.find_crop_bounds(
        align_cap, warp_matrices, warp_mode=warp_mode
    )

    capture_ids = [f"{capture_id:04d}" for capture_id in range(args.start, args.start + args.count)]
    for capture_id in capture_ids:
        cap = capture_mod.Capture.from_filelist(capture_files(args.raw_dir, capture_id))
        cap.compute_undistorted_reflectance(cap.dls_irradiance())
        aligned = imageutils.aligned_capture(
            cap,
            warp_matrices,
            warp_mode,
            cropped_dimensions,
            match_index,
            img_type="reflectance",
        )
        write_stack(stack_dir / f"IMG_{capture_id}_6.tif", aligned)

    metrics = write_previews(stack_dir, preview_dir)
    max_abs_shift = max(
        max(abs(row["shift_y_vs_red_px"]), abs(row["shift_x_vs_red_px"]))
        for row in metrics
        if isinstance(row["shift_y_vs_red_px"], float)
    )
    summary = {
        "raw_dir": str(args.raw_dir),
        "out_root": str(args.out_root),
        "stack_dir": str(stack_dir),
        "preview_dir": str(preview_dir),
        "capture_ids": capture_ids,
        "align_id": args.align_id,
        "match_index": match_index,
        "match_band": align_cap.band_names()[match_index],
        "warp_mode": "MOTION_HOMOGRAPHY",
        "radiometry": "DLS reflectance via Capture.compute_undistorted_reflectance",
        "pansharpened": False,
        "stack_bands": MS_BAND_NAMES,
        "max_abs_shift_pixels": max_abs_shift,
    }
    with (args.out_root / "official_alignment_sample_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
