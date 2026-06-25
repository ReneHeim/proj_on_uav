#!/usr/bin/env python3
"""Apply existing CPU SIFT warp matrices to multispec bands and save at native resolution.

Reads the warp cache produced by a previous SIFT run, applies the warps to
the multispec bands (NOT pan-sharpened), and saves a 5-band uint16 stack at
the multispec native resolution. This gives a CPU SIFT result at 1088x1456
for direct, rescaling-free comparison with the GPU SIFT result.

Usage:
    python -m scripts.diagnostics.alignment.apply_cpu_warps_multispec \\
        --capture /mnt/data/.../IMG_0002_1.tif \\
        --warp-cache /mnt/data/.../warps.npz \\
        --out /path/to/output.tif
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import rasterio
from PIL import Image
from skimage.transform import ProjectiveTransform, warp as _sk_warp, resize

# Set up logging + micasense path
sys.path.insert(0, "/home/davidem/miniforge3/lib/python3.13/site-packages")
from micasense.capture import Capture

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("cpu_warps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True,
                        help="Blue-band path (e.g. .../IMG_0002_1.tif)")
    parser.add_argument("--warp-cache", type=Path, required=True,
                        help="Path to the warp cache .npz from the SIFT run")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output 5-band uint16 stack path")
    parser.add_argument("--rgb-out", type=Path, default=None,
                        help="Optional path for RGB preview PNG")
    args = parser.parse_args()

    log.info("Capture: %s", args.capture)
    log.info("Warp cache: %s", args.warp_cache)
    log.info("Output: %s", args.out)

    # Load warps (micasense format: warp_0..warp_5 + metadata)
    warp_data = np.load(args.warp_cache, allow_pickle=True)
    n_warps = sum(1 for k in warp_data.files if k.startswith("warp_"))
    warps = [np.array(warp_data[f"warp_{i}"]) for i in range(n_warps)]
    raw_meta = warp_data["metadata"].item() if "metadata" in warp_data.files else {}
    if isinstance(raw_meta, str):
        import json as _json
        try:
            meta = _json.loads(raw_meta)
        except Exception:
            meta = {"raw": raw_meta}
    else:
        meta = raw_meta
    log.info("Warp method: %s  seed: %s", meta.get("alignment_method"), meta.get("alignment_seed"))

    # Load 6 bands
    band_suffixes = ("1", "2", "3", "4", "5", "6")
    band_paths = [args.capture.with_name(args.capture.name.replace("_1.tif", f"_{s}.tif"))
                  for s in band_suffixes]
    bands = []
    for p in band_paths:
        with rasterio.open(p) as src:
            bands.append(src.read(1).astype(np.float32))
    log.info("Band shapes: %s", [b.shape for b in bands])
    target_h, target_w = bands[0].shape  # multispec size

    # Apply warps to align multispec bands with ref (band 5)
    # micasense convention: P maps from ref to image, so to warp image -> ref
    # we apply P.inverse. The output is in ref coordinates (pan-sharpened size
    # = bands[5].shape). To compare at multispec size, we resize the warped
    # multispec bands to multispec size. Actually, since we're warping the
    # multispec bands (target_h, target_w) with the multispec SIFT warps,
    # the output is already in multispec resolution.
    aligned = []
    valid_masks = []
    for i, b in enumerate(bands):
        if i == 5:
            # ref band — resize to multispec if needed
            if b.shape != (target_h, target_w):
                b = resize(b, (target_h, target_w), preserve_range=True).astype(np.float32)
            aligned.append(b)
            valid_masks.append(np.ones((target_h, target_w), dtype=bool))
            continue
        if warps[i].sum() == 3.0 and np.allclose(warps[i], np.eye(3)):
            # Identity warp (e.g. fallback) — just resize
            if b.shape != (target_h, target_w):
                b = resize(b, (target_h, target_w), preserve_range=True).astype(np.float32)
            aligned.append(b)
            valid_masks.append(np.ones((target_h, target_w), dtype=bool))
            continue
        # micasense stores warps in skimage standard (x, y) convention.
        P = ProjectiveTransform(matrix=warps[i])
        if b.shape != (target_h, target_w):
            b = resize(b, (target_h, target_w), preserve_range=True).astype(np.float32)
        warped = _sk_warp(b, inverse_map=P.inverse, mode="constant", cval=0.0,
                          preserve_range=True, output_shape=(target_h, target_w))
        valid = _sk_warp(np.ones_like(b, dtype=np.float32), inverse_map=P.inverse,
                         mode="constant", cval=0.0,
                         preserve_range=True, output_shape=(target_h, target_w)) > 0.999
        aligned.append(warped.astype(np.float32))
        valid_masks.append(valid)
        log.info("  band %d (%s) warped, mean before=%.3f after=%.3f",
                 i, band_suffixes[i], b.mean(), warped.mean())

    # Stack 5 multispec bands (exclude panchro band 6)
    aligned_5 = np.stack(aligned[:5])  # (5, H, W)
    valid_overlap = np.logical_and.reduce(valid_masks[:5])
    aligned_5[:, ~valid_overlap] = 0.0
    log.info("Valid all-band overlap: %.2f%%", 100.0 * float(valid_overlap.mean()))
    log.info("Aligned stack shape: %s", aligned_5.shape)

    # Normalize to [0, 1] using band 3 (NIR) max as scale
    norm = float(np.percentile(aligned_5, 99))
    if norm > 0:
        aligned_5 = aligned_5 / norm
    aligned_5 = np.clip(aligned_5, 0.0, 1.0)

    # Save uint16 stack
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = np.rint(aligned_5 * 32767.0).astype(np.uint16)
    profile = {
        "driver": "GTiff", "height": data.shape[1], "width": data.shape[2],
        "count": data.shape[0], "dtype": "uint16",
        "compress": "deflate", "tiled": True,
    }
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(data)
        for i, name in enumerate(["Blue", "Green", "Red", "NIR", "Red edge"], start=1):
            dst.set_band_description(i, name)
        dst.update_tags(
            software="apply_cpu_warps_multispec.py",
            reflectance_scale="32767 = 1.0 reflectance",
            warp_method=meta.get("alignment_method", "?"),
        )
    log.info("Saved %s", args.out)

    # RGB preview
    if args.rgb_out:
        rgb = aligned_5[[2, 1, 0]]
        rgb = np.transpose(rgb, (1, 2, 0))
        lo, hi = float(np.nanpercentile(rgb, 1)), float(np.nanpercentile(rgb, 99))
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        args.rgb_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(args.rgb_out)
        log.info("Saved %s", args.rgb_out)


if __name__ == "__main__":
    main()
