#!/usr/bin/env python3
"""Generate a Blue-aligned CPU SIFT multispec stack from existing CPU warps.

The micasense CPU SIFT pipeline aligns all bands to the Panchromatic band
(band 6, at 2056x2464). The original preprocessed stack `IMG_0002_6.tif`
is at pan-sharpened res (1912x2449, 5 bands Blue-RedEdge).

This script inverts the panchro-to-blue warp and applies it to all bands
(including panchro), producing a Blue-aligned multispec stack at
multispec native res (1088x1456). This is the reference frame used by
the v2 GPU worker, so the 99% similarity metric is meaningful.

Usage:
    python scripts/_make_blue_aligned_cpu_stack.py \\
        --capture /mnt/data/.../IMG_0002_1.tif \\
        --warp-cache /mnt/data/.../warps.npz \\
        --out /path/to/blue_aligned_stack.tif
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from skimage.transform import ProjectiveTransform, warp as _sk_warp, resize


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )


log = logging.getLogger("blue_stack")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True,
                        help="Blue-band path (e.g. .../IMG_0002_1.tif)")
    parser.add_argument("--warp-cache", type=Path, required=True,
                        help="Path to the warp cache .npz from the micasense SIFT run")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output 5-band uint16 Blue-aligned stack")
    parser.add_argument("--rgb-out", type=Path, default=None,
                        help="Optional path for RGB preview PNG")
    args = parser.parse_args()
    setup_logging()

    log.info("Capture: %s", args.capture)
    log.info("Warp cache: %s", args.warp_cache)
    log.info("Output: %s", args.out)

    # Load warps
    warp_data = np.load(args.warp_cache, allow_pickle=True)
    n_warps = sum(1 for k in warp_data.files if k.startswith("warp_"))
    warps = [np.array(warp_data[f"warp_{i}"]) for i in range(n_warps)]
    log.info("Loaded %d warps", n_warps)
    for i, w in enumerate(warps):
        log.info("  warp[%d] sum=%.3f trace=%.3f", i, w.sum(), w[0, 0] + w[1, 1])

    # The micasense warps are in the form warp[i]: ref(5) -> band i
    # where ref=5 is Panchro at pan-sharpened res.
    # To align to BLUE (band 0) instead:
    #   1. Apply warp[0] inverse to band 0 (Blue) to get Panchro coords
    #   2. Apply warp[0] inverse to all other bands to get them in Panchro coords
    #   3. The result is a 5-band stack in Panchro coords, but we want it in
    #      Blue coords at multispec res
    #
    # Simpler approach: warp each band into BLUE's reference frame directly.
    # The CPU computes:
    #   P_i = warp[i]  such that P_i maps (y, x) in Panchro coords -> (y, x) in band i coords
    # We want Q_i = warp from band i -> Blue. That's Q_i = P_0 . P_i^{-1}.
    # Then we apply Q_i to band i to get it in Blue's coords.
    #
    # But micasense stores warps in skimage ProjectiveTransform convention.
    # The warp maps (y, x) -> (y, x) where the order is skimage's (y, x).
    # So: Q_i = ProjectiveTransform(matrix=warp[0]) . ProjectiveTransform(matrix=warp[i]).inverse

    P0 = ProjectiveTransform(matrix=warps[0])
    log.info("P0 (Panchro -> Blue) params:\n%s", P0.params)

    # Load 6 bands at their native resolution
    band_suffixes = ("1", "2", "3", "4", "5", "6")
    band_paths = [args.capture.with_name(args.capture.name.replace("_1.tif", f"_{s}.tif"))
                  for s in band_suffixes]
    bands = []
    for p in band_paths:
        with rasterio.open(p) as src:
            bands.append(src.read(1).astype(np.float32))
    log.info("Band shapes: %s", [b.shape for b in bands])

    # Target: Blue's native (multispec) resolution
    target_h, target_w = bands[0].shape  # multispec (1088, 1456)
    log.info("Target shape (Blue native): (%d, %d)", target_h, target_w)

    # For each band i, build Q_i that maps band i coords -> Blue coords
    # Q_i = P0 . P_i^{-1}
    #   where P_i maps Panchro coords -> band i coords
    #   P_i^{-1} maps band i coords -> Panchro coords
    #   P0 maps Panchro coords -> Blue coords
    # So Q_i maps band i coords -> Blue coords. Apply Q_i to band i.
    aligned = []
    for i, b in enumerate(bands):
        if i == 0:
            # Blue is the ref, no warp
            aligned.append(b.astype(np.float32))
            log.info("  band 0 (Blue): identity, mean=%.3f", b.mean())
            continue
        if i == 5:
            # Panchro: we need to resize it to multispec res first, then warp
            b_resized = resize(b, (target_h, target_w),
                               preserve_range=True).astype(np.float32)
            # For panchro, P_5 is identity. So Q_5 = P0 . I = P0.
            # Micasense stores P_i such that P_i maps moving -> ref. So to apply
            # (output=ref, input=moving), we use inverse_map = P_i.inverse.
            # For panchro with P5=I, inverse_map = I.inverse = I.
            # The Q = P0 composition is applied via Q.inverse to undo the band->ref transform.
            warped = _sk_warp(b_resized, inverse_map=P0.inverse, mode="edge",
                              preserve_range=True, output_shape=(target_h, target_w))
            aligned.append(warped.astype(np.float32))
            log.info("  band 5 (Panchro): warped, mean before=%.3f after=%.3f",
                     b_resized.mean(), warped.mean())
            continue
        Pi = ProjectiveTransform(matrix=warps[i])
        # Q_i = P0 . P_i^{-1}  (micasense-style right-to-left matrix composition)
        # Micasense P_i maps band i coords -> panchro coords -> Blue coords.
        # So Q_i maps band i -> Blue in (y, x) coords.
        # For warp(image=band_i, inverse_map=M): M maps OUTPUT to INPUT.
        # We want M(Blue_coord) = band_i_coord.
        # Since Q_i maps band_i -> Blue, its INVERSE Q_i.inverse maps Blue -> band_i.
        # So inverse_map = Q_i.inverse.
        Q_params = P0.params @ Pi.inverse.params
        Q = ProjectiveTransform(matrix=Q_params)
        # Resize band to multispec res if needed
        if b.shape != (target_h, target_w):
            b = resize(b, (target_h, target_w), preserve_range=True).astype(np.float32)
        warped = _sk_warp(b, inverse_map=Q.inverse, mode="edge",
                          preserve_range=True, output_shape=(target_h, target_w))
        aligned.append(warped.astype(np.float32))
        log.info("  band %d: warped, mean before=%.3f after=%.3f",
                 i, b.mean(), warped.mean())

    aligned_5 = np.stack(aligned[:5])  # (5, H, W)
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
            software="make_blue_aligned_cpu_stack.py",
            reflectance_scale="32767 = 1.0 reflectance",
            warp_method="micasense CPU SIFT (inverted to Blue ref)",
        )
    log.info("Saved %s", args.out)

    # RGB preview
    if args.rgb_out:
        from PIL import Image
        rgb = aligned_5[[2, 1, 0]]
        rgb = np.transpose(rgb, (1, 2, 0))
        lo, hi = float(np.nanpercentile(rgb, 1)), float(np.nanpercentile(rgb, 99))
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        args.rgb_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(args.rgb_out)
        log.info("Saved %s", args.rgb_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
