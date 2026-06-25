#!/usr/bin/env python3
"""Generate a quick RGB PNG preview from a preprocessed reflectance stack.

Usage:
    python -m scripts.archive.legacy_week1_gpu.inspect_week1_output \\
        --tiff /mnt/data/.../week1_gpu/0000SET/preprocessed_stacks/IMG_0002_6.tif \\
        --out /tmp/preview.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiff", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--bands",
        default="2,1,0",
        help="Comma-separated band indices for RGB (default: Red,Green,Blue)",
    )
    args = parser.parse_args()

    with rasterio.open(args.tiff) as src:
        stack = src.read().astype(np.float32) / 32767.0
        tags = dict(src.tags())
    b = [int(x) for x in args.bands.split(",")]
    rgb = np.transpose(stack[b], (1, 2, 0))
    h, w, _ = rgb.shape
    print(f"Stack: {stack.shape}  bands={b}  size={h}x{w}")
    print(f"Tags: {tags}")
    for i, name in enumerate(["Blue", "Green", "Red", "NIR", "Red edge"]):
        print(
            f"  band {i} ({name}): mean={stack[i].mean():.4f}  "
            f"p1={np.percentile(stack[i], 1):.4f}  p99={np.percentile(stack[i], 99):.4f}"
        )
    # Auto-stretch each channel
    rgb_stretched = np.zeros_like(rgb)
    for i in range(3):
        lo, hi = np.percentile(rgb[..., i], [2, 98])
        rgb_stretched[..., i] = np.clip((rgb[..., i] - lo) / max(hi - lo, 1e-6), 0, 1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10 * h / w))
    ax.imshow(rgb_stretched)
    ax.set_title(f"{args.tiff.name}  RGB(bands={b})  {h}x{w}", fontsize=10)
    ax.axis("off")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
