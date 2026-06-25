#!/usr/bin/env python3
"""Fix up existing preprocessed stacks: swap NIR and Red edge bands (3 and 4).

The micasense EXIF metadata in our TIF files labels position 4 as 'NIR
(842 nm)' and position 5 as 'Red edge (717 nm)', but the actual content
is reversed. The v2 GPU worker and runner now apply the swap at
production time, but pre-existing outputs (before the fix) have the
wrong band order.

This script reads each existing preprocessed stack and writes a new
file with bands 3 and 4 swapped. Use --in-place to overwrite.

Usage:
    # Fix all existing outputs in week1_gpu/0000SET/
    python -m scripts.archive.one_off_repairs.fix_band_swap \\
        --indir /mnt/.../week1_gpu/0000SET/preprocessed_stacks

    # In-place overwrite
    python -m scripts.archive.one_off_repairs.fix_band_swap --indir ... --in-place
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("band_swap_fix")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indir", type=Path, required=True, help="Directory of *_6.tif preprocessed stacks"
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite original files (default: write *_swapped.tif)",
    )
    parser.add_argument(
        "--pattern", default="*_6.tif", help="Glob pattern for files to fix (default *_6.tif)"
    )
    args = parser.parse_args()

    files = sorted(args.indir.glob(args.pattern))
    if not files:
        log.error("No files matching %s in %s", args.pattern, args.indir)
        return 1
    log.info("Found %d files in %s", len(files), args.indir)
    if not args.in_place:
        log.info("Writing to *swapped.tif; use --in-place to overwrite originals")

    n_ok, n_skip, n_fail = 0, 0, 0
    for f in files:
        try:
            with rasterio.open(f) as src:
                data = src.read()
                profile = src.profile
                tags = dict(src.tags())
            if data.shape[0] < 5:
                log.warning("SKIP %s: only %d bands (expected 5+)", f.name, data.shape[0])
                n_skip += 1
                continue
            swapped = data.copy()
            swapped[[3, 4]] = data[[4, 3]]
            out_path = f if args.in_place else f.with_name(f.stem + "_swapped.tif")
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(swapped)
                for k, v in tags.items():
                    dst.update_tags(**{k: v})
                dst.update_tags(
                    band_swap_fixed="true (NIR/Red edge positions 4 and 5 swapped in source TIFs)",
                )
            n_ok += 1
            log.info("OK %s -> %s", f.name, out_path.name)
        except Exception as e:
            log.error("FAIL %s: %s", f.name, e)
            n_fail += 1
    log.info("=" * 50)
    log.info("Done. %d OK, %d skipped, %d failed.", n_ok, n_skip, n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
