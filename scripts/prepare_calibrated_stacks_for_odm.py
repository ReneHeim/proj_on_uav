#!/usr/bin/env python3
"""Split calibrated 5-band MicaSense stacks into ODM-readable single-band TIFFs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import rasterio

BAND_SUFFIXES = ("1", "2", "3", "4", "5")
BAND_NAMES = ("Blue", "Green", "Red", "NIR", "Red edge")


def raw_band_path(raw_set: Path, stack_name: str, suffix: str) -> Path:
    raw_name = stack_name.replace("_6.tif", f"_{suffix}.tif")
    matches = list(raw_set.glob(f"**/{raw_name}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one raw source for {raw_name}, found {len(matches)} in {raw_set}"
        )
    return matches[0]


def copy_metadata(raw_path: Path, out_path: Path) -> None:
    subprocess.run(
        [
            "exiftool",
            "-overwrite_original",
            "-q",
            "-q",
            "-tagsFromFile",
            str(raw_path),
            "-all:all",
            "-unsafe",
            str(out_path),
        ],
        check=True,
    )


def split_stack(stack_path: Path, raw_set: Path, out_dir: Path, overwrite: bool) -> int:
    count = 0
    with rasterio.open(stack_path) as src:
        if src.count != 5:
            raise ValueError(f"expected 5 bands in {stack_path}, found {src.count}")
        for band_index, (suffix, band_name) in enumerate(zip(BAND_SUFFIXES, BAND_NAMES), start=1):
            out_name = stack_path.name.replace("_6.tif", f"_{suffix}.tif")
            out_path = out_dir / out_name
            if out_path.exists() and not overwrite:
                count += 1
                continue

            data = src.read(band_index)
            profile = src.profile.copy()
            profile.update(count=1, compress="deflate", tiled=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)
                dst.set_band_description(1, band_name)
                dst.update_tags(
                    software="Prepared for ODM from calibrated MicaSense stack",
                    source_stack=str(stack_path),
                    reflectance_scale="32767 = 1.0 reflectance",
                    calibrated_band=band_name,
                )

            copy_metadata(raw_band_path(raw_set, stack_path.name, suffix), out_path)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stack-dir", type=Path, required=True, help="Folder of calibrated IMG_*_6.tif stacks"
    )
    parser.add_argument(
        "--raw-set", type=Path, required=True, help="Matching raw MicaSense SET folder"
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output ODM images directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stacks = sorted(args.stack_dir.glob("IMG_*_6.tif"))
    if not stacks:
        raise FileNotFoundError(f"no calibrated stacks found in {args.stack_dir}")

    total = 0
    for idx, stack_path in enumerate(stacks, start=1):
        total += split_stack(stack_path, args.raw_set, args.out_dir, args.overwrite)
        if idx % 25 == 0 or idx == len(stacks):
            print(f"[{idx}/{len(stacks)}] prepared {total} single-band TIFFs")

    print(f"prepared_stacks={len(stacks)} prepared_images={total} out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
