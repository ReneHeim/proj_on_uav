#!/usr/bin/env python3
"""Split calibrated MicaSense stacks into ODM-readable TIFFs."""

from __future__ import annotations

import argparse
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

BAND_SUFFIXES = ("1", "2", "3", "4", "5")
BAND_NAMES = ("Blue", "Green", "Red", "Red edge", "NIR")


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


def parse_bands(value: str) -> set[int]:
    bands = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        band = int(item)
        if band < 1 or band > 5:
            raise ValueError(f"band must be in 1..5, got {band}")
        bands.add(band)
    if not bands:
        raise ValueError("at least one band must be selected")
    return bands


def prefixed_name(stack_path: Path, suffix: str, name_prefix: str) -> str:
    return f"{name_prefix}{stack_path.name.replace('_6.tif', f'_{suffix}.tif')}"


def split_stack(
    stack_path: Path,
    raw_set: Path,
    out_dir: Path,
    overwrite: bool,
    selected_bands: set[int],
    as_byte_rgb: bool,
    name_prefix: str,
) -> int:
    count = 0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        src_cm = rasterio.open(stack_path)
    with src_cm as src:
        if src.count != 5:
            raise ValueError(f"expected 5 bands in {stack_path}, found {src.count}")
        for band_index, (suffix, band_name) in enumerate(zip(BAND_SUFFIXES, BAND_NAMES), start=1):
            if band_index not in selected_bands:
                continue
            out_name = prefixed_name(stack_path, suffix, name_prefix)
            out_path = out_dir / out_name
            if out_path.exists() and not overwrite:
                count += 1
                continue

            data = src.read(band_index)
            profile = src.profile.copy()
            if as_byte_rgb:
                data = np.clip(data.astype(np.float32), 0, 32767) * (255.0 / 32767.0)
                out_data = np.repeat(data.astype(np.uint8)[np.newaxis, :, :], 3, axis=0)
                profile.update(
                    count=3,
                    dtype="uint8",
                    compress="deflate",
                    tiled=True,
                    photometric="RGB",
                )
            else:
                out_data = data[np.newaxis, :, :]
                profile.update(count=1, compress="deflate", tiled=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(out_data)
                    if as_byte_rgb:
                        dst.set_band_description(1, f"{band_name} byte R")
                        dst.set_band_description(2, f"{band_name} byte G")
                        dst.set_band_description(3, f"{band_name} byte B")
                    else:
                        dst.set_band_description(1, band_name)
                    dst.update_tags(
                        software="Prepared for ODM from calibrated MicaSense stack",
                        source_stack=str(stack_path),
                        reflectance_scale="32767 = 1.0 reflectance",
                        calibrated_band=band_name,
                        odm_format=(
                            "8-bit RGB duplicate band" if as_byte_rgb else "single-band uint16"
                        ),
                    )

            copy_metadata(raw_band_path(raw_set, stack_path.name, suffix), out_path)
            count += 1
    return count


def write_rgb_composite(
    stack_path: Path,
    raw_set: Path,
    out_dir: Path,
    overwrite: bool,
    name_prefix: str,
) -> int:
    out_name = prefixed_name(stack_path, "3", name_prefix)
    out_path = out_dir / out_name
    if out_path.exists() and not overwrite:
        return 1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        src_cm = rasterio.open(stack_path)
    with src_cm as src:
        if src.count != 5:
            raise ValueError(f"expected 5 bands in {stack_path}, found {src.count}")

        red = src.read(3)
        green = src.read(2)
        blue = src.read(1)
        rgb = np.stack([red, green, blue]).astype(np.float32)
        rgb = np.clip(rgb, 0, 32767) * (255.0 / 32767.0)
        out_data = rgb.astype(np.uint8)

        profile = src.profile.copy()
        profile.update(
            count=3,
            dtype="uint8",
            compress="deflate",
            tiled=True,
            photometric="RGB",
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out_data)
                dst.set_band_description(1, "Red byte R")
                dst.set_band_description(2, "Green byte G")
                dst.set_band_description(3, "Blue byte B")
                dst.update_tags(
                    software="Prepared for ODM from calibrated MicaSense stack",
                    source_stack=str(stack_path),
                    reflectance_scale="32767 = 1.0 reflectance",
                    calibrated_band="RGB composite from Red, Green, Blue",
                    odm_format="8-bit RGB composite",
                )

    copy_metadata(raw_band_path(raw_set, stack_path.name, "3"), out_path)
    return 1


def process_stack(args_tuple: tuple) -> int:
    (
        stack_path,
        raw_set,
        out_dir,
        overwrite,
        selected_bands,
        as_byte_rgb,
        rgb_composite,
        name_prefix,
    ) = args_tuple
    if rgb_composite:
        return write_rgb_composite(stack_path, raw_set, out_dir, overwrite, name_prefix)
    return split_stack(
        stack_path,
        raw_set,
        out_dir,
        overwrite,
        selected_bands,
        as_byte_rgb,
        name_prefix,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stack-dir", type=Path, required=True, help="Folder of calibrated IMG_*_6.tif stacks"
    )
    parser.add_argument(
        "--raw-set", type=Path, required=True, help="Matching raw MicaSense SET folder"
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output ODM images directory")
    parser.add_argument(
        "--bands",
        default="1,2,3,4,5",
        help="Comma-separated 1-based bands to export; use 3 for Red-only ODM geometry",
    )
    parser.add_argument(
        "--as-byte-rgb",
        action="store_true",
        help="Write each selected band as 8-bit RGB for ODM/OpenMVS dense reconstruction.",
    )
    parser.add_argument(
        "--rgb-composite",
        action="store_true",
        help="Write one RGB image per stack using calibrated Red, Green, Blue channels.",
    )
    parser.add_argument(
        "--name-prefix",
        default="",
        help="Prefix output filenames, useful when combining multiple SET folders.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stacks = sorted(args.stack_dir.glob("IMG_*_6.tif"))
    if not stacks:
        raise FileNotFoundError(f"no calibrated stacks found in {args.stack_dir}")

    total = 0
    selected_bands = parse_bands(args.bands)
    jobs = [
        (
            stack_path,
            args.raw_set,
            args.out_dir,
            args.overwrite,
            selected_bands,
            args.as_byte_rgb,
            args.rgb_composite,
            args.name_prefix,
        )
        for stack_path in stacks
    ]
    if args.workers <= 1:
        for idx, job in enumerate(jobs, start=1):
            total += process_stack(job)
            if idx % 25 == 0 or idx == len(stacks):
                print(f"[{idx}/{len(stacks)}] prepared {total} ODM TIFFs", flush=True)
    else:
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_stack, job) for job in jobs]
            for future in as_completed(futures):
                total += future.result()
                done += 1
                if done % 25 == 0 or done == len(stacks):
                    print(f"[{done}/{len(stacks)}] prepared {total} ODM TIFFs", flush=True)

    print(f"prepared_stacks={len(stacks)} prepared_images={total} out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
