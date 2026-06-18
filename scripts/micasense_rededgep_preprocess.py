#!/usr/bin/env python3
"""Preprocess MicaSense RedEdge-P captures into aligned reflectance stacks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

np.mat = np.asmatrix  # Compatibility with MicaSense imageprocessing + NumPy 2.

import rasterio
import skimage.measure
import micasense.capture as capture_mod
from micasense.capture import Capture


ORIG_RANSAC = skimage.measure.ransac
BAND_SUFFIXES = ("1", "2", "3", "4", "5", "6")
MS_BAND_INDEXES = (0, 1, 2, 3, 4)
MS_BAND_NAMES = ("Blue", "Green", "Red", "NIR", "Red edge")


def compat_ransac(*args, **kwargs):
    if "random_state" in kwargs and "rng" not in kwargs:
        kwargs["rng"] = kwargs.pop("random_state")
    return ORIG_RANSAC(*args, **kwargs)


capture_mod.ransac = compat_ransac


def capture_files_from_blue(blue_path: Path) -> list[str]:
    name = blue_path.name
    if not name.endswith("_1.tif"):
        raise ValueError(f"expected *_1.tif capture seed, got {blue_path}")
    return [str(blue_path.with_name(name.replace("_1.tif", f"_{suffix}.tif"))) for suffix in BAND_SUFFIXES]


def iter_capture_seeds(folder: Path) -> Iterable[Path]:
    yield from sorted(folder.glob("**/*_1.tif"))


def load_capture(blue_path: Path) -> Capture:
    files = capture_files_from_blue(blue_path)
    missing = [path for path in files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"capture {blue_path} missing bands: {missing}")
    return Capture.from_filelist(files)


def panel_irradiance(panel_folder: Path, panel_seed: str | None) -> tuple[list[float], dict]:
    seeds = list(iter_capture_seeds(panel_folder))
    if not seeds:
        raise FileNotFoundError(f"no panel captures found in {panel_folder}")
    seed = seeds[0]
    if panel_seed:
        matches = [candidate for candidate in seeds if candidate.name.startswith(panel_seed) or candidate.stem.startswith(panel_seed)]
        if not matches:
            raise FileNotFoundError(f"panel seed {panel_seed!r} not found in {panel_folder}")
        seed = matches[0]

    cap = load_capture(seed)
    detected = int(cap.detect_panels())
    if detected < 5:
        raise RuntimeError(f"only detected {detected} panel bands for {seed}")

    irradiance = [float(value) for value in cap.panel_irradiance()]
    meta = {
        "panel_seed": str(seed),
        "panel_capture_id": cap.uuid,
        "panel_utc": cap.utc_time().isoformat(),
        "panel_location": cap.location(),
        "panel_detected_bands": detected,
        "panel_irradiance": irradiance,
        "panel_albedo": [float(value) for value in cap.panel_albedo()],
    }
    return irradiance, meta


def compute_warps(seed: Path, min_matches: int, verbose: int) -> tuple[list[np.ndarray], dict]:
    cap = load_capture(seed)
    warps = cap.SIFT_align_capture(ref=5, min_matches=min_matches, verbose=verbose)
    meta = {
        "alignment_seed": str(seed),
        "alignment_capture_id": cap.uuid,
        "alignment_utc": cap.utc_time().isoformat(),
        "alignment_location": cap.location(),
        "alignment_min_matches": min_matches,
    }
    return warps, meta


def scale_reflectance_to_uint16(stack: np.ndarray) -> np.ndarray:
    stack = np.nan_to_num(stack, nan=0.0, posinf=2.0, neginf=0.0)
    stack = np.clip(stack, 0.0, 2.0)
    return np.rint(stack * 32767.0).astype(np.uint16)


def write_stack(path: Path, aligned: np.ndarray, include_panchro: bool) -> None:
    indexes = tuple(range(aligned.shape[2])) if include_panchro else MS_BAND_INDEXES
    names = tuple(MS_BAND_NAMES) + (("Panchro",) if include_panchro else tuple())
    data = scale_reflectance_to_uint16(np.moveaxis(aligned[:, :, indexes], 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": "uint16",
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for band_index, name in enumerate(names, start=1):
            dst.set_band_description(band_index, name)
        dst.update_tags(
            software="MicaSense imageprocessing via scripts/micasense_rededgep_preprocess.py",
            reflectance_scale="32767 = 1.0 reflectance",
        )


def output_name(seed: Path) -> str:
    return seed.name.replace("_1.tif", "_6.tif")


def existing_stack_is_valid(path: Path, expected_bands: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as src:
            return src.count == expected_bands and src.width > 0 and src.height > 0
    except rasterio.errors.RasterioIOError:
        return False


def process_capture(seed: Path, outdir: Path, warps: list[np.ndarray], irradiance: list[float], include_panchro: bool) -> dict:
    out_path = outdir / output_name(seed)
    expected_bands = 6 if include_panchro else 5
    if existing_stack_is_valid(out_path, expected_bands):
        return {
            "seed": str(seed),
            "output": str(out_path),
            "skipped_existing": True,
            "bands_written": expected_bands,
        }

    cap = load_capture(seed)
    cap.radiometric_pan_sharpened_aligned_capture(warp_matrices=warps, irradiance_list=irradiance, img_type="reflectance")
    aligned = cap._Capture__aligned_radiometric_pan_sharpened_capture[1]
    write_stack(out_path, aligned, include_panchro=include_panchro)
    return {
        "seed": str(seed),
        "output": str(out_path),
        "skipped_existing": False,
        "capture_id": cap.uuid,
        "utc": cap.utc_time().isoformat(),
        "location": cap.location(),
        "shape": list(aligned.shape),
        "bands_written": 6 if include_panchro else 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-set", type=Path, required=True, help="Flight SET folder containing captures")
    parser.add_argument("--panel-set", type=Path, required=True, help="Panel SET folder used for reflectance calibration")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--alignment-seed", type=str, help="*_1.tif filename to use for SIFT alignment; default first capture")
    parser.add_argument("--panel-seed", type=str, help="panel *_1.tif filename prefix; default first panel capture")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-panchro", action="store_true", help="Write a 6-band stack including panchro")
    parser.add_argument("--min-matches", type=int, default=10)
    parser.add_argument("--verbose-align", type=int, default=1)
    args = parser.parse_args()

    all_seeds = list(iter_capture_seeds(args.input_set))
    if not all_seeds:
        raise FileNotFoundError(f"no captures found in {args.input_set}")
    seeds = all_seeds[: args.limit] if args.limit else all_seeds

    alignment_seed = all_seeds[0]
    if args.alignment_seed:
        matches = [seed for seed in all_seeds if seed.name == args.alignment_seed or seed.stem == args.alignment_seed]
        if not matches:
            raise FileNotFoundError(f"alignment seed {args.alignment_seed!r} not found in {args.input_set}")
        alignment_seed = matches[0]

    irradiance, panel_meta = panel_irradiance(args.panel_set, args.panel_seed)
    warps, alignment_meta = compute_warps(alignment_seed, args.min_matches, args.verbose_align)

    records = []
    failures = []
    skipped = 0
    for index, seed in enumerate(seeds, start=1):
        try:
            record = process_capture(seed, args.outdir, warps, irradiance, args.include_panchro)
            records.append(record)
            if record.get("skipped_existing"):
                skipped += 1
                print(f"[{index}/{len(seeds)}] skipped existing {record['output']}")
            else:
                print(f"[{index}/{len(seeds)}] wrote {record['output']}")
        except Exception as exc:
            failures.append({"seed": str(seed), "error": repr(exc)})
            print(f"[{index}/{len(seeds)}] failed {seed}: {exc}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_set": str(args.input_set),
        "outdir": str(args.outdir),
        "include_panchro": args.include_panchro,
        "panel": panel_meta,
        "alignment": alignment_meta,
        "processed": len(records),
        "skipped_existing": skipped,
        "failed": len(failures),
        "records": records,
        "failures": failures,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
