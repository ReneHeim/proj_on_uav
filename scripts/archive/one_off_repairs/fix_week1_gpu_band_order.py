"""Normalize week1 GPU RedEdge-P stack band order.

The week1 GPU output was produced by mixed script versions. SETs 0001 and
0002 were written in raw RedEdge-P order (B, G, R, NIR, Red edge), while the
analysis pipeline expects Metashape-compatible order (B, G, R, Red edge, NIR).
This script swaps bands 4 and 5 for only those SETs and writes band
descriptions for every stack.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from pathlib import Path

import numpy as np
import rasterio

EXPECTED_DESCRIPTIONS = ("Blue", "Green", "Red", "Red edge", "NIR")
DEFAULT_ROOT = Path(
    "/mnt/data/ONCERCO/processing/local_odm_projects/" "2025_rededgep_no_correction_v3/week1_gpu"
)


def setup_logging(script_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = Path("outputs/archive/legacy_unscoped/logs") / f"{script_name}_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return log_path


def list_stacks(root: Path) -> list[Path]:
    return sorted(root.glob("*SET/preprocessed_stacks/IMG_*_6.tif"))


def write_descriptions(path: Path) -> None:
    with rasterio.open(path, "r+") as ds:
        for idx, desc in enumerate(EXPECTED_DESCRIPTIONS, start=1):
            ds.set_band_description(idx, desc)


def swap_rededge_nir(path: Path) -> dict[str, object]:
    t0 = time.time()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        tags = src.tags()
        band_tags = {idx: src.tags(idx) for idx in range(1, src.count + 1)}
        arr = src.read()

    if arr.shape[0] != 5:
        raise ValueError(f"Expected 5 bands, found {arr.shape[0]} in {path}")

    before_medians = np.nanmedian(arr.reshape(arr.shape[0], -1), axis=1)
    fixed = arr[[0, 1, 2, 4, 3], :, :]
    after_medians = np.nanmedian(fixed.reshape(fixed.shape[0], -1), axis=1)

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(fixed)
        dst.update_tags(**tags)
        for idx, desc in enumerate(EXPECTED_DESCRIPTIONS, start=1):
            dst.set_band_description(idx, desc)
            dst.update_tags(idx, **band_tags.get(idx, {}))

    os.replace(tmp_path, path)
    return {
        "path": str(path),
        "set": path.parts[-3],
        "action": "swap_band4_band5",
        "before_band4_median_uint16": float(before_medians[3]),
        "before_band5_median_uint16": float(before_medians[4]),
        "after_band4_median_uint16": float(after_medians[3]),
        "after_band5_median_uint16": float(after_medians[4]),
        "elapsed_s": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--swap-sets",
        nargs="+",
        default=["0001SET", "0002SET"],
        help="SET folders whose stacks are still in raw B,G,R,NIR,RE order.",
    )
    args = parser.parse_args()

    script_name = Path(__file__).stem
    log_path = setup_logging(script_name)
    report_dir = Path("outputs/archive/legacy_unscoped/reports/metashape_custom_matchtest")
    report_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    t0 = time.time()
    stacks = list_stacks(args.root)
    logging.info("[PHASE] discover stacks: %.1fs n=%d", time.time() - t0, len(stacks))

    swap_sets = set(args.swap_sets)
    swap_paths = [p for p in stacks if p.parts[-3] in swap_sets]
    metadata_only_paths = [p for p in stacks if p.parts[-3] not in swap_sets]
    logging.info(
        "[INPUT] swap_sets=%s swap_files=%d metadata_only_files=%d",
        sorted(swap_sets),
        len(swap_paths),
        len(metadata_only_paths),
    )

    rows: list[dict[str, object]] = []
    t0 = time.time()
    for idx, path in enumerate(swap_paths, start=1):
        rows.append(swap_rededge_nir(path))
        if idx % 25 == 0:
            logging.info("[PROGRESS] swapped %d/%d", idx, len(swap_paths))
    logging.info("[PHASE] swap affected stacks: %.1fs", time.time() - t0)

    t0 = time.time()
    for idx, path in enumerate(metadata_only_paths, start=1):
        write_descriptions(path)
        rows.append({"path": str(path), "set": path.parts[-3], "action": "metadata_only"})
        if idx % 100 == 0:
            logging.info("[PROGRESS] metadata %d/%d", idx, len(metadata_only_paths))
    logging.info("[PHASE] write metadata descriptions: %.1fs", time.time() - t0)

    t0 = time.time()
    manifest_path = report_dir / "week1_gpu_band_order_fix_manifest.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = report_dir / "week1_gpu_band_order_fix_summary.md"
    with summary_path.open("w") as f:
        f.write("# Week1 GPU band-order repair\n\n")
        f.write(f"Root: `{args.root}`\n\n")
        f.write("## Action\n\n")
        f.write(
            "Swapped bands 4 and 5 for SETs `0001SET` and `0002SET`, converting "
            "`Blue, Green, Red, NIR, Red edge` to `Blue, Green, Red, Red edge, NIR`.\n\n"
        )
        f.write(
            "Wrote band descriptions for all stacks as: "
            "`Blue`, `Green`, `Red`, `Red edge`, `NIR`.\n\n"
        )
        f.write("## Counts\n\n")
        f.write(f"- Total stacks discovered: `{len(stacks)}`\n")
        f.write(f"- Stacks with band 4/5 swapped: `{len(swap_paths)}`\n")
        f.write(f"- Metadata-only stacks: `{len(metadata_only_paths)}`\n\n")
        f.write("## Outputs\n\n")
        f.write(f"- Manifest: `{manifest_path}`\n")
        f.write(f"- Log: `{log_path}`\n\n")
        f.write("## Reproducibility\n\n")
        f.write(f"- Script: `{Path(__file__)}`\n")
        f.write(f"- Swap sets: `{sorted(swap_sets)}`\n")
        f.write("- Random seed: not used\n")
    logging.info("[PHASE] write manifest/report: %.1fs", time.time() - t0)
    logging.info("[PHASE] total: %.1fs", time.time() - t_total)
    print(manifest_path)
    print(summary_path)
    print(log_path)


if __name__ == "__main__":
    main()
