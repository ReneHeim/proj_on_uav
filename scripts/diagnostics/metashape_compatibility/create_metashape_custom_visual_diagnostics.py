#!/usr/bin/env python3
"""Create visual diagnostics for matched Metashape/custom RedEdge-P products."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

REFLECTANCE_SCALE = 32767.0
PAIRS = (
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
    path = Path("outputs/archive/legacy_unscoped/logs") / f"create_metashape_custom_visual_diagnostics_{timestamp}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path)],
        force=True,
    )
    return path


def read_stack(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read().astype("float32")[:5]
        profile = {
            "shape": (src.count, src.height, src.width),
            "dtype": src.dtypes[0],
            "crs": src.crs.to_string() if src.crs else None,
            "descriptions": src.descriptions,
        }
    if profile["dtype"] == "uint16" or float(np.nanmax(data)) > 4.0:
        data /= REFLECTANCE_SCALE
    return data, profile


def robust_rgb(stack: np.ndarray) -> np.ndarray:
    rgb = np.stack([stack[2], stack[1], stack[0]], axis=-1)
    out = np.zeros_like(rgb, dtype="float32")
    for i in range(3):
        band = rgb[:, :, i]
        valid = band[np.isfinite(band) & (band > 0)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        out[:, :, i] = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def gray_panel(
    ax, arr: np.ndarray, title: str, cmap: str, vmin: float | None = None, vmax: float | None = None
):
    valid = arr[np.isfinite(arr)]
    if vmin is None and valid.size:
        vmin = float(np.percentile(valid, 2))
    if vmax is None and valid.size:
        vmax = float(np.percentile(valid, 98))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    return im


def make_figure(
    path: Path,
    week: str,
    capture: str,
    ref: np.ndarray,
    custom: np.ndarray,
    ref_meta: dict,
    custom_meta: dict,
) -> None:
    ref_ndvi = (ref[4] - ref[2]) / np.maximum(ref[4] + ref[2], 1e-6)
    custom_ndvi = (custom[4] - custom[2]) / np.maximum(custom[4] + custom[2], 1e-6)
    ref_rg = ref[2] / np.maximum(ref[1], 1e-6)
    custom_rg = custom[2] / np.maximum(custom[1], 1e-6)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    axes[0, 0].imshow(robust_rgb(ref))
    axes[0, 0].set_title(f"Metashape RGB\n{ref_meta['shape']} CRS={ref_meta['crs']}")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(robust_rgb(custom))
    axes[1, 0].set_title(f"Custom RGB\n{custom_meta['shape']} CRS={custom_meta['crs']}")
    axes[1, 0].axis("off")

    panels = [
        (0, 1, ref[2], "Metashape Red", "magma", 0, 0.12),
        (1, 1, custom[2], "Custom Red", "magma", 0, 0.12),
        (0, 2, ref_rg, "Metashape Red/Green", "viridis", 0, 1.6),
        (1, 2, custom_rg, "Custom Red/Green", "viridis", 0, 1.6),
        (0, 3, ref_ndvi, "Metashape NDVI", "RdYlGn", 0, 1),
        (1, 3, custom_ndvi, "Custom NDVI", "RdYlGn", 0, 1),
    ]
    for row, col, arr, title, cmap, vmin, vmax in panels:
        im = gray_panel(axes[row, col], arr, title, cmap, vmin, vmax)
        fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"{week} {capture}: Metashape orthophoto vs custom camera-frame stack", fontsize=14
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/archive/legacy_unscoped/figures/metashape_custom_visual_diagnostics")
    )
    args = parser.parse_args()
    log_path = setup_logging()
    t0 = time.perf_counter()
    outputs = []
    for week, capture, ref_path, custom_path in PAIRS:
        ref_path = Path(ref_path)
        custom_path = Path(custom_path)
        if not ref_path.exists() or not custom_path.exists():
            logging.warning("[SKIP] missing pair %s %s", ref_path, custom_path)
            continue
        ref, ref_meta = read_stack(ref_path)
        custom, custom_meta = read_stack(custom_path)
        out = args.out_dir / f"{week}_{capture}_visual_diagnostic.png"
        make_figure(out, week, capture, ref, custom, ref_meta, custom_meta)
        outputs.append(out)
        logging.info("[WRITE] %s", out)
    logging.info("[PHASE] create visual diagnostics: %.1fs", time.perf_counter() - t0)
    logging.info("[DONE] outputs=%s log=%s", outputs, log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
