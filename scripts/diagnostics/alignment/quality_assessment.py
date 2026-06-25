#!/usr/bin/env python3
"""Quality assessment for SIFT alignment methods.

Compares GPU SIFT (Kornia) vs CPU SIFT (skimage + limit_kp) vs calibrated
fallback. For each method, aligns the 6 bands, saves an aligned 5-band uint16
stack + an RGB preview PNG, then computes quantitative quality metrics
(per-band residual, MAD, gradient correlation) and a side-by-side preview
mosaic.

Usage:
    python -m scripts.diagnostics.alignment.quality_assessment \\
        --capture /mnt/data/.../IMG_0002_1.tif \\
        --cpu-stack outputs/.../IMG_0002_6.tif \\
        --label "0000SET / IMG_0002 (SIFT-friendly)" \\
        --out outputs/quality/0000SET_IMG_0002
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.gridspec import GridSpec
from PIL import Image
from skimage.transform import ProjectiveTransform, estimate_transform, resize

# Set up logging
ROOT = Path(__file__).resolve().parents[1]
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = ROOT / "outputs/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"quality_assessment_{TS}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger("quality")
log.info("Log file: %s", LOG_FILE)

# micasense from base env
sys.path.insert(0, "/home/davidem/miniforge3/lib/python3.13/site-packages")
from micasense.capture import Capture

TORCH_CUDA_PYTHON = "/home/davidem/miniforge3/envs/torch-cuda/bin/python"
GPU_WORKER = Path(__file__).parent / "_gpu_sift_worker.py"


def load_band(p: Path) -> np.ndarray:
    with rasterio.open(p) as src:
        return src.read(1).astype(np.float32)


def write_uint16_stack(path: Path, aligned_bands: np.ndarray, descriptions: list[str]) -> None:
    """Save (5, H, W) float [0, 1] array as a uint16 GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(aligned_bands, 0.0, 1.0)
    data = np.rint(data * 32767.0).astype(np.uint16)
    H, W = data.shape[1], data.shape[2]
    profile = {
        "driver": "GTiff", "height": H, "width": W, "count": data.shape[0],
        "dtype": "uint16", "compress": "deflate", "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i, d in enumerate(descriptions, start=1):
            dst.set_band_description(i, d)
        dst.update_tags(
            software="quality_assessment.py",
            reflectance_scale="32767 = 1.0 reflectance",
        )


def write_rgb_preview(stack: np.ndarray, path: Path) -> None:
    """Write a quick RGB preview from bands [2, 1, 0] of a (5, H, W) array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = stack[[2, 1, 0]].astype(np.float32)  # (3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0))  # -> (H, W, 3)
    lo, hi = float(np.nanpercentile(rgb, 1)), float(np.nanpercentile(rgb, 99))
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(path)


def run_gpu_sift_worker(capture_seed: Path, outdir: Path) -> dict:
    """Run the GPU SIFT worker, save aligned stack + preview."""
    log.info("[GPU] running worker for %s", capture_seed.name)
    outdir.mkdir(parents=True, exist_ok=True)
    band_suffixes = ("1", "2", "3", "4", "5", "6")
    band_paths = []
    for s in band_suffixes:
        p = capture_seed.with_name(capture_seed.name.replace("_1.tif", f"_{s}.tif"))
        band_paths.append(str(p))
    cmd = [TORCH_CUDA_PYTHON, str(GPU_WORKER), "5"] + band_paths
    log.info("[GPU] spawning: %s", TORCH_CUDA_PYTHON)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        log.error("[GPU] worker failed:\n%s", proc.stderr[-1000:])
        raise RuntimeError(f"GPU worker failed: {proc.stderr[-500:]}")
    gpu_data = json.loads(proc.stdout)
    log.info("[GPU] worker: sift=%.2fs match=%.2fs total=%.2fs matches=%s",
             gpu_data["t_sift_s"], gpu_data["t_match_s"], gpu_data["t_total_s"],
             [m["n_matches"] for m in gpu_data["matches_per_band"]])
    return gpu_data


def estimate_warp_from_match_indices(capture_seed: Path, gpu_data: dict) -> list[np.ndarray]:
    """Extract the warps that the GPU worker already estimated (RANSAC in worker)."""
    warps = [np.array(w) for w in gpu_data["warps"]]
    for i, m in enumerate(gpu_data["matches_per_band"]):
        log.info("[GPU] band %d: n_matches=%d  n_inliers=%d",
                 m["band"], m["n_matches"], m.get("n_inliers", 0))
    return warps


def apply_warps_and_save(capture_seed: Path, warps: list[np.ndarray], outdir: Path,
                          label: str) -> tuple[Path, Path]:
    """Apply warps to align bands, save stack + RGB preview."""
    from skimage.transform import warp as _sk_warp
    log.info("[%s] applying warps and saving stack", label)
    out_stack = outdir / f"{label}_aligned_stack.tif"
    out_png = outdir / f"{label}_rgb.png"
    band_suffixes = ("1", "2", "3", "4", "5", "6")
    band_paths = [capture_seed.with_name(capture_seed.name.replace("_1.tif", f"_{s}.tif"))
                  for s in band_suffixes]
    bands = [load_band(p) for p in band_paths]
    target_h, target_w = bands[0].shape
    for i, b in enumerate(bands):
        if b.shape != (target_h, target_w):
            bands[i] = resize(b, (target_h, target_w), preserve_range=True).astype(np.float32)
    # micasense convention: P = estimate_transform('projective', (scale*kpr)[..., ::-1], kpi)
    # P maps ref (kpr) -> image (kpi). To warp image -> ref, use the INVERSE map.
    aligned = []
    valid_masks = []
    for i, b in enumerate(bands):
        if i == 5:
            aligned.append(b)
            valid_masks.append(np.ones((target_h, target_w), dtype=bool))
            continue
        P = ProjectiveTransform(matrix=warps[i])
        # skimage.transform.warp applies the inverse_map to image coordinates
        # to produce output coordinates, i.e. output = inverse(P)(image)
        # We want image warped into ref frame: output[y,x] = P^{-1}(image[y,x])
        warped = _sk_warp(b, inverse_map=P.inverse, mode="constant", cval=0.0,
                          preserve_range=True, output_shape=(target_h, target_w))
        valid = _sk_warp(np.ones_like(b, dtype=np.float32), inverse_map=P.inverse,
                         mode="constant", cval=0.0,
                         preserve_range=True, output_shape=(target_h, target_w)) > 0.999
        aligned.append(warped.astype(np.float32))
        valid_masks.append(valid)
    aligned_5 = np.stack(aligned[:5])  # (5, H, W) — exclude panchro
    valid_overlap = np.logical_and.reduce(valid_masks[:5])
    aligned_5[:, ~valid_overlap] = 0.0
    # Normalize to [0, 1] for the stack (use band 3 NIR max as reference scale)
    norm = float(np.percentile(aligned_5, 99))
    if norm > 0:
        aligned_5 = aligned_5 / norm
    aligned_5 = np.clip(aligned_5, 0.0, 1.0)
    write_uint16_stack(out_stack, aligned_5, ["Blue", "Green", "Red", "NIR", "Red edge"])
    write_rgb_preview(aligned_5, out_png)
    log.info("[%s] saved %s and %s", label, out_stack, out_png)
    return out_stack, out_png


def compute_quality_metrics(stack_a: np.ndarray, stack_b: np.ndarray, ref_idx: int = 3) -> dict:
    """Compute alignment quality between two aligned stacks.

    For each band, computes:
    - MAD (mean absolute difference) vs ref band
    - PSNR vs ref band
    - Gradient correlation vs ref band
    Returns dict {band_name: {mad, psnr, grad_corr}}

    Stacks are resized to a common shape before comparison.
    """
    names = ["Blue", "Green", "Red", "NIR", "Red edge"]
    # Resize to common shape
    target_h = max(stack_a.shape[1], stack_b.shape[1])
    target_w = max(stack_a.shape[2], stack_b.shape[2])
    if stack_a.shape[1:] != (target_h, target_w):
        stack_a = np.stack([resize(stack_a[i], (target_h, target_w), preserve_range=True)
                            for i in range(stack_a.shape[0])])
    if stack_b.shape[1:] != (target_h, target_w):
        stack_b = np.stack([resize(stack_b[i], (target_h, target_w), preserve_range=True)
                            for i in range(stack_b.shape[0])])
    ref = stack_a[ref_idx].astype(np.float32)
    metrics = {}
    for i, name in enumerate(names):
        if i == ref_idx:
            continue
        a = stack_a[i].astype(np.float32)
        b = stack_b[i].astype(np.float32)
        mad_a = float(np.mean(np.abs(a - ref)))
        mad_b = float(np.mean(np.abs(b - ref)))
        def psnr(x, y, max_val=1.0):
            mse = float(np.mean((x - y) ** 2))
            if mse < 1e-12:
                return 100.0
            return 10.0 * np.log10(max_val ** 2 / mse)
        psnr_a = psnr(a, ref)
        psnr_b = psnr(b, ref)
        def grad_corr(x, y):
            gx = np.diff(x, axis=1)
            gy = np.diff(y, axis=1)
            gx = (gx - gx.mean()) / (gx.std() + 1e-6)
            gy = (gy - gy.mean()) / (gy.std() + 1e-6)
            return float(np.mean(gx * gy))
        gc_a = grad_corr(a, ref)
        gc_b = grad_corr(b, ref)
        metrics[name] = {
            "mad_a": mad_a, "mad_b": mad_b, "mad_diff": mad_b - mad_a,
            "psnr_a": psnr_a, "psnr_b": psnr_b, "psnr_diff": psnr_b - psnr_a,
            "grad_corr_a": gc_a, "grad_corr_b": gc_b, "grad_corr_diff": gc_b - gc_a,
        }
    return metrics


def make_mosaic(stack_paths: dict, out_path: Path, title: str) -> None:
    """Make a side-by-side RGB preview mosaic of multiple aligned stacks."""
    fig, axes = plt.subplots(1, len(stack_paths), figsize=(4.5 * len(stack_paths), 5.5))
    if len(stack_paths) == 1:
        axes = [axes]
    for ax, (label, path) in zip(axes, stack_paths.items()):
        with rasterio.open(path) as src:
            stack = src.read().astype(np.float32) / 32767.0
        rgb = stack[[2, 1, 0]]
        rgb = np.transpose(rgb, (1, 2, 0))
        lo, hi = float(np.nanpercentile(rgb, 1)), float(np.nanpercentile(rgb, 99))
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        ax.imshow(rgb)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_diff_mosaic(stack_a_path: Path, stack_b_path: Path, out_path: Path,
                     title: str, ref_band_idx: int = 3) -> None:
    """Show per-band residual vs ref for two methods side-by-side."""
    with rasterio.open(stack_a_path) as src:
        sa = src.read().astype(np.float32) / 32767.0
    with rasterio.open(stack_b_path) as src:
        sb = src.read().astype(np.float32) / 32767.0
    ref_a = sa[ref_band_idx]
    ref_b = sb[ref_band_idx]
    bands = ["Blue", "Green", "Red", "NIR", "Red edge"]
    non_ref_idx = [i for i in range(5) if i != ref_band_idx]
    fig, axes = plt.subplots(2, len(non_ref_idx), figsize=(3.5 * len(non_ref_idx), 7.0))
    for col, i in enumerate(non_ref_idx):
        # top row: method A residual
        ax = axes[0, col]
        d = np.abs(sa[i] - ref_a)
        vmax = float(np.percentile(d, 99))
        im = ax.imshow(d, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"A: |{bands[i]} - NIR|", fontsize=9)
        ax.axis("off")
        # bottom row: method B residual
        ax = axes[1, col]
        d = np.abs(sb[i] - ref_b)
        ax.imshow(d, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"B: |{bands[i]} - NIR|", fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="|residual| (reflectance)")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_quality_table(metrics: dict, out_path: Path, labels: tuple[str, str]) -> None:
    """Write a markdown table of per-band quality metrics."""
    a_label, b_label = labels
    bands = list(metrics.keys())
    lines = [
        f"# SIFT Alignment Quality Comparison",
        "",
        f"A = `{a_label}`  |  B = `{b_label}`",
        "",
        "Positive `mad_diff` means A is better (lower MAD vs ref band).",
        "Positive `psnr_diff` and `grad_corr_diff` mean B is better.",
        "",
        "| Band | MAD A | MAD B | ΔMAD (A-B) | PSNR A | PSNR B | ΔPSNR | grad_corr A | grad_corr B | Δgrad_corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for band in bands:
        m = metrics[band]
        lines.append(
            f"| {band} | {m['mad_a']:.4f} | {m['mad_b']:.4f} | {m['mad_diff']:+.4f} "
            f"| {m['psnr_a']:.2f} | {m['psnr_b']:.2f} | {m['psnr_diff']:+.2f} "
            f"| {m['grad_corr_a']:.4f} | {m['grad_corr_b']:.4f} | {m['grad_corr_diff']:+.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote quality table: %s", out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True,
                        help="Blue-band path (e.g. .../IMG_0002_1.tif)")
    parser.add_argument("--cpu-stack", type=Path, required=True,
                        help="Path to CPU SIFT aligned stack (5-band uint16)")
    parser.add_argument("--calibrated-stack", type=Path, default=None,
                        help="Optional path to calibrated fallback aligned stack")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for all artifacts")
    parser.add_argument("--label", required=True, help="Human-readable label for the capture")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log.info("=" * 70)
    log.info("Quality assessment for: %s", args.label)
    log.info("Capture: %s", args.capture)
    log.info("CPU stack: %s", args.cpu_stack)
    log.info("=" * 70)

    # 1. Run GPU SIFT and save aligned stack
    gpu_data = run_gpu_sift_worker(args.capture, args.out)
    # 2. Extract warps from worker output
    log.info("[GPU] extracting warps from worker output ...")
    warps = estimate_warp_from_match_indices(args.capture, gpu_data)
    # 3. Apply warps and save
    gpu_stack_path, gpu_png_path = apply_warps_and_save(
        args.capture, warps, args.out, "gpu_kornia_5000"
    )

    # 4. Load CPU stack
    with rasterio.open(args.cpu_stack) as src:
        cpu_stack = src.read().astype(np.float32) / 32767.0
    log.info("CPU stack: shape=%s dtype=%s", cpu_stack.shape, cpu_stack.dtype)

    # 5. Compute quality: GPU vs CPU
    log.info("Computing quality metrics: GPU SIFT vs CPU SIFT")
    cpu01 = cpu_stack
    with rasterio.open(gpu_stack_path) as src:
        gpu01 = src.read().astype(np.float32) / 32767.0
    metrics_gpu_cpu = compute_quality_metrics(gpu01, cpu01, ref_idx=3)
    write_quality_table(
        metrics_gpu_cpu, args.out / "quality_table_gpu_vs_cpu.md",
        ("GPU Kornia SIFT", "CPU skimage SIFT (ground truth)"),
    )

    # 6. If calibrated stack provided, also compare
    if args.calibrated_stack and args.calibrated_stack.exists():
        with rasterio.open(args.calibrated_stack) as src:
            cal_stack = src.read().astype(np.float32) / 32767.0
        metrics_gpu_cal = compute_quality_metrics(gpu01, cal_stack, ref_idx=3)
        write_quality_table(
            metrics_gpu_cal, args.out / "quality_table_gpu_vs_calibrated.md",
            ("GPU Kornia SIFT", "Calibrated fallback"),
        )
        # 3-way mosaic
        stack_paths = {
            "GPU SIFT (Kornia)": gpu_stack_path,
            "CPU SIFT (limit_kp_5000)": args.cpu_stack,
            "Calibrated fallback": args.calibrated_stack,
        }
        make_mosaic(stack_paths, args.out / "rgb_mosaic_3way.png",
                    f"RGB preview: {args.label}")
        # Diff mosaic: GPU vs CPU
        make_diff_mosaic(gpu_stack_path, args.cpu_stack,
                         args.out / "residual_mosaic_gpu_vs_cpu.png",
                         f"Per-band residual vs NIR: {args.label}  (GPU vs CPU)")
    else:
        stack_paths = {
            "GPU SIFT (Kornia)": gpu_stack_path,
            "CPU SIFT (limit_kp_5000)": args.cpu_stack,
        }
        make_mosaic(stack_paths, args.out / "rgb_mosaic.png",
                    f"RGB preview: {args.label}")
        make_diff_mosaic(gpu_stack_path, args.cpu_stack,
                         args.out / "residual_mosaic.png",
                         f"Per-band residual vs NIR: {args.label}  (GPU vs CPU)")

    # 7. Also save a "v2 calibrated" stack via GPU warps for reference
    log.info("Done. Artifacts in %s", args.out)


if __name__ == "__main__":
    main()
