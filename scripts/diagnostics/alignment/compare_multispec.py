#!/usr/bin/env python3
"""Direct GPU vs CPU SIFT quality comparison at native multispec resolution (no rescaling).

Loads two aligned 5-band uint16 stacks (GPU + CPU), both at the same
multispec resolution (1088x1456), and computes per-band quality metrics
vs the NIR reference band. Also generates a side-by-side RGB preview PNG
and a per-band residual map.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from skimage.transform import resize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("compare_multispec")


def load_stack(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32) / 32767.0


def compute_metrics(stack_a: np.ndarray, stack_b: np.ndarray, ref_idx: int = 3,
                   names: list[str] = None) -> dict:
    """Per-band quality vs ref band. Both stacks must be same shape."""
    if names is None:
        names = ["Blue", "Green", "Red", "NIR", "Red edge"]
    ref = stack_a[ref_idx]
    metrics = {}
    for i, name in enumerate(names):
        if i == ref_idx:
            continue
        a, b = stack_a[i], stack_b[i]
        mad_a = float(np.mean(np.abs(a - ref)))
        mad_b = float(np.mean(np.abs(b - ref)))
        def psnr(x, y):
            mse = float(np.mean((x - y) ** 2))
            return 100.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)
        psnr_a, psnr_b = psnr(a, ref), psnr(b, ref)
        def grad_corr(x, y):
            gx, gy = np.diff(x, axis=1), np.diff(y, axis=1)
            gx = (gx - gx.mean()) / (gx.std() + 1e-6)
            gy = (gy - gy.mean()) / (gy.std() + 1e-6)
            return float(np.mean(gx * gy))
        gc_a, gc_b = grad_corr(a, ref), grad_corr(b, ref)
        metrics[name] = {
            "mad_a": mad_a, "mad_b": mad_b, "mad_diff": mad_b - mad_a,
            "psnr_a": psnr_a, "psnr_b": psnr_b, "psnr_diff": psnr_b - psnr_a,
            "grad_corr_a": gc_a, "grad_corr_b": gc_b, "grad_corr_diff": gc_b - gc_a,
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-stack", type=Path, required=True)
    parser.add_argument("--cpu-stack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for comparison artifacts")
    parser.add_argument("--label", default="SIFT alignment quality",
                        help="Title for the comparison")
    parser.add_argument("--gpu-name", default="GPU Kornia SIFT (num_features=2000)")
    parser.add_argument("--cpu-name", default="CPU skimage SIFT")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    gpu = load_stack(args.gpu_stack)
    cpu = load_stack(args.cpu_stack)
    log.info("GPU stack: shape=%s mean=%.4f", gpu.shape, gpu.mean())
    log.info("CPU stack: shape=%s mean=%.4f", cpu.shape, cpu.mean())
    assert gpu.shape == cpu.shape, f"Shape mismatch: GPU {gpu.shape} vs CPU {cpu.shape}"
    assert gpu.shape[1:] == (1088, 1456), (
        f"Expected multispec resolution (1088, 1456), got {gpu.shape[1:]}")

    # Quality metrics
    metrics = compute_metrics(gpu, cpu, ref_idx=3)
    table_path = args.out / "quality_table.md"
    names = list(metrics.keys())
    lines = [
        f"# {args.label}",
        "",
        f"A = `{args.gpu_name}`  |  B = `{args.cpu_name}`",
        "",
        f"Both stacks: 5 bands @ multispec native resolution (1088 × 1456), no rescaling.",
        "Reference band: NIR (index 3).",
        "",
        "ΔMAD > 0 means A is better (lower MAD vs ref).",
        "ΔPSNR > 0 means B is better. Δgrad_corr > 0 means B is better.",
        "",
        "| Band | MAD A | MAD B | ΔMAD (A−B) | PSNR A | PSNR B | ΔPSNR (B−A) | grad_corr A | grad_corr B | Δgrad_corr (B−A) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        m = metrics[name]
        lines.append(
            f"| {name} | {m['mad_a']:.4f} | {m['mad_b']:.4f} | {m['mad_diff']:+.4f} "
            f"| {m['psnr_a']:.2f} | {m['psnr_b']:.2f} | {m['psnr_diff']:+.2f} "
            f"| {m['grad_corr_a']:.4f} | {m['grad_corr_b']:.4f} | {m['grad_corr_diff']:+.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- **A** is `{args.gpu_name}` (GPU, num_features=2000, smnn matching)",
        f"- **B** is `{args.cpu_name}` (CPU, skimage, no keypoint limit)",
        "- A lower MAD means the band is more tightly aligned with the NIR reference.",
        "- A higher PSNR means smaller per-pixel error vs the reference.",
        "- A higher gradient correlation means edges are better co-registered.",
        "",
    ])
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Wrote table: %s", table_path)

    # Side-by-side RGB preview
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, stack, name in zip(axes, [gpu, cpu], [args.gpu_name, args.cpu_name]):
        rgb = stack[[2, 1, 0]]  # (3, H, W)
        rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
        lo, hi = float(np.nanpercentile(rgb, 1)), float(np.nanpercentile(rgb, 99))
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        ax.imshow(rgb)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"RGB preview at multispec native resolution (1088×1456): {args.label}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    rgb_path = args.out / "rgb_side_by_side.png"
    fig.savefig(rgb_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote RGB comparison: %s", rgb_path)

    # Per-band residual map
    non_ref = [i for i in range(5) if i != 3]
    fig, axes = plt.subplots(2, len(non_ref), figsize=(4 * len(non_ref), 8.5))
    band_names = ["Blue", "Green", "Red", "NIR", "Red edge"]
    vmax = 0
    for i in non_ref:
        vmax = max(vmax,
                   float(np.percentile(np.abs(gpu[i] - gpu[3]), 99)),
                   float(np.percentile(np.abs(cpu[i] - cpu[3]), 99)))
    for col, i in enumerate(non_ref):
        ax = axes[0, col]
        d = np.abs(gpu[i] - gpu[3])
        ax.imshow(d, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"GPU: |{band_names[i]}−NIR|", fontsize=10)
        ax.axis("off")
        ax = axes[1, col]
        d = np.abs(cpu[i] - cpu[3])
        ax.imshow(d, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"CPU: |{band_names[i]}−NIR|", fontsize=10)
        ax.axis("off")
    fig.colorbar(axes[0, 0].images[0], ax=axes.ravel().tolist(),
                 shrink=0.5, label="|residual| (reflectance)")
    fig.suptitle(f"Per-band residual vs NIR at multispec native resolution: {args.label}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    res_path = args.out / "residual_map.png"
    fig.savefig(res_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote residual map: %s", res_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
