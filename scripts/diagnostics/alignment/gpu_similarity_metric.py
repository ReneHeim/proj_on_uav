#!/usr/bin/env python3
"""99% similarity metric: GPU Kornia SIFT aligned stack vs CPU skimage SIFT aligned stack.

Compares two 5-band uint16 stacks (GPU + CPU) at multispec native resolution
(1088x1456) and computes 4 per-band metrics. The headline number is
`1 - fraction_pixels_above_1pct_reflectance` in a 90% central crop region
(this avoids edge artefacts in the comparison).

Metrics per band:
    1. PSNR (dB)             : 10 * log10(1 / MSE).            Target > 35 dB.
    2. MAD (reflectance)     : mean(|GPU - CPU|).              Target < 0.05.
    3. Fraction-off at 1%    : mean(|GPU - CPU| > 0.01).       Target < 0.01 (i.e. 99% similar).
    4. Sub-pixel shift (px)  : 2D cross-correlation peak.      Target |shift| < 0.3 px.

Usage:
    python -m scripts.diagnostics.alignment.gpu_similarity_metric \\
        --gpu-stack outputs/quality/0000SET_IMG_0002/gpu_v2_aligned_stack.tif \\
        --cpu-stack outputs/quality/multispec_stacks/0000SET_IMG_0002_6.tif \\
        --out outputs/quality/0000SET_IMG_0002 \\
        --label "0000SET / IMG_0002"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def setup_logging(script_name: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = root / "outputs/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_file


log = logging.getLogger("similarity")
SCRIPT = "_gpu_similarity_metric"


def load_stack(path: Path) -> tuple[np.ndarray, dict]:
    """Load (5, H, W) uint16 stack, return float32 [0, 1] + profile metadata."""
    t0 = time.perf_counter()
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32) / 32767.0
        profile = dict(src.profile)
        tags = dict(src.tags())
    log.info("[LOAD] %s shape=%s took=%.2fs", path.name, arr.shape, time.perf_counter() - t0)
    return arr, {"profile": profile, "tags": tags}


def central_crop_mask(shape: tuple[int, int], crop_frac: float = 0.9) -> tuple[slice, slice]:
    """Return (yslice, xslice) for the central crop_frac region."""
    h, w = shape
    dh, dw = int(h * (1 - crop_frac) / 2), int(w * (1 - crop_frac) / 2)
    return slice(dh, h - dh), slice(dw, w - dw)


def per_band_metrics(
    gpu: np.ndarray, cpu: np.ndarray, crop: tuple[slice, slice], names: list[str]
) -> dict:
    """Compute 5 metrics per band, restricted to the central crop.

    Metrics:
      1. PSNR (dB)             : 10 * log10(1 / MSE)
      2. MAD                   : mean(|GPU - CPU|)
      3. Fraction-off at 1%    : mean(|GPU - CPU| > 0.01)   (legacy intensity metric)
      4. SSIM                  : structural similarity index  (perceptual, shift-invariant)
      5. similarity_99 (legacy): 1 - frac_off

    The SSIM is the recommended primary metric because it is shift-invariant
    and matches the user's "99% similar" intuition. Intensity metrics (PSNR,
    MAD, frac_off) are reported as secondary diagnostics.
    """
    from skimage.metrics import structural_similarity as _ssim

    metrics = {}
    for i, name in enumerate(names):
        t0 = time.perf_counter()
        a = gpu[i][crop].astype(np.float32)
        b = cpu[i][crop].astype(np.float32)
        diff = a - b
        absdiff = np.abs(diff)
        mse = float(np.mean(diff**2))
        psnr = 100.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)
        mad = float(np.mean(absdiff))
        frac_off = float(np.mean(absdiff > 0.01))
        # SSIM with data_range=1.0 (reflectance in [0, 1])
        ssim_val = float(_ssim(a, b, data_range=1.0))
        metrics[name] = {
            "psnr_db": psnr,
            "mad": mad,
            "frac_off": frac_off,
            "ssim": ssim_val,
            "similarity_99": 1.0 - frac_off,
            "ssim_similarity_99": ssim_val,  # alias
        }
        log.info(
            "[METRIC] band %d (%s) PSNR=%.2f dB MAD=%.4f frac_off=%.4f " "SSIM=%.4f took=%.3fs",
            i,
            name,
            psnr,
            mad,
            frac_off,
            ssim_val,
            time.perf_counter() - t0,
        )
    return metrics


def subpixel_shift_2d(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Estimate sub-pixel translation (dy, dx) by phase correlation of two images.

    Returns the shift that maps b -> a, in (row, col) = (y, x) pixel units.
    Uses FFT-based phase correlation (fast, exact, sub-pixel via parabolic
    peak fitting in the cross-power spectrum).

    Reference: Reddy & Chatterji 1996, "An FFT-Based Technique for Translation,
    Rotation and Scale-Invariant Image Registration", IEEE TIP 5(8):1266-1271.
    """
    t0 = time.perf_counter()
    h, w = a.shape
    if a.shape != b.shape:
        # resize to common shape
        from skimage.transform import resize

        if a.shape != b.shape:
            b = resize(b, a.shape, preserve_range=True).astype(np.float32)
    a_n = a - a.mean()
    b_n = b - b.mean()
    F_a = np.fft.fft2(a_n)
    F_b = np.fft.fft2(b_n)
    cross = F_a * np.conj(F_b)
    norm = np.abs(cross) + 1e-12
    cross /= norm
    pcm = np.real(np.fft.ifft2(cross))  # phase correlation magnitude, peaks at best shift
    # Integer peak
    y0, x0 = np.unravel_index(int(np.argmax(pcm)), pcm.shape)
    # Sub-pixel refinement via parabolic fit on the 3x3 neighborhood
    y0p = (y0 - 1) % h
    y0n = (y0 + 1) % h
    x0p = (x0 - 1) % w
    x0n = (x0 + 1) % w
    c = pcm[y0, x0]
    up = pcm[y0p, x0]
    dn = pcm[y0n, x0]
    lp = pcm[y0, x0p]
    rn = pcm[y0, x0n]
    denom_y = 2.0 * c - up - dn
    denom_x = 2.0 * c - lp - rn
    dy_sub = 0.0 if abs(denom_y) < 1e-12 else (up - dn) / denom_y
    dx_sub = 0.0 if abs(denom_x) < 1e-12 else (lp - rn) / denom_x
    # Sign convention: positive shift means a is translated by (+dy, +dx) relative to b
    if y0 > h / 2:
        y0 -= h
    if x0 > w / 2:
        x0 -= w
    dy = float(y0 + dy_sub)
    dx = float(x0 + dx_sub)
    log.info("[SHIFT] dy=%.3f dx=%.3f took=%.3fs", dy, dx, time.perf_counter() - t0)
    return dy, dx


def per_band_shift(gpu: np.ndarray, cpu: np.ndarray, names: list[str]) -> dict:
    """Per-band sub-pixel shift of GPU vs CPU. Uses Green band for both as proxy
    to find the dominant shift, since both should be aligned to band 0 (Blue)
    or band 5 (Panchro) in the micasense / Kornia pipelines respectively."""
    # We use the per-band-vs-reference residual to detect a *consistent* shift.
    # The actual shift is computed on the average stack (mean of all 5 bands)
    # which is the most signal-rich.
    avg_gpu = np.mean(gpu, axis=0)
    avg_cpu = np.mean(cpu, axis=0)
    log.info("[SHIFT] using average of 5 bands for sub-pixel shift estimation")
    dy, dx = subpixel_shift_2d(avg_gpu, avg_cpu)
    out = {name: {"dy_px": dy, "dx_px": dx, "mag_px": float(np.hypot(dy, dx))} for name in names}
    return out


def make_diff_mosaic(
    gpu: np.ndarray, cpu: np.ndarray, out_path: Path, title: str, names: list[str]
) -> None:
    """Side-by-side: GPU band | CPU band | |GPU - CPU| heatmap, for non-ref bands."""
    non_ref = [i for i in range(5) if i != 3]  # exclude NIR as "ref" (for visualisation)
    fig, axes = plt.subplots(3, len(non_ref), figsize=(3.2 * len(non_ref), 9.0))
    for col, i in enumerate(non_ref):
        for row, (arr, lbl) in enumerate(
            [(gpu[i], "GPU"), (cpu[i], "CPU"), (np.abs(gpu[i] - cpu[i]), "|GPU-CPU|")]
        ):
            ax = axes[row, col]
            if row < 2:
                lo, hi = float(np.nanpercentile(arr, 1)), float(np.nanpercentile(arr, 99))
                ax.imshow(arr, cmap="gray", vmin=lo, vmax=hi)
            else:
                vmax = float(np.percentile(arr, 99))
                ax.imshow(arr, cmap="magma", vmin=0, vmax=vmax)
            ax.set_title(f"{lbl}: {names[i]}", fontsize=9)
            ax.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("[PLOT] %s", out_path)


def make_shift_heatmap(
    gpu: np.ndarray, cpu: np.ndarray, out_path: Path, title: str, names: list[str]
) -> None:
    """Per-band |GPU - CPU| heatmaps with shared vmax, in a single row."""
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.6))
    vmax = max(float(np.percentile(np.abs(gpu[i] - cpu[i]), 99)) for i in range(n))
    for i, name in enumerate(names):
        ax = axes[i]
        d = np.abs(gpu[i] - cpu[i])
        ax.imshow(d, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{name}", fontsize=10)
        ax.axis("off")
    fig.colorbar(
        axes[-1].images[0], ax=axes.tolist(), shrink=0.7, label="|GPU - CPU| (reflectance)"
    )
    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info("[PLOT] %s", out_path)


def write_markdown_summary(metrics: dict, shifts: dict, out_path: Path, args, target: dict) -> None:
    """Write paper-grade markdown table."""
    names = list(metrics.keys())
    lines = [
        f"# GPU SIFT 99% Similarity vs CPU SIFT Ground Truth",
        "",
        f"**Label**: {args.label}",
        f"**GPU stack**: `{args.gpu_stack}`",
        f"**CPU stack**: `{args.cpu_stack}`",
        f"**Crop region**: 90% central (avoids edge artefacts)",
        "",
        "## Targets",
        "",
        f"- PSNR > {target['psnr_db']} dB",
        f"- MAD < {target['mad']} reflectance units",
        f"- 1 - fraction_off > {target['similarity_99']}  (i.e. 99% pixels within 1% reflectance)",
        f"- |sub-pixel shift| < {target['shift_px']} px",
        "",
        "## Per-band metrics",
        "",
        "| Band | PSNR (dB) | MAD | frac_off (1%) | 99% similarity | dy (px) | dx (px) | |shift| (px) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        m = metrics[name]
        s = shifts[name]
        psnr_ok = "OK" if m["psnr_db"] > target["psnr_db"] else "FAIL"
        mad_ok = "OK" if m["mad"] < target["mad"] else "FAIL"
        sim_ok = "OK" if m["similarity_99"] > target["similarity_99"] else "FAIL"
        shift_ok = "OK" if s["mag_px"] < target["shift_px"] else "FAIL"
        verdict = (
            "PASS"
            if (
                m["psnr_db"] > target["psnr_db"]
                and m["mad"] < target["mad"]
                and m["similarity_99"] > target["similarity_99"]
                and s["mag_px"] < target["shift_px"]
            )
            else "FAIL"
        )
        lines.append(
            f"| {name} | {m['psnr_db']:.2f} {psnr_ok} | {m['mad']:.4f} {mad_ok} "
            f"| {m['frac_off']:.4f} | {m['similarity_99']:.4f} {sim_ok} "
            f"| {s['dy_px']:+.3f} | {s['dx_px']:+.3f} | {s['mag_px']:.3f} {shift_ok} "
            f"| **{verdict}** |"
        )
    avg_sim = float(np.mean([metrics[n]["similarity_99"] for n in names]))
    avg_psnr = float(np.mean([metrics[n]["psnr_db"] for n in names]))
    avg_mad = float(np.mean([metrics[n]["mad"] for n in names]))
    avg_shift = float(np.mean([shifts[n]["mag_px"] for n in names]))
    lines.extend(
        [
            f"| **AVERAGE** | **{avg_psnr:.2f}** | **{avg_mad:.4f}** "
            f"| **{float(np.mean([metrics[n]['frac_off'] for n in names])):.4f}** "
            f"| **{avg_sim:.4f}** | -- | -- | **{avg_shift:.3f}** | -- |",
            "",
            "## Interpretation",
            "",
            f"- **Headline 99% similarity** (mean across bands): **{avg_sim * 100:.2f}%** of pixels "
            f"are within 1% reflectance of the CPU ground truth (in the 90% central crop).",
            f"- **PSNR** (mean): **{avg_psnr:.2f} dB** "
            f"{'above' if avg_psnr > target['psnr_db'] else 'BELOW'} the {target['psnr_db']} dB target.",
            f"- **MAD** (mean): **{avg_mad:.4f}** "
            f"{'below' if avg_mad < target['mad'] else 'ABOVE'} the {target['mad']} target.",
            f"- **Sub-pixel shift** (mean |dy,dx|): **{avg_shift:.3f} px** "
            f"{'below' if avg_shift < target['shift_px'] else 'ABOVE'} the {target['shift_px']} px target.",
            "",
            "OK = metric meets target. FAIL = metric does not. PASS = all 4 metrics meet their target.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("[WRITE] %s", out_path)
    return avg_sim, avg_psnr, avg_mad, avg_shift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-stack", type=Path, required=True)
    parser.add_argument("--cpu-stack", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--crop-frac", type=float, default=0.9, help="Central crop fraction (default 0.9)"
    )
    parser.add_argument("--psnr-target", type=float, default=35.0)
    parser.add_argument("--mad-target", type=float, default=0.05)
    parser.add_argument("--similarity-target", type=float, default=0.99)
    parser.add_argument("--shift-target", type=float, default=0.3)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(SCRIPT)
    log.info("=" * 70)
    log.info("GPU SIFT 99%% similarity metric")
    log.info("=" * 70)
    log.info("Log file: %s", log_file)
    log.info("GPU: %s", args.gpu_stack)
    log.info("CPU: %s", args.cpu_stack)
    log.info("Crop: %.2f central", args.crop_frac)

    t_total0 = time.perf_counter()
    gpu, gpu_meta = load_stack(args.gpu_stack)
    cpu, cpu_meta = load_stack(args.cpu_stack)
    if gpu.shape != cpu.shape:
        log.error("Shape mismatch: GPU %s vs CPU %s", gpu.shape, cpu.shape)
        return 2
    log.info("Stack shape: %s (5 bands @ %dx%d)", gpu.shape, gpu.shape[2], gpu.shape[1])

    h, w = gpu.shape[1], gpu.shape[2]
    crop = central_crop_mask((h, w), args.crop_frac)
    log.info(
        "Central crop: y=%d:%d  x=%d:%d  (of %dx%d)",
        crop[0].start,
        crop[0].stop,
        crop[1].start,
        crop[1].stop,
        h,
        w,
    )

    names = ["Blue", "Green", "Red", "NIR", "Red edge"]
    metrics = per_band_metrics(gpu, cpu, crop, names)
    shifts = per_band_shift(gpu, cpu, names)

    target = {
        "psnr_db": args.psnr_target,
        "mad": args.mad_target,
        "similarity_99": args.similarity_target,
        "shift_px": args.shift_target,
    }
    avg_sim, avg_psnr, avg_mad, avg_shift = write_markdown_summary(
        metrics, shifts, args.out / "gpu_vs_cpu_similarity.md", args, target
    )

    # JSON dump
    json_path = args.out / "gpu_vs_cpu_similarity.json"
    json_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "crop_frac": args.crop_frac,
                "targets": target,
                "metrics": metrics,
                "shifts": shifts,
                "summary": {
                    "avg_similarity_99": avg_sim,
                    "avg_psnr_db": avg_psnr,
                    "avg_mad": avg_mad,
                    "avg_shift_px": avg_shift,
                },
                "log_file": str(log_file),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("[WRITE] %s", json_path)

    # Plots
    make_diff_mosaic(
        gpu, cpu, args.out / "gpu_vs_cpu_mosaic.png", f"GPU vs CPU per band: {args.label}", names
    )
    make_shift_heatmap(
        gpu, cpu, args.out / "gpu_vs_cpu_absdiff.png", f"|GPU - CPU| per band: {args.label}", names
    )

    log.info("[PHASE] total: %.2fs", time.perf_counter() - t_total0)
    log.info(
        "[SUMMARY] avg 99%% similarity = %.4f  (target %.2f)", avg_sim, target["similarity_99"]
    )
    log.info("=" * 70)
    if (
        avg_sim < target["similarity_99"]
        or avg_psnr < target["psnr_db"]
        or avg_mad > target["mad"]
        or avg_shift > target["shift_px"]
    ):
        log.info("RESULT: 99%% similarity target NOT met for %s", args.label)
        return 1
    log.info("RESULT: 99%% similarity target met for %s", args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
