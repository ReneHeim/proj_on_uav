#!/usr/bin/env python3
"""End-to-end v2 quality assessment: GPU Kornia SIFT (v2 worker) vs CPU ground truth.

Pipeline:
  1. Run GPU worker v2 (_gpu_sift_worker_v2.py) to fit warps for the 5
     multispec bands against the Blue ref band.
  2. Apply the warps with skimage.transform.warp at multispec native
     resolution (no resize, no crop) -> aligned 5-band stack.
  3. Write RGB preview and aligned stack.
  4. Run _gpu_similarity_metric.py against the CPU ground truth stack.

Usage:
    python -m scripts.diagnostics.alignment.quality_assessment_v2 \\
        --capture /mnt/data/ONCERCO/.../0000SET/IMG_0002_1.tif \\
        --cpu-stack outputs/quality/multispec_stacks/0000SET_IMG_0002.tif \\
        --out outputs/quality/0000SET_IMG_0002_v2 \\
        --label "0000SET / IMG_0002"
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
import numpy as np
import rasterio
from PIL import Image
from skimage.transform import ProjectiveTransform, warp as _sk_warp, resize


ROOT = Path(__file__).resolve().parents[1]


def setup_logging(script_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = ROOT / "outputs/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_name}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_file


log = logging.getLogger("qa_v2")
SCRIPT = "_quality_assessment_v2"
TORCH_CUDA_PYTHON = "/home/davidem/miniforge3/envs/torch-cuda/bin/python"
WORKER_V2 = ROOT / "scripts" / "_gpu_sift_worker_v2.py"
SIMILARITY = ROOT / "scripts" / "_gpu_similarity_metric.py"


def load_band(p: Path) -> np.ndarray:
    with rasterio.open(p) as src:
        return src.read(1).astype(np.float32)


def write_uint16_stack(path: Path, stack: np.ndarray, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(stack, 0.0, 1.0)
    data = np.rint(data * 32767.0).astype(np.uint16)
    profile = {
        "driver": "GTiff", "height": data.shape[1], "width": data.shape[2],
        "count": data.shape[0], "dtype": "uint16", "compress": "deflate", "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        for i, n in enumerate(names, start=1):
            dst.set_band_description(i, n)
        dst.update_tags(
            software="quality_assessment_v2.py",
            reflectance_scale="32767 = 1.0 reflectance",
            alignment_method="GPU Kornia SIFT v2 (Blue ref, Similarity, 1.5x upscale)",
        )


def write_rgb_preview(stack: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = stack[[2, 1, 0]].astype(np.float32)  # (3, H, W)
    rgb = np.transpose(rgb, (1, 2, 0))  # -> (H, W, 3)
    lo, hi = float(np.nanpercentile(rgb, 1)), float(np.nanpercentile(rgb, 99))
    rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(path)


def run_worker(capture_seed: Path, ref_band_idx: int = 5) -> dict:
    """Run the v2 GPU worker via subprocess, return parsed JSON output.

    ref_band_idx: 0..5 = the band index used as alignment reference.
    Default 5 = Panchro (band 6) at pan-sharpened resolution, matching
    the micasense CPU ground truth reference frame.
    """
    band_suffixes = ("1", "2", "3", "4", "5", "6")
    band_paths = [str(capture_seed.with_name(capture_seed.name.replace("_1.tif", f"_{s}.tif")))
                  for s in band_suffixes]
    cmd = [TORCH_CUDA_PYTHON, str(WORKER_V2), str(ref_band_idx)] + band_paths
    log.info("[WORKER] spawning: %s", " ".join(cmd[:4]) + " ... (6 paths)")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.perf_counter() - t0
    log.info("[WORKER] subprocess took %.2fs, returncode=%d", elapsed, proc.returncode)
    if proc.returncode != 0:
        log.error("[WORKER] stderr (last 2000 chars):\n%s", proc.stderr[-2000:])
        raise RuntimeError(f"v2 worker failed: {proc.stderr[-500:]}")
    # Worker prints progress on stdout; the final line is the JSON
    last_line = proc.stdout.strip().split("\n")[-1]
    gpu_data = json.loads(last_line)
    log.info("[WORKER] sift=%.2fs match=%.2fs total=%.2fs n_inliers_total=%d",
             gpu_data["t_sift_s"], gpu_data["t_match_s"], gpu_data["t_total_s"],
             gpu_data["n_inliers_total"])
    return gpu_data


def apply_warps_multispec(capture_seed: Path, warps: list, ref_band_idx: int = 0
                          ) -> np.ndarray:
    """Apply the fitted warps to align 5 multispec bands to the Blue ref.

    Returns (5, H, W) float32 array where H, W are the multispec native
    resolution (1088 x 1456). The panchro band (index 5) is excluded.

    micasense uses (y, x) order, so warps are applied in that order.
    The ref band is passed through as-is (identity).
    """
    band_suffixes = ("1", "2", "3", "4", "5", "6")
    band_paths = [capture_seed.with_name(capture_seed.name.replace("_1.tif", f"_{s}.tif"))
                  for s in band_suffixes]
    bands = [load_band(p) for p in band_paths]
    # Use multispec band 0 (Blue) as ref: target shape is bands[0].shape
    target_h, target_w = bands[ref_band_idx].shape
    log.info("[WARP] target (ref) shape = (%d, %d)", target_h, target_w)
    aligned = []
    valid_masks = []
    for i, b in enumerate(bands):
        if i == 5:  # skip panchro
            continue
        if i == ref_band_idx:
            aligned.append(b.astype(np.float32))
            valid_masks.append(np.ones((target_h, target_w), dtype=bool))
            continue
        # Warp
        P = ProjectiveTransform(matrix=np.array(warps[i]))
        # The worker fits P such that P(ref_coord) = moving_band_coord, i.e.
        # P maps ref -> moving. For skimage.transform.warp(image, inverse_map=M):
        # M maps OUTPUT coords to INPUT coords. Here OUTPUT=ref, INPUT=moving.
        # So we want M such that M(ref_coord) = moving_band_coord, which is P.
        # Therefore: inverse_map = P (NOT P.inverse).
        warped = _sk_warp(b, inverse_map=P, mode="constant", cval=0.0,
                          preserve_range=True, output_shape=(target_h, target_w))
        valid = _sk_warp(np.ones_like(b, dtype=np.float32), inverse_map=P,
                         mode="constant", cval=0.0,
                         preserve_range=True, output_shape=(target_h, target_w)) > 0.999
        aligned.append(warped.astype(np.float32))
        valid_masks.append(valid)
        log.info("[WARP] band %d: shape=%s -> (%d, %d)  mean before=%.3f after=%.3f",
                 i, b.shape, target_h, target_w, b.mean(), warped.mean())
    aligned_5 = np.stack(aligned)  # bands 0..4
    valid_overlap = np.logical_and.reduce(valid_masks)
    aligned_5[:, ~valid_overlap] = 0.0
    log.info("[WARP] valid all-band overlap: %.2f%%",
             100.0 * float(valid_overlap.mean()))
    # Normalize per stack: divide by 99th percentile of the entire stack
    norm = float(np.percentile(aligned_5, 99))
    if norm > 0:
        aligned_5 = aligned_5 / norm
    aligned_5 = np.clip(aligned_5, 0.0, 1.0)
    return aligned_5


def run_similarity(gpu_stack: Path, cpu_stack: Path, out_dir: Path, label: str) -> int:
    """Run the 99% similarity metric, return its return code."""
    cmd = [
        sys.executable, str(SIMILARITY),
        "--gpu-stack", str(gpu_stack),
        "--cpu-stack", str(cpu_stack),
        "--out", str(out_dir),
        "--label", label,
    ]
    log.info("[SIMILARITY] spawning: %s", " ".join(cmd[:3]) + " ...")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode not in (0, 1):
        log.error("[SIMILARITY] failed: %s", proc.stderr[-1500:])
    log.info("[SIMILARITY] returncode=%d", proc.returncode)
    # Stream the last 50 lines for context
    for line in proc.stdout.strip().split("\n")[-50:]:
        log.info("[SIMILARITY:out] %s", line)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True,
                        help="Blue-band path (e.g. .../IMG_0002_1.tif)")
    parser.add_argument("--cpu-stack", type=Path, required=True,
                        help="Path to CPU SIFT aligned 5-band uint16 stack at multispec native res")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(SCRIPT)
    log.info("=" * 70)
    log.info("v2 quality assessment for: %s", args.label)
    log.info("=" * 70)
    log.info("Log file: %s", log_file)
    log.info("Capture: %s", args.capture)
    log.info("CPU stack: %s", args.cpu_stack)

    t_total0 = time.perf_counter()

    # 1. Run GPU worker v2
    log.info("[PHASE] GPU SIFT worker v2 (ref=Blue band 0 at multispec native res)")
    gpu_data = run_worker(args.capture, ref_band_idx=0)
    # Log per-band diagnostics
    for m in gpu_data["matches_per_band"]:
        log.info("[GPU:band %d] n_features=%d n_match=%d n_inliers=%d scale=%.5f "
                 "trans=(%+.2f,%+.2f) resid_std=%.2f reject=%s",
                 m["band"] + 1, m["n_features"], m["n_matches"],
                 m["n_inliers"], m["fitted_scale"], m["fitted_trans"][0],
                 m["fitted_trans"][1], m["residual_std"], m["reject_reason"])

    # 2. Apply warps and save GPU stack at multispec native res
    log.info("[PHASE] Apply warps to multispec bands (multispec native canvas)")
    gpu_stack = apply_warps_multispec(args.capture, gpu_data["warps"], ref_band_idx=0)
    gpu_stack_path = args.out / "gpu_v2_aligned_stack.tif"
    write_uint16_stack(gpu_stack_path, gpu_stack, ["Blue", "Green", "Red", "NIR", "Red edge"])
    rgb_path = args.out / "gpu_v2_rgb.png"
    write_rgb_preview(gpu_stack, rgb_path)
    log.info("[WROTE] %s  and  %s", gpu_stack_path, rgb_path)

    # 3. Save worker output JSON for diagnostic
    worker_json = args.out / "gpu_v2_worker.json"
    worker_json.write_text(json.dumps(gpu_data, indent=2), encoding="utf-8")
    log.info("[WROTE] %s", worker_json)

    # 4. Run the 99% similarity metric
    log.info("[PHASE] 99%% similarity metric vs CPU ground truth")
    sim_rc = run_similarity(gpu_stack_path, args.cpu_stack, args.out, args.label)

    log.info("[PHASE] total: %.2fs", time.perf_counter() - t_total0)
    log.info("=" * 70)
    log.info("RESULT: similarity returncode = %d (0=pass, 1=fail, 2=shape mismatch)",
             sim_rc)
    return sim_rc


if __name__ == "__main__":
    sys.exit(main())
