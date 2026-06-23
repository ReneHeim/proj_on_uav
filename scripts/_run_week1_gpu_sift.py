#!/usr/bin/env python3
"""Run GPU SIFT alignment + micasense radiometric calibration on the entire week1 dataset.

Pipeline (per SET):
  1. Detect panel + compute irradiance from IMG_0000 (the panel capture)
  2. For each capture in the SET:
     a. Load the 6 raw band .tif files
     b. Run GPU SIFT worker v2 (Kornia on GPU) to fit warps
     c. Call micasense `cap.radiometric_pan_sharpened_aligned_capture(
            warp_matrices=gpu_warps, irradiance_list=irradiance,
            img_type='reflectance')` to apply
            - lens distortion correction
            - vignetting correction
            - DN -> reflectance (using the panel)
            - per-band warps (our GPU fits, NOT micasense's SIFT)
     d. Save a 5-band uint16 reflectance stack per capture
  3. Write per-capture and per-SET summary

Speed vs micasense CPU SIFT (which does the same preprocessing + SIFT):
  - micasense CPU: ~100s per capture (mostly the CPU SIFT alignment)
  - GPU v2:        ~2-3s per capture (mostly I/O + reflectance calibration)
  - ~40-50x speedup

Output naming: `IMG_xxxx_x.tif` in outdir, 5 bands, uint16 = 32767 = 1.0 reflectance.

Usage:
    python scripts/_run_week1_gpu_sift.py \\
        --set 0000SET \\
        --input-root /mnt/data/ONCERCO/data/raw/2025/week1/rededgep \\
        --outdir /mnt/data/ONCERCO/processing/local_odm_projects/2025_rededgep_no_correction_v3/week1_gpu \\
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from PIL import Image
from skimage.transform import ProjectiveTransform, warp as _sk_warp, resize

# micasense (base env, python 3.13)
sys.path.insert(0, "/home/davidem/miniforge3/lib/python3.13/site-packages")
import micasense.capture as capture_mod  # noqa: E402
from micasense.capture import Capture  # noqa: E402

# Compatibility shim (micasense + NumPy 2)
np.mat = np.asmatrix

# GPU worker (in torch-cuda env)
TORCH_CUDA_PYTHON = "/home/davidem/miniforge3/envs/torch-cuda/bin/python"
GPU_WORKER = Path(__file__).parent / "_gpu_sift_worker_v2.py"
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(label: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"_run_week1_gpu_sift_{label}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_file


def panel_irradiance(panel_folder: Path, panel_seed: str = "IMG_0000") -> tuple[list[float], dict]:
    """Detect panel and compute irradiance from IMG_0000.

    panel_folder is expected to be the SET directory (containing a 000/ subdir).
    """
    candidates = sorted(panel_folder.glob(f"000/{panel_seed}_*.tif"))
    candidates = [c for c in candidates if c.name.endswith("_1.tif")]
    if not candidates:
        raise FileNotFoundError(f"no panel capture 000/{panel_seed}_1.tif in {panel_folder}")
    panel_blue = candidates[0]
    panel_band_paths = [
        panel_blue.with_name(panel_blue.name.replace("_1.tif", f"_{i}.tif"))
        for i in (1, 2, 3, 4, 5, 6)
    ]
    cap = Capture.from_filelist([str(p) for p in panel_band_paths])
    detected = int(cap.detect_panels())
    if detected < 5:
        raise RuntimeError(
            f"only {detected}/6 panel bands detected for {panel_blue}; "
            f"panel may be off-frame or shadowed"
        )
    irradiance = [float(v) for v in cap.panel_irradiance()]
    meta = {
        "panel_seed": str(panel_blue),
        "panel_capture_id": cap.uuid,
        "panel_utc": cap.utc_time().isoformat(),
        "panel_location": cap.location(),
        "panel_detected_bands": detected,
        "panel_irradiance": irradiance,
        "panel_albedo": [float(v) for v in cap.panel_albedo()],
    }
    return irradiance, meta


def run_gpu_sift_worker(capture_seed: Path, ref_index: int = 5) -> dict:
    """Spawn the v2 worker, return its JSON output."""
    band_paths = [str(capture_seed.with_name(
        capture_seed.name.replace("_1.tif", f"_{i}.tif"))) for i in (1, 2, 3, 4, 5, 6)]
    cmd = [TORCH_CUDA_PYTHON, str(GPU_WORKER), str(ref_index)] + band_paths
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"GPU worker failed: {proc.stderr[-500:]}")
    last_line = proc.stdout.strip().split("\n")[-1]
    return json.loads(last_line)


def _warp_work_share_seed(pan_shape, warp_matrices: list, irradiance: list, blue_path: str, outdir: str) -> str:
    """Module-level worker for multiprocessing.Pool.

    Returns the output path on success.
    """
    import sys
    sys.path.insert(0, "/home/davidem/miniforge3/lib/python3.13/site-packages")
    import numpy as np
    np.mat = np.asmatrix
    from micasense.capture import Capture
    from pathlib import Path
    from skimage.transform import ProjectiveTransform

    seed = Path(blue_path)
    band_paths = [str(seed.with_name(seed.name.replace("_1.tif", f"_{i}.tif")))
                  for i in (1, 2, 3, 4, 5, 6)]
    cap = Capture.from_filelist(band_paths)
    # micasense expects warp_matrices as a list of ProjectiveTransform objects,
    # NOT a list of lists. Convert.
    if isinstance(warp_matrices[0], (list, np.ndarray)):
        warp_objs = [ProjectiveTransform(matrix=np.array(w)) for w in warp_matrices]
    else:
        warp_objs = list(warp_matrices)
    cap.radiometric_pan_sharpened_aligned_capture(
        warp_matrices=warp_objs, irradiance_list=irradiance, img_type="reflectance"
    )
    aligned = cap._Capture__aligned_radiometric_pan_sharpened_capture[1]
    # aligned has shape (H_pan, W_pan, 6) including panchro as last band.
    # We want multispec only (5 bands: Blue, Green, Red, NIR, Red edge).
    multispec = aligned[:, :, :5]  # (H, W, 5)
    out_path = Path(outdir) / seed.name.replace("_1.tif", "_6.tif")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.rint(multispec * 32767.0), 0, 65535).astype(np.uint16)
    # rasterio wants (bands, H, W)
    data = np.moveaxis(data, -1, 0)  # (5, H, W)
    H, W = data.shape[1], data.shape[2]
    profile = {
        "driver": "GTiff", "height": H, "width": W, "count": 5,
        "dtype": "uint16", "compress": "deflate", "tiled": True,
    }
    import rasterio
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data)
        for i, n in enumerate(["Blue", "Green", "Red", "NIR", "Red edge"], start=1):
            dst.set_band_description(i, n)
        dst.update_tags(
            software="run_week1_gpu_sift.py",
            reflectance_scale="32767 = 1.0 reflectance",
            alignment="GPU Kornia SIFT v2 (bridge-based NIR)",
            panel_capture_id=cap.uuid,
        )
    return str(out_path)


def process_capture(
    seed: Path, outdir: Path, irradiance: list[float], gpu_warps: list, n_inliers_total: int
) -> dict:
    """Run the micasense preprocessing + GPU warps in a subprocess (separate process)."""
    args = (target_shape_for_dummy, gpu_warps, irradiance, str(seed), str(outdir))
    # We use multiprocessing.Pool.apply_async which needs module-level functions
    # but multiprocessing on the base env works. The micasense state
    # is in the subprocess.
    out_path = _warp_work_share_seed(None, gpu_warps, irradiance, str(seed), str(outdir))
    return {
        "seed": str(seed),
        "output": out_path,
        "n_inliers_total": n_inliers_total,
    }


# Dummy placeholder to make process_capture signature work
target_shape_for_dummy = None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", required=True, help="SET name (e.g. 0000SET)")
    parser.add_argument("--input-root", type=Path, required=True,
                        help="Root directory containing <SET>/000/ folders")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Output root directory; per-SET subdir will be created")
    parser.add_argument("--panel-seed", default="IMG_0000",
                        help="Panel capture name (default IMG_0000)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of captures (0 = all)")
    parser.add_argument("--ref-index", type=int, default=0,
                        help="SIFT reference band index. 0 = Blue band 1 (only valid "
                             "option for the v2 worker which excludes panchro from SIFT).")
    parser.add_argument("--capture-list", type=Path, default=None,
                        help="Optional capture list file (one capture per line)")
    args = parser.parse_args()

    set_dir = args.input_root / args.set
    panel_dir = set_dir  # panel is in the same SET
    set_outdir = args.outdir / args.set / "preprocessed_stacks"
    set_outdir.mkdir(parents=True, exist_ok=True)
    log_file = setup_logging(args.set)
    log = logging.getLogger("week1_gpu")
    log.info("Log file: %s", log_file)
    log.info("=" * 70)
    log.info("week1 GPU SIFT runner for SET=%s", args.set)
    log.info("=" * 70)

    # Step 1: panel calibration (single-threaded, ~10s)
    t0 = time.perf_counter()
    log.info("[PHASE] Panel calibration from %s/000/%s_1.tif", args.set, args.panel_seed)
    try:
        irradiance, panel_meta = panel_irradiance(panel_dir, args.panel_seed)
    except Exception as e:
        log.error("[FAIL] panel calibration: %s", e)
        return 2
    log.info("[PHASE] Panel: %d/6 bands detected, irradiance=%s (%.1fs)",
             panel_meta["panel_detected_bands"],
             [f"{v:.4f}" for v in irradiance],
             time.perf_counter() - t0)

    # Step 2: list captures
    if args.capture_list:
        with open(args.capture_list) as f:
            capture_names = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    else:
        all_1 = sorted(set_dir.glob("000/*_1.tif"))
        capture_names = [p.name for p in all_1]
    if args.limit:
        capture_names = capture_names[: args.limit]
    log.info("[PHASE] %d captures to process", len(capture_names))

    # Step 3: for each capture, run GPU SIFT then micasense radiometric apply
    # We use the multiprocessing Pool for the radiometric apply (CPU-bound, micasense)
    # and run the GPU SIFT inline (it can only run one at a time on the GPU).
    t0 = time.perf_counter()
    pool = mp.Pool(processes=max(1, args.workers - 1))
    results = []
    failures = []
    log.info("[PHASE] Starting per-capture processing (workers=%d, GPU sequential)",
             args.workers)
    t_first_done = None
    for i, capture_name in enumerate(capture_names, start=1):
        seed = set_dir / "000" / capture_name
        if not seed.exists():
            log.warning("[%d/%d] SKIP missing %s", i, len(capture_names), seed)
            continue
        t_cap0 = time.perf_counter()
        # GPU SIFT (sequential, ~2s per capture)
        try:
            t_sift0 = time.perf_counter()
            worker_out = run_gpu_sift_worker(seed, ref_index=args.ref_index)
            t_sift = time.perf_counter() - t_sift0
            gpu_warps = worker_out["warps"]
            n_inliers = worker_out.get("n_inliers_total", 0)
        except Exception as e:
            log.error("[%d/%d] FAIL GPU SIFT %s: %s", i, len(capture_names), capture_name, e)
            failures.append({"seed": str(seed), "error": str(e)})
            continue
        # Micasense radiometric apply (parallel via pool)
        try:
            async_result = pool.apply_async(
                _warp_work_share_seed,
                (None, gpu_warps, irradiance, str(seed), str(set_outdir))
            )
            t_apply0 = time.perf_counter()
            out_path = async_result.get(timeout=120)
            t_apply = time.perf_counter() - t_apply0
        except Exception as e:
            log.error("[%d/%d] FAIL micasense apply %s: %s", i, len(capture_names), capture_name, e)
            failures.append({"seed": str(seed), "error": str(e)})
            continue
        t_cap = time.perf_counter() - t_cap0
        results.append({
            "seed": str(seed),
            "output": out_path,
            "n_inliers_total": n_inliers,
            "t_sift_s": t_sift,
            "t_apply_s": t_apply,
            "t_total_s": t_cap,
        })
        # Live progress: every 10 captures or first 5
        if i <= 5 or i % 10 == 0 or i == len(capture_names):
            t_elapsed = time.perf_counter() - t0
            t_per_capture = t_elapsed / i
            eta_s = t_per_capture * (len(capture_names) - i)
            throughput = i / t_elapsed  # captures/sec
            log.info(
                "[%d/%d] (%.1f%%) %s: SIFT=%.2fs apply=%.2fs total=%.2fs | "
                "throughput=%.2f cap/s  ETA=%.0fs (%.1f min)",
                i, len(capture_names), 100.0 * i / len(capture_names),
                capture_name, t_sift, t_apply, t_cap,
                throughput, eta_s, eta_s / 60,
            )
    pool.close()
    pool.join()
    t_total = time.perf_counter() - t0
    log.info("[PHASE] Processed %d captures in %.1fs (%.2fs/capture mean, %.2f cap/s)",
             len(results), t_total, t_total / max(1, len(results)),
             len(results) / max(0.001, t_total))

    # Step 4: write summary
    summary = {
        "set": args.set,
        "input_root": str(args.input_root),
        "outdir": str(args.outdir),
        "panel_meta": panel_meta,
        "n_captures": len(capture_names),
        "n_processed": len(results),
        "n_failures": len(failures),
        "t_total_s": t_total,
        "t_per_capture_s": t_total / max(1, len(results)),
        "ref_index": args.ref_index,
        "results": results,
        "failures": failures,
    }
    summary_path = args.outdir / args.set / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("[WROTE] %s", summary_path)
    log.info("=" * 70)
    log.info("Done. %d processed, %d failed, %.1fs total", len(results), len(failures), t_total)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
