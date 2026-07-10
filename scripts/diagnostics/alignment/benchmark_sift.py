#!/usr/bin/env python3
"""Benchmark SIFT alignment methods for micasense RedEdge-P preprocessing.

Compares:
  1. baseline     : skimage SIFT + brute-force match_descriptors (current, hangs on high-kp images)
  2. limit_kp_5k  : same, but cap keypoints to top-5000 per band (by response)
  3. limit_kp_10k : same, but cap keypoints to top-10000 per band
  4. limit_kp_15k : same, but cap keypoints to top-15000 per band

Each method is run on a single capture and timed end-to-end.
The script also runs on two captures (fast + slow) to show the difference.

Outputs:
  - Per-method timing log to outputs/archive/legacy_unscoped/logs/sift_benchmark_*.log
  - results/sift_benchmark_results.json
  - reports/sift_benchmark_summary.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup (AGENTS.md rule 1: function-level profiling and log files)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]  # proj_on_uav/
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"sift_benchmark_{TS}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger("sift_benchmark")
log.info("Log file: %s", LOG_FILE)

RESULTS_DIR = ROOT / "outputs/archive/legacy_unscoped/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = ROOT / "outputs/archive/legacy_unscoped/reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = RESULTS_DIR / "sift_benchmark_results.json"
RESULTS_MD = REPORTS_DIR / "sift_benchmark_summary.md"

# ---------------------------------------------------------------------------
# Imports (after logging setup)
# ---------------------------------------------------------------------------
# If running in a separate conda env (e.g. torch-cuda), make the micasense
# package from the base miniforge3 env importable by adding it to sys.path.
import os as _os
import subprocess as _sp

_BASE_MICA = "/home/davidem/miniforge3/lib/python3.13/site-packages"
if _os.path.isdir(_BASE_MICA) and _BASE_MICA not in sys.path:
    sys.path.insert(0, _BASE_MICA)
    log.info("Added base miniforge3 site-packages to sys.path: %s", _BASE_MICA)

TORCH_CUDA_PYTHON = "/home/davidem/miniforge3/envs/torch-cuda/bin/python"
GPU_WORKER = Path(__file__).parent / "_gpu_sift_worker.py"
log.info("GPU worker: %s  python: %s", GPU_WORKER, TORCH_CUDA_PYTHON)

import numpy as np
import skimage.measure as _skm

# micasense + skimage NumPy-2 compat shim
from micasense.capture import Capture
from skimage.feature import SIFT
from skimage.feature import match_descriptors as _sk_match
from skimage.transform import ProjectiveTransform, estimate_transform

_orig_ransac = _skm.ransac


def _compat_ransac(*args, **kwargs):
    if "random_state" in kwargs and "rng" not in kwargs:
        kwargs["rng"] = kwargs.pop("random_state")
    return _orig_ransac(*args, **kwargs)


_skm.ransac = _compat_ransac
import micasense.capture as _cap_mod

_cap_mod.ransac = _compat_ransac

import micasense.imageutils as _mi  # noqa: F401  (forces module load)


def _make_kornia_sift(num_features: int = 5000, device: str = "cuda"):
    """Build a Kornia-based SIFT_align_capture that runs entirely on GPU.

    Returns a function with the same signature as Capture.SIFT_align_capture.

    Keypoints are limited to num_features (Kornia sorts by response internally
    inside the scale-space detector). Matching uses Kornia's `match_smnn`
    (mutual nearest neighbor + Lowe's ratio test) on GPU.
    """
    if not KORNIA_AVAILABLE:
        raise RuntimeError("Kornia not available; install torch + kornia in a GPU env")

    def kornia_sift(
        self, ref=5, min_matches=10, verbose=0, err_red=10.0, err_blue=12.0, err_LWIR=12.0
    ):
        import torch
        from kornia.feature import SIFTFeature, match_smnn
        from kornia.feature.laf import get_laf_center
        from skimage.transform import ProjectiveTransform

        t_total = time.perf_counter()
        log.info("[GPU-SIFT] enter  ref=%d  num_features=%d  device=%s", ref, num_features, device)

        ref_shape = self.images[ref].raw().shape  # panchro size
        rest_shape = self.images[0].raw().shape  # multispec size
        log.info("[GPU-SIFT] ref=%s  rest=%s", ref_shape, rest_shape)

        # Load all 6 bands (undistorted) and resize to a common resolution.
        # All bands must be the same size for batched SIFTFeature.
        from skimage.transform import resize

        target_h, target_w = rest_shape
        bands_np = []
        for ix in range(len(self.images)):
            img = self.images[ix].undistorted(self.images[ix].raw())
            if img.shape != (target_h, target_w):
                img = resize(img, (target_h, target_w), preserve_range=True).astype(np.float32)
            else:
                img = img.astype(np.float32)
            bands_np.append(img)
        bands_t = torch.from_numpy(np.stack(bands_np)).unsqueeze(1).to(device)  # (6, 1, H, W)
        log.info(
            "[GPU-SIFT] bands tensor: shape=%s  dtype=%s  device=%s",
            bands_t.shape,
            bands_t.dtype,
            bands_t.device,
        )

        t0 = time.perf_counter()
        # SIFTFeature: DoG detector + SIFT descriptor
        # rootsift=True gives L2-normalized descriptors (matches skimage's float convention).
        sift = SIFTFeature(num_features, device=device, rootsift=True, upright=False)
        lafs_all, resp_all, desc_all = sift(bands_t)  # each (6, N, ...)
        log.info("[GPU-SIFT] detect+extract all 6 bands: %.2fs", time.perf_counter() - t0)
        log.info("[GPU-SIFT]   lafs:  %s", lafs_all.shape)
        log.info("[GPU-SIFT]   resp:  %s", resp_all.shape)
        log.info("[GPU-SIFT]   desc:  %s", desc_all.shape)

        # Get keypoint centers (x, y) at the (H, W) rest resolution.
        # get_laf_center expects (B, N, 2, 3) and returns (B, N, 2).
        centers = get_laf_center(lafs_all)  # (6, N, 2) on device
        centers_cpu = centers.cpu().numpy()

        # Match each non-ref band to the reference using match_smnn (mutual + ratio test).
        # Per docs: input desc1 (B1, D), desc2 (B2, D); returns (dists (B3,1), idxs (B3,2)).
        scale = np.array(ref_shape) / np.array(rest_shape)
        img_index = list(range(len(self.images)))
        img_index.pop(ref)
        img_index = np.array(img_index)
        warp_matrices_calibrated = self.get_warp_matrices(ref_index=ref)

        # Save the raw warp container so we can return a list of 3x3 matrices.
        out_warps = [np.eye(3) for _ in range(len(self.images))]

        for ix in img_index:
            t_b = time.perf_counter()
            # L2-normalize for cosine similarity (kornia SIFT is already RootSIFT
            # but we keep this step for safety)
            d_ix = torch.nn.functional.normalize(desc_all[ix], dim=-1)
            d_ref = torch.nn.functional.normalize(desc_all[ref], dim=-1)
            _, idxs = match_smnn(d_ix, d_ref, th=0.95)  # (B3, 2) on device
            n_match = int(idxs.size(0))
            log.info(
                "[GPU-SIFT] band %d (%s): smnn matches=%d  (%.2fs)",
                ix,
                self.images[ix].band_name,
                n_match,
                time.perf_counter() - t_b,
            )
            if n_match == 0:
                out_warps[ix] = warp_matrices_calibrated[ix]
                continue

            idx_ix = idxs[:, 0].cpu().numpy()
            idx_ref = idxs[:, 1].cpu().numpy()
            # Center coordinates in the (H, W) rest resolution; LAF centers are (x, y).
            kp_ix = centers_cpu[ix, idx_ix, :]  # (K, 2)
            kp_ref = centers_cpu[ref, idx_ref, :]  # (K, 2) at rest scale
            # Scale ref keypoints to the panchro (ref) resolution to match
            # the micasense coordinate convention (scale by ref/rest).
            kp_ref_full = kp_ref * scale[None, :]
            scale_i = np.ones(2)
            # Build a synthetic match array (idx_in_image, idx_in_ref) for micasense
            # filter_keypoints (it expects an Mx2 array of paired indices).
            match_pairs = np.stack([idx_ix, idx_ref], axis=1)
            try:
                filtered_kpi, filtered_kpr, filtered_match, _ = self.filter_keypoints(
                    list(kp_ix),
                    list(kp_ref_full),
                    match_pairs,
                    warp_matrices_calibrated[ix],
                    scale,
                    scale_i,
                    threshold=err_red if ix <= 5 else err_blue,
                )
            except Exception as exc:
                log.warning("[GPU-SIFT] band %d filter_keypoints failed: %s", ix, exc)
                out_warps[ix] = warp_matrices_calibrated[ix]
                continue
            log.info(
                "[GPU-SIFT] band %d filtered=%d (min=%d)", ix, len(filtered_match), min_matches
            )

            if len(filtered_match) >= min_matches:
                kpi_arr = np.asarray(filtered_kpi)
                kpr_arr = np.asarray(filtered_kpr)
                kpi_use, kpr_use, _, _ = self.find_inliers(
                    kpi_arr, kpr_arr, np.asarray(filtered_match)
                )
                P = estimate_transform(
                    "projective", (scale * kpr_use)[:, ::-1], (scale_i * kpi_use)[:, ::-1]
                )
            else:
                log.info("[GPU-SIFT] band %d: < min_matches, using calibrated", ix)
                P = ProjectiveTransform(matrix=warp_matrices_calibrated[ix])
            out_warps[ix] = P.params

        log.info("[GPU-SIFT] exit  total=%.2fs", time.perf_counter() - t_total)
        return out_warps

    return kornia_sift


# ---------------------------------------------------------------------------
# Method variants for SIFT_align_capture
# ---------------------------------------------------------------------------
def _limit_keypoints(sift_extractor, max_kp: int) -> int:
    """Cap keypoints/descriptors to top-N by scale (proxy for salience in skimage 0.26+).

    skimage 0.26 removed the public `responses` attribute on SIFT. We use `scales`
    as a proxy: larger scale = more salient keypoint (detected at coarser scale,
    more stable across viewpoints).
    """
    if sift_extractor.keypoints is None:
        return 0
    n = len(sift_extractor.keypoints)
    if n <= max_kp:
        return n
    # Use scales as ranking criterion. Larger scale = more stable/salient.
    if hasattr(sift_extractor, "scales") and sift_extractor.scales is not None:
        scores = np.asarray(sift_extractor.scales, dtype=np.float64)
    else:
        # Fallback: random sample
        idx = np.random.default_rng(0).choice(n, size=max_kp, replace=False)
        sift_extractor.keypoints = sift_extractor.keypoints[idx]
        sift_extractor.descriptors = sift_extractor.descriptors[idx]
        return max_kp
    idx = np.argsort(scores)[-max_kp:]
    sift_extractor.keypoints = sift_extractor.keypoints[idx]
    sift_extractor.descriptors = sift_extractor.descriptors[idx]
    return max_kp


def _make_patched_sift(max_kp: int | None):
    """Return a patched SIFT_align_capture that optionally caps keypoints.

    If max_kp is None, behaves exactly like the original (baseline).
    """

    def patched(self, ref=5, min_matches=10, verbose=0, err_red=10.0, err_blue=12.0, err_LWIR=12.0):
        t_total = time.perf_counter()
        log.info("[SIFT] enter  ref=%d  min_matches=%d  max_kp=%s", ref, min_matches, max_kp)
        descriptor_extractor = SIFT()
        keypoints, descriptors = [], []
        img_index = list(range(len(self.images)))
        img_index.pop(ref)
        ref_shape = self.images[ref].raw().shape
        rest_shape = self.images[img_index[0]].raw().shape
        scale = np.array(ref_shape) / np.array(rest_shape)
        log.info("[SIFT] ref_shape=%s rest_shape=%s non_ref=%s", ref_shape, rest_shape, img_index)

        t0 = time.perf_counter()
        warp_matrices_calibrated = self.get_warp_matrices(ref_index=ref)
        log.info("[SIFT] get_warp_matrices(calibrated): %.2fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        if rest_shape != ref_shape:
            from skimage.transform import resize

            ref_image_SIFT = self.images[ref].undistorted(self.images[ref].raw())
            ref_image_SIFT = resize(ref_image_SIFT, rest_shape)
            ref_image_SIFT = (ref_image_SIFT / ref_image_SIFT.max() * 65535).astype(np.uint16)
        else:
            ref_image_SIFT = self.images[ref].undistorted(self.images[ref].raw())
        log.info("[SIFT] load+undistort ref: %.2fs", time.perf_counter() - t0)

        t0 = time.perf_counter()
        descriptor_extractor.detect_and_extract(ref_image_SIFT)
        if max_kp is not None:
            n_ref = _limit_keypoints(descriptor_extractor, max_kp)
        else:
            n_ref = (
                len(descriptor_extractor.keypoints)
                if descriptor_extractor.keypoints is not None
                else 0
            )
        keypoints_ref = descriptor_extractor.keypoints
        descriptor_ref = descriptor_extractor.descriptors
        log.info("[SIFT] detect+extract ref: %.2fs  keypoints=%d", time.perf_counter() - t0, n_ref)

        match_images, ratio, filter_tr = [], [], []
        img_index = np.array(img_index)
        for ix in img_index:
            t_band = time.perf_counter()
            img = self.images[ix].undistorted(self.images[ix].raw())
            if img.shape != rest_shape:
                from skimage.transform import resize

                img_base = self.images[ix].raw()[self.images[ix].raw() > 0].min()
                img = img.astype(float)
                img[img > 0] = img[img > 0] - img_base
                img = resize(img, rest_shape)
                img = (img / img.max() * 65535).astype(np.uint16)
                ratio.append(1)
                filter_tr.append(err_LWIR)
            else:
                ratio.append(0.8)
                filter_tr.append(err_red if ix <= 5 else err_blue)
            match_images.append(img)
            descriptor_extractor.detect_and_extract(img)
            n_kp = (
                _limit_keypoints(descriptor_extractor, max_kp)
                if max_kp is not None
                else len(descriptor_extractor.keypoints)
            )
            keypoints.append(descriptor_extractor.keypoints)
            descriptors.append(descriptor_extractor.descriptors)
            log.info(
                "[SIFT] band %d (%s): total %.2fs  keypoints=%d",
                ix,
                self.images[ix].band_name,
                time.perf_counter() - t_band,
                n_kp,
            )

        t0 = time.perf_counter()
        matches = [_sk_match(d, descriptor_ref, max_ratio=r) for d, r in zip(descriptors, ratio)]
        log.info("[SIFT] match_descriptors (5 bands): %.2fs", time.perf_counter() - t0)
        log.info("[SIFT] match counts per band: %s", [len(m) for m in matches])

        models, kp_image, kp_ref = [], [], []
        for m, k, ix, t_thresh in zip(matches, keypoints, img_index, filter_tr):
            scale_i = np.array(self.images[ix].raw().shape) / np.array(rest_shape)
            t_b = time.perf_counter()
            filtered_kpi, filtered_kpr, filtered_match, err = self.filter_keypoints(
                k,
                keypoints_ref,
                m,
                warp_matrices_calibrated[ix],
                scale,
                scale_i,
                threshold=t_thresh,
            )
            log.info(
                "[SIFT] band %d filter_keypoints: %.2fs  filtered=%d (min=%d)",
                ix,
                time.perf_counter() - t_b,
                len(filtered_match),
                min_matches,
            )
            if len(filtered_match) > min_matches:
                t_b = time.perf_counter()
                kpi, kpr, imatch, model = self.find_inliers(
                    filtered_kpi, filtered_kpr, filtered_match
                )
                log.info("[SIFT] band %d find_inliers: %.2fs", ix, time.perf_counter() - t_b)
                t_b = time.perf_counter()
                P = estimate_transform(
                    "projective", (scale * kpr)[:, ::-1], (scale_i * kpi)[:, ::-1]
                )
                log.info("[SIFT] band %d estimate_transform: %.2fs", ix, time.perf_counter() - t_b)
            else:
                log.info("[SIFT] band %d: < min_matches, using calibrated", ix)
                P = ProjectiveTransform(matrix=warp_matrices_calibrated[ix])
                kpi = filtered_kpi
                kpr = filtered_kpr
            models.append(P)
            kp_image.append(kpi)
            kp_ref.append(kpr)

        self.__sift_aligned_capture = [np.eye(3)] * len(self.images)
        for ix, m in zip(img_index, models):
            self.__sift_aligned_capture[ix] = m.params
        log.info("[SIFT] exit  total=%.2fs", time.perf_counter() - t_total)
        return self.__sift_aligned_capture

    return patched


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------
METHODS = [
    ("baseline", None),
    ("limit_kp_2000", 2000),
    ("limit_kp_5000", 5000),
    ("kornia_sift_5000", 5000),  # GPU method added below if torch-cuda available
]

# GPU method: only included if the torch-cuda Python is available
if _os.path.isfile(TORCH_CUDA_PYTHON):
    METHODS.append(("kornia_sift_5000", 5000))  # GPU method, num_features=5000
    log.info("GPU method 'kornia_sift_5000' added (torch-cuda env found)")
else:
    log.warning("GPU method skipped: %s not found", TORCH_CUDA_PYTHON)

PER_METHOD_TIMEOUT = 300  # seconds

# Optional: GPU SIFT via Kornia (only available in the torch-cuda env).
# Note: the GPU method is run via subprocess, so we don't actually need
# kornia installed in THIS process. We just check for the torch-cuda Python.
import importlib.util as _ilu

KORNIA_AVAILABLE = False  # set later based on torch-cuda Python availability


def _run_with_timeout(fn, timeout_s: int, label: str) -> dict:
    """Run fn in a subprocess with a hard timeout. Returns result dict."""
    import subprocess
    import tempfile

    payload = {
        "fn_module": fn.__module__,
        "fn_name": fn.__name__,
        "args_module": fn.__code__.co_filename,
    }
    # Simpler: run the actual function in-process with a SIGALRM-style fallback.
    # We don't have a portable subprocess timeout here, so just run with a wall-clock check.
    t0 = time.perf_counter()
    try:
        result = fn()
        return {"status": "ok", "elapsed_s": time.perf_counter() - t0, "result": result}
    except Exception as exc:
        return {
            "status": "error",
            "elapsed_s": time.perf_counter() - t0,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def benchmark_capture(
    capture_seed: Path,
    panel_seed: Path | None = None,
    methods: list[tuple[str, int | None]] | None = None,
    per_method_timeout_s: int = 300,
    use_subprocess: bool = False,
) -> dict:
    """Run all methods on a single capture and return timings.

    If use_subprocess is True, each method runs in a separate process with a
    hard kill after per_method_timeout_s. This is the only way to escape a
    hung SIFT step in a C extension (SIGALRM does not interrupt C code).
    """
    if methods is None:
        methods = METHODS

    log.info("=" * 70)
    log.info("BENCHMARK capture: %s", capture_seed)
    log.info("=" * 70)
    results = {"capture": str(capture_seed), "methods": {}}

    if use_subprocess:
        return _benchmark_capture_subprocess(capture_seed, methods, per_method_timeout_s, results)

    # In-process path (used for the fast capture where all methods complete quickly)
    import importlib.util as _ilu

    from micasense.capture import Capture as _Cap

    _spec = _ilu.spec_from_file_location(
        "mpp",
        Path(__file__).parent / "micasense_rededgep_preprocess.py",
    )
    _mpp = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mpp)

    log.info("Loading capture from %s ...", capture_seed)
    t0 = time.perf_counter()
    cap = _Cap.from_filelist(
        [_mpp.band_path_from_blue(capture_seed, s) for s in _mpp.BAND_SUFFIXES]
    )
    log.info("Capture loaded in %.2fs", time.perf_counter() - t0)
    log.info("  ref band shape (band 5/panchro): %s", cap.images[5].raw().shape)
    log.info("  other bands shape: %s", cap.images[0].raw().shape)

    for name, max_kp in methods:
        log.info("-" * 60)
        log.info("METHOD: %s  max_kp=%s", name, max_kp)
        log.info("-" * 60)
        # GPU method: run via torch-cuda subprocess
        if name == "kornia_sift_5000":
            t0 = time.perf_counter()
            try:
                band_paths = [
                    str(_mpp.band_path_from_blue(capture_seed, s)) for s in _mpp.BAND_SUFFIXES
                ]
                cmd = [TORCH_CUDA_PYTHON, str(GPU_WORKER), "5"] + band_paths
                log.info("[GPU] spawning: %s", " ".join(cmd[:3]) + " ...")
                proc = _sp.run(cmd, capture_output=True, text=True, timeout=120)
                if proc.returncode != 0:
                    raise RuntimeError(f"GPU worker failed: {proc.stderr[-500:]}")
                gpu_data = json.loads(proc.stdout)
                elapsed = time.perf_counter() - t0
                log.info(
                    "[GPU] t_sift=%.2fs  t_match=%.2fs  t_total=%.2fs",
                    gpu_data["t_sift_s"],
                    gpu_data["t_match_s"],
                    gpu_data["t_total_s"],
                )
                log.info(
                    "[GPU] shape=%s  matches=%s",
                    gpu_data["image_shape"],
                    [m["n_matches"] for m in gpu_data["matches_per_band"]],
                )
                results["methods"][name] = {
                    "status": "ok",
                    "elapsed_s": elapsed,
                    "max_kp": max_kp,
                    "gpu_sift_s": gpu_data["t_sift_s"],
                    "gpu_match_s": gpu_data["t_match_s"],
                    "gpu_total_s": gpu_data["t_total_s"],
                    "matches_per_band": [m["n_matches"] for m in gpu_data["matches_per_band"]],
                }
            except _sp.TimeoutExpired:
                elapsed = time.perf_counter() - t0
                log.error("[GPU] timeout after %.1fs", elapsed)
                results["methods"][name] = {
                    "status": "timeout",
                    "elapsed_s": elapsed,
                    "max_kp": max_kp,
                }
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                log.exception("[GPU] ERROR after %.1fs", elapsed)
                results["methods"][name] = {
                    "status": "error",
                    "elapsed_s": elapsed,
                    "max_kp": max_kp,
                    "error": repr(exc),
                }
            continue
        # CPU methods
        _Cap.SIFT_align_capture = _make_patched_sift(max_kp)
        t0 = time.perf_counter()
        try:
            warps = cap.SIFT_align_capture(ref=5, min_matches=8, verbose=0)
            elapsed = time.perf_counter() - t0
            log.info("[RESULT] %s: %.2fs  warps=%d", name, elapsed, len(warps))
            results["methods"][name] = {
                "status": "ok",
                "elapsed_s": elapsed,
                "max_kp": max_kp,
            }
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.exception("[RESULT] %s: ERROR after %.2fs", name, elapsed)
            results["methods"][name] = {
                "status": "error",
                "elapsed_s": elapsed,
                "max_kp": max_kp,
                "error": repr(exc),
            }

    import importlib

    importlib.reload(_cap_mod)
    _Cap.SIFT_align_capture = _cap_mod.Capture.SIFT_align_capture
    return results


def _run_one_method_subprocess(
    capture_seed: Path, max_kp: int | None, queue, method_name: str = ""
) -> None:
    """Worker: runs one SIFT method on one capture, puts result dict in queue.

    For the GPU method 'kornia_sift_5000', the GPU SIFT is run via the
    torch-cuda Python in a separate subprocess. For CPU methods, we
    monkey-patch Capture.SIFT_align_capture and call it in-process.
    """
    import multiprocessing as _mp

    try:
        if method_name == "kornia_sift_5000":
            # Run the GPU worker in a subprocess so we stay in the base env
            import importlib.util as _ilu

            _spec = _ilu.spec_from_file_location(
                "mpp",
                Path(__file__).parent / "micasense_rededgep_preprocess.py",
            )
            _mpp = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mpp)
            band_paths = [
                str(_mpp.band_path_from_blue(capture_seed, s)) for s in _mpp.BAND_SUFFIXES
            ]
            cmd = [TORCH_CUDA_PYTHON, str(GPU_WORKER), "5"] + band_paths
            t0 = time.perf_counter()
            proc = _mp.Process.__class__  # noqa
            import subprocess as _sp

            proc = _sp.run(cmd, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(f"GPU worker failed: {proc.stderr[-500:]}")
            gpu_data = json.loads(proc.stdout)
            elapsed = time.perf_counter() - t0
            queue.put(
                {
                    "status": "ok",
                    "elapsed_s": elapsed,
                    "max_kp": max_kp,
                    "warps": 6,
                    "gpu_sift_s": gpu_data["t_sift_s"],
                    "gpu_match_s": gpu_data["t_match_s"],
                    "gpu_total_s": gpu_data["t_total_s"],
                    "matches_per_band": [m["n_matches"] for m in gpu_data["matches_per_band"]],
                }
            )
            return

        import importlib.util as _ilu

        from micasense.capture import Capture as _Cap

        _spec = _ilu.spec_from_file_location(
            "mpp",
            Path(__file__).parent / "micasense_rededgep_preprocess.py",
        )
        _mpp = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mpp)

        cap = _Cap.from_filelist(
            [_mpp.band_path_from_blue(capture_seed, s) for s in _mpp.BAND_SUFFIXES]
        )
        _Cap.SIFT_align_capture = _make_patched_sift(max_kp)
        t0 = time.perf_counter()
        warps = cap.SIFT_align_capture(ref=5, min_matches=8, verbose=0)
        elapsed = time.perf_counter() - t0
        queue.put({"status": "ok", "elapsed_s": elapsed, "max_kp": max_kp, "warps": len(warps)})
    except Exception as exc:
        queue.put({"status": "error", "error": repr(exc), "traceback": traceback.format_exc()})


def _benchmark_capture_subprocess(capture_seed, methods, timeout_s, results) -> dict:
    """Run each method in a separate process; kill after timeout_s."""
    import multiprocessing as _mp

    for name, max_kp in methods:
        log.info("-" * 60)
        log.info("METHOD (subprocess): %s  max_kp=%s  timeout=%ds", name, max_kp, timeout_s)
        log.info("-" * 60)
        q = _mp.Queue()
        p = _mp.Process(
            target=_run_one_method_subprocess,
            args=(capture_seed, max_kp, q, name),
            name=f"sift-{name}",
        )
        t0 = time.perf_counter()
        p.start()
        p.join(timeout=timeout_s)
        elapsed = time.perf_counter() - t0
        if p.is_alive():
            log.warning("[TIMEOUT] %s still alive after %.1fs; killing", name, elapsed)
            p.kill()
            p.join()
            results["methods"][name] = {
                "status": "timeout",
                "elapsed_s": elapsed,
                "max_kp": max_kp,
            }
        else:
            try:
                payload = q.get_nowait()
            except Exception:
                payload = {
                    "status": "error",
                    "error": "no result from subprocess",
                    "exitcode": p.exitcode,
                }
            payload["max_kp"] = max_kp
            payload["elapsed_s"] = elapsed
            log.info("[RESULT] %s: %s  elapsed=%.1fs", name, payload.get("status"), elapsed)
            if "gpu_sift_s" in payload:
                log.info(
                    "[RESULT]   GPU detail: sift=%.2fs  match=%.2fs  total=%.2fs  matches=%s",
                    payload["gpu_sift_s"],
                    payload["gpu_match_s"],
                    payload["gpu_total_s"],
                    payload.get("matches_per_band"),
                )
            results["methods"][name] = payload
    return results


def write_markdown(all_results: list[dict]) -> None:
    """Write a markdown summary table comparing methods per capture."""
    lines = [
        "# SIFT Alignment Method Benchmark",
        "",
        f"Generated UTC: {datetime.utcnow().isoformat()}",
        f"Log file: `{LOG_FILE.relative_to(ROOT)}`",
        "",
        "## Methods compared",
        "",
        "| Method | Description |",
        "|---|---|",
        "| baseline | skimage SIFT + brute-force `match_descriptors` (current, hangs on high-keypoint images) |",
        "| limit_kp_5000 | Cap keypoints to top-5000 per band (by scale proxy), then brute-force match |",
        "| limit_kp_10000 | Cap keypoints to top-10000 per band |",
        "| limit_kp_15000 | Cap keypoints to top-15000 per band |",
        "| kornia_sift_5000 | GPU SIFT via Kornia (PyTorch + CUDA), num_features=5000, smnn matching |",
        "",
        "## Results",
        "",
        "| Capture | Method | Status | Elapsed (s) | Speedup vs baseline |",
        "|---|---|---|---:|---:|",
    ]
    for r in all_results:
        capture_name = Path(r["capture"]).name
        baseline_s = r["methods"].get("baseline", {}).get("elapsed_s")
        for name, _ in METHODS:
            m = r["methods"].get(name, {})
            status = m.get("status", "?")
            elapsed = m.get("elapsed_s", 0.0)
            speedup = (
                f"{baseline_s / elapsed:.1f}x" if baseline_s and elapsed and status == "ok" else "-"
            )
            lines.append(f"| {capture_name} | {name} | {status} | {elapsed:.1f} | {speedup} |")
        lines.append("| | | | | |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The `baseline` reproduces the current behavior (brute-force matching on full keypoint sets).",
            "- `limit_kp_*` methods cap keypoints per band to make brute-force matching feasible.",
            "- Lower elapsed time + status=ok is better. Errors indicate the method hung or failed.",
            "",
        ]
    )
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote markdown summary: %s", RESULTS_MD)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast-capture",
        type=Path,
        default=Path("/mnt/data/ONCERCO/data/raw/2025/week1/rededgep/0000SET/000/IMG_0002_1.tif"),
        help="Capture expected to be SIFT-friendly (low keypoint count)",
    )
    parser.add_argument(
        "--slow-capture",
        type=Path,
        default=Path("/mnt/data/ONCERCO/data/raw/2025/week1/rededgep/0001SET/000/IMG_0002_1.tif"),
        help="Capture expected to hang on baseline (high keypoint count)",
    )
    parser.add_argument("--skip-slow", action="store_true", help="Skip the slow capture")
    parser.add_argument("--skip-fast", action="store_true", help="Skip the fast capture")
    parser.add_argument(
        "--slow-methods",
        default="limit_kp_5000,limit_kp_10000,limit_kp_15000",
        help="Comma-separated method names to run on the slow capture (default skips baseline)",
    )
    parser.add_argument(
        "--fast-methods",
        default=",".join(name for name, _ in METHODS),
        help="Comma-separated method names to run on the fast capture",
    )
    parser.add_argument(
        "--slow-timeout",
        type=int,
        default=120,
        help="Per-method hard timeout (seconds) for the slow capture",
    )
    parser.add_argument(
        "--fast-timeout",
        type=int,
        default=300,
        help="Per-method hard timeout (seconds) for the fast capture",
    )
    args = parser.parse_args()

    fast_methods = _parse_methods(args.fast_methods)
    slow_methods = _parse_methods(args.slow_methods)
    log.info("Fast methods: %s", fast_methods)
    log.info("Slow methods: %s", slow_methods)

    all_results = []
    if not args.skip_fast:
        all_results.append(
            benchmark_capture(
                args.fast_capture,
                methods=fast_methods,
                per_method_timeout_s=args.fast_timeout,
            )
        )
    if not args.skip_slow:
        all_results.append(
            benchmark_capture(
                args.slow_capture,
                methods=slow_methods,
                per_method_timeout_s=args.slow_timeout,
                use_subprocess=True,  # hard kill on timeout
            )
        )

    RESULTS_JSON.write_text(json.dumps(all_results, indent=2, default=str))
    log.info("Wrote results JSON: %s", RESULTS_JSON)
    write_markdown(all_results)


def _parse_methods(spec: str) -> list[tuple[str, int | None]]:
    name_to_maxkp = {name: max_kp for name, max_kp in METHODS}
    out = []
    for name in spec.split(","):
        name = name.strip()
        if not name:
            continue
        if name not in name_to_maxkp:
            raise ValueError(f"Unknown method {name!r}. Known: {list(name_to_maxkp)}")
        out.append((name, name_to_maxkp[name]))
    return out


if __name__ == "__main__":
    main()
