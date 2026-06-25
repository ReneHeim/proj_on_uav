#!/usr/bin/env python3
"""Diagnose where micasense SIFT_align_capture hangs.

Monkey-patches SIFT_align_capture to add per-step timing logs.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Configure logging BEFORE importing micasense
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("diag")

import numpy as np

# Patch skimage.measure.ransac for NumPy 2 compat
import skimage.measure as _skm
from micasense.capture import Capture
from skimage.feature import SIFT
from skimage.transform import ProjectiveTransform, estimate_transform

_orig_ransac = _skm.ransac


def _compat_ransac(*args, **kwargs):
    if "random_state" in kwargs and "rng" not in kwargs:
        kwargs["rng"] = kwargs.pop("random_state")
    return _orig_ransac(*args, **kwargs)


_skm.ransac = _compat_ransac
import micasense.capture as _cap

_cap.ransac = _compat_ransac


def _timed(label):
    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            log.info("  >> %s: start", label)
            return self

        def __exit__(self, *a):
            log.info("  << %s: %.2fs", label, time.perf_counter() - self.t0)

    return _T()


def _patched_sift_align(
    self, ref=5, min_matches=10, verbose=0, err_red=10.0, err_blue=12.0, err_LWIR=12.0
):
    log.info("SIFT_align_capture: enter  ref=%d  min_matches=%d", ref, min_matches)
    t_total = time.perf_counter()
    descriptor_extractor = SIFT()
    keypoints = []
    descriptors = []
    img_index = list(range(len(self.images)))
    img_index.pop(ref)
    ref_shape = self.images[ref].raw().shape
    rest_shape = self.images[img_index[0]].raw().shape
    scale = np.array(ref_shape) / np.array(rest_shape)
    log.info(
        "  ref_shape=%s  rest_shape=%s  scale=%s  non_ref_bands=%s",
        ref_shape,
        rest_shape,
        scale,
        img_index,
    )

    with _timed("get_warp_matrices (calibrated)"):
        warp_matrices_calibrated = self.get_warp_matrices(ref_index=ref)

    if rest_shape != ref_shape:
        with _timed("load+undistort+resize ref image"):
            ref_image_SIFT = self.images[ref].undistorted(self.images[ref].raw())
            from skimage.transform import resize

            ref_image_SIFT = resize(ref_image_SIFT, rest_shape)
            ref_image_SIFT = (ref_image_SIFT / ref_image_SIFT.max() * 65535).astype(np.uint16)
    else:
        with _timed("load+undistort ref image (no resize)"):
            ref_image_SIFT = self.images[ref].undistorted(self.images[ref].raw())

    with _timed("detect_and_extract ref image"):
        descriptor_extractor.detect_and_extract(ref_image_SIFT)
    keypoints_ref = descriptor_extractor.keypoints
    descriptor_ref = descriptor_extractor.descriptors
    log.info(
        "  ref keypoints=%d  descriptors.shape=%s",
        len(keypoints_ref),
        descriptor_ref.shape if descriptor_ref is not None else None,
    )

    match_images = []
    ratio = []
    filter_tr = []
    img_index = np.array(img_index)

    for ix in img_index:
        t_band = time.perf_counter()
        with _timed(f"load+undistort band {ix} ({self.images[ix].band_name})"):
            img = self.images[ix].undistorted(self.images[ix].raw())
        if img.shape != rest_shape:
            from skimage.transform import resize

            with _timed(f"resize band {ix}"):
                img_base = self.images[ix].raw()[self.images[ix].raw() > 0].min()
                img = img.astype(float)
                img[img > 0] = img[img > 0] - img_base
                img = resize(img, rest_shape)
                img = (img / img.max() * 65535).astype(np.uint16)
            ratio.append(1)
            filter_tr.append(err_LWIR)
        else:
            ratio.append(0.8)
            if ix <= 5:
                filter_tr.append(err_red)
            else:
                filter_tr.append(err_blue)
        match_images.append(img)
        with _timed(f"detect_and_extract band {ix}"):
            descriptor_extractor.detect_and_extract(img)
        keypoints.append(descriptor_extractor.keypoints)
        descriptors.append(descriptor_extractor.descriptors)
        log.info("  band %d keypoints=%d", ix, len(descriptor_extractor.keypoints))
        log.info("  band %d total so far: %.2fs", ix, time.perf_counter() - t_band)

    with _timed("match_descriptors for all bands"):
        from skimage.feature import match_descriptors

        matches = [
            match_descriptors(d, descriptor_ref, max_ratio=r) for d, r in zip(descriptors, ratio)
        ]

    # Find inliers per band
    models = []
    kp_image = []
    kp_ref = []
    for m, k, ix, t in zip(matches, keypoints, img_index, filter_tr):
        scale_i = np.array(self.images[ix].raw().shape) / np.array(rest_shape)
        with _timed(f"filter_keypoints band {ix}"):
            filtered_kpi, filtered_kpr, filtered_match, err = self.filter_keypoints(
                k, keypoints_ref, m, warp_matrices_calibrated[ix], scale, scale_i, threshold=t
            )
        log.info("  band %d filtered_match=%d (min=%d)", ix, len(filtered_match), min_matches)
        if len(filtered_match) > min_matches:
            with _timed(f"find_inliers band {ix}"):
                kpi, kpr, imatch, model = self.find_inliers(
                    filtered_kpi, filtered_kpr, filtered_match
                )
            with _timed(f"estimate_transform band {ix}"):
                P = estimate_transform(
                    "projective", (scale * kpr)[:, ::-1], (scale_i * kpi)[:, ::-1]
                )
        else:
            log.info("  band %d: < min_matches, using calibrated", ix)
            P = ProjectiveTransform(matrix=warp_matrices_calibrated[ix])
            kpi = filtered_kpi
            kpr = filtered_kpr
        models.append(P)
        kp_image.append(kpi)
        kp_ref.append(kpr)

    self.__sift_aligned_capture = [np.eye(3)] * len(self.images)
    for ix, m in zip(img_index, models):
        self.__sift_aligned_capture[ix] = m.params
    log.info("SIFT_align_capture: exit  total=%.2fs", time.perf_counter() - t_total)
    return self.__sift_aligned_capture


# Monkey-patch
Capture.SIFT_align_capture = _patched_sift_align
log.info("Patched Capture.SIFT_align_capture")

# Now run the pre-cache logic
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "mpp", Path(__file__).parent / "micasense_rededgep_preprocess.py"
)
mpp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mpp)
load_capture = mpp.load_capture
load_warp_cache = mpp.load_warp_cache
save_warp_cache = mpp.save_warp_cache
iter_capture_seeds = mpp.iter_capture_seeds
read_capture_list = mpp.read_capture_list
resolve_capture_list = mpp.resolve_capture_list
panel_irradiance = mpp.panel_irradiance
capture_files_from_blue = mpp.capture_files_from_blue
mpp.capture_mod.ransac = mpp.compat_ransac

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-set", type=Path, required=True)
    parser.add_argument("--panel-set", type=Path, required=True)
    parser.add_argument("--capture-list", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--warp-cache", type=Path, required=True)
    parser.add_argument("--alignment-seed", type=str, required=True)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--qa-preview-dir", type=Path, default=None)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.warp_cache.parent.mkdir(parents=True, exist_ok=True)

    all_seeds = sorted(args.input_set.glob("**/*_1.tif"))
    seeds = resolve_capture_list(read_capture_list(args.capture_list), all_seeds)
    log.info("Resolved %d seeds from capture list", len(seeds))

    seed_matches = [
        s for s in all_seeds if s.stem == args.alignment_seed or s.name == args.alignment_seed
    ]
    if not seed_matches:
        log.error("Alignment seed not found: %s", args.alignment_seed)
        sys.exit(1)
    alignment_seed = seed_matches[0]
    log.info("Alignment seed: %s", alignment_seed)

    log.info("Computing panel irradiance...")
    irradiance, panel_meta = panel_irradiance(args.panel_set, None)
    log.info("Panel irradiance: %s", irradiance)

    log.info("Computing warps via SIFT on %s ...", alignment_seed.name)
    t0 = time.perf_counter()
    cap = load_capture(alignment_seed)
    warps = cap.SIFT_align_capture(ref=5, min_matches=args.min_matches, verbose=2)
    log.info("SIFT done in %.2fs, warps=%d matrices", time.perf_counter() - t0, len(warps))
    save_warp_cache(
        args.warp_cache, warps, {"alignment_seed": str(alignment_seed), "alignment_method": "sift"}
    )
    log.info("Saved warp cache to %s", args.warp_cache)
