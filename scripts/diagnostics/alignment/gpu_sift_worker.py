#!/usr/bin/env python3
"""GPU SIFT worker run inside the torch-cuda conda env.

Spawned by _benchmark_sift.py / _quality_assessment.py. Reads 6
band image paths from sys.argv, runs Kornia SIFT + match_smnn on GPU,
estimates projective warps with RANSAC, and writes a JSON file with
timing + match counts + warps to stdout.

This script intentionally lives outside the micasense/benchmark tree
because the torch-cuda env has its own numpy/scipy and cannot import
micasense directly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from kornia.feature import SIFTFeature, match_smnn
from kornia.feature.laf import get_laf_center
from skimage.measure import ransac as _sk_ransac
from skimage.transform import ProjectiveTransform, resize


def main():
    if len(sys.argv) != 8:
        print("usage: _gpu_sift_worker.py <ref_index> <band1> ... <band6>", file=sys.stderr)
        sys.exit(2)
    ref = int(sys.argv[1])
    paths = [Path(p) for p in sys.argv[2:8]]
    bands = []
    for p in paths:
        import rasterio

        with rasterio.open(p) as src:
            bands.append(src.read(1).astype(np.float32))
    sizes = sorted(set(b.shape for b in bands), key=lambda s: s[0] * s[1])
    target_h, target_w = sizes[0]
    for i, b in enumerate(bands):
        if b.shape != (target_h, target_w):
            bands[i] = resize(b, (target_h, target_w), preserve_range=True).astype(np.float32)
    bands_t = torch.from_numpy(np.stack(bands)).unsqueeze(1).cuda()  # (6,1,H,W)

    t0 = time.perf_counter()
    sift = SIFTFeature(2000, device="cuda", rootsift=True, upright=False)

    def detect_one(band_t: torch.Tensor) -> tuple[np.ndarray, torch.Tensor]:
        lafs_i, _, descs_i = sift(band_t.unsqueeze(0))
        with torch.no_grad():
            centers = get_laf_center(lafs_i).squeeze(0).detach().cpu().numpy()
            descs_norm = torch.nn.functional.normalize(descs_i.squeeze(0).detach(), dim=-1)
        return centers, descs_norm

    t_sift0 = time.perf_counter()
    ref_centers, ref_descs = detect_one(bands_t[ref])
    torch.cuda.synchronize()
    t_sift = time.perf_counter() - t_sift0

    ref_shape = bands[ref].shape
    rest_shape = bands[0].shape
    scale = np.array(ref_shape) / np.array(rest_shape)

    t_match0 = time.perf_counter()
    matches_per_band = []
    warps_out = [np.eye(3).tolist() for _ in range(6)]
    img_index = [i for i in range(6) if i != ref]
    for ix in img_index:
        ix_centers, ix_descs = detect_one(bands_t[ix])
        _, idxs = match_smnn(ix_descs, ref_descs, th=0.95)
        n_match = int(idxs.size(0))
        matches_per_band.append({"band": ix, "n_matches": n_match})
        if n_match >= 8:
            idx_ix = idxs[:, 0].cpu().numpy()
            idx_ref = idxs[:, 1].cpu().numpy()
            kp_ix = ix_centers[idx_ix, :]  # (K, 2) in (x, y)
            kp_ref = ref_centers[idx_ref, :] * scale[None, :]
            # micasense uses (y, x) order, Kornia returns (x, y)
            kp_ix_yx = kp_ix[:, ::-1]
            kp_ref_yx = kp_ref[:, ::-1]
            try:
                # Fit Euclidean (translation + rotation, NO scale) with
                # tight residual. RedEdge-P bands are co-registered to <3px
                # after lens correction. Fitting a full projective lets RANSAC
                # absorb noise into a spurious scale, so we constrain to rigid
                # motion which is what the physics actually predicts.
                from skimage.transform import EuclideanTransform

                model, inliers = _sk_ransac(
                    (kp_ref_yx, kp_ix_yx),
                    EuclideanTransform,
                    min_samples=2,
                    residual_threshold=1.5,
                    max_trials=3000,
                )
                if model is not None and inliers.sum() >= 4:
                    params = model.params
                    sx, sy = params[1, 1], params[0, 0]  # (y, x) in micasense
                    tx, ty = params[0, 2], params[1, 2]
                    h, w = target_h, target_w
                    matches_per_band[-1]["fitted_scale"] = [float(sx), float(sy)]
                    matches_per_band[-1]["fitted_trans"] = [float(tx), float(ty)]
                    # RedEdge-P bands are near-identity after lens correction.
                    # Allow generous tolerance: scale 0.90-1.10, trans up to 30% of image.
                    scale_ok = 0.90 < sx < 1.10 and 0.90 < sy < 1.10
                    trans_ok = abs(tx) < 0.3 * h and abs(ty) < 0.3 * w
                    if scale_ok and trans_ok:
                        warps_out[ix] = params.tolist()
                        matches_per_band[-1]["n_inliers"] = int(inliers.sum())
                    else:
                        # Reject: keep identity warp, record why
                        matches_per_band[-1]["n_inliers"] = 0
                        matches_per_band[-1]["reject_reason"] = (
                            f"scale=({sx:.4f},{sy:.4f}) trans=({tx:.2f},{ty:.2f}) "
                            f"outside [0.90,1.10]x[0.3*{h},0.3*{w}]"
                        )
                else:
                    matches_per_band[-1]["n_inliers"] = 0
            except Exception:
                matches_per_band[-1]["n_inliers"] = 0
        else:
            matches_per_band[-1]["n_inliers"] = 0
        del ix_descs
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t_match = time.perf_counter() - t_match0

    print(
        json.dumps(
            {
                "t_sift_s": t_sift,
                "t_match_s": t_match,
                "t_total_s": time.perf_counter() - t0,
                "image_shape": [target_h, target_w],
                "matches_per_band": matches_per_band,
                "warps": warps_out,
            }
        )
    )


if __name__ == "__main__":
    main()
