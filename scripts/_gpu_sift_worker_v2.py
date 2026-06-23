#!/usr/bin/env python3
"""GPU SIFT worker v2 — multispec-native alignment, matched to CPU ground truth.

Reference frame: Blue band (band 1, index 0) at multispec native resolution
(1088x1456). This matches the CPU ground-truth multispec stack produced by
_apply_cpu_warps_multispec.py, so the 99% similarity metric is meaningful.

Fixes vs v1 (_gpu_sift_worker.py):
  A. Reference band = Blue (band 1) at multispec native resolution.
     Both ref and other multispec bands live in the same coordinate system,
     so the warp is near-identity (no crop, no resize artefacts).
  B. SimilarityTransform (rotation + translation + uniform scale) with
     tighter scale guard (|scale - 1| < 0.03) instead of Euclidean.
  C. Higher num_features (4096) and looser match threshold (0.90) for
     better coverage of difficult bands like NIR.
  D. The panchromatic band (band 6) is excluded from SIFT alignment; it
     lives at a different physical resolution (2056x2464) and would
     introduce scale artefacts. The output 5-band stack is at multispec
     native res, matching the CPU ground truth.

Reads 6 band image paths from sys.argv. The multispec bands (1-5) are at
1088x1456. The panchro band (6) is at 2056x2464 and is excluded from the
SIFT alignment but its warp is reported as identity (caller may choose to
downsample to multispec or upscale to pan-sharpened at display time).

The first argv is the ref index in the multispec set (0 = Blue band 1).

Usage:
    /home/davidem/miniforge3/envs/torch-cuda/bin/python \\
        scripts/_gpu_sift_worker_v2.py \\
        0 \\
        /path/IMG_0002_1.tif /path/IMG_0002_2.tif ... /path/IMG_0002_6.tif

Outputs a JSON line on stdout with timing, match counts, and warps.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from skimage.measure import ransac as _sk_ransac
from skimage.transform import SimilarityTransform
from kornia.feature import SIFTFeature, match_smnn
from kornia.feature.laf import get_laf_center


# --- Configurable worker parameters ---
NUM_FEATURES = 4096          # Kornia SIFT features per band
UPSCALE = 1.0                # No upscale; work at multispec native res
MATCH_TH = 0.95              # match_smnn distance ratio threshold
RANSAC_RESIDUAL = 2.0        # px
RANSAC_TRIALS = 5000
# Use ProjectiveTransform to match the micasense CPU SIFT. The CPU uses
# ProjectiveTransform with full 8-DOF (so does this worker). Add a projective
# guard: the warp's "non-similarity" component (a - d, b + c) must be small
# (RedEdge-P bands are physically near-rigid). A loose guard allows the
# full projective flexibility the CPU uses, while rejecting degenerate fits.
MODEL_CLS = "ProjectiveTransform"
SCALE_GUARD = 0.01           # |scale - 1| must be < 1% (RedEdge-P physical co-registration)
PROJ_GUARD = 0.05            # |a - d|, |b + c| must be < 0.05; very loose to allow CPU-like fits
TRANS_FRAC_GUARD = 0.15      # |tx|, |ty| must be < this * image_dim; else reject fit
MIN_INLIERS = 2              # minimum inliers to accept the fit (NIR often has only 2-3 matches)
MIN_MATCHES = 6              # minimum matches to attempt a fit


def detect_sift(sift: SIFTFeature, band_t: torch.Tensor
                ) -> tuple[np.ndarray, torch.Tensor]:
    """Run SIFT on a single band, return (keypoint_xy Nx2, descriptors Nx128)."""
    with torch.no_grad():
        lafs_i, _, descs_i = sift(band_t.unsqueeze(0))
        centers = get_laf_center(lafs_i).squeeze(0).detach().cpu().numpy()
        descs = torch.nn.functional.normalize(descs_i.squeeze(0).detach(), dim=-1)
    return centers, descs


def main() -> None:
    if len(sys.argv) != 8:
        print("usage: _gpu_sift_worker_v2.py <ref_index 0..5> <band1> ... <band6>",
              file=sys.stderr)
        sys.exit(2)
    ref = int(sys.argv[1])
    paths = [Path(p) for p in sys.argv[2:8]]
    print(f"[WORKER-V2] ref index = {ref} (band {ref+1})  upscale = {UPSCALE}", flush=True)

    # --- Phase 1: Load bands ---
    t0 = time.perf_counter()
    import rasterio
    bands = []
    for p in paths:
        with rasterio.open(p) as src:
            bands.append(src.read(1).astype(np.float32))
    print(f"[WORKER-V2] band shapes = {[b.shape for b in bands]}  load={time.perf_counter() - t0:.2f}s",
          flush=True)

    # We work at multispec native res (1088x1456). The multispec bands
    # (1-5) are already at this res. The panchro band (6) is at 2056x2464
    # and is excluded from SIFT alignment.
    target_h, target_w = bands[ref].shape
    if (target_h, target_w) != (1088, 1456):
        print(f"[WORKER-V2] WARNING: ref band at {target_h}x{target_w}, "
              f"expected multispec 1088x1456", flush=True)
    print(f"[WORKER-V2] working at multispec native res ({target_h}, {target_w})", flush=True)

    # SIFT alignment indices: all multispec bands except the ref
    work_indices = [i for i in range(5) if i != ref]  # 0..4 only
    print(f"[WORKER-V2] SIFT work indices = {work_indices} (panchro band 6 excluded)", flush=True)

    # --- Phase 2: Build tensor on GPU (multispec bands only) ---
    work_bands = [bands[i] for i in work_indices]
    bands_t = torch.from_numpy(np.stack(work_bands + [bands[ref]])).unsqueeze(1)  # (5, 1, H, W)
    tensor_ref_idx = len(work_indices)  # index of the ref band in the tensor
    print(f"[WORKER-V2] tensor shape (B,1,H,W) = {tuple(bands_t.shape)}", flush=True)

    if UPSCALE != 1.0:
        up_h, up_w = int(target_h * UPSCALE), int(target_w * UPSCALE)
        bands_up = torch.nn.functional.interpolate(
            bands_t, size=(up_h, up_w), mode="bilinear", align_corners=False
        ).cuda()
        print(f"[WORKER-V2] upsampled to ({up_h}, {up_w})", flush=True)
    else:
        bands_up = bands_t.cuda()
        print(f"[WORKER-V2] no upscale; using native ({target_h}, {target_w})", flush=True)

    # --- Phase 3: SIFT detection on ref ---
    t_sift0 = time.perf_counter()
    sift = SIFTFeature(NUM_FEATURES, device="cuda", rootsift=True, upright=False)
    ref_centers_up, ref_descs = detect_sift(sift, bands_up[tensor_ref_idx])
    torch.cuda.synchronize()
    t_sift = time.perf_counter() - t_sift0
    print(f"[WORKER-V2] ref SIFT: n_features={len(ref_centers_up)}  t={t_sift:.2f}s", flush=True)

    # Convert ref keypoints to (x, y) at target (multispec native) resolution
    ref_centers = ref_centers_up / UPSCALE

    # --- Phase 4: Match and fit per band ---
    t_match0 = time.perf_counter()
    matches_per_band = []
    # 6 warps total (1 per band). Ref band (index `ref`) and panchro (index 5)
    # get identity warps.
    warps_out = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]] for _ in range(6)]
    n_inliers_total = 0
    for tensor_i, band_i in enumerate(work_indices):
        ix_centers_up, ix_descs = detect_sift(sift, bands_up[tensor_i])
        _, idxs = match_smnn(ix_descs, ref_descs, th=MATCH_TH)
        n_match = int(idxs.size(0))
        del ix_descs
        torch.cuda.empty_cache()
        record = {"band": band_i, "n_features": int(len(ix_centers_up)),
                  "n_matches": n_match, "n_inliers": 0,
                  "fitted_scale": 1.0, "fitted_trans": [0.0, 0.0],
                  "residual_std": 0.0, "reject_reason": None}
        if n_match < MIN_MATCHES:
            record["reject_reason"] = f"n_match={n_match} < MIN_MATCHES={MIN_MATCHES}"
            matches_per_band.append(record)
            print(f"[WORKER-V2] band {band_i+1}: n_features={len(ix_centers_up)} "
                  f"n_match={n_match}  -> REJECT ({record['reject_reason']})", flush=True)
            continue
        ix_centers = ix_centers_up / UPSCALE
        idx_ix = idxs[:, 0].cpu().numpy()
        idx_ref = idxs[:, 1].cpu().numpy()
        kp_ix = ix_centers[idx_ix, :]
        kp_ref = ref_centers[idx_ref, :]
        # Both Kornia SIFT keypoints and skimage ProjectiveTransform use (x, y)
        # order: `predict([x, y]) -> [x', y']` and `params = [[a, b, tx], [c, d, ty], [0, 0, 1]]`.
        # Pass keypoints directly (no reversal) so the fit is in the right basis.
        # micasense CPU reverses because its skimage SIFT keypoints are in (y, x);
        # Kornia returns (x, y), so we must NOT reverse.
        # RANSAC fits: model @ kp_ref = kp_ix, so model maps ref -> moving.
        # Choose model class
        if MODEL_CLS == "ProjectiveTransform":
            from skimage.transform import ProjectiveTransform
            model_cls = ProjectiveTransform
            min_samples_ransac = 4
        else:
            model_cls = SimilarityTransform
            min_samples_ransac = 2
        try:
            model, inliers = _sk_ransac(
                (kp_ref, kp_ix),  # (x, y) order
                model_cls,
                min_samples=min_samples_ransac,
                residual_threshold=RANSAC_RESIDUAL,
                max_trials=RANSAC_TRIALS,
            )
        except Exception as e:
            record["reject_reason"] = f"ransac_error: {e}"
            matches_per_band.append(record)
            print(f"[WORKER-V2] band {band_i+1}: RANSAC error {e}", flush=True)
            continue
        if model is None or inliers is None or inliers.sum() < MIN_INLIERS:
            record["reject_reason"] = (
                f"n_inliers={int(inliers.sum()) if inliers is not None else 0} "
                f"< MIN_INLIERS={MIN_INLIERS}"
            )
            matches_per_band.append(record)
            print(f"[WORKER-V2] band {band_i+1}: n_match={n_match} "
                  f"n_inliers={int(inliers.sum()) if inliers is not None else 0}  "
                  f"-> REJECT ({record['reject_reason']})", flush=True)
            continue
        a, b_, c, d = (model.params[0, 0], model.params[0, 1],
                       model.params[1, 0], model.params[1, 1])
        # For Projective, a/d may differ; use a combined "scale" as the
        # geometric mean of singular values of the 2x2 linear part.
        if MODEL_CLS == "ProjectiveTransform":
            U, S, Vt = np.linalg.svd(np.array([[a, b_], [c, d]]))
            scale = float(np.exp(np.mean(np.log(S))))
            proj_asym = float(abs(a - d))  # non-uniform scale
            proj_shear = float(abs(b_ + c))  # non-similarity shear
        else:
            scale = float(np.sqrt(a * a + c * c))
            proj_asym = 0.0
            proj_shear = 0.0
        # Residual std computed in (x, y) order (no reversal)
        res = model.residuals(kp_ref, kp_ix)
        resid_std = float(np.std(res[inliers])) if inliers is not None and inliers.sum() > 1 else 0.0
        # In (x, y) convention: params = [[a, b, tx], [c, d, ty], [0, 0, 1]]
        # model maps ref -> moving in (x, y) coords
        tx = float(model.params[0, 2])
        ty = float(model.params[1, 2])
        record.update({
            "n_inliers": int(inliers.sum()),
            "fitted_scale": scale,
            "fitted_proj_asym": proj_asym,
            "fitted_proj_shear": proj_shear,
            "fitted_trans": [tx, ty],
            "residual_std": resid_std,
        })
        # Validation chain with relaxed fallback: try strict first, then relax
        accepted = False
        rejection_reasons = []
        for scale_try, trans_try in [
            (SCALE_GUARD, TRANS_FRAC_GUARD),
            (SCALE_GUARD * 4, TRANS_FRAC_GUARD * 2),  # 4x scale, 2x trans
            (0.05, 0.3),  # very permissive final fallback
        ]:
            scale_ok = abs(scale - 1.0) <= scale_try
            trans_ok = (abs(tx) <= trans_try * target_w
                        and abs(ty) <= trans_try * target_h)
            # Projective guard (only for Projective model)
            proj_ok = True
            if MODEL_CLS == "ProjectiveTransform" and PROJ_GUARD > 0:
                proj_ok = (proj_asym <= PROJ_GUARD and proj_shear <= PROJ_GUARD)
            if scale_ok and trans_ok and proj_ok:
                if scale_try > SCALE_GUARD or (MODEL_CLS == "ProjectiveTransform" and PROJ_GUARD > 0):
                    print(f"[WORKER-V2] band {band_i+1}: accepted (scale_try={scale_try}, "
                          f"trans_try={trans_try}, asym={proj_asym:.4f}, shear={proj_shear:.4f})",
                          flush=True)
                accepted = True
                break
            else:
                why = []
                if not scale_ok:
                    why.append(f"|scale-1|={abs(scale - 1.0):.4f}>{scale_try}")
                if not trans_ok:
                    why.append(f"|tx|,|ty|=({tx:.1f},{ty:.1f})>{trans_try}*({target_w},{target_h})")
                if not proj_ok:
                    why.append(f"asym={proj_asym:.4f} or shear={proj_shear:.4f} > {PROJ_GUARD}")
                rejection_reasons.append("; ".join(why))
        if not accepted:
            record["reject_reason"] = f"rejected at all guards: {rejection_reasons[-1]}"
            print(f"[WORKER-V2] band {band_i+1}: REJECT ({record['reject_reason']})",
                  flush=True)
            matches_per_band.append(record)
            continue
        # ACCEPT
        warps_out[band_i] = model.params.tolist()
        n_inliers_total += int(inliers.sum())
        print(f"[WORKER-V2] band {band_i+1}: n_match={n_match} n_inliers={int(inliers.sum())} "
              f"scale={scale:.5f} asym={proj_asym:.4f} shear={proj_shear:.4f} "
              f"trans=(tx={tx:+.2f},ty={ty:+.2f}) resid_std={resid_std:.2f}  OK",
              flush=True)
        matches_per_band.append(record)
    t_match = time.perf_counter() - t_match0
    torch.cuda.synchronize()

    # --- Phase 4.5: Fall back to mean warp for rejected bands ---
    # RedEdge-P bands have physically similar inter-band offsets. If a band's
    # RANSAC fit is degenerate (e.g., NIR has very different texture), use the
    # mean of the accepted warps as a physical-prior fallback.
    accepted_warps = []
    accepted_record_idxs = []
    rejected_record_idxs = []
    for i, m in enumerate(matches_per_band):
        if m.get("reject_reason") is None and m.get("n_inliers", 0) >= MIN_INLIERS:
            accepted_warps.append(np.array(warps_out[m["band"]]))
            accepted_record_idxs.append(i)
        else:
            rejected_record_idxs.append(i)
    if accepted_warps and rejected_record_idxs:
        # Mean warp (per-parameter)
        mean_warp = np.mean(accepted_warps, axis=0)
        # Force the 3x3 matrix to be a proper similarity: enforce a/d symmetric
        # and b=-c (no shear), normalize so the (a, d) eigenvalues have mean 1
        a, b_, c, d = mean_warp[0, 0], mean_warp[0, 1], mean_warp[1, 0], mean_warp[1, 1]
        scale = np.sqrt(np.sqrt(a * d - b_ * c))  # geometric mean
        theta = 0.5 * np.arctan2(b_ + c, a + d)  # mean rotation
        # Build a clean similarity matrix from scale, theta, mean translation
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        clean = np.array([
            [scale * cos_t, -scale * sin_t, mean_warp[0, 2]],
            [scale * sin_t,  scale * cos_t, mean_warp[1, 2]],
            [0, 0, 1]
        ])
        for i in rejected_record_idxs:
            band_i = matches_per_band[i]["band"]
            warps_out[band_i] = clean.tolist()
            matches_per_band[i]["fitted_scale"] = float(scale)
            matches_per_band[i]["fitted_trans"] = [float(clean[1, 2]), float(clean[0, 2])]
            matches_per_band[i]["fitted_proj_asym"] = 0.0
            matches_per_band[i]["fitted_proj_shear"] = 0.0
            matches_per_band[i]["fallback_used"] = "mean_of_accepted_warps"
            old_reason = matches_per_band[i].get("reject_reason", "n/a")
            matches_per_band[i]["reject_reason"] = None
            matches_per_band[i]["previous_reject_reason"] = old_reason
            print(f"[WORKER-V2] band {band_i+1}: FALLBACK to mean warp "
                  f"(scale={scale:.5f}, trans=({clean[1, 2]:+.2f},{clean[0, 2]:+.2f}), "
                  f"prev reason: {old_reason})", flush=True)

    out = {
        "worker_version": "v2",
        "upscale": UPSCALE,
        "num_features": NUM_FEATURES,
        "match_th": MATCH_TH,
        "ransac_residual": RANSAC_RESIDUAL,
        "ransac_trials": RANSAC_TRIALS,
        "model_cls": MODEL_CLS,
        "scale_guard": SCALE_GUARD,
        "trans_frac_guard": TRANS_FRAC_GUARD,
        "min_inliers": MIN_INLIERS,
        "min_matches": MIN_MATCHES,
        "ref_band": ref + 1,
        "t_sift_s": float(t_sift),
        "t_match_s": float(t_match),
        "t_total_s": float(time.perf_counter() - t0),
        "image_shape": [int(target_h), int(target_w)],
        "n_inliers_total": int(n_inliers_total),
        "matches_per_band": matches_per_band,
        "warps": warps_out,
    }
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
