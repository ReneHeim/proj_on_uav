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
from skimage.transform import ProjectiveTransform, SimilarityTransform
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
MIN_INLIERS = 8              # minimum real inliers to accept a fit
MIN_MATCHES = 6              # minimum matches to attempt a fit
MAX_RESIDUAL_STD = 1.5       # px; reject unstable low-consensus fits
NIR_BAND = 3                 # 0-based: band 4, NIR
RED_BAND = 2                 # 0-based: band 3, Red
RED_EDGE_BAND = 4            # 0-based: band 5, Red edge


def detect_sift(sift: SIFTFeature, band_t: torch.Tensor
                ) -> tuple[np.ndarray, torch.Tensor]:
    """Run SIFT on a single band, return (keypoint_xy Nx2, descriptors Nx128)."""
    with torch.no_grad():
        lafs_i, _, descs_i = sift(band_t.unsqueeze(0))
        centers = get_laf_center(lafs_i).squeeze(0).detach().cpu().numpy()
        descs = torch.nn.functional.normalize(descs_i.squeeze(0).detach(), dim=-1)
    return centers, descs


def normalize_for_sift(bands_t: torch.Tensor) -> torch.Tensor:
    """Robust per-band normalization to [0, 1] for feature detection."""
    flat = bands_t.flatten(2)
    lo = torch.quantile(flat, 0.01, dim=2).view(-1, 1, 1, 1)
    hi = torch.quantile(flat, 0.99, dim=2).view(-1, 1, 1, 1)
    return torch.clamp((bands_t - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def gradient_for_sift(bands_t: torch.Tensor) -> torch.Tensor:
    """Gradient-magnitude feature image for cross-spectral matching."""
    norm = normalize_for_sift(bands_t)
    gx = torch.zeros_like(norm)
    gy = torch.zeros_like(norm)
    gx[:, :, :, 1:] = norm[:, :, :, 1:] - norm[:, :, :, :-1]
    gy[:, :, 1:, :] = norm[:, :, 1:, :] - norm[:, :, :-1, :]
    return normalize_for_sift(torch.sqrt(gx * gx + gy * gy + 1e-12))


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

    # --- Phase 2: Build tensor on GPU (multispec bands only, natural order) ---
    bands_t = torch.from_numpy(np.stack(bands[:5])).unsqueeze(1)  # (5, 1, H, W)
    print(f"[WORKER-V2] tensor shape (B,1,H,W) = {tuple(bands_t.shape)}", flush=True)

    if UPSCALE != 1.0:
        up_h, up_w = int(target_h * UPSCALE), int(target_w * UPSCALE)
        raw_up = torch.nn.functional.interpolate(
            bands_t, size=(up_h, up_w), mode="bilinear", align_corners=False
        ).cuda()
        print(f"[WORKER-V2] upsampled to ({up_h}, {up_w})", flush=True)
    else:
        raw_up = bands_t.cuda()
        print(f"[WORKER-V2] no upscale; using native ({target_h}, {target_w})", flush=True)
    grad_up = gradient_for_sift(raw_up)

    # --- Phase 3: SIFT detection and pairwise fitting helpers ---
    t_sift0 = time.perf_counter()
    sift = SIFTFeature(NUM_FEATURES, device="cuda", rootsift=True, upright=False)
    feature_cache: dict[tuple[str, int], tuple[np.ndarray, torch.Tensor]] = {}

    def features(mode: str, band_i: int) -> tuple[np.ndarray, torch.Tensor]:
        key = (mode, band_i)
        if key not in feature_cache:
            tensor = raw_up if mode == "raw" else grad_up
            centers_up, descs = detect_sift(sift, tensor[band_i])
            feature_cache[key] = (centers_up / UPSCALE, descs)
        return feature_cache[key]

    def validate_model(model, inliers, kp_ref: np.ndarray, kp_mov: np.ndarray,
                       ref_i: int, mov_i: int, mode: str) -> dict:
        if model is None or inliers is None:
            return {
                "valid": False, "reject_reason": "ransac returned no model",
                "ref_band": ref_i, "moving_band": mov_i, "mode": mode,
            }
        a, b_, c, d = (model.params[0, 0], model.params[0, 1],
                       model.params[1, 0], model.params[1, 1])
        if MODEL_CLS == "ProjectiveTransform":
            _, s_vals, _ = np.linalg.svd(np.array([[a, b_], [c, d]]))
            scale = float(np.exp(np.mean(np.log(s_vals))))
            proj_asym = float(abs(a - d))
            proj_shear = float(abs(b_ + c))
        else:
            scale = float(np.sqrt(a * a + c * c))
            proj_asym = 0.0
            proj_shear = 0.0
        res = model.residuals(kp_ref, kp_mov)
        n_inliers = int(inliers.sum())
        resid_std = float(np.std(res[inliers])) if n_inliers > 1 else float("inf")
        tx = float(model.params[0, 2])
        ty = float(model.params[1, 2])
        reasons = []
        if n_inliers < MIN_INLIERS:
            reasons.append(f"n_inliers={n_inliers} < MIN_INLIERS={MIN_INLIERS}")
        if resid_std > MAX_RESIDUAL_STD:
            reasons.append(f"residual_std={resid_std:.2f} > {MAX_RESIDUAL_STD}")
        if abs(scale - 1.0) > 0.03:
            reasons.append(f"|scale-1|={abs(scale - 1.0):.4f} > 0.03")
        if abs(tx) > TRANS_FRAC_GUARD * target_w or abs(ty) > TRANS_FRAC_GUARD * target_h:
            reasons.append(
                f"|tx|,|ty|=({tx:.1f},{ty:.1f}) > {TRANS_FRAC_GUARD}*({target_w},{target_h})"
            )
        if MODEL_CLS == "ProjectiveTransform" and PROJ_GUARD > 0:
            if proj_asym > PROJ_GUARD or proj_shear > PROJ_GUARD:
                reasons.append(
                    f"asym={proj_asym:.4f} or shear={proj_shear:.4f} > {PROJ_GUARD}"
                )
        return {
            "valid": not reasons,
            "reject_reason": "; ".join(reasons) if reasons else None,
            "ref_band": ref_i,
            "moving_band": mov_i,
            "mode": mode,
            "n_inliers": n_inliers,
            "fitted_scale": scale,
            "fitted_proj_asym": proj_asym,
            "fitted_proj_shear": proj_shear,
            "fitted_trans": [tx, ty],
            "residual_std": resid_std,
            "params": model.params,
        }

    def update_transform_fields(record: dict) -> None:
        params = np.asarray(record["params"])
        a, b_, c, d = params[0, 0], params[0, 1], params[1, 0], params[1, 1]
        if MODEL_CLS == "ProjectiveTransform":
            _, s_vals, _ = np.linalg.svd(np.array([[a, b_], [c, d]]))
            record["fitted_scale"] = float(np.exp(np.mean(np.log(s_vals))))
            record["fitted_proj_asym"] = float(abs(a - d))
            record["fitted_proj_shear"] = float(abs(b_ + c))
        else:
            record["fitted_scale"] = float(np.sqrt(a * a + c * c))
            record["fitted_proj_asym"] = 0.0
            record["fitted_proj_shear"] = 0.0
        record["fitted_trans"] = [float(params[0, 2]), float(params[1, 2])]

    def fit_pair(ref_i: int, mov_i: int, mode: str) -> dict:
        ref_centers, ref_descs_i = features(mode, ref_i)
        mov_centers, mov_descs = features(mode, mov_i)
        _, idxs = match_smnn(mov_descs, ref_descs_i, th=MATCH_TH)
        n_match = int(idxs.size(0))
        record = {
            "ref_band": ref_i,
            "moving_band": mov_i,
            "mode": mode,
            "n_features_ref": int(len(ref_centers)),
            "n_features_moving": int(len(mov_centers)),
            "n_matches": n_match,
            "n_inliers": 0,
            "valid": False,
            "reject_reason": None,
        }
        if n_match < MIN_MATCHES:
            record["reject_reason"] = f"n_match={n_match} < MIN_MATCHES={MIN_MATCHES}"
            return record
        idx_mov = idxs[:, 0].cpu().numpy()
        idx_ref = idxs[:, 1].cpu().numpy()
        kp_mov = mov_centers[idx_mov, :]
        kp_ref = ref_centers[idx_ref, :]
        if MODEL_CLS == "ProjectiveTransform":
            model_cls = ProjectiveTransform
            min_samples_ransac = 4
        else:
            model_cls = SimilarityTransform
            min_samples_ransac = 2
        try:
            model, inliers = _sk_ransac(
                (kp_ref, kp_mov),
                model_cls,
                min_samples=min_samples_ransac,
                residual_threshold=RANSAC_RESIDUAL,
                max_trials=RANSAC_TRIALS,
            )
        except Exception as exc:
            record["reject_reason"] = f"ransac_error: {exc}"
            return record
        record.update(validate_model(model, inliers, kp_ref, kp_mov, ref_i, mov_i, mode))
        return record

    # Prime raw features for the Blue reference so timing remains comparable.
    ref_centers, _ = features("raw", ref)
    torch.cuda.synchronize()
    t_sift = time.perf_counter() - t_sift0
    print(f"[WORKER-V2] ref SIFT: n_features={len(ref_centers)}  t={t_sift:.2f}s", flush=True)

    # --- Phase 4: Match and fit per band ---
    t_match0 = time.perf_counter()
    matches_per_band = []
    # 6 warps total (1 per band). Ref band (index `ref`) and panchro (index 5)
    # get identity warps.
    warps_out = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]] for _ in range(6)]
    n_inliers_total = 0
    direct_results: dict[int, dict] = {}
    for band_i in work_indices:
        if band_i == NIR_BAND:
            continue
        fit = fit_pair(ref, band_i, "raw")
        direct_results[band_i] = fit
        record = {
            "band": band_i,
            "n_features": fit.get("n_features_moving", 0),
            "n_matches": fit["n_matches"],
            "n_inliers": fit.get("n_inliers", 0),
            "fitted_scale": fit.get("fitted_scale", 1.0),
            "fitted_trans": fit.get("fitted_trans", [0.0, 0.0]),
            "residual_std": fit.get("residual_std", 0.0),
            "reject_reason": fit.get("reject_reason"),
            "fitted_proj_asym": fit.get("fitted_proj_asym", 0.0),
            "fitted_proj_shear": fit.get("fitted_proj_shear", 0.0),
            "selected_for_band": fit.get("valid", False),
            "candidate_ref_band": ref,
            "candidate_mode": "raw",
        }
        if fit.get("valid", False):
            warps_out[band_i] = fit["params"].tolist()
            n_inliers_total += int(fit["n_inliers"])
            print(f"[WORKER-V2] band {band_i+1}: raw Blue candidate OK "
                  f"n_match={fit['n_matches']} n_inliers={fit['n_inliers']} "
                  f"trans=({fit['fitted_trans'][0]:+.2f},{fit['fitted_trans'][1]:+.2f}) "
                  f"resid_std={fit['residual_std']:.2f}", flush=True)
        else:
            print(f"[WORKER-V2] band {band_i+1}: raw Blue candidate REJECT "
                  f"({fit.get('reject_reason')})", flush=True)
        matches_per_band.append(record)

    if NIR_BAND in work_indices:
        nir_candidates = []
        direct_nir = fit_pair(ref, NIR_BAND, "raw")
        direct_nir["candidate_name"] = "blue_to_nir_raw"
        nir_candidates.append(direct_nir)

        for bridge_i, bridge_name in [(RED_EDGE_BAND, "rededge"), (RED_BAND, "red")]:
            bridge_blue = direct_results.get(bridge_i)
            if not bridge_blue or not bridge_blue.get("valid", False):
                continue
            for mode in ["raw", "gradient"]:
                bridge_nir = fit_pair(bridge_i, NIR_BAND, mode)
                bridge_nir["candidate_name"] = f"blue_to_{bridge_name}_to_nir_{mode}"
                if bridge_nir.get("valid", False):
                    bridge_nir["params"] = bridge_nir["params"] @ bridge_blue["params"]
                    update_transform_fields(bridge_nir)
                    bridge_nir["candidate_ref_band"] = bridge_i
                    bridge_nir["bridge_warp_band"] = bridge_i
                nir_candidates.append(bridge_nir)

        valid_nir = [c for c in nir_candidates if c.get("valid", False)]
        selected = min(
            valid_nir,
            key=lambda c: (float(c.get("residual_std", np.inf)), -int(c.get("n_inliers", 0))),
            default=None,
        )
        if selected is not None:
            warps_out[NIR_BAND] = selected["params"].tolist()
            n_inliers_total += int(selected["n_inliers"])
            print(f"[WORKER-V2] band {NIR_BAND+1}: selected {selected['candidate_name']} "
                  f"n_match={selected['n_matches']} n_inliers={selected['n_inliers']} "
                  f"resid_std={selected['residual_std']:.2f}", flush=True)
        else:
            print(f"[WORKER-V2] band {NIR_BAND+1}: all NIR candidates rejected", flush=True)

        record = {
            "band": NIR_BAND,
            "n_features": features("raw", NIR_BAND)[0].shape[0],
            "n_matches": selected["n_matches"] if selected else direct_nir["n_matches"],
            "n_inliers": selected.get("n_inliers", 0) if selected else 0,
            "fitted_scale": selected.get("fitted_scale", 1.0) if selected else 1.0,
            "fitted_trans": selected.get("fitted_trans", [0.0, 0.0]) if selected else [0.0, 0.0],
            "residual_std": selected.get("residual_std", 0.0) if selected else 0.0,
            "reject_reason": None if selected else "all NIR candidates rejected",
            "fitted_proj_asym": selected.get("fitted_proj_asym", 0.0) if selected else 0.0,
            "fitted_proj_shear": selected.get("fitted_proj_shear", 0.0) if selected else 0.0,
            "selected_for_band": selected is not None,
            "candidate_ref_band": selected.get("ref_band") if selected else ref,
            "candidate_mode": selected.get("mode") if selected else "raw",
            "selected_candidate": selected.get("candidate_name") if selected else None,
            "nir_candidates": [
                {k: v for k, v in c.items() if k != "params"}
                for c in nir_candidates
            ],
        }
        matches_per_band.append(record)
    t_match = time.perf_counter() - t_match0
    torch.cuda.synchronize()

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
