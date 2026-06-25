#!/usr/bin/env python3
"""Preprocess MicaSense RedEdge-P captures into aligned reflectance stacks."""

from __future__ import annotations

import argparse
import cProfile
import logging
import json
import pstats
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

np.mat = np.asmatrix  # Compatibility with MicaSense imageprocessing + NumPy 2.

import micasense.capture as capture_mod
import rasterio
import skimage.measure
from micasense.capture import Capture

ORIG_RANSAC = skimage.measure.ransac
BAND_SUFFIXES = ("1", "2", "3", "4", "5", "6")
RAW_MS_BAND_NAMES = ("Blue", "Green", "Red", "NIR", "Red edge")
METASHAPE_MS_BAND_INDEXES = (0, 1, 2, 4, 3)
METASHAPE_MS_BAND_NAMES = ("Blue", "Green", "Red", "Red edge", "NIR")
PANCHRO_BAND_INDEX = 5
PANCHRO_BAND_NAME = "Panchro"
DEFAULT_METASHAPE_CORRECTION = {
    "Blue": 1.0,
    "Green": 1.0,
    "Red": 1.0,
    "Red edge": 1.0,
    "NIR": 1.0,
    "Panchro": 1.0,
}
RADIOMETRY_MODES = (
    "micasense_panel",
    "micasense_dls",
    "panel_dls_tie",
    "metashape_compatible",
)


def setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


@contextmanager
def timed_phase(name: str):
    t0 = time.perf_counter()
    logging.info("[PHASE] %s: start", name)
    try:
        yield
    finally:
        logging.info("[PHASE] %s: %.1fs", name, time.perf_counter() - t0)


@contextmanager
def timeout_after(seconds: int | None, label: str):
    if not seconds:
        yield
        return

    def handler(signum, frame):
        raise TimeoutError(f"{label} exceeded {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def compat_ransac(*args, **kwargs):
    if "random_state" in kwargs and "rng" not in kwargs:
        kwargs["rng"] = kwargs.pop("random_state")
    return ORIG_RANSAC(*args, **kwargs)


capture_mod.ransac = compat_ransac


def capture_files_from_blue(blue_path: Path) -> list[str]:
    name = blue_path.name
    if not name.endswith("_1.tif"):
        raise ValueError(f"expected *_1.tif capture seed, got {blue_path}")
    return [
        str(blue_path.with_name(name.replace("_1.tif", f"_{suffix}.tif")))
        for suffix in BAND_SUFFIXES
    ]


def band_path_from_blue(blue_path: Path, suffix: str) -> Path:
    return blue_path.with_name(blue_path.name.replace("_1.tif", f"_{suffix}.tif"))


def iter_capture_seeds(folder: Path) -> Iterable[Path]:
    yield from sorted(folder.glob("**/*_1.tif"))


def read_capture_list(path: Path) -> list[str]:
    entries = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    if not entries:
        raise ValueError(f"capture list is empty: {path}")
    return entries


def resolve_capture_list(entries: list[str], all_seeds: list[Path]) -> list[Path]:
    by_name = {seed.name: seed for seed in all_seeds}
    by_stem = {seed.stem: seed for seed in all_seeds}
    resolved = []
    missing = []
    for entry in entries:
        candidate = Path(entry)
        if candidate.exists():
            resolved.append(candidate)
            continue
        match = by_name.get(entry) or by_stem.get(entry) or by_name.get(f"{entry}_1.tif")
        if match:
            resolved.append(match)
        else:
            missing.append(entry)
    if missing:
        raise FileNotFoundError(f"capture-list entries not found under input-set: {missing[:10]}")
    return resolved


def load_capture(blue_path: Path) -> Capture:
    files = capture_files_from_blue(blue_path)
    missing = [path for path in files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"capture {blue_path} missing bands: {missing}")
    return Capture.from_filelist(files)


def capture_detected_panel_bands(seed: Path) -> int:
    with timed_phase(f"panel check load {seed.name}"):
        cap = load_capture(seed)
    with timed_phase(f"panel check detect {seed.name}"):
        return int(cap.detect_panels())


def exclude_detected_panels(seeds: list[Path], min_detected_bands: int) -> tuple[list[Path], list[dict]]:
    kept = []
    excluded = []
    for seed in seeds:
        try:
            detected = capture_detected_panel_bands(seed)
        except Exception as exc:
            logging.warning("[CLASSIFY] panel detection failed for %s: %s", seed, exc)
            kept.append(seed)
            continue
        if detected >= min_detected_bands:
            excluded.append(
                {
                    "seed": str(seed),
                    "role": "panel",
                    "reason": f"detect_panels returned {detected} bands",
                    "panel_detected_bands": detected,
                }
            )
            logging.info("[CLASSIFY] excluding panel %s detected_bands=%s", seed, detected)
        else:
            kept.append(seed)
    return kept, excluded


def evenly_sample(items: list[Path], count: int | None) -> list[Path]:
    if not count or count >= len(items):
        return items
    indexes = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[index] for index in sorted(set(indexes))]


def score_alignment_candidate(seed: Path, max_size: int) -> dict:
    panchro_path = band_path_from_blue(seed, "6")
    if not panchro_path.exists():
        raise FileNotFoundError(f"missing panchro band for alignment scoring: {panchro_path}")
    with rasterio.open(panchro_path) as src:
        scale = max(src.height, src.width) / float(max_size)
        out_height = max(1, int(src.height / max(scale, 1.0)))
        out_width = max(1, int(src.width / max(scale, 1.0)))
        image = src.read(1, out_shape=(out_height, out_width)).astype("float32")

    lo = float(np.nanpercentile(image, 1))
    hi = float(np.nanpercentile(image, 99))
    normalized = np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gy, gx = np.gradient(normalized)
    gradient = np.hypot(gx, gy)
    hist, _ = np.histogram(normalized, bins=64, range=(0.0, 1.0), density=False)
    probabilities = hist.astype("float64") / max(int(hist.sum()), 1)
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum() / 6.0)
    gradient_p95 = float(np.nanpercentile(gradient, 95))
    gradient_mean = float(np.nanmean(gradient))
    edge_density = float(np.mean(gradient > max(gradient_p95 * 0.25, 1e-6)))
    saturation_fraction = float(np.mean((normalized <= 0.01) | (normalized >= 0.99)))
    score = gradient_p95 * 4.0 + gradient_mean * 2.0 + entropy + edge_density * 0.5
    score -= saturation_fraction * 1.5
    return {
        "seed": str(seed),
        "panchro": str(panchro_path),
        "score": float(score),
        "gradient_p95": gradient_p95,
        "gradient_mean": gradient_mean,
        "edge_density": edge_density,
        "entropy": entropy,
        "saturation_fraction": saturation_fraction,
    }


def rank_alignment_candidates(
    seeds: list[Path], candidate_count: int | None, max_size: int
) -> list[dict]:
    candidates = evenly_sample(seeds, candidate_count)
    rankings = []
    for seed in candidates:
        try:
            rankings.append(score_alignment_candidate(seed, max_size=max_size))
        except Exception as exc:
            logging.warning("[ALIGN-CANDIDATE] failed to score %s: %s", seed, exc)
    rankings.sort(key=lambda record: record["score"], reverse=True)
    for rank, record in enumerate(rankings[:10], start=1):
        logging.info(
            "[ALIGN-CANDIDATE] rank=%s seed=%s score=%.4f edge_density=%.4f entropy=%.4f saturation=%.4f",
            rank,
            record["seed"],
            record["score"],
            record["edge_density"],
            record["entropy"],
            record["saturation_fraction"],
        )
    return rankings


def panel_irradiance(panel_folder: Path, panel_seed: str | None) -> tuple[list[float], dict]:
    with timed_phase("panel list seeds"):
        seeds = list(iter_capture_seeds(panel_folder))
    if not seeds:
        raise FileNotFoundError(f"no panel captures found in {panel_folder}")
    seed = seeds[0]
    if panel_seed:
        matches = [
            candidate
            for candidate in seeds
            if candidate.name.startswith(panel_seed) or candidate.stem.startswith(panel_seed)
        ]
        if not matches:
            raise FileNotFoundError(f"panel seed {panel_seed!r} not found in {panel_folder}")
        seed = matches[0]

    logging.info("[PANEL] seed=%s", seed)
    with timed_phase("panel load capture"):
        cap = load_capture(seed)
    with timed_phase("panel detect_panels"):
        detected = int(cap.detect_panels())
    if detected < 5:
        raise RuntimeError(f"only detected {detected} panel bands for {seed}")

    with timed_phase("panel irradiance"):
        irradiance = [float(value) for value in cap.panel_irradiance()]
    with timed_phase("panel DLS irradiance"):
        dls_irradiance = [float(value) for value in cap.dls_irradiance()]
    dls_to_panel_scale = [
        panel_value / dls_value if dls_value > 0 else 1.0
        for panel_value, dls_value in zip(irradiance, dls_irradiance)
    ]
    meta = {
        "panel_seed": str(seed),
        "panel_capture_id": cap.uuid,
        "panel_utc": cap.utc_time().isoformat(),
        "panel_location": cap.location(),
        "panel_detected_bands": detected,
        "panel_irradiance": irradiance,
        "panel_dls_irradiance": dls_irradiance,
        "panel_dls_to_panel_scale": dls_to_panel_scale,
        "panel_albedo": [float(value) for value in cap.panel_albedo()],
    }
    return irradiance, meta


def compute_warps(
    seed: Path,
    min_matches: int,
    verbose: int,
    allow_calibrated_fallback: bool,
    alignment_method: str,
    alignment_timeout: int | None,
) -> tuple[list[np.ndarray], dict]:
    logging.info("[ALIGN] seed=%s min_matches=%s", seed, min_matches)
    with timed_phase("alignment load capture"):
        cap = load_capture(seed)
    resolved_alignment_method = alignment_method
    alignment_warning = None
    if alignment_method == "calibrated":
        with timed_phase("alignment calibrated matrices"):
            warps = cap.get_warp_matrices(ref_index=5)
        logging.warning(
            "Using calibrated camera warp matrices for %s. This is fast, but "
            "should be treated as diagnostic until visually validated.",
            seed,
        )
    else:
        try:
            with timeout_after(alignment_timeout, f"SIFT alignment for {seed.name}"):
                with timed_phase("alignment SIFT_align_capture"):
                    warps = cap.SIFT_align_capture(ref=5, min_matches=min_matches, verbose=verbose)
        except (TimeoutError, UnboundLocalError) as exc:
            if not allow_calibrated_fallback:
                raise RuntimeError(
                    "SIFT band alignment failed and calibrated fallback is disabled. "
                    "Choose a better --alignment-seed, usually a sharp mid-flight "
                    "capture, or rerun with --allow-calibrated-fallback only for "
                    "diagnostic/non-production outputs."
                ) from exc
            # micasense.capture.SIFT_align_capture can leave kpi/kpr undefined when
            # a band has too few valid SIFT matches and the calibrated fallback is used.
            resolved_alignment_method = "calibrated_fallback_after_sift_failure"
            alignment_warning = repr(exc)
            logging.warning(
                "SIFT alignment failed; using calibrated camera warp matrices "
                "for %s: %s",
                seed,
                exc,
            )
            with timed_phase("alignment calibrated fallback"):
                warps = cap.get_warp_matrices(ref_index=5)
    meta = {
        "alignment_seed": str(seed),
        "alignment_capture_id": cap.uuid,
        "alignment_utc": cap.utc_time().isoformat(),
        "alignment_location": cap.location(),
        "alignment_min_matches": min_matches,
        "alignment_method": resolved_alignment_method,
    }
    if alignment_warning:
        meta["alignment_warning"] = alignment_warning
    return warps, meta


def load_warp_cache(path: Path) -> tuple[list[np.ndarray], dict]:
    with timed_phase("load warp cache"):
        with np.load(path, allow_pickle=False) as data:
            indexes = sorted(
                int(key.split("_", 1)[1]) for key in data.files if key.startswith("warp_")
            )
            warps = [data[f"warp_{index}"] for index in indexes]
            meta = json.loads(str(data["metadata"].item())) if "metadata" in data.files else {}
    logging.info("[ALIGN] loaded %s warp matrices from %s", len(warps), path)
    return warps, meta


def save_warp_cache(path: Path, warps: list[np.ndarray], metadata: dict) -> None:
    with timed_phase("save warp cache"):
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"warp_{index}": warp for index, warp in enumerate(warps)}
        arrays["metadata"] = np.array(json.dumps(metadata))
        np.savez_compressed(path, **arrays)
    logging.info("[ALIGN] saved %s warp matrices to %s", len(warps), path)


def scale_reflectance_to_uint16(stack: np.ndarray) -> np.ndarray:
    with timed_phase("scale reflectance to uint16"):
        stack = np.nan_to_num(stack, nan=0.0, posinf=2.0, neginf=0.0)
        stack = np.clip(stack, 0.0, 2.0)
        return np.rint(stack * 32767.0).astype(np.uint16)


def load_metashape_correction(path: Path | None, radiometry_mode: str) -> dict[str, float]:
    if radiometry_mode != "metashape_compatible":
        return dict(DEFAULT_METASHAPE_CORRECTION)
    if path is None:
        raise ValueError(
            "--radiometry-mode metashape_compatible requires --metashape-correction-json"
        )
    with timed_phase("load Metashape correction JSON"):
        payload = json.loads(path.read_text())
    factors = payload.get("band_correction_factors", payload)
    correction = dict(DEFAULT_METASHAPE_CORRECTION)
    for band_name in METASHAPE_MS_BAND_NAMES:
        if band_name not in factors:
            raise KeyError(f"missing correction factor for {band_name!r} in {path}")
        correction[band_name] = float(factors[band_name])
    if "Panchro" in factors:
        correction["Panchro"] = float(factors["Panchro"])
    logging.info("[RADIOMETRY] loaded Metashape correction factors: %s", correction)
    return correction


def capture_irradiance_for_mode(
    cap: Capture,
    radiometry_mode: str,
    panel_irradiance_values: list[float],
    panel_dls_to_panel_scale: list[float] | None,
) -> list[float]:
    if radiometry_mode in {"micasense_panel", "metashape_compatible"}:
        return panel_irradiance_values
    dls_irradiance = [float(value) for value in cap.dls_irradiance()]
    if radiometry_mode == "micasense_dls":
        return dls_irradiance
    if radiometry_mode == "panel_dls_tie":
        if panel_dls_to_panel_scale is None:
            raise ValueError("panel_dls_tie requires panel_dls_to_panel_scale metadata")
        if len(panel_dls_to_panel_scale) != len(dls_irradiance):
            raise ValueError(
                f"panel_dls_to_panel_scale length {len(panel_dls_to_panel_scale)} "
                f"does not match capture DLS length {len(dls_irradiance)}"
            )
        return [
            dls_value * float(scale)
            for dls_value, scale in zip(dls_irradiance, panel_dls_to_panel_scale)
        ]
    raise ValueError(f"unsupported radiometry mode: {radiometry_mode}")


def output_indexes_and_names(include_panchro: bool) -> tuple[tuple[int, ...], tuple[str, ...]]:
    indexes = tuple(METASHAPE_MS_BAND_INDEXES)
    names = tuple(METASHAPE_MS_BAND_NAMES)
    if include_panchro:
        indexes = indexes + (PANCHRO_BAND_INDEX,)
        names = names + (PANCHRO_BAND_NAME,)
    return indexes, names


def reorder_and_correct_aligned(
    aligned: np.ndarray,
    include_panchro: bool,
    correction: dict[str, float],
) -> tuple[np.ndarray, tuple[str, ...]]:
    indexes, names = output_indexes_and_names(include_panchro)
    stack = np.moveaxis(aligned[:, :, indexes], 2, 0).astype("float32", copy=False)
    factors = np.array([correction.get(name, 1.0) for name in names], dtype="float32")
    stack = stack * factors[:, None, None]
    return stack, names


def write_stack(
    path: Path,
    aligned: np.ndarray,
    include_panchro: bool,
    radiometry_mode: str,
    correction: dict[str, float],
    correction_source: Path | None,
) -> None:
    stack, names = reorder_and_correct_aligned(aligned, include_panchro, correction)
    data = scale_reflectance_to_uint16(stack)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": "uint16",
        "compress": "deflate",
        "tiled": True,
    }
    logging.info("[WRITE] path=%s shape=%s dtype=%s", path, data.shape, data.dtype)
    with timed_phase("write GeoTIFF"):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)
            for band_index, name in enumerate(names, start=1):
                dst.set_band_description(band_index, name)
            dst.update_tags(
                software="MicaSense imageprocessing via scripts/micasense_rededgep_preprocess.py",
                reflectance_scale="32767 = 1.0 reflectance",
                raw_band_order=", ".join(RAW_MS_BAND_NAMES + (PANCHRO_BAND_NAME,)),
                output_band_order=", ".join(names),
                radiometry_mode=radiometry_mode,
                metashape_correction_json=str(correction_source) if correction_source else "",
                metashape_correction_factors=json.dumps(
                    {name: correction.get(name, 1.0) for name in names},
                    sort_keys=True,
                ),
            )


def write_rgb_preview(path: Path, aligned: np.ndarray) -> None:
    with timed_phase(f"write QA preview {path.name}"):
        from PIL import Image

        rgb = aligned[:, :, (2, 1, 0)].astype("float32")
        lo = float(np.nanpercentile(rgb, 1))
        hi = float(np.nanpercentile(rgb, 99))
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.rint(rgb * 255).astype("uint8")).save(path)


def output_name(seed: Path) -> str:
    return seed.name.replace("_1.tif", "_6.tif")


def existing_stack_is_valid(
    path: Path,
    expected_bands: int,
    expected_band_names: tuple[str, ...],
    radiometry_mode: str,
    correction_source: Path | None,
) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as src:
            if src.count != expected_bands or src.width <= 0 or src.height <= 0:
                return False
            descriptions = tuple(description or "" for description in src.descriptions)
            if descriptions != expected_band_names:
                return False
            tags = src.tags()
            if tags.get("radiometry_mode", "micasense_panel") != radiometry_mode:
                return False
            expected_correction = str(correction_source) if correction_source else ""
            if tags.get("metashape_correction_json", "") != expected_correction:
                return False
            return True
    except rasterio.errors.RasterioIOError:
        return False


def process_capture(
    seed: Path,
    outdir: Path,
    warps: list[np.ndarray],
    panel_irradiance_values: list[float],
    panel_dls_to_panel_scale: list[float] | None,
    include_panchro: bool,
    radiometry_mode: str,
    correction: dict[str, float],
    correction_source: Path | None,
    qa_preview_dir: Path | None,
) -> dict:
    out_path = outdir / output_name(seed)
    expected_bands = 6 if include_panchro else 5
    expected_band_names = output_indexes_and_names(include_panchro)[1]
    with timed_phase(f"existing output check {seed.name}"):
        output_exists = existing_stack_is_valid(
            out_path,
            expected_bands,
            expected_band_names,
            radiometry_mode,
            correction_source,
        )
    if output_exists:
        return {
            "seed": str(seed),
            "output": str(out_path),
            "skipped_existing": True,
            "bands_written": expected_bands,
        }

    logging.info("[CAPTURE] seed=%s", seed)
    with timed_phase(f"load capture {seed.name}"):
        cap = load_capture(seed)
    with timed_phase(f"capture irradiance {seed.name}"):
        irradiance = capture_irradiance_for_mode(
            cap,
            radiometry_mode=radiometry_mode,
            panel_irradiance_values=panel_irradiance_values,
            panel_dls_to_panel_scale=panel_dls_to_panel_scale,
        )
    with timed_phase(f"radiometric pan-sharpen align {seed.name}"):
        cap.radiometric_pan_sharpened_aligned_capture(
            warp_matrices=warps, irradiance_list=irradiance, img_type="reflectance"
        )
    with timed_phase(f"extract aligned array {seed.name}"):
        aligned = cap._Capture__aligned_radiometric_pan_sharpened_capture[1]
        logging.info("[CAPTURE] aligned shape=%s dtype=%s", aligned.shape, aligned.dtype)
    with timed_phase(f"write stack {seed.name}"):
        write_stack(
            out_path,
            aligned,
            include_panchro=include_panchro,
            radiometry_mode=radiometry_mode,
            correction=correction,
            correction_source=correction_source,
        )
    preview_path = None
    if qa_preview_dir:
        preview_path = qa_preview_dir / output_name(seed).replace(".tif", "_rgb.png")
        write_rgb_preview(preview_path, aligned)
    return {
        "seed": str(seed),
        "output": str(out_path),
        "skipped_existing": False,
        "capture_id": cap.uuid,
        "utc": cap.utc_time().isoformat(),
        "location": cap.location(),
        "shape": list(aligned.shape),
        "bands_written": 6 if include_panchro else 5,
        "output_band_order": list(output_indexes_and_names(include_panchro)[1]),
        "radiometry_mode": radiometry_mode,
        "capture_irradiance": irradiance,
        "metashape_correction_json": str(correction_source) if correction_source else None,
        "qa_preview": str(preview_path) if preview_path else None,
    }


def process_capture_worker(
    args: tuple[
        Path,
        Path,
        list[np.ndarray],
        list[float],
        list[float] | None,
        bool,
        str,
        dict[str, float],
        Path | None,
        Path | None,
    ],
) -> dict:
    (
        seed,
        outdir,
        warps,
        panel_irradiance_values,
        panel_dls_to_panel_scale,
        include_panchro,
        radiometry_mode,
        correction,
        correction_source,
        qa_preview_dir,
    ) = args
    return process_capture(
        seed,
        outdir,
        warps,
        panel_irradiance_values,
        panel_dls_to_panel_scale,
        include_panchro,
        radiometry_mode,
        correction,
        correction_source,
        qa_preview_dir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-set", type=Path, required=True, help="Flight SET folder containing captures"
    )
    parser.add_argument(
        "--panel-set",
        type=Path,
        required=True,
        help="Panel SET folder used for reflectance calibration",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--alignment-seed",
        type=str,
        help="*_1.tif filename to use for SIFT alignment; default middle capture",
    )
    parser.add_argument(
        "--auto-alignment-seed",
        action="store_true",
        help="Automatically choose an alignment seed by ranking panchro texture/sharpness.",
    )
    parser.add_argument(
        "--alignment-candidate-count",
        type=int,
        default=120,
        help="Number of evenly spaced captures to score for --auto-alignment-seed. Use 0 for all.",
    )
    parser.add_argument(
        "--alignment-score-size",
        type=int,
        default=512,
        help="Maximum image side used for fast alignment candidate scoring.",
    )
    parser.add_argument(
        "--alignment-candidates-out",
        type=Path,
        help="Optional JSON file for the ranked alignment candidates.",
    )
    parser.add_argument(
        "--panel-seed", type=str, help="panel *_1.tif filename prefix; default first panel capture"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--capture-list",
        type=Path,
        help=(
            "Optional newline-delimited list of flight captures to process. "
            "Entries can be *_1.tif names, stems, or absolute paths."
        ),
    )
    parser.add_argument(
        "--auto-exclude-panels",
        action="store_true",
        help=(
            "Backward-compatible alias for --panel-strategy full. Full panel "
            "detection loads every capture and is slow on complete weeks."
        ),
    )
    parser.add_argument(
        "--panel-strategy",
        choices=("none", "full"),
        default="none",
        help=(
            "Panel filtering strategy. 'none' performs no per-capture panel scan; "
            "'full' runs MicaSense detect_panels on every capture."
        ),
    )
    parser.add_argument("--panel-detect-min-bands", type=int, default=5)
    parser.add_argument(
        "--include-panchro", action="store_true", help="Write a 6-band stack including panchro"
    )
    parser.add_argument(
        "--radiometry-mode",
        choices=RADIOMETRY_MODES,
        default="micasense_panel",
        help=(
            "micasense_panel uses one panel-derived irradiance for the whole SET. "
            "micasense_dls uses each capture's DLS irradiance. panel_dls_tie scales each "
            "capture's DLS irradiance by the same-flight panel/DLS ratio. "
            "metashape_compatible additionally applies frozen per-band correction factors."
        ),
    )
    parser.add_argument(
        "--metashape-correction-json",
        type=Path,
        help="Frozen correction JSON produced by calibrate_metashape_compatibility.py.",
    )
    parser.add_argument("--min-matches", type=int, default=10)
    parser.add_argument("--verbose-align", type=int, default=1)
    parser.add_argument(
        "--alignment-method",
        choices=("sift", "calibrated"),
        default="sift",
        help="Use SIFT alignment or calibrated camera matrices. Calibrated is for diagnostics.",
    )
    parser.add_argument(
        "--alignment-timeout",
        type=int,
        default=300,
        help="Seconds before aborting SIFT alignment. Use 0 to disable.",
    )
    parser.add_argument(
        "--warp-cache",
        type=Path,
        help="Optional .npz path for cached warp matrices. Existing cache is reused by default.",
    )
    parser.add_argument(
        "--refresh-warps",
        action="store_true",
        help="Recompute warp matrices even when --warp-cache already exists.",
    )
    parser.add_argument(
        "--qa-preview-dir",
        type=Path,
        help="Optional directory for RGB PNG previews of processed stacks.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel capture-processing workers. Start with 2 for RedEdge-P.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional structured timing log path. Defaults to stdout only.",
    )
    parser.add_argument(
        "--profile-out",
        type=Path,
        help="Optional cProfile output path for the whole run.",
    )
    parser.add_argument(
        "--profile-summary",
        type=Path,
        help="Optional text pstats summary path, sorted by cumulative time.",
    )
    parser.add_argument(
        "--allow-calibrated-fallback",
        action="store_true",
        help=(
            "Allow camera-calibration warp matrices if SIFT alignment fails. "
            "Use only for diagnostics because it can leave RedEdge-P bands "
            "spatially misregistered."
        ),
    )
    args = parser.parse_args()
    setup_logging(args.log_file)
    logging.info("[START] input_set=%s panel_set=%s outdir=%s", args.input_set, args.panel_set, args.outdir)
    correction = load_metashape_correction(args.metashape_correction_json, args.radiometry_mode)

    with timed_phase("input list seeds"):
        all_seeds = list(iter_capture_seeds(args.input_set))
    if not all_seeds:
        raise FileNotFoundError(f"no captures found in {args.input_set}")
    if args.capture_list:
        with timed_phase("resolve capture list"):
            seeds = resolve_capture_list(read_capture_list(args.capture_list), all_seeds)
        seed_source = str(args.capture_list)
    else:
        seeds = all_seeds
        seed_source = "input_set_recursive"

    seed_count_before_panel_exclusion = len(seeds)
    excluded_captures = []
    panel_strategy = "full" if args.auto_exclude_panels else args.panel_strategy
    if panel_strategy == "full":
        with timed_phase("auto exclude detected panels"):
            seeds, excluded_captures = exclude_detected_panels(
                seeds, args.panel_detect_min_bands
            )
    if not seeds:
        raise FileNotFoundError("no captures left to process after filtering")
    processing_seeds = seeds[: args.limit] if args.limit else seeds

    alignment_seed = seeds[len(seeds) // 2]
    alignment_candidate_rankings = []
    if args.alignment_seed:
        matches = [
            seed
            for seed in all_seeds
            if seed.name == args.alignment_seed or seed.stem == args.alignment_seed
        ]
        if not matches:
            raise FileNotFoundError(
                f"alignment seed {args.alignment_seed!r} not found in {args.input_set}"
            )
        alignment_seed = matches[0]
    if args.auto_alignment_seed:
        with timed_phase("rank alignment candidates"):
            alignment_candidate_rankings = rank_alignment_candidates(
                seeds,
                args.alignment_candidate_count,
                args.alignment_score_size,
            )
        if not alignment_candidate_rankings:
            raise RuntimeError("no alignment candidates could be scored")
        alignment_seed = Path(alignment_candidate_rankings[0]["seed"])
        logging.info("[ALIGN] auto-selected alignment seed %s", alignment_seed)
        if args.alignment_candidates_out:
            args.alignment_candidates_out.parent.mkdir(parents=True, exist_ok=True)
            args.alignment_candidates_out.write_text(
                json.dumps(alignment_candidate_rankings, indent=2)
            )

    with timed_phase("panel calibration"):
        irradiance, panel_meta = panel_irradiance(args.panel_set, args.panel_seed)
    panel_dls_to_panel_scale = panel_meta.get("panel_dls_to_panel_scale")
    with timed_phase("alignment warp computation"):
        if args.warp_cache and args.warp_cache.exists() and not args.refresh_warps:
            warps, alignment_meta = load_warp_cache(args.warp_cache)
            alignment_meta["alignment_method"] = f"cached_{alignment_meta.get('alignment_method', 'unknown')}"
            alignment_meta["warp_cache"] = str(args.warp_cache)
        else:
            warps, alignment_meta = compute_warps(
                alignment_seed,
                args.min_matches,
                args.verbose_align,
                args.allow_calibrated_fallback,
                args.alignment_method,
                args.alignment_timeout,
            )
            if args.warp_cache:
                alignment_meta["warp_cache"] = str(args.warp_cache)
                save_warp_cache(args.warp_cache, warps, alignment_meta)

    records = []
    failures = []
    skipped = 0
    workers = max(1, args.workers)
    logging.info("[WORKERS] capture processing workers=%s", workers)
    with timed_phase("process captures"):
        if workers == 1:
            for index, seed in enumerate(processing_seeds, start=1):
                try:
                    record = process_capture(
                        seed,
                        args.outdir,
                        warps,
                        irradiance,
                        panel_dls_to_panel_scale,
                        args.include_panchro,
                        args.radiometry_mode,
                        correction,
                        args.metashape_correction_json,
                        args.qa_preview_dir,
                    )
                    records.append(record)
                    if record.get("skipped_existing"):
                        skipped += 1
                        logging.info(
                            "[%s/%s] skipped existing %s",
                            index,
                            len(processing_seeds),
                            record["output"],
                        )
                    else:
                        logging.info(
                            "[%s/%s] wrote %s", index, len(processing_seeds), record["output"]
                        )
                except Exception as exc:
                    failures.append({"seed": str(seed), "error": repr(exc)})
                    logging.exception(
                        "[%s/%s] failed %s: %s", index, len(processing_seeds), seed, exc
                    )
        else:
            worker_args = [
                (
                    seed,
                    args.outdir,
                    warps,
                    irradiance,
                    panel_dls_to_panel_scale,
                    args.include_panchro,
                    args.radiometry_mode,
                    correction,
                    args.metashape_correction_json,
                    args.qa_preview_dir,
                )
                for seed in processing_seeds
            ]
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_seed = {
                    executor.submit(process_capture_worker, item): item[0] for item in worker_args
                }
                completed = 0
                for future in as_completed(future_to_seed):
                    seed = future_to_seed[future]
                    completed += 1
                    try:
                        record = future.result()
                        records.append(record)
                        if record.get("skipped_existing"):
                            skipped += 1
                            logging.info(
                                "[%s/%s] skipped existing %s",
                                completed,
                                len(processing_seeds),
                                record["output"],
                            )
                        else:
                            logging.info(
                                "[%s/%s] wrote %s",
                                completed,
                                len(processing_seeds),
                                record["output"],
                            )
                    except Exception as exc:
                        failures.append({"seed": str(seed), "error": repr(exc)})
                        logging.exception(
                            "[%s/%s] failed %s: %s",
                            completed,
                            len(processing_seeds),
                            seed,
                            exc,
                        )
            records.sort(key=lambda record: record["seed"])

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_set": str(args.input_set),
        "capture_list": str(args.capture_list) if args.capture_list else None,
        "seed_source": seed_source,
        "seed_count_before_panel_exclusion": seed_count_before_panel_exclusion,
        "seed_count_after_filtering": len(seeds),
        "seed_count_for_processing": len(processing_seeds),
        "panel_strategy": panel_strategy,
        "excluded_captures": excluded_captures,
        "outdir": str(args.outdir),
        "include_panchro": args.include_panchro,
        "raw_band_order": list(RAW_MS_BAND_NAMES) + [PANCHRO_BAND_NAME],
        "output_band_order": list(output_indexes_and_names(args.include_panchro)[1]),
        "radiometry_mode": args.radiometry_mode,
        "metashape_correction_json": str(args.metashape_correction_json)
        if args.metashape_correction_json
        else None,
        "metashape_correction_factors": correction,
        "workers": workers,
        "panel": panel_meta,
        "alignment": alignment_meta,
        "alignment_candidate_rankings": alignment_candidate_rankings[:20],
        "processed": len(records),
        "skipped_existing": skipped,
        "failed": len(failures),
        "records": records,
        "failures": failures,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    with timed_phase("write manifest"):
        (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logging.info("[DONE] processed=%s skipped=%s failed=%s", len(records), skipped, len(failures))
    return 2 if failures else 0


if __name__ == "__main__":
    parsed_profile_out = None
    parsed_profile_summary = None
    try:
        profile_parser = argparse.ArgumentParser(add_help=False)
        profile_parser.add_argument("--profile-out", type=Path)
        profile_parser.add_argument("--profile-summary", type=Path)
        profile_args, _ = profile_parser.parse_known_args()
        parsed_profile_out = profile_args.profile_out
        parsed_profile_summary = profile_args.profile_summary
    except Exception:
        parsed_profile_out = None
        parsed_profile_summary = None

    if parsed_profile_out or parsed_profile_summary:
        profiler = cProfile.Profile()
        try:
            exit_code = profiler.runcall(main)
        finally:
            if parsed_profile_out:
                parsed_profile_out.parent.mkdir(parents=True, exist_ok=True)
                profiler.dump_stats(parsed_profile_out)
            if parsed_profile_summary:
                parsed_profile_summary.parent.mkdir(parents=True, exist_ok=True)
                with parsed_profile_summary.open("w") as handle:
                    stats = pstats.Stats(profiler, stream=handle).sort_stats("cumulative")
                    stats.print_stats(80)
        raise SystemExit(exit_code)
    raise SystemExit(main())
