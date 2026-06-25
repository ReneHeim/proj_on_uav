#!/usr/bin/env python3
"""Process one ONCERCO RedEdge-P week with Agisoft Metashape.

Run this script with Metashape's bundled Python, for example:

    metashape.sh -r scripts/metashape_process_rededgep_week.py -- \
      --raw-week /mnt/data/ONCERCO/data/raw/2025/week3/rededgep \
      --out-root /tmp/metashape_repro_week3 \
      --project /tmp/metashape_repro_week3/week3.psx \
      --mode panels_sun

The script intentionally uses the real Agisoft reflectance calibration rather
than trying to reproduce its proprietary implementation in NumPy.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def import_metashape():
    try:
        import Metashape  # type: ignore
    except Exception as exc:  # pragma: no cover - only runs outside Metashape.
        raise RuntimeError(
            "Metashape Python module is not available. Run this script with "
            "Agisoft Metashape's Python, for example `metashape.sh -r ...`, "
            "not the system Python."
        ) from exc
    return Metashape


def setup_logging(out_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = out_root / "logs" / f"metashape_process_rededgep_week_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )
    return log_path


def phase(name: str, started: float) -> float:
    logging.info("[PHASE] %s: %.1fs", name, time.perf_counter() - started)
    return time.perf_counter()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-week", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("panels_sun", "panels_only", "sun_only"),
        default="panels_sun",
        help="Reflectance calibration inputs passed to chunk.calibrateReflectance().",
    )
    parser.add_argument("--limit-captures", type=int, help="Debug limit by capture id.")
    parser.add_argument(
        "--downscale",
        type=int,
        default=2,
        help="Metashape alignment downscale. Use 1 for highest quality, 2 for draft/repro tests.",
    )
    parser.add_argument(
        "--skip-dense", action="store_true", help="Use sparse/depth defaults where possible."
    )
    parser.add_argument(
        "--crs", default="EPSG::4326", help="Output CRS string for exported rasters."
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def capture_id(path: Path) -> str:
    return path.name.split("_")[1]


def collect_photo_groups(raw_week: Path, limit_captures: int | None) -> list[list[str]]:
    seeds = sorted(raw_week.glob("**/IMG_*_1.tif"))
    if limit_captures:
        seeds = seeds[:limit_captures]
    groups = []
    for seed in seeds:
        files = [
            str(seed.with_name(seed.name.replace("_1.tif", f"_{suffix}.tif")))
            for suffix in range(1, 7)
        ]
        if all(Path(path).exists() for path in files):
            groups.append(files)
        else:
            logging.warning("[SKIP] incomplete capture %s", seed)
    if not groups:
        raise FileNotFoundError(f"no complete RedEdge-P captures found under {raw_week}")
    return groups


def write_capture_list(path: Path, groups: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["capture_id", "band1", "band2", "band3", "band4", "band5", "band6"])
        for group in groups:
            writer.writerow([capture_id(Path(group[0])), *group])


def set_normalize_band_sensitivity(Metashape, chunk) -> None:
    # Metashape versions expose sensor/band calibration slightly differently.
    # Prefer the documented high-level calibration; this setter is best-effort.
    for sensor in chunk.sensors:
        try:
            sensor.normalize_bands = True
            logging.info("[CAL] set normalize_bands=True on sensor %s", sensor.label)
        except Exception:
            pass
        try:
            calib = sensor.calibration
            if hasattr(calib, "normalize_bands"):
                calib.normalize_bands = True
                logging.info(
                    "[CAL] set calibration.normalize_bands=True on sensor %s", sensor.label
                )
        except Exception:
            pass


def calibrate_reflectance(Metashape, chunk, mode: str) -> dict:
    use_panels = mode in {"panels_sun", "panels_only"}
    use_sun = mode in {"panels_sun", "sun_only"}
    located = None
    if use_panels:
        logging.info("[CAL] locateReflectancePanels")
        located = chunk.locateReflectancePanels()
        logging.info("[CAL] locateReflectancePanels result=%s", located)
    set_normalize_band_sensitivity(Metashape, chunk)
    logging.info("[CAL] calibrateReflectance panels=%s sun=%s", use_panels, use_sun)
    chunk.calibrateReflectance(
        use_reflectance_panels=use_panels,
        use_sun_sensor=use_sun,
    )
    return {
        "mode": mode,
        "use_reflectance_panels": use_panels,
        "use_sun_sensor": use_sun,
        "locate_reflectance_panels_result": str(located),
    }


def process_geometry(Metashape, chunk, downscale: int, skip_dense: bool) -> None:
    logging.info("[GEOM] matchPhotos downscale=%s", downscale)
    chunk.matchPhotos(downscale=downscale, generic_preselection=True, reference_preselection=True)
    logging.info("[GEOM] alignCameras")
    chunk.alignCameras()
    if skip_dense:
        logging.info("[GEOM] skip dense/depth by request")
        return
    logging.info("[GEOM] buildDepthMaps")
    chunk.buildDepthMaps(downscale=downscale, filter_mode=Metashape.MildFiltering)
    logging.info("[GEOM] buildDem")
    chunk.buildDem(source_data=Metashape.DepthMapsData)


def export_per_camera_orthophotos(Metashape, chunk, out_dir: Path, crs: str) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    coordinate_system = Metashape.CoordinateSystem(crs)
    cameras = [camera for camera in chunk.cameras if camera.transform and camera.enabled]
    logging.info("[EXPORT] cameras=%s out_dir=%s", len(cameras), out_dir)
    for index, camera in enumerate(cameras, start=1):
        label = Path(camera.label).stem
        capture = label.rsplit("_", 1)[0] if "_" in label else label
        out_path = out_dir / f"{capture}_6.tif"
        try:
            logging.info("[EXPORT] %s/%s %s -> %s", index, len(cameras), camera.label, out_path)
            chunk.exportRaster(
                path=str(out_path),
                source_data=Metashape.OrthomosaicData,
                projection=coordinate_system,
                image_compression=Metashape.ImageCompression(),
                white_background=False,
                save_alpha=True,
            )
            records.append({"camera": camera.label, "output": str(out_path), "status": "ok"})
        except Exception as exc:
            logging.exception("[EXPORT] failed %s: %s", camera.label, exc)
            records.append(
                {
                    "camera": camera.label,
                    "output": str(out_path),
                    "status": "error",
                    "error": repr(exc),
                }
            )
    return records


def main(argv: list[str]) -> int:
    # Metashape passes script arguments after `--`; tolerate a leading separator.
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(args.out_root)
    Metashape = import_metashape()
    started = time.perf_counter()

    logging.info("[START] raw_week=%s project=%s mode=%s", args.raw_week, args.project, args.mode)
    groups = collect_photo_groups(args.raw_week, args.limit_captures)
    capture_list_path = args.out_root / "metadata" / "capture_list.csv"
    write_capture_list(capture_list_path, groups)
    started = phase("collect photo groups", started)

    doc = Metashape.Document()
    if args.project.exists() and not args.overwrite:
        logging.info("[PROJECT] open existing %s", args.project)
        doc.open(str(args.project))
        chunk = doc.chunk
    else:
        if args.project.exists() and args.overwrite:
            logging.info("[PROJECT] overwrite existing %s", args.project)
        chunk = doc.addChunk()
        flattened = [path for group in groups for path in group]
        logging.info("[PROJECT] addPhotos count=%s layout=multiplane", len(flattened))
        chunk.addPhotos(flattened, filegroups=[6] * len(groups), layout=Metashape.MultiplaneLayout)
        doc.save(str(args.project))
    started = phase("project setup/add photos", started)

    calibration_meta = calibrate_reflectance(Metashape, chunk, args.mode)
    doc.save()
    started = phase("reflectance calibration", started)

    process_geometry(Metashape, chunk, args.downscale, args.skip_dense)
    doc.save()
    started = phase("geometry processing", started)

    if not args.skip_dense:
        logging.info("[ORTHO] buildOrthomosaic")
        chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)
        doc.save()
    started = phase("orthomosaic build", started)

    export_records = []
    if not args.skip_dense:
        export_records = export_per_camera_orthophotos(
            Metashape,
            chunk,
            args.out_root / "orthophotos",
            args.crs,
        )
    started = phase("export rasters", started)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/metashape_process_rededgep_week.py",
        "metashape_version": getattr(Metashape.app, "version", "unknown"),
        "raw_week": str(args.raw_week),
        "project": str(args.project),
        "out_root": str(args.out_root),
        "capture_count": len(groups),
        "capture_list": str(capture_list_path),
        "calibration": calibration_meta,
        "downscale": args.downscale,
        "skip_dense": args.skip_dense,
        "crs": args.crs,
        "exports": export_records,
        "log": str(log_path),
    }
    manifest_path = args.out_root / "metadata" / "metashape_process_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logging.info("[DONE] manifest=%s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
