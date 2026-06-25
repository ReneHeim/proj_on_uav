"""Create Metashape-style per-capture orthophotos from calibrated MicaSense stacks.

The script uses an ODM/OpenSfM reconstruction and DSM for geometry, but reads
the already calibrated 5-band RedEdge-P stacks as the radiometric source. The
final output is one EPSG:4326 Float32 GeoTIFF per capture, named IMG_####_6.tif.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from orthority.factory import FrameCameras
from orthority.ortho import Ortho
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject

BAND_NAMES = ("Blue", "Green", "Red", "NIR", "Red edge")
REFLECTANCE_SCALE = 32767.0


def capture_stem(stack_path: Path) -> str:
    if not stack_path.name.endswith("_6.tif"):
        raise ValueError(f"expected IMG_*_6.tif stack, got {stack_path.name}")
    return stack_path.name[: -len("_6.tif")]


def parse_capture_list(value: str | None) -> set[str] | None:
    if not value:
        return None
    captures = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item.endswith("_6.tif"):
            item = capture_stem(Path(item))
        captures.add(item)
    return captures


def select_stacks(stack_dir: Path, captures: set[str] | None, limit: int | None) -> list[Path]:
    stacks = sorted(stack_dir.glob("IMG_*_6.tif"))
    if captures is not None:
        stacks = [path for path in stacks if capture_stem(path) in captures]
    if limit is not None:
        stacks = stacks[:limit]
    if not stacks:
        raise FileNotFoundError(f"no stacks selected in {stack_dir}")
    return stacks


def pose_spread(cameras: FrameCameras, capture: str) -> dict:
    xyz_values = []
    missing = []
    for suffix in range(1, 6):
        name = f"{capture}_{suffix}.tif"
        try:
            cam = cameras.get(name)
        except Exception:
            missing.append(name)
            continue
        xyz_values.append(np.asarray(cam.pos, dtype=np.float64))

    if missing or len(xyz_values) != 5:
        return {"complete": False, "missing": missing}

    xyz = np.vstack(xyz_values)
    return {
        "complete": True,
        "max_xyz_range": float(np.ptp(xyz, axis=0).max()),
    }


def orthorectify_to_temp(
    stack_path: Path,
    temp_path: Path,
    dem_path: Path,
    cameras: FrameCameras,
    world_crs: CRS,
    reference_band: int,
    overwrite: bool,
) -> None:
    capture = capture_stem(stack_path)
    camera = cameras.get(f"{capture}_{reference_band}.tif")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    ortho = Ortho(stack_path, dem_path, camera, crs=world_crs, dem_band=1)
    ortho.process(
        temp_path,
        dtype="uint16",
        compress="deflate",
        build_ovw=False,
        overwrite=overwrite,
        progress=False,
    )


def reproject_scaled_to_epsg4326(temp_path: Path, out_path: Path, overwrite: bool) -> dict:
    if out_path.exists() and not overwrite:
        with rasterio.open(out_path) as src:
            return {
                "output": str(out_path),
                "skipped_existing": True,
                "width": src.width,
                "height": src.height,
                "crs": src.crs.to_string() if src.crs else None,
                "bands": src.count,
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dst_crs = CRS.from_epsg(4326)
    with rasterio.open(temp_path) as src:
        left, bottom, right, top = array_bounds(src.height, src.width, src.transform)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, left, bottom, right, top
        )
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=5,
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=np.nan,
            compress="deflate",
            tiled=True,
            interleave="pixel",
        )

        stats = []
        with rasterio.open(out_path, "w", **profile) as dst:
            for band_index, band_name in enumerate(BAND_NAMES, start=1):
                source = src.read(band_index).astype(np.float32) / REFLECTANCE_SCALE
                source_mask = src.read_masks(band_index)
                dest = np.full((height, width), np.nan, dtype=np.float32)
                dest_mask = np.zeros((height, width), dtype=np.uint8)
                reproject(
                    source,
                    dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                reproject(
                    source_mask,
                    dest_mask,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    src_nodata=0,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
                dest[dest_mask == 0] = np.nan
                dst.write(dest, band_index)
                dst.set_band_description(band_index, band_name)
                valid = dest[np.isfinite(dest)]
                stats.append(
                    {
                        "band": band_name,
                        "valid_pixels": int(valid.size),
                        "min": float(valid.min()) if valid.size else math.nan,
                        "max": float(valid.max()) if valid.size else math.nan,
                        "mean": float(valid.mean()) if valid.size else math.nan,
                    }
                )
            dst.update_tags(
                software="Orthority + orthorectify_micasense_per_capture.py",
                source_temp=str(temp_path),
                reflectance_scale="float32 reflectance; source uint16 divided by 32767",
            )

    return {
        "output": str(out_path),
        "skipped_existing": False,
        "width": width,
        "height": height,
        "crs": dst_crs.to_string(),
        "bands": 5,
        "stats": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odm-project", type=Path, required=True)
    parser.add_argument("--stack-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--captures", help="Comma-separated capture stems or IMG_*_6.tif names")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reference-band", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    reconstruction = args.odm_project / "opensfm" / "reconstruction.json"
    dem_path = args.odm_project / "odm_dem" / "dsm.tif"
    if not reconstruction.exists():
        raise FileNotFoundError(reconstruction)
    if not dem_path.exists():
        raise FileNotFoundError(dem_path)

    with rasterio.open(dem_path) as dem_ds:
        world_crs = dem_ds.crs
    if world_crs is None:
        raise RuntimeError(f"DSM has no CRS: {dem_path}")

    cameras = FrameCameras(
        int_param=reconstruction,
        ext_param=reconstruction,
        io_kwargs={"crs": world_crs},
        cam_kwargs={"distort": True, "alpha": 1.0},
    )
    stacks = select_stacks(args.stack_dir, parse_capture_list(args.captures), args.limit)
    temp_dir = args.temp_dir or args.out_dir.parent / f".{args.out_dir.name}_tmp"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    records = []
    failures = []
    for index, stack_path in enumerate(stacks, start=1):
        capture = capture_stem(stack_path)
        temp_path = temp_dir / f"{capture}_6_UTM.tif"
        out_path = args.out_dir / f"{capture}_6.tif"
        try:
            if out_path.exists() and not args.overwrite:
                with rasterio.open(out_path) as ds:
                    record = {
                        "path": str(out_path),
                        "skipped_existing": True,
                        "crs": ds.crs.to_string() if ds.crs else None,
                        "width": ds.width,
                        "height": ds.height,
                        "count": ds.count,
                        "dtypes": list(ds.dtypes),
                        "descriptions": list(ds.descriptions),
                    }
                record.update(
                    {
                        "capture": capture,
                        "source_stack": str(stack_path),
                        "temp_utm": str(temp_path),
                        "reference_band": args.reference_band,
                        "pose_spread": None,
                    }
                )
                records.append(record)
                print(f"[{index}/{len(stacks)}] skipped existing {out_path}")
                continue
            spread = pose_spread(cameras, capture)
            if not spread.get("complete"):
                raise RuntimeError(f"incomplete ODM band poses: {spread}")
            orthorectify_to_temp(
                stack_path,
                temp_path,
                dem_path,
                cameras,
                world_crs,
                args.reference_band,
                args.overwrite,
            )
            record = reproject_scaled_to_epsg4326(temp_path, out_path, args.overwrite)
            record.update(
                {
                    "capture": capture,
                    "source_stack": str(stack_path),
                    "temp_utm": str(temp_path),
                    "reference_band": args.reference_band,
                    "pose_spread": spread,
                }
            )
            records.append(record)
            print(f"[{index}/{len(stacks)}] wrote {out_path}")
        except Exception as exc:
            failures.append(
                {"capture": capture, "source_stack": str(stack_path), "error": repr(exc)}
            )
            print(f"[{index}/{len(stacks)}] failed {capture}: {exc}")

    if not args.keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "odm_project": str(args.odm_project),
        "stack_dir": str(args.stack_dir),
        "out_dir": str(args.out_dir),
        "temp_dir": str(temp_dir),
        "kept_temp": args.keep_temp,
        "world_crs_used_for_orthority": world_crs.to_string(),
        "final_crs": "EPSG:4326",
        "reference_band": args.reference_band,
        "selected": len(stacks),
        "processed": len(records),
        "failed": len(failures),
        "records": records,
        "failures": failures,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"processed={len(records)} failed={len(failures)} manifest={args.out_dir / 'manifest.json'}"
    )
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
