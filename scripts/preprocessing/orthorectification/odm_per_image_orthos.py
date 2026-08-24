#!/usr/bin/env python3
"""Create Metashape-style per-capture orthophotos from an ODM project.

This uses ODM's panchro OpenSfM camera poses and DEM, then samples the aligned
undistorted RedEdge-P band images into one Float32 GeoTIFF per capture.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

BAND_SUFFIXES = ("1", "2", "3", "4", "5")


def read_offsets(dataset: Path) -> tuple[int, int]:
    coords = dataset / "odm_georeferencing" / "coords.txt"
    lines = coords.read_text().splitlines()
    x, y = lines[1].split()[:2]
    return int(x), int(y)


def camera_origin(
    rotation_vec: list[float], translation: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    rotation = Rotation.from_rotvec(rotation_vec).as_matrix()
    translation_vec = np.asarray(translation, dtype=np.float64)
    origin = -rotation.T @ translation_vec
    return rotation, origin


def dem_index(transform: Affine, x: np.ndarray | float, y: np.ndarray | float):
    col = (x - transform.c) / transform.a
    row = (y - transform.f) / transform.e
    return col, row


def dem_xy_center(transform: Affine, col: np.ndarray, row: np.ndarray):
    x = (col + 0.5) * transform.a + transform.c
    y = (row + 0.5) * transform.e + transform.f
    return x, y


def corner_to_dem_index(
    cpx: float,
    cpy: float,
    rotation: np.ndarray,
    origin: np.ndarray,
    focal_px: float,
    dem_min: float,
    offset_x: int,
    offset_y: int,
    transform: Affine,
) -> tuple[float, float]:
    a1, b1, c1 = rotation[0]
    a2, b2, c2 = rotation[1]
    a3, b3, c3 = rotation[2]
    xs, ys, zs = origin
    za = dem_min

    m = a3 * b1 * cpy - a1 * b3 * cpy - (a3 * b2 - a2 * b3) * cpx - (a2 * b1 - a1 * b2) * focal_px
    xa = (
        offset_x
        + (
            m * xs
            + (
                b3 * c1 * cpy
                - b1 * c3 * cpy
                - (b3 * c2 - b2 * c3) * cpx
                - (b2 * c1 - b1 * c2) * focal_px
            )
            * za
            - (
                b3 * c1 * cpy
                - b1 * c3 * cpy
                - (b3 * c2 - b2 * c3) * cpx
                - (b2 * c1 - b1 * c2) * focal_px
            )
            * zs
        )
        / m
    )
    ya = (
        offset_y
        + (
            m * ys
            - (
                a3 * c1 * cpy
                - a1 * c3 * cpy
                - (a3 * c2 - a2 * c3) * cpx
                - (a2 * c1 - a1 * c2) * focal_px
            )
            * za
            + (
                a3 * c1 * cpy
                - a1 * c3 * cpy
                - (a3 * c2 - a2 * c3) * cpx
                - (a2 * c1 - a1 * c2) * focal_px
            )
            * zs
        )
        / m
    )
    return dem_index(transform, xa, ya)


def capture_id(shot_id: str) -> str:
    return re.sub(r"_6\.tif$", "", shot_id)


def output_name(shot_id: str) -> str:
    match = re.search(r"IMG_(\d+)_6\.tif$", shot_id)
    if match:
        return f"IMG_{match.group(1)}_6.tif"
    return shot_id


def valid_dem_values(dem: np.ndarray, nodata: float | None) -> np.ndarray:
    valid = np.isfinite(dem)
    if nodata is not None and np.isfinite(nodata):
        valid &= dem != nodata
    return dem[valid]


def orthorectify_capture(
    dataset: Path,
    shot_id: str,
    shot: dict,
    cameras: dict,
    dem_ds,
    dem: np.ndarray,
    dem_min: float,
    outdir: Path,
) -> bool:
    camera = cameras[shot["camera"]]
    focal = float(camera.get("focal_x", camera.get("focal")))
    rotation, origin = camera_origin(shot["rotation"], shot["translation"])
    offset_x, offset_y = read_offsets(dataset)

    base = capture_id(shot_id)
    image_paths = [
        dataset / "opensfm" / "undistorted" / "images" / f"{base}_{suffix}.tif"
        for suffix in BAND_SUFFIXES
    ]
    if not all(path.exists() for path in image_paths):
        missing = [path.name for path in image_paths if not path.exists()]
        print(f"skip {shot_id}: missing {', '.join(missing)}")
        return False

    with rasterio.open(image_paths[0]) as src0:
        img_h = src0.height
        img_w = src0.width

    half_w = (img_w - 1) / 2.0
    half_h = (img_h - 1) / 2.0
    focal_px = focal * max(img_h, img_w)
    corners = [
        corner_to_dem_index(
            -half_w,
            -half_h,
            rotation,
            origin,
            focal_px,
            dem_min,
            offset_x,
            offset_y,
            dem_ds.transform,
        ),
        corner_to_dem_index(
            half_w,
            -half_h,
            rotation,
            origin,
            focal_px,
            dem_min,
            offset_x,
            offset_y,
            dem_ds.transform,
        ),
        corner_to_dem_index(
            half_w,
            half_h,
            rotation,
            origin,
            focal_px,
            dem_min,
            offset_x,
            offset_y,
            dem_ds.transform,
        ),
        corner_to_dem_index(
            -half_w,
            half_h,
            rotation,
            origin,
            focal_px,
            dem_min,
            offset_x,
            offset_y,
            dem_ds.transform,
        ),
    ]
    cols = [c[0] for c in corners]
    rows = [c[1] for c in corners]
    min_col = max(0, min(dem.shape[1] - 1, math.floor(min(cols))))
    max_col = max(0, min(dem.shape[1] - 1, math.ceil(max(cols))))
    min_row = max(0, min(dem.shape[0] - 1, math.floor(min(rows))))
    max_row = max(0, min(dem.shape[0] - 1, math.ceil(max(rows))))

    grid_rows, grid_cols = np.mgrid[min_row : max_row + 1, min_col : max_col + 1]
    za = dem[grid_rows, grid_cols].astype(np.float64)
    dem_valid = np.isfinite(za)
    if dem_ds.nodata is not None and np.isfinite(dem_ds.nodata):
        dem_valid &= za != dem_ds.nodata
    xa, ya = dem_xy_center(dem_ds.transform, grid_cols, grid_rows)
    xa = xa - offset_x
    ya = ya - offset_y

    dx = xa - origin[0]
    dy = ya - origin[1]
    dz = za - origin[2]
    a1, b1, c1 = rotation[0]
    a2, b2, c2 = rotation[1]
    a3, b3, c3 = rotation[2]
    den = a3 * dx + b3 * dy + c3 * dz
    x = half_w - (focal_px * (a1 * dx + b1 * dy + c1 * dz) / den)
    y = half_h - (focal_px * (a2 * dx + b2 * dy + c2 * dz) / den)
    src_x = img_w - 1 - x
    src_y = img_h - 1 - y
    valid = (
        dem_valid
        & np.isfinite(src_x)
        & np.isfinite(src_y)
        & (src_x >= 0)
        & (src_y >= 0)
        & (src_x <= img_w - 1)
        & (src_y <= img_h - 1)
    )
    if not valid.any():
        print(f"skip {shot_id}: no DEM overlap")
        return False

    valid_rows, valid_cols = np.where(valid)
    crop_r0, crop_r1 = valid_rows.min(), valid_rows.max() + 1
    crop_c0, crop_c1 = valid_cols.min(), valid_cols.max() + 1
    src_x = src_x[crop_r0:crop_r1, crop_c0:crop_c1]
    src_y = src_y[crop_r0:crop_r1, crop_c0:crop_c1]
    valid = valid[crop_r0:crop_r1, crop_c0:crop_c1]

    out = np.full((len(BAND_SUFFIXES), valid.shape[0], valid.shape[1]), np.nan, dtype=np.float32)
    coords = np.vstack([src_y.ravel(), src_x.ravel()])
    for index, path in enumerate(image_paths):
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
        sampled = map_coordinates(band, coords, order=1, mode="constant", cval=np.nan).reshape(
            valid.shape
        )
        sampled[~valid] = np.nan
        out[index] = sampled.astype(np.float32)

    out_transform = dem_ds.transform * Affine.translation(min_col + crop_c0, min_row + crop_r0)
    profile = {
        "driver": "GTiff",
        "height": out.shape[1],
        "width": out.shape[2],
        "count": out.shape[0],
        "dtype": "float32",
        "crs": dem_ds.crs,
        "transform": out_transform,
        "nodata": np.nan,
        "compress": "deflate",
        "tiled": True,
    }
    out_path = outdir / output_name(shot_id)
    outdir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out)
        for idx, name in enumerate(("Blue", "Green", "Red", "NIR", "Rededge"), start=1):
            dst.set_band_description(idx, name)
    print(f"wrote {out_path} {out.shape[2]}x{out.shape[1]}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="ODM project directory")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--images", help="Comma-separated panchro shot ids; default processes all shots"
    )
    parser.add_argument("--limit", type=int, help="Maximum number of shots to process")
    args = parser.parse_args()

    reconstruction = json.loads((args.dataset / "opensfm" / "reconstruction.json").read_text())[0]
    shots = reconstruction["shots"]
    cameras = reconstruction["cameras"]
    selected = args.images.split(",") if args.images else list(shots)
    if args.limit:
        selected = selected[: args.limit]

    dem_path = args.dataset / "odm_dem" / "dsm.tif"
    with rasterio.open(dem_path) as dem_ds:
        dem = dem_ds.read(1)
        dem_values = valid_dem_values(dem, dem_ds.nodata)
        if dem_values.size == 0:
            raise RuntimeError(f"DEM has no valid values: {dem_path}")
        dem_min = float(dem_values.min())
        ok = 0
        for shot_id in selected:
            if shot_id not in shots:
                print(f"skip {shot_id}: not in reconstruction")
                continue
            ok += orthorectify_capture(
                args.dataset, shot_id, shots[shot_id], cameras, dem_ds, dem, dem_min, args.outdir
            )
    print(f"processed {ok}/{len(selected)} captures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
