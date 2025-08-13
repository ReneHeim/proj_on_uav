import os
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio as rio
import yaml
from rasterio.transform import from_bounds


def _write_multiband_tif(path: Path, width=10, height=10, count=5, crs="EPSG:32632"):
    transform = from_bounds(0, 0, 10, 10, width, height)
    data = np.zeros((count, height, width), dtype=np.float32)
    for b in range(count):
        data[b, :, :] = (b + 1) * 0.01
    with rio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


def _write_singleband_tif(path: Path, width=10, height=10, crs="EPSG:32632"):
    transform = from_bounds(0, 0, 10, 10, width, height)
    data = np.ones((height, width), dtype=np.float32)
    with rio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def _write_camera_file(path: Path, photo_id: str):
    # matches parser: skip 2 rows, tab-separated, no header
    # columns: ['PhotoID','X','Y','Z','Omega','Phi','Kappa','r11','r12','r13','r21','r22','r23','r31','r32','r33']
    lines = [
        "# header1\n",
        "# header2\n",
        f"{photo_id}\t0\t0\t10\t0\t0\t0\t1\t0\t0\t0\t1\t0\t0\t0\t1\n",
    ]
    path.write_text("".join(lines))


def _write_polygon(path: Path, crs="EPSG:32632"):
    gdf = gpd.GeoDataFrame(
        {"id": [1]}, geometry=gpd.GeoSeries.from_wkt(["POLYGON((0 0,10 0,10 10,0 10,0 0))"])
    )
    gdf.set_crs(crs, inplace=True)
    # write to GeoPackage
    gdf.to_file(path, driver="GPKG")


def test_script_01_extract_e2e(tmp_path: Path):
    base = tmp_path
    orthos = base / "orthos"
    out = base / "out"
    orthos.mkdir()
    out.mkdir()

    ortho_path = orthos / "IMG_0001_0.tif"
    dem_path = base / "dem.tif"
    cam_path = base / "cams.txt"
    poly_path = base / "polys.gpkg"

    _write_multiband_tif(ortho_path)
    _write_singleband_tif(dem_path)
    _write_camera_file(cam_path, "IMG_0001_0")
    _write_polygon(poly_path)

    cfg = {
        "base_path": str(base),
        "inputs": {
            "date_time": {"start": "2024-01-01 12:00:00", "time_zone": "UTC"},
            "paths": {
                "cam_path": str(cam_path),
                "dem_path": str(dem_path),
                "orthophoto_path": str(orthos / "*.tif"),
                "ori": [str(orthos)],
                "mosaic_path": str(orthos / "mosaic.tif"),
                "ground_truth_coordinates": str(base / "gps.csv"),
                "polygon_file_path": str(poly_path),
            },
            "settings": {
                "number_of_processor": 1,
                "filter_radius": 1,
                "file_name": "e2e",
                "bands": 5,
                "target_crs": "EPSG:32632",
            },
        },
        "outputs": {"paths": {"main_out": str(out), "plot_out": str(out / "plots")}},
    }
    (base / "gps.csv").write_text("id,lon,lat\n1,0,0\n")
    cfg_path = base / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # run script 01
    proc = subprocess.run(
        ["python", "-m", "src.01_main_extract_data", "--config", str(cfg_path)], capture_output=True
    )
    assert proc.returncode == 0, proc.stderr.decode()

    # expect at least one parquet output
    outputs = list(out.glob("*.parquet"))
    if not outputs:
        # show stderr/stdout for debugging
        print(proc.stdout.decode())
        print(proc.stderr.decode())
    assert outputs, "No parquet produced by extract script"
