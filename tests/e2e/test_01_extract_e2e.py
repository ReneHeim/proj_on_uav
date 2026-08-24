import os
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import rasterio as rio
import yaml
from rasterio.transform import from_bounds


def _write_multiband_tif(path, width=10, height=10, count=5, crs="EPSG:32632"):
    transform = from_bounds(0, 0, 10, 10, width, height)
    data = np.zeros((count, height, width), dtype=np.float32)
    for b in range(count):
        data[b] = (b + 1) * 0.01
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


def _write_singleband_tif(path, width=10, height=10, crs="EPSG:32632"):
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


def _write_camera_file(path, photo_id):
    lines = [
        "# header1\n",
        "# header2\n",
        f"{photo_id}\t0\t0\t10\t0\t0\t0\t1\t0\t0\t0\t1\t0\t0\t0\t1\n",
    ]
    path.write_text("".join(lines))


def _build_config(base, cam_path, dem_path, orthos_path, poly_path, out_dir):
    return {
        "base_path": str(base),
        "inputs": {
            "date_time": {"start": "2024-01-01 12:00:00", "time_zone": "UTC"},
            "paths": {
                "cam_path": str(cam_path),
                "dem_path": str(dem_path),
                "orthophoto_path": str(orthos_path / "*.tif"),
                "ori": [str(orthos_path)],
                "mosaic_path": str(orthos_path / "mosaic.tif"),
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
        "outputs": {"paths": {"main_out": str(out_dir), "plot_out": str(out_dir / "plots")}},
    }


def _run_extract(cfg_path):
    proc = subprocess.run(
        ["python", "-m", "main_extract", "--config", str(cfg_path), "--no-polygon"],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    return proc


def test_script_01_extract_e2e(tmp_path):
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

    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=gpd.GeoSeries.from_wkt(["POLYGON((0 0,10 0,10 10,0 10,0 0))"]),
    )
    gdf.set_crs("EPSG:32632", inplace=True)
    gdf.to_file(poly_path, driver="GPKG")

    cfg = _build_config(base, cam_path, dem_path, orthos, poly_path, out)
    (base / "gps.csv").write_text("id,lon,lat\n1,0,0\n")
    cfg_path = base / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    proc = _run_extract(cfg_path)

    outputs = list(out.glob("*.parquet"))
    if not outputs:
        print(proc.stdout.decode())
        print(proc.stderr.decode())
    assert outputs, "No parquet produced by extract script"

    df = pl.read_parquet(outputs[0])
    expected_cols = {
        "Xw",
        "Yw",
        "band1",
        "band2",
        "band3",
        "band4",
        "band5",
        "vza",
        "vaa",
        "sunelev",
        "saa",
    }
    missing = expected_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    total_pixels = 10 * 10
    assert 0 < len(df) <= total_pixels, f"Unexpected row count: {len(df)}"

    for col in [f"band{i}" for i in range(1, 6)]:
        vals = df[col].to_numpy()
        assert np.all(np.isfinite(vals)), f"Non-finite values in {col}"

    vza = df["vza"].to_numpy()
    valid_vza = vza[np.isfinite(vza)]
    assert np.all((valid_vza >= 0) & (valid_vza <= 90)), f"vza out of range: {valid_vza}"


def test_pipeline_produces_valid_angles(tmp_path):
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

    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=gpd.GeoSeries.from_wkt(["POLYGON((0 0,10 0,10 10,0 10,0 0))"]),
    )
    gdf.set_crs("EPSG:32632", inplace=True)
    gdf.to_file(poly_path, driver="GPKG")

    cfg = _build_config(base, cam_path, dem_path, orthos, poly_path, out)
    (base / "gps.csv").write_text("id,lon,lat\n1,0,0\n")
    cfg_path = base / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    _run_extract(cfg_path)
    outputs = list(out.glob("*.parquet"))
    assert outputs, "No parquet produced"

    df = pl.read_parquet(outputs[0])

    vza = df["vza"].to_numpy()
    vaa = df["vaa"].to_numpy()
    sunelev = df["sunelev"].to_numpy()
    saa_col = df["saa"].to_numpy()

    finite_vza = vza[np.isfinite(vza)]
    assert len(finite_vza) > 0, "No valid vza values"
    assert finite_vza.min() >= 0, f"vza min too low: {finite_vza.min()}"
    assert finite_vza.max() <= 90, f"vza max too high: {finite_vza.max()}"

    sunelev_finite = sunelev[np.isfinite(sunelev)]
    assert len(sunelev_finite) > 0
    assert sunelev_finite.min() >= -90

    finite_vaa = vaa[np.isfinite(vaa)]
    assert len(finite_vaa) > 0
    assert finite_vaa.min() >= 0
    assert finite_vaa.max() <= 360

    finite_saa = saa_col[np.isfinite(saa_col)]
    assert len(finite_saa) > 0
    assert finite_saa.min() >= 0
    assert finite_saa.max() <= 360
