import os
import subprocess
from pathlib import Path

import geopandas as gpd
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


def _build_config(base, cam_path, dem_path, orthos_path, poly_path, main_out, plot_out):
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
        "outputs": {"paths": {"main_out": str(main_out), "plot_out": str(plot_out)}},
    }


def test_script_02_filter_e2e(tmp_path):
    # Re-use output from previous test location if running together is not guaranteed;
    # Here we only sanity-run the CLI to ensure it accepts --help
    proc = subprocess.run(["python", "-m", "filtering", "--help"], capture_output=True)
    assert proc.returncode == 0


def test_filter_pipeline(tmp_path: Path):
    base = tmp_path
    orthos = base / "orthos"
    main_out = base / "out"
    plots_out = base / "plots"
    poly_out = base / "polygon_out"
    for d in [orthos, main_out, plots_out, poly_out]:
        d.mkdir()

    ortho_path = orthos / "IMG_0001_0.tif"
    dem_path = base / "dem.tif"
    cam_path = base / "cams.txt"
    poly_path = base / "polys.gpkg"

    _write_multiband_tif(ortho_path)
    _write_singleband_tif(dem_path)
    _write_camera_file(cam_path, "IMG_0001_0")

    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=gpd.GeoSeries.from_wkt(["POLYGON((0 0,10 0,10 10,0 10,0 0))"]),
    )
    gdf.set_crs("EPSG:32632", inplace=True)
    gdf.to_file(poly_path, driver="GPKG")

    cfg = _build_config(base, cam_path, dem_path, orthos, poly_path, main_out, poly_out)
    (base / "gps.csv").write_text("id,lon,lat\n1,0,0\n")
    cfg_path = base / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    # Run extract WITH polygon filtering to produce plot_id column
    proc = subprocess.run(
        ["python", "-m", "main_extract", "--config", str(cfg_path)],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode()

    # Verify extract produced parquet with plot_id
    extract_parquets = list(main_out.glob("*.parquet"))
    assert extract_parquets, "No parquet from extract"

    df_extract = pl.read_parquet(extract_parquets[0])
    assert "plot_id" in df_extract.columns, f"Missing plot_id. Columns: {df_extract.columns}"
    assert len(df_extract) > 0

    # Run filter pipeline
    proc_filter = subprocess.run(
        ["python", "-m", "filtering", "--config", str(cfg_path)],
        capture_output=True,
    )
    assert proc_filter.returncode == 0, proc_filter.stderr.decode()

    # Verify per-polygon output parquets exist
    polygon_parquets = list(poly_out.glob("*.parquet"))
    assert polygon_parquets, f"No polygon parquets in {poly_out}"

    for pq_path in polygon_parquets:
        df = pl.read_parquet(pq_path)
        assert len(df) > 0, f"Empty dataframe in {pq_path.name}"
        assert "plot_id" in df.columns
        assert "OSAVI" in df.columns
        assert "ExcessGreen" in df.columns

        expected_osavi = 1.16 * (df["band5"] - df["band3"]) / (
            df["band5"] + df["band3"] + 0.16
        )
        expected_excess_green = 2 * df["band2"] - df["band3"] - df["band1"]
        np.testing.assert_allclose(df["OSAVI"], expected_osavi)
        np.testing.assert_allclose(df["ExcessGreen"], expected_excess_green)

        plot_ids = df["plot_id"].unique().to_list()
        assert len(plot_ids) == 1, f"Multiple plot_ids in single file: {plot_ids}"

        for col in ["band1", "band2", "band3", "band4", "band5"]:
            if col in df.columns:
                vals = df[col].to_numpy()
                finite_vals = vals[np.isfinite(vals)]
                if len(finite_vals) > 0:
                    assert finite_vals.min() >= 0, f"{col} has negative values"
                    assert finite_vals.max() <= 1.0, f"{col} > 1.0"
