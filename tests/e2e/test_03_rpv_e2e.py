import os
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import yaml
from shapely.geometry import Polygon


def _make_config(base_dir, polygon_path, bands=5):
    return {
        "base_path": str(base_dir),
        "inputs": {
            "date_time": {"start": "2024-01-01 12:00:00", "time_zone": "UTC"},
            "paths": {
                "cam_path": str(base_dir / "dummy_cams.txt"),
                "dem_path": str(base_dir / "dummy_dem.tif"),
                "orthophoto_path": str(base_dir / "dummy_*.tif"),
                "ori": [str(base_dir)],
                "mosaic_path": str(base_dir / "dummy_mosaic.tif"),
                "ground_truth_coordinates": str(base_dir / "dummy_gps.csv"),
                "polygon_file_path": str(polygon_path),
            },
            "settings": {
                "number_of_processor": 1,
                "filter_radius": 1,
                "file_name": "e2e_rpv",
                "bands": bands,
                "target_crs": "EPSG:32632",
            },
        },
        "outputs": {
            "paths": {
                "main_out": str(base_dir / "dummy_out"),
                "plot_out": str(base_dir / "dummy_plots"),
            }
        },
    }


def _create_synthetic_rpv_data(base_dir, n_points=600):
    np.random.seed(42)

    xcam = 250.0
    ycam = 250.0
    delta_z = 100.0
    sunelev = 60.0
    saa = 180.0
    elev = 50.0

    Xw = np.random.uniform(0, 500, n_points)
    Yw = np.random.uniform(0, 500, n_points)

    dx = Xw - xcam
    dy = Yw - ycam
    vx_val = xcam - Xw
    vy_val = ycam - Yw
    vz_val = np.full(n_points, delta_z)
    v_norm = np.sqrt(vx_val**2 + vy_val**2 + vz_val**2)

    cos_vza = np.clip(vz_val / (v_norm + 1e-12), -1.0, 1.0)
    vza = np.degrees(np.arccos(cos_vza))
    vaa = np.degrees(np.arctan2(vx_val, vy_val)) % 360
    sza = 90.0 - sunelev
    raa = np.abs(saa - vaa) % 360
    raa = np.where(raa <= 180, raa, 360 - raa)

    from src.modelling.rpv import rpv_2

    rho0_true, k_true, theta_true = 0.3, 0.7, 0.1
    angle_pack = (np.full(n_points, sza), vza, raa)
    R = rpv_2(angle_pack, rho0_true, k_true, theta_true)
    noise = np.random.normal(0, 0.015, n_points)
    band1 = np.clip(R + noise, 0.001, 0.999)

    df = pl.DataFrame(
        {
            "Xw": Xw,
            "Yw": Yw,
            "xcam": np.full(n_points, xcam, dtype=np.float32),
            "ycam": np.full(n_points, ycam, dtype=np.float32),
            "delta_z": vz_val.astype(np.float32),
            "elev": np.full(n_points, elev, dtype=np.float32),
            "sunelev": np.full(n_points, sunelev, dtype=np.float32),
            "saa": np.full(n_points, saa, dtype=np.float32),
            "vx": vx_val.astype(np.float32),
            "vy": vy_val.astype(np.float32),
            "vz": vz_val.astype(np.float32),
            "v_norm": v_norm.astype(np.float32),
            "vza": vza.astype(np.float32),
            "vaa": vaa.astype(np.float32),
            "sza": np.full(n_points, sza, dtype=np.float32),
            "raa": raa.astype(np.float32),
            "band1": band1.astype(np.float32),
            "band2": np.random.uniform(0.05, 0.10, n_points).astype(np.float32),
            "band3": np.random.uniform(0.02, 0.06, n_points).astype(np.float32),
            "band4": np.random.uniform(0.01, 0.05, n_points).astype(np.float32),
            "band5": np.random.uniform(0.10, 0.30, n_points).astype(np.float32),
        }
    )
    return df


def test_script_03_rpv_e2e():
    proc = subprocess.run(["python", "-m", "rpv_modelling", "--help"], capture_output=True)
    assert proc.returncode == 0


def test_rpv_pipeline(tmp_path: Path):
    base_dir = tmp_path / "rpv_data"
    base_dir.mkdir()

    data_dir = base_dir / "week1" / "polygon_df"
    data_dir.mkdir(parents=True)

    poly_path = base_dir / "polys.gpkg"
    gdf_poly = gpd.GeoDataFrame(
        {"id": [1], "ifz_id": [1], "cult": ["wheat"], "trt": ["control"]},
        geometry=[Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])],
        crs="EPSG:32632",
    )
    gdf_poly.to_file(poly_path, driver="GPKG")

    df = _create_synthetic_rpv_data(base_dir, n_points=600)
    parquet_path = data_dir / "plot_1.parquet"
    df.write_parquet(parquet_path)

    dummy_files = [
        "dummy_cams.txt",
        "dummy_dem.tif",
        "dummy_gps.csv",
        "dummy_mosaic.tif",
        "dummy_ortho.tif",
    ]
    for fname in dummy_files:
        (base_dir / fname).touch()
    (base_dir / "dummy_*.tif").touch()
    (base_dir / "dummy_out").mkdir(exist_ok=True)
    (base_dir / "dummy_plots").mkdir(exist_ok=True)

    cfg = _make_config(base_dir, poly_path, bands=5)
    cfg_path = base_dir / "cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    proc = subprocess.run(
        [
            "python",
            "-m",
            "rpv_modelling",
            "--config",
            str(cfg_path),
            "--base-dir",
            str(base_dir),
            "--band",
            "band1",
        ],
        capture_output=True,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr.decode()}"

    results_dir = base_dir / "RPV_Results" / "V12" / "week1"
    csv_path = results_dir / "rpv_week1_band1_results.csv"
    assert csv_path.exists(), f"Expected CSV not found: {csv_path}"

    import polars as pl

    results = pl.read_csv(csv_path)
    assert len(results) > 0

    rho0 = results["rho0"].to_list()[0]
    k = results["k"].to_list()[0]
    theta = results["theta"].to_list()[0]
    nrmse = results["nrmse"].to_list()[0]

    assert 0.01 < rho0 < 2.0, f"rho0 out of range: {rho0}"
    assert 0.0 <= k <= 3.0, f"k out of range: {k}"
    assert -1.0 <= theta <= 1.0, f"theta out of range: {theta}"
    assert 0.0 <= nrmse <= 1.0, f"nrmse out of range: {nrmse}"
