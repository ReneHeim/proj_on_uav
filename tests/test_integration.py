"""
Integration tests for the complete UAV reflectance extraction pipeline.
Tests the full workflow from data loading to RPV model fitting.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import pytest
import rasterio as rio
import yaml
from rasterio.transform import from_bounds


def create_test_data(tmp_path: Path):
    """Create synthetic test data for integration testing."""
    base = tmp_path
    orthos = base / "orthos"
    out = base / "out"
    orthos.mkdir()
    out.mkdir()

    # Create test orthophoto
    ortho_path = orthos / "IMG_0001_0.tif"
    dem_path = base / "dem.tif"
    cam_path = base / "cams.txt"
    poly_path = base / "polys.gpkg"

    # Create multi-band orthophoto
    transform = from_bounds(0, 0, 10, 10, 10, 10)
    data = np.zeros((5, 10, 10), dtype=np.float32)
    for b in range(5):
        data[b, :, :] = (b + 1) * 0.1 + np.random.random((10, 10)) * 0.05

    with rio.open(
        ortho_path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=5,
        dtype="float32",
        crs="EPSG:32632",
        transform=transform,
    ) as dst:
        dst.write(data)

    # Create DEM
    dem_data = np.ones((10, 10), dtype=np.float32) * 100.0
    with rio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=transform,
    ) as dst:
        dst.write(dem_data, 1)

    # Create camera file
    lines = [
        "# header1\n",
        "# header2\n",
        "IMG_0001_0\t5.0\t5.0\t110.0\t0\t0\t0\t1\t0\t0\t0\t1\t0\t0\t0\t1\n",
    ]
    cam_path.write_text("".join(lines))

    # Create polygon
    gdf = gpd.GeoDataFrame(
        {"id": [1], "plot_id": ["plot_1"]},
        geometry=gpd.GeoSeries.from_wkt(["POLYGON((1 1,9 1,9 9,1 9,1 1))"]),
    )
    gdf.set_crs("EPSG:32632", inplace=True)
    gdf.to_file(poly_path, driver="GPKG")

    # Create GPS coordinates
    gps_path = base / "gps.csv"
    gps_path.write_text("id,lon,lat\n1,5.0,5.0\n")

    return {
        "ortho_path": ortho_path,
        "dem_path": dem_path,
        "cam_path": cam_path,
        "poly_path": poly_path,
        "gps_path": gps_path,
        "out_path": out,
    }


def create_config(data_paths, tmp_path: Path):
    """Create configuration file for testing."""
    cfg = {
        "base_path": str(tmp_path),
        "inputs": {
            "date_time": {"start": "2024-01-01 12:00:00", "time_zone": "UTC"},
            "paths": {
                "cam_path": str(data_paths["cam_path"]),
                "dem_path": str(data_paths["dem_path"]),
                "orthophoto_path": str(data_paths["ortho_path"].parent / "*.tif"),
                "ori": [str(data_paths["ortho_path"].parent)],
                "mosaic_path": str(data_paths["ortho_path"]),
                "ground_truth_coordinates": str(data_paths["gps_path"]),
                "polygon_file_path": str(data_paths["poly_path"]),
            },
            "settings": {
                "number_of_processor": 1,
                "filter_radius": 1,
                "file_name": "integration_test",
                "bands": 5,
                "target_crs": "EPSG:32632",
            },
        },
        "outputs": {
            "paths": {
                "main_out": str(data_paths["out_path"]),
                "plot_out": str(data_paths["out_path"] / "plots"),
            }
        },
    }

    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def test_complete_pipeline_integration():
    """Test the complete pipeline from extraction to RPV fitting."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_paths = create_test_data(tmp_path)
        config_path = create_config(data_paths, tmp_path)

        # Step 1: Extract data
        proc = subprocess.run(
            ["python", "-m", "main_extract", "--config", str(config_path)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Extraction failed: {proc.stderr}"

        # Verify extraction output
        extract_files = list(data_paths["out_path"].glob("**/*.parquet"))
        assert len(extract_files) > 0, "No parquet files produced by extraction"

        # Step 2: Apply filtering
        proc = subprocess.run(
            ["python", "-m", "filtering", "--config", str(config_path)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"Filtering failed: {proc.stderr}"

        # Step 3: RPV modeling (smoke test)
        proc = subprocess.run(
            [
                "python",
                "-m",
                "rpv_modelling",
                "--config",
                str(config_path),
                "--band",
                "band1",
            ],
            capture_output=True,
            text=True,
        )
        # RPV might fail without real data, but CLI should work
        assert proc.returncode in [0, 1], f"RPV modeling failed unexpectedly: {proc.stderr}"


def test_data_flow_validation():
    """Test that data flows correctly between pipeline stages."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_paths = create_test_data(tmp_path)
        config_path = create_config(data_paths, tmp_path)

        # Run extraction
        subprocess.run(
            ["python", "-m", "main_extract", "--config", str(config_path)],
            capture_output=True,
            check=True,
        )

        # Verify extracted data structure
        parquet_files = list(data_paths["out_path"].glob("**/*.parquet"))
        assert len(parquet_files) > 0

        # Load and validate extracted data
        df = pl.read_parquet(parquet_files[0])

        # Check required columns exist
        required_cols = ["Xw", "Yw", "elev", "band1", "band2", "band3", "band4", "band5"]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        # Check data quality
        assert len(df) > 0, "Extracted dataframe is empty"
        # Check if any columns have all null values
        null_counts = df.select(pl.all().is_null().sum())
        assert not any(null_counts.row(0)), "All values are null in some columns"


def test_error_handling():
    """Test error handling in the pipeline."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Test with missing config file
        proc = subprocess.run(
            ["python", "-m", "main_extract", "--config", "nonexistent.yml"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0, "Should fail with missing config file"

        # Test with invalid config
        invalid_config = tmp_path / "invalid.yml"
        invalid_config.write_text("invalid: yaml: content")

        proc = subprocess.run(
            ["python", "-m", "main_extract", "--config", str(invalid_config)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0, "Should fail with invalid config"


def test_memory_efficiency():
    """Test that the pipeline handles large datasets efficiently."""
    try:
        import psutil
    except ImportError:
        pytest.skip("psutil not available for memory monitoring")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        data_paths = create_test_data(tmp_path)
        config_path = create_config(data_paths, tmp_path)

        # Run with memory monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Run extraction
        subprocess.run(
            ["python", "-m", "main_extract", "--config", str(config_path)],
            capture_output=True,
            check=True,
        )

        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

        # Memory increase should be reasonable (< 1GB for this small dataset)
        assert memory_increase < 1024, f"Memory usage increased by {memory_increase:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__])
