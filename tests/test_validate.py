import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.core.validate import validate_extract_output


class TestValidateExtractOutput:
    def test_clean_data_passes(self):
        """Clean data with all required columns passes validation."""
        df = pl.DataFrame(
            {
                "Xw": [1.0, 2.0],
                "Yw": [1.0, 2.0],
                "band1": [0.1, 0.2],
                "band2": [0.2, 0.3],
                "band3": [0.3, 0.4],
                "band4": [0.4, 0.5],
                "band5": [0.5, 0.6],
                "elev": [100.0, 101.0],
                "delta_z": [50.0, 51.0],
                "delta_x": [1.0, 2.0],
                "delta_y": [1.0, 2.0],
                "distance_xy": [1.4, 2.8],
                "angle_rad": [0.5, 0.6],
                "vza": [30.0, 45.0],
                "vaa_rad": [1.0, 2.0],
                "vaa_temp": [1.0, 2.0],
                "vaa": [90.0, 180.0],
                "xcam": [0.0, 0.0],
                "ycam": [0.0, 0.0],
                "sunelev": [40.0, 40.0],
                "saa": [180.0, 180.0],
                "path": ["/a.tif", "/b.tif"],
            }
        )
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            df.write_parquet(out / "test1.parquet")
            df.write_parquet(out / "test2.parquet")
            result = validate_extract_output(out)
            assert result["ok"]
            assert result["n_files"] == 2
            assert result["n_corrupt"] == 0
            assert result["schema_issues"] == []

    def test_missing_required_columns_fails(self):
        """Missing required column causes validation failure."""
        df = pl.DataFrame(
            {
                "Xw": [1.0],
                "Yw": [1.0],
                "band1": [0.1],
                "band2": [0.2],
                "band3": [0.3],
                "band4": [0.4],
                "band5": [0.5],
                "elev": [100.0],
                "delta_z": [50.0],
                "delta_x": [1.0],
                "delta_y": [1.0],
                "distance_xy": [1.0],
                "angle_rad": [0.5],
                "vza": [30.0],
                "vaa_rad": [1.0],
                "vaa_temp": [1.0],
                "vaa": [90.0],
                "xcam": [0.0],
                "ycam": [0.0],
                "sunelev": [40.0],
                "saa": [180.0],
                # missing "path"
            }
        )
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            df.write_parquet(out / "test1.parquet")
            result = validate_extract_output(out)
            assert not result["ok"]
            assert "path" in result["missing_columns"]

    def test_vza_out_of_range_fails(self):
        """VZA > 90 deg should fail validation."""
        df = pl.DataFrame(
            {
                "Xw": [1.0],
                "Yw": [1.0],
                "band1": [0.1],
                "band2": [0.2],
                "band3": [0.3],
                "band4": [0.4],
                "band5": [0.5],
                "elev": [100.0],
                "delta_z": [50.0],
                "delta_x": [1.0],
                "delta_y": [1.0],
                "distance_xy": [1.0],
                "angle_rad": [0.5],
                "vza": [110.0],
                "vaa_rad": [1.0],
                "vaa_temp": [1.0],
                "vaa": [90.0],
                "xcam": [0.0],
                "ycam": [0.0],
                "sunelev": [40.0],
                "saa": [180.0],
                "path": ["/a.tif"],
            }
        )
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            df.write_parquet(out / "test1.parquet")
            result = validate_extract_output(out)
            assert not result["ok"]
            assert len(result["range_issues"]) > 0

    def test_inconsistent_schemas_fails(self):
        """Two files with different columns should fail."""
        df1 = pl.DataFrame(
            {
                "Xw": [1.0],
                "Yw": [1.0],
                "band1": [0.1],
                "band2": [0.2],
                "band3": [0.3],
                "band4": [0.4],
                "band5": [0.5],
                "elev": [100.0],
                "delta_z": [50.0],
                "delta_x": [1.0],
                "delta_y": [1.0],
                "distance_xy": [1.0],
                "angle_rad": [0.5],
                "vza": [30.0],
                "vaa_rad": [1.0],
                "vaa_temp": [1.0],
                "vaa": [90.0],
                "xcam": [0.0],
                "ycam": [0.0],
                "sunelev": [40.0],
                "saa": [180.0],
                "path": ["/a.tif"],
            }
        )
        df2 = df1.with_columns(pl.lit(1).alias("extra_col"))
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            df1.write_parquet(out / "test1.parquet")
            df2.write_parquet(out / "test2.parquet")
            result = validate_extract_output(out)
            assert not result["ok"]
            assert len(result["schema_issues"]) > 0

    def test_empty_dir_fails(self):
        """Empty directory should fail validation."""
        with tempfile.TemporaryDirectory() as d:
            result = validate_extract_output(Path(d))
            assert not result["ok"]

    def test_optional_plot_id_accepted(self):
        """plot_id column should be accepted as optional."""
        df = pl.DataFrame(
            {
                "Xw": [1.0],
                "Yw": [1.0],
                "band1": [0.1],
                "band2": [0.2],
                "band3": [0.3],
                "band4": [0.4],
                "band5": [0.5],
                "elev": [100.0],
                "delta_z": [50.0],
                "delta_x": [1.0],
                "delta_y": [1.0],
                "distance_xy": [1.0],
                "angle_rad": [0.5],
                "vza": [30.0],
                "vaa_rad": [1.0],
                "vaa_temp": [1.0],
                "vaa": [90.0],
                "xcam": [0.0],
                "ycam": [0.0],
                "sunelev": [40.0],
                "saa": [180.0],
                "path": ["/a.tif"],
                "plot_id": [1],
            }
        )
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            df.write_parquet(out / "test1.parquet")
            result = validate_extract_output(out)
            assert result["ok"]
            assert result["files_without_plot_id"] == []
