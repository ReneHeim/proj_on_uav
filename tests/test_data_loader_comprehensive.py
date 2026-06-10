"""
Comprehensive unit tests for the data_loader module.
Tests polygon-based data loading and splitting functionality.
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import pytest

from src.filter.data_loader import files_without_plot_id, load_by_polygon


class TestDataLoader:
    """Test data loading functionality."""

    def create_test_parquet(self, path):
        """Helper function to create test parquet file with plot_id column."""
        data = []

        for i in range(10):
            data.append(
                {
                    "Xw": 5.0 + np.random.randn() * 2,
                    "Yw": 5.0 + np.random.randn() * 2,
                    "band1": np.random.rand(),
                    "band2": np.random.rand(),
                    "band3": np.random.rand(),
                    "elev": 100.0 + np.random.randn() * 5,
                    "plot_id": "plot_1",
                }
            )

        for i in range(8):
            data.append(
                {
                    "Xw": 15.0 + np.random.randn() * 2,
                    "Yw": 5.0 + np.random.randn() * 2,
                    "band1": np.random.rand(),
                    "band2": np.random.rand(),
                    "band3": np.random.rand(),
                    "elev": 100.0 + np.random.randn() * 5,
                    "plot_id": "plot_2",
                }
            )

        for i in range(12):
            data.append(
                {
                    "Xw": 5.0 + np.random.randn() * 2,
                    "Yw": 15.0 + np.random.randn() * 2,
                    "band1": np.random.rand(),
                    "band2": np.random.rand(),
                    "band3": np.random.rand(),
                    "elev": 100.0 + np.random.randn() * 5,
                    "plot_id": "plot_3",
                }
            )

        for i in range(5):
            data.append(
                {
                    "Xw": 25.0 + np.random.randn() * 2,
                    "Yw": 25.0 + np.random.randn() * 2,
                    "band1": np.random.rand(),
                    "band2": np.random.rand(),
                    "band3": np.random.rand(),
                    "elev": 100.0 + np.random.randn() * 5,
                    "plot_id": "outside",
                }
            )

        df = pl.DataFrame(data)
        df.write_parquet(path)

    def _verify_output_file(self, filepath, expected_plot_id):
        """Verify a single output parquet file."""
        df = pl.read_parquet(filepath)
        assert len(df) > 0, f"File {filepath} is empty"
        assert "plot_id" in df.columns
        assert all(df["plot_id"] == expected_plot_id)
        return df

    def test_load_by_polygon_basic(self):
        """Test basic polygon-based data loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()
            parquet_path = data_dir / "test_data_basic.parquet"
            output_dir = Path(tmp_dir) / "output_basic"

            self.create_test_parquet(parquet_path)

            result = load_by_polygon(str(data_dir), str(output_dir))
            assert result == "done"
            assert output_dir.exists()

            output_files = sorted(output_dir.glob("*.parquet"))
            expected_ids = {"plot_1", "plot_2", "plot_3", "outside"}
            assert len(output_files) == len(expected_ids)

            found_ids = set()
            for f in output_files:
                df = self._verify_output_file(f, f.stem)
                found_ids.add(f.stem)
            assert found_ids == expected_ids

    def test_load_by_polygon_empty_polygons(self):
        """Test loading with parquet that has no plot_id column (no polygons found)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()
            parquet_path = data_dir / "test_data_empty.parquet"
            output_dir = Path(tmp_dir) / "output_empty"

            df = pl.DataFrame(
                {
                    "Xw": [5.0, 15.0],
                    "Yw": [5.0, 5.0],
                    "band1": [0.1, 0.2],
                    "elev": [100.0, 101.0],
                }
            )
            df.write_parquet(parquet_path)

            result = load_by_polygon(str(data_dir), str(output_dir))
            assert result == "done"

            output_files = list(output_dir.glob("*.parquet"))
            assert len(output_files) == 0

    def test_load_by_polygon_no_overlap(self):
        """Test loading when data has plot_id that matches no other data groups."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()
            parquet_path = data_dir / "test_data_no_overlap.parquet"
            output_dir = Path(tmp_dir) / "output_no_overlap"

            df = pl.DataFrame(
                {
                    "Xw": [25.0] * 5,
                    "Yw": [25.0] * 5,
                    "band1": np.random.rand(5),
                    "band2": np.random.rand(5),
                    "band3": np.random.rand(5),
                    "elev": np.random.randn(5) * 5 + 100,
                    "plot_id": ["far_plot"] * 5,
                }
            )
            df.write_parquet(parquet_path)

            result = load_by_polygon(str(data_dir), str(output_dir))
            assert result == "done"

            output_files = list(output_dir.glob("*.parquet"))
            assert len(output_files) == 1
            far_file = output_files[0]
            assert far_file.stem == "far_plot"
            result_df = pl.read_parquet(far_file)
            assert len(result_df) == 5
            assert all(result_df["plot_id"] == "far_plot")

    def test_load_by_polygon_single_polygon(self):
        """Test loading with single polygon/plot_id value."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()
            parquet_path = data_dir / "test_data_single.parquet"
            output_dir = Path(tmp_dir) / "output_single"

            df = pl.DataFrame(
                {
                    "Xw": [5.0] * 10,
                    "Yw": [5.0] * 10,
                    "band1": np.random.rand(10),
                    "band2": np.random.rand(10),
                    "band3": np.random.rand(10),
                    "elev": np.random.randn(10) * 5 + 100,
                    "plot_id": ["single_plot"] * 10,
                }
            )
            df.write_parquet(parquet_path)

            result = load_by_polygon(str(data_dir), str(output_dir))
            assert result == "done"

            output_files = list(output_dir.glob("*.parquet"))
            assert len(output_files) == 1
            assert output_files[0].stem == "single_plot"

            result_df = pl.read_parquet(output_files[0])
            assert len(result_df) == 10

    def test_load_by_polygon_preserves_columns(self):
        """Test that loading preserves all original columns."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()
            parquet_path = data_dir / "test_data_columns.parquet"
            output_dir = Path(tmp_dir) / "output_columns"

            self.create_test_parquet(parquet_path)

            result = load_by_polygon(str(data_dir), str(output_dir))
            assert result == "done"

            expected_columns = {"Xw", "Yw", "band1", "band2", "band3", "elev", "plot_id"}
            for f in output_dir.glob("*.parquet"):
                df = pl.read_parquet(f)
                assert set(df.columns) == expected_columns, (
                    f"Columns mismatch in {f.name}: got {set(df.columns)}"
                )

    def test_load_by_polygon_data_types(self):
        """Test that numeric data types are preserved as Float64."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()
            parquet_path = data_dir / "test_data_types.parquet"
            output_dir = Path(tmp_dir) / "output_types"

            self.create_test_parquet(parquet_path)

            result = load_by_polygon(str(data_dir), str(output_dir))
            assert result == "done"

            numeric_cols = ["Xw", "Yw", "band1", "band2", "band3", "elev"]
            for f in output_dir.glob("*.parquet"):
                df = pl.read_parquet(f)
                for col in numeric_cols:
                    assert df[col].dtype == pl.Float64, (
                        f"Column {col} in {f.name} has dtype {df[col].dtype}"
                    )
                assert df["plot_id"].dtype == pl.String

    def test_load_by_polygon_coordinate_ranges(self):
        """Test that coordinates are within expected ranges per polygon."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()
            parquet_path = data_dir / "test_data_ranges.parquet"
            output_dir = Path(tmp_dir) / "output_ranges"

            self.create_test_parquet(parquet_path)

            result = load_by_polygon(str(data_dir), str(output_dir))
            assert result == "done"

            for f in output_dir.glob("*.parquet"):
                df = pl.read_parquet(f)
                plot_id = f.stem

                # Verify coordinates are finite and within reasonable bounds for the polygon region
                assert df["Xw"].is_finite().all()
                assert df["Yw"].is_finite().all()
                assert df["Xw"].min() >= -10.0 and df["Xw"].max() <= 30.0
                assert df["Yw"].min() >= -10.0 and df["Yw"].max() <= 30.0

                if plot_id == "plot_2":
                    # plot_2's Xw should be shifted right compared to plot_1/3
                    assert df["Xw"].mean() > 10.0
                elif plot_id == "plot_3":
                    # plot_3's Yw should be shifted up compared to plot_1/2
                    assert df["Yw"].mean() > 10.0
                elif plot_id == "outside":
                    assert df["Xw"].min() >= 20.0
                    assert df["Yw"].min() >= 20.0

    def test_load_by_polygon_specific(self):
        """Test loading with specific polygon selection."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()
            parquet_path = data_dir / "test_data_specific.parquet"
            output_dir = Path(tmp_dir) / "output_specific"

            self.create_test_parquet(parquet_path)

            result = load_by_polygon(str(data_dir), str(output_dir), specific="plot_1")
            assert result == "done"

            output_files = list(output_dir.glob("*.parquet"))
            assert len(output_files) == 1
            assert output_files[0].stem == "plot_1"

            df = pl.read_parquet(output_files[0])
            assert len(df) > 0
            assert all(df["plot_id"] == "plot_1")

    def test_files_without_plot_id_direct(self):
        """Test files_without_plot_id returns files missing plot_id column."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()

            df_good = pl.DataFrame(
                {"Xw": [1.0, 2.0], "Yw": [1.0, 2.0], "plot_id": ["a", "b"]}
            )
            df_good.write_parquet(data_dir / "good.parquet")

            df_bad = pl.DataFrame(
                {"Xw": [3.0, 4.0], "Yw": [3.0, 4.0], "elev": [100.0, 101.0]}
            )
            df_bad.write_parquet(data_dir / "bad.parquet")

            missing = files_without_plot_id(data_dir)
            assert len(missing) == 1
            assert missing[0].name == "bad.parquet"


class TestErrorHandling:
    """Test error handling in data loading."""

    def create_test_parquet(self, path):
        """Helper function to create test parquet file with plot_id."""
        df = pl.DataFrame(
            {
                "Xw": [5.0, 15.0, 5.0],
                "Yw": [5.0, 5.0, 15.0],
                "band1": [0.1, 0.2, 0.3],
                "band2": [0.2, 0.3, 0.4],
                "band3": [0.3, 0.4, 0.5],
                "elev": [100.0, 101.0, 102.0],
                "plot_id": ["p1", "p2", "p3"],
            }
        )
        df.write_parquet(path)

    def test_load_by_polygon_nonexistent_df_folder(self):
        """Test loading with nonexistent df_folder handles gracefully."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "output_error"
            nonexistent = Path(tmp_dir) / "nonexistent_folder"

            result = load_by_polygon(str(nonexistent), str(output_dir))
            assert result == "done"

            output_files = list(output_dir.glob("*.parquet"))
            assert len(output_files) == 0

    def test_load_by_polygon_nonexistent_polygon_specific(self):
        """Test loading with specific polygon that doesn't exist in data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()
            parquet_path = data_dir / "test_data.parquet"
            output_dir = Path(tmp_dir) / "output_error"
            self.create_test_parquet(parquet_path)

            result = load_by_polygon(str(data_dir), str(output_dir), specific="nonexistent")
            assert result == "done"

            output_files = list(output_dir.glob("*.parquet"))
            assert len(output_files) == 1
            assert output_files[0].stem == "nonexistent"

    def test_load_by_polygon_invalid_format(self):
        """Test loading with corrupted parquet file in df_folder raises error."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()
            corrupted = data_dir / "corrupted.parquet"
            corrupted.write_bytes(b"not a valid parquet")
            output_dir = Path(tmp_dir) / "output_invalid"

            with pytest.raises(Exception):
                load_by_polygon(str(data_dir), str(output_dir))


class TestPerformance:
    """Test performance aspects of data loading."""

    def create_test_polygons(self, path):
        """Helper function to create test polygon file."""
        polygons = gpd.GeoDataFrame(
            {
                "id": [1, 2, 3],
                "plot_id": ["plot_1", "plot_2", "plot_3"],
                "geometry": gpd.GeoSeries.from_wkt(
                    [
                        "POLYGON((1 1, 9 1, 9 9, 1 9, 1 1))",
                        "POLYGON((11 1, 19 1, 19 9, 11 9, 11 1))",
                        "POLYGON((1 11, 9 11, 9 19, 1 19, 1 11))",
                    ]
                ),
            }
        )
        polygons.set_crs("EPSG:32632", inplace=True)
        polygons.to_file(path, driver="GPKG")

    def test_load_by_polygon_large_dataset(self):
        """Test loading with large dataset."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            data_dir.mkdir()
            parquet_path = data_dir / "test_large_data.parquet"
            output_dir = Path(tmp_dir) / "output_perf"

            n_points = 10000
            data = []
            for i in range(n_points):
                data.append(
                    {
                        "Xw": np.random.uniform(0, 20),
                        "Yw": np.random.uniform(0, 20),
                        "band1": np.random.rand(),
                        "band2": np.random.rand(),
                        "band3": np.random.rand(),
                        "elev": 100.0 + np.random.randn() * 5,
                        "plot_id": f"plot_{np.random.randint(1, 4)}",
                    }
                )

            df = pl.DataFrame(data)
            df.write_parquet(parquet_path)

            result = load_by_polygon(str(data_dir), str(output_dir))
            assert result == "done"

            output_files = list(output_dir.glob("*.parquet"))
            assert len(output_files) >= 1


if __name__ == "__main__":
    pytest.main([__file__])
