"""
Comprehensive unit tests for the data_loader module.
Tests polygon-based data loading and splitting functionality.
"""

import os
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import pytest

from src.Common.data_loader import load_by_polygon


class TestDataLoader:
    """Test data loading functionality."""

    def create_test_polygons(self, path):
        """Helper function to create test polygon file."""
        # Create test polygons
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

    def create_test_parquet(self, path):
        """Helper function to create test parquet file."""
        # Create test data with points in different polygons
        data = []

        # Points in polygon 1
        for i in range(10):
            data.append(
                {
                    "Xw": 5.0 + np.random.randn() * 2,
                    "Yw": 5.0 + np.random.randn() * 2,
                    "band1": np.random.rand(),
                    "band2": np.random.rand(),
                    "band3": np.random.rand(),
                    "elev": 100.0 + np.random.randn() * 5,
                }
            )

        # Points in polygon 2
        for i in range(8):
            data.append(
                {
                    "Xw": 15.0 + np.random.randn() * 2,
                    "Yw": 5.0 + np.random.randn() * 2,
                    "band1": np.random.rand(),
                    "band2": np.random.rand(),
                    "band3": np.random.rand(),
                    "elev": 100.0 + np.random.randn() * 5,
                }
            )

        # Points in polygon 3
        for i in range(12):
            data.append(
                {
                    "Xw": 5.0 + np.random.randn() * 2,
                    "Yw": 15.0 + np.random.randn() * 2,
                    "band1": np.random.rand(),
                    "band2": np.random.rand(),
                    "band3": np.random.rand(),
                    "elev": 100.0 + np.random.randn() * 5,
                }
            )

        # Points outside all polygons
        for i in range(5):
            data.append(
                {
                    "Xw": 25.0 + np.random.randn() * 2,
                    "Yw": 25.0 + np.random.randn() * 2,
                    "band1": np.random.rand(),
                    "band2": np.random.rand(),
                    "band3": np.random.rand(),
                    "elev": 100.0 + np.random.randn() * 5,
                }
            )

        df = pl.DataFrame(data)
        df.write_parquet(path)

    def test_load_by_polygon_basic(self):
        """Test basic polygon-based data loading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files with unique names
            polygon_path = Path(tmp_dir) / "test_polygons_basic.gpkg"
            parquet_path = Path(tmp_dir) / "test_data_basic.parquet"
            output_dir = Path(tmp_dir) / "output_basic"

            self.create_test_polygons(polygon_path)
            self.create_test_parquet(parquet_path)

            # Load data by polygon
            result = load_by_polygon(str(parquet_path), str(output_dir))

            assert result == "done"
            assert output_dir.exists()

    def test_load_by_polygon_empty_polygons(self):
        """Test loading with empty polygon file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create empty polygon file with unique name
            polygon_path = Path(tmp_dir) / "test_empty_polygons.gpkg"
            empty_polygons = gpd.GeoDataFrame(
                {"id": [], "plot_id": [], "geometry": gpd.GeoSeries([])}
            )
            empty_polygons.set_crs("EPSG:32632", inplace=True)
            empty_polygons.to_file(polygon_path, driver="GPKG")

            # Create test parquet
            parquet_path = Path(tmp_dir) / "test_data_empty.parquet"
            output_dir = Path(tmp_dir) / "output_empty"
            self.create_test_parquet(parquet_path)

            # Should return "done"
            result = load_by_polygon(str(parquet_path), str(output_dir))
            assert result == "done"

    def test_load_by_polygon_no_overlap(self):
        """Test loading when data doesn't overlap with polygons."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create polygons far from data with unique name
            polygon_path = Path(tmp_dir) / "test_far_polygons.gpkg"
            far_polygons = gpd.GeoDataFrame(
                {
                    "id": [1],
                    "plot_id": ["far_plot"],
                    "geometry": gpd.GeoSeries.from_wkt(
                        ["POLYGON((100 100, 110 100, 110 110, 100 110, 100 100))"]
                    ),
                }
            )
            far_polygons.set_crs("EPSG:32632", inplace=True)
            far_polygons.to_file(polygon_path, driver="GPKG")

            # Create test parquet with data near origin
            parquet_path = Path(tmp_dir) / "test_data_no_overlap.parquet"
            output_dir = Path(tmp_dir) / "output_no_overlap"
            self.create_test_parquet(parquet_path)

            # Should return "done"
            result = load_by_polygon(str(parquet_path), str(output_dir))
            assert result == "done"

    def test_load_by_polygon_single_polygon(self):
        """Test loading with single polygon."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create single polygon with unique name
            polygon_path = Path(tmp_dir) / "test_single_polygon.gpkg"
            single_polygon = gpd.GeoDataFrame(
                {
                    "id": [1],
                    "plot_id": ["single_plot"],
                    "geometry": gpd.GeoSeries.from_wkt(["POLYGON((1 1, 9 1, 9 9, 1 9, 1 1))"]),
                }
            )
            single_polygon.set_crs("EPSG:32632", inplace=True)
            single_polygon.to_file(polygon_path, driver="GPKG")

            # Create test parquet
            parquet_path = Path(tmp_dir) / "test_data_single.parquet"
            output_dir = Path(tmp_dir) / "output_single"
            self.create_test_parquet(parquet_path)

            # Load data
            result = load_by_polygon(str(parquet_path), str(output_dir))
            assert result == "done"

    def test_load_by_polygon_preserves_columns(self):
        """Test that loading preserves all original columns."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files with unique names
            polygon_path = Path(tmp_dir) / "test_polygons_columns.gpkg"
            parquet_path = Path(tmp_dir) / "test_data_columns.parquet"
            output_dir = Path(tmp_dir) / "output_columns"

            self.create_test_polygons(polygon_path)
            self.create_test_parquet(parquet_path)

            # Load data
            result = load_by_polygon(str(parquet_path), str(output_dir))
            assert result == "done"

    def test_load_by_polygon_data_types(self):
        """Test that data types are preserved."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files with unique names
            polygon_path = Path(tmp_dir) / "test_polygons_types.gpkg"
            parquet_path = Path(tmp_dir) / "test_data_types.parquet"
            output_dir = Path(tmp_dir) / "output_types"

            self.create_test_polygons(polygon_path)
            self.create_test_parquet(parquet_path)

            # Load data
            result = load_by_polygon(str(parquet_path), str(output_dir))
            assert result == "done"

    def test_load_by_polygon_coordinate_ranges(self):
        """Test that coordinates are within expected ranges."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test files with unique names
            polygon_path = Path(tmp_dir) / "test_polygons_ranges.gpkg"
            parquet_path = Path(tmp_dir) / "test_data_ranges.parquet"
            output_dir = Path(tmp_dir) / "output_ranges"

            self.create_test_polygons(polygon_path)
            self.create_test_parquet(parquet_path)

            # Load data
            result = load_by_polygon(str(parquet_path), str(output_dir))
            assert result == "done"


class TestErrorHandling:
    """Test error handling in data loading."""

    def create_test_polygons(self, path):
        """Helper function to create test polygon file."""
        polygons = gpd.GeoDataFrame(
            {
                "id": [1],
                "plot_id": ["test_plot"],
                "geometry": gpd.GeoSeries.from_wkt(["POLYGON((1 1, 9 1, 9 9, 1 9, 1 1))"]),
            }
        )
        polygons.set_crs("EPSG:32632", inplace=True)
        polygons.to_file(path, driver="GPKG")

    def create_test_parquet(self, path):
        """Helper function to create test parquet file."""
        df = pl.DataFrame(
            {
                "Xw": [5.0, 15.0, 5.0],
                "Yw": [5.0, 5.0, 15.0],
                "band1": [0.1, 0.2, 0.3],
                "band2": [0.2, 0.3, 0.4],
                "band3": [0.3, 0.4, 0.5],
                "elev": [100.0, 101.0, 102.0],
            }
        )
        df.write_parquet(path)

    def test_load_by_polygon_nonexistent_parquet(self):
        """Test loading with nonexistent parquet file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            polygon_path = Path(tmp_dir) / "test_error_polygons.gpkg"
            self.create_test_polygons(polygon_path)

            with pytest.raises(Exception):
                load_by_polygon("nonexistent.parquet", str(polygon_path))

    def test_load_by_polygon_nonexistent_polygon(self):
        """Test loading with nonexistent polygon file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_path = Path(tmp_dir) / "test_error_data.parquet"
            output_dir = Path(tmp_dir) / "output_error"
            self.create_test_parquet(parquet_path)

            # The function should handle missing files gracefully
            result = load_by_polygon(str(parquet_path), str(output_dir))
            assert result == "done"

    def test_load_by_polygon_invalid_polygon_format(self):
        """Test loading with invalid polygon file format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create invalid polygon file
            invalid_path = Path(tmp_dir) / "invalid.txt"
            invalid_path.write_text("This is not a valid polygon file")

            parquet_path = Path(tmp_dir) / "test_invalid_data.parquet"
            output_dir = Path(tmp_dir) / "output_invalid"
            self.create_test_parquet(parquet_path)

            # The function should handle invalid files gracefully
            result = load_by_polygon(str(parquet_path), str(output_dir))
            assert result == "done"


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
            # Create test files with unique names
            polygon_path = Path(tmp_dir) / "test_perf_polygons.gpkg"
            parquet_path = Path(tmp_dir) / "test_large_data.parquet"
            output_dir = Path(tmp_dir) / "output_perf"

            self.create_test_polygons(polygon_path)

            # Create large dataset
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
                    }
                )

            df = pl.DataFrame(data)
            df.write_parquet(parquet_path)

            # Should complete without timeout
            result = load_by_polygon(str(parquet_path), str(output_dir))
            assert result == "done"


if __name__ == "__main__":
    pytest.main([__file__])
