"""
Comprehensive unit tests for the raster module.
Tests coordinate transformations, raster operations, and alignment functions.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import rasterio as rio
from pyproj import Transformer
from rasterio.transform import from_bounds

from src.extract.raster import (
    _auto_figsize,
    _finite_mask,
    check_alignment,
    coregister_and_resample,
    latlon_to_utm32n_series,
    pixelToWorldCoords,
    plotting_raster,
    read_orthophoto_bands,
    to_numpy2,
    worldToPixelCoords,
    xy_np,
    xyval,
)


class TestCoordinateTransformations:
    """Test coordinate transformation functions."""

    def test_pixel_to_world_coords(self):
        """Test pixel to world coordinate transformation."""
        geo_transform = [1000.0, 1.0, 0.0, 2000.0, 0.0, -1.0]

        world_x, world_y = pixelToWorldCoords(10, 20, geo_transform)

        assert world_x == 1010.0
        assert world_y == 1980.0

    def test_world_to_pixel_coords_int(self):
        """Test world to pixel coordinate transformation with integer output."""
        geo_transform = [1000.0, 1.0, 0.0, 2000.0, 0.0, -1.0]

        pixel_x, pixel_y = worldToPixelCoords(1010.0, 1980.0, geo_transform, dtype="int")

        assert pixel_x == 10
        assert pixel_y == 20
        assert isinstance(pixel_x, int)
        assert isinstance(pixel_y, int)

    def test_world_to_pixel_coords_float(self):
        """Test world to pixel coordinate transformation with float output."""
        geo_transform = [1000.0, 1.0, 0.0, 2000.0, 0.0, -1.0]

        pixel_x, pixel_y = worldToPixelCoords(1010.5, 1980.5, geo_transform, dtype="float")

        assert abs(pixel_x - 11.0) < 1e-6
        assert abs(pixel_y - 20.0) < 1e-6
        assert isinstance(pixel_x, float)
        assert isinstance(pixel_y, float)

    def test_xyval_function(self):
        """Test xyval function for raster array."""
        test_array = np.array([[1, 2], [3, 4]])

        x, y, values = xyval(test_array)

        assert len(x) == 4
        assert len(y) == 4
        assert len(values) == 4

        expected_x = [0, 1, 0, 1]
        expected_y = [0, 0, 1, 1]
        expected_values = [1, 2, 3, 4]

        assert set(x) == set(expected_x)
        assert set(y) == set(expected_y)
        assert set(values) == set(expected_values)

    def test_to_numpy2_transform(self):
        """Test transform to numpy array conversion."""
        from affine import Affine

        transform = Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)

        result = to_numpy2(transform)

        assert result.shape == (3, 3)
        assert result[0, 0] == 1.0
        assert result[0, 2] == 100.0
        assert result[1, 1] == -1.0
        assert result[1, 2] == 200.0

    def test_xy_np_center_offset(self):
        """Test xy_np function with center offset."""
        from affine import Affine

        transform = Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)

        x, y = xy_np(transform, 10, 20, offset="center")

        assert len(x) == 1
        assert len(y) == 1
        assert abs(x[0] - 110.5) < 1e-6
        assert abs(y[0] - 179.5) < 1e-6

    def test_xy_np_multiple_points(self):
        """Test xy_np function with multiple points."""
        from affine import Affine

        transform = Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)

        rows = [10, 20, 30]
        cols = [5, 15, 25]

        x, y = xy_np(transform, rows, cols, offset="ul")

        assert len(x) == 3
        assert len(y) == 3

    def test_xy_np_invalid_offset(self):
        """Test xy_np function with invalid offset."""
        from affine import Affine

        transform = Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0)

        with pytest.raises(ValueError, match="Invalid offset"):
            xy_np(transform, 10, 20, offset="invalid")


class TestRasterOperations:
    """Test raster reading and processing operations."""

    def create_test_raster(self, path, width=10, height=10, bands=5, crs="EPSG:32632"):
        """Helper function to create test raster files."""
        transform = from_bounds(0, 0, width, height, width, height)

        data = np.zeros((bands, height, width), dtype=np.float32)
        for b in range(bands):
            data[b, :, :] = (b + 1) * 0.1

        with rio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=bands,
            dtype="float32",
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data)

    def test_read_orthophoto_bands_basic(self):
        """Test basic orthophoto band reading."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raster_path = Path(tmp_dir) / "test.tif"
            self.create_test_raster(raster_path)

            result = read_orthophoto_bands(str(raster_path))

            assert isinstance(result, pl.DataFrame)
            assert len(result) == 100
            assert "band1" in result.columns
            assert "band5" in result.columns
            assert "Xw" in result.columns
            assert "Yw" in result.columns

    def test_read_orthophoto_bands_with_transform(self):
        """Test orthophoto reading with coordinate transformation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raster_path = Path(tmp_dir) / "test.tif"
            self.create_test_raster(raster_path)

            result = read_orthophoto_bands(str(raster_path), transform_to_utm=True)

            assert isinstance(result, pl.DataFrame)
            assert "Xw" in result.columns
            assert "Yw" in result.columns

    def test_read_orthophoto_bands_different_band_count(self):
        """Test reading orthophoto with non-standard band count."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raster_path = Path(tmp_dir) / "test_3bands.tif"
            self.create_test_raster(raster_path, bands=3)

            result = read_orthophoto_bands(str(raster_path))

            assert isinstance(result, pl.DataFrame)
            assert len(result) == 100
            assert "band1" in result.columns
            assert "band2" in result.columns
            assert "band3" in result.columns
            assert "band4" not in result.columns

    def test_read_orthophoto_bands_no_transform(self):
        """Test reading orthophoto without coordinate transform."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raster_path = Path(tmp_dir) / "test_no_transform.tif"
            self.create_test_raster(raster_path)

            result = read_orthophoto_bands(str(raster_path), transform_to_utm=False)

            assert isinstance(result, pl.DataFrame)
            assert len(result) == 100
            assert "Xw" in result.columns
            assert "Yw" in result.columns

    def test_coregister_and_resample(self):
        """Test coregistration and resampling."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_path = Path(tmp_dir) / "ref.tif"
            self.create_test_raster(ref_path, width=10, height=10)

            input_path = Path(tmp_dir) / "input.tif"
            self.create_test_raster(input_path, width=8, height=8)

            output_path = Path(tmp_dir) / "output.tif"

            result = coregister_and_resample(str(input_path), str(ref_path), str(output_path))

            assert Path(result).exists()

            with rio.open(result) as dst:
                assert dst.width == 10
                assert dst.height == 10

    def test_check_alignment_same_raster(self):
        """Test alignment check with same raster."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raster_path = Path(tmp_dir) / "test.tif"
            self.create_test_raster(raster_path)

            assert check_alignment(str(raster_path), str(raster_path))

    def test_check_alignment_different_sizes(self):
        """Test alignment check with different sized rasters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raster1_path = Path(tmp_dir) / "raster1.tif"
            raster2_path = Path(tmp_dir) / "raster2.tif"

            self.create_test_raster(raster1_path, width=10, height=10)
            self.create_test_raster(raster2_path, width=8, height=8)

            result = check_alignment(str(raster1_path), str(raster2_path))
            assert isinstance(result, bool)

    def test_check_alignment_different_crs(self):
        """Test alignment check with same bounds but different CRS."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            raster1_path = Path(tmp_dir) / "raster_utm32.tif"
            raster2_path = Path(tmp_dir) / "raster_wgs84.tif"

            self.create_test_raster(raster1_path, width=10, height=10, crs="EPSG:32632")
            self.create_test_raster(raster2_path, width=10, height=10, crs="EPSG:4326")

            result = check_alignment(str(raster1_path), str(raster2_path))
            assert isinstance(result, bool)
            assert result is False

    def test_latlon_to_utm32n_series(self):
        """Test latitude/longitude to UTM conversion."""
        lat = 52.0
        lon = 13.0

        result = latlon_to_utm32n_series(lat, lon)

        assert len(result) == 2
        x, y = result

        assert 400000 < x < 900000
        assert 5000000 < y < 6000000

        lats = [52.0, 52.1, 52.2]
        lons = [13.0, 13.1, 13.2]

        for lat, lon in zip(lats, lons):
            result = latlon_to_utm32n_series(lat, lon)
            assert len(result) == 2
            x, y = result

            assert 400000 < x < 900000
            assert 5000000 < y < 6000000


T = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)


@pytest.mark.parametrize("lat,lon", [(52.0, 13.0), (52.1, 13.1), (52.2, 12.0)])
def test_latlon_to_utm32n_precise(lat, lon, tol=1.0):
    exp_x, exp_y = T.transform(lon, lat)
    x, y = latlon_to_utm32n_series(lat, lon)
    assert math.isfinite(x) and math.isfinite(y)
    assert abs(x - exp_x) < tol and abs(y - exp_y) < tol


def test_monotonic_local():
    lat, lon = 52.0, 13.0
    x0, y0 = latlon_to_utm32n_series(lat, lon)
    xE, yE = latlon_to_utm32n_series(lat, lon + 0.05)
    xN, yN = latlon_to_utm32n_series(lat + 0.05, lon)
    assert xE > x0 and yN > y0


class TestPlottingFunctions:
    """Test plotting functions."""

    def _make_test_df(self):
        return pl.DataFrame(
            {
                "Xw": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5],
                "Yw": [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5],
                "band1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.15, 0.25, 0.35, 0.45, 0.55],
                "band2": [0.2, 0.3, 0.4, 0.5, 0.6, 0.25, 0.35, 0.45, 0.55, 0.65],
                "band3": [0.3, 0.4, 0.5, 0.6, 0.7, 0.35, 0.45, 0.55, 0.65, 0.75],
                "OSAVI": [0.55, 0.60, 0.65, 0.70, 0.75, 0.58, 0.63, 0.68, 0.73, 0.78],
                "NDVI": [0.65, 0.70, 0.75, 0.80, 0.85, 0.68, 0.73, 0.78, 0.83, 0.88],
                "elev": [100.0, 101.0, 102.0, 103.0, 104.0, 100.5, 101.5, 102.5, 103.5, 104.5],
            }
        )

    def test_plotting_raster_basic(self):
        """Test basic raster plotting."""
        df = self._make_test_df()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plots"

            plotting_raster(df, str(output_path), "test_file", ny=5, nx=5, dpi=30)

            assert (output_path / "bands_data").exists()
            panels_png = output_path / "bands_data" / "panels_test_file.png"
            dist_png = output_path / "bands_data" / "band_distributions_test_file.png"
            assert panels_png.exists()
            assert dist_png.exists()

    def test_plotting_raster_empty_data(self):
        """Test plotting with empty dataframe."""
        df = pl.DataFrame({"Xw": [], "Yw": [], "band1": [], "elev": []})

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plots"

            plotting_raster(df, str(output_path), "empty_file")

    def test_plotting_raster_with_density(self):
        """Test plotting with density plot enabled."""
        df = self._make_test_df()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plots"

            plotting_raster(df, str(output_path), "density_test", plot_density=True, ny=5, nx=5, dpi=30)

            panels_png = output_path / "bands_data" / "panels_density_test.png"
            assert panels_png.exists()

    def test_plotting_raster_with_kde(self):
        """Test plotting with KDE density mode."""
        df = self._make_test_df()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plots"

            plotting_raster(
                df, str(output_path), "kde_test", plot_density=True, density_mode="kde", ny=5, nx=5, dpi=30
            )

            panels_png = output_path / "bands_data" / "panels_kde_test.png"
            assert panels_png.exists()

    def test_plotting_raster_with_custom_columns(self):
        """Test plotting with custom columns."""
        df = self._make_test_df()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plots"

            plotting_raster(
                df, str(output_path), "custom_test", custom_columuns=["OSAVI", "NDVI"], ny=5, nx=5, dpi=30
            )

            panels_png = output_path / "bands_data" / "panels_custom_test.png"
            assert panels_png.exists()

    def test_plotting_raster_with_auto_figsize(self):
        """Test plotting with auto_figsize enabled."""
        df = self._make_test_df()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plots"

            plotting_raster(df, str(output_path), "autofig_test", auto_figsize=True, ny=5, nx=5, dpi=30)

            panels_png = output_path / "bands_data" / "panels_autofig_test.png"
            assert panels_png.exists()

    def test_plotting_raster_with_band_kde(self):
        """Test plotting with band_kde enabled."""
        df = self._make_test_df()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plots"

            plotting_raster(df, str(output_path), "bandkde_test", band_kde=True, ny=5, nx=5, dpi=30)

            kde_png = output_path / "bands_data" / "band_kde_bandkde_test.png"
            assert kde_png.exists()

    def test_plotting_raster_fill_empty(self):
        """Test plotting with fill_empty disabled."""
        df = self._make_test_df()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "plots"

            plotting_raster(df, str(output_path), "fillempty_test", fill_empty=False, ny=5, nx=5, dpi=30)

            panels_png = output_path / "bands_data" / "panels_fillempty_test.png"
            assert panels_png.exists()


def test_auto_figsize():
    """Test _auto_figsize returns reasonable figure dimensions."""
    figsize, panelsize = _auto_figsize(
        nx=1000,
        ny=800,
        rows=2,
        cols=3,
        pixels_per_bin=4.0,
        dpi=200,
        min_panel_size=(4.0, 3.5),
        max_panel_size=(60.0, 120.0),
    )

    assert isinstance(figsize, tuple) and len(figsize) == 2
    assert isinstance(panelsize, tuple) and len(panelsize) == 2
    assert figsize[0] > 0 and figsize[1] > 0
    assert panelsize[0] > 0 and panelsize[1] > 0

    panel_w = (1000 * 4.0) / 200
    panel_h = (800 * 4.0) / 200
    assert panelsize[0] == panel_w
    assert panelsize[1] == panel_h
    assert figsize[0] == 3 * panel_w
    assert figsize[1] == 2 * panel_h


def test_auto_figsize_clamped():
    """Test _auto_figsize clamps to min/max panel sizes."""
    figsize, panelsize = _auto_figsize(
        nx=10,
        ny=10,
        rows=1,
        cols=1,
        pixels_per_bin=1.0,
        dpi=100,
        min_panel_size=(4.0, 3.5),
        max_panel_size=(60.0, 120.0),
    )

    assert panelsize[0] == 4.0
    assert panelsize[1] == 3.5


def test_finite_mask():
    """Test _finite_mask with NaN, Inf, and finite values."""
    arr = np.array([1.0, np.nan, 3.0, np.inf, -np.inf, 5.0, 2.0], dtype=np.float64)

    mask = _finite_mask(arr)

    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert bool(mask[0]) is True
    assert bool(mask[1]) is False
    assert bool(mask[2]) is True
    assert bool(mask[3]) is False
    assert bool(mask[4]) is False
    assert bool(mask[5]) is True
    assert bool(mask[6]) is True
    assert mask.sum() == 4


def test_finite_mask_multiple_arrays():
    """Test _finite_mask with multiple input arrays."""
    a = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    b = np.array([np.inf, 2.0, 3.0], dtype=np.float64)

    mask = _finite_mask(a, b)

    assert mask.tolist() == [False, False, True]


class TestErrorHandling:
    """Test error handling in raster operations."""

    def test_read_nonexistent_file(self):
        """Test reading nonexistent raster file."""
        with pytest.raises(Exception):
            read_orthophoto_bands("nonexistent_file.tif")

    def test_coregister_nonexistent_files(self):
        """Test coregistration with nonexistent files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.tif"

            with pytest.raises(Exception):
                coregister_and_resample(
                    "nonexistent_input.tif", "nonexistent_ref.tif", str(output_path)
                )

    def test_check_alignment_nonexistent_files(self):
        """Test alignment check with nonexistent files."""
        result = check_alignment("nonexistent1.tif", "nonexistent2.tif")
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__])
