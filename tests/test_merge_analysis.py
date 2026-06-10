import numpy as np
import polars as pl
import rasterio as rio
from rasterio.transform import from_bounds

from src.extract.merge_analysis import (
    analyze_kdtree_matching,
    merge_data,
    reproject_dem_to_band_grid_single,
    sample_dem_at_band_pixels,
)
from src.extract.raster import read_orthophoto_bands


def test_kdtree_matching_no_crash():
    df_dem = pl.DataFrame({"Xw": [0.0, 1.0, 2.0], "Yw": [0.0, 1.0, 2.0]})
    df_all = pl.DataFrame({"Xw": [0.0, 1.0, 2.0], "Yw": [0.0, 1.0, 2.0]})

    stats = analyze_kdtree_matching(df_dem, df_all, precision=0)
    assert "exact_matches" in stats
    assert stats["exact_matches"] >= 3


def test_merge_data(tmp_path):
    width, height = 10, 10
    crs = "EPSG:32632"
    transform = from_bounds(0, 0, 10, 10, width, height)

    band_path = tmp_path / "ortho.tif"
    data = np.zeros((5, height, width), dtype=np.float32)
    for b in range(5):
        data[b] = (b + 1) * 0.01
    with rio.open(
        band_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=5,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

    dem_path = tmp_path / "dem.tif"
    dem_data = np.arange(100.0, 100.0 + height * width, dtype=np.float32).reshape(height, width)
    with rio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(dem_data, 1)

    df_allbands = read_orthophoto_bands(str(band_path))
    result = merge_data(df_allbands, str(band_path), str(dem_path))

    expected_cols = ["Xw", "Yw"] + [f"band{i}" for i in range(1, 6)] + ["elev"]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"

    assert len(result) == width * height

    elev_col = result["elev"].to_numpy()
    assert np.all(np.isfinite(elev_col))

    xs, ys = result["Xw"].to_list(), result["Yw"].to_list()
    assert len(xs) == width * height
    assert np.all(np.isfinite(np.array(xs)))
    assert np.all(np.isfinite(np.array(ys)))

    for i in range(1, 6):
        vals = result[f"band{i}"].to_numpy()
        assert np.allclose(vals, float(i) * 0.01)

    elev_rounded = elev_col.round().astype(int)
    expected_range = set(range(100, 100 + height * width))
    for v in elev_rounded:
        assert v in expected_range, f"Unexpected elevation value: {v}"


def test_analyze_kdtree_50_percent_overlap():
    df_dem = pl.DataFrame(
        {
            "Xw": [0.0, 1.0, 2.0, 3.0],
            "Yw": [0.0, 1.0, 2.0, 3.0],
        }
    )
    df_all = pl.DataFrame(
        {
            "Xw": [0.0, 1.0, 5.0, 6.0],
            "Yw": [0.0, 1.0, 5.0, 6.0],
        }
    )

    stats = analyze_kdtree_matching(df_dem, df_all, precision=0, max_distance=1.0)
    assert stats["exact_matches"] == 2
    assert stats["no_matches"] >= 2


def test_analyze_kdtree_no_overlap():
    df_dem = pl.DataFrame(
        {
            "Xw": [0.0, 1.0, 2.0],
            "Yw": [0.0, 1.0, 2.0],
        }
    )
    df_all = pl.DataFrame(
        {
            "Xw": [100.0, 101.0, 102.0],
            "Yw": [100.0, 101.0, 102.0],
        }
    )

    stats = analyze_kdtree_matching(df_dem, df_all, precision=0, max_distance=1.0)
    assert stats["exact_matches"] == 0
    assert stats["no_matches"] == 3


def test_analyze_kdtree_large():
    np.random.seed(42)
    n = 1200
    xs = np.random.uniform(0, 100, n).tolist()
    ys = np.random.uniform(0, 100, n).tolist()
    df_dem = pl.DataFrame({"Xw": xs, "Yw": ys})
    df_all = pl.DataFrame({"Xw": xs, "Yw": ys})

    stats = analyze_kdtree_matching(df_dem, df_all, precision=2)
    assert stats["exact_matches"] == n
    assert stats["near_matches"] == 0


def test_reproject_dem_to_band_grid_single(tmp_path):
    crs = "EPSG:32632"
    dem_path = tmp_path / "dem.tif"
    band_path = tmp_path / "band.tif"
    out_path = tmp_path / "dem_reproj.tif"

    dem_width, dem_height = 5, 5
    dem_transform = from_bounds(0, 0, 10, 10, dem_width, dem_height)
    dem_data = np.ones((dem_height, dem_width), dtype=np.float32) * 42.0
    with rio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=dem_width,
        height=dem_height,
        count=1,
        dtype="float32",
        crs=crs,
        transform=dem_transform,
    ) as dst:
        dst.write(dem_data, 1)

    band_width, band_height = 10, 10
    band_transform = from_bounds(0, 0, 10, 10, band_width, band_height)
    band_data = np.zeros((1, band_height, band_width), dtype=np.float32)
    with rio.open(
        band_path,
        "w",
        driver="GTiff",
        width=band_width,
        height=band_height,
        count=1,
        dtype="float32",
        crs=crs,
        transform=band_transform,
    ) as dst:
        dst.write(band_data)

    result_path = reproject_dem_to_band_grid_single(str(dem_path), str(band_path), str(out_path))
    assert result_path == str(out_path)

    with rio.open(out_path) as src:
        assert src.width == band_width
        assert src.height == band_height
        assert src.crs == rio.open(band_path).crs


def test_sample_dem_at_band_pixels(tmp_path):
    crs = "EPSG:32632"
    width, height = 3, 3
    transform = from_bounds(0, 0, 3, 3, width, height)

    band_path = tmp_path / "band.tif"
    band_data = np.zeros((1, height, width), dtype=np.float32)
    with rio.open(
        band_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(band_data)

    dem_path = tmp_path / "dem.tif"
    dem_data = np.arange(10.0, 10.0 + width * height, dtype=np.float32).reshape(height, width)
    with rio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(dem_data, 1)

    sampled = sample_dem_at_band_pixels(str(band_path), str(dem_path))
    assert sampled.shape == (height, width)
    expected = np.arange(10.0, 19.0).reshape(height, width)
    assert np.allclose(sampled, expected)


def test_sample_dem_at_band_pixels_with_nan(tmp_path):
    crs = "EPSG:32632"
    width, height = 3, 3
    transform = from_bounds(0, 0, 3, 3, width, height)

    band_path = tmp_path / "band.tif"
    band_data = np.zeros((1, height, width), dtype=np.float32)
    with rio.open(
        band_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(band_data)

    dem_path = tmp_path / "dem_nan.tif"
    dem_data = np.arange(10.0, 19.0, dtype=np.float32).reshape(height, width)
    dem_data[0, 0] = np.nan
    dem_data[1, 1] = np.nan
    with rio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(dem_data, 1)

    sampled = sample_dem_at_band_pixels(str(band_path), str(dem_path))
    assert sampled.shape == (height, width)
    assert np.isnan(sampled[0, 0])
    assert np.isnan(sampled[1, 1])
    valid_mask = ~np.isnan(sampled)
    if valid_mask.any():
        assert valid_mask.sum() == 7
