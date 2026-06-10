import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
from shapely.geometry import Point, Polygon

from src.extract.polygon_filtering import (
    apply_polygon_shrinkage,
    check_data_polygon_overlap,
    combine_chunk_results,
    create_union_polygon,
    filter_df_by_polygon,
    load_and_prepare_polygons,
    prepare_chunks,
)


def test_overlap_detection():
    df = pl.DataFrame(
        {
            "Xw": [0.2, 0.8],
            "Yw": [0.2, 0.8],
            "band1": [1, 1],
            "elev": [100, 100],
        }
    )

    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf_poly = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:32632")

    has_overlap, data_bounds = check_data_polygon_overlap(df, gdf_poly)
    assert has_overlap is True
    assert len(data_bounds) == 4


# ---------------------------------------------------------------------------
# 1. filter_df_by_polygon with synthetic data
# ---------------------------------------------------------------------------


def test_filter_df_by_polygon_synthetic_data(tmp_path):
    xs = np.arange(0, 10, 1.0)
    ys = np.arange(0, 10, 1.0)
    X, Y = np.meshgrid(xs, ys)
    df = pl.DataFrame(
        {
            "Xw": X.ravel(),
            "Yw": Y.ravel(),
            "band1": np.ones(X.size, dtype=np.float64),
            "elev": np.full(X.size, 100.0, dtype=np.float64),
        }
    )

    rect = Polygon([(2.5, 2.5), (7.5, 2.5), (7.5, 7.5), (2.5, 7.5)])
    gdf_poly = gpd.GeoDataFrame({"id": ["plot_A"], "geometry": [rect]}, crs="EPSG:32632")
    poly_path = tmp_path / "rect.gpkg"
    gdf_poly.to_file(poly_path, driver="GPKG")

    result = filter_df_by_polygon(
        df, str(poly_path), target_crs="EPSG:32632", id_field="id", shrinkage=0, debug=False
    )

    assert result is not None
    assert "plot_id" in result.columns

    expected_count = 25
    assert len(result) == expected_count
    assert all(result["plot_id"] == "plot_A")

    for row in result.iter_rows(named=True):
        assert 3.0 <= row["Xw"] <= 7.0
        assert 3.0 <= row["Yw"] <= 7.0


def test_filter_df_by_polygon_target_crs_conversion(tmp_path):
    xs = np.arange(0, 6, 1.0)
    ys = np.arange(0, 6, 1.0)
    X, Y = np.meshgrid(xs, ys)
    df = pl.DataFrame(
        {
            "Xw": X.ravel(),
            "Yw": Y.ravel(),
            "band1": np.ones(X.size, dtype=np.float64),
            "elev": np.full(X.size, 100.0, dtype=np.float64),
        }
    )

    rect = Polygon([(2.5, 2.5), (5.5, 2.5), (5.5, 5.5), (2.5, 5.5)])
    gdf_poly = gpd.GeoDataFrame({"id": ["plot_B"], "geometry": [rect]}, crs="EPSG:32632")
    poly_path = tmp_path / "rect_crs.gpkg"
    gdf_poly.to_file(poly_path, driver="GPKG")

    result = filter_df_by_polygon(
        df, str(poly_path), target_crs="EPSG:32632", id_field="id", shrinkage=0, debug=False
    )

    assert result is not None
    assert len(result) == 9


# ---------------------------------------------------------------------------
# 2. load_and_prepare_polygons
# ---------------------------------------------------------------------------


def test_load_and_prepare_polygons_valid(tmp_path):
    rect = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
    gdf = gpd.GeoDataFrame(geometry=[rect], crs="EPSG:32632")
    path = tmp_path / "valid.gpkg"
    gdf.to_file(path, driver="GPKG")

    result = load_and_prepare_polygons(str(path), "EPSG:32632")
    assert len(result) == 1
    assert result.crs.to_string() == "EPSG:32632"


def test_load_and_prepare_polygons_empty_file(tmp_path):
    path = tmp_path / "empty.gpkg"
    gdf_empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:32632")
    gdf_empty.to_file(path, driver="GPKG")

    result = load_and_prepare_polygons(str(path), "EPSG:32632")
    assert result.empty


def test_load_and_prepare_polygons_crs_conversion(tmp_path):
    from pyproj import CRS

    rect = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
    gdf = gpd.GeoDataFrame(geometry=[rect], crs="EPSG:4326")
    path = tmp_path / "crs_conversion.gpkg"
    gdf.to_file(path, driver="GPKG")

    result = load_and_prepare_polygons(str(path), "EPSG:32632")
    assert len(result) == 1
    assert result.crs.to_string() != "EPSG:4326"


# ---------------------------------------------------------------------------
# 3. apply_polygon_shrinkage
# ---------------------------------------------------------------------------


def test_apply_polygon_shrinkage_reduces_area():
    rect = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    gdf = gpd.GeoDataFrame(geometry=[rect], crs="EPSG:32632")

    original_area = gdf.geometry.area.iloc[0]
    result = apply_polygon_shrinkage(gdf, shrinkage=0.1)
    shrunk_area = result.geometry.area.iloc[0]

    assert shrunk_area < original_area


def test_apply_polygon_shrinkage_zero_returns_same():
    rect = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    gdf = gpd.GeoDataFrame(geometry=[rect], crs="EPSG:32632")

    result = apply_polygon_shrinkage(gdf, shrinkage=0)
    assert result.geometry.area.iloc[0] == gdf.geometry.area.iloc[0]


# ---------------------------------------------------------------------------
# 4. create_union_polygon
# ---------------------------------------------------------------------------


def test_create_union_polygon():
    p1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    p2 = Polygon([(3, 3), (8, 3), (8, 8), (3, 8)])
    gdf = gpd.GeoDataFrame(geometry=[p1, p2], crs="EPSG:32632")

    union = create_union_polygon(gdf)
    assert union.area >= max(poly.area for poly in gdf.geometry)
    assert p1.within(union) or p1.intersects(union)
    assert p2.within(union) or p2.intersects(union)


# ---------------------------------------------------------------------------
# 5. prepare_chunks
# ---------------------------------------------------------------------------


def test_prepare_chunks():
    n = 500
    df = pl.DataFrame(
        {
            "Xw": np.random.rand(n),
            "Yw": np.random.rand(n),
            "band1": np.ones(n, dtype=np.float64),
            "elev": np.full(n, 100.0, dtype=np.float64),
        }
    )

    chunk_indices, max_workers, n_points, n_chunks = prepare_chunks(df)

    assert n_points == n
    assert n_chunks >= 1
    assert len(chunk_indices) == n_chunks

    covered = set()
    for start, end in chunk_indices:
        assert 0 <= start < end <= n
        for i in range(start, end):
            covered.add(i)

    assert len(covered) == n
    assert covered == set(range(n))


def test_prepare_chunks_small_df():
    df = pl.DataFrame(
        {
            "Xw": [0.5],
            "Yw": [0.5],
            "band1": [1.0],
            "elev": [100.0],
        }
    )

    chunk_indices, max_workers, n_points, n_chunks = prepare_chunks(df)
    assert n_chunks == 1
    assert chunk_indices[0] == (0, 1)


# ---------------------------------------------------------------------------
# 6. combine_chunk_results
# ---------------------------------------------------------------------------


def test_combine_chunk_results():
    chunk1 = gpd.GeoDataFrame(
        {"plot_id": ["A", "A"], "Xw": [1.0, 2.0], "Yw": [1.0, 2.0]},
        geometry=gpd.points_from_xy([1.0, 2.0], [1.0, 2.0]),
        crs="EPSG:32632",
    )
    chunk2 = gpd.GeoDataFrame(
        {"plot_id": ["B"], "Xw": [3.0], "Yw": [3.0]},
        geometry=gpd.points_from_xy([3.0], [3.0]),
        crs="EPSG:32632",
    )

    result = combine_chunk_results([chunk1, chunk2], n_points=100, phase_time=1.0)
    assert result is not None
    assert len(result) == 3
    assert sorted(result["plot_id"].to_list()) == ["A", "A", "B"]


def test_combine_chunk_results_empty():
    result = combine_chunk_results([], n_points=100, phase_time=1.0)
    assert result is None


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


def test_filter_df_by_polygon_empty_df(tmp_path):
    rect = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    gdf = gpd.GeoDataFrame({"id": ["A"], "geometry": [rect]}, crs="EPSG:32632")
    path = tmp_path / "empty.gpkg"
    gdf.to_file(path, driver="GPKG")

    df = pl.DataFrame(
        schema={"Xw": pl.Float64, "Yw": pl.Float64, "band1": pl.Float64, "elev": pl.Float64}
    )
    result = filter_df_by_polygon(df, str(path), target_crs="EPSG:32632", shrinkage=0, debug=False)
    assert result is None


def test_filter_df_by_polygon_no_points_inside(tmp_path):
    xs = np.arange(0, 3, 1.0)
    ys = np.arange(0, 3, 1.0)
    X, Y = np.meshgrid(xs, ys)
    df = pl.DataFrame(
        {
            "Xw": X.ravel(),
            "Yw": Y.ravel(),
            "band1": np.ones(X.size, dtype=np.float64),
            "elev": np.full(X.size, 100.0, dtype=np.float64),
        }
    )

    far_rect = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])
    gdf = gpd.GeoDataFrame({"id": ["far"], "geometry": [far_rect]}, crs="EPSG:32632")
    path = tmp_path / "far.gpkg"
    gdf.to_file(path, driver="GPKG")

    result = filter_df_by_polygon(df, str(path), target_crs="EPSG:32632", shrinkage=0, debug=False)
    assert result is None


def test_empty_polygon_file_returns_none(tmp_path):
    df = pl.DataFrame(
        {
            "Xw": [1.0, 2.0],
            "Yw": [1.0, 2.0],
            "band1": [1.0, 1.0],
            "elev": [100.0, 100.0],
        }
    )

    path = tmp_path / "empty_poly.gpkg"
    gdf_empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:32632")
    gdf_empty.to_file(path, driver="GPKG")

    result = filter_df_by_polygon(df, str(path), target_crs="EPSG:32632", shrinkage=0, debug=False)
    assert result is None
