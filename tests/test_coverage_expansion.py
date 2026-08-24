"""Behavioral coverage for public helpers and orchestration paths.

These tests use small synthetic rasters/tables and patch only external or
computationally expensive collaborators.  They are deliberately assertions
about returned data and generated files, rather than line-execution tests.
"""

from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pytest
import rasterio
from rasterio.transform import from_bounds, from_origin
from shapely.geometry import Polygon

from oncerco_uav.core import preprocess as preprocess_module
from oncerco_uav.core import search as search_module
from oncerco_uav.core.validate import REQUIRED_COLUMNS, validate_extract_output, validate_single_parquet
from oncerco_uav.extract import camera as camera_module
from oncerco_uav.extract import merge_analysis as merge_module
from oncerco_uav.extract import polygon_filtering as polygon_module
from oncerco_uav.extract import raster as raster_module
from oncerco_uav.filter import data_loader
from oncerco_uav.filter import filters as filters_module
from oncerco_uav.modelling import plotting as modelling_plotting
from oncerco_uav.modelling import processing as modelling_processing
from oncerco_uav.pipelines import extract_data as extract_pipeline
from oncerco_uav.pipelines import filtering as filtering_pipeline
from oncerco_uav.pipelines import modelling as modelling_pipeline
from oncerco_uav.stats import Logistic_regression as logistic_module
from oncerco_uav.stats import plotting as stats_plotting
from oncerco_uav.stats import processing as stats_processing


def _minimal_preprocess_frame(include_derived=False):
    frame = pl.DataFrame(
        {
            "Xw": [0.0, 1.0, 0.0],
            "Yw": [0.0, 0.0, 1.0],
            "xcam": [2.0, 2.0, 2.0],
            "ycam": [3.0, 3.0, 3.0],
            "delta_z": [4.0, 4.0, 4.0],
            "sunelev": [40.0, 40.0, 40.0],
            "saa": [180.0, 180.0, 180.0],
            "band1": [0.1, 0.2, 0.3],
            "band2": [0.2, 0.3, 0.4],
            "band3": [0.1, 0.2, 0.3],
            "band4": [0.4, 0.5, 0.6],
            "band5": [0.5, 0.6, 0.7],
        }
    )
    if include_derived:
        frame = preprocess_module.df_preprocess(frame)
    return frame


def test_df_preprocess_builds_indices_and_debug_repairs_values():
    out = preprocess_module.df_preprocess(_minimal_preprocess_frame())
    assert {"vx", "vy", "vz", "v_norm", "vza", "vaa", "sza", "NDVI", "raa", "OSAVI"}.issubset(
        out.columns
    )
    assert out["vaa"][0] == pytest.approx(33.69, abs=0.01)
    assert out["sza"][0] == pytest.approx(50.0)

    corrupted = out.with_columns(pl.lit(999.0).alias("NDVI"))
    repaired = preprocess_module.df_preprocess(corrupted, debug=True)
    expected_ndvi = (repaired["band5"] - repaired["band3"]) / (
        repaired["band5"] + repaired["band3"]
    )
    assert np.allclose(repaired["NDVI"].to_numpy(), expected_ndvi.to_numpy())


def test_df_preprocess_checks_existing_vectors_and_drops_nulls():
    clean = preprocess_module.df_preprocess(_minimal_preprocess_frame(), debug=False)
    checked = preprocess_module.df_preprocess(clean, debug=True)
    assert checked.height == clean.height
    assert np.allclose(checked["v_norm"].to_numpy(), clean["v_norm"].to_numpy())

    with pytest.raises(AssertionError, match="vx mismatch"):
        preprocess_module.df_preprocess(clean.with_columns(pl.lit(0.0).alias("vx")))

    with_nan = _minimal_preprocess_frame().with_columns(
        pl.Series("band1", [0.1, float("nan"), 0.3])
    )
    assert preprocess_module.df_preprocess(with_nan).height == 2


def _validation_frame(**updates):
    values = {column: [0.0, 1.0] for column in REQUIRED_COLUMNS}
    values.update(
        {
            "band1": [0.1, 0.2],
            "band2": [0.2, 0.3],
            "band3": [0.3, 0.4],
            "band4": [0.4, 0.5],
            "band5": [0.5, 0.6],
            "vza": [10.0, 20.0],
            "vaa": [10.0, 20.0],
            "sunelev": [40.0, 40.0],
            "saa": [180.0, 180.0],
            "elev": [100.0, 101.0],
        }
    )
    values.update(updates)
    return pl.DataFrame(values)


def test_validate_reports_corruption_ranges_and_optional_schema(tmp_path):
    good = _validation_frame(plot_id=["a", "b"], extra_debug=[1, 2])
    good.write_parquet(tmp_path / "good.parquet")
    (tmp_path / "corrupt.parquet").write_bytes(b"not parquet")
    result = validate_extract_output(tmp_path)
    assert not result["ok"]
    assert result["n_corrupt"] == 1
    assert "extra_debug" in result["extra_columns"]
    assert "corrupt.parquet" in result["schema_issues"][0]["file"]

    bad = _validation_frame(vza=[None, None], band1=[-0.1, 4.5])
    bad.write_parquet(tmp_path / "bad.parquet")
    result = validate_extract_output(tmp_path)
    assert not result["ok"]
    assert any("all null" in issue for issue in result["range_issues"])
    assert any("max=" in issue or "min=" in issue for issue in result["range_issues"])
    with pytest.raises(ValueError, match="Data validation failed"):
        validate_extract_output(tmp_path, raise_on_error=True)


def test_validate_single_parquet_and_missing_plot_id(tmp_path):
    path = tmp_path / "single.parquet"
    _validation_frame().write_parquet(path)
    result = validate_single_parquet(path)
    assert result["ok"]
    assert result["n_files"] == 1
    assert result["files_without_plot_id"] == ["single.parquet"]


def test_search_helpers_cover_ordering_errors_and_unknown_weeks(tmp_path, monkeypatch):
    ordered = search_module.order_path_list(
        ["/a/plot_10.parquet", "/a/no_plot.parquet", "/a/plot_2.parquet", "/a/plot_bad.parquet"]
    )
    assert ordered[:2] == ["/a/plot_2.parquet", "/a/plot_10.parquet"]
    assert ordered[2:] == ["/a/no_plot.parquet", "/a/plot_bad.parquet"]

    week_dir = tmp_path / "week3"
    week_dir.mkdir()
    (week_dir / "plot_1.parquet").write_bytes(b"placeholder")
    assert search_module.search_directory(str(week_dir), "plot")
    assert search_module.search_directory(str(week_dir), "missing") is None

    original_glob = search_module.glob.glob
    monkeypatch.setattr(search_module.glob, "glob", lambda _: (_ for _ in ()).throw(OSError("denied")))
    assert search_module.search_directory(str(week_dir), "plot") is None
    monkeypatch.setattr(search_module.glob, "glob", original_glob)

    hidden = tmp_path / "$hidden"
    hidden.mkdir()
    (hidden / "plot_hidden.parquet").write_bytes(b"x")
    unknown = tmp_path / "relevant"
    unknown.mkdir()
    (unknown / "plot_unknown.parquet").write_bytes(b"x")
    results = search_module.optimized_recursive_search([], "plot", str(tmp_path), remove_unkwown=False)
    assert "week3" in results
    assert "unknown" in results
    assert all("$hidden" not in path for paths in results.values() for path in paths)
    assert "unknown" not in search_module.optimized_recursive_search([], "plot", str(tmp_path))


def _camera_row(photo_id, x, y, z):
    return "\t".join(
        [photo_id, str(x), str(y), str(z)] + ["0"] * 3 + ["1", "0", "0", "0", "1", "0", "0", "0", "1"]
    )


def test_camera_position_crs_return_modes_and_duplicate_without_raster(tmp_path):
    path = tmp_path / "cameras.txt"
    path.write_text(
        "header 1\nheader 2\n"
        + _camera_row("IMG", 9.0, 51.0, 100)
        + "\n"
        + _camera_row("IMG", 10.0, 52.0, 200)
        + "\n",
        encoding="utf-8",
    )
    lon, lat, z, geo_lon, geo_lat = camera_module.get_camera_position(
        path, "IMG", target_crs="EPSG:32632", return_geographic=True
    )
    assert (geo_lon, geo_lat, z) == pytest.approx((9.0, 51.0, 100.0))
    assert lon > 400_000 and lat > 5_000_000
    assert camera_module.get_camera_position(path, "IMG") == pytest.approx((9.0, 51.0, 100.0))
    with pytest.raises(ValueError, match="not found"):
        camera_module.get_camera_position(path, "missing")


def test_plot_angles_writes_all_views(tmp_path):
    df = pl.DataFrame(
        {
            "Xw": [0.0, 1.0, 2.0],
            "Yw": [0.0, 1.0, 2.0],
            "elev": [10.0, 11.0, 12.0],
            "vza": [5.0, 10.0, 15.0],
        }
    )
    camera_module.plot_angles(df, 0.0, 0.0, 100.0, tmp_path, "IMG")
    for view in ("top_down", "side_view", "3d_view"):
        assert (tmp_path / view / "angle_data_IMG.png").exists()


def test_data_loader_helpers_and_derived_columns(tmp_path):
    first = pl.DataFrame({"plot_id": ["a", "b"], "band1": [1.0, 2.0], "band2": [2.0, 3.0]})
    second = pl.DataFrame({"plot_id": ["a"], "band1": [4.0], "band2": [5.0]})
    no_id = pl.DataFrame({"band1": [1.0]})
    first.write_parquet(tmp_path / "first.parquet")
    second.write_parquet(tmp_path / "second.parquet")
    no_id.write_parquet(tmp_path / "no_id.parquet")

    loaded = data_loader._read_folder(tmp_path)
    assert len(loaded) == 3
    assert data_loader.unique_plot_ids(loaded) == {"a", "b"}
    split = data_loader.split_by_polygon(loaded, {"a", "missing"})
    assert split["a"].height == 2
    assert split["missing"].height == 0
    assert list(data_loader.batched(range(5), 2)) == [[0, 1], [2, 3], [4]]

    out = tmp_path / "out"
    data_loader.load_by_polygon(
        str(tmp_path),
        str(out),
        specific="a",
        derived_columns=[(pl.col("band1") + pl.col("band2")).alias("sum")],
    )
    result = pl.read_parquet(out / "a.parquet")
    assert sorted(result["sum"].to_list()) == [3.0, 9.0]
    polygon_dict = data_loader.create_polygon_dict(
        {"first": first, "no_id": no_id}, ["a", "b"]
    )
    assert polygon_dict["a"].height == 1
    assert polygon_dict["b"].height == 1


def test_modelling_processing_returns_success_and_error_rows(tmp_path, monkeypatch):
    path = tmp_path / "plot.parquet"
    pl.DataFrame({"value": [1.0, 2.0, 3.0]}).write_parquet(path)

    monkeypatch.setattr(modelling_processing, "df_preprocess", lambda df, debug=False: df)
    monkeypatch.setattr(modelling_processing, "rpv_fit", lambda dg, **kwargs: (1, 2, 3, 4, 5, 6))
    rows = pl.DataFrame(
        {
            "ifz_id": pl.Series("ifz_id", [12, 13], dtype=pl.Int64),
            "cult": ["A", "B"],
            "trt": ["yes", "no"],
            "geometry": ["POINT (0 0)", None],
            "paths": [str(path), str(tmp_path / "missing.parquet")],
            "value": [1.0, 0.0],
        }
    )
    result = modelling_processing.process_weekly_data_rpv(
        {"week1": rows}, band="band5", sample_total_dataset=2, filter={"column": "value", "sign": ">", "threshold": 0}
    )
    assert result.height == 2
    assert result.filter(pl.col("status") == "success").height == 1
    assert result.filter(pl.col("status").str.starts_with("error:")).height == 1
    assert result.filter(pl.col("status") == "success")["plot_id"].item() == 12


def test_modelling_plotting_skips_existing_and_bad_paths(tmp_path, monkeypatch):
    valid = tmp_path / "valid.parquet"
    pl.DataFrame({"Xw": [0.0], "Yw": [0.0]}).write_parquet(valid)
    panel_dir = tmp_path / "bands_data"
    panel_dir.mkdir()
    (panel_dir / "panels_valid.png").write_bytes(b"already there")
    calls = []
    monkeypatch.setattr(modelling_plotting, "plotting_raster", lambda *args, **kwargs: calls.append(args))
    gdf = pl.DataFrame({"paths": [str(valid), str(tmp_path / "bad.parquet"), None]})
    modelling_plotting.plot_df("week1", gdf, tmp_path)
    assert calls == []
    (panel_dir / "panels_valid.png").unlink()
    modelling_plotting.plot_df("week1", gdf, tmp_path)
    assert len(calls) == 1
    assert calls[0][2] == "valid"


def _write_test_raster(path, *, width=8, height=8, count=2, crs="EPSG:32632", transform=None):
    transform = transform or from_origin(500000, 5000000, 1, 1)
    data = np.stack([np.full((height, width), i + 1, dtype=np.float32) for i in range(count)])
    with rasterio.open(
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


def test_raster_read_without_transform_coregistration_and_alignment(tmp_path):
    source = tmp_path / "source.tif"
    reference = tmp_path / "reference.tif"
    _write_test_raster(source, count=2)
    _write_test_raster(reference, count=2, width=4, height=4, transform=from_origin(500000, 5000000, 2, 2))
    no_transform = raster_module.read_orthophoto_bands(source, transform_to_utm=False)
    assert no_transform.height == 64
    assert no_transform["Xw"].min() > 499999

    output = tmp_path / "nested" / "coreg.tif"
    result = raster_module.coregister_and_resample(source, reference, output, target_resolution=(2, 2))
    assert result == output
    with rasterio.open(output) as src:
        assert (src.width, src.height) == (4, 4)
    assert raster_module.check_alignment(reference, reference)
    assert not raster_module.check_alignment(reference, source)
    assert not raster_module.check_alignment(reference, tmp_path / "missing.tif")


def test_raster_numeric_and_density_helpers(tmp_path):
    transform = from_origin(100, 200, 2, 2)
    assert raster_module.xy_np(transform, [0, 1], [0, 1], offset="ur")[0] == [102.0, 104.0]
    assert raster_module.xy_np(transform, 0, 0, offset="ll")[1] == [198.0]
    x = np.array([0.1, 0.2, 0.9, 0.95])
    y = np.array([0.1, 0.8, 0.2, 0.9])
    result = raster_module._auto_coarsen_for_occupancy(
        x, y, 0, 1, 0, 1, 20, 20, occupancy_target=0.8, min_bins=2, debug=True
    )
    assert result[0] <= 20 and result[1] <= 20
    no_change = raster_module._auto_coarsen_for_occupancy(x, y, 0, 1, 0, 1, 4, 4, None, 2, False)
    assert no_change[0:2] == (4, 4)
    with pytest.raises(ValueError, match="Invalid offset"):
        raster_module.xy_np(transform, 0, 0, offset="bad")

    xbins, ybins = raster_module._make_bins(0, 1, 0, 1, 4, 4)
    for mode, kwargs in [("hist", {}), ("kde", {"kde_bw": "scott"}), ("kde", {"kde_bw": (0.2, 0.3)}), ("kde_exact", {}), ("other", {})]:
        density = raster_module._density_grid(x, y, xbins, ybins, mode, kwargs.get("kde_bw"), True)
        assert density.shape == (4, 4)

    grid = raster_module._grid_mean_for_series(
        pl.Series("band", [1.0, np.nan, 3.0, 4.0]),
        np.array([True, True, True, True]),
        x,
        y,
        xbins,
        ybins,
        np.ones((4, 4)),
        fill_empty=True,
        debug=True,
        name="band",
    )
    assert np.isfinite(grid).any()
    pdf = raster_module._kde1d_fast(np.linspace(0.1, 0.9, 20), np.linspace(0, 1, 50), bins=32)
    assert np.trapezoid(pdf, np.linspace(0, 1, 50)) == pytest.approx(1.0, rel=0.1)
    assert np.all(raster_module._kde1d_fast(np.array([1.0, 2.0]), np.array([0.0, 1.0])) == 0)


def test_plotting_raster_generates_panels_hist_scatter_and_kde(tmp_path):
    rng = np.random.default_rng(7)
    n = 80
    df = pl.DataFrame(
        {
            "Xw": rng.uniform(0, 10, n),
            "Yw": rng.uniform(0, 10, n),
            "band1": rng.uniform(0.1, 0.4, n),
            "band2": rng.uniform(0.2, 0.5, n),
            "band3": rng.uniform(0.3, 0.6, n),
            "elev": rng.uniform(90, 110, n),
            "custom": rng.uniform(0, 1, n),
        }
    )
    raster_module.plotting_raster(
        df,
        tmp_path,
        "synthetic",
        nx=8,
        ny=8,
        min_bins=2,
        max_bands=3,
        plot_density=True,
        density_mode="kde",
        density_kde_bw=0.5,
        density_log=True,
        scatter_quicklook=True,
        scatter_max=10,
        custom_columuns=["custom", "missing"],
        band_kde=True,
        band_kde_points=32,
        auto_figsize=False,
        dpi=20,
        debug=True,
    )
    out = tmp_path / "bands_data"
    assert (out / "panels_synthetic.png").exists()
    assert (out / "band_distributions_synthetic.png").exists()
    assert (out / "band_kde_synthetic.png").exists()
    assert (out / "scatter_quicklook_synthetic.png").exists()


def test_polygon_chunk_and_debug_plot_branches(tmp_path):
    from oncerco_uav.extract import polygon_filtering as polygon_module

    polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    geographic_gdf = gpd.GeoDataFrame({"id": ["inside"]}, geometry=[polygon], crs="EPSG:4326")
    path = tmp_path / "polygons.gpkg"
    geographic_gdf.to_file(path, driver="GPKG")
    assert polygon_module.is_pos_inside_polygon(1, 1, {"Polygon_path": str(path)})
    assert not polygon_module.is_pos_inside_polygon(3, 3, {"Polygon_path": str(path)})
    with pytest.raises(TypeError):
        polygon_module.is_pos_inside_polygon("1", 1, {"Polygon_path": str(path)})
    with pytest.raises(FileNotFoundError):
        polygon_module.is_pos_inside_polygon(1, 1, {"Polygon_path": str(tmp_path / "none")})

    projected_gdf = geographic_gdf.to_crs("EPSG:32632")
    projected_polygon = projected_gdf.geometry.iloc[0]
    points = pl.DataFrame(
        {
            "Xw": [projected_polygon.centroid.x, projected_polygon.centroid.x + 1, 900000.0],
            "Yw": [projected_polygon.centroid.y, projected_polygon.centroid.y + 1, 9000000.0],
            "band1": [1.0, 0.0, 1.0],
        }
    )
    matched = polygon_module.process_chunk((0, 3), points, projected_gdf, "EPSG:32632")
    assert matched is not None and matched["plot_id"].tolist() == ["inside", "inside"]
    no_match = polygon_module.process_chunk((0, 3), points, projected_gdf, "EPSG:32632", id_field="missing")
    assert no_match is not None and set(no_match["plot_id"]) == {"plot_0"}
    chunks, count, _ = polygon_module.process_chunks_parallel(
        points, [(0, 2), (2, 3)], 2, projected_gdf, "EPSG:32632", "id", 2
    )
    assert count == 2 and len(chunks) == 1

    far = gpd.GeoDataFrame({"plot_id": ["far"]}, geometry=[Polygon([(20, 20), (22, 20), (22, 22), (20, 22)])], crs="EPSG:32632")
    polygon_module.plot_no_overlap(far, [0, 0, 1, 1], plots_out=tmp_path, img_name="far", debug=True)
    assert (tmp_path / "polygon_filtering_data" / "no_overlap_far.png").exists()


def test_extract_pipeline_helpers_and_process_path(tmp_path, monkeypatch):
    out_dir = tmp_path / "extract"
    out_dir.mkdir()
    processed = out_dir / "run_IMG_12_3.tif.parquet"
    processed.write_bytes(b"already")
    assert extract_pipeline.check_already_processed(out_dir) == {12}

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("IMG_12_3.tif", "IMG_13_3.tif", "unmatched.tif"):
        (image_dir / name).write_bytes(b"image")
    remaining = extract_pipeline.remove_images_already_processed(str(image_dir / "*.tif"), out_dir)
    assert {p.name for p in remaining} == {"IMG_13_3.tif", "unmatched.tif"}

    ori = tmp_path / "ori"
    ori.mkdir()
    (ori / "a.tif").write_bytes(b"a")
    assert extract_pipeline.retrieve_orthophoto_paths([str(ori)]) == [str(ori / "a.tif").replace("/", "\\")]

    monkeypatch.setattr(extract_pipeline.solar.solar, "get_altitude", lambda *args: 12.5)
    monkeypatch.setattr(extract_pipeline.solar.solar, "get_azimuth", lambda *args: -5.0)
    elev, azimuth = extract_pipeline.extract_sun_angles("IMG", 9.0, 51.0, "2024-06-01 12:00:00")
    assert (elev, azimuth) == pytest.approx((12.5, 355.0))
    assert extract_pipeline.extract_sun_angles("IMG", 9.0, 51.0, "bad") == (0.0, 0.0)

    df = pl.DataFrame(
        {
            "Xw": [1.0, 2.0],
            "Yw": [3.0, 4.0],
            "elev": [100.0, 101.0],
            "band1": [0.2, 0.0],
            "band2": [0.3, 0.4],
        }
    )
    source = {
        "target_crs": "EPSG:32632",
        "dem_path": str(tmp_path / "dem.tif"),
        "bands": 2,
        "Polygon_path": str(tmp_path / "polygons.gpkg"),
        "plot out": tmp_path / "plots",
        "start date": "2024-06-01 12:00:00",
        "time zone": "UTC",
        "name": "run",
    }
    source["plot out"].mkdir()
    calls = []
    checks = iter([False, True])
    monkeypatch.setattr(extract_pipeline, "get_camera_position", lambda *args, **kwargs: (1.0, 2.0, 100.0, 9.0, 51.0))
    monkeypatch.setattr(extract_pipeline, "check_alignment", lambda *args: next(checks))
    monkeypatch.setattr(extract_pipeline, "coregister_and_resample", lambda *args, **kwargs: str(tmp_path / "aligned.tif"))
    monkeypatch.setattr(extract_pipeline, "read_orthophoto_bands", lambda *args, **kwargs: df)
    monkeypatch.setattr(extract_pipeline, "merge_data", lambda *args, **kwargs: df)
    monkeypatch.setattr(extract_pipeline, "filter_df_by_polygon", lambda frame, **kwargs: frame)
    monkeypatch.setattr(extract_pipeline, "extract_sun_angles", lambda *args: (30.0, 180.0))
    monkeypatch.setattr(extract_pipeline, "calculate_angles", lambda frame, *args: frame)
    monkeypatch.setattr(extract_pipeline, "plotting_raster", lambda *args, **kwargs: calls.append("raster"))
    monkeypatch.setattr(extract_pipeline, "plot_angles", lambda *args, **kwargs: calls.append("angles"))
    monkeypatch.setattr(extract_pipeline, "save_parquet", lambda frame, *args: calls.append(frame))

    extract_pipeline.process_orthophoto(
        str(tmp_path / "IMG_1_3.tif"),
        "camera.txt",
        [],
        out_dir,
        source,
        0,
        "exiftool",
        polygon_filtering=True,
        alignment=True,
    )
    assert calls[:2] == ["raster", "angles"]
    assert calls[2].height == 1
    assert calls[2]["path"].to_list() == ["IMG_1_3.tif"]


def test_extract_pipeline_save_and_main_validation_branches(tmp_path, monkeypatch):
    class FallbackWriter:
        def __init__(self):
            self.calls = []

        def write_parquet(self, path, **kwargs):
            self.calls.append(kwargs.get("compression"))
            if kwargs.get("compression") in {"zstd", "snappy"}:
                raise RuntimeError("codec unavailable")
            Path(path).write_bytes(b"parquet")

    writer = FallbackWriter()
    extract_pipeline.save_parquet(writer, tmp_path, {"name": "run"}, 2, "IMG.tif")
    assert writer.calls == ["zstd", "snappy", None]

    config = SimpleNamespace(
        main_extract_out=tmp_path / "main_out",
        main_extract_cam_path="camera.txt",
        main_extract_dem_path="dem.tif",
        main_extract_ori=["ori"],
        main_extract_name="run",
        main_extract_path_list_tag="*.tif",
        bands=2,
        main_polygon_path="polygons.gpkg",
        start_date="2024-06-01 12:00:00",
        time_zone="UTC",
        plot_out=tmp_path / "plots",
        target_crs="EPSG:32632",
    )
    (tmp_path / "main_out").mkdir()
    image = tmp_path / "IMG_4_3.tif"
    image.write_bytes(b"image")
    monkeypatch.setattr(extract_pipeline, "config_object", lambda _: config)
    monkeypatch.setattr(extract_pipeline, "logging_config", lambda: None)
    monkeypatch.setattr(extract_pipeline, "remove_images_already_processed", lambda *args: (_ for _ in ()).throw(OSError("checkpoint")))
    monkeypatch.setattr(extract_pipeline.glob, "glob", lambda pattern: [str(image)])
    monkeypatch.setattr(extract_pipeline, "retrieve_orthophoto_paths", lambda _: [])
    processed = []
    monkeypatch.setattr(extract_pipeline, "process_orthophoto", lambda *args, **kwargs: processed.append(kwargs))
    monkeypatch.setattr(
        extract_pipeline,
        "validate_extract_output",
        lambda _: {"ok": False, "schema_issues": [{"file": "bad"}], "range_issues": ["bad range"], "files_without_plot_id": ["x"]},
    )
    monkeypatch.setattr("sys.argv", ["extract_data", "--config", "dummy", "--no-polygon", "--alignment"])
    extract_pipeline.main()
    assert processed and processed[0]["polygon_filtering"] is False and processed[0]["alignment"] is True

    monkeypatch.setattr(extract_pipeline, "remove_images_already_processed", lambda *args: [])
    monkeypatch.setattr(
        extract_pipeline,
        "validate_extract_output",
        lambda _: {"ok": True, "n_files": 0, "schema_issues": [], "range_issues": [], "files_without_plot_id": []},
    )
    monkeypatch.setattr(extract_pipeline, "glob", SimpleNamespace(glob=lambda _: []))
    monkeypatch.setattr("sys.argv", ["extract_data", "--config", "dummy"])
    extract_pipeline.main()


def test_filtering_pipeline_main_success_and_empty(tmp_path, monkeypatch):
    config = SimpleNamespace(
        main_extract_out=tmp_path,
        main_extract_out_polygons_df=tmp_path / "plots",
    )
    monkeypatch.setattr(filtering_pipeline, "config_object", lambda _: config)
    monkeypatch.setattr(filtering_pipeline, "logging_config", lambda: None)
    monkeypatch.setattr("sys.argv", ["filtering", "--config", "dummy"])
    monkeypatch.setattr(filtering_pipeline.glob, "glob", lambda _: [])
    with pytest.raises(RuntimeError, match="No parquet"):
        filtering_pipeline.main()

    source = tmp_path / "one.parquet"
    source.write_bytes(b"placeholder")
    captured = {}
    monkeypatch.setattr(filtering_pipeline.glob, "glob", lambda _: [str(source)])
    monkeypatch.setattr(filtering_pipeline, "load_by_polygon", lambda *args, **kwargs: captured.update(kwargs) or "done")
    filtering_pipeline.main()
    assert len(captured["derived_columns"]) == 2


def test_modelling_pipeline_main_with_public_synthetic_polygon(tmp_path, monkeypatch):
    polygon_path = tmp_path / "polygons.gpkg"
    gpd.GeoDataFrame(
        {"cultivar": ["A"], "trt": ["untreated"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:32632",
    ).to_file(polygon_path, driver="GPKG")
    config = SimpleNamespace(base_path=str(tmp_path), bands=1, main_polygon_path=str(polygon_path))
    plot_path = tmp_path / "week1" / "plot_0.parquet"
    plot_path.parent.mkdir()
    plot_path.write_bytes(b"placeholder")
    monkeypatch.setattr(modelling_pipeline, "config_object", lambda _: config)
    monkeypatch.setattr(modelling_pipeline, "logging_config", lambda: None)
    monkeypatch.setattr(modelling_pipeline, "optimized_recursive_search", lambda *args, **kwargs: {"week1": [str(plot_path)]})
    stats_calls = []
    monkeypatch.setattr(modelling_pipeline, "process_weekly_data_stats", lambda *args, **kwargs: stats_calls.append(args))
    monkeypatch.setattr(
        modelling_pipeline,
        "process_weekly_data_rpv",
        lambda *args, **kwargs: pl.DataFrame({"geometry": ["POINT (0 0)"], "rho0": [0.1], "status": ["success"]}),
    )

    monkeypatch.setattr("sys.argv", ["modelling", "--config", "dummy", "--band", "band1", "--base-dir", str(tmp_path)])
    modelling_pipeline.main()
    assert (tmp_path / "RPV_Results" / "V12" / "week1" / "rpv_week1_band1_results.csv").exists()
    assert (tmp_path / "RPV_Results" / "V12" / "rpv_results.csv").exists()
    assert stats_calls

    monkeypatch.setattr("sys.argv", ["modelling", "--config", "dummy", "--band", "0", "--base-dir", str(tmp_path)])
    modelling_pipeline.main()


def _stats_frame(n=12):
    rng = np.random.default_rng(4)
    return pl.DataFrame(
        {
            "sunelev": rng.uniform(30, 60, n),
            "saa": rng.uniform(0, 360, n),
            "vaa": rng.uniform(0, 360, n),
            "vza": rng.uniform(0, 75, n),
            "OSAVI": rng.uniform(0, 1, n),
            "NDVI": rng.uniform(0, 1, n),
            "excess_green": rng.uniform(-0.2, 0.2, n),
            **{f"band{i}": rng.uniform(0.1, 0.9, n) for i in range(1, 6)},
        }
    )


def test_stats_processing_plot_anova_and_skip_paths(tmp_path, monkeypatch):
    out = tmp_path / "stats"
    frame = _stats_frame()
    angle_calls = []

    def fake_angle(*args, **kwargs):
        angle_calls.append(kwargs["band"])
        Path(kwargs["out"]).write_bytes(b"plot")

    monkeypatch.setattr(stats_plotting, "angle_kde_plot", fake_angle)
    from oncerco_uav.stats import ANOVA as anova_module

    monkeypatch.setattr(anova_module, "ANOVA_preprocess", lambda dg, **kwargs: dg)
    monkeypatch.setattr(anova_module, "ANOVA_optimized", lambda dg, **kwargs: pl.DataFrame({"p_value": [0.1]}))
    monkeypatch.setattr(
        stats_processing,
        "plotting_raster",
        lambda dg, root, name, **kwargs: (root / "bands_data").mkdir(parents=True, exist_ok=True) or (root / "bands_data" / f"panels_{name}.png").write_bytes(b"panel"),
    )
    stats_processing.process_stats(frame, tmp_path / "plot.parquet", "week1", out)
    assert len(angle_calls) == 5
    assert (out / "anova" / "anova_results_plot.csv").exists()
    assert (out / "bands_data" / "panels_plot.png").exists()
    stats_processing.process_stats(frame, tmp_path / "plot.parquet", "week1", out)
    assert len(angle_calls) == 5


def test_stats_logistic_and_weekly_orchestration(tmp_path, monkeypatch):
    healthy = tmp_path / "healthy.parquet"
    diseased = tmp_path / "diseased.parquet"
    frame = _stats_frame(8)
    frame.write_parquet(healthy)
    frame.write_parquet(diseased)
    gdf = pl.DataFrame(
        {
            "ifz_id": [90013, 90001, 90014],
            "cult": ["A", "A", "B"],
            "trt": ["yes", "no", "yes"],
            "geometry": [None, None, None],
            "paths": [str(healthy), str(diseased), str(healthy)],
            "value": [1.0, 1.0, 0.0],
        }
    )
    monkeypatch.setattr(stats_processing, "df_preprocess", lambda dg, debug=False: dg)
    monkeypatch.setattr(stats_processing, "preprocess_healthy_diseased", lambda h, d, **kwargs: pd.DataFrame({"status": ["healthy", "diseased"], "band5": [0.2, 0.8]}))
    monkeypatch.setattr(stats_processing, "format_logistic_results", lambda *args, **kwargs: pl.DataFrame({"section": ["AUROC"], "value": [0.8]}))
    monkeypatch.setattr(logistic_module, "logistic_regression", lambda data: {"AUROC_metrics": {"AUROC_nadir": 0.8}, "Effect_size": {}})
    stats_processing.process_logistic_regression(tmp_path / "out", "week1", gdf)
    assert list((tmp_path / "out" / "logistic_regression").glob("*.csv"))

    process_calls = []
    def fake_stats(dg, path, week, out):
        process_calls.append(path)
        if len(process_calls) == 2:
            raise RuntimeError("synthetic stats failure")

    monkeypatch.setattr(stats_processing, "process_logistic_regression", lambda *args, **kwargs: None)
    monkeypatch.setattr(stats_processing, "process_stats", fake_stats)
    weekly = gdf.with_columns(pl.Series("ifz_id", [1, 2, 3], dtype=pl.Int64)).with_columns(
        pl.Series("paths", [str(healthy), str(diseased), None])
    )
    stats_processing.process_weekly_data_stats(
        {"week1": weekly}, tmp_path / "weekly", filter={"column": "band1", "sign": "<", "threshold": 2}
    )
    assert len(process_calls) == 2


def test_stats_angle_plot_and_filter_visualization_branches(tmp_path):
    frame = pl.DataFrame(
        {
            "band1": [0.1] * 6 + [0.2] * 6,
            "vza": [10.0] * 6 + [30.0] * 6,
        }
    )
    output = tmp_path / "angle.png"
    stats_plotting.angle_kde_plot(
        frame,
        band="band1",
        bins=[(0, 20), (20, 40), (40, 60)],
        angle="vza",
        xlim=(0, 1),
        points=32,
        linewidth=1,
        colors=["red", "blue"],
        dpi=20,
        out=output,
    )
    assert output.exists()
    stats_plotting.angle_kde_plot(
        frame, "band1", [(0, 20)], "vza", None, 16, 1, None, 20, out=None
    )

    values = pl.DataFrame({"Xw": [0.0, 1.0], "Yw": [0.0, 1.0], "value": [0.1, 0.9]})
    filters_module.plot_heatmap(values, "value", None, sample_size=2)
    filters_module.add_mask_and_plot(values, "value", 0.5, above=False, output_path=None)
    with pytest.raises(ValueError, match="Missing required columns"):
        filters_module.plot_spectrogram(values, 1, [475])
    with pytest.raises(ValueError, match="Number of wavelengths"):
        filters_module.plot_spectrogram(pl.DataFrame({"band1": [0.1]}), 1, [475, 560])
    filters_module.plot_spectrogram(
        pl.DataFrame({f"band{i}": np.linspace(0.1, 0.9, 5) for i in range(1, 3)}),
        2,
        [475, 560],
        sample_size=3,
        output_path=str(tmp_path),
    )


def test_merge_analysis_alignment_visualization_and_failure(tmp_path, monkeypatch):
    band_path = tmp_path / "band.tif"
    dem_path = tmp_path / "dem.tif"
    _write_test_raster(band_path, width=4, height=4, count=1)
    _write_test_raster(dem_path, width=3, height=3, count=1)
    sampled = merge_module.sample_dem_at_band_pixels(band_path, dem_path)
    assert sampled.shape == (3, 3)

    bands = pl.DataFrame({"Xw": [0.001, 1.004, 2.0], "Yw": [0.0, 1.0, 2.0], "band1": [1.0, 2.0, 3.0]})
    dem = pl.DataFrame({"Xw": [0.0, 1.0, 5.0], "Yw": [0.0, 1.0, 5.0], "elev": [10.0, 11.0, 12.0]})
    folder = tmp_path / "alignments"
    (folder / "coordinate_alignment_2.png").parent.mkdir(parents=True)
    (folder / "coordinate_alignment_2.png").write_bytes(b"old")
    alignment = merge_module.visualize_coordinate_alignment(dem, bands, precision=2, folder_name=str(folder))
    assert alignment["common_points"] == 2
    assert alignment["saved_index"] == 3
    assert (folder / "coordinate_alignment_3.png").exists()

    monkeypatch.setattr(merge_module, "reproject_dem_to_band_grid_single", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("reprojection failed")))
    with pytest.raises(RuntimeError, match="reprojection failed"):
        merge_module.merge_data(bands, band_path, dem_path, debug=True)


def test_polygon_crs_timeout_sampling_and_error_branches(tmp_path, monkeypatch):
    polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    projected = gpd.GeoDataFrame({"id": ["p"]}, geometry=[polygon], crs="EPSG:4326").to_crs("EPSG:32632")
    projected_path = tmp_path / "projected.gpkg"
    projected.to_file(projected_path, driver="GPKG")
    assert polygon_module.is_pos_inside_polygon(1, 1, {"Polygon_path": str(projected_path)})

    with pytest.raises(TypeError):
        polygon_module.is_pos_inside_polygon(None, 1, {"Polygon_path": str(projected_path)})
    real_read_file = polygon_module.gpd.read_file
    monkeypatch.setattr(polygon_module.gpd, "read_file", lambda _: (_ for _ in ()).throw(RuntimeError("read failed")))
    with pytest.raises(RuntimeError, match="read failed"):
        polygon_module.is_pos_inside_polygon(1, 1, {"Polygon_path": str(projected_path)})
    monkeypatch.setattr(polygon_module.gpd, "read_file", real_read_file)
    polygon_module.setup_timeout(1)
    polygon_module.disable_timeout()

    no_crs = gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs=None)
    geographic_wrong = gpd.GeoDataFrame({"id": [1]}, geometry=[Polygon([(1000, 1000), (1002, 1000), (1002, 1002), (1000, 1002)])], crs="EPSG:4326")
    other_crs = gpd.GeoDataFrame({"id": [1]}, geometry=[polygon], crs="EPSG:3857")
    monkeypatch.setattr(polygon_module.gpd, "read_file", lambda path: {"none": no_crs, "wrong": geographic_wrong, "other": other_crs}[str(path)])
    assert polygon_module.load_and_prepare_polygons("none", "EPSG:32632").crs.to_string() == "EPSG:32632"
    assert polygon_module.load_and_prepare_polygons("wrong", "EPSG:32632").crs.to_string() == "EPSG:32632"
    assert polygon_module.load_and_prepare_polygons("other", "EPSG:32632").crs.to_string() == "EPSG:32632"

    poly_with_ids = gpd.GeoDataFrame({"id": ["id1"], "plot_id": ["plot1"]}, geometry=[polygon], crs="EPSG:32632")
    data_bounds = [0, 0, 2, 2]
    filtered = gpd.GeoDataFrame(
        {"Xw": [0.5, 1.5], "Yw": [0.5, 1.5], "band1": [1.0, 0.0], "plot_id": ["id1", "id1"]},
        geometry=gpd.points_from_xy([0.5, 1.5], [0.5, 1.5]),
        crs="EPSG:32632",
    )
    polygon_module.plot_results(poly_with_ids, filtered, "EPSG:32632", "ids", data_bounds, plots_out=tmp_path, img_name="ids")
    polygon_module.plot_results(
        gpd.GeoDataFrame({"plot_id": ["plot1"]}, geometry=[polygon], crs="EPSG:32632"),
        filtered,
        "EPSG:32632",
        "plotid",
        data_bounds,
        plots_out=tmp_path,
        img_name="plotid",
    )
    polygon_module.plot_results(
        gpd.GeoDataFrame({"other": ["x"]}, geometry=[polygon], crs="EPSG:32632"),
        filtered,
        "EPSG:32632",
        "fallback",
        data_bounds,
        plots_out=tmp_path,
        img_name="fallback",
    )
    large = filtered.iloc[[0] * 6].copy()
    polygon_module.plot_results(poly_with_ids, large, "EPSG:32632", "pandas", data_bounds, sample_for_debug=2, plots_out=tmp_path, img_name="pandas")
    large_geo = gpd.GeoDataFrame(large, geometry=gpd.points_from_xy(large.Xw, large.Yw), crs="EPSG:32632")
    polygon_module.plot_results(poly_with_ids, large_geo, "EPSG:32632", "geo", data_bounds, sample_for_debug=2, plots_out=tmp_path, img_name="geo")

    # The function intentionally catches malformed point frames and returns None.
    assert polygon_module.process_chunk(
        (0, 1), pl.DataFrame({"bad": [1]}), poly_with_ids, "EPSG:32632"
    ) is None

    monkeypatch.setattr(polygon_module, "process_chunk", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("chunk failed")))
    chunks, processed_count, _ = polygon_module.process_chunks_parallel(
        pl.DataFrame({"Xw": [1.0], "Yw": [1.0]}), [(0, 1)], 1, poly_with_ids, "EPSG:32632", "id", 1
    )
    assert processed_count == 1 and chunks == []
    assert polygon_module.combine_chunk_results([object()], 1, 1.0) is None

    monkeypatch.setattr(polygon_module.plt, "savefig", lambda *args, **kwargs: None)
    polygon_module.plot_no_overlap(poly_with_ids, [10, 10, 11, 11], plots_out=None, img_name="none")


def _logistic_input(n=60, with_groups=False):
    rng = np.random.default_rng(9)
    status = np.where(np.arange(n) % 2 == 0, "healthy", "diseased")
    vza = np.where(np.arange(n) < n // 2, 10.0, 45.0) + rng.normal(0, 1, n)
    frame = pd.DataFrame(
        {
            "status": status,
            "band5": rng.uniform(0.1, 0.9, n),
            "vza": vza,
            "raa": rng.uniform(-160, 160, n),
        }
    )
    if with_groups:
        # Each group contains both classes so GroupKFold has a valid training
        # and validation class distribution.
        frame["group"] = (np.arange(n) // 2) % 10
    return frame


def test_logistic_validation_geometry_group_and_nested_paths():
    frame = _logistic_input()
    with pytest.raises(KeyError, match="Requested bands"):
        logistic_module._auroc_fast(frame, bands=("band1",))
    with pytest.raises(KeyError, match="required columns"):
        logistic_module._auroc_fast(frame.drop(columns=["raa"]))
    with pytest.raises(KeyError, match="group_col"):
        logistic_module._auroc_fast(frame, group_col="missing")

    result = logistic_module._auroc_fast(
        frame,
        same_size=False,
        geometry_match=False,
        n_splits=2,
        v_bins=2,
        r_bins=2,
    )
    assert 0 <= result["AUROC_main"] <= 1

    grouped = logistic_module._auroc_fast(
        _logistic_input(with_groups=True),
        group_col="group",
        geometry_match=False,
        same_size=True,
        n_splits=2,
        v_bins=2,
        r_bins=2,
    )
    assert grouped["settings"]["group_col"] == "group"

    nested = logistic_module._auroc_fast(
        _logistic_input(72),
        geometry_match=False,
        same_size=False,
        nested=True,
        n_splits=3,
        v_bins=2,
        r_bins=2,
        Cs=(1.0,),
    )
    assert 0 <= nested["AUROC_full"] <= 1


def test_logistic_effect_size_and_result_normalization_edges():
    only_diseased = pd.DataFrame(
        {"status": ["diseased", "diseased"], "band5": [0.5, 0.6], "vza": [10, 10], "vza_bin": ["0-20", "0-20"], "raa_bin": ["0-90", "0-90"]}
    )
    assert np.isnan(logistic_module._calculate_cohens_d(only_diseased)["Cohen_d_nadir"])
    one_each = only_diseased.assign(status=["healthy", "diseased"])
    assert np.isnan(logistic_module._calculate_cohens_d(one_each)["Cohen_d_nadir"])
    zero_variance = pd.DataFrame(
        {
            "status": ["healthy", "diseased"],
            "band5": [0.5, 0.5],
            "vza": [10, 10],
            "vza_bin": ["0-20", "0-20"],
            "raa_bin": ["0-90", "0-90"],
        }
    )
    assert np.isnan(logistic_module._calculate_cohens_d(zero_variance)["Cohen_d_nadir"])

    assert logistic_module._extract_result_dict({"AUROC_metrics": {}, "Effect_size": {}})["AUROC_metrics"] == {}
    struct = pl.DataFrame({"AUROC_metrics": [{"AUROC_nadir": 0.5}], "Effect_size": [{}]})
    assert logistic_module._extract_result_dict(struct)["AUROC_metrics"]["AUROC_nadir"] == 0.5
    with pytest.raises(ValueError, match="single-row"):
        logistic_module._extract_result_dict(pl.DataFrame({"x": [1, 2]}))
    with pytest.raises(TypeError, match="dict or"):
        logistic_module._extract_result_dict([1])

    malformed = {
        "AUROC_metrics": {"bad": "not numeric"},
        "Effect_size": {
            "Cohen_d_nadir": "bad",
            "top_bins_by_|d|": {"vza_bin=0-20_raa_bin=0-90": "bad", "other": 1.0},
        },
    }
    long = logistic_module.format_logistic_results(malformed, shape="long")
    assert long.height == 1
    wide = logistic_module.format_logistic_results(
        {"AUROC_metrics": {"AUROC_nadir": 0.5}, "Effect_size": {"Cohen_d_nadir": "bad"}},
        shape="wide",
    )
    assert wide["Cohen_d_nadir"][0] is None


def test_remaining_small_error_and_fallback_paths(tmp_path, monkeypatch):
    from datetime import datetime

    from oncerco_uav.core.config_object import AttrDict
    from oncerco_uav.core.logging import logging_config
    from oncerco_uav.extract.date_time import convert_to_timezone

    assert repr(AttrDict({"answer": 42})) == "AttrDict({'answer': 42})"
    monkeypatch.chdir(tmp_path)
    logging_config()
    assert (tmp_path / "process.log").exists()
    assert convert_to_timezone(datetime(2024, 1, 1), "not-a-timezone").tzinfo is not None

    week_file = tmp_path / "some_week7_plot.parquet"
    week_file.write_bytes(b"placeholder")
    found = search_module.optimized_recursive_search([], "plot", str(tmp_path))
    assert found["week7"] == [str(week_file)]

    with pytest.raises(Exception):
        camera_module.calculate_angles(pl.DataFrame({}), 0, 0, 1, 30, 180)

    sample_path = tmp_path / "sample.parquet"
    pl.DataFrame({"value": [1.0, 2.0]}).write_parquet(sample_path)
    monkeypatch.setattr(modelling_processing, "df_preprocess", lambda dg, debug=False: dg)
    monkeypatch.setattr(modelling_processing, "rpv_fit", lambda dg, **kwargs: (1, 2, 3, 4, 5, 6))
    rows = pl.DataFrame(
        {
            "ifz_id": [1, 2],
            "cult": ["A", "B"],
            "trt": ["yes", "no"],
            "geometry": [None, None],
            "paths": [None, str(sample_path)],
            "value": [0.0, 1.0],
        }
    )
    result = modelling_processing.process_weekly_data_rpv(
        {"week": rows}, "band5", sample_total_dataset=10, filter={"column": "value", "sign": "<", "threshold": 2}
    )
    assert result.height == 1

    invalid_rpv = pl.DataFrame(
        {"vza": [10.0], "band5": [2.0], "sza": [20.0], "raa": [0.0]}
    )
    from oncerco_uav.modelling import rpv as rpv_module

    with pytest.raises(ValueError, match="No valid samples"):
        rpv_module.rpv_fit(invalid_rpv, band="band5", n_samples_bins=1)

    class AlwaysFail:
        def write_parquet(self, *args, **kwargs):
            raise RuntimeError("all codecs failed")

    with pytest.raises(RuntimeError, match="all codecs failed"):
        extract_pipeline.save_parquet(AlwaysFail(), tmp_path, {"name": "run"}, 0, "bad.tif")

    original_kde = stats_plotting._kde1d_fast
    monkeypatch.setattr(stats_plotting, "_kde1d_fast", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("kde failed")))
    stats_plotting.angle_kde_plot(
        pl.DataFrame({"band1": [0.1] * 6, "vza": [10.0] * 6}),
        "band1",
        [(0, 20)],
        "vza",
        (0, 1),
        10,
        1,
        None,
        20,
        out=tmp_path / "failed_kde.png",
    )
    monkeypatch.setattr(stats_plotting, "_kde1d_fast", original_kde)
