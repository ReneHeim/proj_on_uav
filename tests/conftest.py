import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
import rasterio as rio
from rasterio.transform import from_origin


@pytest.fixture(autouse=True)
def _no_plot_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_reflectance_df():
    return pl.DataFrame({
        "Xw": [1.0, 2.0, 3.0, 4.0, 5.0],
        "Yw": [1.0, 2.0, 3.0, 4.0, 5.0],
        "band1": [0.1, 0.15, 0.2, 0.25, 0.3],
        "band2": [0.2, 0.25, 0.3, 0.35, 0.4],
        "band3": [0.15, 0.2, 0.25, 0.3, 0.35],
        "band4": [0.3, 0.35, 0.4, 0.45, 0.5],
        "band5": [0.4, 0.45, 0.5, 0.55, 0.6],
        "elev": [100.0, 101.0, 102.0, 103.0, 104.0],
        "sunelev": [40.0, 40.0, 40.0, 40.0, 40.0],
        "saa": [180.0, 180.0, 180.0, 180.0, 180.0],
        "vza": [10.0, 20.0, 30.0, 40.0, 50.0],
        "vaa": [90.0, 95.0, 100.0, 105.0, 110.0],
        "xcam": [0.0, 0.0, 0.0, 0.0, 0.0],
        "ycam": [0.0, 0.0, 0.0, 0.0, 0.0],
        "delta_z": [100.0, 101.0, 102.0, 103.0, 104.0],
    })


@pytest.fixture
def temp_raster(tmp_path):
    raster_path = tmp_path / "test_raster.tif"
    width, height = 10, 10
    data = np.arange(width * height, dtype=np.float32).reshape(1, height, width)
    transform = from_origin(500000.0, 5000000.0, 1.0, 1.0)
    with rio.open(
        raster_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=transform,
    ) as dst:
        dst.write(data)
    return raster_path
