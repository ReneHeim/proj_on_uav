import numpy as np
import polars as pl
import pytest
import rasterio
from rasterio.transform import from_origin

from src.extract.camera import calculate_angles, get_camera_position

# ---------------------------------------------------------------------------
# Value correctness
# ---------------------------------------------------------------------------


def test_get_camera_position_uses_nearest_duplicate(tmp_path):
    camera_path = tmp_path / "cameras.txt"
    camera_path.write_text(
        "# Cameras (2)\n"
        "# PhotoID, X, Y, Z, Omega, Phi, Kappa, r11, r12, r13, r21, r22, r23, r31, r32, r33\n"
        "IMG_0001_6\t9.0\t51.0\t100\t0\t0\t0\t1\t0\t0\t0\t1\t0\t0\t0\t1\n"
        "IMG_0001_6\t10.0\t52.0\t200\t0\t0\t0\t1\t0\t0\t0\t1\t0\t0\t0\t1\n",
        encoding="utf-8",
    )
    raster_path = tmp_path / "IMG_0001_6.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(9.99, 52.01, 0.002, 0.002),
    ) as dst:
        dst.write(np.ones((1, 10, 10), dtype=np.float32))

    x, y, z = get_camera_position(
        camera_path,
        "IMG_0001_6",
        orthophoto_path=raster_path,
    )

    assert x == pytest.approx(10.0)
    assert y == pytest.approx(52.0)
    assert z == pytest.approx(200.0)


def test_calculate_angles_basic():
    df = pl.DataFrame(
        {
            "Xw": [0.0, 1.0],
            "Yw": [0.0, 1.0],
            "elev": [0.0, 0.0],
            "band1": [1000, 1000],
        }
    )
    out = calculate_angles(df, xcam=0.0, ycam=0.0, zcam=10.0, sunelev=30.0, saa=150.0)
    for col in ["delta_x", "delta_y", "delta_z", "vza", "vaa"]:
        assert col in out.columns


def test_calculate_angles_vza_values():
    df = pl.DataFrame(
        {
            "Xw": [0.0, 10.0],
            "Yw": [0.0, 0.0],
            "elev": [0.0, 0.0],
            "band1": [1000, 1000],
        }
    )
    out = calculate_angles(df, xcam=0.0, ycam=0.0, zcam=10.0, sunelev=30.0, saa=150.0)

    vza_values = out["vza"].to_list()
    # Point at (0,0,0): distance_xy=0, delta_z=10, arctan2(0,10)=0 → vza≈0
    assert vza_values[0] == pytest.approx(0.0, abs=0.1)
    # Point at (10,0,0): distance_xy=10, delta_z=10, arctan2(10,10)=π/4 → vza≈45
    assert vza_values[1] == pytest.approx(45.0, abs=0.1)


def test_calculate_angles_vaa_in_range():
    df = pl.DataFrame(
        {
            "Xw": [10.0, 0.0, -10.0, 0.0],
            "Yw": [0.0, 10.0, 0.0, -10.0],
            "elev": [0.0, 0.0, 0.0, 0.0],
            "band1": [1000, 1000, 1000, 1000],
        }
    )
    out = calculate_angles(df, xcam=0.0, ycam=0.0, zcam=10.0, sunelev=30.0, saa=150.0)
    vaa = out["vaa"].to_list()
    for v in vaa:
        assert 0.0 <= v <= 360.0


def test_calculate_angles_adds_columns():
    df = pl.DataFrame(
        {
            "Xw": [0.0],
            "Yw": [0.0],
            "elev": [0.0],
            "band1": [1000],
        }
    )
    out = calculate_angles(df, xcam=0.0, ycam=0.0, zcam=10.0, sunelev=30.0, saa=150.0)
    for col in ["xcam", "ycam", "sunelev", "saa"]:
        assert col in out.columns
    assert out["xcam"][0] == 0.0
    assert out["ycam"][0] == 0.0
    assert out["sunelev"][0] == 30.0
    assert out["saa"][0] == 150.0


def test_calculate_angles_delta_z():
    df = pl.DataFrame(
        {
            "Xw": [0.0, 0.0],
            "Yw": [0.0, 0.0],
            "elev": [2.0, 5.0],
            "band1": [1000, 1000],
        }
    )
    out = calculate_angles(df, xcam=0.0, ycam=0.0, zcam=10.0, sunelev=30.0, saa=150.0)
    delta_z = out["delta_z"].to_list()
    assert delta_z[0] == pytest.approx(8.0, abs=0.01)
    assert delta_z[1] == pytest.approx(5.0, abs=0.01)


def test_calculate_angles_directly_above():
    df = pl.DataFrame(
        {
            "Xw": [0.0],
            "Yw": [0.0],
            "elev": [0.0],
            "band1": [1000],
        }
    )
    out = calculate_angles(df, xcam=0.0, ycam=0.0, zcam=100.0, sunelev=30.0, saa=150.0)
    assert out["vza"][0] == pytest.approx(0.0, abs=0.5)


def test_calculate_angles_horizon_level():
    df = pl.DataFrame(
        {
            "Xw": [10.0],
            "Yw": [0.0],
            "elev": [0.0],
            "band1": [1000],
        }
    )
    out = calculate_angles(df, xcam=0.0, ycam=0.0, zcam=0.0, sunelev=30.0, saa=150.0)
    assert out["vza"][0] == pytest.approx(90.0, abs=0.5)


def test_calculate_angles_zero_elevation():
    df = pl.DataFrame(
        {
            "Xw": [0.0, 3.0],
            "Yw": [0.0, 4.0],
            "elev": [0.0, 0.0],
            "band1": [1000, 1000],
        }
    )
    out = calculate_angles(df, xcam=0.0, ycam=0.0, zcam=5.0, sunelev=30.0, saa=150.0)
    vza0 = out["vza"][0]
    vza1 = out["vza"][1]
    assert vza0 == pytest.approx(0.0, abs=0.5)
    # delta_z=5, distance_xy=5 → arctan2(5,5) = 45°
    assert vza1 == pytest.approx(45.0, abs=0.5)


def test_calculate_angles_large_elevation_variation():
    df = pl.DataFrame(
        {
            "Xw": [0.0, 0.0, 0.0],
            "Yw": [0.0, 0.0, 0.0],
            "elev": [0.0, 50.0, 100.0],
            "band1": [1000, 1000, 1000],
        }
    )
    out = calculate_angles(df, xcam=0.0, ycam=0.0, zcam=150.0, sunelev=30.0, saa=150.0)
    # elev=0: delta_z=150, vza≈0
    # elev=100: delta_z=50, vza≈0
    assert out["delta_z"][0] == pytest.approx(150.0, abs=0.01)
    assert out["delta_z"][2] == pytest.approx(50.0, abs=0.01)


# ---------------------------------------------------------------------------
# band1==65535 masking
# ---------------------------------------------------------------------------


def test_calculate_angles_masks_65535():
    df = pl.DataFrame(
        {
            "Xw": [0.0, 10.0],
            "Yw": [0.0, 0.0],
            "elev": [0.0, 0.0],
            "band1": [65535, 1000],
        }
    )
    out = calculate_angles(df, xcam=0.0, ycam=0.0, zcam=10.0, sunelev=30.0, saa=150.0)
    assert out["vza"][0] is None
    assert out["vaa"][0] is None
    assert out["vza"][1] is not None
    assert out["vaa"][1] is not None
