import polars as pl

from src.Utils.extract_data.camera import calculate_angles


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
