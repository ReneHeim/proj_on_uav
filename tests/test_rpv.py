import polars as pl

from src.Common.preprocess import  df_preprocess
from src.Utils.RPV_modelling.rpv import rpv_fit


def test_rpv_preprocess_and_fit():
    # minimal dataframe containing required columns
    df = pl.DataFrame(
        {
            "Xw": [0.0, 1.0, 0.0, 1.0],
            "Yw": [0.0, 1.0, 1.0, 0.0],
            "delta_z": [10.0, 10.0, 10.0, 10.0],
            "xcam": [0.0, 0.0, 0.0, 0.0],
            "ycam": [0.0, 0.0, 0.0, 0.0],
            "sunelev": [30.0, 30.0, 30.0, 30.0],
            "saa": [150.0, 150.0, 150.0, 150.0],
            "band3": [0.1, 0.1, 0.1, 0.1],
            "band5": [0.2, 0.2, 0.2, 0.2],
            "band1": [0.2, 0.2, 0.2, 0.2],
            "band2": [0.2, 0.2, 0.2, 0.2],
        }
    )
    df = df_preprocess(df)
    assert all(c in df.columns for c in ["vza", "vaa", "sza", "NDVI", "raa"])

    # band reflectance within [0,1] is required by rpv_fit sampling
    rho0, k, theta, rc, rmse, nrmse = rpv_fit(df, band="band1", n_samples_bins=1)
    assert 0 < rho0 < 1
    assert 0 < k < 3
    assert -1 < theta < 1
