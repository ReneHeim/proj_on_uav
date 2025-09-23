import numpy as np
import pandas as pd
import polars as pl
import logging
from tqdm import tqdm


def df_preprocess(df, debug=False, load_indeces=False):
    EPS = 1e-2  # Tolerance for floating point comparison

    # Drop NaN and nulls
    len_before = len(df)
    df = df.fill_nan(None)
    df = df.drop_nulls()
    logging.info(
        f"Dropped {len_before - len(df)} NaN and nulls the: {round((len_before - len(df))/len_before * 100,3)} %"
    )

    if "vx" in df.columns:
        vx = df["vx"]
        assert np.allclose(vx, df["xcam"] - df["Xw"], atol=EPS), "vx mismatch"
    else:
        vx = df["xcam"] - df["Xw"]
        df = df.with_columns(pl.Series("vx", vx))

    if "vy" in df.columns:
        vy = df["vy"]
        assert np.allclose(vy, df["ycam"] - df["Yw"], atol=EPS), "vy mismatch"
    else:
        vy = df["ycam"] - df["Yw"]
        df = df.with_columns(pl.Series("vy", vy))

    if "vz" in df.columns:
        vz = df["vz"]
        assert np.allclose(vz, df["delta_z"], atol=EPS), "vz mismatch"
    else:
        vz = df["delta_z"]
        df = df.with_columns(pl.Series("vz", vz))

    if "v_norm" in df.columns:
        v_norm = df["v_norm"]
        assert np.allclose(v_norm, np.sqrt(vx**2 + vy**2 + vz**2), atol=EPS), "v_norm mismatch"
    else:
        v_norm = np.sqrt(vx**2 + vy**2 + vz**2)
        df = df.with_columns(pl.Series("v_norm", v_norm))

    # Formulas as expressions
    cos_vza = (df["vz"] / (df["v_norm"] + 1e-12)).clip(-1.0, 1.0)  # 1  guard /0 and range
    vza_formula = np.degrees(np.arccos(cos_vza))
    vaa_formula = np.where(
        (df["vx"].to_numpy() == 0) & (df["vy"].to_numpy() == 0),
        np.nan,
        np.degrees(np.arctan2(df["vx"].to_numpy(), df["vy"].to_numpy())) % 360,
    )

    sza_formula = 90 - df["sunelev"]

    raa_formula = np.abs(df["saa"] - vaa_formula) % 360
    raa_formula = np.where(raa_formula <= 180, raa_formula, 360 - raa_formula)

    # indexes
    ndvi_formula = (df["band5"] - df["band3"]) / (df["band5"] + df["band3"])
    excess_green_formula = 2 * df["band2"] - df["band3"] - df["band1"]

    Y = 0.16
    osavi_formula = (1 + Y) * (df["band5"] - df["band3"]) / (df["band5"] + df["band3"] + Y)
    # if load_indeces:

    # Check or create columns
    for col, formula in [
        ("vza", vza_formula),
        ("vaa", vaa_formula),
        ("sza", sza_formula),
        ("NDVI", ndvi_formula),
        ("raa", raa_formula),
        ("excess_green", excess_green_formula),
        ("OSAVI", osavi_formula),
    ]:
        if col in df.columns and debug == True:
            if not np.allclose(df[col], formula, atol=EPS):
                print(f"Column '{col}' values do not match formula!")
                df = df.with_columns(pl.Series(col, formula))
        else:
            df = df.with_columns(pl.Series(col, formula))
    return df
