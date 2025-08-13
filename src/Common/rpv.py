import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error


def rpv_1(angle_pack, rho0, k, theta, rc=1.0):
    s, v, rphi = np.radians(angle_pack)  # θi, θr, Δφ  (deg → rad)
    cs, cv = np.cos(s), np.cos(v)
    sin_s, sin_v = np.sin(s), np.sin(v)

    g = np.arccos(cs * cv + sin_s * sin_v * np.cos(rphi))  # phase angle
    # Corrected phase kernel calculation
    F = (1 - theta**2) / (1 + theta**2 - 2 * theta * np.cos(g)) ** 1.5
    G = np.sqrt(np.tan(s) ** 2 + np.tan(v) ** 2 - 2 * np.tan(s) * np.tan(v) * np.cos(rphi))
    hot = 1 + (1 - rc) / (1 + G)  # hotspot (ρc=1 ⇒ disabled)

    return rho0 * (cs ** (k - 1)) * (cv ** (k - 1)) * (cs + cv) ** (-k) * F * hot


def rpv_2(angle_pack, rho0, k, theta, rc=1.0):  # 1  keep hotspot neutral
    s, v, dphi = np.radians(angle_pack)
    cs, cv = np.cos(s), np.cos(v)
    sin_s, sin_v = np.sin(s), np.sin(v)

    g = np.arccos(np.clip(cs * cv + sin_s * sin_v * np.cos(dphi), -1.0, 1.0))

    F = (1 - theta**2) / (1 + theta**2 + 2 * theta * np.cos(g)) ** 1.5
    G = np.sqrt(np.tan(s) ** 2 + np.tan(v) ** 2 - 2 * np.tan(s) * np.tan(v) * np.cos(dphi))
    hot = 1 + (1 - rc) / (1 + G)  # 3  deactivate unless rc is freed

    return rho0 * (cs ** (k - 1)) * (cv ** (k - 1)) * (cs + cv) ** (-k) * F * hot


def rpv_side(angle_pack, rho0, k, theta, sigma, rc=1.0):
    s, v, dphi = np.radians(angle_pack)
    cs, cv = np.cos(s), np.cos(v)
    g = np.arccos(np.clip(cs * cv + np.sin(s) * np.sin(v) * np.cos(dphi), -1, 1))
    F = (1 - theta**2) / (1 + theta**2 + 2 * theta * np.cos(g)) ** 1.5
    G = np.sqrt(np.tan(s) ** 2 + np.tan(v) ** 2 - 2 * np.tan(s) * np.tan(v) * np.cos(dphi))
    H = 1 + (1 - rc) / (1 + G)
    S = 1 + sigma * (np.sin(g / 2) ** 2) * (np.sin(dphi) ** 2)  # ← new bit
    return rho0 * (cs ** (k - 1)) * (cv ** (k - 1)) * (cs + cv) ** (-k) * F * H * S


def rpv_df_preprocess(df, debug=False):
    EPS = 1e-2  # Tolerance for floating point comparison

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
    vaa_formula = np.degrees(np.arctan2(df["vx"], df["vy"])) % 360
    sza_formula = 90 - df["sunelev"]
    ndvi_formula = (df["band5"] - df["band3"]) / (df["band5"] + df["band3"])
    raa_formula = np.abs(df["saa"] - vaa_formula) % 360
    raa_formula = np.where(raa_formula <= 180, raa_formula, 360 - raa_formula)

    # Check or create columns
    for col, formula in [
        ("vza", vza_formula),
        ("vaa", vaa_formula),
        ("sza", sza_formula),
        ("NDVI", ndvi_formula),
        ("raa", raa_formula),
    ]:
        if col in df.columns and debug == True:
            if not np.allclose(df[col], formula, atol=EPS):
                print(f"Column '{col}' values do not match formula!")
                df = df.with_columns(pl.Series(col, formula))
        else:
            df = df.with_columns(pl.Series(col, formula))
    return df


def rpv_fit(df, band, n_samples_bins):
    """
    Fit the RPV model to spectral data.

    Args:
        df: DataFrame with angle information
        band: Spectral band name to use for fitting

    Returns:
        Tuple of (rho0, k, theta, rc, rmse, nrmse)
    """
    edges = np.array([0, 15, 25, 35, 45, 60])  # VZA bins
    bins = np.digitize(df["vza"], edges, right=False) - 1  # 0-based
    df = df.with_columns(pl.Series("bin", bins))

    frames: list[pl.DataFrame] = []
    for b in range(len(edges) - 1):
        subset = df.filter((pl.col("bin") == b) & (pl.col(band).is_between(0, 1)))
        if subset.height > 0:
            n_take = min(n_samples_bins, subset.height)
            frames.append(subset.sample(n=n_take, seed=42))
    if not frames:
        raise ValueError("No valid samples available for RPV fitting")
    df_fit = pl.concat(frames)

    sza, vza, raa, R = [df_fit[col].to_numpy() for col in ["sza", "vza", "raa", band]]
    mask = np.isfinite(sza) & np.isfinite(vza) & np.isfinite(raa) & np.isfinite(R)

    k_prior, lam = 1, 0.05

    def resid(pars, sza, vza, raa, R):
        rho0, k, theta = pars
        data_err = rpv_2((sza, vza, raa), rho0, k, theta) - R
        prior_err = np.sqrt(lam) * (k - k_prior)
        return np.concatenate([data_err, [prior_err]])

    p0 = [np.median(R), 0.1, 0]  # bowl-centred
    bounds = ([1e-3, 0.0, -1], [1.1, 3, 1])

    res = least_squares(
        resid,
        p0,
        bounds=bounds,
        args=(sza, vza, raa, R),
        loss="cauchy",
        jac="3-point",
        max_nfev=30_000,
    )
    # print("Median R", np.median(R))

    rho0, k, theta = res.x
    rc = 1.0

    R_hat = rpv_2((sza, vza, raa), rho0, k, theta, rc)
    rmse = np.sqrt(mean_squared_error(R, R_hat))
    nrmse = rmse / R.mean()

    return rho0, k, theta, rc, rmse, nrmse
