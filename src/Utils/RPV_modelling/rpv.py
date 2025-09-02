import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error


def rpv_2(angle_pack, rho0, k, theta, rc=1.0):
    s, v, dphi = np.radians(angle_pack)
    cs, cv = np.cos(s), np.cos(v)
    sin_s, sin_v = np.sin(s), np.sin(v)
    g = np.arccos(np.clip(cs * cv + sin_s * sin_v * np.cos(dphi), -1.0, 1.0))
    F = (1 - theta**2) / (1 + theta**2 + 2 * theta * np.cos(g)) ** 1.5
    G = np.sqrt(np.tan(s) ** 2 + np.tan(v) ** 2 - 2 * np.tan(s) * np.tan(v) * np.cos(dphi))
    hot = 1 + (1 - rc) / (1 + G)
    return rho0 * (cs ** (k - 1)) * (cv ** (k - 1)) * (cs + cv) ** (k - 1) * F * hot


def rpv_1(angle_pack, rho0, k, theta, rc=1.0):
    s, v, dphi = np.radians(angle_pack)
    cs, cv = np.cos(s), np.cos(v)
    sin_s, sin_v = np.sin(s), np.sin(v)

    g = np.arccos(np.clip(cs * cv + sin_s * sin_v * np.cos(dphi), -1.0, 1.0))

    F = (1 - theta**2) / (1 + theta**2 + 2 * theta * np.cos(g)) ** 1.5
    G = np.sqrt(np.tan(s) ** 2 + np.tan(v) ** 2 - 2 * np.tan(s) * np.tan(v) * np.cos(dphi))
    hot = 1 + (1 - rc) / (1 + G)  # 3deactivate unless rc is freed

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

    def resid(pars, sza, vza, raa, R):
        rho0, k, theta = pars
        data_err = rpv_2((sza, vza, raa), rho0, k, theta) - R
        return np.hstack([data_err])

    p0 = [np.median(R), 0.1, 0]
    bounds = ([1e-3, 0.0, -1], [2, 3, 1])

    res = least_squares(
        resid,
        p0,
        bounds=bounds,
        args=(sza, vza, raa, R),
        loss="cauchy",
        jac="3-point",
        max_nfev=30_000,
    )

    rho0, k, theta = res.x
    rc = 1.0

    R_hat = rpv_2((sza, vza, raa), rho0, k, theta, rc)
    rmse = np.sqrt(mean_squared_error(R, R_hat))
    nrmse = rmse / R.mean()

    return rho0, k, theta, rc, rmse, nrmse
