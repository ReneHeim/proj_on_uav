import numpy as np, polars as pl, matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import mean_squared_error

def rpv_1(angle_pack, rho0, k, theta, rc=1.0):
    s, v, rphi = np.radians(angle_pack)                 # θi, θr, Δφ  (deg → rad)
    cs, cv = np.cos(s), np.cos(v); sin_s, sin_v = np.sin(s), np.sin(v)

    g  = np.arccos(cs*cv + sin_s*sin_v*np.cos(rphi))    # phase angle
    # Corrected phase kernel calculation
    F  = (1 - theta**2) / (1 + theta**2 - 2*theta*np.cos(g))**1.5
    G  = np.sqrt(np.tan(s)**2 + np.tan(v)**2 - 2*np.tan(s)*np.tan(v)*np.cos(rphi))
    hot= 1 + (1 - rc)/(1 + G)                           # hotspot (ρc=1 ⇒ disabled)

    return rho0 * (cs**(k-1)) * (cv**(k-1)) * (cs + cv)**(-k) * F * hot



def rpv_2(angle_pack, rho0, k, Theta, rc=1.0):
    """Rahman–Pinty–Verstraete BRF (canonical RAMI form)."""
    s, v, dphi = np.radians(angle_pack)              # θi, θr, Δφ  [rad]
    cs, cv   = np.cos(s), np.cos(v)
    sin_s, sin_v = np.sin(s), np.sin(v)

    g  = np.arccos(cs*cv + sin_s*sin_v*np.cos(dphi)) # phase angle
    F  = (1 - Theta**2) / (1 + Theta**2 + 2*Theta*np.cos(g))**1.5  # phase kernel
    G  = np.sqrt(np.tan(s)**2 + np.tan(v)**2 - 2*np.tan(s)*np.tan(v)*np.cos(dphi))
    hot = 1 + (1 - rc)/(1 + G)

    return rho0 * (cs**(k-1)) * (cv**(k-1)) * (cs + cv)**(-k) * F * hot



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
    vza_formula = np.degrees(np.arccos(df["vz"] / df["v_norm"]))
    vaa_formula = (np.degrees(np.arctan2(df["vx"], df["vy"])) % 360)
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
        ("raa", raa_formula)
    ]:
        if col in df.columns and  debug==True:
            if not np.allclose(df[col], formula, atol=EPS):
                print(f"Column '{col}' values do not match formula!")
                df = df.with_columns(pl.Series(col, formula))
        else:
            df = df.with_columns(pl.Series(col, formula))
    return df


def rpv_fit(df, band):
    """
    Fit the RPV model to spectral data.

    Args:
        df: DataFrame with angle information
        band: Spectral band name to use for fitting

    Returns:
        Tuple of (rho0, k, theta, rc, rmse, nrmse)
    """
    # 1. Data selection
    df_fit = (
        df.filter((pl.col(band).is_finite()) & (pl.col(band) > 0) & (pl.col(band) < 1))
        .sample(n=100000, with_replacement=False, shuffle=True, seed=42)
    )
    sza, vza, raa, R = [df_fit[col].to_numpy() for col in ["sza", "vza", "raa", band]]
    mask = np.isfinite(sza) & np.isfinite(vza) & np.isfinite(raa) & np.isfinite(R)
    sza, vza, raa, R = [x[mask] for x in (sza, vza, raa, R)]

    # 2. Define residual for a 3-parameter fit
    def rpv_res_3params(pars, sza, vza, raa, R):
        # Unpack three parameters
        rho0, k, theta = pars
        return rpv_2((sza, vza, raa), rho0, k, theta) - R

    # 3. Initial guess and bounds for 3 parameters
    p0 = [np.median(R), 0.7, -0.10]
    bounds = ([1e-3, 0.2, -0.30],
              [0.70, 0.95, 0.00])

    # 4. Robust fit with the residual function
    res = least_squares(
        rpv_res_3params, p0,
        bounds=bounds,  # Add this to use the bounds
        args=(sza, vza, raa, R),
        loss="soft_l1",
        max_nfev=30000, verbose=0
    )

    # 5. Unpack the three results (not four)
    rho0, k, theta = res.x
    rc = 1.0  # Use default value since we're not fitting this parameter

    # 6. Evaluate model performance
    R_hat = rpv_2((sza, vza, raa), rho0, k, theta, rc)
    rmse = np.sqrt(mean_squared_error(R, R_hat))
    nrmse = rmse / R.mean()

    return rho0, k, theta, rc, rmse, nrmse