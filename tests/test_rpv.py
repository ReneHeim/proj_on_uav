import numpy as np
import polars as pl
import pytest

from src.core.preprocess import df_preprocess
from src.modelling.rpv import rpv_1, rpv_2, rpv_fit, rpv_side


# ---------------------------------------------------------------------------
# Existing test (kept)
# ---------------------------------------------------------------------------

def test_rpv_preprocess_and_fit():
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

    rho0, k, theta, rc, rmse, nrmse = rpv_fit(df, band="band1", n_samples_bins=1)
    assert 0 < rho0 < 1
    assert 0 < k < 3
    assert -1 < theta < 1


# ---------------------------------------------------------------------------
# rpv_1, rpv_2, rpv_side pure function tests
# ---------------------------------------------------------------------------

def test_rpv_1_basic():
    result = rpv_1((30, 0, 0), rho0=0.1, k=1.0, theta=0.0)
    assert 0 <= result <= 2


def test_rpv_2_basic():
    result = rpv_2((30, 0, 0), rho0=0.1, k=1.0, theta=0.0)
    assert 0 <= result <= 2


def test_rpv_side_basic():
    result = rpv_side((30, 0, 0), rho0=0.1, k=1.0, theta=0.0, sigma=0.0)
    assert 0 <= result <= 2


def test_rpv_doubling_rho0_doubles_output():
    r1 = rpv_1((30, 10, 20), rho0=0.1, k=1.0, theta=0.1)
    r2 = rpv_1((30, 10, 20), rho0=0.2, k=1.0, theta=0.1)
    assert r2 == pytest.approx(2 * r1, rel=1e-6)


def test_rpv_output_in_range():
    angles = [
        (10, 5, 0),
        (30, 20, 60),
        (60, 45, 120),
        (80, 70, 180),
        (5, 0, 30),
    ]
    for sza, vza, raa in angles:
        r = rpv_2((sza, vza, raa), rho0=0.3, k=0.8, theta=0.2)
        assert 0 <= r <= 2, f"rpv_2({sza},{vza},{raa}) = {r} out of [0,2]"


def test_rpv_extreme_angles():
    result = rpv_1((85, 80, 179), rho0=0.5, k=1.5, theta=-0.5)
    assert 0 <= result <= 2

    result = rpv_2((0.1, 0.1, 0.1), rho0=0.2, k=0.5, theta=0.9)
    assert 0 <= result <= 2


def test_rpv_1_and_rpv_2_differ():
    # rpv_1 divides by (cs+cv)^k, rpv_2 multiplies by (cs+cv)^(k-1)
    result1 = rpv_1((30, 10, 45), rho0=0.3, k=1.5, theta=0.2)
    result2 = rpv_2((30, 10, 45), rho0=0.3, k=1.5, theta=0.2)
    assert result1 != pytest.approx(result2, rel=1e-4)


# ---------------------------------------------------------------------------
# rpv_fit with synthetic data
# ---------------------------------------------------------------------------

def test_rpv_fit_synthetic():
    true_rho0 = 0.3
    true_k = 0.8
    true_theta = 0.1

    np.random.seed(42)
    n = 200
    sza = np.random.uniform(20, 50, n)
    vza = np.random.uniform(0, 40, n)
    raa = np.random.uniform(0, 180, n)

    reflectance = rpv_2((sza, vza, raa), true_rho0, true_k, true_theta)
    noise = np.random.normal(0, 0.005, n)
    reflectance = reflectance + noise

    df = pl.DataFrame({
        "sza": sza,
        "vza": vza,
        "raa": raa,
        "band1": reflectance,
    })

    rho0, k, theta, rc, rmse, nrmse = rpv_fit(df, band="band1", n_samples_bins=5)

    assert rho0 == pytest.approx(true_rho0, abs=0.1)
    assert k == pytest.approx(true_k, abs=0.3)
    assert theta == pytest.approx(true_theta, abs=0.15)


# ---------------------------------------------------------------------------
# df_preprocess edge cases
# ---------------------------------------------------------------------------

def test_df_preprocess_empty():
    df = pl.DataFrame({
        "Xw": pl.Series([], dtype=pl.Float64),
        "Yw": pl.Series([], dtype=pl.Float64),
        "xcam": pl.Series([], dtype=pl.Float64),
        "ycam": pl.Series([], dtype=pl.Float64),
        "delta_z": pl.Series([], dtype=pl.Float64),
        "sunelev": pl.Series([], dtype=pl.Float64),
        "saa": pl.Series([], dtype=pl.Float64),
        "band1": pl.Series([], dtype=pl.Float64),
        "band2": pl.Series([], dtype=pl.Float64),
        "band3": pl.Series([], dtype=pl.Float64),
        "band5": pl.Series([], dtype=pl.Float64),
    })
    with pytest.raises(ZeroDivisionError):
        df_preprocess(df)


def test_df_preprocess_missing_vx_vy_vz_creates_them():
    df = pl.DataFrame({
        "Xw": [0.0],
        "Yw": [0.0],
        "xcam": [1.0],
        "ycam": [2.0],
        "delta_z": [10.0],
        "sunelev": [30.0],
        "saa": [150.0],
        "band1": [0.2],
        "band2": [0.2],
        "band3": [0.1],
        "band5": [0.2],
    })
    result = df_preprocess(df)
    assert "vx" in result.columns
    assert "vy" in result.columns
    assert "vz" in result.columns
    assert result["vx"][0] == pytest.approx(1.0)
    assert result["vy"][0] == pytest.approx(2.0)
    assert result["vz"][0] == pytest.approx(10.0)


def test_df_preprocess_computes_ndvi():
    df = pl.DataFrame({
        "Xw": [0.0],
        "Yw": [0.0],
        "xcam": [0.0],
        "ycam": [0.0],
        "delta_z": [10.0],
        "sunelev": [30.0],
        "saa": [150.0],
        "band1": [0.1],
        "band2": [0.2],
        "band3": [0.3],
        "band5": [0.7],
    })
    result = df_preprocess(df)
    expected_ndvi = (0.7 - 0.3) / (0.7 + 0.3)
    assert result["NDVI"][0] == pytest.approx(expected_ndvi, rel=1e-6)
