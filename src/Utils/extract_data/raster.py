#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# version ='1.0'
# ---------------------------------------------------------------------------

"""
This code copiles all Common that will be called later by other codes. For the sake of clarity
these Common are defined in this separated piece of code.
"""
import math
import os
from timeit import default_timer as timer
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
import affine
import polars as pl
import rasterio as rio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject


def pixelToWorldCoords(pX, pY, geoTransform):
    """Input image pixel coordinates and get world coordinates according to geotransform using gdal"""

    def applyGeoTransform(inX, inY, geoTransform):
        outX = geoTransform[0] + inX * geoTransform[1] + inY * geoTransform[2]
        outY = geoTransform[3] + inX * geoTransform[4] + inY * geoTransform[5]
        return outX, outY

    mX, mY = applyGeoTransform(pX, pY, geoTransform)
    return mX, mY


def worldToPixelCoords(wX, wY, geoTransform, dtype="int"):
    """Input world coordinates and get pixel coordinates according to reverse geotransform using gdal"""
    reverse_transform = ~affine.Affine.from_gdal(*geoTransform)
    px, py = reverse_transform * (wX, wY)
    if dtype == "int":
        px, py = int(px + 0.5), int(py + 0.5)
    else:
        px, py = px + 0.5, py + 0.5
    return px, py


def xyval(A):
    """
    Function to list all pixel coords including their associated value starting from the  upper left corner (?)
    :param A: raster band as numpy array
    :return: x and y of each pixel and the associated value
    """
    import numpy as np

    x, y = np.indices(A.shape)
    return x.ravel(), y.ravel(), A.ravel()


def to_numpy2(transform):
    return np.array(
        [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f, 0, 0, 1],
        dtype="float64",
    ).reshape((3, 3))


def xy_np(transform, rows, cols, offset="center"):
    if isinstance(rows, int) and isinstance(cols, int):
        pts = np.array([[rows, cols, 1]]).T
    else:
        assert len(rows) == len(cols)
        pts = np.ones((3, len(rows)), dtype=int)
        pts[0] = rows
        pts[1] = cols

    if offset == "center":
        coff, roff = (0.5, 0.5)
    elif offset == "ul":
        coff, roff = (0, 0)
    elif offset == "ur":
        coff, roff = (1, 0)
    elif offset == "ll":
        coff, roff = (0, 1)
    elif offset == "lr":
        coff, roff = (1, 1)
    else:
        raise ValueError("Invalid offset")

    _transnp = to_numpy2(transform)
    _translt = to_numpy2(transform.translation(coff, roff))
    locs = _transnp @ _translt @ pts
    return locs[0].tolist(), locs[1].tolist()


# ------------------------------
# Orthophoto Reading: Reflectance Bands
# ------------------------------
def read_orthophoto_bands(each_ortho, transform_to_utm=True, target_crs="EPSG:32632"):
    start_bands = timer()
    try:
        with rio.open(each_ortho) as rst:
            num_bands = rst.count
            b_all = rst.read(masked=True)  # Read all bands

            rows, cols = np.indices((rst.height, rst.width))
            rows_flat, cols_flat = rows.flatten(), cols.flatten()
            Xw, Yw = rio.transform.xy(rst.transform, rows_flat, cols_flat, offset="center")

            if transform_to_utm:
                transformer = Transformer.from_crs(rst.crs, target_crs, always_xy=True)
                x_trans, y_trans = transformer.transform(Xw, Yw)
                Xw = np.array(x_trans)
                Yw = np.array(y_trans)
            else:
                Xw = np.array(Xw)
                Yw = np.array(Yw)

            band_values = b_all.reshape(num_bands, -1).T

            data = {"Xw": pl.Series(Xw), "Yw": pl.Series(Yw)}
            for idx in range(num_bands):
                data[f"band{idx + 1}"] = pl.Series(band_values[:, idx])
            df_allbands = pl.DataFrame(data)

        end_bands = timer()
        logging.info(
            f"Processed orthophoto bands for {os.path.basename(each_ortho)} in {end_bands - start_bands:.2f} seconds"
        )
        return df_allbands
    except Exception as e:
        logging.error(f"Error reading bands from {each_ortho}: {e}")
        raise


def coregister_and_resample(
    input_path, ref_path, output_path, target_resolution=None, resampling=Resampling.nearest
):
    """
    Reproject and resample the input raster (orthophoto) to match the reference raster (DEM).
    If target_resolution is provided (e.g., (10, 10)), the output will be resampled to that resolution.
    """
    try:
        with rio.open(ref_path) as ref_src:
            ref_crs = ref_src.crs
            ref_transform = ref_src.transform
            ref_width = ref_src.width
            ref_height = ref_src.height
            ref_bounds = ref_src.bounds

        with rio.open(input_path) as src:
            # Compute the transform, width, and height of the output raster
            if target_resolution:
                # If a target resolution is provided, recalculate transform for the new resolution
                # We'll use calculate_default_transform but override the resolution
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs, ref_crs, ref_width, ref_height, *ref_bounds
                )

                # Now adjust the transform for the target resolution
                # The calculate_default_transform gives a transform that matches the reference image,
                # We just need to scale it if we want a different resolution.
                xres, yres = target_resolution
                # We know dst_transform:
                # dst_transform = | xres, 0, x_min|
                #                 | 0, -yres, y_max|
                #                 | 0, 0, 1       |
                # If we want a different resolution, we can recompute width/height
                x_min, y_min, x_max, y_max = ref_bounds
                dst_width = int((x_max - x_min) / xres)
                dst_height = int((y_max - y_min) / abs(yres))
                dst_transform = rio.transform.from_bounds(
                    x_min, y_min, x_max, y_max, dst_width, dst_height
                )
            else:
                # Use the reference transform and size directly
                dst_transform, dst_width, dst_height = ref_transform, ref_width, ref_height

            dst_kwargs = src.meta.copy()
            dst_kwargs.update(
                {
                    "crs": ref_crs,
                    "transform": dst_transform,
                    "width": dst_width,
                    "height": dst_height,
                }
            )

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with rio.open(output_path, "w", **dst_kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=ref_crs,
                        resampling=resampling,
                    )

        logging.info(f"Co-registration and resampling completed. Output saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error in co-registration: {e}")
        raise


def check_alignment(dem_path, ortho_path):
    """
    Check if the DEM and orthophoto are aligned:
    - Same CRS
    - Same pixel size
    - Same alignment of pixel boundaries

    Returns True if aligned, False otherwise.
    """
    try:
        with rio.open(dem_path) as dem_src, rio.open(ortho_path) as ortho_src:
            # Check CRS
            if dem_src.crs != ortho_src.crs:
                logging.info(f"CRS mismatch: DEM={dem_src.crs}, Ortho={ortho_src.crs}")
                return False

            # Extract transforms
            dem_transform = dem_src.transform
            ortho_transform = ortho_src.transform

            # Check pixel sizes
            dem_res = (dem_transform.a, dem_transform.e)  # (xres, yres)
            ortho_res = (ortho_transform.a, ortho_transform.e)
            if not (abs(dem_res[0]) == abs(ortho_res[0]) and abs(dem_res[1]) == abs(ortho_res[1])):
                logging.info(f"Resolution mismatch: DEM={dem_res}, Ortho={ortho_res}")
                return False

            # Check alignment: We must ensure that the pixel grid aligns.
            # We'll check if the difference in the origins aligns with a multiple of the pixel size.
            # For alignment: (ortho_transform.c - dem_transform.c) should be an integer multiple of the xres
            # and (ortho_transform.f - dem_transform.f) should be an integer multiple of the yres.

            x_offset = (ortho_transform.c - dem_transform.c) / dem_res[0]
            y_offset = (ortho_transform.f - dem_transform.f) / dem_res[1]

            # If x_offset and y_offset are close to integers, we consider them aligned
            if not (
                abs(x_offset - round(x_offset)) < 1e-6 and abs(y_offset - round(y_offset)) < 1e-6
            ):
                logging.info(f"Pixel alignment mismatch: x_offset={x_offset}, y_offset={y_offset}")
                return False

            # If we reach here, DEM and orthophoto are aligned
            logging.info(
                f" DEM and orthophoto are aligned: x_offset={x_offset}, y_offset={y_offset}"
            )
            return True
    except Exception as e:
        logging.error(f"Error checking alignment: {e}")
        return False


def latlon_to_utm32n_series(lat_deg, lon_deg):
    """
    Convert geographic coordinates (lat, lon in degrees, WGS84)
    to UTM Zone 32N (EPSG:32632) using the standard UTM formulas.

    Returns:
      (easting, northing) in meters.
    """
    # WGS84 ellipsoid constants
    a = 6378137.0  # semi-major axis (meters)
    f = 1 / 298.257223563  # flattening
    e2 = 2 * f - f**2  # eccentricity squared
    e = math.sqrt(e2)

    # UTM parameters for Zone 32N
    k0 = 0.9996
    E0 = 500000.0  # false easting
    N0 = 0.0  # false northing (northern hemisphere)
    lambda0 = math.radians(9.0)  # central meridian for Zone 32N (9°E)

    # Convert input latitude and longitude from degrees to radians
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)

    # Compute auxiliary values
    N_val = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    T = math.tan(phi) ** 2
    # Second eccentricity squared
    ep2 = e2 / (1 - e2)
    C = ep2 * math.cos(phi) ** 2
    A = (lam - lambda0) * math.cos(phi)

    # Meridional arc length (M)
    M = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * phi)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * phi)
        - (35 * e2**3 / 3072) * math.sin(6 * phi)
    )

    # Calculate Easting and Northing using standard UTM series formulas
    easting = E0 + k0 * N_val * (
        A + (1 - T + C) * A**3 / 6 + (5 - 18 * T + T**2 + 72 * C - 58 * ep2) * A**5 / 120
    )

    northing = N0 + k0 * (
        M
        + N_val
        * math.tan(phi)
        * (
            A**2 / 2
            + (5 - T + 9 * C + 4 * C**2) * A**4 / 24
            + (61 - 58 * T + T**2 + 600 * C - 330 * ep2) * A**6 / 720
        )
    )

    return easting, northing


# python
# python
# python
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import gaussian_kde


def _ensure_outdir(base_path: str) -> str:
    outdir = os.path.join(base_path, "bands_data")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _has_columns(df, required: Sequence[str]) -> Tuple[bool, List[str]]:
    missing = [c for c in required if c not in df.columns]
    return (len(missing) == 0, missing)


def _extract_xy(df) -> Tuple[np.ndarray, np.ndarray]:
    x = df["Xw"].to_numpy()
    y = df["Yw"].to_numpy()
    return x, y


def _finite_mask(*arrays: Iterable[np.ndarray]) -> np.ndarray:
    # Mask rows where any provided array has invalid (NaN/inf) values
    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def _log_stats_debug(debug: bool, name: str, arr: np.ndarray) -> None:
    if not debug:
        return
    n = arr.size
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    n_finite = int(np.isfinite(arr).sum())
    msg = f"[plotting_raster] {name}: size={n}, n_nan={n_nan}, n_inf={n_inf}, n_finite={n_finite}"
    if n_finite > 0:
        try:
            msg += (
                f", min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}, "
                f"mean={np.nanmean(arr):.6g}"
            )
        except Exception:
            pass
    logging.info(msg)


def _compute_extent(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    return float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(y)), float(np.nanmax(y))


def _make_bins(xmin: float, xmax: float, ymin: float, ymax: float, nx: int, ny: int):
    xbins = np.linspace(xmin, xmax, nx + 1)
    ybins = np.linspace(ymin, ymax, ny + 1)
    return xbins, ybins


def _counts2d(x: np.ndarray, y: np.ndarray, xbins: np.ndarray, ybins: np.ndarray) -> np.ndarray:
    counts, _, _ = np.histogram2d(y, x, bins=[ybins, xbins])
    return counts


def _auto_coarsen_for_occupancy(
    x: np.ndarray,
    y: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    nx: int,
    ny: int,
    occupancy_target: Optional[float],
    min_bins: int,
    debug: bool,
) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Automatically coarsens the 2D binning configuration to meet a specified occupancy
    target by iteratively reducing the number of bins. The occupancy target represents
    the fraction of non-zero bins among all bins. By adjusting the bin sizes, this
    function ensures the desired occupancy is fulfilled if feasible.

    Parameters:
    x : np.ndarray
        Array of x-coordinates for the data points to be binned.
    y : np.ndarray
        Array of y-coordinates for the data points to be binned.
    xmin : float
        Minimum bound for the x-axis range.
    xmax : float
        Maximum bound for the x-axis range.
    ymin : float
        Minimum bound for the y-axis range.
    ymax : float
        Maximum bound for the y-axis range.
    nx : int
        Initial number of bins along the x-axis.
    ny : int
        Initial number of bins along the y-axis.
    occupancy_target : Optional[float]
        Target fraction of non-zero bins, between 0 and 1. If None or invalid,
        no coarsening is applied.
    min_bins : int
        Minimum number of bins allowed along any axis during the coarsening process.
    debug : bool
        If True, provides detailed logging of the coarsening steps.

    Returns:
    Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - Final number of bins along the x-axis.
        - Final number of bins along the y-axis.
        - Array of x-axis bin edges (xbins).
        - Array of y-axis bin edges (ybins).
        - 2D array of bin counts after coarsening.
    """
    xbins, ybins = _make_bins(xmin, xmax, ymin, ymax, nx, ny)
    counts_all = _counts2d(x, y, xbins, ybins)
    if occupancy_target is None or occupancy_target <= 0 or occupancy_target > 1:
        return nx, ny, xbins, ybins, counts_all

    total_bins = counts_all.size
    zero_bins = int((counts_all == 0).sum())
    occupancy = 1.0 - zero_bins / total_bins if total_bins else 0.0
    if debug:
        logging.info(
            f"[plotting_raster] 2D counts: bins={total_bins}, zero_bins={zero_bins} "
            f"({(100*zero_bins/total_bins if total_bins else 0):.2f}%), occupancy={occupancy:.3f}"
        )
    if occupancy >= occupancy_target:
        return nx, ny, xbins, ybins, counts_all

    cur_nx, cur_ny = nx, ny
    while (cur_nx > min_bins or cur_ny > min_bins) and occupancy < occupancy_target:
        cur_nx = max(min_bins, int(cur_nx * 0.8))
        cur_ny = max(min_bins, int(cur_ny * 0.8))
        xbins, ybins = _make_bins(xmin, xmax, ymin, ymax, cur_nx, cur_ny)
        counts_all = _counts2d(x, y, xbins, ybins)
        total_bins = counts_all.size
        zero_bins = int((counts_all == 0).sum())
        occupancy = 1.0 - zero_bins / total_bins if total_bins else 0.0
        if debug:
            logging.info(
                f"[plotting_raster] Coarsened to {cur_nx}x{cur_ny}; occupancy={occupancy:.3f}"
            )
    return cur_nx, cur_ny, xbins, ybins, counts_all


def _grid_mean_for_series(
    series,
    m_coord: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xbins: np.ndarray,
    ybins: np.ndarray,
    counts_all: np.ndarray,
    fill_empty: bool,
    debug: bool,
    name: str,
) -> np.ndarray:
    """
    Compute a 2D grid of mean values from a given series based on provided coordinate masks,
    bins, and additional parameters.

    This function processes a numerical series, filters its values according to a given
    coordinate mask, and calculates a 2D grid of means for the masked values. The grid is
    constructed using histogram bin edges in both x and y dimensions. If specified, this
    function can also fill empty bins in the grid using a neighborhood-based approach. Debugging
    information can be logged by enabling the debug flag.

    Parameters:
        series (pd.Series): Input data series to process.
        m_coord (np.ndarray): Boolean mask indicating which coordinates from the series to consider.
        x (np.ndarray): Array containing x-coordinates corresponding to the series data.
        y (np.ndarray): Array containing y-coordinates corresponding to the series data.
        xbins (np.ndarray): Array defining the edges of bins along the x axis.
        ybins (np.ndarray): Array defining the edges of bins along the y axis.
        counts_all (np.ndarray): Array representing existing counts for all bins.
        fill_empty (bool): Flag indicating whether empty bins should be filled based on a uniform filter.
        debug (bool): Flag to enable detailed debug logging.
        name (str): Name used in debug logging to identify the operation context.

    Returns:
        np.ndarray: 2D grid of mean values calculated based on the input data and coordinates.
    """
    v_full = series.to_numpy()
    v = v_full[m_coord]
    vm = np.isfinite(v)

    if debug:
        n = v.size
        n_finite = int(vm.sum())
        n_nan = int(np.isnan(v).sum())
        logging.info(
            f"[plotting_raster] {name}: values after coord-mask size={n}, "
            f"finite={n_finite} ({(100*n_finite/max(n,1)):.2f}%), nan={n_nan}"
        )
        _log_stats_debug(debug, f"{name} (masked values)", v)

    sums, _, _ = np.histogram2d(y[vm], x[vm], bins=[ybins, xbins], weights=v[vm])
    counts_valid, _, _ = np.histogram2d(y[vm], x[vm], bins=[ybins, xbins])

    grid = np.divide(sums, counts_valid, out=np.full_like(sums, np.nan), where=counts_valid > 0)

    if debug:
        n_cells = grid.size
        n_nan_cells = int(np.isnan(grid).sum())
        n_finite_cells = n_cells - n_nan_cells
        zero_bins_all = int((counts_all == 0).sum())
        zero_bins_valid = int((counts_valid == 0).sum())
        logging.info(
            f"[plotting_raster] {name}: grid size={grid.shape}, "
            f"finite_cells={n_finite_cells}/{n_cells} "
            f"({(100*n_finite_cells/max(n_cells,1)):.2f}%), "
            f"nan_cells={n_nan_cells}, zero_bins_all={zero_bins_all}, "
            f"zero_bins_valid={zero_bins_valid}"
        )
        try:
            if n_finite_cells > 0:
                p2, p50, p98 = np.nanpercentile(grid, [2, 50, 98])
                logging.info(
                    f"[plotting_raster] {name}: grid percentiles p2={p2:.6g}, "
                    f"p50={p50:.6g}, p98={p98:.6g}"
                )
        except Exception:
            pass

    if fill_empty:
        val = grid.copy()
        mask = np.isfinite(val).astype(float)
        val[np.isnan(val)] = 0.0
        num = uniform_filter(val, size=3, mode="nearest")
        den = uniform_filter(mask, size=3, mode="nearest")
        filled = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        grid = np.where(np.isfinite(grid), grid, filled)
        if debug:
            n_after = grid.size - int(np.isnan(grid).sum())
            logging.info(
                f"[plotting_raster] {name}: filled empty bins -> finite_cells={n_after}/{grid.size}"
            )

    return grid


def _density_grid(
    x: np.ndarray,
    y: np.ndarray,
    xbins: np.ndarray,
    ybins: np.ndarray,
    mode: str,
    kde_bw,            # None | "scott" | float | tuple(float, float)
    log_scale: bool,
) -> np.ndarray:
    """
    Fast density computation.

    Modes:
    - "hist": 2D histogram counts (fastest).
    - "kde":  2D histogram convolved with a Gaussian kernel (fast KDE).
              Bandwidth can be None/"scott"/float/(float,float) in coordinate units.
    - "kde_exact": legacy exact gaussian_kde evaluation on the grid (slow; use only for small data).
    """
    # Bin widths in coordinate units (for bandwidth -> sigma_bins conversion)
    dx = float(np.diff(xbins).mean()) if len(xbins) > 1 else 1.0
    dy = float(np.diff(ybins).mean()) if len(ybins) > 1 else 1.0

    if mode == "hist":
        dens, _, _ = np.histogram2d(y, x, bins=[ybins, xbins])
        dens = dens.astype(float)

    elif mode == "kde":
        counts, _, _ = np.histogram2d(y, x, bins=[ybins, xbins])
        counts = counts.astype(float)

        # 2) Choose bandwidth in meters (coordinate units), then convert to sigma in bins
        if kde_bw is None or (isinstance(kde_bw, str) and kde_bw.lower() == "scott"):
            n = max(1, x.size)
            factor = n ** (-1 / 6)  # Scott's factor for 2D
            sx = float(np.nanstd(x)) if np.isfinite(np.nanstd(x)) else dx
            sy = float(np.nanstd(y)) if np.isfinite(np.nanstd(y)) else dy
            bw_x_m = max(1e-12, sx * factor)
            bw_y_m = max(1e-12, sy * factor)
        elif isinstance(kde_bw, (list, tuple)) and len(kde_bw) == 2:
            bw_x_m = float(kde_bw[0])
            bw_y_m = float(kde_bw[1])
        else:
            # Assume scalar isotropic bandwidth in meters
            bw_x_m = bw_y_m = float(kde_bw)

        # Convert to sigma in bins
        sigma_x_bins = max(1e-9, bw_x_m / dx)
        sigma_y_bins = max(1e-9, bw_y_m / dy)

        # 3) Convolve histogram with Gaussian kernel (fast KDE)
        from scipy.ndimage import gaussian_filter

        dens = gaussian_filter(counts, sigma=(sigma_y_bins, sigma_x_bins), mode="nearest")

    elif mode == "kde_exact":
        # Fallback to exact gaussian_kde (slow; O(Npoints * Ncells))
        coords = np.vstack([x, y])
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(coords, bw_method=None if kde_bw in (None, "scott", "Scott") else kde_bw)
        x_centers = 0.5 * (xbins[:-1] + xbins[1:])
        y_centers = 0.5 * (ybins[:-1] + ybins[1:])
        XX, YY = np.meshgrid(x_centers, y_centers)
        dens = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(len(y_centers), len(x_centers))
    else:
        # Unknown mode -> default to histogram
        dens, _, _ = np.histogram2d(y, x, bins=[ybins, xbins])
        dens = dens.astype(float)

    if log_scale:
        dens = np.log1p(dens)
    return dens



def _auto_figsize(
    nx: int,
    ny: int,
    rows: int,
    cols: int,
    pixels_per_bin: float,
    dpi: int,
    min_panel_size: Tuple[float, float],
    max_panel_size: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    def clamp(val, lo, hi):
        return max(lo, min(val, hi))

    panel_w_in = clamp((nx * pixels_per_bin) / dpi, min_panel_size[0], max_panel_size[0])
    panel_h_in = clamp((ny * pixels_per_bin) / dpi, min_panel_size[1], max_panel_size[1])
    return (cols * panel_w_in, rows * panel_h_in), (panel_w_in, panel_h_in)


def _plot_panels(
    panels: List[Tuple[str, np.ndarray, Dict[str, Any]]],
    extent: List[float],
    out_path: str,
    rows: int,
    cols: int,
    dpi: int,
    figsize: Tuple[float, float],
    debug: bool,
) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, dpi=dpi)
    for ax, (title, grid, kw) in zip(axes.ravel(), panels):
        im = ax.imshow(grid, origin="lower", extent=extent, interpolation="bilinear", **kw)
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        fig.colorbar(im, ax=ax, shrink=0.85)

    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    if debug:
        logging.info(f"[plotting_raster] Saved panels to: {out_path}")
    plt.close(fig)


def _scatter_quicklook(
    x: np.ndarray,
    y: np.ndarray,
    out_path: str,
    dpi: int,
    max_points: int,
    size: float,
    alpha: float,
    extent: Tuple[float, float, float, float],
    debug: bool,
) -> None:
    try:
        n = x.size
        if n > max_points:
            idx = np.random.choice(n, size=max_points, replace=False)
            xs, ys = x[idx], y[idx]
            sampled = max_points
        else:
            xs, ys = x, y
            sampled = n

        if debug:
            logging.info(f"[plotting_raster] Scatter quicklook: plotting {sampled} points")

        xmin, xmax, ymin, ymax = extent
        width_m = xmax - xmin
        height_m = ymax - ymin
        aspect = width_m / max(height_m, 1e-9)
        base_w = 10.0
        fig_w = base_w
        fig_h = max(6.0, base_w / max(aspect, 1e-6))

        fig_s, ax_s = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax_s.scatter(xs, ys, s=size, alpha=alpha, c="k")
        ax_s.set_title("XY point scatter quicklook")
        ax_s.set_xlabel("X (m)")
        ax_s.set_ylabel("Y (m)")
        ax_s.set_aspect("equal", adjustable="box")
        fig_s.tight_layout()
        fig_s.savefig(out_path, dpi=dpi)
        if debug:
            logging.info(f"[plotting_raster] Saved scatter quicklook to: {out_path}")
        plt.close(fig_s)
    except Exception as e:
        logging.error(f"[plotting_raster] Failed scatter quicklook: {e}")


def _band_kde_plot(
    df,
    bands: List[str],
    out_path: str,
    xlim: Tuple[float, float],
    points: int,
    linewidth: float,
    colors: Optional[List[str]],
    dpi: int,
    debug: bool,
) -> None:
    try:
        x_min, x_max = xlim
        x_grid = np.linspace(x_min, x_max, int(points))
        fig_k, ax_k = plt.subplots(figsize=(10, 6), dpi=dpi)

        cycle = colors or plt.rcParams["axes.prop_cycle"].by_key().get("color", None)

        for i, b in enumerate(bands):
            v = df[b].to_numpy()
            # Restrict to finite values within [x_min, x_max] to match chart limits
            v = v[np.isfinite(v)]
            v = v[(v >= x_min) & (v <= x_max)]
            if v.size < 5:
                continue

            # Fast KDE via histogram smoothing (orders of magnitude faster than gaussian_kde on big data)
            y_pdf = _kde1d_fast(
                v,
                x_grid,
                bw=None,  # or set a float bandwidth in data units (e.g., 0.01)
                bins=2058,  # can lower to 512 for even faster
                vmin=x_min,
                vmax=x_max,
            )

            color = None
            if cycle and i < len(cycle):
                color = cycle[i]
            ax_k.plot(x_grid, y_pdf, label=b, linewidth=linewidth, color=color)

        ax_k.set_xlim(x_min, x_max)
        ax_k.set_ylim(bottom=0.0)
        ax_k.set_xlabel("Value")
        ax_k.set_ylabel("Density (KDE)")
        ax_k.set_title("Per-band value distributions (KDE)")
        ax_k.grid(True, alpha=0.3)
        ax_k.legend(ncol=2)
        fig_k.tight_layout()
        fig_k.savefig(out_path, dpi=dpi)
        if debug:
            logging.info(f"[plotting_raster] Saved combined band KDE to: {out_path}")
        plt.close(fig_k)
    except Exception as e:
        logging.error(f"[plotting_raster] Failed to create band KDE chart: {e}")
from scipy.ndimage import gaussian_filter1d

def _kde1d_fast(
        v: np.ndarray,
        x_grid: np.ndarray,
        bw: float | None = None,
        bins: int = 1024,
        vmin: float | None = None,
        vmax: float | None = None,
) -> np.ndarray:
    """
    Approximate 1D KDE efficiently via histogram + Gaussian smoothing.

    Steps:
    1) Bin values into a fine histogram.
    2) Smooth counts with gaussian_filter1d using sigma derived from bandwidth.
    3) Interpolate smoothed density to x_grid and normalize to integrate to ~1.

    Args:
        v: 1D array of finite samples.
        x_grid: Points where the PDF should be evaluated.
        bw: Bandwidth in data units. If None, use Scott's rule.
        bins: Number of histogram bins for the smoothing grid.
        vmin, vmax: Optional clipping range. If None, inferred from data.

    Returns:
        y_pdf evaluated at x_grid (approximately normalized).
    """
    v = v[np.isfinite(v)]
    if v.size < 5:
        return np.zeros_like(x_grid)

    # Range and histogram grid
    lo = np.min(v) if vmin is None else float(vmin)
    hi = np.max(v) if vmax is None else float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x_grid)

    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(v, bins=edges)

    # Bandwidth: Scott's rule if not provided
    if bw is None:
        s = np.std(v)
        n = v.size
        # Scott's rule: bw = 1.06 * s * n^(-1/5); fallback if s==0
        bw = 1.06 * (s if s > 0 else (hi - lo) / 6.0) * (n ** (-1.0 / 5.0))
        if bw <= 0 or not np.isfinite(bw):
            bw = max((hi - lo) / 100.0, 1e-12)

    # Convert bandwidth to sigma in bins
    bin_width = centers[1] - centers[0]
    sigma_bins = max(bw / bin_width, 1e-6)

    # Smooth counts
    smooth = gaussian_filter1d(counts.astype(float), sigma=sigma_bins, mode="nearest")

    # Convert to density (divide by N and bin width)
    density_centers = smooth / (v.size * bin_width)

    # Interpolate to requested x_grid
    y_pdf = np.interp(x_grid, centers, density_centers, left=0.0, right=0.0)

    # Normalize lightly to ensure area ≈ 1 over [lo, hi]
    area = np.trapz(y_pdf, x_grid)
    if area > 0 and np.isfinite(area):
        y_pdf = y_pdf / area

    return y_pdf

def _band_histograms(
    df,
    bands: List[str],
    m_coord: np.ndarray,
    out_path: str,
    clip: Tuple[float, float],
    dpi: int,
    debug: bool,
) -> None:
    cols_h = min(3, len(bands))
    rows_h = int(np.ceil(len(bands) / cols_h)) if cols_h > 0 else 1
    fig_h, axes_h = plt.subplots(rows_h, cols_h, figsize=(5 * cols_h, 3.5 * rows_h), squeeze=False, dpi=dpi)

    for ax, b in zip(axes_h.ravel(), bands):
        try:
            v = df[b].to_numpy()[m_coord]
            v = v[np.isfinite(v)]
            if v.size:
                lo_h, hi_h = np.percentile(v, clip)
                ax.hist(v, bins=50, range=(lo_h, hi_h))
                if debug:
                    logging.info(
                        f"[plotting_raster] Hist {b}: n={v.size}, clip_range=({lo_h:.6g},{hi_h:.6g})"
                    )
            else:
                if debug:
                    logging.info(f"[plotting_raster] Hist {b}: no finite values after masking.")
        except Exception as e:
            logging.error(f"[plotting_raster] Failed histogram for {b}: {e}")
        ax.set_title(b)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    for ax in axes_h.ravel()[len(bands) :]:
        ax.axis("off")

    fig_h.tight_layout()
    fig_h.savefig(out_path, dpi=dpi)
    if debug:
        logging.info(f"[plotting_raster] Saved histograms to: {out_path}")
    plt.close(fig_h)


def plotting_raster(
    df_merged,
    path: str,
    file_name: str,
    bands_prefix: str = "band",
    custom_columuns: Optional[list] = None,
    nx: int = 1500,
    ny: int = 1500,
    max_bands: int = 6,
    clip: Tuple[int, int] = (2, 98),
    debug: bool = False,
    fill_empty: bool = False,
    occupancy_target: Optional[float] = None,
    min_bins: int = 50,
    plot_density: bool = False,
    density_log: bool = False,
    density_cmap: str = "magma",
    density_clip: Tuple[int, int] = (2, 98),
    scatter_quicklook: bool = False,
    scatter_max: int = 200000,
    scatter_alpha: float = 0.25,
    scatter_size: float = 1.0,
    density_vmin: Optional[float] = None,
    density_vmax: Optional[float] = None,
    density_discrete: bool = False,
    # Kernel density for spatial density
    density_mode: str = "kde",          # "hist" or "kde"
    density_kde_bw: Optional[Any] = None,
    # Combined per-band KDE chart
    band_kde: bool = True,
    band_kde_xlim: Tuple[float, float] = (0.0, 1.0),
    band_kde_points: int = 512,
    band_kde_linewidth: float = 2.0,
    band_kde_colors: Optional[List[str]] = None,
    # Figure sizing
    auto_figsize: bool = True,
    dpi: int = 200,
    pixels_per_bin: float = 4.0,
    min_panel_size: Tuple[float, float] = (4.0, 3.5),
    max_panel_size: Tuple[float, float] = (60.0, 120.0),
) -> None:
    """
    Orchestrates gridding, density visualization, and panel plotting by delegating to
    small focused helpers. Saves:
    - panels_<file_name>.png
    - band_distributions_<file_name>.png
    - scatter_quicklook_<file_name>.png (optional)
    - band_kde_<file_name>.png (optional)
    """

    # Setup and validation
    outdir = _ensure_outdir(path)
    if debug:
        logging.info(
            f"[plotting_raster] start: file_name={file_name}, nx={nx}, ny={ny}, "
            f"max_bands={max_bands}, clip={clip}, bands_prefix='{bands_prefix}', "
            f"fill_empty={fill_empty}, occupancy_target={occupancy_target}, "
            f"plot_density={plot_density}, density_mode={density_mode}, "
            f"auto_figsize={auto_figsize}, dpi={dpi}, pixels_per_bin={pixels_per_bin}"
        )

    ok, missing = _has_columns(df_merged, ["Xw", "Yw"])
    if not ok:
        logging.error(f"[plotting_raster] Missing required columns: {missing}")
        return

    # Extract and clean coordinates
    try:
        x, y = _extract_xy(df_merged)
    except Exception as e:
        logging.error(f"[plotting_raster] Failed to extract Xw/Yw arrays: {e}")
        return

    _log_stats_debug(debug, "Xw", x)
    _log_stats_debug(debug, "Yw", y)

    m_coord = _finite_mask(x, y)
    if debug:
        kept = int(m_coord.sum())
        total = x.size
        pct = 100 * kept / total if total else 0
        logging.info(f"[plotting_raster] Coord mask: kept={kept}/{total} ({pct:.2f}%)")

    x, y = x[m_coord], y[m_coord]
    if x.size == 0 or y.size == 0:
        logging.warning("[plotting_raster] No finite coordinates after masking; aborting.")
        return

    xmin, xmax, ymin, ymax = _compute_extent(x, y)
    if debug:
        logging.info(
            f"[plotting_raster] Extent: xmin={xmin:.6f}, xmax={xmax:.6f}, "
            f"ymin={ymin:.6f}, ymax={ymax:.6f}"
        )

    # Binning and occupancy
    nx, ny, xbins, ybins, counts_all = _auto_coarsen_for_occupancy(
        x, y, xmin, xmax, ymin, ymax, nx, ny, occupancy_target, min_bins, debug
    )

    # Panels: density (optional), band means, elevation
    panels: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []

    if plot_density:
        dens = _density_grid(x, y, xbins, ybins, density_mode, density_kde_bw, density_log)
        if density_vmin is not None or density_vmax is not None:
            vmin = 0.0 if density_vmin is None else float(density_vmin)
            vmax = float(np.nanmax(dens)) if density_vmax is None else float(density_vmax)
        else:
            try:
                vmin, vmax = np.nanpercentile(dens, density_clip)
            except Exception:
                vmin, vmax = (np.nanmin(dens), np.nanmax(dens))

        if density_discrete and density_mode != "kde":
            import matplotlib as mpl

            boundaries = np.arange(np.floor(vmin), np.ceil(vmax) + 1)
            norm = mpl.colors.BoundaryNorm(boundaries=boundaries, ncolors=256)
            dens_kw = dict(cmap=density_cmap, norm=norm)
        else:
            dens_kw = dict(cmap=density_cmap, vmin=vmin, vmax=vmax)

        title = "point_density_kde" if density_mode == "kde" else "point_density"
        panels.append((title, dens, dens_kw))

    # Compute band grids

    # Bands to process
    if bands_prefix is not None:
        bands = [c for c in df_merged.columns if isinstance(c, str) and c.startswith(bands_prefix)]
        bands = bands[:max_bands]
        if debug:
            logging.info(f"[plotting_raster] Bands discovered (limited to {max_bands}): {bands}")

        for b in bands:
            try:
                grid = _grid_mean_for_series(
                    df_merged[b],
                    m_coord,
                    x,
                    y,
                    xbins,
                    ybins,
                    counts_all,
                    fill_empty=fill_empty,
                    debug=debug,
                    name=b,
                )
                panels.append((b, grid, dict(cmap="viridis")))
            except Exception as e:
                logging.error(f"[plotting_raster] Failed to compute grid for {b}: {e}")
    if custom_columuns is not None:
        for b in custom_columuns:
            try:
                grid = _grid_mean_for_series(
                    df_merged[b],
                    m_coord,
                    x,
                    y,
                    xbins,
                    ybins,
                    counts_all,
                    fill_empty=fill_empty,
                    debug=debug,
                    name=b,
                )
                panels.append((b, grid, dict(cmap="viridis")))
            except Exception as e:
                logging.error(f"[plotting_raster] Failed to compute grid for {b}: {e}")


    # Elevation panel
    if "elev" in df_merged.columns:
        elev = _grid_mean_for_series(
            df_merged["elev"],
            m_coord,
            x,
            y,
            xbins,
            ybins,
            counts_all,
            fill_empty=fill_empty,
            debug=debug,
            name="elev",
        )
        try:
            vmin_e, vmax_e = np.nanpercentile(elev, clip)
        except Exception:
            vmin_e, vmax_e = (np.nan, np.nan)
        panels.append(("elevation", elev, dict(cmap="terrain", vmin=vmin_e, vmax=vmax_e)))
    else:
        logging.warning("[plotting_raster] Column 'elev' not found; elevation panel omitted.")

    # Layout and figure size
    n_panels = len(panels)
    cols = min(3, n_panels) if n_panels > 0 else 1
    rows = int(np.ceil(n_panels / cols)) if n_panels > 0 else 1
    extent = [xmin, xmax, ymin, ymax]

    if auto_figsize:
        figsize, _ = _auto_figsize(nx, ny, rows, cols, pixels_per_bin, dpi, min_panel_size, max_panel_size)
        if debug:
            logging.info(f"[plotting_raster] Auto figsize={figsize} at dpi={dpi}")
    else:
        figsize = (5 * cols, 4 * rows)
        if debug:
            logging.info(f"[plotting_raster] Fixed figsize={figsize}")

    # Plot and save panels
    panels_path = os.path.join(outdir, f"panels_{file_name}.png")
    _plot_panels(panels, extent, panels_path, rows, cols, dpi, figsize, debug)

    # Optional scatter quicklook
    if scatter_quicklook:
        scatter_path = os.path.join(outdir, f"scatter_quicklook_{file_name}.png")
        _scatter_quicklook(
            x,
            y,
            scatter_path,
            dpi,
            max_points=scatter_max,
            size=scatter_size,
            alpha=scatter_alpha,
            extent=(xmin, xmax, ymin, ymax),
            debug=debug,
        )

    # Band histograms
    if bands_prefix is not None:
        hist_path = os.path.join(outdir, f"band_distributions_{file_name}.png")
        _band_histograms(df_merged, bands, m_coord, hist_path, clip, dpi, debug)


    # Optional combined band KDE chart
    if band_kde and bands_prefix is not None:
        kde_path = os.path.join(outdir, f"band_kde_{file_name}.png")
        _band_kde_plot(
            df_merged,
            bands,
            kde_path,
            xlim=band_kde_xlim,
            points=band_kde_points,
            linewidth=band_kde_linewidth,
            colors=band_kde_colors,
            dpi=dpi,
            debug=debug,
        )

    if debug:
        logging.info("[plotting_raster] done.")