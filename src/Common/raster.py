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
def plotting_raster(
        df_merged,
        path,
        file_name,
        bands_prefix="band",
        nx=1500,
        ny=1500,
        max_bands=6,
        clip=(2, 98),
        debug=False,
        fill_empty=False,
        occupancy_target=None,
        min_bins=50,
        plot_density=False,
        density_log=False,
        density_cmap="magma",
        density_clip=(2, 98),
        scatter_quicklook=False,
        scatter_max=200000,
        scatter_alpha=0.25,
        scatter_size=1.0,
        density_vmin=None,
        density_vmax=None,
        density_discrete=False,
        auto_figsize=True,          # enable auto sizing based on grid density
        dpi=200,                    # DPI used for sizing; also used for saving
        pixels_per_bin=4,         # desired pixels per grid cell (bin)
        min_panel_size=(4.0, 3.5),  # per-panel min size in inches (w, h)
        max_panel_size=(60.0, 120.0) # per-panel max size in inches (w, h)
):
    """
    Parameters
    ----------
    df_merged : DataFrame-like
        Input table with at least:
        - Xw (float): world X coordinate (meters)
        - Yw (float): world Y coordinate (meters)
        - <bands_prefix>* (float): band columns (e.g., band1, band2, ...)
        - elev (float, optional): per-point elevation
        Supports Polars or Pandas.

    path : str
        Base output directory. A subfolder 'bands_data' will be created inside it.

    file_name : str
        Base name (without extension) used for saved image files.

    bands_prefix : str, default "band"
        Prefix used to auto-detect band columns to rasterize.

    nx : int, default 1500
        Number of grid bins along X (columns). Higher → finer grid, but sparser bins.

    ny : int, default 1500
        Number of grid bins along Y (rows).

    max_bands : int, default 6
        Maximum number of band panels rendered, even if more bands exist.

    clip : tuple(int, int), default (2, 98)
        Percentile clip for color scaling of band and elevation panels.

    debug : bool, default False
        If True, logs detailed diagnostics (counts, coverage, percentiles, file paths).

    fill_empty : bool, default False
        If True, fills NaN grid cells for visualization with a simple 3×3 neighborhood average.
        Recommended only for display; analytics should use the unfilled grid.

    occupancy_target : float or None, default None
        If set to a value in (0, 1], the function auto-coarsens nx, ny until at least this
        fraction of grid cells are non-empty. Useful to avoid overly sparse grids.

    min_bins : int, default 50
        Lower bound for nx and ny during auto-coarsening governed by occupancy_target.

    plot_density : bool, default False
        If True, includes a panel showing point density (2D bin counts).

    density_log : bool, default False
        If True, uses log1p scaling on density counts for visualization. Leave False when
        counts are already small (e.g., 0–6).

    density_cmap : str, default "magma"
        Colormap used for the density panel.

    density_clip : tuple(int, int), default (2, 98)
        Percentile clip for density color scaling when explicit vmin/vmax are not provided.

    scatter_quicklook : bool, default False
        If True, saves a raw XY scatter quicklook image (sampled if needed) to validate spatial distribution.

    scatter_max : int, default 200000
        Maximum number of points to draw in the scatter quicklook (random sample if exceeded).

    scatter_alpha : float, default 0.25
        Alpha (transparency) for scatter points in the quicklook.

    scatter_size : float, default 1.0
        Marker size for scatter points in the quicklook.

    density_vmin : float or None, default None
        Explicit minimum value for density color scale. For small integer counts, set to 0.

    density_vmax : float or None, default None
        Explicit maximum value for density color scale. For small integer counts, set to e.g. 6.

    density_discrete : bool, default False
        If True, uses discrete color levels for density (e.g., one color per integer count).

    auto_figsize : bool, default True
        If True, figure size adapts to grid density so each cell has enough pixels, reducing
        rendering artifacts for very fine grids.

    dpi : int, default 200
        DPI used both for figure rendering/saving and for computing automatic figure size.

    pixels_per_bin : float, default 4
        Desired pixels per grid cell when auto_figsize=True. Larger values produce larger, crisper panels.

    min_panel_size : tuple(float, float), default (4.0, 3.5)
        Minimum per-panel (width, height) in inches when auto sizing.

    max_panel_size : tuple(float, float), default (60.0, 120.0)
        Maximum per-panel (width, height) in inches when auto sizing.

    Notes
    -----
    - Per-cell means are computed using only finite band values and divide by the number
      of valid samples in each bin (band-specific counts).
    - Many NaNs indicate a grid that is too fine for point density; lower nx/ny,
      enable occupancy_target, or use fill_empty=True for display.
    - Outputs are saved as PNGs in '{path}/bands_data':
        * panels_<file_name>.png
        * band_distributions_<file_name>.png
        * scatter_quicklook_<file_name>.png (if scatter_quicklook=True)
    """



    if debug:
        logging.info(
            f"[plotting_raster] start: file_name={file_name}, nx={nx}, ny={ny}, "
            f"max_bands={max_bands}, clip={clip}, bands_prefix='{bands_prefix}', "
            f"fill_empty={fill_empty}, occupancy_target={occupancy_target}, "
            f"plot_density={plot_density}, density_log={density_log}, "
            f"auto_figsize={auto_figsize}, dpi={dpi}, pixels_per_bin={pixels_per_bin}"
        )

    # Basic presence/emptiness checks
    try:
        is_empty = df_merged is None or (hasattr(df_merged, "is_empty") and df_merged.is_empty())
    except Exception:
        is_empty = df_merged is None or (len(df_merged) == 0)

    if is_empty:
        logging.error(f"[plotting_raster] No data found for {file_name}")
        return

    # Column checks
    required_cols = ["Xw", "Yw"]
    missing_required = [c for c in required_cols if c not in df_merged.columns]
    if missing_required:
        logging.error(f"[plotting_raster] Missing required columns: {missing_required}")
        return

    # Prepare output
    outdir = os.path.join(path, "bands_data")
    os.makedirs(outdir, exist_ok=True)
    if debug:
        logging.info(f"[plotting_raster] Output directory: {outdir}")

    # Extract coordinates
    try:
        x = df_merged["Xw"].to_numpy()
        y = df_merged["Yw"].to_numpy()
    except Exception as e:
        logging.error(f"[plotting_raster] Failed to extract Xw/Yw arrays: {e}")
        return

    if debug:
        def safe_stats(arr, name):
            n = arr.size
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            n_finite = int(np.isfinite(arr).sum())
            msg = (
                f"[plotting_raster] {name}: size={n}, n_nan={n_nan}, n_inf={n_inf}, "
                f"n_finite={n_finite}"
            )
            try:
                if n_finite > 0:
                    msg += (
                        f", min={np.nanmin(arr):.3f}, max={np.nanmax(arr):.3f}, "
                        f"mean={np.nanmean(arr):.3f}"
                    )
            except Exception:
                pass
            logging.info(msg)
        safe_stats(x, "Xw")
        safe_stats(y, "Yw")

    # Mask invalid coordinates
    m = ~np.isnan(x) & ~np.isnan(y) & np.isfinite(x) & np.isfinite(y)
    if debug:
        logging.info(
            f"[plotting_raster] Coord mask: kept={int(m.sum())}/{x.size} "
            f"({(100*m.mean() if x.size else 0):.2f}%)"
        )
    x, y = x[m], y[m]

    if x.size == 0 or y.size == 0:
        logging.warning("[plotting_raster] No finite coordinates after masking; aborting.")
        return

    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    if debug:
        logging.info(
            f"[plotting_raster] Extent: xmin={xmin:.3f}, xmax={xmax:.3f}, "
            f"ymin={ymin:.3f}, ymax={ymax:.3f}"
        )

    def make_bins(nx, ny):
        xbins = np.linspace(xmin, xmax, nx + 1)
        ybins = np.linspace(ymin, ymax, ny + 1)
        return xbins, ybins

    xbins, ybins = make_bins(nx, ny)
    if debug:
        logging.info(
            f"[plotting_raster] Bins: nx+1={len(xbins)}, ny+1={len(ybins)}, "
            f"dx~={(xmax - xmin) / max(nx, 1):.3f}, dy~={(ymax - ymin) / max(ny, 1):.3f}"
        )

    # Compute bin counts (y first as rows, x as cols)
    counts_all, _, _ = np.histogram2d(y, x, bins=[ybins, xbins])
    total_points = int(counts_all.sum())
    zero_bins = int((counts_all == 0).sum())
    total_bins = counts_all.size
    occupancy = 1.0 - (zero_bins / total_bins if total_bins else 0.0)
    if debug:
        logging.info(
            f"[plotting_raster] 2D counts: points={total_points}, bins={total_bins}, "
            f"zero_bins={zero_bins} ({(100*zero_bins/total_bins if total_bins else 0):.2f}%), "
            f"occupancy={occupancy:.3f}"
        )

    # Optional adaptive coarsening to reach an occupancy target
    if occupancy_target is not None and 0 < occupancy_target <= 1.0 and occupancy < occupancy_target:
        if debug:
            logging.info(
                f"[plotting_raster] Adaptive coarsening to reach occupancy_target={occupancy_target}"
            )
        cur_nx, cur_ny = nx, ny
        while (cur_nx > min_bins or cur_ny > min_bins):
            cur_nx = max(min_bins, int(cur_nx * 0.8))
            cur_ny = max(min_bins, int(cur_ny * 0.8))
            xbins, ybins = make_bins(cur_nx, cur_ny)
            counts_all, _, _ = np.histogram2d(y, x, bins=[ybins, xbins])
            zero_bins = int((counts_all == 0).sum())
            total_bins = counts_all.size
            occupancy = 1.0 - (zero_bins / total_bins if total_bins else 0.0)
            if debug:
                logging.info(
                    f"[plotting_raster] Coarsened to {cur_nx}x{cur_ny} bins; occupancy={occupancy:.3f}"
                )
            if occupancy >= occupancy_target:
                nx, ny = cur_nx, cur_ny
                break
        if debug:
            logging.info(f"[plotting_raster] Final grid: nx={nx}, ny={ny}, occupancy={occupancy:.3f}")

    # Recompute bins/counts if changed
    xbins, ybins = make_bins(nx, ny)
    counts_all, _, _ = np.histogram2d(y, x, bins=[ybins, xbins])

    def grid_mean(series, name="unknown"):
        v_full = series.to_numpy()
        v = v_full[m]
        vm = np.isfinite(v)

        if debug:
            n = v.size
            n_finite = int(vm.sum())
            n_nan = int(np.isnan(v).sum())
            logging.info(
                f"[plotting_raster] {name}: values after coord-mask size={n}, "
                f"finite={n_finite} ({(100*n_finite/max(n,1)):.2f}%), nan={n_nan}"
            )
            if n_finite > 0:
                try:
                    logging.info(
                        f"[plotting_raster] {name}: min={np.nanmin(v):.6g}, "
                        f"max={np.nanmax(v):.6g}, mean={np.nanmean(v):.6g}"
                    )
                except Exception:
                    pass

        # Weighted sums and VALID counts per grid cell
        sums, _, _ = np.histogram2d(y[vm], x[vm], bins=[ybins, xbins], weights=v[vm])
        counts_valid, _, _ = np.histogram2d(y[vm], x[vm], bins=[ybins, xbins])

        # Mean over valid values only
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
                f"nan_cells={n_nan_cells}, "
                f"zero_bins_all={zero_bins_all}, zero_bins_valid={zero_bins_valid}"
            )
            if n_finite_cells > 0:
                try:
                    p2, p50, p98 = np.nanpercentile(grid, [2, 50, 98])
                    logging.info(
                        f"[plotting_raster] {name}: grid percentiles p2={p2:.6g}, "
                        f"p50={p50:.6g}, p98={p98:.6g}"
                    )
                except Exception:
                    pass

        # Optional simple hole-filling
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

    # Select bands
    bands = [c for c in df_merged.columns if isinstance(c, str) and c.startswith(bands_prefix)]
    bands = bands[:max_bands]
    if debug:
        logging.info(f"[plotting_raster] Bands discovered (limited to {max_bands}): {bands}")

    # Build panels (optional density, band grids, elevation)
    panels = []

    # Density panel
    if plot_density:
        if density_log:
            density_grid = np.log1p(counts_all.astype(float))
            density_desc = "log1p(counts)"
        else:
            density_grid = counts_all.astype(float)
            density_desc = "counts"

        if density_vmin is not None or density_vmax is not None:
            vmin = 0.0 if density_vmin is None else float(density_vmin)
            vmax = float(np.nanmax(density_grid)) if density_vmax is None else float(density_vmax)
        else:
            try:
                vmin, vmax = np.nanpercentile(density_grid, density_clip)
            except Exception:
                vmin, vmax = (np.nanmin(density_grid), np.nanmax(density_grid))

        if density_discrete:
            import matplotlib as mpl
            boundaries = np.arange(np.floor(vmin), np.ceil(vmax) + 1)
            norm = mpl.colors.BoundaryNorm(boundaries=boundaries, ncolors=256)
            density_kwargs = dict(cmap=density_cmap, norm=norm)
        else:
            density_kwargs = dict(cmap=density_cmap, vmin=vmin, vmax=vmax)

        if debug:
            logging.info(
                f"[plotting_raster] Density panel: {density_desc}, "
                f"range=({vmin},{vmax}), discrete={density_discrete}, "
                f"max_count={int(np.nanmax(counts_all))}"
            )

        panels.append(("point_density", density_grid, density_kwargs))

    # Band means
    means = []
    for b in bands:
        try:
            means.append((b, grid_mean(df_merged[b], name=b)))
        except Exception as e:
            logging.error(f"[plotting_raster] Failed to compute grid for {b}: {e}")
    panels.extend((b, g, dict(cmap="viridis")) for (b, g) in means)

    # Elevation
    if "elev" not in df_merged.columns:
        logging.warning("[plotting_raster] Column 'elev' not found; elevation panel will be empty.")
        elev = np.full((ny, nx), np.nan)
    else:
        elev = grid_mean(df_merged["elev"], name="elev")
    try:
        lo_e, hi_e = np.nanpercentile(elev, clip)
    except Exception:
        lo_e, hi_e = (np.nan, np.nan)
    panels.append(("elevation", elev, dict(cmap="terrain", vmin=lo_e, vmax=hi_e)))

    # Figure layout
    n_panels = len(panels)
    cols = min(3, n_panels)
    rows = int(np.ceil(n_panels / cols))
    extent = [xmin, xmax, ymin, ymax]

    # Auto figure size that respects bin density
    def clamp(val, lo, hi):
        return max(lo, min(val, hi))

    if auto_figsize:
        # Desired per-panel size from grid density (pixels per bin -> inches)
        panel_w_in = clamp((nx * pixels_per_bin) / dpi, min_panel_size[0], max_panel_size[0])
        panel_h_in = clamp((ny * pixels_per_bin) / dpi, min_panel_size[1], max_panel_size[1])
        # Total figure size
        fig_w = cols * panel_w_in
        fig_h = rows * panel_h_in
        if debug:
            logging.info(
                f"[plotting_raster] Auto figsize: panel=({panel_w_in:.2f}in, {panel_h_in:.2f}in), "
                f"figure=({fig_w:.2f}in, {fig_h:.2f}in) at dpi={dpi}"
            )
        figsize = (fig_w, fig_h)
    else:
        # Fallback to legacy sizing
        figsize = (5 * cols, 4 * rows)
        if debug:
            logging.info(f"[plotting_raster] Fixed figsize: {figsize}")

    # Create panels figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, dpi=dpi)
    for ax, (title, grid, kw) in zip(axes.ravel(), panels):
        im = ax.imshow(grid, origin="lower", extent=extent, interpolation="bilinear", **kw)
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        fig.colorbar(im, ax=ax, shrink=0.85)

    # Hide unused axes
    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")

    fig.tight_layout()
    panels_path = os.path.join(outdir, f"panels_{file_name}.png")
    fig.savefig(panels_path, dpi=dpi)
    if debug:
        logging.info(f"[plotting_raster] Saved panels to: {panels_path}")
    plt.close(fig)

    # Optional raw XY scatter quicklook
    if scatter_quicklook:
        try:
            n = x.size
            if n > scatter_max:
                idx = np.random.choice(n, size=scatter_max, replace=False)
                xs, ys = x[idx], y[idx]
                sampled = scatter_max
            else:
                xs, ys = x, y
                sampled = n

            if debug:
                logging.info(f"[plotting_raster] Scatter quicklook: plotting {sampled} points")

            # Scale scatter figure to preserve aspect and visibility
            width_m = xmax - xmin
            height_m = ymax - ymin
            aspect = width_m / max(height_m, 1e-9)
            base_w = 10.0
            fig_w = base_w
            fig_h = max(6.0, base_w / max(aspect, 1e-6))
            fig_s, ax_s = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax_s.scatter(xs, ys, s=scatter_size, alpha=scatter_alpha, c="k")
            ax_s.set_title("XY point scatter quicklook")
            ax_s.set_xlabel("X (m)")
            ax_s.set_ylabel("Y (m)")
            ax_s.set_aspect("equal", adjustable="box")
            fig_s.tight_layout()
            scatter_path = os.path.join(outdir, f"scatter_quicklook_{file_name}.png")
            fig_s.savefig(scatter_path, dpi=dpi)
            if debug:
                logging.info(f"[plotting_raster] Saved scatter quicklook to: {scatter_path}")
            plt.close(fig_s)
        except Exception as e:
            logging.error(f"[plotting_raster] Failed scatter quicklook: {e}")

    # Histograms for band distributions
    cols_h = min(3, len(bands))
    rows_h = int(np.ceil(len(bands) / cols_h)) if cols_h > 0 else 1
    # Modest size; independent of grid density
    fig_h, axes_h = plt.subplots(rows_h, cols_h, figsize=(5 * cols_h, 3.5 * rows_h), squeeze=False, dpi=dpi)

    for ax, b in zip(axes_h.ravel(), bands):
        try:
            v = df_merged[b].to_numpy()[m]
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

    for ax in axes_h.ravel()[len(bands):]:
        ax.axis("off")

    fig_h.tight_layout()
    hist_path = os.path.join(outdir, f"band_distributions_{file_name}.png")
    fig_h.savefig(hist_path, dpi=dpi)
    if debug:
        logging.info(f"[plotting_raster] Saved histograms to: {hist_path}")
    plt.close(fig_h)

    if debug:
        logging.info("[plotting_raster] done.")