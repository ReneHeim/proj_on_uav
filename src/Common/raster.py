#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# version ='1.0'
# ---------------------------------------------------------------------------

'''
This code copiles all Common that will be called later by other codes. For the sake of clarity
these Common are defined in this separated piece of code.
'''
import logging
import os

import affine
import matplotlib.pyplot as plt

import numpy as np
import math

from timeit import default_timer as timer
import polars as pl
import rasterio as rio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject, calculate_default_transform


def pixelToWorldCoords(pX, pY, geoTransform):
    ''' Input image pixel coordinates and get world coordinates according to geotransform using gdal
    '''

    def applyGeoTransform(inX, inY, geoTransform):
        outX = geoTransform[0] + inX * geoTransform[1] + inY * geoTransform[2]
        outY = geoTransform[3] + inX * geoTransform[4] + inY * geoTransform[5]
        return outX, outY

    mX, mY = applyGeoTransform(pX, pY, geoTransform)
    return mX, mY

def worldToPixelCoords(wX, wY, geoTransform, dtype='int'):
    ''' Input world coordinates and get pixel coordinates according to reverse geotransform using gdal
    '''
    reverse_transform = ~ affine.Affine.from_gdal(*geoTransform)
    px, py = reverse_transform * (wX, wY)
    if dtype == 'int':
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
    return np.array([transform.a,
                     transform.b,
                     transform.c,
                     transform.d,
                     transform.e,
                     transform.f, 0, 0, 1], dtype='float64').reshape((3,3))

def xy_np(transform, rows, cols, offset='center'):
    if isinstance(rows, int) and isinstance(cols, int):
        pts = np.array([[rows, cols, 1]]).T
    else:
        assert len(rows) == len(cols)
        pts = np.ones((3, len(rows)), dtype=int)
        pts[0] = rows
        pts[1] = cols

    if offset == 'center':
        coff, roff = (0.5, 0.5)
    elif offset == 'ul':
        coff, roff = (0, 0)
    elif offset == 'ur':
        coff, roff = (1, 0)
    elif offset == 'll':
        coff, roff = (0, 1)
    elif offset == 'lr':
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
            Xw, Yw = rio.transform.xy(rst.transform, rows_flat, cols_flat, offset='center')

            if transform_to_utm:
                transformer = Transformer.from_crs(rst.crs, target_crs, always_xy=True)
                x_trans, y_trans = transformer.transform(Xw, Yw)
                Xw = np.array(x_trans)
                Yw = np.array(y_trans)
            else:
                Xw = np.array(Xw)
                Yw = np.array(Yw)

            band_values = b_all.reshape(num_bands, -1).T

            data = {
                'Xw': pl.Series(Xw),
                'Yw': pl.Series(Yw)
            }
            for idx in range(num_bands):
                data[f'band{idx + 1}'] = pl.Series(band_values[:, idx])
            df_allbands = pl.DataFrame(data)


        end_bands = timer()
        logging.info(
            f"Processed orthophoto bands for {os.path.basename(each_ortho)} in {end_bands - start_bands:.2f} seconds")
        return df_allbands
    except Exception as e:
        logging.error(f"Error reading bands from {each_ortho}: {e}")
        raise



def coregister_and_resample(input_path, ref_path, output_path, target_resolution=None, resampling=Resampling.nearest):
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
                    src.crs,
                    ref_crs,
                    ref_width,
                    ref_height,
                    *ref_bounds
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
                dst_transform = rio.transform.from_bounds(x_min, y_min, x_max, y_max, dst_width, dst_height)
            else:
                # Use the reference transform and size directly
                dst_transform, dst_width, dst_height = ref_transform, ref_width, ref_height

            dst_kwargs = src.meta.copy()
            dst_kwargs.update({
                "crs": ref_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
            })

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
                        resampling=resampling
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
            if not (abs(x_offset - round(x_offset)) < 1e-6 and abs(y_offset - round(y_offset)) < 1e-6):
                logging.info(f"Pixel alignment mismatch: x_offset={x_offset}, y_offset={y_offset}")
                return False

            # If we reach here, DEM and orthophoto are aligned
            logging.info(f" DEM and orthophoto are aligned: x_offset={x_offset}, y_offset={y_offset}")
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
    a = 6378137.0                       # semi-major axis (meters)
    f = 1 / 298.257223563               # flattening
    e2 = 2*f - f**2                     # eccentricity squared
    e = math.sqrt(e2)

    # UTM parameters for Zone 32N
    k0 = 0.9996
    E0 = 500000.0                       # false easting
    N0 = 0.0                            # false northing (northern hemisphere)
    lambda0 = math.radians(9.0)         # central meridian for Zone 32N (9°E)

    # Convert input latitude and longitude from degrees to radians
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)

    # Compute auxiliary values
    N_val = a / math.sqrt(1 - e2 * math.sin(phi)**2)
    T = math.tan(phi)**2
    # Second eccentricity squared
    ep2 = e2 / (1 - e2)
    C = ep2 * math.cos(phi)**2
    A = (lam - lambda0) * math.cos(phi)

    # Meridional arc length (M)
    M = a * (
          (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * phi
        - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * math.sin(2*phi)
        + (15*e2**2/256 + 45*e2**3/1024) * math.sin(4*phi)
        - (35*e2**3/3072) * math.sin(6*phi)
    )

    # Calculate Easting and Northing using standard UTM series formulas
    easting = E0 + k0 * N_val * (
          A
        + (1 - T + C) * A**3 / 6
        + (5 - 18*T + T**2 + 72*C - 58*ep2) * A**5 / 120
    )

    northing = N0 + k0 * (
          M
        + N_val * math.tan(phi) * (
              A**2 / 2
            + (5 - T + 9*C + 4*C**2) * A**4 / 24
            + (61 - 58*T + T**2 + 600*C - 330*ep2) * A**6 / 720
        )
    )

    return easting, northing

def plotting_raster(df_merged, path, file_name, bands_prefix="band", nx=1500, ny=1500, max_bands=6, clip=(2, 98)):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    outdir = os.path.join(path, "bands_data")
    os.makedirs(outdir, exist_ok=True)

    x = df_merged["Xw"].to_numpy()
    y = df_merged["Yw"].to_numpy()
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]

    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    xbins = np.linspace(xmin, xmax, nx + 1)
    ybins = np.linspace(ymin, ymax, ny + 1)
    counts, _, _ = np.histogram2d(y, x, bins=[ybins, xbins])

    def grid_mean(series):
        v = series.to_numpy()[m]
        vm = np.isfinite(v)
        sums, _, _ = np.histogram2d(y[vm], x[vm], bins=[ybins, xbins], weights=v[vm])
        return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)

    bands = [c for c in df_merged.columns if c.startswith(bands_prefix)][:max_bands]
    means = [(b, grid_mean(df_merged[b])) for b in bands]
    elev = grid_mean(df_merged["elev"])

    n_panels = len(means) + 1
    cols = min(3, n_panels)
    rows = int(np.ceil(n_panels / cols))
    extent = [xmin, xmax, ymin, ymax]

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    for ax, (band, grid) in zip(axes.ravel(), means):
        im = ax.imshow(grid, origin="lower", extent=extent, cmap="viridis", interpolation="bilinear")
        ax.set_title(band)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        fig.colorbar(im, ax=ax, shrink=0.85)

    lo, hi = np.nanpercentile(elev, clip)
    ax_elev = axes.ravel()[len(means)]
    im = ax_elev.imshow(elev, origin="lower", extent=extent, cmap="terrain", vmin=lo, vmax=hi, interpolation="bilinear")
    ax_elev.set_title("Elevation")
    ax_elev.set_xlabel("X (m)")
    ax_elev.set_ylabel("Y (m)")
    fig.colorbar(im, ax=ax_elev, shrink=0.85)

    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"panels_{file_name}.png"), dpi=200)
    plt.close(fig)

    cols_h = min(3, len(bands))
    rows_h = int(np.ceil(len(bands) / cols_h))
    fig_h, axes_h = plt.subplots(rows_h, cols_h, figsize=(5 * cols_h, 3.5 * rows_h), squeeze=False)
    for ax, b in zip(axes_h.ravel(), bands):
        v = df_merged[b].to_numpy()[m]
        v = v[np.isfinite(v)]
        if v.size:
            lo, hi = np.percentile(v, clip)
            ax.hist(v, bins=50, range=(lo, hi))
        ax.set_title(b)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    for ax in axes_h.ravel()[len(bands):]:
        ax.axis("off")

    fig_h.tight_layout()
    fig_h.savefig(os.path.join(outdir, f"band_distributions_{file_name}.png"), dpi=200)
    plt.close(fig_h)
