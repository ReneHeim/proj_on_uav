#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# version ='1.0'
# ---------------------------------------------------------------------------

'''
This code copiles all functions that will be called later by other codes. For the sake of clarity
these functions are defined in this separated piece of code.
'''


import numpy as np
import math
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