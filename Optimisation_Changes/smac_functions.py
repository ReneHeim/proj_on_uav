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
