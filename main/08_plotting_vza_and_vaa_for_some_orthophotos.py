#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# version ='1.0'
# ---------------------------------------------------------------------------

'''
This code coputes the Viewing Zenith Angle (VZA) and the Viewing Azimuth Angle (VAA) for 2 sample images and stores computed values in an array with similar coordinate system as
the reference image. A function is written that takes 4 inputs, namely: the path to the orthorectified image, the path to the same image while not yet orthorectified (with all
relevant exif data), the path to the Degital Elevation Model (DEM), and the path to the file containing camera positions. The code outputs the VZA, VAA raster files and a KML vector
file containing the location of the camera
'''

from smac_functions import *
import shutil
import os

def plot_vza_image(im_path, 
                   im_exif = r"D:\on_uav_data\raw\20200906_Bot\multi\fl2_nad\20200906_151812_IMG_0150_6.tif",
                   dem_path = r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\All_flight\20200906_bot_corn_comb_1m_dem.tif", 
                   cam_path = r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\All_flight\20200906_bot_corn_comb_cameras.txt"):

    # Load modules      
    from tqdm import tqdm
    import pandas as pd
    import glob
    import os
    import rasterio as rio
    from functools import reduce, partial
    import math
    import numpy as np
    import exiftool as exif
    from pathlib import Path, PureWindowsPath
    from timeit import default_timer as timer
    import pyarrow
    import matplotlib.pyplot as plt
    import simplekml
    
    # Specify output directory
    out = r"D:\on_uav_data\proc\Sample plot vza vaa"

    # Open DEM and turn it in a table with 3 columns
    start = timer()
    with rio.open(dem_path) as dem:
        d1 = dem.read(1)
        arr_dem = np.array(d1)
    Xp_dem, Yp_dem, val_dem = xyval(arr_dem)
    res_dem = []
    with rio.open(dem_path) as dem_layer:
        for i, j in zip(Xp_dem, Yp_dem):
            res_dem.append(dem_layer.xy(i, j))
        df = pd.DataFrame(res_dem)
        df_dem = pd.concat([pd.Series(val_dem), df], axis=1)
        df_dem.columns = ['elev', 'Xw', 'Yw']
        df_dem = df_dem.round({'elev': 2, 'Xw': 1, 'Yw': 1})
    end = timer()
    print('Break DEM into pixel: ', end - start, 'seconds')
   
    
    # Importing camera positions and the associated orthophotos
    start2 = timer()
    campos = pd.read_csv(cam_path, sep='\t', skiprows=2, header=None)
    campos.columns = ['PhotoID', 'X', 'Y', 'Z', 'Omega', 'Phi', 'Kappa', 'r11', 'r12', 'r13', 'r21',
                      'r22', 'r23', 'r31', 'r32', 'r33']
    path, file = os.path.split(im_path)
    name, ext = os.path.splitext(file)
    campos1 = campos[campos['PhotoID'].str.match(name)] 
    xcam = campos1['X'].values  # easting
    ycam = campos1['Y'].values  # northing
    zcam = campos1['Z'].values
    del campos, campos1  # trying to reduce memory load
    end2 = timer()
    print('Finding camera position for first ortho in first chunk: ', end2 - start2, 'seconds')

    # Open orthophoto and turn it in a taale with 12 columns (2 for x and y, and 1 for each of the 10 bands)
    start3 = timer()
    bands = {}
    with rio.open(im_path) as rst:
        h = rst.height
        w = rst.width
        crs = rst.crs
        transform = rst.transform
        for counter in range(1, 11, 1):
            res = []
            b1 = rst.read(counter)
            arr = np.array(b1)
            Xp, Yp, val = xyval(arr) # this function is loaded from smac_functions.py
            for i, j in zip(Xp, Yp):
                res.append(rst.xy(i, j))  # input px, py -> see rasterio documentation
            df = pd.DataFrame(res)
            df_ortho = pd.concat([pd.Series(val), df], axis=1)
            df_ortho.columns = ['band'+str(counter), 'Xw', 'Yw']
            df_ortho = df_ortho.round({'Xw': 1, 'Yw': 1})  # rounding coords for matches; careful when aligning raster
            # outputs and original raster files.
            bands[f"band{counter}"] = df_ortho
        my_reduce = partial(pd.merge, on=["Xw", "Yw"], how='outer') 
        df_allbands = reduce(my_reduce, bands.values())
        end3 = timer()
        print('Break all ortho bands into pixel: ', end3 - start3, 'seconds')

        # Combine ortho values and DEM values while setting NaN's
        start4 = timer()
        dfs = [df_dem, df_allbands]  
        df_merged = reduce(lambda left, right: pd.merge(left, right, on=["Xw", "Yw"]), dfs)  # combine all dfs above
        exifobj = im_exif
        with exif.ExifToolHelper() as et:
            sunelev = float(et.get_tags(exifobj, 'XMP:SolarElevation')[0]['XMP:SolarElevation'])* (180/math.pi)
            saa = float(et.get_tags(exifobj, 'XMP:SolarAzimuth')[0]['XMP:SolarAzimuth'])* (180/math.pi)
        end4 = timer()
        print('Combine ortho values and DEM: ', end4 - start4, 'seconds')
        

        # Calculate VZA 
        df_merged['vza'] = df_merged.apply(lambda x: np.arctan((zcam[0] - x.elev)/math.sqrt(math.pow(xcam[0] - x.Xw, 2)+math.pow(ycam[0] - x.Yw, 2))), axis=1)
        df_merged['vza'] = round(90 - (df_merged['vza'] * (180/math.pi)), 2)  # substr 90° to get correct angle
        df_merged['vza'] = np.where((df_merged['band1'] == 65535), np.nan, df_merged['vza'])  # set nan when band = nan

        # Calculate VAA
        df_merged['vaa_rad'] = df_merged.apply(lambda x: 2 * math.atan((x.Xw - xcam[0])/(math.sqrt((x.Xw - xcam[0])**2 + (x.Yw - ycam[0])**2)+x.Yw - ycam[0])), axis=1)
        df_merged['vaa'] = round(180 + (df_merged['vaa_rad'] * (180/math.pi)), 2) - saa
        df_merged['vaa'] = np.where((df_merged['band1'] == 65535), np.nan, df_merged['vaa'])  # set nan when band = nan
        df_merged['vaa_rad'] = np.where((df_merged['band1'] == 65535), np.nan, df_merged['vaa_rad'])
        
        # Calculate VAA acoording to Roosjen (2018)
        df_merged['vaa_rad_roosjen'] = df_merged.apply(lambda x: math.acos((ycam[0] - x.Yw)/(math.sqrt((x.Xw - xcam[0])**2 + (x.Yw - ycam[0])**2))) if x.Xw - xcam[0] < 0 else - math.acos((ycam[0] - x.Yw)/(math.sqrt((x.Xw - xcam[0])**2 + (x.Yw - ycam[0])**2))), axis=1)
        df_merged['vaa_roosjen'] = np.where (round((df_merged['vaa_rad_roosjen'] * (180/math.pi)), 2) - saa > -180, round((df_merged['vaa_rad_roosjen'] * (180/math.pi)), 2) - saa, round((df_merged['vaa_rad_roosjen'] * (180/math.pi)), 2) - saa + 360)
        df_merged['vaa_roosjen'] = np.where((df_merged['band1'] == 65535), np.nan, df_merged['vaa_roosjen'])  # set nan when band = nan
        df_merged['vaa_rad_roosjen'] = np.where((df_merged['band1'] == 65535), np.nan, df_merged['vaa_rad_roosjen'])
 
        # Insert camera position values
        df_merged.insert(1,'xcam', xcam[0])
        df_merged.insert(2,'ycam', ycam[0])
  
    result = df_merged
    del  df_allbands
    
    # Save vza and reopen to view it
    arr_vza = result.vza.to_numpy()
    arr_vza = np.reshape(arr_vza, (h, w))
    with rio.open(os.path.join(out, 'vza_' + file ), "w",
               driver="GTiff",
               height=h,
               width=w,
               count=1,
               dtype=arr_vza.dtype,
               crs=crs,
               transform=transform) as dst:
        dst.write(arr_vza, indexes=1)  # indexes must match count
    with rio.open(im_path) as rst:
       RGB_image = rst.read([6,4,2])/20000
    RGB_image = np.stack(RGB_image, axis=2)
    
    fig, axs = plt.subplots(ncols = 2)
    axs[0].imshow(RGB_image)
    axs[1].imshow(arr_vza)
    plt.show()
    
    # Save vaa and reopen to view it
    arr_vaa = result.vaa.to_numpy()
    arr_vaa = np.reshape(arr_vaa, (h, w))
    with rio.open(os.path.join(out, 'vaa_' + file ), "w",
               driver="GTiff",
               height=h,
               width=w,
               count=1,
               dtype=arr_vza.dtype,
               crs=crs,
               transform=transform) as dst:
        dst.write(arr_vaa, indexes=1)  # indexes must match count
    with rio.open(im_path) as rst:
       RGB_image = rst.read([6,4,2])/20000
    RGB_image = np.stack(RGB_image, axis=2)
    
    fig, axs = plt.subplots(ncols = 2)
    axs[0].imshow(RGB_image)
    axs[1].imshow(arr_vaa)
    plt.show()
    
    # Save vaa roosjen and reopen to view it
    arr_vaa_roosjen = result.vaa_roosjen.to_numpy()
    arr_vaa_roosjen = np.reshape(arr_vaa_roosjen, (h, w))
    with rio.open(os.path.join(out, 'vaa_roosjen_' + file ), "w",
               driver="GTiff",
               height=h,
               width=w,
               count=1,
               dtype=arr_vza.dtype,
               crs=crs,
               transform=transform) as dst:
        dst.write(arr_vaa_roosjen, indexes=1)  # indexes must match count
    with rio.open(im_path) as rst:
       RGB_image = rst.read([6,4,2])/20000
    RGB_image = np.stack(RGB_image, axis=2)
    
    fig, axs = plt.subplots(ncols = 2)
    axs[0].imshow(RGB_image)
    axs[1].imshow(arr_vaa_roosjen)
    plt.show()
    
    # Save comera position in KML file
    shp = simplekml.Kml()
    shp.newpoint(name = 'camera position', coords = [(xcam[0], ycam[0])])
    shp.save(os.path.join(out, 'vza_' + name + ".kml" ))
    # End of function

# Specify input images
Images = [r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\All_flight\20200906_bot_corn_comb_1m_orthophotos\20200906_151812_IMG_0150_6.tif",
          r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\All_flight\20200906_bot_corn_comb_1m_orthophotos\20200906_162448_IMG_1457_6.tif"]

# Specify output directory
out = r"D:\on_uav_data\proc\Sample plot vza vaa"

# Loop through images using the defined function
for image in Images:
    plot_vza_image(image)
    shutil.copy(image, os.path.join(out, os.path.split(image)[1]))