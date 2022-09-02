#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/06/22
# version ='1.0'
# ---------------------------------------------------------------------------
"""
This script retrieves pixel-wise reflectance values from Metashape orthophotos. For each pixel, the
observation/illumination angles will be calculated based on Roosjen, 2017. Orthophotos are used instead of
ortho mosaics as the camera positions are available for each ortho photo but not for the mosaic. The pixel resolution
is determined by the orthophotos and dem.

The following input variables are required to build a oblique-reflectance data frame:

    - camera positions as omega, phi, kappa txt file (Agisoft Metashape)
    - digital elevation model (dem)
    - list of ortho photos (same crs as dem)
    - ...
"""

# <editor-fold desc="01_loading libs and paths">
import pandas as pd
import glob
import os
from smac_functions import *
import rasterio as rio
from functools import reduce, partial
import math
import numpy as np
import exiftool as exif
from tqdm import tqdm
from pathlib import Path, PureWindowsPath
from timeit import default_timer as timer

# section 1: loading paths

out = r"D:\on_uav_data\out"

cam_path = r"D:\on_uav_data\20200906_Bot\agisoft\20200906_bot_corn_comb_cameras.txt"
dem_path = r"D:\on_uav_data\20200906_Bot\agisoft\20200906_bot_corn_comb_10cm_dem.tif"

ori = [r"D:\on_uav_data\20200906_Bot\multi\fl2_nad",
       r"D:\on_uav_data\20200906_Bot\multi\fl3_oblique",
       r"D:\on_uav_data\20200906_Bot\multi\fl4_nad"]
# </editor-fold>

# <editor-fold desc="02_splitting loaded data into chunks">
path_list = []  # listing all orthophoto paths
path_list = glob.glob(r'D:\on_uav_data\out\otsu\*.tif')
chunks = np.array_split(path_list, 15)
# </editor-fold>

# <editor-fold desc="03_retrieving pixel coords for DEM">
# section 2: get XY + value for each dem
start = timer()

with rio.open(dem_path) as dem:
    d1 = dem.read(1)
    arr_dem = np.array(d1)

Xp_dem, Yp_dem, val_dem = xyval(arr_dem)

res_dem = []
with rio.open(dem_path) as dem_layer:
    for i, j in zip(Xp_dem, Yp_dem):
        # coords2pixels = map_layer.index(235059.32,810006.31)
        res_dem.append(dem_layer.xy(i, j))

    #res_dem = dem_layer.xy() for zip(i, j) in Xp_dem

    df = pd.DataFrame(res_dem)
    df_dem = pd.concat([pd.Series(val_dem), df], axis=1)
    df_dem.columns = ['elev', 'Xw', 'Yw']
    df_dem = df_dem.round({'elev': 2, 'Xw': 1, 'Yw': 1})

end = timer()
print('Break DEM into pixel: ', end - start, 'seconds')
# </editor-fold>

for iteration, chunk in enumerate(chunks):

    df_list = []
    ori_list = []  # listing all multispectral images containing exif data

    for each_ortho in tqdm(chunk):  # select just a few in chunk to test loop

        start2 = timer()
        # section 3: Importing camera positions and the associated orthophotos

        campos = pd.read_csv(cam_path, sep='\t', skiprows=2, header=None)
        campos.columns = ['PhotoID', 'X', 'Y', 'Z', 'Omega', 'Phi', 'Kappa', 'r11', 'r12', 'r13', 'r21',
                          'r22', 'r23', 'r31', 'r32', 'r33']

        path, file = os.path.split(each_ortho)
        name, ext = os.path.splitext(file)

        reduced_string = file[5:]
        reduced_name = name[5:]

        campos1 = campos[campos['PhotoID'].str.match(reduced_name)]
        xcam = campos1['X'].values  # easting
        ycam = campos1['Y'].values  # northing
        zcam = campos1['Z'].values

        del campos, campos1  # trying to reduce memory load

        end2 = timer()
        print('Finding camera position for first ortho in first chunk: ', end2 - start2, 'seconds')

        start3 = timer()
        # section 4: Get sun elevation and sun azimuth angle

        for item in ori:
            ori_list.append(glob.glob(item + "//*.tif"))  # paths from all non-calibrated images as these have relevant EXIF

        def flatten(t):
            return [item for sublist in t for item in sublist]

        path_flat = flatten(ori_list)
        
        path_norm = []
        for i in path_flat:
            path_norm.append(str(PureWindowsPath(i)))
        

        filter_object = filter(lambda a: reduced_string in a, path_norm)

        # Convert the filter object to list
        exifobj = list(filter_object)

        with exif.ExifToolHelper() as et:
            #metdat = et.get_metadata(exifobj[0])
            #print(metdat)
            
            sunelev = et.get_tags(exifobj[0], 'XMP:SolarElevation')
            saa = et.get_tags(exifobj[0], 'XMP:SolarAzimuth')

        end3 = timer()
        print('Getting SAA and Sun Elevation from ortho EXIF data: ', end3 - start3, 'seconds')

        start4 = timer()
        # section 5: get XY + value for each ortho photo

        bands = {}

        with rio.open(each_ortho) as rst:

            for counter in range(1, 11, 1):

                res = []
                b1 = rst.read(counter)
                arr = np.array(b1)

                Xp, Yp, val = xyval(arr) # this function is loaded from smac_functions.py

                for i, j in zip(Xp, Yp):
                    # coords2pixels = map_layer.index(235059.32,810006.31)  # input lon,lat
                    res.append(rst.xy(i, j))  # input px, py -> see rasterio documentation

                df = pd.DataFrame(res)
                df_ortho = pd.concat([pd.Series(val), df], axis=1)
                df_ortho.columns = ['band'+str(counter), 'Xw', 'Yw']
                df_ortho = df_ortho.round({'Xw': 1, 'Yw': 1})  # rounding coords for matches; careful when aligning raster
                # outputs and original raster files.
                bands[f"band{counter}"] = df_ortho

        my_reduce = partial(pd.merge, on=["Xw", "Yw"], how='outer')
        df_allbands = reduce(my_reduce, bands.values())

        end4 = timer()
        print('Break all ortho bands into pixel: ', end4 - start4, 'seconds')

        #start5 = timer()
        # section 6: get XY + value for each dem

        # with rio.open(dem_path) as dem:
        #     d1 = dem.read(1)
        #     arr_dem = np.array(d1)
        #
        # Xp_dem, Yp_dem, val_dem = xyval(arr_dem)
        #
        # res_dem = []
        # with rio.open(dem_path) as dem_layer:
        #     start5_1 = timer()
        #     for i, j in zip(Xp_dem, Yp_dem):
        #         # coords2pixels = map_layer.index(235059.32,810006.31)
        #         res_dem.append(dem_layer.xy(i, j))
        #     end5_1 = timer()
        #     print(end5_1 - start5_1, 'seconds for 5_1 section')
        #
        #     #res_dem = dem_layer.xy() for zip(i, j) in Xp_dem
        #
        #     df = pd.DataFrame(res_dem)
        #     df_dem = pd.concat([pd.Series(val_dem), df], axis=1)
        #     df_dem.columns = ['elev', 'Xw', 'Yw']
        #     df_dem = df_dem.round({'elev': 2, 'Xw': 1, 'Yw': 1})
        #
        # end5 = timer()
        # print(end5 - start5, 'seconds for fifth section')

        # section 7: Calculate slope using the dem

        # gdal.DEMProcessing('slope.tif', dem_path, 'slope')
        #
        # with rio.open('slope.tif') as dataset:
        #     slope = dataset.read(1)
        #
        # Xp_slp, Yp_slp, val_slp = xyval(slope)
        #
        # res_slp = []
        # with rio.open(dem_path) as slp_layer:
        #     for i, j in zip(Xp_slp, Yp_slp):
        #         # coords2pixels = map_layer.index(235059.32,810006.31)
        #         res_slp.append(slp_layer.xy(i, j))  # input px, py
        #
        #     df = pd.DataFrame(res_slp)
        #     df_slp = pd.concat([pd.Series(val_slp), df], axis=1)
        #     df_slp.columns = ['slp', 'Xw', 'Yw']
        #     df_slp = df_slp.round({'slp': 2, 'Xw': 1, 'Yw': 1})

        # section 8: calculate aspect using the dem

        # gdal.DEMProcessing('aspect.tif', dem_path, 'aspect')
        #
        # with rio.open('aspect.tif') as dataset:
        #     aspect = dataset.read(1)
        #
        # Xp_asp, Yp_asp, val_asp = xyval(aspect) # this function is loaded from smac functions
        #
        # res_asp = []
        # with rio.open(dem_path) as asp_layer:
        #     for i, j in zip(Xp_asp, Yp_asp):
        #         # coords2pixels = map_layer.index(235059.32,810006.31)
        #         res_asp.append(asp_layer.xy(i, j))  # input px, py
        #
        #     df = pd.DataFrame(res_asp)
        #     df_asp = pd.concat([pd.Series(val_asp), df], axis=1) #[pd.Series(Xp_asp), pd.Series(Yp_asp), pd.Series(val_asp), df]
        #     df_asp.columns = ['asp', 'Xw', 'Yw'] #['Xp_asp', 'Yp_asp', 'asp', 'Xw', 'Yw']
        #     df_asp = df_asp.round({'asp': 2, 'Xw': 1, 'Yw': 1})

        start6 = timer()
        # section 9: Combine ortho values and DEM values while setting NaN's

        dfs = [df_dem, df_allbands]  #, df_slp, df_asp] TODO: slp and asp not needed without correcting vza

        df_merged = reduce(lambda left, right: pd.merge(left, right, on=["Xw", "Yw"]), dfs)  # combine all dfs above

        #df_merged['asp'] = np.where((df_merged['band1'] == 65535), np.nan, df_merged['asp'])
        #df_merged['slp'] = np.where((df_merged['band1'] == 65535), np.nan, df_merged['slp'])

        #df_merged['slp_rad'] = round((df_merged['slp']/180)*math.pi, 2) #convert to rad -> 1 rad = (deg/180)*pi
        #df_merged['asp_rad'] = round((df_merged['asp']/180)*math.pi, 2) #convert to rad

        # section 10: calculate VZA

        df_merged['vza'] = df_merged.apply(lambda x: np.arctan((zcam[0] - x.elev)/math.sqrt(math.pow(xcam[0] - x.Xw, 2)+math.pow(ycam[0] - x.Yw, 2))), axis=1)
        df_merged['vza'] = round(90 - (df_merged['vza'] * (180/math.pi)), 2)  # substr 90° to get correct angle
        #df_merged['vza_rad'] = round((df_merged['vza']/180)*math.pi, 2)

        df_merged['vza'] = np.where((df_merged['band1'] == 65535), np.nan, df_merged['vza'])  # set nan when band = nan
        #df_merged['vza_rad'] = np.where((df_merged['band1'] == 65535), np.nan, df_merged['vza_rad'])

        # section 11: calculate VAA

        df_merged['vaa_rad'] = df_merged.apply(lambda x: 2 * math.atan((x.Xw - xcam[0])/(math.sqrt((x.Xw - xcam[0])**2 + (x.Yw - ycam[0])**2)+x.Yw - ycam[0])), axis=1)
        df_merged['vaa'] = round(180 + (df_merged['vaa_rad'] * (180/math.pi)), 2)

        df_merged['vaa'] = np.where((df_merged['band1'] == 65535), np.nan, df_merged['vaa'])  # set nan when band = nan
        df_merged['vaa_rad'] = np.where((df_merged['band1'] == 65535), np.nan, df_merged['vaa_rad'])

        #df_merged['vaa'] = df_merged.apply(lambda x: np.arctan((x.Yw - ycam[0])/(x.Xw-xcam[0])), axis=1)
        #df_merged['vaa'] = 90-(df_merged['vaa'] * (180/math.pi))
        #df_merged['vaa_rad'] = (df_merged['vaa']/180)*math.pi
        #df_merged['vza_corr_rad'] = df_merged.apply(lambda x: np.cos(x.vza)*np.cos(x.slp)+np.sin(x.vza)*np.sin(x.slp)*np.cos(x.vaa-x.asp), axis=1)
        #df_merged['vza_corr'] = df_merged['vza_corr_rad'] * (180/math.pi)

        df_merged.insert(0, 'path', file)
        df_merged.insert(1,'xcam', xcam[0])
        df_merged.insert(2,'ycam', ycam[0])
        df_merged.insert(3, 'sunelev', round(float(sunelev[0]['XMP:SolarElevation']), 2))
        df_merged.insert(4, 'saa', round(float(saa[0]['XMP:SolarAzimuth']), 2))

        df_list.append(df_merged)
        #print(sys.getsizeof(df_list))
        

        end6 = timer()
        print('Combine DEM and ortho values + calculate VZA and VAA', end6 - start6, 'seconds')

    result = pd.concat(df_list)
    result.to_csv(f"{out}allorthopixels_10cm_band_part{iteration}.csv")

    del result, df_merged, df_allbands, df_list
