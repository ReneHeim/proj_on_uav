#!/usr/bin/env python3
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim and Nathan Okole
# Created Date: 2023/02/07
# version ='1.2'
# ---------------------------------------------------------------------------

"""
This script unlocks pixel-wise reflectance values from Metashape orthophotos. For each pixel, the
view zenith and azimuth angles will be calculated. Orthophotos are used instead of
orthomosaics as the camera positions are available for each ortho photo but not for the mosaic. The pixel resolution
is determined by the orthophotos and the digital elevation model (dem).

The following input variables are required to build a oblique-reflectance data frame:

    - camera positions as omega, phi, kappa txt file (Agisoft Metashape)
    - digital elevation model (dem)
    - list of ortho photos (same crs as dem)
    - ...
    
This codes implements parallelization to speed up the process using the multiprocessing module
"""

# Load libraries 
import pandas as pd
import glob
import os
from smac_functions import *
from functools import reduce, partial
import numpy as np
from joblib import Parallel, delayed


# Define all necessary directories in a list of dictionnaries for 2 dates, 2 datasets (all flights together and nadir flight only) and 3 resolutions (6 elements)
# Each Dictionnary is comprised of 6 elements:
#   the output directory
#   the camera position textfile
#   the directory to the dem
#   the directory to the original images that still have their exif data
#   the name under which the output data will be saved
#   the directory to orthophotos
sources =[      
          {'out': r"D:\on_uav_data\proc\extract\20200906_Bot\20200906_bot_corn_comb_20cm\All_flight",
            'cam_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_20cm\All_flight\20200906_bot_corn_comb_cameras.txt",
            'dem_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_20cm\All_flight\20200906_bot_corn_comb_20cm_dem.tif", 
            'ori': [r"D:\on_uav_data\raw\20200906_Bot\multi\fl2_nad",
                   r"D:\on_uav_data\raw\20200906_Bot\multi\fl3_oblique",
                   r"D:\on_uav_data\raw\20200906_Bot\multi\fl4_nad"], 
            'name': '20200906_bot_corn_comb_20cm',
            'path_list_tag': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_20cm\All_flight\20200906_bot_corn_comb_20cm_orthophotos\*.tif"},
           
          {'out': r"D:\on_uav_data\proc\extract\20200906_Bot\20200906_bot_corn_comb_20cm\Nadir_flight",
            'cam_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_20cm\Nadir_flight\20200906_bot_corn_comb_cameras.txt",
            'dem_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_20cm\Nadir_flight\20200906_bot_corn_comb_20cm_dem.tif", 
            'ori': [r"D:\on_uav_data\raw\20200906_Bot\multi\fl2_nad",
                   r"D:\on_uav_data\raw\20200906_Bot\multi\fl3_oblique",
                   r"D:\on_uav_data\raw\20200906_Bot\multi\fl4_nad"], 
            'name': '20200906_bot_corn_comb_20cm',
            'path_list_tag': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_20cm\Nadir_flight\20200906_bot_corn_comb_20cm_orthophotos\*.tif"}, 
          
          
          {'out': r"D:\on_uav_data\proc\extract\20200906_Bot\20200906_bot_corn_comb_50cm\All_flight",
            'cam_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_50cm\All_flight\20200906_bot_corn_comb_cameras.txt",
            'dem_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_50cm\All_flight\20200906_bot_corn_comb_50cm_dem.tif", 
            'ori': [r"D:\on_uav_data\raw\20200906_Bot\multi\fl2_nad",
                   r"D:\on_uav_data\raw\20200906_Bot\multi\fl3_oblique",
                   r"D:\on_uav_data\raw\20200906_Bot\multi\fl4_nad"], 
            'name': '20200906_bot_corn_comb_50cm',
            'path_list_tag': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_50cm\All_flight\20200906_bot_corn_comb_50cm_orthophotos\*.tif"},
           
          {'out': r"D:\on_uav_data\proc\extract\20200906_Bot\20200906_bot_corn_comb_50cm\Nadir_flight",
            'cam_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_50cm\Nadir_flight\20200906_bot_corn_comb_cameras.txt",
            'dem_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_50cm\Nadir_flight\20200906_bot_corn_comb_50cm_dem.tif", 
            'ori': [r"D:\on_uav_data\raw\20200906_Bot\multi\fl2_nad",
                   r"D:\on_uav_data\raw\20200906_Bot\multi\fl3_oblique",
                   r"D:\on_uav_data\raw\20200906_Bot\multi\fl4_nad"], 
            'name': '20200906_bot_corn_comb_50cm',
            'path_list_tag': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_50cm\Nadir_flight\20200906_bot_corn_comb_50cm_orthophotos\*.tif"}, 
          
          
          {'out': r"D:\on_uav_data\proc\extract\20200906_Bot\20200906_bot_corn_comb_1m\All_flight",
            'cam_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\All_flight\20200906_bot_corn_comb_cameras.txt",
            'dem_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\All_flight\20200906_bot_corn_comb_1m_dem.tif", 
            'ori': [r"D:\on_uav_data\raw\20200906_Bot\multi\fl2_nad",
                   r"D:\on_uav_data\raw\20200906_Bot\multi\fl3_oblique",
                   r"D:\on_uav_data\raw\20200906_Bot\multi\fl4_nad"], 
            'name': '20200906_bot_corn_comb_1m',
            'path_list_tag': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\All_flight\20200906_bot_corn_comb_1m_orthophotos\*.tif"},
           
          {'out': r"D:\on_uav_data\proc\extract\20200906_Bot\20200906_bot_corn_comb_1m\Nadir_flight",
            'cam_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\Nadir_flight\20200906_bot_corn_comb_cameras.txt",
            'dem_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\Nadir_flight\20200906_bot_corn_comb_1m_dem.tif", 
            'ori': [r"D:\on_uav_data\raw\20200906_Bot\multi\fl2_nad",
                   r"D:\on_uav_data\raw\20200906_Bot\multi\fl3_oblique",
                   r"D:\on_uav_data\raw\20200906_Bot\multi\fl4_nad"], 
            'name': '20200906_bot_corn_comb_1m',
            'path_list_tag': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\Nadir_flight\20200906_bot_corn_comb_1m_orthophotos\*.tif"}, 
          
          
          
          {'out': r"D:\on_uav_data\proc\extract\20200907_Bot\20200907_bot_corn_comb_20cm\All_flight",
            'cam_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_20cm\All_flight\20200907_bot_corn_comb_cameras.txt",
            'dem_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_20cm\All_flight\20200907_bot_corn_comb_20cm_dem.tif", 
            'ori': [r"D:\on_uav_data\raw\20200907_Bot\multi\fl3",
                   r"D:\on_uav_data\raw\20200907_Bot\multi\fl4",
                   r"D:\on_uav_data\raw\20200907_Bot\multi\fl5"],
            'name': '20200907_bot_corn_comb_20cm',
            'path_list_tag': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_20cm\All_flight\20200907_bot_corn_comb_20cm_orthophotos\*.tif"},
           
          {'out': r"D:\on_uav_data\proc\extract\20200907_Bot\20200907_bot_corn_comb_20cm\Nadir_flight",
            'cam_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_20cm\Nadir_flight\20200907_bot_corn_comb_cameras.txt",
            'dem_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_20cm\Nadir_flight\20200907_bot_corn_comb_20cm_dem.tif", 
            'ori': [r"D:\on_uav_data\raw\20200907_Bot\multi\fl3",
                   r"D:\on_uav_data\raw\20200907_Bot\multi\fl4",
                   r"D:\on_uav_data\raw\20200907_Bot\multi\fl5"],
            'name': '20200907_bot_corn_comb_20cm',
            'path_list_tag': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_20cm\Nadir_flight\20200907_bot_corn_comb_20cm_orthophotos\*.tif"},
          
          
          {'out': r"D:\on_uav_data\proc\extract\20200907_Bot\20200907_bot_corn_comb_50cm\All_flight",
            'cam_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_50cm\All_flight\20200907_bot_corn_comb_cameras.txt",
            'dem_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_50cm\All_flight\20200907_bot_corn_comb_50cm_dem.tif", 
            'ori': [r"D:\on_uav_data\raw\20200907_Bot\multi\fl3",
                   r"D:\on_uav_data\raw\20200907_Bot\multi\fl4",
                   r"D:\on_uav_data\raw\20200907_Bot\multi\fl5"],
            'name': '20200907_bot_corn_comb_50cm',
            'path_list_tag': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_50cm\All_flight\20200907_bot_corn_comb_50cm_orthophotos\*.tif"},
           
          {'out': r"D:\on_uav_data\proc\extract\20200907_Bot\20200907_bot_corn_comb_50cm\Nadir_flight",
            'cam_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_50cm\Nadir_flight\20200907_bot_corn_comb_cameras.txt",
            'dem_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_50cm\Nadir_flight\20200907_bot_corn_comb_50cm_dem.tif", 
            'ori': [r"D:\on_uav_data\raw\20200907_Bot\multi\fl3",
                   r"D:\on_uav_data\raw\20200907_Bot\multi\fl4",
                   r"D:\on_uav_data\raw\20200907_Bot\multi\fl5"],
            'name': '20200907_bot_corn_comb_50cm',
            'path_list_tag': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_50cm\Nadir_flight\20200907_bot_corn_comb_50cm_orthophotos\*.tif"},
          
          
          {'out': r"D:\on_uav_data\proc\extract\20200907_Bot\20200907_bot_corn_comb_1m\All_flight",
            'cam_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_1m\All_flight\20200907_bot_corn_comb_cameras.txt",
            'dem_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_1m\All_flight\20200907_bot_corn_comb_1m_dem.tif", 
            'ori': [r"D:\on_uav_data\raw\20200907_Bot\multi\fl3",
                   r"D:\on_uav_data\raw\20200907_Bot\multi\fl4",
                   r"D:\on_uav_data\raw\20200907_Bot\multi\fl5"],
            'name': '20200907_bot_corn_comb_1m',
            'path_list_tag': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_1m\All_flight\20200907_bot_corn_comb_1m_orthophotos\*.tif"},
           
          {'out': r"D:\on_uav_data\proc\extract\20200907_Bot\20200907_bot_corn_comb_1m\Nadir_flight",
            'cam_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_1m\Nadir_flight\20200907_bot_corn_comb_cameras.txt",
            'dem_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_1m\Nadir_flight\20200907_bot_corn_comb_1m_dem.tif", 
            'ori': [r"D:\on_uav_data\raw\20200907_Bot\multi\fl3",
                   r"D:\on_uav_data\raw\20200907_Bot\multi\fl4",
                   r"D:\on_uav_data\raw\20200907_Bot\multi\fl5"],
            'name': '20200907_bot_corn_comb_1m',
            'path_list_tag': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_1m\Nadir_flight\20200907_bot_corn_comb_1m_orthophotos\*.tif"}
    ]

# Loop trough the list of input elements
for source in sources:  
    
    # Define the function that will be used as input for the multiprocessing
    # This function takes as input a tuple of 2 elements : an index, and a chunk of images  (1/15th of all orthophotos)
    def build_database(tuple_chunk):  
        iteration=tuple_chunk[0]
        chunk=tuple_chunk[1]
        df_list = []
          
        # import modules need for the function
        from tqdm import tqdm
        import pandas as pd
        import glob
        import os
        import rasterio as rio
        from functools import reduce, partial
        import math
        import numpy as np
        import exiftool as exif
        from tqdm import tqdm
        from pathlib import Path, PureWindowsPath
        from timeit import default_timer as timer
        import pyarrow
        
        # Assign values in the dictionnary to variables
        out = source['out']
        cam_path = source['cam_path']
        dem_path = source['dem_path']
        ori = source['ori']
        
        # Turn DEM into table with pixels in rows and coordinates in columns
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
        
    
        # List all path of images containing exif data from different flights and put them in one linge list
        ori_list = []
        for item in ori:
            ori_list.append(glob.glob(item + "\\*.tif"))  # paths from all non-calibrated images as these have relevant EXIF
        def flatten(t):
            return [item for sublist in t for item in sublist]
        path_flat = flatten(ori_list)
        path_norm = []
        for i in path_flat:
            path_norm.append(str(PureWindowsPath(i)))
        
        # Loop trhough images in each chunk
        for each_ortho in tqdm(chunk):  # select just a few in chunk to test loop
    
            # Import camera positions and the search the camera coords for the associated orthophoto
            start2 = timer()
            campos = pd.read_csv(cam_path, sep='\t', skiprows=2, header=None)
            campos.columns = ['PhotoID', 'X', 'Y', 'Z', 'Omega', 'Phi', 'Kappa', 'r11', 'r12', 'r13', 'r21',
                              'r22', 'r23', 'r31', 'r32', 'r33']   
            path, file = os.path.split(each_ortho)
            name, ext = os.path.splitext(file)    
            reduced_string = file[5:]
            reduced_name = name[5:]   
            campos1 = campos[campos['PhotoID'].str.match(name)] ### was producing error with reduced_name
            xcam = campos1['X'].values  # easting
            ycam = campos1['Y'].values  # northing
            zcam = campos1['Z'].values    
            del campos, campos1  # trying to reduce memory load    
            end2 = timer()
            print('Finding camera position for first ortho in first chunk: ', end2 - start2, 'seconds')
    
            # Get sun elevation and sun azimuth angle
            start3 = timer()
            filter_object = filter(lambda a: reduced_string in a, path_norm)
            # Convert the filter object to list
            exifobj = list(filter_object)   
            with exif.ExifToolHelper() as et:
                sunelev = float(et.get_tags(exifobj[0], 'XMP:SolarElevation')[0]['XMP:SolarElevation'])* (180/math.pi)
                saa = float(et.get_tags(exifobj[0], 'XMP:SolarAzimuth')[0]['XMP:SolarAzimuth'])* (180/math.pi)
            end3 = timer()
            print('Getting SAA and Sun Elevation from ortho EXIF data: ', end3 - start3, 'seconds')
            start4 = timer()
            
            # Get XY + value for each ortho photo
            bands = {}
            with rio.open(each_ortho) as rst:
                # Loop throug bands
                for counter in range(1, 11, 1):
                    res = []
                    b1 = rst.read(counter)
                    arr = np.array(b1)
                    Xp, Yp, val = xyval(arr) # this function is loaded from smac_functions.py
                    # Loop trhough pixels
                    for i, j in zip(Xp, Yp):
                        # coords2pixels = map_layer.index(235059.32,810006.31)  # input lon,lat
                        res.append(rst.xy(i, j))  # input px, py -> see rasterio documentation
                    df = pd.DataFrame(res)
                    df_ortho = pd.concat([pd.Series(val), df], axis=1)
                    df_ortho.columns = ['band'+str(counter), 'Xw', 'Yw']
                    df_ortho = df_ortho.round({'Xw': 1, 'Yw': 1})  # rounding coords for matches; careful when aligning raster
                    # outputs and original raster files.
                    bands[f"band{counter}"] = df_ortho
            my_reduce = partial(pd.merge, on=["Xw", "Yw"], how='outer') # define a function to merge bands together
            df_allbands = reduce(my_reduce, bands.values()) # iterates functions for all 11 bands
            end4 = timer()
            print('Break all ortho bands into pixel: ', end4 - start4, 'seconds')
    
            # Combine ortho values and DEM values, Compute vza and vaa and insert some imoprtant columns
            start5 = timer()
            dfs = [df_dem, df_allbands]  #, df_slp, df_asp] TODO: slp and asp not needed without correcting vza
            df_merged = reduce(lambda left, right: pd.merge(left, right, on=["Xw", "Yw"]), dfs)  # combine all dfs above
            
            df_merged['vza'] = df_merged.apply(lambda x: np.arctan((zcam[0] - x.elev)/math.sqrt(math.pow(xcam[0] - x.Xw, 2)+math.pow(ycam[0] - x.Yw, 2))), axis=1)
            df_merged['vza'] = round(90 - (df_merged['vza'] * (180/math.pi)), 2)  # substr 90° to get correct angle
            df_merged['vza'] = np.where((df_merged['band1'] == 65535), np.nan, df_merged['vza'])  # set nan when band = nan
            
            # Calculate VAA acoording to Roosjen (2018)
            df_merged['vaa_rad'] = df_merged.apply(lambda x: math.acos((ycam[0] - x.Yw)/(math.sqrt((x.Xw - xcam[0])**2 + (x.Yw - ycam[0])**2))) if x.Xw - xcam[0] < 0 else - math.acos((ycam[0] - x.Yw)/(math.sqrt((x.Xw - xcam[0])**2 + (x.Yw - ycam[0])**2))), axis=1)
            df_merged['vaa'] = np.where (round((df_merged['vaa_rad'] * (180/math.pi)), 2) - saa > -180, round((df_merged['vaa_rad'] * (180/math.pi)), 2) - saa, round((df_merged['vaa_rad'] * (180/math.pi)), 2) - saa + 360)
            df_merged['vaa'] =np.where((df_merged['band1'] == 65535), np.nan, df_merged['vaa'])  # set nan when band = nan
            df_merged['vaa_rad'] = np.where((df_merged['band1'] == 65535), np.nan, df_merged['vaa_rad'])
            
            df_merged.insert(0, 'path', file)
            df_merged.insert(1,'xcam', xcam[0])
            df_merged.insert(2,'ycam', ycam[0])
            df_merged.insert(3, 'sunelev', round(sunelev, 2))
            df_merged.insert(4, 'saa', round(saa, 2))
    
            df_list.append(df_merged)
            end5 = timer()
            print('Combine DEM and ortho values + calculate VZA and VAA', end5 - start5, 'seconds')
    
        # Save table as a feather file and free memory
        result = pd.concat(df_list)
        result=result.reset_index().drop('index', axis=1)
        if not os.path.isdir(source['out']):
            os.makedirs(source['out'])
        result.to_feather(f"{source['out']}\\{source['name']}_{iteration}.feather")
        del result, df_merged, df_allbands, df_list
    
    
    # Split loaded data into chunks">
    path_list = []  # listing all orthophoto paths
    path_list = glob.glob(source['path_list_tag'])
    chunks = np.array_split(path_list, 15)
    
    # Parallelize using the defined function and chunks as inputs
    
    Parallel(n_jobs=8)(delayed(build_database)(i) for i in list(enumerate(chunks)))
    
