#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/06/22
# Version      : 1.0
# ---------------------------------------------------------------------------
"""
This script retrieves pixel-wise reflectance values from Metashape orthophotos. For each pixel, the
observation/illumination angles are calculated based on the methodology outlined in Roosjen (2017).
Orthophotos are used instead of ortho mosaics as the camera positions are available for each orthophoto
but not for the mosaic. The pixel resolution is determined by the orthophotos and the digital elevation model (DEM).

Input requirements:
    - Camera positions in omega, phi, kappa text file format (Agisoft Metashape)
    - Digital Elevation Model (DEM)
    - List of orthophotos (in the same CRS as the DEM)

This script utilizes parallelization to enhance performance by processing images in parallel.
"""

# Import necessary libraries
import pandas as pd
import glob
import os
from smac_functions import *
from config_object import config
from functools import reduce, partial
import numpy as np
from joblib import Parallel, delayed

# Define a dictionary for each input source containing paths and settings
sources = [
    {
        'out': config.main_extract_out,               # Output directory for results
        'cam_path': config.main_extract_cam_path,     # Path to camera position file
        'dem_path': config.main_extract_dem_path,     # Path to DEM file
        'ori': config.main_extract_ori,               # Path to directory with original images (containing EXIF data)
        'name': config.main_extract_name,             # Output file name prefix
        'path_list_tag': config.main_extract_path_list_tag  # Path list tag for orthophotos
    }
]

# Loop through each source dictionary to process all provided data
for source in sources:

    # Define function to process each chunk of images in parallel
    # This function takes as input a tuple: an index, and a list of images (1/15th of all orthophotos)
    def build_database(tuple_chunk):
        iteration = tuple_chunk[0]
        chunk = tuple_chunk[1]
        df_list = []  # Initialize list to store dataframes for each image in the chunk

        # Import required libraries within function scope
        from tqdm import tqdm
        import pandas as pd
        import rasterio as rio
        import math
        import numpy as np
        import exiftool as exif
        from pathlib import Path, PureWindowsPath
        from timeit import default_timer as timer

        # Assign source variables to local variables for readability
        out = source['out']
        cam_path = source['cam_path']
        dem_path = source['dem_path']
        ori = source['ori']

        # Step 1: Process DEM to extract pixel-based elevation and coordinates
        start = timer()
        with rio.open(dem_path) as dem:
            arr_dem = np.array(dem.read(1))  # Read DEM as a numpy array
        Xp_dem, Yp_dem, val_dem = xyval(arr_dem)  # Extract pixel coordinates and values

        # Create DataFrame for DEM data with coordinates and elevations for each pixel
        res_dem = []
        with rio.open(dem_path) as dem_layer:
            for i, j in zip(Xp_dem, Yp_dem):
                res_dem.append(dem_layer.xy(i, j))  # Convert pixel indices to geographic coordinates
        df_dem = pd.DataFrame(res_dem, columns=['Xw', 'Yw'])
        df_dem['elev'] = val_dem
        df_dem = df_dem.round({'elev': 2, 'Xw': 1, 'Yw': 1})  # Round coordinates and elevation for consistent merging
        end = timer()
        print('DEM processing time: ', end - start, 'seconds')

        # Step 2: Generate a list of all paths to original images with EXIF data
        ori_list = [glob.glob(item + "\\*.tif") for item in ori]
        path_flat = [str(PureWindowsPath(path)) for sublist in ori_list for path in sublist]

        # Step 3: Process each image in the current chunk
        for each_ortho in tqdm(chunk):

            # Step 3a: Retrieve camera position for the current orthophoto
            start2 = timer()
            campos = pd.read_csv(cam_path, sep='\t', skiprows=2, header=None)
            campos.columns = ['PhotoID', 'X', 'Y', 'Z', 'Omega', 'Phi', 'Kappa', 'r11', 'r12', 'r13',
                              'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
            path, file = os.path.split(each_ortho)
            name, _ = os.path.splitext(file)
            campos1 = campos[campos['PhotoID'].str.match(name)]
            xcam, ycam, zcam = campos1['X'].values[0], campos1['Y'].values[0], campos1['Z'].values[0]
            end2 = timer()
            print('Camera position retrieval time: ', end2 - start2, 'seconds')

            # Step 3b: Retrieve solar elevation and azimuth angles from EXIF data (Roosjen 2017)
            start3 = timer()
            exifobj = [path for path in path_flat if name in path]
            with exif.ExifToolHelper() as et:
                sunelev = float(et.get_tags(exifobj[0], 'XMP:SolarElevation')[0]['XMP:SolarElevation']) * (180 / math.pi)
                saa = float(et.get_tags(exifobj[0], 'XMP:SolarAzimuth')[0]['XMP:SolarAzimuth']) * (180 / math.pi)
            end3 = timer()
            print('Solar angle retrieval time: ', end3 - start3, 'seconds')

            # Step 3c: Extract reflectance data for each pixel in orthophoto bands
            start4 = timer()
            bands = {}
            with rio.open(each_ortho) as rst:
                for counter in range(1, 11):
                    b1 = rst.read(counter)
                    Xp, Yp, val = xyval(b1)  # Get pixel values and coordinates
                    res = [rst.xy(i, j) for i, j in zip(Xp, Yp)]
                    df_ortho = pd.DataFrame(res, columns=['Xw', 'Yw'])
                    df_ortho['band' + str(counter)] = val
                    bands[f"band{counter}"] = df_ortho
            df_allbands = reduce(partial(pd.merge, on=["Xw", "Yw"], how='outer'), bands.values())
            end4 = timer()
            print('Orthophoto band processing time: ', end4 - start4, 'seconds')

            # Step 3d: Combine DEM and orthophoto data, then calculate VZA and VAA based on Roosjen (2017)
            start5 = timer()
            dfs = [df_dem, df_allbands]
            df_merged = reduce(lambda left, right: pd.merge(left, right, on=["Xw", "Yw"]), dfs)

            # Calculate view zenith angle (VZA) for each pixel
            df_merged['vza'] = df_merged.apply(lambda x: np.arctan((zcam - x.elev) / math.sqrt((xcam - x.Xw)**2 + (ycam - x.Yw)**2)), axis=1)
            df_merged['vza'] = round(90 - (df_merged['vza'] * (180 / math.pi)), 2)
            df_merged['vza'] = np.where(df_merged['band1'] == 65535, np.nan, df_merged['vza'])  # Set NaN for missing data

            # Calculate view azimuth angle (VAA) based on Roosjen (2018) and subtract solar azimuth angle (SAA)
            df_merged['vaa_rad'] = df_merged.apply(lambda x: math.acos((ycam - x.Yw) / (math.sqrt((x.Xw - xcam)**2 + (x.Yw - ycam)**2))) if x.Xw - xcam < 0 else -math.acos((ycam - x.Yw) / (math.sqrt((x.Xw - xcam)**2 + (x.Yw - ycam)**2))), axis=1)
            df_merged['vaa'] = round((df_merged['vaa_rad'] * (180 / math.pi)) - saa, 2)
            df_merged['vaa'] = np.where(df_merged['band1'] == 65535, np.nan, df_merged['vaa'])

            # Insert metadata columns
            df_merged.insert(0, 'path', file)
            df_merged.insert(1, 'xcam', xcam)
            df_merged.insert(2, 'ycam', ycam)
            df_merged.insert(3, 'sunelev', round(sunelev, 2))
            df_merged.insert(4, 'saa', round(saa, 2))
            df_list.append(df_merged)
            end5 = timer()
            print('Data merging and angle calculations time: ', end5 - start5, 'seconds')

        # Save results as a feather file and free memory
        result = pd.concat(df_list).reset_index(drop=True)
        if not os.path.isdir(out):
            os.makedirs(out)
        result.to_feather(f"{out}\\{source['name']}_{iteration}.feather")
        del result, df_merged, df_allbands, df_list

    # Split orthophoto paths into chunks for parallel processing
    path_list = glob.glob(source['path_list_tag'])
    chunks = np.array_split(path_list, 15)

    # Process each chunk in parallel
    Parallel(n_jobs=8)(delayed(build_database)(i) for i in list(enumerate(chunks)))
