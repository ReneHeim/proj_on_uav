#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/06/22
# Version      : 1.0
# ---------------------------------------------------------------------------

import pandas as pd
import glob
import os
import logging
from smac_functions import *
from config_object import config
from functools import reduce, partial
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import rasterio as rio
import math
import numpy as np
import exiftool
from pathlib import Path, PureWindowsPath
from timeit import default_timer as timer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("process.log"),
        logging.StreamHandler()
    ]
)

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
    exiftool_path = r"C:\Program Files\ExifTool\exiftool.exe"
    # Define function to process each chunk of images in parallel
    def build_database(tuple_chunk):

        iteration = tuple_chunk[0]
        chunk = tuple_chunk[1]
        df_list = []

        # Assign source variables to local variables
        out = source['out']

        cam_path = source['cam_path']

        dem_path = source['dem_path']
        ori = source['ori']

        # Log start of DEM processing
        logging.info(f"Processing DEM for iteration {iteration}")
        start = timer()
        print("here")

        # Log start of DEM processing
        logging.info(f"Processing DEM for iteration {iteration}")
        start = timer()

        try:
            with rio.open(dem_path) as dem:
                arr_dem = dem.read(1)  # Read DEM as a numpy array
                transform = dem.transform  # Affine transformation for pixel-to-world coordinates

            # Create a meshgrid for all indices
            rows, cols = np.indices(arr_dem.shape)

            # Use Rasterio's vectorized function to get all coordinates at once
            x_coords, y_coords = rio.transform.xy(transform, rows, cols, offset='center')
            end = timer()

            logging.info(f"DEM processing completed for iteration {iteration} in {end - start:.2f} seconds")

        # Flatten the arrays and create a DataFrame
            df_dem = pd.DataFrame({
                "Xw": np.array(x_coords).ravel(),
                "Yw": np.array(y_coords).ravel(),
                "elev": arr_dem.ravel()
            }).round({"elev": 2, "Xw": 1, "Yw": 1})

            end = timer()
        except Exception as e:
            logging.error(f"Error processing DEM: {e}")
            return

        try:
            ori_list = [glob.glob(item + "\\*.tif") for item in ori]
            path_flat = [str(PureWindowsPath(path)) for sublist in ori_list for path in sublist]
        except Exception as e:
            logging.error(f"Error retrieving original images: {e}")
            return

        # Process each image in the current chunk
        for each_ortho in tqdm(chunk):
            try:
                path, file = os.path.split(each_ortho)
                name, _ = os.path.splitext(file)
                logging.info(f"Processing orthophoto {file} for iteration {iteration}")

                # Step 1: Retrieve camera position
                start2 = timer()
                campos = pd.read_csv(cam_path, sep='\t', skiprows=2, header=None)
                campos.columns = ['PhotoID', 'X', 'Y', 'Z', 'Omega', 'Phi', 'Kappa', 'r11', 'r12', 'r13',
                                  'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
                campos1 = campos[campos['PhotoID'].str.match(name)]
                xcam, ycam, zcam = campos1['X'].values[0], campos1['Y'].values[0], campos1['Z'].values[0]
                end2 = timer()
                logging.info(f"Camera position retrieved for {file} in {end2 - start2:.2f} seconds")

                # Step 2: Retrieve solar angles from EXIF
                start3 = timer()
                exifobj = [path for path in path_flat if name in path]
                try:
                    with exiftool.ExifToolHelper(executable=exiftool_path) as et:
                        metadata = et.get_metadata(exifobj[0])
                        metadata =metadata[0]  # For debugging purposes
                        sunelev = float(metadata.get('XMP:SolarElevation', 0)) * (180 / math.pi)
                        saa = float(metadata.get('XMP:SolarAzimuth', 0)) * (180 / math.pi)
                    end3 = timer()
                    print('Getting SAA and Sun Elevation from ortho EXIF data: ', end3 - start3, 'seconds')
                except Exception as e:
                    logging.error(f"Error processing orthophoto {file}: {e}")

                end3 = timer()
                logging.info(f"Solar angles retrieved for {file} in {end3 - start3:.2f} seconds")

                start4 = timer()
                with rio.open(each_ortho) as rst:
                    nodata = rst.nodata  # Get the nodata value
                    num_bands = rst.count  # Total number of bands
                    height = rst.height
                    width = rst.width

                    # Read all bands into a 3D numpy array of shape (bands, rows, cols)
                    # If memory is a concern, you can process bands individually
                    b_all = rst.read()  # shape: (num_bands, height, width)

                    # Create a valid data mask where data is not nodata
                    if nodata is not None:
                        valid_mask = np.all(b_all != nodata, axis=0)
                    else:
                        # If nodata is not defined, assume all data is valid
                        valid_mask = np.all(b_all != 0, axis=0)  # Adjust if 0 is a valid data value

                    # Get the indices of valid data points
                    rows, cols = np.where(valid_mask)

                    # Get the world coordinates for these indices (vectorized)
                    Xw, Yw = rio.transform.xy(rst.transform, rows, cols)

                    # Extract band values at valid indices
                    # Shape of band_values: (num_valid_pixels, num_bands)
                    band_values = b_all[:, rows, cols].T

                    # Prepare data for DataFrame
                    data = {
                        'Xw': Xw,
                        'Yw': Yw,
                    }
                    for idx in range(num_bands):
                        data[f'band{idx + 1}'] = band_values[:, idx]

                    # Create a single DataFrame with all bands
                    df_allbands = pd.DataFrame(data)

                end4 = timer()
                logging.info(f"Orthophoto bands processed for {file} in {end4 - start4:.2f} seconds")

                # Step 4: Merge DEM and orthophoto data
                start5 = timer()
                dfs = [df_dem, df_allbands]
                df_merged = reduce(lambda left, right: pd.merge(left, right, on=["Xw", "Yw"]), dfs)

                # Calculate angles
                df_merged['vza'] = df_merged.apply(
                    lambda x: np.arctan((zcam - x.elev) / math.sqrt((xcam - x.Xw)**2 + (ycam - x.Yw)**2)), axis=1)
                df_merged['vza'] = round(90 - (df_merged['vza'] * (180 / math.pi)), 2)
                df_merged['vza'] = np.where(df_merged['band1'] == 65535, np.nan, df_merged['vza'])

                df_merged['vaa_rad'] = df_merged.apply(
                    lambda x: math.acos((ycam - x.Yw) / (math.sqrt((x.Xw - xcam)**2 + (x.Yw - ycam)**2))) if x.Xw - xcam < 0 else -math.acos((ycam - x.Yw) / (math.sqrt((x.Xw - xcam)**2 + (x.Yw - ycam)**2))), axis=1)
                df_merged['vaa'] = round((df_merged['vaa_rad'] * (180 / math.pi)) - saa, 2)
                df_merged['vaa'] = np.where(df_merged['band1'] == 65535, np.nan, df_merged['vaa'])

                df_merged.insert(0, 'path', file)
                df_merged.insert(1, 'xcam', xcam)
                df_merged.insert(2, 'ycam', ycam)
                df_merged.insert(3, 'sunelev', round(sunelev, 2))
                df_merged.insert(4, 'saa', round(saa, 2))
                df_list.append(df_merged)
                end5 = timer()
                logging.info(f"Data merging and angle calculations completed for {file} in {end5 - start5:.2f} seconds")

            except Exception as e:
                logging.error(f"Error processing orthophoto {file}: {e}")

        # Save results
        try:
            result = pd.concat(df_list).reset_index(drop=True)
            if not os.path.isdir(out):
                os.makedirs(out)
            result.to_feather(f"{out}\\{source['name']}_{iteration}.feather")
            logging.info(f"Results saved for iteration {iteration}")
        except Exception as e:
            logging.error(f"Error saving results for iteration {iteration}: {e}")

    # Split orthophoto paths into chunks for parallel processing
    path_list = glob.glob(source['path_list_tag'])
    chunks = np.array_split(path_list, 150)
    logging.info(f"Starting parallel processing with {len(chunks)} chunks")

    # Process each chunk in parallel
    Parallel(n_jobs=1)(delayed(build_database)(i) for i in list(enumerate(chunks)))

