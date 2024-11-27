#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/06/22
# Version     : 1.5
# ----------------------------------------------------------------------------

import glob
import os
import logging
import math
import numpy as np
from functools import reduce, partial
from joblib import Parallel, delayed
from pathlib import Path, PureWindowsPath
from timeit import default_timer as timer
from tqdm import tqdm
import rasterio as rio
import exiftool
import polars as pl  # Using Polars instead of Pandas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("process.log"),
        logging.StreamHandler()
    ]
)

# Assuming smac_functions and config_object are custom modules
from smac_functions import *  # Replace with specific imports as needed
from config_object import config


def process_dem(dem_path):
    """
    Reads the DEM file and returns a Polars DataFrame with coordinates and elevation.

    Args:
        dem_path (str): Path to the DEM file.

    Returns:
        pl.DataFrame: DataFrame containing 'Xw', 'Yw', and 'elev' columns.
    """
    logging.info("Processing DEM")
    start = timer()

    try:
        with rio.open(dem_path) as dem:
            arr_dem = dem.read(1)  # Read DEM as a NumPy array
            transform = dem.transform  # Affine transformation for pixel-to-world coordinates

        # Create a meshgrid for all indices
        rows, cols = np.indices(arr_dem.shape)

        # Use Rasterio's vectorized function to get all coordinates at once
        x_coords, y_coords = rio.transform.xy(transform, rows, cols, offset='center')

        # Flatten the arrays and create a Polars DataFrame
        df_dem = pl.DataFrame({
            "Xw": np.array(x_coords).ravel(),
            "Yw": np.array(y_coords).ravel(),
            "elev": arr_dem.ravel()
        }).with_columns([
            pl.col("elev").round(2),
            pl.col("Xw").round(1),
            pl.col("Yw").round(1)
        ])

        end = timer()
        logging.info(f"DEM processing completed in {end - start:.2f} seconds")
        return df_dem

    except Exception as e:
        logging.error(f"Error processing DEM: {e}")
        return None


def load_camera_positions(cam_path):
    """
    Loads camera positions from a file into a Polars DataFrame.

    Args:
        cam_path (str): Path to the camera positions file.

    Returns:
        pl.DataFrame: DataFrame containing camera positions.
    """
    try:
        campos = pl.read_csv(
            cam_path,
            separator='\t',
            skip_rows=2,
            has_header=False
        )
        campos.columns = ['PhotoID', 'X', 'Y', 'Z', 'Omega', 'Phi', 'Kappa',
                          'r11', 'r12', 'r13', 'r21', 'r22', 'r23',
                          'r31', 'r32', 'r33']
        return campos
    except Exception as e:
        logging.error(f"Error loading camera positions: {e}")
        return None


def get_solar_angles(exiftool_path, exifobj_path):
    """
    Retrieves solar elevation and azimuth angles from EXIF data.

    Args:
        exiftool_path (str): Path to the ExifTool executable.
        exifobj_path (str): Path to the image file to extract EXIF data from.

    Returns:
        tuple: sunelev (float), saa (float)
    """
    try:
        with exiftool.ExifToolHelper(executable=exiftool_path) as et:
            metadata_list = et.get_metadata(exifobj_path)
            if metadata_list:
                metadata = metadata_list[0]
                sunelev = float(metadata.get('XMP:SolarElevation', 0))
                saa = float(metadata.get('XMP:SolarAzimuth', 0))
                return sunelev, saa
            else:
                logging.error(f"No metadata found for {exifobj_path}")
                return None, None
    except Exception as e:
        logging.error(f"Error retrieving EXIF data from {exifobj_path}: {e}")
        return None, None


def process_orthophoto_bands(each_ortho):
    """
    Processes the bands of an orthophoto and returns a Polars DataFrame.

    Args:
        each_ortho (str): Path to the orthophoto file.

    Returns:
        pl.DataFrame: DataFrame containing 'Xw', 'Yw', and band values.
    """
    start = timer()
    try:
        with rio.open(each_ortho) as rst:
            nodata = rst.nodata  # Get the nodata value
            num_bands = rst.count  # Total number of bands

            # Read all bands into a 3D NumPy array of shape (bands, rows, cols)
            b_all = rst.read()  # shape: (num_bands, height, width)

            # Create a valid data mask where data is not nodata
            if nodata is not None:
                valid_mask = np.all(b_all != nodata, axis=0)
            else:
                # If nodata is not defined, assume all data is valid
                valid_mask = np.all(b_all != 0, axis=0)  # Adjust if 0 is a valid data value

            # Get the indices of valid data points
            rows, cols = np.where(valid_mask)

            if len(rows) == 0:
                logging.warning(f"No valid data found in {each_ortho}")
                return None

            # Get the world coordinates for these indices (vectorized)
            Xw, Yw = rio.transform.xy(rst.transform, rows, cols)

            # Extract band values at valid indices
            # Shape of band_values: (num_valid_pixels, num_bands)
            band_values = b_all[:, rows, cols].T

            # Prepare data for Polars DataFrame
            data = {
                'Xw': Xw,
                'Yw': Yw,
            }
            for idx in range(num_bands):
                data[f'band{idx + 1}'] = band_values[:, idx]

            # Create a Polars DataFrame with all bands
            df_allbands = pl.DataFrame(data)

        end = timer()
        logging.info(f"Orthophoto bands processed for {each_ortho} in {end - start:.2f} seconds")
        return df_allbands

    except Exception as e:
        logging.error(f"Error processing orthophoto bands for {each_ortho}: {e}")
        return None


def merge_data_and_calculate_angles(df_dem, df_allbands, xcam, ycam, zcam, sunelev, saa, file):
    """
    Merges DEM and orthophoto data, calculates angles, and returns the merged DataFrame.

    Args:
        df_dem (pl.DataFrame): DataFrame containing DEM data.
        df_allbands (pl.DataFrame): DataFrame containing orthophoto band data.
        xcam (float): Camera X coordinate.
        ycam (float): Camera Y coordinate.
        zcam (float): Camera Z coordinate.
        sunelev (float): Solar elevation angle.
        saa (float): Solar azimuth angle.
        file (str): Filename of the orthophoto.

    Returns:
        pl.DataFrame: Merged DataFrame with calculated angles.
    """
    try:
        # Merge df_dem and df_allbands on 'Xw' and 'Yw' columns
        df_merged = df_dem.join(df_allbands, on=["Xw", "Yw"], how="inner")

        # Calculate viewing zenith angle (vza)
        df_merged = df_merged.with_columns(
            (
                    90 - (
                    pl.arctangent(
                        (zcam - pl.col('elev')) /
                        pl.sqrt((xcam - pl.col('Xw'))**2 + (ycam - pl.col('Yw'))**2)
                    ) * (180 / math.pi)
            )
            ).round(2).alias('vza')
        )

        # Replace 'vza' with None where 'band1' equals 65535
        df_merged = df_merged.with_columns(
            pl.when(pl.col('band1') == 65535)
            .then(None)
            .otherwise(pl.col('vza'))
            .alias('vza')
        )

        # Calculate viewing azimuth angle (vaa)
        df_merged = df_merged.with_columns(
            pl.when(xcam - pl.col('Xw') < 0)
            .then(
                -pl.arccos(
                    (ycam - pl.col('Yw')) /
                    pl.sqrt((pl.col('Xw') - xcam)**2 + (pl.col('Yw') - ycam)**2)
                )
            )
            .otherwise(
                pl.arccos(
                    (ycam - pl.col('Yw')) /
                    pl.sqrt((pl.col('Xw') - xcam)**2 + (pl.col('Yw') - ycam)**2)
                )
            )
            .alias('vaa_rad')
        )

        # Convert 'vaa_rad' to degrees and adjust by 'saa'
        df_merged = df_merged.with_columns(
            ((pl.col('vaa_rad') * (180 / math.pi) - saa).round(2)).alias('vaa')
        )

        # Replace 'vaa' with None where 'band1' equals 65535
        df_merged = df_merged.with_columns(
            pl.when(pl.col('band1') == 65535)
            .then(None)
            .otherwise(pl.col('vaa'))
            .alias('vaa')
        )

        # Add additional columns: 'path', 'xcam', 'ycam', 'sunelev', 'saa'
        df_merged = df_merged.with_columns([
            pl.lit(file).alias('path'),
            pl.lit(xcam).alias('xcam'),
            pl.lit(ycam).alias('ycam'),
            pl.lit(round(sunelev, 2)).alias('sunelev'),
            pl.lit(round(saa, 2)).alias('saa')
        ])

        # Reorder columns to place the new columns at the beginning
        cols_order = ['path', 'xcam', 'ycam', 'sunelev', 'saa'] + [
            col for col in df_merged.columns if col not in ['path', 'xcam', 'ycam', 'sunelev', 'saa']
        ]
        df_merged = df_merged.select(cols_order)

        return df_merged

    except Exception as e:
        logging.error(f"Error merging data and calculating angles for {file}: {e}")
        return None


def build_database(tuple_chunk, source, exiftool_path):
    """
    Processes a chunk of orthophoto images.

    Args:
        tuple_chunk (tuple): Tuple containing iteration index and list of orthophoto paths.
        source (dict): Dictionary containing source configuration.
        exiftool_path (str): Path to the ExifTool executable.

    Returns:
        None
    """
    iteration = tuple_chunk[0]
    chunk = tuple_chunk[1]
    df_list = []

    # Assign source variables to local variables
    out = source['out']
    cam_path = source['cam_path']
    dem_path = source['dem_path']
    ori = source['ori']

    logging.info(f"Processing DEM for iteration {iteration}")
    df_dem = process_dem(dem_path)
    if df_dem is None:
        logging.error(f"Skipping iteration {iteration} due to DEM processing error.")
        return

    # Load camera positions once
    campos = load_camera_positions(cam_path)
    if campos is None:
        logging.error(f"Skipping iteration {iteration} due to camera positions loading error.")
        return

    # Get list of original images
    try:
        ori_list = [glob.glob(os.path.join(item, "*.tif")) for item in ori]
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
            campos1 = campos.filter(pl.col('PhotoID').str.contains(name))
            if len(campos1) == 0:
                logging.error(f"No camera position found for {name}")
                continue
            xcam = campos1['X'][0]
            ycam = campos1['Y'][0]
            zcam = campos1['Z'][0]
            end2 = timer()
            logging.info(f"Camera position retrieved for {file} in {end2 - start2:.2f} seconds")

            # Step 2: Retrieve solar angles from EXIF
            start3 = timer()
            exifobj = [p for p in path_flat if name in p]
            if not exifobj:
                logging.error(f"No original image found for {name}")
                continue
            sunelev, saa = get_solar_angles(exiftool_path, exifobj[0])
            if sunelev is None or saa is None:
                logging.error(f"Skipping {file} due to missing EXIF data")
                continue
            end3 = timer()
            logging.info(f"Solar angles retrieved for {file} in {end3 - start3:.2f} seconds")

            # Step 3: Process orthophoto bands
            df_allbands = process_orthophoto_bands(each_ortho)
            if df_allbands is None:
                logging.error(f"Skipping {file} due to error in processing bands")
                continue

            # Step 4: Merge data and calculate angles
            df_merged = merge_data_and_calculate_angles(
                df_dem, df_allbands, xcam, ycam, zcam, sunelev, saa, file
            )
            if df_merged is None:
                logging.error(f"Skipping {file} due to error in merging data and calculating angles")
                continue

            df_list.append(df_merged)

        except Exception as e:
            logging.error(f"Error processing orthophoto {file}: {e}")
            continue

    # Save results
    try:
        if df_list:
            result = pl.concat(df_list)
            if not os.path.isdir(out):
                os.makedirs(out)
            # Save to Feather format using Polars
            result.write_ipc(os.path.join(out, f"{source['name']}_{iteration}.feather"))
            logging.info(f"Results saved for iteration {iteration}")
        else:
            logging.warning(f"No data to save for iteration {iteration}")
    except Exception as e:
        logging.error(f"Error saving results for iteration {iteration}: {e}")


def process_source(source):
    """
    Processes a single source configuration.

    Args:
        source (dict): Dictionary containing source configuration.

    Returns:
        None
    """
    exiftool_path = r"C:\Program Files\ExifTool\exiftool.exe"

    # Split orthophoto paths into chunks for parallel processing
    path_list = glob.glob(source['path_list_tag'])
    chunks = np.array_split(path_list, 150)
    logging.info(f"Starting parallel processing with {len(chunks)} chunks")

    # Process each chunk in parallel
    Parallel(n_jobs=1)(
        delayed(build_database)(i, source, exiftool_path) for i in list(enumerate(chunks))
    )


def main():
    """
    Main function to process all sources defined in the configuration.
    """
    # Define a list of sources to process
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

    # Process each source
    for source in sources:
        process_source(source)


if __name__ == "__main__":
    main()
