#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import polars as pl
import glob
import os
import logging
from smac_functions import *
from config_object import config
from joblib import Parallel, delayed
from tqdm import tqdm
import rasterio as rio
import math
from timeit import default_timer as timer
import exiftool
from pathlib import PureWindowsPath

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("process.log"),
        logging.StreamHandler()
    ]
)

# Define input sources
sources = [
    {
        'out': config.main_extract_out,
        'cam_path': config.main_extract_cam_path,
        'dem_path': config.main_extract_dem_path,
        'ori': config.main_extract_ori,
        'name': config.main_extract_name,
        'path_list_tag': config.main_extract_path_list_tag
    }
]

# Process each source independently
for source in sources:
    exiftool_path = r"C:\Program Files\ExifTool\exiftool.exe"

    # Function to process a chunk of images
    def build_database(tuple_chunk):
        iteration = tuple_chunk[0]
        image = tuple_chunk[1]

        # Assign source variables to local variables
        out = source['out']
        cam_path = source['cam_path']
        dem_path = source['dem_path']
        ori = source['ori']

        logging.info(f"Starting DEM processing for iteration {iteration}")
        start_DEM_i = timer()

        try:
            start_dem_read = timer()
            # Read the DEM file and convert to DataFrame
            with rio.open(dem_path) as dem:
                arr_dem = dem.read(1)  # Read the elevation data
                transform = dem.transform  # Affine transformation
                rows, cols = np.indices(arr_dem.shape)
                rows_flat, cols_flat = rows.flatten(), cols.flatten()
                x_coords, y_coords = rio.transform.xy(transform, rows_flat, cols_flat, offset='center')

            # Create a Polars DataFrame for DEM with Float32 precision
            df_dem = pl.DataFrame({
                "Xw": pl.Series(x_coords, dtype=pl.Float32),
                "Yw": pl.Series(y_coords, dtype=pl.Float32),
                "elev": pl.Series(arr_dem.ravel(), dtype=pl.Float32)
            })
            end_dem_read = timer()
            logging.info(f"DEM processing completed for iteration {iteration} in {end_dem_read - start_dem_read:.2f} seconds")

        except Exception as e:
            logging.error(f"Error processing DEM for iteration {iteration}: {e}")
            return

        try:
            # Retrieve orthophoto paths
            start_ori = timer()
            ori_list = [glob.glob(item + "\\*.tif") for item in ori]
            path_flat = [str(PureWindowsPath(path)) for sublist in ori_list for path in sublist]
            end_ori = timer()
            logging.info(f"Retrieved orthophoto paths for iteration {iteration} in {end_ori - start_ori:.2f} seconds")
        except Exception as e:
            logging.error(f"Error retrieving orthophoto paths for iteration {iteration}: {e}")
            return

        # Process each orthophoto in the chunk
        for each_ortho in tqdm(image, desc=f"Processing iteration {iteration}"):
            try:
                start_ortho = timer()
                path, file = os.path.split(each_ortho)
                name, _ = os.path.splitext(file)
                logging.info(f"Processing orthophoto {file} for iteration {iteration}")

                # Retrieve camera position
                start_campos = timer()
                campos = pl.read_csv(cam_path, separator='\t', skip_rows=2, has_header=False)
                campos.columns = ['PhotoID', 'X', 'Y', 'Z', 'Omega', 'Phi', 'Kappa', 'r11', 'r12', 'r13',
                                  'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
                campos1 = campos.filter(pl.col('PhotoID').str.contains(name))
                xcam, ycam, zcam = campos1['X'][0], campos1['Y'][0], campos1['Z'][0]
                end_campos = timer()
                logging.info(f"Retrieved camera position for {file} in {end_campos - start_campos:.2f} seconds")

                # Retrieve solar angles from EXIF
                start_exif = timer()
                exifobj = [path for path in path_flat if name in path]
                try:
                    with exiftool.ExifToolHelper(executable=exiftool_path) as et:
                        metadata = et.get_metadata(exifobj[0])[0]
                        sunelev = float(metadata.get('XMP:SolarElevation', 0))
                        saa = float(metadata.get('XMP:SolarAzimuth', 0))
                except Exception as e:
                    logging.error(f"Error processing EXIF data for {file}: {e}")
                end_exif = timer()
                logging.info(f"Retrieved EXIF data for {file} in {end_exif - start_exif:.2f} seconds")

                # Read orthophoto bands
                start_bands = timer()
                with rio.open(each_ortho) as rst:
                    num_bands = rst.count
                    b_all = rst.read()
                    rows, cols = np.indices((rst.height, rst.width))
                    rows_flat, cols_flat = rows.flatten(), cols.flatten()
                    Xw, Yw = rio.transform.xy(rst.transform, rows_flat, cols_flat)

                    band_values = b_all[:, rows_flat, cols_flat].T
                    data = {'Xw': pl.Series(Xw, dtype=pl.Float32), 'Yw': pl.Series(Yw, dtype=pl.Float32)}
                    for idx in range(num_bands):
                        data[f'band{idx + 1}'] = band_values[:, idx]
                    df_allbands = pl.DataFrame(data)
                end_bands = timer()
                logging.info(f"Processed orthophoto bands for {file} in {end_bands - start_bands:.2f} seconds")

                # Merge DEM and band data
                start_merge = timer()
                df_dem = df_dem.with_columns([
                    pl.col("Xw").round(3),
                    pl.col("Yw").round(3)
                ]).unique()

                df_allbands = df_allbands.with_columns([
                    pl.col("Xw").round(3),
                    pl.col("Yw").round(3)
                ]).unique()

                df_merged = df_dem.join(df_allbands, on=["Xw", "Yw"], how="inner")
                end_merge = timer()
                logging.info(f"Merged data for {file} in {end_merge - start_merge:.2f} seconds")


                # Calculate angles using NumPy for atan2 with Polars expressions


                # Start timer for angle calculations
                start_angles = timer()

                # Add delta values
                df_merged = df_merged.with_columns([
                    (zcam - pl.col("elev")).alias("delta_z"),
                    (xcam - pl.col("Xw")).alias("delta_x"),
                    (ycam - pl.col("Yw")).alias("delta_y")
                ])

                # Calculate distance_xy
                df_merged = df_merged.with_columns([
                    (pl.col("delta_x").pow(2) + pl.col("delta_y").pow(2)).sqrt().alias("distance_xy")
                ])

                # Calculate angle_rad (viewing zenith angle) using pl.arctan2
                df_merged = df_merged.with_columns([
                    pl.arctan2(pl.col("delta_z"), pl.col("distance_xy")).alias("angle_rad")
                ])

                # Calculate vza (Viewing Zenith Angle)
                df_merged = df_merged.with_columns([
                    (90 - (pl.col("angle_rad") * (180 / np.pi))).round(2).alias("vza")
                ])

                # Calculate vaa_rad (viewing azimuth angle) using pl.arctan2
                df_merged = df_merged.with_columns([
                    pl.arctan2(pl.col("delta_x"), pl.col("delta_y")).alias("vaa_rad")
                ])

                # Calculate vaa_temp and vaa (Viewing Azimuth Angle)
                df_merged = df_merged.with_columns([
                    ((pl.col("vaa_rad") * (180 / np.pi)) - saa).alias("vaa_temp")
                ])

                df_merged = df_merged.with_columns([
                    ((pl.col("vaa_temp") + 360) % 360).alias("vaa")
                ])

                # Handle band1 == 65535 condition for vza and vaa
                df_merged = df_merged.with_columns([
                    pl.when(pl.col("band1") == 65535).then(None).otherwise(pl.col("vza")).alias("vza"),
                    pl.when(pl.col("band1") == 65535).then(None).otherwise(pl.col("vaa")).alias("vaa")
                ])

                # Add constant columns
                df_merged = df_merged.with_columns([
                    pl.lit(file).alias("path"),
                    pl.lit(xcam).alias("xcam"),
                    pl.lit(ycam).alias("ycam"),
                    pl.lit(sunelev).alias("sunelev"),
                    pl.lit(saa).alias("saa")
                ])

                # End timer
                end_angles = timer()
                logging.info(f"Calculated angles for {file} in {end_angles - start_angles:.2f} seconds")

                # Save to Parquet
                start_write = timer()
                df_merged.write_parquet(f"{out}\\{source['name']}_{iteration}_{file}.parquet", compression='zstd', compression_level=2)
                end_write = timer()
                logging.info(f"Saved image result for {file} in {end_write - start_write:.2f} seconds")
            except Exception as e:
                logging.error(f"Error processing orthophoto {file}: {e}")

        end_DEM_i = timer()
        logging.info(f"Total time for iteration {iteration}: {end_DEM_i - start_DEM_i:.2f} seconds")

    # Split paths into images and process in parallel
    path_list = glob.glob(source['path_list_tag'])
    images = np.array_split(path_list, len(path_list))  # Adjust image size for memory and performance
    logging.info(f"Starting parallel processing with {len(images)} chunks")

    Parallel(n_jobs=1)(delayed(build_database)(i) for i in list(enumerate(images)))
