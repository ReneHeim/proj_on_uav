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
import numpy as np
from timeit import default_timer as timer
import exiftool
from pathlib import PureWindowsPath
import traceback

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("process.log"),
            logging.StreamHandler()
        ]
    )

def read_dem(dem_path, precision):
    start = timer()
    try:
        with rio.open(dem_path) as dem:
            arr_dem = dem.read(1)  # elevation data
            # Ensure arr_dem is float32
            if arr_dem.dtype != np.float32:
                arr_dem = arr_dem.astype(np.float32)

            transform = dem.transform
            rows, cols = np.indices(arr_dem.shape)
            rows_flat, cols_flat = rows.flatten(), cols.flatten()
            x_coords, y_coords = rio.transform.xy(transform, rows_flat, cols_flat, offset='center')

        # Convert coordinates to float32
        x_coords = np.array(x_coords, dtype=np.float32)
        y_coords = np.array(y_coords, dtype=np.float32)

        df_dem = pl.DataFrame({
            "Xw": pl.Series(x_coords, dtype=pl.Float32),
            "Yw": pl.Series(y_coords, dtype=pl.Float32),
            "elev": pl.Series(arr_dem.ravel(), dtype=pl.Float32)
        })

        df_dem = df_dem.with_columns([
            pl.col("Xw").round(precision),
            pl.col("Yw").round(precision)
        ]).unique()

        end = timer()
        logging.info(f"DEM read and processed in {end - start:.2f} seconds")
        return df_dem
    except Exception as e:
        logging.error(f"Error reading DEM: {e}")
        raise

def retrieve_orthophoto_paths(ori):
    start = timer()
    try:
        ori_list = [glob.glob(os.path.join(item, "*.tif")) for item in ori]
        path_flat = [str(PureWindowsPath(path)) for sublist in ori_list for path in sublist]
        end = timer()
        logging.info(f"Retrieved orthophoto paths in {end - start:.2f} seconds")
        return path_flat
    except Exception as e:
        logging.error(f"Error retrieving orthophoto paths: {e}")
        raise

def extract_sun_angles(name, path_flat, exiftool_path):
    start = timer()
    sunelev, saa = 0.0, 0.0
    try:
        exifobj = [pth for pth in path_flat if name in pth]
        if len(exifobj) == 0:
            logging.warning(f"No matching orthophoto found for EXIF data extraction: {name}")
            return sunelev, saa

        with exiftool.ExifToolHelper(executable=exiftool_path) as et:
            metadata = et.get_metadata(exifobj[0])[0]
            sunelev = float(metadata.get('XMP:SolarElevation', 0))
            saa = float(metadata.get('XMP:SolarAzimuth', 0))
        end = timer()
        logging.info(f"Retrieved EXIF (Sun Angles) for {name} in {end - start:.2f} seconds")
        return sunelev, saa
    except Exception as e:
        logging.error(f"Error processing EXIF data for {name}: {e}")
        return sunelev, saa

def get_camera_position(cam_path, name):
    start = timer()
    try:
        campos = pl.read_csv(cam_path, separator='\t', skip_rows=2, has_header=False)
        campos.columns = ['PhotoID', 'X', 'Y', 'Z', 'Omega', 'Phi', 'Kappa', 'r11', 'r12', 'r13',
                          'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
        campos = campos.with_columns([
            pl.col("X").cast(pl.Float32),
            pl.col("Y").cast(pl.Float32),
            pl.col("Z").cast(pl.Float32)
        ])
        campos1 = campos.filter(pl.col('PhotoID').str.contains(name))
        xcam, ycam, zcam = campos1['X'][0], campos1['Y'][0], campos1['Z'][0]
        end = timer()
        logging.info(f"Retrieved camera position for {name} in {end - start:.2f} seconds")
        return float(xcam), float(ycam), float(zcam)
    except Exception as e:
        logging.error(f"Error retrieving camera position for {name}: {e}")
        raise

def read_orthophoto_bands(each_ortho, precision):
    start_bands = timer()
    try:
        with rio.open(each_ortho) as rst:
            num_bands = rst.count
            b_all = rst.read()  # possibly uint16 or float
            if b_all.dtype == np.float64:
                b_all = b_all.astype(np.float32)

            rows, cols = np.indices((rst.height, rst.width))
            rows_flat, cols_flat = rows.flatten(), cols.flatten()
            Xw, Yw = rio.transform.xy(rst.transform, rows_flat, cols_flat, offset='center')

            # Convert coordinates to float32
            Xw = np.array(Xw, dtype=np.float32)
            Yw = np.array(Yw, dtype=np.float32)

            band_values = b_all[:, rows_flat, cols_flat].T
            if band_values.dtype == np.float64:
                band_values = band_values.astype(np.float32)

            data = {
                'Xw': pl.Series(Xw, dtype=pl.Float32),
                'Yw': pl.Series(Yw, dtype=pl.Float32)
            }

            # Assume band_values suitable as UInt16 if reflectance-like
            # Adjust dtype if needed:
            # For simplicity, let's keep them as UInt16 as per original logic
            for idx in range(num_bands):
                data[f'band{idx + 1}'] = pl.Series(band_values[:, idx], dtype=pl.UInt16)

            df_allbands = pl.DataFrame(data)
            df_allbands = df_allbands.with_columns([
                pl.col("Xw").round(precision),
                pl.col("Yw").round(precision)
            ]).unique()

        end_bands = timer()
        logging.info(f"Processed orthophoto bands for {os.path.basename(each_ortho)} in {end_bands - start_bands:.2f} seconds")
        return df_allbands
    except Exception as e:
        logging.error(f"Error reading bands from {each_ortho}: {e}")
        raise

def merge_data(df_dem, df_allbands, precision):
    start_merge = timer()
    try:
        # df_dem already processed; ensure rounding (may be redundant)
        df_dem = df_dem.with_columns([
            pl.col("Xw").round(precision),
            pl.col("Yw").round(precision)
        ]).unique()

        # df_allbands already rounded and unique
        df_merged = df_dem.join(df_allbands, on=["Xw", "Yw"], how="inner")
        end_merge = timer()
        logging.info(f"Merged DEM and band data in {end_merge - start_merge:.2f} seconds")
        return df_merged
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        raise

def calculate_angles(df_merged, xcam, ycam, zcam, sunelev, saa):
    start_angles = timer()
    try:

        # Step 1: Add delta_z, delta_x, delta_y
        df_merged = df_merged.with_columns([
            (pl.lit(zcam, dtype=pl.Float32) - pl.col("elev")).alias("delta_z"),
            (pl.lit(xcam, dtype=pl.Float32) - pl.col("Xw")).alias("delta_x"),
            (pl.lit(ycam, dtype=pl.Float32) - pl.col("Yw")).alias("delta_y"),
        ])


        # Step 2: Calculate distance_xy
        df_merged = df_merged.with_columns([
            ((pl.col("delta_x").pow(2) + pl.col("delta_y").pow(2)).sqrt()).alias("distance_xy"),
        ])

        # Step 3: Calculate angle_rad (viewing zenith angle)
        df_merged = df_merged.with_columns([
            pl.arctan2(pl.col("delta_z"), pl.col("distance_xy")).alias("angle_rad")
        ])

        # Step 4: Calculate vza (Viewing Zenith Angle)
        df_merged = df_merged.with_columns([
            (90 - (pl.col("angle_rad") * (180 / np.pi))).round(2).alias("vza")
        ])

        # Step 5: Calculate vaa_rad (viewing azimuth angle)
        df_merged = df_merged.with_columns([
            pl.arctan2(pl.col("delta_x"), pl.col("delta_y")).alias("vaa_rad")
        ])

        # Step 6: Calculate vaa_temp and vaa (Viewing Azimuth Angle)
        # Calculate vaa_temp and vaa (Viewing Azimuth Angle)
        df_merged = df_merged.with_columns([
            ((pl.col("vaa_rad") * (180 / np.pi)) - saa).alias("vaa_temp")
        ])

        df_merged = df_merged.with_columns([
            ((pl.col("vaa_temp") + 360) % 360).alias("vaa")
        ])

        # Step 7: Handle no-data condition
        df_merged = df_merged.with_columns([
            pl.when(pl.col("band1") == 65535).then(None).otherwise(pl.col("vza")).alias("vza"),
            pl.when(pl.col("band1") == 65535).then(None).otherwise(pl.col("vaa")).alias("vaa")
        ])

        # Step 8: Add metadata columns
        df_merged = df_merged.with_columns([
            pl.lit(xcam, dtype=pl.Float32).alias("xcam"),
            pl.lit(ycam, dtype=pl.Float32).alias("ycam"),
            pl.lit(sunelev, dtype=pl.Float32).alias("sunelev"),
            pl.lit(saa, dtype=pl.Float32).alias("saa")
        ])

        end_angles = timer()
        logging.info(f"Calculated angles in {end_angles - start_angles:.2f} seconds")
        return df_merged

    except Exception as e:
        logging.error(f"Error calculating angles: {e}")
        logging.error(traceback.format_exc())
        raise



def save_parquet(df, out, source, iteration, file):
    start_write = timer()
    try:
        df.write_parquet(f"{out}\\{source['name']}_{iteration}_{file}.parquet", compression='zstd', compression_level=2)
        end_write = timer()
        logging.info(f"Saved image result for {file} in {end_write - start_write:.2f} seconds")
    except Exception as e:
        logging.error(f"Error saving parquet for {file}: {e}")
        raise

def process_orthophoto(each_ortho, df_dem, cam_path, path_flat, out, source, iteration, exiftool_path, precision):
    try:
        start_ortho = timer()
        path, file = os.path.split(each_ortho)
        name, _ = os.path.splitext(file)
        logging.info(f"Processing orthophoto {file} for iteration {iteration}")

        # Get camera position
        xcam, ycam, zcam = get_camera_position(cam_path, name)

        # Get solar angles
        sunelev, saa = extract_sun_angles(name, path_flat, exiftool_path)

        # Read orthophoto bands
        df_allbands = read_orthophoto_bands(each_ortho, precision)

        # Merge DEM and band data
        df_merged = merge_data(df_dem, df_allbands, precision)

        # Log structure of df_merged

        # Ensure necessary columns exist
        required_columns = ["Xw", "Yw", "elev"]
        if not all(col in df_merged.columns for col in required_columns):
            raise ValueError(f"Missing required columns in df_merged: {required_columns}")

        # Calculate angles
        df_merged = calculate_angles(df_merged, xcam, ycam, zcam, sunelev, saa)

        # Add path column
        df_merged = df_merged.with_columns([
            pl.lit(file).alias("path")
        ])

        # Save to Parquet
        save_parquet(df_merged, out, source, iteration, file)

        end_ortho = timer()
        logging.info(f"Finished processing orthophoto {file} for iteration {iteration} in {end_ortho - start_ortho:.2f} seconds")

    except Exception as e:
        logging.error(f"Error processing orthophoto {file}: {e}")
        logging.error(traceback.format_exc())


def build_database(tuple_chunk, source, exiftool_path):
    iteration = tuple_chunk[0]
    images = tuple_chunk[1]

    out = source['out']
    cam_path = source['cam_path']
    dem_path = source['dem_path']
    ori = source['ori']
    precision = source['precision']

    logging.info(f"Starting DEM processing for iteration {iteration}")
    start_DEM_i = timer()

    # Read DEM once per chunk
    df_dem = read_dem(dem_path, precision)

    # Retrieve orthophoto paths once per chunk
    path_flat = retrieve_orthophoto_paths(ori)

    for each_ortho in tqdm(images, desc=f"Processing iteration {iteration}"):
        process_orthophoto(each_ortho, df_dem, cam_path, path_flat, out, source, iteration, exiftool_path, precision)

    end_DEM_i = timer()
    logging.info(f"Total time for iteration {iteration}: {end_DEM_i - start_DEM_i:.2f} seconds")

def main():
    configure_logging()

    sources = [
        {
            'out': config.main_extract_out,
            'cam_path': config.main_extract_cam_path,
            'dem_path': config.main_extract_dem_path,
            'ori': config.main_extract_ori,
            'name': config.main_extract_name,
            'path_list_tag': config.main_extract_path_list_tag,
            "precision": config.precision
        }
    ]

    exiftool_path = r"C:\Program Files\ExifTool\exiftool.exe"

    for source in sources:
        path_list = glob.glob(source['path_list_tag'])
        # Adjust chunking based on data size and resources
        num_chunks = 10
        images_split = np.array_split(path_list, num_chunks)
        logging.info(f"Starting parallel processing with {len(images_split)} chunks")

        Parallel(n_jobs=1)(delayed(build_database)(i, source, exiftool_path) for i in enumerate(images_split))

if __name__ == "__main__":
    main()
