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

def read_dem(dem_path):
    start = timer()
    try:
        with rio.open(dem_path) as dem:
            arr_dem = dem.read(1)  # elevation data, often float32 if set in the raster
            # Ensure arr_dem is float32:
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

        # Round and unique can remain, but ensure no unnecessary large intermediate:
        df_dem = df_dem.with_columns([
            pl.col("Xw").round(3),
            pl.col("Yw").round(3)
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

def process_orthophoto(each_ortho, df_dem, cam_path, path_flat, out, source, iteration, exiftool_path):
    try:
        start_ortho = timer()
        path, file = os.path.split(each_ortho)
        name, _ = os.path.splitext(file)
        logging.info(f"Processing orthophoto {file} for iteration {iteration}")

        # Retrieve camera position (already float32)
        start_campos = timer()
        campos = pl.read_csv(cam_path, separator='\t', skip_rows=2, has_header=False)
        campos.columns = ['PhotoID', 'X', 'Y', 'Z', 'Omega', 'Phi', 'Kappa', 'r11', 'r12', 'r13',
                          'r21', 'r22', 'r23', 'r31', 'r32', 'r33']
        # Convert camera coords to float32
        campos = campos.with_columns([
            pl.col("X").cast(pl.Float32),
            pl.col("Y").cast(pl.Float32),
            pl.col("Z").cast(pl.Float32)
        ])
        campos1 = campos.filter(pl.col('PhotoID').str.contains(name))
        xcam, ycam, zcam = campos1['X'][0], campos1['Y'][0], campos1['Z'][0]
        end_campos = timer()
        logging.info(f"Retrieved camera position for {file} in {end_campos - start_campos:.2f} seconds")

        # Retrieve solar angles from EXIF
        start_exif = timer()
        exifobj = [pth for pth in path_flat if name in pth]
        try:
            with exiftool.ExifToolHelper(executable=exiftool_path) as et:
                metadata = et.get_metadata(exifobj[0])[0]
                sunelev = float(metadata.get('XMP:SolarElevation', 0))
                saa = float(metadata.get('XMP:SolarAzimuth', 0))
        except Exception as e:
            logging.error(f"Error processing EXIF data for {file}: {e}")
            sunelev, saa = 0.0, 0.0  # fallback
        end_exif = timer()
        logging.info(f"Retrieved EXIF data for {file} in {end_exif - start_exif:.2f} seconds")

        # Read orthophoto bands
        start_bands = timer()
        with rio.open(each_ortho) as rst:
            num_bands = rst.count
            b_all = rst.read()  # This might be uint16 for reflectance or 8-bit
            # Ensure band arrays are as small as possible
            # If they are float64, convert:
            if b_all.dtype == np.float64:
                b_all = b_all.astype(np.float32)

            rows, cols = np.indices((rst.height, rst.width))
            rows_flat, cols_flat = rows.flatten(), cols.flatten()
            Xw, Yw = rio.transform.xy(rst.transform, rows_flat, cols_flat, offset='center')

            # Convert coordinates to float32
            Xw = np.array(Xw, dtype=np.float32)
            Yw = np.array(Yw, dtype=np.float32)

            band_values = b_all[:, rows_flat, cols_flat].T
            # If possible, cast band_values to uint16 or float32:
            # If original data is reflectance and fits in uint16, keep it as uint16
            if band_values.dtype == np.float64:
                band_values = band_values.astype(np.float32)

            data = {'Xw': pl.Series(Xw, dtype=pl.Float32),
                    'Yw': pl.Series(Yw, dtype=pl.Float32)}
            for idx in range(num_bands):
                # If data is reflectance 0-65535, keep uint16:
                # If it's already uint16, no need to cast.
                # If float needed, cast to float32.
                # We'll assume it's uint16 or float32 based on the data source
                # If reflectance is stored as int16 or uint16, do:
                # band_values[:, idx] = band_values[:, idx].astype(np.uint16)
                data[f'band{idx + 1}'] = pl.Series(band_values[:, idx], dtype=pl.UInt16)

            df_allbands = pl.DataFrame(data)
        end_bands = timer()
        logging.info(f"Processed orthophoto bands for {file} in {end_bands - start_bands:.2f} seconds")

        # Merge DEM and band data
        start_merge = timer()
        # df_dem already float32
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

        # Calculate angles
        start_angles = timer()

        df_merged = df_merged.with_columns([
            (pl.lit(zcam, dtype=pl.Float32) - pl.col("elev")).alias("delta_z"),
            (pl.lit(xcam, dtype=pl.Float32) - pl.col("Xw")).alias("delta_x"),
            (pl.lit(ycam, dtype=pl.Float32) - pl.col("Yw")).alias("delta_y")
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

        df_merged = df_merged.with_columns([
            pl.lit(file).alias("path"),
            pl.lit(xcam, dtype=pl.Float32).alias("xcam"),
            pl.lit(ycam, dtype=pl.Float32).alias("ycam"),
            pl.lit(sunelev, dtype=pl.Float32).alias("sunelev"),
            pl.lit(saa, dtype=pl.Float32).alias("saa")
        ])

        end_angles = timer()
        logging.info(f"Calculated angles for {file} in {end_angles - start_angles:.2f} seconds")

        # Save to Parquet
        start_write = timer()
        df_merged.write_parquet(f"{out}\\{source['name']}_{iteration}_{file}.parquet", compression='zstd', compression_level=2)
        end_write = timer()
        logging.info(f"Saved image result for {file} in {end_write - start_write:.2f} seconds")

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

    logging.info(f"Starting DEM processing for iteration {iteration}")
    start_DEM_i = timer()

    # Read DEM once per chunk
    df_dem = read_dem(dem_path)

    # Retrieve orthophoto paths once per chunk
    path_flat = retrieve_orthophoto_paths(ori)

    for each_ortho in tqdm(images, desc=f"Processing iteration {iteration}"):
        process_orthophoto(each_ortho, df_dem, cam_path, path_flat, out, source, iteration, exiftool_path)

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
            'path_list_tag': config.main_extract_path_list_tag
        }
    ]

    exiftool_path = r"C:\Program Files\ExifTool\exiftool.exe"

    for source in sources:
        path_list = glob.glob(source['path_list_tag'])
        # Consider fewer chunks if you have memory issues
        # For example, if you have millions of images, do not split into so many chunks
        # Instead, try something like:
        num_chunks = 10  # arbitrary smaller number of chunks
        images_split = np.array_split(path_list, num_chunks)
        logging.info(f"Starting parallel processing with {len(images_split)} chunks")

        Parallel(n_jobs=1)(delayed(build_database)(i, source, exiftool_path) for i in enumerate(images_split))

if __name__ == "__main__":
    main()
