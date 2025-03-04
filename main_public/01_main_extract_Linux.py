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
from pathlib import Path
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import json
import subprocess
import subprocess
import re
import json
from pathlib import Path

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




def fix_path(path_str):
    # Replace backslashes with forward slashes using regex.
    return re.sub(r"\\+", "/", path_str)

def extract_sun_angles(name, path_flat, exiftool_path):
    start = timer()
    sunelev, saa = 0.0, 0.0  # Default values if metadata is missing

    try:
        # Find the correct file path that contains the given name.
        exifobj = [pth for pth in path_flat if name in pth]
        if not exifobj:
            logging.warning(f"No matching orthophoto found for EXIF data extraction: {name}")
            return sunelev, saa

        # Fix the file path with regex to replace any backslashes with forward slashes.
        original_path = exifobj[0]
        fixed_path = fix_path(original_path)
        # Also, ensure it's a proper POSIX path:
        file_path = Path(fixed_path).as_posix()

        # Build the command as a list; passing the file path directly avoids shell issues.
        cmd = [exiftool_path, "-j", "-SolarElevation", "-SolarAzimuth", file_path]
        output = subprocess.run(cmd, capture_output=True, text=True)

        if output.returncode != 0:
            logging.warning(f"EXIF metadata extraction failed for {name}. Error: {output.stderr}")
            return sunelev, saa  # Return default values

        # Use regex to extract values from the text output (if you still prefer regex over JSON parsing):
        # Alternatively, if exiftool returns JSON, you can parse it:
        metadata_list = json.loads(output.stdout)
        if not metadata_list:
            logging.warning(f"No EXIF metadata found for {name}")
            return sunelev, saa  # Return default values

        metadata = metadata_list[0]
        sunelev = float(metadata.get('Solar Elevation', 0))
        saa = float(metadata.get('Solar Azimuth', 0))

        end = timer()
        logging.info(f"Retrieved EXIF (Sun Angles) for {name} in {end - start:.2f} seconds")

    except Exception as e:
        logging.warning(f"EXIF metadata extraction failed for {name}, skipping. Error: {e}")

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
        # Ensure DEM has only one elevation per coordinate
        df_dem = df_dem.group_by(["Xw", "Yw"]).agg([
            pl.col("elev").mean().alias("elev")  # Choose mean, min, or max
        ])

        # Ensure rounding precision
        df_dem = df_dem.with_columns([
            pl.col("Xw").round(precision),
            pl.col("Yw").round(precision)
        ]).unique()

        df_allbands = df_allbands.with_columns([
            pl.col("Xw").round(precision),
            pl.col("Yw").round(precision)
        ]).unique()

        # Use LazyFrame for memory optimization
        df_dem_lazy = df_dem.lazy()
        df_allbands_lazy = df_allbands.lazy()

        df_merged = df_dem_lazy.join(df_allbands_lazy, on=["Xw", "Yw"], how="inner").collect()

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
        output_path = Path(out) / f"{source['name']}_{iteration}_{file}.parquet"
        df.write_parquet(str(output_path), compression='zstd', compression_level=2)
        end_write = timer()
        logging.info(f"Saved image result for {file} in {end_write - start_write:.2f} seconds")
    except Exception as e:
        logging.error(f"Error saving parquet for {file}: {e}")
        raise

def check_alignment(dem_path, ortho_path):
    """
    Check if the DEM and orthophoto are aligned:
    - Same CRS
    - Same pixel size
    - Same alignment of pixel boundaries

    Returns True if aligned, False otherwise.
    """
    try:
        with rasterio.open(dem_path) as dem_src, rasterio.open(ortho_path) as ortho_src:
            # Check CRS
            if dem_src.crs != ortho_src.crs:
                logging.info(f"CRS mismatch: DEM={dem_src.crs}, Ortho={ortho_src.crs}")
                return False

            # Extract transforms
            dem_transform = dem_src.transform
            ortho_transform = ortho_src.transform

            # Check pixel sizes
            dem_res = (dem_transform.a, dem_transform.e)  # (xres, yres)
            ortho_res = (ortho_transform.a, ortho_transform.e)
            if not (abs(dem_res[0]) == abs(ortho_res[0]) and abs(dem_res[1]) == abs(ortho_res[1])):
                logging.info(f"Resolution mismatch: DEM={dem_res}, Ortho={ortho_res}")
                return False

            # Check alignment: We must ensure that the pixel grid aligns.
            # We'll check if the difference in the origins aligns with a multiple of the pixel size.
            # For alignment: (ortho_transform.c - dem_transform.c) should be an integer multiple of the xres
            # and (ortho_transform.f - dem_transform.f) should be an integer multiple of the yres.

            x_offset = (ortho_transform.c - dem_transform.c) / dem_res[0]
            y_offset = (ortho_transform.f - dem_transform.f) / dem_res[1]

            # If x_offset and y_offset are close to integers, we consider them aligned
            if not (abs(x_offset - round(x_offset)) < 1e-6 and abs(y_offset - round(y_offset)) < 1e-6):
                logging.info(f"Pixel alignment mismatch: x_offset={x_offset}, y_offset={y_offset}")
                return False

            # If we reach here, DEM and orthophoto are aligned
            logging.info(f" DEM and orthophoto are aligned: x_offset={x_offset}, y_offset={y_offset}")
            return True
    except Exception as e:
        logging.error(f"Error checking alignment: {e}")
        return False


def coregister_and_resample(input_path, ref_path, output_path, target_resolution=None, resampling=Resampling.nearest):
    """
    Reproject and resample the input raster (orthophoto) to match the reference raster (DEM).
    If target_resolution is provided (e.g., (10, 10)), the output will be resampled to that resolution.
    """
    try:
        with rasterio.open(ref_path) as ref_src:
            ref_crs = ref_src.crs
            ref_transform = ref_src.transform
            ref_width = ref_src.width
            ref_height = ref_src.height
            ref_bounds = ref_src.bounds

        with rasterio.open(input_path) as src:
            # Compute the transform, width, and height of the output raster
            if target_resolution:
                # If a target resolution is provided, recalculate transform for the new resolution
                # We'll use calculate_default_transform but override the resolution
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs,
                    ref_crs,
                    ref_width,
                    ref_height,
                    *ref_bounds
                )

                # Now adjust the transform for the target resolution
                # The calculate_default_transform gives a transform that matches the reference image,
                # We just need to scale it if we want a different resolution.
                xres, yres = target_resolution
                # We know dst_transform:
                # dst_transform = | xres, 0, x_min|
                #                 | 0, -yres, y_max|
                #                 | 0, 0, 1       |
                # If we want a different resolution, we can recompute width/height
                x_min, y_min, x_max, y_max = ref_bounds
                dst_width = int((x_max - x_min) / xres)
                dst_height = int((y_max - y_min) / abs(yres))
                dst_transform = rasterio.transform.from_bounds(x_min, y_min, x_max, y_max, dst_width, dst_height)
            else:
                # Use the reference transform and size directly
                dst_transform, dst_width, dst_height = ref_transform, ref_width, ref_height

            dst_kwargs = src.meta.copy()
            dst_kwargs.update({
                "crs": ref_crs,
                "transform": dst_transform,
                "width": dst_width,
                "height": dst_height,
            })

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with rasterio.open(output_path, "w", **dst_kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=ref_crs,
                        resampling=resampling
                    )

        logging.info(f"Co-registration and resampling completed. Output saved to {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error in co-registration: {e}")
        raise


# Integration example within process_orthophoto:
def process_orthophoto(each_ortho, df_dem, cam_path, path_flat, out, source, iteration, exiftool_path, precision):
    try:
        start_ortho = timer()
        path, file = os.path.split(each_ortho)
        name, _ = os.path.splitext(file)
        logging.info(f"Processing orthophoto {file} for iteration {iteration}")

        # Paths
        dem_path = source['dem_path']

        # Co-registration Check
        # Before proceeding, ensure DEM and orthophoto are aligned
        if not check_alignment(dem_path, each_ortho):
            # Not aligned, co-register orthophoto to DEM
            coreg_path = os.path.join(out, f"coreg_{file}")
            # Optionally, define a target resolution (e.g., (10,10)) if you want to change pixel size
            target_resolution = None  # or (10, 10)
            each_ortho = coregister_and_resample(each_ortho, dem_path, coreg_path, target_resolution=target_resolution, resampling=Resampling.bilinear)

            # Verify alignment again after co-registration
            if not check_alignment(dem_path, each_ortho):
                raise ValueError("Co-registration failed: orthophoto and DEM are still not aligned.")

        # Get camera position
        xcam, ycam, zcam = get_camera_position(cam_path, name)

        # Get solar angles
        sunelev, saa = extract_sun_angles(name, path_flat, exiftool_path)

        # Read orthophoto bands
        df_allbands = read_orthophoto_bands(each_ortho, precision)

        # Merge DEM and band data
        df_merged = merge_data(df_dem, df_allbands, precision)

        # Log structure of df_merged
        logging.info(f"df_merged columns before angle calculation: {df_merged.columns}")

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

    exiftool_path = r"exiftool"

    for source in sources:
        path_list = glob.glob(source['path_list_tag'])
        # Adjust chunking based on data size and resources
        num_chunks = 10
        images_split = np.array_split(path_list, num_chunks)
        logging.info(f"Starting parallel processing with {len(images_split)} chunks")


        Parallel(n_jobs=1)(delayed(build_database)(i, source, exiftool_path) for i in enumerate(images_split))

if __name__ == "__main__":
    main()
