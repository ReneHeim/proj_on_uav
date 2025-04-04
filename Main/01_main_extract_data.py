#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
from types import NoneType

import numpy as np
import pytz
import glob
import os
import logging
import rasterio as rio
from timeit import default_timer as timer
from pathlib import PureWindowsPath, Path
import traceback
import re

from numpy.ma.core import masked
from pyproj import Transformer
from rasterio.warp import reproject, Resampling, calculate_default_transform
from tqdm import tqdm

from Main.functions.date_time_functions import convert_to_timezone
from Main.functions.merge_analysis_functions import visualize_coordinate_alignment, analyze_kdtree_matching, merge_data
from Main.functions.polygon_filtering_functions import is_pos_inside_polygon, filter_df_by_polygon
from Main.functions.raster_functions import *  # Your helper functions, e.g., xyval, latlon_to_utm32n_series, etc.
from config_object import config_object
import geopandas as gpd
from shapely.geometry import Point
import polars as pl
import pysolar as solar


def config_objecture_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("process.log", encoding='utf-8'),  # Note the encoding parameter
            logging.StreamHandler()
        ]
    )


# ------------------------------
# DEM Reading: True Elevation Data with CRS Transformation
# ------------------------------
def read_dem(dem_path, precision, transform_to_utm=True, target_crs="EPSG:32632"):
    start = timer()
    try:
        with rio.open(dem_path) as dem:
            arr_dem = dem.read(1)  # elevation data
            if arr_dem.dtype != np.float32:
                arr_dem = arr_dem.astype(np.float32)
            transform = dem.transform
            rows, cols = np.indices(arr_dem.shape)
            rows_flat, cols_flat = rows.flatten(), cols.flatten()
            # Get native (pixel center) coordinates in DEM's CRS
            x_coords, y_coords = rio.transform.xy(transform, rows_flat, cols_flat, offset='center')

            # Transform to target CRS if requested (e.g., UTM for polygons)
            if transform_to_utm:
                from pyproj import Transformer
                transformer = Transformer.from_crs(dem.crs, target_crs, always_xy=True)
                x_trans, y_trans = transformer.transform(x_coords, y_coords)
                x_coords = np.array(x_trans, dtype=np.float32).round(precision)
                y_coords = np.array(y_trans, dtype=np.float32).round(precision)
            else:
                x_coords = np.array(x_coords, dtype=np.float32).round(precision)
                y_coords = np.array(y_coords, dtype=np.float32).round(precision)

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


# ------------------------------
# Orthophoto Reading: Reflectance Bands
# ------------------------------
def read_orthophoto_bands(each_ortho, precision, transform_to_utm=True, target_crs="EPSG:32632"):
    start_bands = timer()
    try:
        with rio.open(each_ortho) as rst:
            num_bands = rst.count
            b_all = rst.read(masked=True)  # Read all bands

            rows, cols = np.indices((rst.height, rst.width))
            rows_flat, cols_flat = rows.flatten(), cols.flatten()
            Xw, Yw = rio.transform.xy(rst.transform, rows_flat, cols_flat, offset='center')

            if transform_to_utm:
                transformer = Transformer.from_crs(rst.crs, target_crs, always_xy=True)
                x_trans, y_trans = transformer.transform(Xw, Yw)
                Xw = np.array(x_trans)
                Yw = np.array(y_trans)
            else:
                Xw = np.array(Xw)
                Yw = np.array(Yw)

            band_values = b_all.reshape(num_bands, -1).T

            data = {
                'Xw': pl.Series(Xw),
                'Yw': pl.Series(Yw)
            }
            for idx in range(num_bands):
                data[f'band{idx + 1}'] = pl.Series(band_values[:, idx])
            df_allbands = pl.DataFrame(data)


        end_bands = timer()
        logging.info(
            f"Processed orthophoto bands for {os.path.basename(each_ortho)} in {end_bands - start_bands:.2f} seconds")
        return df_allbands
    except Exception as e:
        logging.error(f"Error reading bands from {each_ortho}: {e}")
        raise


# ------------------------------
# Calculate Viewing Angles
# ------------------------------
def calculate_angles(df_merged, xcam, ycam, zcam, sunelev, saa):
    start_angles = timer()
    try:
        df_merged = df_merged.with_columns([
            (pl.lit(zcam, dtype=pl.Float32) - pl.col("elev")).alias("delta_z"),
            (pl.lit(xcam, dtype=pl.Float32) - pl.col("Xw")).alias("delta_x"),
            (pl.lit(ycam, dtype=pl.Float32) - pl.col("Yw")).alias("delta_y")
        ])
        df_merged = df_merged.with_columns([
            ((pl.col("delta_x").pow(2) + pl.col("delta_y").pow(2)).sqrt()).alias("distance_xy")
        ])
        df_merged = df_merged.with_columns([
            pl.arctan2(pl.col("delta_z"), pl.col("distance_xy")).alias("angle_rad")
        ])
        df_merged = df_merged.with_columns([
            (90 - (pl.col("angle_rad") * (180 / np.pi))).round(2).alias("vza")
        ])
        df_merged = df_merged.with_columns([
            pl.arctan2(pl.col("delta_x"), pl.col("delta_y")).alias("vaa_rad")
        ])
        df_merged = df_merged.with_columns([
            ((pl.col("vaa_rad") * (180 / np.pi)) - saa).alias("vaa_temp")
        ])
        df_merged = df_merged.with_columns([
            ((pl.col("vaa_temp") + 360) % 360).alias("vaa")
        ])

        # Compute the lower and upper bounds for the central 90%
        p05 = df_merged.select(pl.col("elev").quantile(0.02)).item()
        p95 = df_merged.select(pl.col("elev").quantile(0.98)).item()

        # Use these bounds to mask out values outside the central 90%
        df_merged = df_merged.with_columns([
            pl.when(pl.col("band1") == 65535).then(None).otherwise(pl.col("vza")).alias("vza"),
            pl.when(pl.col("band1") == 65535).then(None).otherwise(pl.col("vaa")).alias("vaa"),
            pl.when((pl.col("elev") < p05) | (pl.col("elev") > p95))
            .then(None)
            .otherwise(pl.col("elev"))
            .alias("elev")
        ])

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


def coregister_and_resample(input_path, ref_path, output_path, target_resolution=None, resampling=Resampling.nearest):
    """
    Reproject and resample the input raster (orthophoto) to match the reference raster (DEM).
    If target_resolution is provided (e.g., (10, 10)), the output will be resampled to that resolution.
    """
    try:
        with rio.open(ref_path) as ref_src:
            ref_crs = ref_src.crs
            ref_transform = ref_src.transform
            ref_width = ref_src.width
            ref_height = ref_src.height
            ref_bounds = ref_src.bounds

        with rio.open(input_path) as src:
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
                dst_transform = rio.transform.from_bounds(x_min, y_min, x_max, y_max, dst_width, dst_height)
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

            with rio.open(output_path, "w", **dst_kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
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


def check_alignment(dem_path, ortho_path):
    """
    Check if the DEM and orthophoto are aligned:
    - Same CRS
    - Same pixel size
    - Same alignment of pixel boundaries

    Returns True if aligned, False otherwise.
    """
    try:
        with rio.open(dem_path) as dem_src, rio.open(ortho_path) as ortho_src:
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


# ------------------------------
# Save Output to Parquet
# ------------------------------
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


# ------------------------------
# Utility Functions for Paths and EXIF
# ------------------------------
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
    return re.sub(r"\\+", "/", path_str)


def extract_sun_angles(name, lon, lat, datetime_str, timezone="UTC"):
    """
    Calculate sun angles using pysolar library based on timestamp and position.

    Args:
        name (str): Name of the image file (for logging)
        lon (float): longitude
        lat (float): latitude
        datetime_str (str): Datetime string in the format 'YYYY-MM-DD HH:MM:SS'
        timezone (str): Timezone of the datetime string

    Returns:
        tuple: (solar_elevation, solar_azimuth) in degrees
    """
    start = timer()
    sunelev, saa = 0.0, 0.0

    try:

        dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
        dt_with_tz = convert_to_timezone(dt, timezone)

        # Calculate solar position using pysolar
        solar_elevation = solar.solar.get_altitude(lat, lon, dt_with_tz)
        solar_azimuth = solar.solar.get_azimuth(lat, lon, dt_with_tz)

        # Ensure azimuth is in the range [0, 360]
        if solar_azimuth < 0:
            solar_azimuth += 360

        sunelev = solar_elevation
        saa = solar_azimuth

        end = timer()
        logging.info(
            f"Calculated sun angles for {name} using pysolar: elevation={sunelev:.2f}°, azimuth={saa:.2f}° in {end - start:.2f} seconds")
    except Exception as e:
        logging.warning(f"Sun angle calculation with pysolar failed for {name}, skipping. Error: {e}")

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
        lon, lat, zcam = campos1['X'][0], campos1['Y'][0], campos1['Z'][0]

        end = timer()
        logging.info(f"Retrieved camera position for {name} in {end - start:.2f} seconds")
        return float(lon), float(lat), float(zcam)
    except Exception as e:
        logging.error(f"Error retrieving camera position for {name}: {e}")
        raise


# ------------------------------
# Core Processing Function for an Orthophoto
# ------------------------------
def process_orthophoto(orthophoto, cam_path, path_flat, out, source, iteration, exiftool_path, precision,
                       polygon_filtering=False, alignment=False):
    try:

        #PART 1: Read the Raster
        start_ortho = timer()
        path, file = os.path.split(orthophoto)
        name, _ = os.path.splitext(file)

        logging.info(f"Processing orthophoto {file} for iteration {iteration}")

        # Get camera position from the camera file
        lon, lat, zcam = get_camera_position(cam_path, name)

        # Optional: Ensure DEM and orthophoto are aligned
        if alignment:
            if not check_alignment(source['dem_path'], orthophoto):
                coreg_path = os.path.join(out, f"coreg_{file}")
                orthophoto = coregister_and_resample(orthophoto, source['dem_path'], coreg_path, target_resolution=None,
                                                     resampling=Resampling.bilinear)
                if not check_alignment(source['dem_path'], orthophoto):
                    raise ValueError("Co-registration failed: orthophoto and DEM are still not aligned.")


        # Read orthophoto bands (assumed to be transformed already if needed)
        df_allbands = read_orthophoto_bands(orthophoto, precision)

        #PART 2: Merge orthophoto bands with DEM data

        df_merged = merge_data(df_allbands, orthophoto, source['dem_path'], debug="verbose")
        logging.info(f"df_merged columns before angle calculation: {df_merged.columns}")

        # Check required columns exist
        required_columns = ["Xw", "Yw", "band1", "band2", "band3"]
        if not all(col in df_merged.columns for col in required_columns):
            raise ValueError(f"Missing required columns in df_merged: {required_columns}")


        #Paert 3: Filter by polygon if specified
        if polygon_filtering:
            df_merged = filter_df_by_polygon(df_merged,polygon_path = source["Polygon_path"],
                                             plots_out= source["plot out"] ,target_crs="EPSG:32632",
                                             img_name= file)
            if type(df_merged) == NoneType:
                raise ValueError("No Points are inside the polygon, skipping this image.")

        #Part 4: Retrieve solar angles from position and time and filter
        sunelev, saa = extract_sun_angles(name, lon, lat, source["start date"], source["time zone"])

        # Calculate viewing angles
        df_merged = calculate_angles(df_merged, lon, lat, zcam, sunelev, saa)
        # Add file name column
        df_merged = df_merged.with_columns([pl.lit(file).alias("path")])
        # Filter black pixels
        len_before = len(df_merged)
        df_merged = df_merged.filter(pl.col("band1") != 0)
        logging.info(f"Black pixel filtering: {len_before} -> {len(df_merged)} | Percentage of points filtered: {(len_before-len(df_merged))/len_before * 100}%" )


        #PART 5: Save the merged data
        plotting_raster(df_merged, source["plot out"]+"/bands_data", file)

        # Save merged data as parquet
        save_parquet(df_merged, out, source, iteration, file)
        end_ortho = timer()
        logging.info(
            f"Finished processing orthophoto {file} for iteration {iteration} in {end_ortho - start_ortho:.2f} seconds")
    except Exception as e:
        logging.error(f"Error processing orthophoto {file}: {e}")
        logging.error(traceback.format_exc())


# ------------------------------
# Images management for dataframe creation
# ------------------------------
def build_database(tuple_chunk, source, exiftool_path):
    iteration = tuple_chunk[0]
    image = tuple_chunk[1]
    out = source['out']
    cam_path = source['cam_path']
    dem_path = source['dem_path']
    ori = source['ori']
    precision = source['precision']
    logging.info(f"Starting DEM processing for iteration {iteration}")
    start_DEM_i = timer()
    path_flat = retrieve_orthophoto_paths(ori)
    process_orthophoto(image, cam_path, path_flat, out, source, iteration, exiftool_path, precision,
                       polygon_filtering=True)

    end_DEM_i = timer()
    logging.info(f"Total time for iteration {iteration}: {end_DEM_i - start_DEM_i:.2f} seconds")


def main():
    config_objecture_logging()
    config = config_object("config_file.yaml")

    source= {'out': config.main_extract_out,
               'cam_path': config.main_extract_cam_path,
               'dem_path': config.main_extract_dem_path,
               'ori': config.main_extract_ori,
               'name': config.main_extract_name,
               'path_list_tag': config.main_extract_path_list_tag,
               'precision': config.precision,
               'Polygon_path': config.main_polygon_path,
               'start date': config.start_date,
               'time zone': config.time_zone,
               'plot out': config.plot_out}

    exiftool_path = r"exiftool"
    path_list = glob.glob(source['path_list_tag'])
    logging.info(f"Processing {len(path_list)} images sequentially")

    # Process each image directly without splitting into chunks
    for i, image_path in tqdm(enumerate(path_list)):
        logging.info(f"Processing image {i + 1}/{len(path_list)}: {os.path.basename(image_path)}")
        # Create a single-item list to maintain compatibility with build_database
        build_database((i, image_path), source, exiftool_path)


if __name__ == "__main__":
    main()
