#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os
import re
import traceback
from datetime import datetime
from pathlib import Path, PureWindowsPath
from timeit import default_timer as timer

import numpy as np
import polars as pl
import pysolar as solar
from rasterio.enums import Resampling
from tqdm import tqdm

from src.Utils.extract_data.camera import calculate_angles, get_camera_position, plot_angles
from src.Common.config_object import config_object
from src.Utils.extract_data.date_time import convert_to_timezone
from src.Utils.extract_data.merge_analysis import merge_data
from src.Utils.extract_data.polygon_filtering import filter_df_by_polygon
from src.Utils.extract_data.raster import (
    check_alignment,
    coregister_and_resample,
    plotting_raster,
    read_orthophoto_bands,
)
from src.Common.logging import logging_config


# ------------------------------
# Save Output to Parquet
# ------------------------------
def save_parquet(df, out, source, iteration, file):
    start_write = timer()
    output_path = Path(out) / f"{source['name']}_{iteration}_{file}.parquet"
    last_err = None
    for comp in ("zstd", "snappy", None):
        try:
            if comp is None:
                df.write_parquet(str(output_path))
            else:
                df.write_parquet(str(output_path), compression=comp)
            end_write = timer()
            logging.info(
                f"Saved image result for {file} in {end_write - start_write:.2f} seconds (compression={comp})"
            )
            return
        except Exception as e:
            last_err = e
            logging.warning(f"Parquet write failed with compression={comp}: {e}")
            continue
    logging.error(f"Error saving parquet for {file}: {last_err}")
    raise last_err


# ------------------------------
# Check Images done
# ------------------------------
def check_already_processed(out_dir):
    processed = set()  # use a set for O(1) look-ups
    for p in Path(out_dir).glob("*.parquet"):
        m = re.match(r".*IMG_(\d+)_\d+\.tif\.parquet$", p.name)
        if m:  # accept only valid filenames
            processed.add(int(m.group(1)))  # capture the image number
    return processed


def remove_images_already_processed(inp_dir, out_dir):
    processed = check_already_processed(out_dir)
    inp_dir = inp_dir.replace("*.tif", "")
    imgs = list(Path(inp_dir).glob("*.tif"))
    nums = [
        int(re.match(r".*IMG_(\d+)_\d+\.tif$", p.name).group(1))
        for p in imgs
        if re.match(r".*IMG_(\d+)_\d+\.tif$", p.name)
    ]
    to_drop = [i for i, n in enumerate(nums) if n in processed]
    for i in sorted(to_drop, reverse=True):  # delete back-to-front
        del imgs[i]

    logging.info(
        f"Images To Process: {len(imgs)}, Images Already Processed:{len(processed)}, Total number of images: {len(glob.glob(inp_dir + "*.tif"))}"
    )
    return imgs  # remaining *.tif* paths


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
            f"Calculated sun angles for {name} using pysolar: elevation={sunelev:.2f}°, azimuth={saa:.2f}° in {end - start:.2f} seconds"
        )
    except Exception as e:
        logging.warning(
            f"Sun angle calculation with pysolar failed for {name}, skipping. Error: {e}"
        )

    return sunelev, saa


# ------------------------------
# Core Processing Function for an Orthophoto
# ------------------------------
def process_orthophoto(
    orthophoto,
    cam_path,
    path_flat,
    out,
    source,
    iteration,
    exiftool_path,
    polygon_filtering=False,
    alignment=False,
):
    try:
        # PART 1: Read the Raster
        start_ortho = timer()
        path, file = os.path.split(orthophoto)
        name, _ = os.path.splitext(file)

        logging.info(f"Processing orthophoto {file} for iteration {iteration}")

        # Get camera position from the camera file
        lon, lat, zcam = get_camera_position(cam_path, name, source["target_crs"])

        # Optional: Ensure DEM and orthophoto are aligned
        if alignment:
            if not check_alignment(source["dem_path"], orthophoto):
                coreg_path = os.path.join(out, f"coreg_{file}")
                orthophoto = coregister_and_resample(
                    orthophoto,
                    source["dem_path"],
                    coreg_path,
                    target_resolution=None,
                    resampling=Resampling.bilinear,
                )
                if not check_alignment(source["dem_path"], orthophoto):
                    raise ValueError(
                        "Co-registration failed: orthophoto and DEM are still not aligned."
                    )

        # Read orthophoto bands (assumed to be transformed already if needed)
        df_allbands = read_orthophoto_bands(orthophoto)

        # PART 2: Merge orthophoto bands with DEM data

        df_merged = merge_data(df_allbands, orthophoto, source["dem_path"], debug="verbose")
        logging.info(f"df_merged columns before angle calculation: {df_merged.columns}")

        # Check required columns exist
        required_columns = ["Xw", "Yw", "elev"] + [
            f"band{x}" for x in range(1, source["bands"] + 1)
        ]
        if not all(col in df_merged.columns for col in required_columns):
            raise ValueError(f"Missing required columns in df_merged: {required_columns}")

        # Paert 3: Filter by polygon if specified
        if polygon_filtering:
            df_merged = filter_df_by_polygon(
                df_merged,
                polygon_path=source["Polygon_path"],
                plots_out=source["plot out"],
                target_crs=source["target_crs"],
                img_name=file,
            )
            if df_merged is None:
                raise ValueError("No Points are inside the polygon, skipping this image.")

        # Part 4: Retrieve solar angles from position and time and filter
        sunelev, saa = extract_sun_angles(name, lon, lat, source["start date"], source["time zone"])

        # Calculate viewing angles
        df_merged = calculate_angles(df_merged, lon, lat, zcam, sunelev, saa)
        # Add file name column
        df_merged = df_merged.with_columns([pl.lit(file).alias("path")])
        # Filter black pixels
        len_before = len(df_merged)
        for column in df_merged.columns:
            if "band" in column:
                df_merged = df_merged.with_columns([pl.col(column).replace(0, np.nan)])

        logging.info(
            f"Black pixel filtering: {len_before} -> {len(df_merged)} | Percentage of points filtered: {(len_before-len(df_merged))/len_before * 100}%"
        )

        # PART 5: plot and save the merged data

        plotting_raster(df_merged, source["plot out"] / "bands_data", file,ny=380,
            nx=630,band_kde=False, auto_figsize=False)

        plot_angles(df_merged, lon, lat, zcam, source["plot out"] / "angles_data", file)

        # Save merged data as parquet
        save_parquet(df_merged, out, source, iteration, file)
        end_ortho = timer()
        logging.info(
            f"Finished processing orthophoto {file} for iteration {iteration} in {end_ortho - start_ortho:.2f} seconds"
        )
    except Exception as e:
        logging.error(f"Error processing orthophoto {file}: {e}")
        logging.error(traceback.format_exc())


def main():
    logging_config()

    parser = argparse.ArgumentParser(
        description="Extract per-pixel data and angles from orthophotos"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config_file_example.yml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--alignment",
        action="store_true",
        help="Enable DEM/orthophoto co-registration if misaligned",
    )
    parser.add_argument(
        "--no-polygon",
        action="store_true",
        help="Disable polygon filtering",
    )
    args = parser.parse_args()

    config = config_object(args.config)

    source = {
        "out": config.main_extract_out,
        "cam_path": config.main_extract_cam_path,
        "dem_path": config.main_extract_dem_path,
        "ori": config.main_extract_ori,
        "name": config.main_extract_name,
        "path_list_tag": config.main_extract_path_list_tag,
        "bands": config.bands,
        "Polygon_path": config.main_polygon_path,
        "start date": config.start_date,
        "time zone": config.time_zone,
        "plot out": config.plot_out,
        "target_crs": config.target_crs,
    }

    exiftool_path = r"exiftool"
    try:
        print(source["path_list_tag"])
        path_list = remove_images_already_processed(source["path_list_tag"], source["out"])
    except Exception as e:
        logging.error(f"Error Loading Last checkpoint: {e}, e")
        path_list = glob.glob(source["path_list_tag"])

    logging.info(f"Processing {len(path_list)} images sequentially")

    # Process each image directly without splitting into chunks
    for i, image_path in tqdm(enumerate(path_list)):
        logging.info(f"Processing image {i + 1}/{len(path_list)}: {os.path.basename(image_path)}")
        # Create a single-item list to maintain compatibility with build_database
        ori = source["ori"]
        logging.info(f"Starting DEM processing for iteration {i}")

        start_DEM_i = timer()
        path_flat = retrieve_orthophoto_paths(ori)
        process_orthophoto(
            image_path,
            source["cam_path"],
            path_flat,
            source["out"],
            source,
            i,
            exiftool_path,
            polygon_filtering=not args.no_polygon,
            alignment=args.alignment,
        )
        end_DEM_i = timer()
        logging.info(f"Total time for iteration {i}: {end_DEM_i - start_DEM_i:.2f} seconds")


if __name__ == "__main__":
    main()
