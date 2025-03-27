#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
import pytz
import glob
import os
import logging
import rasterio as rio
from timeit import default_timer as timer
from pathlib import PureWindowsPath, Path
import traceback
import re
from pyproj import Transformer
from rasterio.warp import reproject, Resampling, calculate_default_transform
from tqdm import tqdm

from Main.functions.date_time_functions import convert_to_timezone
from Main.functions.raster_functions import *  # Your helper functions, e.g., xyval, latlon_to_utm32n_series, etc.
from config_object import config
import geopandas as gpd
from shapely.geometry import Point
import polars as pl
import pysolar as solar

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("process.log"),
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
            b_all = rst.read()  # Read all bands
            print(max(b_all[0][0]))
            if b_all.dtype == np.float64:
                b_all = b_all.astype(np.float32)
            rows, cols = np.indices((rst.height, rst.width))
            rows_flat, cols_flat = rows.flatten(), cols.flatten()
            Xw, Yw = rio.transform.xy(rst.transform, rows_flat, cols_flat, offset='center')
            Xw = np.array(Xw, dtype=np.float32)
            Yw = np.array(Yw, dtype=np.float32)

            # Transform coordinates if needed to match DEM (e.g., UTM)
            if transform_to_utm:
                transformer = Transformer.from_crs(rst.crs, target_crs, always_xy=True)
                x_trans, y_trans = transformer.transform(Xw, Yw)
                Xw = np.array(x_trans, dtype=np.float32).round(precision)
                Yw = np.array(y_trans, dtype=np.float32).round(precision)
            else:
                Xw = np.array(Xw, dtype=np.float32).round(precision)
                Yw = np.array(Yw, dtype=np.float32).round(precision)

            band_values = b_all.reshape(num_bands, -1).T


            data = {
                'Xw': pl.Series(Xw, dtype=pl.Float32),
                'Yw': pl.Series(Yw, dtype=pl.Float32)
            }
            for idx in range(num_bands):
                data[f'band{idx + 1}'] = pl.Series(band_values[:, idx], dtype=pl.Float32)
            df_allbands = pl.DataFrame(data)
            df_allbands = df_allbands.with_columns([
                pl.col("Xw").round(precision),
                pl.col("Yw").round(precision)
            ]).unique()
        end_bands = timer()
        logging.info(
            f"Processed orthophoto bands for {os.path.basename(each_ortho)} in {end_bands - start_bands:.2f} seconds")
        return df_allbands
    except Exception as e:
        logging.error(f"Error reading bands from {each_ortho}: {e}")
        raise


# ------------------------------
# Merge DEM and Orthophoto Data on Coordinates
# ------------------------------
def merge_data(df_dem, df_allbands, precision):
    start_merge = timer()
    try:
        df_dem = df_dem.group_by(["Xw", "Yw"]).agg([
            pl.col("elev").mean().alias("elev")
        ])
        df_dem = df_dem.with_columns([
            pl.col("Xw").round(precision),
            pl.col("Yw").round(precision)
        ]).unique()
        df_allbands = df_allbands.with_columns([
            pl.col("Xw").round(precision),
            pl.col("Yw").round(precision)
        ]).unique()
        df_dem_lazy = df_dem.lazy()
        df_allbands_lazy = df_allbands.lazy()
        df_merged = df_dem_lazy.join(df_allbands_lazy, on=["Xw", "Yw"], how="inner").collect()
        end_merge = timer()
        logging.info(f"Merged DEM and band data in {end_merge - start_merge:.2f} seconds")
        return df_merged
    except Exception as e:
        logging.error(f"Error merging data: {e}")
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
        print(output_path)
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
        logging.info(f"Calculated sun angles for {name} using pysolar: elevation={sunelev:.2f}°, azimuth={saa:.2f}° in {end - start:.2f} seconds")
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


def filter_df_by_polygon(df, polygon_path, target_crs="EPSG:32632", precision=2):
    # Read the polygon file
    gdf_poly = gpd.read_file(polygon_path)
    points_before = len(df)

    # *** IMPORTANT: Correct the CRS if needed ***
    # If the coordinate values suggest a projected system (e.g., UTM) but the file is tagged as EPSG:4326,
    # reassign the correct CRS:
    if gdf_poly.crs.to_string() == "EPSG:4326":
        # For example, if you know they should be UTM Zone 32N:
        gdf_poly.crs = "EPSG:32632"

    # Filter valid, non-empty geometries
    valid_gdf = gdf_poly[gdf_poly.is_valid & (~gdf_poly.is_empty)]
    if valid_gdf.empty:
        raise ValueError("No valid polygons found in the file.")

    # Convert to target CRS if needed
    if valid_gdf.crs != target_crs:
        valid_gdf = valid_gdf.to_crs(target_crs)

    # Union all valid polygons into one geometry
    # Use union_all() if available, else unary_union:
    try:
        union_poly = valid_gdf.union_all()  # Newer API
    except AttributeError:
        union_poly = valid_gdf.unary_union

    print("Union polygon bounds:", union_poly.bounds)

    # Convert the Polars DataFrame to Pandas
    df_pd = df.to_pandas()

    # Create shapely Points for each row
    df_pd["geometry"] = df_pd.apply(lambda row: Point(row["Xw"], row["Yw"]), axis=1)
    # Filter rows where the point is within the union polygon
    gdf_points = gpd.GeoDataFrame(df_pd, geometry="geometry", crs=target_crs)
    gdf_filtered = gdf_points[gdf_points["geometry"].within(union_poly)].copy()
    points_after = len(df_pd)


    logging.info("Points filtered:" + str(points_after - points_before))
    # Drop the geometry column and convert back to Polars DataFrame
    gdf_filtered = gdf_filtered.drop(columns=["geometry"])
    return pl.from_pandas(gdf_filtered)


# ------------------------------
# Core Processing Function for an Orthophoto
# ------------------------------
def process_orthophoto(each_ortho, cam_path, path_flat, out, source, iteration, exiftool_path, precision,
                       polygon_filtering=False, alignment=False):
    try:
        start_ortho = timer()
        path, file = os.path.split(each_ortho)
        name, _ = os.path.splitext(file)
        logging.info(f"Processing orthophoto {file} for iteration {iteration}")

        dem_path = source['dem_path']
        # Get camera position from the camera file
        lon, lat, zcam = get_camera_position(cam_path, name)
        print(f"longitude: {lon}, latitude: {lat}, zcam: {zcam}")

        # Optional: Check if the image is within a polygon
        if polygon_filtering:
            easting, northing = latlon_to_utm32n_series(lat, lon)
            point = Point(easting, northing)
            gdf = gpd.read_file(source["Polygon_path"])
            inside = any(point.within(polygon) for polygon in gdf["geometry"])
            logging.info(f"Point inside Polygon: {str(inside)}")
            if not inside:
                raise ValueError(f"The Image {file} is not inside any polygon")

        # Optional: Ensure DEM and orthophoto are aligned
        if alignment:
            if not check_alignment(dem_path, each_ortho):
                coreg_path = os.path.join(out, f"coreg_{file}")
                each_ortho = coregister_and_resample(each_ortho, dem_path, coreg_path, target_resolution=None,
                                                     resampling=Resampling.bilinear)
                if not check_alignment(dem_path, each_ortho):
                    raise ValueError("Co-registration failed: orthophoto and DEM are still not aligned.")

        # Read DEM with transformation so that coordinates match the polygon system (UTM)
        df_dem = read_dem(dem_path, precision, transform_to_utm=True, target_crs="EPSG:32632")

        # Read orthophoto bands (assumed to be transformed already if needed)
        df_allbands = read_orthophoto_bands(each_ortho, precision)


        # Merge DEM and orthophoto data on (Xw, Yw)
        df_merged = merge_data(df_dem, df_allbands, precision)
        logging.info(f"df_merged columns before angle calculation: {df_merged.columns}")

        # Check required columns exist
        required_columns = ["Xw", "Yw", "band1", "band2", "band3"]
        if not all(col in df_merged.columns for col in required_columns):
            raise ValueError(f"Missing required columns in df_merged: {required_columns}")


        if polygon_filtering:
         polygon_path = source["Polygon_path"]  # path to your polygon file
         df_merged = filter_df_by_polygon(df_merged, polygon_path, target_crs="EPSG:32632", precision=2)

        # Retrieve solar angles from position and time
        sunelev, saa = extract_sun_angles(name, lon, lat, source["start date"], source["time zone"])

        # Calculate viewing angles
        df_merged = calculate_angles(df_merged, lon, lat, zcam, sunelev, saa)
        # Add file name column
        df_merged = df_merged.with_columns([pl.lit(file).alias("path")])




        # Filter merged DataFrame by polygon
        # After merging, filter rows outside the polygon:


        # For debugging: convert to pandas and plot distributions
        import matplotlib.pyplot as plt
        df_pd = df_merged.to_pandas()
        print(df_pd.describe())
        plt.figure(figsize=(8, 6))
        plt.hist(df_pd['elev'], bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Elevation')
        plt.ylabel('Frequency')
        plt.title('Elevation Distribution')
        plt.show()
        for band in [col for col in df_pd.columns if col.startswith('band')]:
            plt.figure(figsize=(8, 6))
            plt.hist(df_pd[band], bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel(f'{band} Values')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of {band}')
            plt.show()

        # Save merged data as parquet
        save_parquet(df_merged, out, source, iteration, file)
        end_ortho = timer()
        logging.info(
            f"Finished processing orthophoto {file} for iteration {iteration} in {end_ortho - start_ortho:.2f} seconds")
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
    path_flat = retrieve_orthophoto_paths(ori)
    for each_ortho in tqdm(images, desc=f"Processing iteration {iteration}"):
        process_orthophoto(each_ortho, cam_path, path_flat, out, source, iteration, exiftool_path, precision,
                           polygon_filtering=False)
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
            "precision": config.precision,
            "Polygon_path": config.main_polygon_path,
            "start date": config.start_date,
            "time zone": config.time_zone,
        }
    ]
    exiftool_path = r"exiftool"
    for source in sources:
        path_list = glob.glob(source['path_list_tag'])
        logging.info(f"Processing {len(path_list)} images sequentially")

        # Process each image directly without splitting into chunks
        for i, image_path in enumerate(path_list):
            logging.info(f"Processing image {i+1}/{len(path_list)}: {os.path.basename(image_path)}")
            # Create a single-item list to maintain compatibility with build_database
            single_image = [image_path]
            # Process directly with the current iteration number
            build_database((i, single_image), source, exiftool_path)

if __name__ == "__main__":
    main()
