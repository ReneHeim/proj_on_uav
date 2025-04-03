import os

import polars as pl
import logging
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.stats import gaussian_kde
from timeit import default_timer as timer


def merge_data(df_allbands, band_path, dem_path, debug="verbose"):
    """
    Merges the DEM and orthophoto (band) data by reprojecting the DEM onto the band grid
    and then sampling the DEM at each band pixel center. This assigns each band pixel the correct DEM value.

    Parameters:
        df_allbands: Polars DataFrame with orthophoto band data.
        band_path: Path to the orthophoto raster.
        dem_path: Path to the DEM raster.
        debug: Debug flag.

    Returns:
        df_merged: Polars DataFrame with band data and the assigned 'elev' column.
    """
    start_merge = timer()
    try:
        logging.info("Starting merge_data: Reprojecting DEM to band grid...")
        # Define a temporary file path for the reprojected DEM
        temp_dem_path = os.path.join(os.path.dirname(band_path), "temp_reprojected_dem.tif")

        # Reproject DEM to match the band image grid
        reproject_dem_to_band_grid(dem_path, band_path, temp_dem_path, resampling_method=Resampling.bilinear)
        logging.info(f"Reprojected DEM saved to {temp_dem_path}")

        logging.info("Sampling reprojected DEM at band pixel centers...")
        # Sample the reprojected DEM at band pixel centers
        dem_sampled = sample_dem_at_band_pixels(band_path, temp_dem_path)
        logging.info("Sampling complete.")

        # Add the sampled DEM values to the band DataFrame.
        # Since the reprojected DEM is forced to match the band grid, its flattened array
        # will have the same number of pixels as the DataFrame rows.
        df_merged = df_allbands.with_columns(pl.Series("elev", dem_sampled.flatten(), dtype=pl.Float32))

        if debug:
            merged_size = len(df_merged)
            logging.info(f"Reproject and sample merge: {merged_size} points merged.")

        # Optionally remove the temporary file
        if os.path.exists(temp_dem_path):
            os.remove(temp_dem_path)
            logging.info(f"Temporary reprojected DEM file {temp_dem_path} removed.")

        end_merge = timer()
        merge_time = end_merge - start_merge
        logging.info(f"Merged DEM and band data by sampling in {merge_time:.2f} seconds")
        logging.info(f"Processing speed: {df_merged.height / merge_time:.2f} points/second")
        return df_merged
    except Exception as e:
        logging.error(f"Error merging data: {e}")
        raise

def reproject_dem_to_band_grid(dem_path, band_path, output_dem_path, resampling_method=Resampling.bilinear):
    """
    Reprojects and resamples the DEM to the grid of the band image by forcing the output
    grid to match the band image's transform, width, and height exactly.

    Parameters:
        dem_path: Path to the input DEM raster.
        band_path: Path to the band (orthophoto) raster to use as reference.
        output_dem_path: Path where the reprojected DEM will be saved.
        resampling_method: Resampling method (default bilinear for smoother values).

    Returns:
        output_dem_path (str): Path to the reprojected DEM.
    """
    start_reproj = timer()
    logging.info("Starting DEM reprojection using band grid parameters...")

    # Use the band image's grid parameters directly.
    with rio.open(band_path) as band_src:
        band_crs = band_src.crs
        dst_transform = band_src.transform
        dst_width = band_src.width
        dst_height = band_src.height
        logging.info(f"Band image parameters: CRS: {band_crs}, width: {dst_width}, height: {dst_height}")

    with rio.open(dem_path) as dem_src:
        logging.info(f"DEM source CRS: {dem_src.crs}")
        dst_kwargs = dem_src.meta.copy()
        dst_kwargs.update({
            'crs': band_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height
        })

        with rio.open(output_dem_path, 'w', **dst_kwargs) as dst:
            for i in range(1, dem_src.count + 1):
                logging.info(f"Reprojecting DEM band {i}...")
                reproject(
                    source=rio.band(dem_src, i),
                    destination=rio.band(dst, i),
                    src_transform=dem_src.transform,
                    src_crs=dem_src.crs,
                    dst_transform=dst_transform,
                    dst_crs=band_crs,
                    resampling=resampling_method
                )
    end_reproj = timer()
    logging.info(f"DEM reprojection completed in {end_reproj - start_reproj:.2f} seconds.")
    return output_dem_path


def sample_dem_at_band_pixels(band_path, dem_reprojected_path):
    """
    Samples the reprojected DEM at the band image's pixel centers by reading the entire DEM array.
    If the DEM array shape doesn't match the band image shape, it is padded or cropped to match.

    Parameters:
        band_path: Path to the band (orthophoto) raster.
        dem_reprojected_path: Path to the DEM reprojected onto the band grid.

    Returns:
        dem_values (np.ndarray): An array of DEM values with the same shape as the band image.
    """
    start_sample = timer()
    logging.info("Starting DEM sampling by reading the full array...")

    # Read the band image to get its dimensions
    with rio.open(band_path) as band_src:
        band_array = band_src.read(1)
    band_shape = band_array.shape
    logging.info(f"Band image shape: {band_shape}")

    # Read the DEM reprojected file
    with rio.open(dem_reprojected_path) as dem_src:
        logging.info(f"Reprojected DEM dimensions: width={dem_src.width}, height={dem_src.height}")
        dem_values = dem_src.read(1)
    dem_shape = dem_values.shape
    logging.info(f"Reprojected DEM shape: {dem_shape}")

    # If there is a shape mismatch, pad or crop the DEM array to match the band image.
    if dem_shape != band_shape:
        logging.warning(f"Mismatch in shapes: DEM shape {dem_shape} vs Band shape {band_shape}. "
                        f"Attempting to adjust DEM array to match the band image.")
        new_dem = np.full(band_shape, 0, dtype=dem_values.dtype)  # fill with 0 (or np.nan)
        # Determine the overlapping region dimensions
        h = min(dem_shape[0], band_shape[0])
        w = min(dem_shape[1], band_shape[1])
        new_dem[:h, :w] = dem_values[:h, :w]
        dem_values = new_dem
        logging.info(f"Adjusted DEM shape: {dem_values.shape}")

    end_sample = timer()
    num_pixels = band_shape[0] * band_shape[1]
    sample_time = end_sample - start_sample
    logging.info(f"DEM sampling completed in {sample_time:.2f} seconds for {num_pixels} pixels "
                 f"({num_pixels / sample_time:.2f} pixels/second).")
    return dem_values


def visualize_coordinate_alignment(df_dem, df_allbands, precision, folder_name="Plots/coordinate_alignments"):
    """Visualize how well the coordinates align between datasets

    Args:
        df_dem: DEM DataFrame
        df_allbands: Orthophoto bands DataFrame
        precision: Coordinate precision
        folder_name: Name of the folder to save figures in (default: "coordinate_alignments")

    Returns:
        Dictionary with overlap statistics
    """
    import os
    import re
    import matplotlib.pyplot as plt

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Get the next available index by checking existing files
    existing_files = os.listdir(folder_name) if os.path.exists(folder_name) else []
    indices = []

    # Extract indices from existing filenames
    pattern = re.compile(r'coordinate_alignment_(\d+)\.png$')
    for file in existing_files:
        match = pattern.match(file)
        if match:
            indices.append(int(match.group(1)))
    # Determine next index (0 if no files exist, otherwise max+1)
    next_index = 0 if not indices else max(indices) + 1
    # Round coordinates for analysis
    df_dem = df_dem.with_columns([
        pl.col("Xw").round(precision),
        pl.col("Yw").round(precision)
    ])
    df_allbands = df_allbands.with_columns([
        pl.col("Xw").round(precision),
        pl.col("Yw").round(precision)
    ])

    # Get coordinate differences
    dem_coords = set(zip(df_dem["Xw"].to_list(), df_dem["Yw"].to_list()))
    bands_coords = set(zip(df_allbands["Xw"].to_list(), df_allbands["Yw"].to_list()))

    # Find the common points
    common_coords = dem_coords.intersection(bands_coords)

    # Calculate the percentage of overlap
    overlap_dem = 100 * len(common_coords) / len(dem_coords)
    overlap_bands = 100 * len(common_coords) / len(bands_coords)

    # Create a scatter plot of a sample of points from each dataset
    plt.figure(figsize=(12, 10))

    # Sample points for clearer visualization if datasets are large
    max_points = 5000
    dem_sample = list(dem_coords)[:max_points] if len(dem_coords) > max_points else dem_coords
    bands_sample = list(bands_coords)[:max_points] if len(bands_coords) > max_points else bands_coords

    # Plot points
    if dem_sample:
        x_dem, y_dem = zip(*dem_sample)
        plt.scatter(x_dem, y_dem, c='blue', alpha=0.5, s=3, label=f'DEM ({overlap_dem:.1f}% overlap)')

    if bands_sample:
        x_bands, y_bands = zip(*bands_sample)
        plt.scatter(x_bands, y_bands, c='red', alpha=0.5, s=3, label=f'Bands ({overlap_bands:.1f}% overlap)')

    plt.legend()
    plt.title('Coordinate Alignment Between DEM and Band Data')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Save the figure with auto-incremented index in the designated folder
    filename = os.path.join(folder_name, f'coordinate_alignment_{next_index}.png')
    plt.savefig(filename, dpi=200)

    logging.info(f"Coordinate alignment analysis saved to {filename} (index: {next_index})")
    logging.info(f"  - DEM points: {len(dem_coords):,}")
    logging.info(f"  - Band points: {len(bands_coords):,}")
    logging.info(f"  - Overlapping points: {len(common_coords):,}")
    logging.info(f"  - Overlap percentage: DEM {overlap_dem:.2f}%, Bands {overlap_bands:.2f}%")

    return {
        "dem_points": len(dem_coords),
        "band_points": len(bands_coords),
        "common_points": len(common_coords),
        "dem_overlap_pct": overlap_dem,
        "bands_overlap_pct": overlap_bands,
        "saved_index": next_index
    }

def analyze_kdtree_matching(df_dem, df_allbands, precision, max_distance=1.0):
    """Analyze potential matches using K-d tree nearest neighbor search"""
    from scipy.spatial import cKDTree
    import numpy as np

    # Round coordinates for consistent analysis
    df_dem = df_dem.with_columns([
        pl.col("Xw").round(precision),
        pl.col("Yw").round(precision)
    ])
    df_allbands = df_allbands.with_columns([
        pl.col("Xw").round(precision),
        pl.col("Yw").round(precision)
    ])

    # Extract coordinates as numpy arrays
    dem_coords = np.array(list(zip(df_dem["Xw"].to_list(), df_dem["Yw"].to_list())))
    bands_coords = np.array(list(zip(df_allbands["Xw"].to_list(), df_allbands["Yw"].to_list())))

    # Build K-d tree for band coordinates
    start_build = timer()
    tree = cKDTree(bands_coords)
    build_time = timer() - start_build

    # Find nearest neighbors for DEM points
    start_query = timer()
    distances, indices = tree.query(dem_coords, k=1, distance_upper_bound=max_distance)
    query_time = timer() - start_query

    # Count matches at different distance thresholds
    exact_matches = np.sum(distances == 0)
    near_matches = np.sum((distances > 0) & (distances <= max_distance))
    no_matches = np.sum(distances > max_distance)

    # Calculate percentages
    total_points = len(dem_coords)
    exact_pct = 100 * exact_matches / total_points
    near_pct = 100 * near_matches / total_points
    no_match_pct = 100 * no_matches / total_points

    logging.info(f"K-d Tree Nearest Neighbor Analysis:")
    logging.info(f"  - Tree build time: {build_time:.4f}s")
    logging.info(f"  - Query time: {query_time:.4f}s ({total_points / query_time:.2f} points/s)")
    logging.info(f"  - Exact matches (distance=0): {exact_matches:,} ({exact_pct:.2f}%)")
    logging.info(f"  - Near matches (0<d≤{max_distance}): {near_matches:,} ({near_pct:.2f}%)")
    logging.info(f"  - No matches (d>{max_distance}): {no_matches:,} ({no_match_pct:.2f}%)")

    return {
        "exact_matches": int(exact_matches),
        "near_matches": int(near_matches),
        "no_matches": int(no_matches),
        "exact_pct": exact_pct,
        "build_time": build_time,
        "query_time": query_time
    }