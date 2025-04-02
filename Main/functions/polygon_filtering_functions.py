import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import os
import signal
import pandas as pd
import concurrent.futures
from functools import partial
from shapely.geometry import Point
import geopandas as gpd
import polars as pl
from pyproj import CRS
from shapely.geometry.geo import box


def is_pos_inside_polygon(lat: float, lon: float, config: dict) -> bool:
    """
    Check if a geographical point is inside any polygon in a GeoDataFrame.

    Parameters:
    -----------
    lat : float
        Latitude of the point to check.
    lon : float
        Longitude of the point to check.
    config : dict
        Dictionary containing the path to the polygon file at key "Polygon_path".

    Returns:
    --------
    bool
        True if the point is inside any polygon, False otherwise.

    Raises:
    -------
    FileNotFoundError
        If the polygon file cannot be found.
    ValueError
        If the input coordinates are invalid or if there's an issue with coordinate transformation.
    """

    # Validate inputs
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise TypeError("Latitude and longitude must be numeric values")

    polygon_path = config.get("Polygon_path")
    if not polygon_path or not os.path.exists(polygon_path):
        raise FileNotFoundError(f"Polygon file not found at: {polygon_path}")

    try:
        # Read the polygon file
        gdf = gpd.read_file(polygon_path)

        # Create the point geometry based on CRS
        if str(gdf.crs) == "EPSG:4326":
            # If the polygons are in WGS84 (lat/lon), create a point with lon/lat order
            point = Point(lon, lat)
            logging.debug(f"Using WGS84 coordinates: Point({lon}, {lat})")
        else:
            # If the polygons are in a projected CRS, convert lat/lon to the same CRS
            point_gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
            point_gdf = point_gdf.to_crs(gdf.crs)
            point = point_gdf.geometry.iloc[0]
            logging.debug(f"Transformed point to {gdf.crs}: {point}")

        # Check if the point is inside any polygon
        inside = any(point.within(polygon) for polygon in gdf["geometry"])
        logging.info(f"Point inside Polygon: {str(inside)}")

        return inside

    except Exception as e:
        logging.error(f"Error checking if point is inside polygon: {str(e)}")
        raise


class TimeoutException(Exception):
    """Exception raised when a function times out."""
    pass


def setup_timeout(max_runtime_seconds):
    """Set up a timeout handler for Unix systems."""
    if os.name != 'nt':  # Not Windows
        def timeout_handler(signum, frame):
            raise TimeoutException("Function timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max_runtime_seconds)


def disable_timeout():
    """Disable the timeout alarm if on Unix."""
    if os.name != 'nt':
        signal.alarm(0)


def load_and_prepare_polygons(polygon_path, target_crs):
    """Load polygon file and handle CRS transformations."""
    phase_start = time.time()

    # Read the polygon file with optimized settings
    gdf_poly = gpd.read_file(polygon_path)

    if gdf_poly.crs is None:
        logging.warning(f"Polygon file has no defined CRS. Assuming {target_crs}")
        gdf_poly.set_crs(target_crs, inplace=True)
    else:
        original_crs = gdf_poly.crs
        logging.info(f"Original polygon CRS: {original_crs}")

        # Check if bounds suggest geographic coordinates
        bounds = gdf_poly.total_bounds
        is_likely_geographic = (
                abs(bounds[0]) <= 180 and
                abs(bounds[1]) <= 90 and
                abs(bounds[2]) <= 180 and
                abs(bounds[3]) <= 90
        )

        # Handle CRS conversion if needed
        if original_crs == CRS.from_epsg(4326):
            if not is_likely_geographic:
                logging.warning(
                    f"Polygon file claims to be WGS84 but coordinates don't look geographic. "
                    f"Bounds: {bounds}. Forcing CRS to {target_crs}"
                )
                gdf_poly.set_crs(target_crs, inplace=True, allow_override=True)
            else:
                logging.info(f"Converting polygons from {original_crs} to {target_crs}")
                gdf_poly = gdf_poly.to_crs(target_crs)
        elif original_crs != CRS.from_string(target_crs):
            logging.info(f"Converting polygons from {original_crs} to {target_crs}")
            gdf_poly = gdf_poly.to_crs(target_crs)

    # Process polygons - only keep valid ones
    gdf_poly = gdf_poly[gdf_poly.is_valid & ~gdf_poly.geometry.is_empty]

    phase_time = time.time() - phase_start
    logging.info(f"Loaded {len(gdf_poly)} polygons in {phase_time:.2f}s")

    return gdf_poly


def apply_polygon_shrinkage(gdf_poly, shrinkage):
    """Apply shrinkage to the polygons."""
    if shrinkage <= 0:
        return gdf_poly

    phase_start = time.time()

    # Pre-compute areas once
    gdf_poly['area'] = gdf_poly.geometry.area

    # Calculate buffer distance and store in a numpy array for vectorized operation
    buffer_distances = -1 * np.sqrt(gdf_poly['area'].values) * shrinkage

    # Apply buffer with parallel processing
    gdf_poly['geometry'] = [
        geom.buffer(dist) if dist > -area ** 0.4 else geom
        for geom, dist, area in zip(gdf_poly.geometry, buffer_distances, gdf_poly['area'])
    ]

    # Remove empty geometries
    valid_before = len(gdf_poly)
    gdf_poly = gdf_poly[~gdf_poly.geometry.is_empty]

    # Log shrinkage stats
    if len(gdf_poly) < valid_before:
        logging.warning(f"{valid_before - len(gdf_poly)} polygons became empty after shrinking")

    logging.info(f"Applied shrinkage factor {shrinkage} to polygons")

    phase_time = time.time() - phase_start
    logging.info(f"Processed polygons in {phase_time:.2f}s")

    return gdf_poly


def create_union_polygon(gdf_poly):
    """Create a union of all polygons for faster containment checks."""
    phase_start = time.time()

    union_poly = gdf_poly.geometry.unary_union
    logging.info(f"Union polygon bounds: {union_poly.bounds}")

    phase_time = time.time() - phase_start
    logging.info(f"Created union polygon in {phase_time:.2f}s")

    return union_poly


def check_data_polygon_overlap(df, polygons_gdf, debug=True, max_runtime_seconds=600):
    """
    Check if the data points and polygons have overlapping extents.

    Args:
        df: Polars DataFrame with point data
        polygons_gdf: GeoDataFrame containing polygons
        debug: Whether to enable debug output
        max_runtime_seconds: Maximum runtime for timeout warnings

    Returns:
        Tuple of (has_overlap, data_bounds)
    """
    phase_start = time.time()

    # Get data bounds with individual operations
    xmin = df.select(pl.min("Xw")).item()
    xmax = df.select(pl.max("Xw")).item()
    ymin = df.select(pl.min("Yw")).item()
    ymax = df.select(pl.max("Yw")).item()

    # Create bounds array
    data_bounds = [xmin, ymin, xmax, ymax]

    # Get the total bounds of all polygons
    poly_bounds = polygons_gdf.total_bounds  # [xmin, ymin, xmax, ymax]

    # Check for potential overlap of bounding boxes
    overlap_x = min(data_bounds[2], poly_bounds[2]) - max(data_bounds[0], poly_bounds[0])
    overlap_y = min(data_bounds[3], poly_bounds[3]) - max(data_bounds[1], poly_bounds[1])

    # If bounding boxes don't overlap, no point in checking individual polygons
    if overlap_x <= 0 or overlap_y <= 0:
        has_overlap = False
    else:
        # For more precision, we could check each polygon individually
        # This would be slower but more accurate for complex arrangements
        # In most cases, the bounding box check is sufficient
        has_overlap = True

        # Optional: Count how many polygons actually overlap with data bounds
        data_box = box(xmin, ymin, xmax, ymax)
        overlapping_polygons = sum(1 for geom in polygons_gdf.geometry if geom.intersects(data_box))

        if overlapping_polygons == 0:
            has_overlap = False
            logging.warning("Bounding boxes overlap, but no individual polygons intersect data extent")
        else:
            logging.info(f"Found {overlapping_polygons} polygons that may contain points")

    phase_time = time.time() - phase_start
    logging.info(f"Checked for overlap in {phase_time:.2f}s. Has overlap: {has_overlap}")

    return has_overlap, data_bounds

def plot_no_overlap(gdf_poly, data_bounds, polygon_basename, start_time, max_runtime_seconds,
                    plots_out=None, img_name = None):
    """Generate a plot showing why there is no overlap."""
    plt.figure(figsize=(10, 8))
    gdf_poly.plot(alpha=0.5, edgecolor='red')
    plt.plot([data_bounds[0], data_bounds[2], data_bounds[2], data_bounds[0], data_bounds[0]],
             [data_bounds[1], data_bounds[1], data_bounds[3], data_bounds[3], data_bounds[1]],
             'b--', label='Data bounds')
    plt.title("No Overlap Between Data and Polygons")
    plt.legend()

    if plots_out != None:
        plt.savefig(f'{plots_out}/polygon_filtering_data/no_overlap_{img_name}.png', dpi=200)
    else:
        plt.savefig(f'no_overlap_issue.png', dpi=200)
    if time.time() - start_time < max_runtime_seconds - 5:  # Only show if we have time
        plt.show()
    else:
        plt.close()


def process_chunk(chunk_indices, df, polygons_gdf, target_crs, id_field="id"):
    """
    Process a chunk of points to check which polygon they fall within.

    Args:
        chunk_indices: Tuple of (start_idx, end_idx) for the chunk
        df: The Polars dataframe containing all points
        polygons_gdf: GeoDataFrame containing polygons with ID field
        target_crs: Coordinate reference system
        id_field: Field in the GeoJSON/shapefile containing the plot ID

    Returns:
        GeoDataFrame with points and their assigned plot IDs
    """
    try:
        start_idx, end_idx = chunk_indices
        # Get chunk from the dataframe
        chunk = df.slice(start_idx, end_idx - start_idx)
        chunk_pd = chunk.to_pandas()

        # Create points GeoDataFrame
        points = [Point(x, y) for x, y in zip(chunk_pd['Xw'], chunk_pd['Yw'])]
        points_gdf = gpd.GeoDataFrame(chunk_pd, geometry=points, crs=target_crs)

        # Initialize plot_id column with None
        points_gdf['plot_id'] = None

        # For each polygon, find points within it and assign plot ID
        for idx, polygon in polygons_gdf.iterrows():
            # Points within this specific polygon
            mask = points_gdf.geometry.within(polygon.geometry)
            if mask.any():
                # Get the ID from the polygon
                plot_id = polygon[id_field] if id_field in polygon else f"plot_{idx}"
                # Assign the ID to all points within this polygon
                points_gdf.loc[mask, 'plot_id'] = plot_id

        # Return only points that got assigned a plot_id (i.e., are in any polygon)
        result = points_gdf[points_gdf['plot_id'].notna()]
        return result if not result.empty else None

    except Exception as e:
        logging.error(f"Error processing chunk {start_idx}-{end_idx}: {e}")
        return None

def prepare_chunks(df):
    """Prepare chunk boundaries for parallel processing."""
    # Calculate total chunks
    n_points = len(df)

    # Create CPU-optimized chunks
    available_cpus = max(1, os.cpu_count() - 1)  # Leave one CPU free

    # Choose points per chunk based on data size and CPU count
    points_per_chunk = min(1000000, max(10000, n_points // (available_cpus * 2)))
    n_chunks = (n_points + points_per_chunk - 1) // points_per_chunk

    # Determine worker count
    max_workers = min(available_cpus, n_chunks)

    # Prepare chunk indices
    chunk_indices = [(i * points_per_chunk, min((i + 1) * points_per_chunk, n_points))
                     for i in range(n_chunks)]

    logging.info(f"Prepared {n_chunks} chunks with ~{points_per_chunk:,} points each")
    logging.info(f"Will use {max_workers} workers for processing")

    return chunk_indices, max_workers, n_points, n_chunks


def process_chunks_parallel(df, chunk_indices, max_workers, polygons_gdf, target_crs, id_field,
                            n_chunks, start_time, max_runtime_seconds):
    """Process chunks in parallel using ThreadPoolExecutor."""
    phase_start = time.time()

    # Use a partial function with fixed arguments
    process_func = partial(process_chunk,
                           df=df,
                           polygons_gdf=polygons_gdf,
                           target_crs=target_crs,
                           id_field=id_field)

    filtered_chunks = []
    processed_count = 0

    # Process chunks in parallel with timeout awareness
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chunk = {executor.submit(process_func, chunk_idx): i
                           for i, chunk_idx in enumerate(chunk_indices)}

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            # Check if we're approaching timeout
            if time.time() - start_time > max_runtime_seconds * 0.9:
                logging.warning(
                    f"Approaching timeout. Processed {processed_count}/{n_chunks} chunks. Cancelling remaining tasks.")
                # Cancel pending futures
                for f in future_to_chunk:
                    f.cancel()
                break

            chunk_idx = future_to_chunk[future]
            processed_count += 1

            try:
                result = future.result()
                if result is not None:
                    filtered_chunks.append(result)
            except Exception as e:
                logging.error(f"Chunk {chunk_idx} processing failed: {e}")

            # Log progress periodically
            if processed_count % max(1, n_chunks // 10) == 0:
                logging.info(
                    f"Processed {processed_count}/{n_chunks} chunks ({(processed_count / n_chunks * 100):.1f}%)")

    return filtered_chunks, processed_count, time.time() - phase_start


def combine_chunk_results(filtered_chunks, n_points, phase_time):
    """Combine processed chunks into a single DataFrame."""
    if not filtered_chunks:
        return None

    try:
        gdf_filtered = pd.concat(filtered_chunks, ignore_index=True)
        points_after = len(gdf_filtered)
        points_filtered = n_points - points_after
        percentage_filtered = (points_filtered / n_points * 100) if n_points > 0 else 0

        logging.info(f"Points inside polygons: {points_after:,} ({100 - percentage_filtered:.2f}%)")
        logging.info(f"Points filtered out: {points_filtered:,} ({percentage_filtered:.2f}%)")
        logging.info(f"Filtered points in {phase_time:.2f}s ({points_after / phase_time:.2f} points/s)")

        # Clean up and convert back to Polars
        gdf_filtered = gdf_filtered.drop(columns=["geometry"])
        return pl.from_pandas(gdf_filtered)
    except Exception as e:
        logging.error(f"Error combining filtered results: {e}")
        return None


def plot_results(gdf_poly, gdf_filtered, target_crs, polygon_basename, sample_for_debug=5000, plots_out=None, img_name = None):
    """Generate a visualization of the filtered points within polygons."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot polygons
    gdf_poly.plot(ax=ax, alpha=0.3, edgecolor='black')

    points_after = len(gdf_filtered)

    # Sample points for visualization
    if points_after > sample_for_debug:
        # Convert to pandas for easy sampling
        sample_df = gdf_filtered.sample(sample_for_debug)
        sample_points = gpd.GeoDataFrame(
            sample_df,
            geometry=[Point(x, y) for x, y in zip(sample_df['Xw'], sample_df['Yw'])],
            crs=target_crs
        )
        sample_points.plot(ax=ax, markersize=2, color='green', alpha=0.5, label='Points Inside')
        title = f"Points Inside Polygons (Sample of {sample_for_debug:,} from {points_after:,})"
    else:
        # Plot all points if few enough
        inside_points = gpd.GeoDataFrame(
            gdf_filtered,
            geometry=[Point(x, y) for x, y in zip(gdf_filtered['Xw'], gdf_filtered['Yw'])],
            crs=target_crs
        )
        inside_points.plot(ax=ax, markersize=2, color='green', alpha=0.5, label='Points Inside')
        title = f"All {points_after:,} Points Inside Polygons"

    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    # Save the plot
    plt.tight_layout()
    print(gdf_filtered)
    if plots_out != None:
        os.makedirs(f"{plots_out}/polygon_filtering_data", exist_ok=True)
        plt.savefig(f'{plots_out}/polygon_filtering_data/polygon_filtering_{img_name}.png', dpi=200)
    else:
        plt.savefig(f'polygon_filtering_{polygon_basename}.png', dpi=200)
    plt.close()
    plt.show()


def filter_df_by_polygon(df, polygon_path, target_crs="EPSG:32632", id_field="id",
                         shrinkage=0.1, debug=True, max_runtime_seconds=600,
                         sample_for_debug=5000, plots_out=None, img_name = None):
    """
    Filter a Polars DataFrame to only include points that fall within polygons.
    Implements timeouts and performance optimizations.

    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing points with 'Xw' and 'Yw' columns
    polygon_path : str
        Path to the polygon file (e.g., shapefile, GeoJSON)
    target_crs : str, default="EPSG:32632"
        Target coordinate reference system
    shrinkage : float, default=0.1
        Shrinkage factor for polygons (0 = no shrinkage, 1 = maximum shrinkage)
    debug : bool, default=True
        Whether to generate debug plots and additional logging
    max_runtime_seconds : int, default=600
        Maximum time in seconds to let the function run before returning original df
    sample_for_debug : int, default=5000
        Maximum number of points to use for debug visualizations

    Returns:
    --------
    polars.DataFrame
        Filtered DataFrame containing only points inside the polygon
    """
    start_time = time.time()
    points_before = len(df)
    polygon_basename = os.path.basename(polygon_path)

    # Set up timeout handler
    try:
        setup_timeout(max_runtime_seconds)
        logging.info(f"Points before filtering: {points_before:,}")

        # --- PHASE 1: Load and prepare polygons ---
        # Load and prepare polygons
        polygons_gdf = load_and_prepare_polygons(polygon_path, target_crs)

        if polygons_gdf.empty:
            logging.error(f"No valid polygons in {polygon_path}")
            return df

        # Apply polygon shrinkage if needed
        polygons_gdf = apply_polygon_shrinkage(polygons_gdf, shrinkage)

        if polygons_gdf.empty:
            logging.warning(f"All polygons empty after shrinkage. Returning original data.")
            return df

        # Check for overlap between data and polygons
        has_overlap, data_bounds = check_data_polygon_overlap(df, polygons_gdf, debug, max_runtime_seconds)

        if not has_overlap:
            logging.warning("No overlap between data and polygons. Returning original data.")
            if debug:
                plot_no_overlap(polygons_gdf, data_bounds, polygon_basename, start_time, max_runtime_seconds, img_name=img_name, plots_out=plots_out)
            return df


        # Check if we've been running too long already
        if time.time() - start_time > max_runtime_seconds * 0.6:
            logging.warning(f"Already used > 60% of max runtime ({max_runtime_seconds}s). Returning original data.")
            return df

        # --- PHASE 5: Prepare chunks for parallfel processing ---
        chunk_indices, max_workers, n_points, n_chunks = prepare_chunks(df)

        # --- PHASE 6: Process chunks in parallel ---
        filtered_chunks, processed_count, phase_time = process_chunks_parallel(
            df, chunk_indices, max_workers, polygons_gdf, target_crs,
            id_field,  # This is the default id_field, you can make it a parameter
            n_chunks, start_time, max_runtime_seconds
        )

        # Check if we processed anything
        if processed_count == 0:
            logging.warning("No chunks were processed successfully. Returning original data.")
            return df

        # Check if we need to return early due to timeout
        if time.time() - start_time > max_runtime_seconds * 0.95:
            if filtered_chunks:
                logging.warning(f"Timeout approaching. Returning partial results from {len(filtered_chunks)} chunks.")
                try:
                    partial_result = pd.concat(filtered_chunks, ignore_index=True)
                    return pl.from_pandas(partial_result.drop(columns=["geometry"]))
                except Exception as e:
                    logging.error(f"Error creating partial result: {e}")
                    return df
            else:
                logging.warning("Timeout approaching with no filtered results. Returning original data.")
                return df

        # --- PHASE 7: Combine results ---
        if not filtered_chunks:
            logging.warning("No points found inside polygons. Returning original data.")
            return df

        result_df = combine_chunk_results(filtered_chunks, n_points, phase_time)
        if result_df is None or len(result_df) == 0:
            logging.warning("No points after filtering. Returning original data.")
            return df


        print(df)

        # --- PHASE 8: Debug visualization ---
        if debug and time.time() - start_time < max_runtime_seconds * 0.9:
            try:
                gdf_filtered = pd.concat(filtered_chunks, ignore_index=True)
                plot_results(polygons_gdf, gdf_filtered, target_crs, polygon_basename, sample_for_debug, plots_out, img_name )
            except Exception as e:
                logging.error(f"Error generating debug plot: {e}")
            return result_df


    except TimeoutException:
        logging.warning(f"Function timed out after {max_runtime_seconds} seconds. Returning original data.")
        return df

    except Exception as e:
        logging.error(f"Error in polygon filtering: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return df

    finally:
        disable_timeout()