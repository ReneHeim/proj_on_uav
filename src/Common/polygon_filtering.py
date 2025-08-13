import concurrent.futures
import logging
import os
import signal
import time
from functools import partial

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from pyproj import CRS
from shapely.geometry import Point
from shapely.geometry.geo import box


# Not used
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
    if os.name != "nt":  # Not Windows

        def timeout_handler(signum, frame):
            raise TimeoutException("Function timed out")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max_runtime_seconds)


def disable_timeout():
    """Disable the timeout alarm if on Unix."""
    if os.name != "nt":
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
            abs(bounds[0]) <= 180
            and abs(bounds[1]) <= 90
            and abs(bounds[2]) <= 180
            and abs(bounds[3]) <= 90
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
    gdf_poly["area"] = gdf_poly.geometry.area

    # Calculate buffer distance and store in a numpy array for vectorized operation
    buffer_distances = -1 * np.sqrt(gdf_poly["area"].values) * shrinkage

    # Apply buffer with parallel processing | Exclude small areas to avoid too muh shrinkage
    gdf_poly["geometry"] = [
        geom.buffer(dist) if dist > -(area**0.4) else geom
        for geom, dist, area in zip(gdf_poly.geometry, buffer_distances, gdf_poly["area"])
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


def check_data_polygon_overlap(df, polygons_gdf):
    """
    Check if the data points and polygons have overlapping extents.

    Args:
        df: Polars DataFrame with point data
        polygons_gdf: GeoDataFrame containing polygons
    Returns:
        Tuple of (has_overlap, data_bounds)
    """
    phase_start = time.time()

    # Get data bounds with individual operations
    xmin = df.select(pl.col("Xw").min()).item()
    xmax = df.select(pl.col("Xw").max()).item()
    ymin = df.select(pl.col("Yw").min()).item()
    ymax = df.select(pl.col("Yw").max()).item()

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
            logging.warning(
                "Bounding boxes overlap, but no individual polygons intersect data extent"
            )
        else:
            logging.info(f"Found {overlapping_polygons} polygons that may contain points")

    phase_time = time.time() - phase_start
    logging.info(f"Checked for overlap in {phase_time:.2f}s. Has overlap: {has_overlap}")

    return has_overlap, data_bounds


def plot_no_overlap(gdf_poly, data_bounds, plots_out=None, img_name=None, debug=False):
    """
    Generate a plot showing why there is no overlap between data and polygons.
    also shows the approximate closest distance between data and polygons.

    Args:
        gdf_poly: GeoDataFrame containing polygons
        data_bounds: List of [xmin, ymin, xmax, ymax] for data extent
        polygon_basename: Base name for the plot file
        start_time: Start time of the process
        plots_out: Output directory for plots
        img_name: Optional name for the image file
    """
    # Create figure with two subplots - main plot and inset for context
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Get total bounds of polygons
    poly_bounds = gdf_poly.total_bounds  # [xmin, ymin, xmax, ymax]

    # Create data bounding box as polygon for better visualization
    data_box = box(data_bounds[0], data_bounds[1], data_bounds[2], data_bounds[3])
    data_box_gdf = gpd.GeoDataFrame(geometry=[data_box], crs=gdf_poly.crs)

    # Plot polygons
    gdf_poly.plot(ax=ax, alpha=0.5, edgecolor="red", facecolor="lightcoral", label="Polygon Areas")

    # Plot data bounds with distinct style
    data_box_gdf.plot(
        ax=ax, facecolor="none", edgecolor="blue", linestyle="--", linewidth=2, label="Data Bounds"
    )

    # Add polygon IDs as text on each polygon
    for idx, poly in gdf_poly.iterrows():
        # Get the centroid of the polygon for label placement
        centroid = poly.geometry.centroid

        # Determine the polygon ID to display
        if "id" in poly:
            poly_id = poly["id"]
        elif "plot_id" in poly:
            poly_id = poly["plot_id"]
        else:
            poly_id = f"Plot {idx}"

        # Add text annotation with the polygon ID
        ax.text(
            centroid.x,
            centroid.y,
            poly_id,
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3"),
        )

    # Calculate minimum distance between polygons and data bounds
    min_distance = float("inf")
    closest_poly_idx = None

    for idx, poly in gdf_poly.iterrows():
        distance = poly.geometry.distance(data_box)
        if distance < min_distance:
            min_distance = distance
            closest_poly_idx = idx

    # Create a line connecting closest points if we found a minimum distance
    if closest_poly_idx is not None and min_distance < float("inf"):
        # Get the closest polygon
        closest_poly = gdf_poly.iloc[closest_poly_idx].geometry

        # Get the closest points between data_box and closest_poly
        # This is approximate - shapely doesn't have a direct "closest points" function
        # We'll use interpolation on the boundary to find points

        # Simple approximation: connect centroids with a line
        data_centroid = data_box.centroid
        poly_centroid = closest_poly.centroid

        # Draw connection line
        ax.plot(
            [data_centroid.x, poly_centroid.x],
            [data_centroid.y, poly_centroid.y],
            "k-",
            alpha=0.6,
            linewidth=1.5,
            linestyle=":",
        )

        # Add distance text at midpoint
        midpoint_x = (data_centroid.x + poly_centroid.x) / 2
        midpoint_y = (data_centroid.y + poly_centroid.y) / 2

        # Format distance nicely - show in kilometers if large, meters if small
        distance_text = (
            f"≈ {min_distance:.1f} m" if min_distance < 1000 else f"≈ {min_distance / 1000:.2f} km"
        )

        ax.text(
            midpoint_x,
            midpoint_y,
            distance_text,
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
        )

    # Determine axis limits to show both
    x_min = min(data_bounds[0], poly_bounds[0])
    y_min = min(data_bounds[1], poly_bounds[1])
    x_max = max(data_bounds[2], poly_bounds[2])
    y_max = max(data_bounds[3], poly_bounds[3])

    # Add some padding
    padding = 0.1  # 10% padding
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Add a context inset if areas are far apart
    # (Defined as: distance is more than 5x the average dimension of the boxes)
    avg_data_dim = ((data_bounds[2] - data_bounds[0]) + (data_bounds[3] - data_bounds[1])) / 2
    avg_poly_dim = ((poly_bounds[2] - poly_bounds[0]) + (poly_bounds[3] - poly_bounds[1])) / 2
    avg_dim = (avg_data_dim + avg_poly_dim) / 2

    if min_distance > 5 * avg_dim:
        # Areas are far apart, add an inset with the global view
        axins = ax.inset_axes([0.05, 0.05, 0.3, 0.3])
        gdf_poly.plot(ax=axins, color="red", alpha=0.5)
        data_box_gdf.plot(ax=axins, color="blue", alpha=0.5)
        axins.set_title("Overview")
        axins.set_xticks([])
        axins.set_yticks([])

    # Set title and labels
    ax.set_title("No Overlap Between Data and Polygons", fontsize=14)
    ax.set_xlabel("Easting (m)", fontsize=12)
    ax.set_ylabel("Northing (m)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Save the plot
    plt.tight_layout()

    if plots_out is not None:
        os.makedirs(f"{plots_out}/polygon_filtering_data", exist_ok=True)
        output_path = f"{plots_out}/polygon_filtering_data/no_overlap_{img_name}.png"
        plt.savefig(output_path, dpi=300)
        logging.info(f"No overlap plot saved to {output_path}")
    else:
        plt.savefig(f"no_overlap_issue.png", dpi=300)

    # Only display if we have enough time remaining
    if debug == True:
        plt.show()
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
        # Use vectorized approach for faster point creation from 30 seconds to 0.5 seconds

        points_gdf = gpd.GeoDataFrame(
            chunk_pd, geometry=gpd.points_from_xy(chunk_pd["Xw"], chunk_pd["Yw"]), crs=target_crs
        )

        # Initialize plot_id column with None
        points_gdf["plot_id"] = None

        # For each polygon, find points within it and assign plot ID
        for idx, polygon in polygons_gdf.iterrows():
            # Points within this specific polygon
            mask = points_gdf.geometry.within(polygon.geometry)
            if mask.any():
                # Get the ID from the polygon
                plot_id = polygon[id_field] if id_field in polygon else f"plot_{idx}"
                # Assign the ID to all points within this polygon
                points_gdf.loc[mask, "plot_id"] = plot_id

        # Return only points that got assigned a plot_id (i.e., are in any polygon)
        result = points_gdf[points_gdf["plot_id"].notna()]
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
    chunk_indices = [
        (i * points_per_chunk, min((i + 1) * points_per_chunk, n_points)) for i in range(n_chunks)
    ]

    logging.info(f"Prepared {n_chunks} chunks with ~{points_per_chunk:,} points each")
    logging.info(f"Will use {max_workers} workers for processing")

    return chunk_indices, max_workers, n_points, n_chunks


def process_chunks_parallel(
    df, chunk_indices, max_workers, polygons_gdf, target_crs, id_field, n_chunks
):
    """Process chunks in parallel using ThreadPoolExecutor."""
    phase_start = time.time()

    # Use a partial function with fixed arguments
    process_func = partial(
        process_chunk, df=df, polygons_gdf=polygons_gdf, target_crs=target_crs, id_field=id_field
    )

    filtered_chunks = []
    processed_count = 0

    # Process chunks in parallel with timeout awareness
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(process_func, chunk_idx): i for i, chunk_idx in enumerate(chunk_indices)
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):

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
                    f"Processed {processed_count}/{n_chunks} chunks ({(processed_count / n_chunks * 100):.1f}%)"
                )

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
        logging.info(
            f"Filtered points in {phase_time:.2f}s ({points_after / phase_time:.2f} points/s)"
        )

        # Clean up and convert back to Polars
        gdf_filtered = gdf_filtered.drop(columns=["geometry"])
        return pl.from_pandas(gdf_filtered)
    except Exception as e:
        logging.error(f"Error combining filtered results: {e}")
        return None


def plot_results(
    gdf_poly,
    gdf_filtered,
    target_crs,
    polygon_basename,
    data_bounds,
    sample_for_debug=5000,
    plots_out=None,
    img_name=None,
):
    """
    Generate a visualization of the filtered points within polygons.
    Randomly samples points and displays polygon IDs on the map.

    Args:
        gdf_poly: GeoDataFrame containing polygons with ID field
        gdf_filtered: GeoDataFrame containing filtered points
        target_crs: Coordinate reference system
        polygon_basename: Base name for the plot file
        sample_for_debug: Maximum number of points to sample for visualization
        plots_out: Output directory for plots
        img_name: Optional name for the image file
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot polygons with a semi-transparent fill
    gdf_poly.plot(ax=ax, alpha=0.3, edgecolor="black")

    points_after = len(gdf_filtered)

    # Sample points for visualization
    if points_after > sample_for_debug:
        # Ensure truly random sampling by using numpy's random state
        import numpy as np

        np.random.seed()  # Use system time as seed for true randomness

        # For pandas DataFrame
        if isinstance(gdf_filtered, pd.DataFrame):
            # Take a fully random sample (not sequential)
            random_indices = np.random.choice(
                points_after, size=min(sample_for_debug, points_after), replace=False
            )
            sample_df = gdf_filtered.iloc[random_indices]

        # For GeoDataFrame
        elif isinstance(gdf_filtered, gpd.GeoDataFrame):
            # Use GeoDataFrame's sample method with random_state=None for true randomness
            sample_df = gdf_filtered.sample(min(sample_for_debug, points_after), random_state=None)

        # For Polars DataFrame
        elif hasattr(gdf_filtered, "sample") and callable(getattr(gdf_filtered, "sample")):
            # If it's a Polars DataFrame with sample method
            sample_df = gdf_filtered.sample(n=min(sample_for_debug, points_after))
            # Convert to pandas for the rest of the processing
            if hasattr(sample_df, "to_pandas"):
                sample_df = sample_df.to_pandas()

        # Fallback to manual random sampling
        else:
            logging.info(f"Using manual random sampling for {type(gdf_filtered)}")
            # Convert to pandas first if needed
            if hasattr(gdf_filtered, "to_pandas"):
                temp_df = gdf_filtered.to_pandas()
            else:
                temp_df = pd.DataFrame(gdf_filtered)

            random_indices = np.random.choice(
                len(temp_df), size=min(sample_for_debug, len(temp_df)), replace=False
            )
            sample_df = temp_df.iloc[random_indices]

        # Create GeoDataFrame from the sampled points
        sample_points = gpd.GeoDataFrame(
            sample_df,
            geometry=gpd.points_from_xy(sample_df["Xw"], sample_df["Yw"]),
            crs=target_crs,
        )

        # Plot sampled points

        sample_points[sample_points.band1 != 0].plot(
            ax=ax, markersize=3, color="green", alpha=0.6, label="Valid Points"
        )
        sample_points[sample_points.band1 == 0].plot(
            ax=ax, markersize=3, color="grey", alpha=0.6, label="Null Points"
        )

        title = f"Points Inside Polygons (Random Sample of {len(sample_points):,} from {points_after:,} total)   "
    else:
        # Plot all points if few enough
        inside_points = gpd.GeoDataFrame(
            gdf_filtered,
            geometry=[Point(x, y) for x, y in zip(gdf_filtered["Xw"], gdf_filtered["Yw"])],
            crs=target_crs,
        )
        inside_points.plot(ax=ax, markersize=3, color="green", alpha=0.6, label="Points Inside")
        title = f"All {points_after:,} Points Inside Polygons"

    # Add polygon IDs as text on each polygon
    for idx, poly in gdf_poly.iterrows():
        # Get the centroid of the polygon for label placement
        centroid = poly.geometry.centroid

        # Determine the polygon ID to display
        if "id" in poly:
            poly_id = poly["id"]
        elif "plot_id" in poly:
            poly_id = poly["plot_id"]
        else:
            poly_id = f"plot {idx}"

        # Add text annotation with the polygon ID
        plt.text(
            centroid.x,
            centroid.y,
            poly_id,
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3"),
        )

    # Add a color-coded point count per polygon if we have points
    if points_after > 0 and "plot_id" in gdf_filtered.columns:
        # Get count by polygon

        gdf_filtered = gdf_filtered[gdf_filtered.band1 != 0]
        polygon_counts = gdf_filtered.groupby("plot_id").size()

        # Add a table with counts to the figure
        if len(polygon_counts) > 0:
            cell_text = [[f"{id}", f"{count}"] for id, count in polygon_counts.items()]
            count_table = plt.table(
                cellText=cell_text,
                colLabels=["Polygon ID", "Valid Point Count"],
                loc="lower right",
                cellLoc="center",
                bbox=[0.65, 0.02, 0.3, min(0.3, 0.02 * len(polygon_counts))],
            )
            count_table.auto_set_font_size(False)
            count_table.set_fontsize(9)
            for key, cell in count_table.get_celld().items():
                if key[0] == 0:  # Header row
                    cell.set_text_props(weight="bold")

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Improve axis appearance
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # Create data bounding box as polygon for better visualization
    data_box = box(data_bounds[0], data_bounds[1], data_bounds[2], data_bounds[3])
    data_box_gdf = gpd.GeoDataFrame(geometry=[data_box], crs=gdf_poly.crs)

    data_box_gdf.plot(
        ax=ax, facecolor="none", edgecolor="blue", linestyle="--", linewidth=2, label="Data Bounds"
    )

    # Save the plot
    plt.tight_layout()
    if plots_out is not None:
        os.makedirs(f"{plots_out}/polygon_filtering_data", exist_ok=True)
        plt.savefig(f"{plots_out}/polygon_filtering_data/polygon_filtering_{img_name}.png", dpi=300)
    else:
        plt.savefig(f"polygon_filtering_{polygon_basename}.png", dpi=300)

    # Show plot first, then close
    plt.show()
    plt.close()


def filter_df_by_polygon(
    df,
    polygon_path,
    target_crs="EPSG:32632",
    id_field="id",
    shrinkage=0.1,
    debug=True,
    sample_for_debug=5000,
    plots_out=None,
    img_name=None,
):
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
        has_overlap, data_bounds = check_data_polygon_overlap(df, polygons_gdf)

        if not has_overlap:
            logging.warning("No overlap between data and polygons. Returning None.")
            if debug:
                plot_no_overlap(
                    polygons_gdf, data_bounds, img_name=img_name, plots_out=plots_out, debug=debug
                )
            return None

        # --- PHASE 5: Prepare chunks for parallel processing ---
        chunk_indices, max_workers, n_points, n_chunks = prepare_chunks(df)

        # --- PHASE 6: Process chunks in parallel ---
        filtered_chunks, processed_count, phase_time = process_chunks_parallel(
            df,
            chunk_indices,
            max_workers,
            polygons_gdf,
            target_crs,
            id_field,  # This is the default id_field, you can make it a parameter
            n_chunks,
        )

        # Check if we processed anything
        if processed_count == 0:
            logging.warning("No chunks were processed successfully. Returning original data.")
            return df

        # --- PHASE 7: Combine results ---
        if not filtered_chunks:
            logging.warning("No points found inside polygons. Returning None.")
            return None

        result_df = combine_chunk_results(filtered_chunks, n_points, phase_time)
        if result_df is None or len(result_df) == 0:
            logging.warning("No points after filtering. Returning original data.")
            return df

            # --- PHASE 8: Debug visualization ---
        try:
            gdf_filtered = pd.concat(filtered_chunks, ignore_index=True)
            plot_results(
                polygons_gdf,
                gdf_filtered,
                target_crs,
                polygon_basename,
                data_bounds,
                sample_for_debug,
                plots_out,
                img_name,
            )
        except Exception as e:
            logging.error(f"Error generating debug plot: {e}")
        return result_df

    except Exception as e:
        logging.error(f"Error in polygon filtering: {str(e)}")
        import traceback

        logging.error(traceback.format_exc())
        return df

    finally:
        disable_timeout()
