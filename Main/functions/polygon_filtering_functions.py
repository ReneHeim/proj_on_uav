import logging
from shapely.geometry import Point
import geopandas as gpd
import os
import polars as pl


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


def filter_df_by_polygon(df, polygon_path, target_crs="EPSG:32632", shrinkage=0.1,
                         debug=True, max_runtime_seconds=60, sample_for_debug=5000):
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
    max_runtime_seconds : int, default=60
        Maximum time in seconds to let the function run before returning original df
    sample_for_debug : int, default=5000
        Maximum number of points to use for debug visualizations

    Returns:
    --------
    polars.DataFrame
        Filtered DataFrame containing only points inside the polygon
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import logging
    import time
    import os
    import signal
    from functools import partial
    from shapely.geometry import Point
    import geopandas as gpd
    import polars as pl
    from pyproj import CRS

    start_time = time.time()

    # Set up a timeout handler
    class TimeoutException(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutException("Function timed out")

    # Set the timeout if on Unix (Windows doesn't support SIGALRM)
    if os.name != 'nt':  # Not Windows
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max_runtime_seconds)

    try:
        points_before = len(df)
        logging.info(f"Points before filtering: {points_before:,}")
        polygon_basename = os.path.basename(polygon_path)

        # ----- PHASE 1: LOAD AND PREPARE POLYGONS -----
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

        if gdf_poly.empty:
            logging.error(f"No valid polygons in {polygon_path}")
            return df

        # Log phase timing
        phase_time = time.time() - phase_start
        logging.info(f"Phase 1: Loaded {len(gdf_poly)} polygons in {phase_time:.2f}s")

        # ----- PHASE 2: POLYGON SHRINKAGE -----
        phase_start = time.time()

        # Apply shrinkage with performance-optimized approach
        if shrinkage > 0:
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

            if gdf_poly.empty:
                logging.warning(f"All polygons empty after shrinkage. Returning original data.")
                return df

            logging.info(f"Applied shrinkage factor {shrinkage} to polygons")

        # Log phase timing
        phase_time = time.time() - phase_start
        logging.info(f"Phase 2: Processed polygons in {phase_time:.2f}s")

        # ----- PHASE 3: CREATE UNION POLYGON -----
        phase_start = time.time()

        # Union all geometries into a single polygon for faster containment check
        union_poly = gdf_poly.geometry.unary_union
        logging.info(f"Union polygon bounds: {union_poly.bounds}")

        # Log phase timing
        phase_time = time.time() - phase_start
        logging.info(f"Phase 3: Created union polygon in {phase_time:.2f}s")

        # ----- PHASE 4: PREPARE POINTS FOR SPATIAL OPERATION -----
        phase_start = time.time()

        # Check if there could be an overlap (quick check before full spatial operation)
        data_sample = df.select(pl.col("Xw").min(), pl.col("Xw").max(),
                                pl.col("Yw").min(), pl.col("Yw").max()).collect()

        data_bounds = [
            data_sample.item(0, 0), data_sample.item(0, 2),  # xmin, ymin
            data_sample.item(0, 1), data_sample.item(0, 3)  # xmax, ymax
        ]

        poly_bounds = union_poly.bounds  # (xmin, ymin, xmax, ymax)

        # Check for potential overlap
        overlap_x = min(data_bounds[2], poly_bounds[2]) - max(data_bounds[0], poly_bounds[0])
        overlap_y = min(data_bounds[3], poly_bounds[3]) - max(data_bounds[1], poly_bounds[1])

        if overlap_x <= 0 or overlap_y <= 0:
            logging.warning("No overlap between data and polygons. Returning original data.")

            # Debug plot for no overlap case
            if debug:
                plt.figure(figsize=(10, 8))
                gdf_poly.plot(alpha=0.5, edgecolor='red')
                plt.plot([data_bounds[0], data_bounds[2], data_bounds[2], data_bounds[0], data_bounds[0]],
                         [data_bounds[1], data_bounds[1], data_bounds[3], data_bounds[3], data_bounds[1]],
                         'b--', label='Data bounds')
                plt.title("No Overlap Between Data and Polygons")
                plt.legend()
                plt.savefig('no_overlap_issue.png', dpi=200)
                if time.time() - start_time < max_runtime_seconds - 5:  # Only show if we have time
                    plt.show()
                else:
                    plt.close()

            return df

        # Check if we've been running too long already
        if time.time() - start_time > max_runtime_seconds * 0.6:
            logging.warning(f"Already used > 60% of max runtime ({max_runtime_seconds}s). Returning original data.")
            return df

        # Optimize point conversion to geometry using vectorized operations
        # Rather than converting the entire dataframe at once, which can be memory-intensive,
        # process in chunks to maintain memory efficiency

        # Get number of points and determine chunk size based on available memory
        n_points = len(df)
        points_per_chunk = min(1000000, max(10000, n_points // 10))  # Adaptive chunking
        n_chunks = (n_points + points_per_chunk - 1) // points_per_chunk

        # Log phase timing
        phase_time = time.time() - phase_start
        logging.info(f"Phase 4: Prepared for filtering in {phase_time:.2f}s")

        # ----- PHASE 5: FILTER POINTS IN CHUNKS -----
        phase_start = time.time()

        # Function to process a chunk of points
        def process_chunk(chunk_df, union_polygon, target_crs):
            chunk_pd = chunk_df.to_pandas()
            # Create points from coordinates
            points = [Point(x, y) for x, y in zip(chunk_pd['Xw'], chunk_pd['Yw'])]
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(chunk_pd, geometry=points, crs=target_crs)
            # Filter points within polygon
            return gdf[gdf.geometry.within(union_polygon)]

        filtered_chunks = []

        # Process in chunks to avoid memory issues
        for i in range(n_chunks):
            # Check if we're approaching timeout
            if time.time() - start_time > max_runtime_seconds * 0.9:
                logging.warning(f"Approaching timeout. Processed {i}/{n_chunks} chunks. Returning original data.")
                return df

            # Get chunk
            start_idx = i * points_per_chunk
            end_idx = min((i + 1) * points_per_chunk, n_points)
            chunk = df.slice(start_idx, end_idx - start_idx)

            # Process chunk
            filtered_chunk = process_chunk(chunk, union_poly, target_crs)
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)

            # Log progress for long-running operations
            if i % max(1, n_chunks // 10) == 0:
                logging.info(f"Processed {i + 1}/{n_chunks} chunks ({((i + 1) / n_chunks * 100):.1f}%)")

        # Combine results from all chunks
        if not filtered_chunks:
            logging.warning("No points found inside polygons. Returning original data.")
            return df

        gdf_filtered = pd.concat(filtered_chunks, ignore_index=True)

        # Get stats on filtered data
        points_after = len(gdf_filtered)
        points_filtered = points_before - points_after
        percentage_filtered = (points_filtered / points_before * 100) if points_before > 0 else 0

        logging.info(f"Points inside polygons: {points_after:,} ({100 - percentage_filtered:.2f}%)")
        logging.info(f"Points filtered out: {points_filtered:,} ({percentage_filtered:.2f}%)")

        # Log phase timing
        phase_time = time.time() - phase_start
        logging.info(f"Phase 5: Filtered points in {phase_time:.2f}s ({points_after / phase_time:.2f} points/s)")

        # ----- PHASE 6: PREPARE RESULTS AND DEBUG PLOTS -----
        phase_start = time.time()

        # If all points are filtered out, return original
        if points_after == 0:
            logging.warning("All points filtered out. Returning original data.")
            return df

        # Clean up and convert back to Polars
        gdf_filtered = gdf_filtered.drop(columns=["geometry"])
        result_df = pl.from_pandas(gdf_filtered)

        # Debug plots - only if we're not close to timeout
        if debug and time.time() - start_time < max_runtime_seconds * 0.9:
            # Plot results
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Plot polygons
            gdf_poly.plot(ax=ax, alpha=0.3, edgecolor='black')

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
            plt.savefig(f'polygon_filtering_{polygon_basename}.png', dpi=200)

            # Only show if we have time
            if time.time() - start_time < max_runtime_seconds - 2:
                plt.show()
            else:
                plt.close()

        # Log phase timing
        phase_time = time.time() - phase_start
        total_time = time.time() - start_time
        logging.info(f"Phase 6: Prepared results in {phase_time:.2f}s")
        logging.info(
            f"Total runtime: {total_time:.2f}s for {points_before:,} points ({points_before / total_time:.2f} points/s)")

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
        # Disable the alarm if on Unix
        if os.name != 'nt':
            signal.alarm(0)