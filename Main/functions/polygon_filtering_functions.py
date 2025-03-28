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


def filter_df_by_polygon(df, polygon_path, target_crs="EPSG:32632", shrinkage=0.1, debug=True):
    """
    Filter a Polars DataFrame to only include points that fall within polygons from a file.

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

    Returns:
    --------
    polars.DataFrame
        Filtered DataFrame containing only points inside the polygon
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import logging
    from shapely.geometry import Point
    import geopandas as gpd
    import polars as pl
    import os
    from pyproj import CRS

    # Read the polygon file
    gdf_poly = gpd.read_file(polygon_path)
    points_before = len(df)

    logging.info(f"Points before filtering: {points_before}")

    # Get the file name for debug plots
    polygon_basename = os.path.basename(polygon_path)

    # ------ CRS HANDLING SECTION ------
    # Check and report the original CRS
    if gdf_poly.crs is None:
        logging.warning(f"Polygon file has no defined CRS. Assuming {target_crs}")
        gdf_poly.set_crs(target_crs, inplace=True)
    else:
        original_crs = gdf_poly.crs
        logging.info(f"Original polygon CRS: {original_crs}")

        # Check if this looks like a geographic CRS with coordinates in degrees (WGS84 etc.)
        bounds = gdf_poly.total_bounds
        is_likely_geographic = (
                abs(bounds[0]) <= 180 and
                abs(bounds[1]) <= 90 and
                abs(bounds[2]) <= 180 and
                abs(bounds[3]) <= 90
        )

        # If it's WGS84, convert to the target CRS
        if original_crs == CRS.from_epsg(4326):
            if not is_likely_geographic:
                logging.warning(
                    f"WARNING: Polygon file claims to be WGS84 (EPSG:4326) but coordinates don't look geographic. "
                    f"Bounds: {bounds}. Forcing CRS to {target_crs}"
                )
                gdf_poly.set_crs(target_crs, inplace=True, allow_override=True)
            else:
                logging.info(f"Converting polygons from {original_crs} to {target_crs}")
                gdf_poly = gdf_poly.to_crs(target_crs)

        # If it's not the target CRS, convert to target
        elif original_crs != CRS.from_string(target_crs):
            logging.info(f"Converting polygons from {original_crs} to {target_crs}")
            gdf_poly = gdf_poly.to_crs(target_crs)

    # --------------------------------

    # Filter valid, non-empty geometries and report counts
    original_count = len(gdf_poly)
    gdf_poly = gdf_poly[gdf_poly.is_valid]
    valid_count = len(gdf_poly)

    if valid_count < original_count:
        logging.warning(f"Removed {original_count - valid_count} invalid geometries")

    valid_gdf = gdf_poly[~gdf_poly.geometry.is_empty]
    nonempty_count = len(valid_gdf)

    if nonempty_count < valid_count:
        logging.warning(f"Removed {valid_count - nonempty_count} empty geometries")

    if valid_gdf.empty:
        logging.error(f"No valid polygons found in the file: {polygon_path}")
        # Return original dataframe instead of None
        return df

    logging.info(f"Using {nonempty_count} valid polygons from {polygon_path}")

    # Debug plot: Original polygons with coordinate display
    if debug:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Original polygons
        valid_gdf.plot(ax=axes[0], alpha=0.5, edgecolor='black')

        # Add coordinate info
        bounds = valid_gdf.total_bounds
        axes[0].set_title(f"Original Polygons - {polygon_basename}\n({len(valid_gdf)} polygons)\nCRS: {valid_gdf.crs}")
        axes[0].text(
            0.05, 0.05,
            f"X range: {bounds[0]:.2f} to {bounds[2]:.2f}\nY range: {bounds[1]:.2f} to {bounds[3]:.2f}",
            transform=axes[0].transAxes,
            bbox=dict(facecolor='white', alpha=0.7)
        )
        axes[0].grid(True)

    # Shrink each polygon by applying a negative buffer - the buffer is proportional to polygon size
    original_polys = valid_gdf.copy()

    if shrinkage > 0:
        # Calculate area-based buffer distances for each polygon
        valid_gdf['area'] = valid_gdf.geometry.area
        valid_gdf['buffer_distance'] = -1 * valid_gdf['area'].apply(lambda x: np.sqrt(x) * shrinkage)

        # Log the buffer distances
        min_buffer = valid_gdf['buffer_distance'].min()
        max_buffer = valid_gdf['buffer_distance'].max()
        avg_buffer = valid_gdf['buffer_distance'].mean()
        logging.info(f"Buffer distances: min={min_buffer:.2f}, max={max_buffer:.2f}, avg={avg_buffer:.2f}")

        # Apply the calculated buffer to each polygon
        valid_gdf['original_geometry'] = valid_gdf.geometry  # Keep original for reference

        valid_gdf['geometry'] = valid_gdf.apply(
            lambda row: row.geometry.buffer(
                row.buffer_distance) if row.buffer_distance > -row.area ** 0.4 else row.geometry,
            axis=1
        )

        # Check for topology errors in buffered geometries
        valid_gdf['is_valid'] = valid_gdf.geometry.is_valid
        invalid_mask = ~valid_gdf['is_valid']
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            logging.warning(f"{invalid_count} geometries became invalid after buffering. Attempting to fix.")
            valid_gdf.loc[invalid_mask, 'geometry'] = valid_gdf.loc[invalid_mask, 'geometry'].apply(
                lambda g: g.buffer(0))

        # Remove any polygons that might have become empty after shrinking
        valid_before = len(valid_gdf)
        valid_gdf = valid_gdf[~valid_gdf.geometry.is_empty]
        valid_after = len(valid_gdf)

        if valid_after < valid_before:
            logging.warning(f"{valid_before - valid_after} polygons became empty after shrinking")

        if valid_gdf.empty:
            logging.warning(f"All polygons became empty after applying shrinkage factor of {shrinkage}")
            # Return original dataframe instead of None or empty dataframe
            return df

    # Debug plot: Shrunk polygons
    if debug and shrinkage > 0:
        # Plot 2: Shrunk polygons
        valid_gdf.plot(ax=axes[1], alpha=0.5, edgecolor='red')
        original_polys.boundary.plot(ax=axes[1], color='black', linestyle='--', alpha=0.7)
        axes[1].set_title(f"Shrunk Polygons\n(shrinkage={shrinkage})")

        # Add info about shrinkage
        axes[1].text(
            0.05, 0.05,
            f"Avg. buffer: {avg_buffer:.2f}\nPolygons: {valid_after}/{valid_before}",
            transform=axes[1].transAxes,
            bbox=dict(facecolor='white', alpha=0.7)
        )
        axes[1].grid(True)

    # Union all valid polygons into one geometry
    try:
        union_poly = valid_gdf.geometry.unary_union
        logging.info(f"Created union polygon with bounds: {union_poly.bounds}")
    except Exception as e:
        logging.error(f"Error creating union polygon: {str(e)}")
        if debug:
            valid_gdf.plot(figsize=(10, 8))
            plt.title("Polygons causing union error")
            plt.savefig('problematic_polygons.png', dpi=300)
            plt.show()
        # Return original DataFrame in case of error
        return df

    # Convert the Polars DataFrame to Pandas for spatial operations
    df_pd = df.to_pandas()

    # Create shapely Points for each row
    df_pd["geometry"] = df_pd.apply(lambda row: Point(row["Xw"], row["Yw"]), axis=1)

    # Create a GeoDataFrame with the correct CRS
    gdf_points = gpd.GeoDataFrame(df_pd, geometry="geometry", crs=target_crs)

    # Extra check for data bounds vs. polygon bounds
    if debug:
        data_bounds = gdf_points.total_bounds
        poly_bounds = valid_gdf.total_bounds

        logging.info(
            f"Data bounds: X ({data_bounds[0]:.2f}, {data_bounds[2]:.2f}), Y ({data_bounds[1]:.2f}, {data_bounds[3]:.2f})")
        logging.info(
            f"Polygon bounds: X ({poly_bounds[0]:.2f}, {poly_bounds[2]:.2f}), Y ({poly_bounds[1]:.2f}, {poly_bounds[3]:.2f})")

        # Check for overlap between data and polygons
        overlap_x = min(data_bounds[2], poly_bounds[2]) - max(data_bounds[0], poly_bounds[0])
        overlap_y = min(data_bounds[3], poly_bounds[3]) - max(data_bounds[1], poly_bounds[1])

        # Check if there's no overlap
        if overlap_x <= 0 or overlap_y <= 0:
            logging.warning("WARNING: No overlap between data and polygons! Filter will remove all points.")

            # Create a more detailed plot to visualize the issue
            if debug:
                plt.figure(figsize=(12, 10))
                valid_gdf.plot(alpha=0.5, edgecolor='red', label='Polygons')

                # Plot a sample of points
                sample_size = min(5000, len(gdf_points))
                gdf_points_sample = gdf_points.sample(sample_size) if len(gdf_points) > sample_size else gdf_points
                gdf_points_sample.plot(markersize=2, alpha=0.6, color='blue', label='Data Points')

                plt.title("No Overlap Between Data and Polygons")
                plt.legend()
                plt.grid(True)
                plt.savefig('no_overlap_issue.png', dpi=300)
                plt.show()

            # Return a warning message
            logging.warning("Returning the original DataFrame as there's no overlap with the polygons.")
            return df
        else:
            logging.info(f"Data and polygons overlap by X: {overlap_x:.2f}, Y: {overlap_y:.2f}")

    # Sample points for visualization (if the dataset is large)
    if debug:
        sample_size = min(5000, len(gdf_points))
        gdf_points_sample = gdf_points.sample(sample_size) if len(gdf_points) > sample_size else gdf_points

    # Perform the actual filtering
    gdf_filtered = gdf_points[gdf_points.geometry.within(union_poly)].copy()

    points_after = len(gdf_filtered)
    points_filtered = points_before - points_after

    percentage_filtered = (points_filtered / points_before * 100) if points_before > 0 else 0
    logging.info(f"Points filtered: {points_filtered:,} out of {points_before:,} ({percentage_filtered:.2f}%)")
    logging.info(f"Points remaining: {points_after:,} ({100 - percentage_filtered:.2f}%)")

    # If all points are filtered out, this is likely an error
    if points_after == 0:
        logging.warning(
            "All points were filtered out! This may indicate a coordinate system mismatch or polygon issue.")
        logging.warning("Returning the original DataFrame to avoid pipeline failure.")

        # Debug plot: Points and polygon
        if debug:
            # Plot 3: Points and polygon overlap problem
            valid_gdf.plot(ax=axes[2], alpha=0.3, edgecolor='black')

            # Plot a sample of points
            if not gdf_points_sample.empty:
                gdf_points_sample.plot(ax=axes[2], markersize=2, color='red', alpha=0.5, label='All Outside')

            axes[2].set_title("Filtering Issue: All Points Outside Polygons")
            axes[2].legend()
            axes[2].grid(True)

            plt.tight_layout()
            plt.savefig(f'polygon_filtering_all_outside_{polygon_basename}.png', dpi=300)
            plt.show()

            # Create a more detailed diagnostic plot
            plt.figure(figsize=(12, 10))
            valid_gdf.plot(alpha=0.5, edgecolor='red', label='Polygons')

            # Plot a sample of points
            gdf_points_sample.plot(markersize=2, alpha=0.6, color='blue', label='Data Points')

            plt.title("Diagnostic: All Points Outside Polygons")
            plt.legend()
            plt.grid(True)
            plt.savefig('all_outside_issue.png', dpi=300)
            plt.show()

        # Return the original DataFrame instead of an empty one
        return df

    # Debug plot: Points and polygon
    if debug and points_after > 0:
        # Plot 3: Points and polygon
        valid_gdf.plot(ax=axes[2], alpha=0.3, edgecolor='black')

        # Plot a sample of points colored by whether they're inside or outside
        if not gdf_points_sample.empty:
            # Mark which points are inside
            gdf_points_sample['inside'] = gdf_points_sample.geometry.within(union_poly)

            # Plot points inside the polygon
            inside_points = gdf_points_sample[gdf_points_sample['inside']]
            if not inside_points.empty:
                inside_points.plot(ax=axes[2], markersize=2, color='green', alpha=0.5, label='Inside')

            # Plot points outside the polygon
            outside_points = gdf_points_sample[~gdf_points_sample['inside']]
            if not outside_points.empty:
                outside_points.plot(ax=axes[2], markersize=2, color='red', alpha=0.5, label='Outside')

        axes[2].set_title(f"Point Filtering\n({points_after:,} inside, {points_filtered:,} outside)")

        # Add percentage info
        axes[2].text(
            0.05, 0.05,
            f"Kept: {100 - percentage_filtered:.1f}%\nFiltered: {percentage_filtered:.1f}%",
            transform=axes[2].transAxes,
            bbox=dict(facecolor='white', alpha=0.7)
        )

        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(f'polygon_filtering_{polygon_basename}.png', dpi=300)
        plt.show()

        # Show a more detailed point distribution for better verification
        if not gdf_filtered.empty and len(gdf_filtered) > 0:
            plt.figure(figsize=(12, 10))

            # Plot the polygons
            valid_gdf.plot(alpha=0.3, edgecolor='black')

            # Plot the filtered points (sample if too many)
            filtered_sample = gdf_filtered.sample(min(5000, len(gdf_filtered))) if len(
                gdf_filtered) > 5000 else gdf_filtered
            filtered_sample.plot(markersize=2, alpha=0.6, color='blue')

            plt.title(f"Detailed View: {len(filtered_sample)} sample points (out of {len(gdf_filtered):,} filtered)")
            plt.grid(True)
            plt.savefig(f'detailed_points_{polygon_basename}.png', dpi=300)
            plt.show()

    # Drop the geometry column and convert back to Polars DataFrame
    gdf_filtered = gdf_filtered.drop(columns=["geometry"])

    # Return the filtered DataFrame
    return pl.from_pandas(gdf_filtered)