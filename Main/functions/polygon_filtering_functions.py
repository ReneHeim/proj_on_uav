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




def filter_df_by_polygon(df, polygon_path, target_crs="EPSG:32632", shrinkage=0.1):
    # Read the polygon file
    gdf_poly = gpd.read_file(polygon_path)
    points_before = len(f"Points before filtering: {len(gdf_poly)}")

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

    # Shrink each polygon by applying a negative buffer - the buffer is proportional to polygon size
    if shrinkage > 0:
        # Calculate area-based buffer distances for each polygon
        valid_gdf['area'] = valid_gdf.geometry.area
        valid_gdf['buffer_distance'] = -1 * valid_gdf['area'].apply(lambda x: np.sqrt(x) * shrinkage)

        # Apply the calculated buffer to each polygon
        valid_gdf['geometry'] = valid_gdf.apply(
            lambda row: row.geometry.buffer(row.buffer_distance) if not row.geometry.is_empty else row.geometry,
            axis=1
        )

        # Remove any polygons that might have become empty after shrinking
        valid_gdf = valid_gdf[~valid_gdf.geometry.is_empty]

        if valid_gdf.empty:
            logging.warning(f"All polygons became empty after applying shrinkage factor of {shrinkage}")
            return df.filter(pl.lit(False))  # Return empty dataframe with same schema

    # Union all valid polygons into one geometry
    try:
        union_poly = valid_gdf.geometry.unary_union  # This works with both newer and older versions
    except AttributeError:
        try:
            union_poly = valid_gdf.union_all()  # Newer API
        except AttributeError:
            union_poly = valid_gdf.unary_union  # Older API

    print("Union polygon bounds:", union_poly.bounds)

    # Convert the Polars DataFrame to Pandas
    df_pd = df.to_pandas()

    # Create shapely Points for each row
    df_pd["geometry"] = df_pd.apply(lambda row: Point(row["Xw"], row["Yw"]), axis=1)

    # Filter rows where the point is within the union polygon
    gdf_points = gpd.GeoDataFrame(df_pd, geometry="geometry", crs=target_crs)
    gdf_filtered = gdf_points[gdf_points["geometry"].within(union_poly)].copy()

    points_after = len(gdf_filtered)
    points_filtered = points_before - points_after

    logging.info(f"Points filtered: {points_filtered} ({points_filtered / points_before * 100:.2f}%)")

    # Drop the geometry column and convert back to Polars DataFrame
    gdf_filtered = gdf_filtered.drop(columns=["geometry"])
    return pl.from_pandas(gdf_filtered)