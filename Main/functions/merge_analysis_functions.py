import polars as pl
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from timeit import default_timer as timer


def visualize_coordinate_alignment(df_dem, df_allbands, precision, folder_name="coordinate_alignments"):
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