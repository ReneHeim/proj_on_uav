import logging
import os
import traceback
from timeit import default_timer as timer

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from rasterio.warp import transform

# ------------------------------
# Calculate Viewing Angles
# ------------------------------
def calculate_angles(df_merged, xcam, ycam, zcam, sunelev, saa):
    """
    Calculate view zenith and azimuth angles for each point in a point cloud
    relative to a camera (e.g., drone) position.

    Args:
        df_merged (pl.DataFrame): Point cloud, must contain columns: 'elev', 'Xw', 'Yw', 'band1'.
        xcam, ycam, zcam (float): Camera coordinates (projected).
        sunelev (float): Sun elevation (deg), stashed in output for reference.
        saa (float): Solar azimuth angle (deg), used for relative azimuth calculation.

    Returns:
        pl.DataFrame: Original DataFrame with new columns:
            - delta_x, delta_y, delta_z: Vector from ground point to camera (meters)
            - distance_xy: Horizontal distance ground↔camera
            - angle_rad, vza: View zenith angle (rad, deg)
            - vaa_rad, vaa_temp, vaa: View azimuth angles (deg, radians), relative to solar azimuth
            - xcam, ycam, sunelev, saa: Repeated as constants for reference

    Notes:
        - Sets vza/vaa to None where band1==65535 (masked).
        - Masks out points outside 2nd-98th elev percentile.
        - If input columns missing, will raise; not validated.
        - No file IO; no data is removed.
    """
    start_angles = timer()
    try:
        # ------------------------------------------------------------------
        # 1. components of the view vector (camera  minus  ground point)
        # ------------------------------------------------------------------
        df_merged = df_merged.with_columns([
            (pl.lit(zcam, dtype=pl.Float32) - pl.col("elev")).alias("delta_z"),
            (pl.lit(xcam, dtype=pl.Float32) - pl.col("Xw")).alias("delta_x"),
            (pl.lit(ycam, dtype=pl.Float32) - pl.col("Yw")).alias("delta_y")
        ])

        # horizontal distance
        df_merged = df_merged.with_columns(
            ((pl.col("delta_x")**2 + pl.col("delta_y")**2).sqrt()).alias("distance_xy")
        )

        # ------------------------------------------------------------------
        # 2. View-Zenith Angle  (0° at nadir, 90° at horizon)
        #    angle_rad kept for backward compatibility
        # ------------------------------------------------------------------
        df_merged = df_merged.with_columns(
            pl.arctan2(pl.col("distance_xy"),            # NOTE: distance first,
                       pl.col("delta_z"))                #       height second
              .alias("angle_rad")
        ).with_columns(
            (pl.col("angle_rad") * 180 / np.pi)          # radians → degrees
              .round(2)
              .alias("vza")
        )

        # ------------------------------------------------------------------
        # 3. View-Azimuth Angle  (absolute, radians)
        #    Note the argument order (east, north)
        # ------------------------------------------------------------------
        df_merged = df_merged.with_columns(
            pl.arctan2(pl.col("delta_x"), pl.col("delta_y")).alias("vaa_rad")
        )

        # same relative-azimuth logic, preserving vaa_temp → vaa
        df_merged = df_merged.with_columns(
            ((pl.col("vaa_rad") * 180 / np.pi) - saa).alias("vaa_temp")
        )
        df_merged = df_merged.with_columns(
            (((pl.col("vaa_temp") + 360) % 360)).alias("vaa")
        )

        # ------------------------------------------------------------------
        # 4. masking logic (unchanged)
        # ------------------------------------------------------------------
        p05 = df_merged.select(pl.col("elev").quantile(0.02, interpolation="nearest")).item()
        p95 = df_merged.select(pl.col("elev").quantile(0.98, interpolation="nearest")).item()

        df_merged = df_merged.with_columns([
            pl.when(pl.col("band1") == 65535).then(None).otherwise(pl.col("vza")).alias("vza"),
            pl.when(pl.col("band1") == 65535).then(None).otherwise(pl.col("vaa")).alias("vaa"),
            pl.when((pl.col("elev") < p05) | (pl.col("elev") > p95))
              .then(None)
              .otherwise(pl.col("elev"))
              .alias("elev")
        ])

        # ------------------------------------------------------------------
        # 5. Stash constants (unchanged)
        # ------------------------------------------------------------------
        df_merged = df_merged.with_columns([
            pl.lit(xcam, dtype=pl.Float32).alias("xcam"),
            pl.lit(ycam, dtype=pl.Float32).alias("ycam"),
            pl.lit(sunelev, dtype=pl.Float32).alias("sunelev"),
            pl.lit(saa,     dtype=pl.Float32).alias("saa")
        ])

        end_angles = timer()
        logging.info(f"Calculated angles in {end_angles - start_angles:.2f} seconds")
        return df_merged

    except Exception as e:
        logging.error(f"Error calculating angles: {e}")
        logging.error(traceback.format_exc())
        raise



def get_camera_position(cam_path, name, target_crs=None ):
    """
    Extract the 3D position of a specific camera/image from a text file.

    Args:
        cam_path (str or Path): Path to camera position file (tab-separated, no header, skip 2 rows).
        name (str): Substring to match the 'PhotoID' of the image/camera.
        target_crs (str, optional): EPSG code (e.g., 'EPSG:32632'). If set, position is reprojected.

    Returns:
        tuple (float, float, float): (x, y, z) camera coordinates. May be lon/lat or projected.

    Raises:
        - Any file or parsing errors will be logged and re-raised.
        - If name not found, may raise IndexError.

    Notes:
        - Assumes file structure and delimiters are correct.
        - Uses rasterio.transform for CRS change (if needed).
    """
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

        if target_crs is not None:
            lon , lat = transform('EPSG:4326', target_crs, [lon], [lat])
            lon , lat = lon[0], lat[0]

        end = timer()
        logging.info(f"Retrieved camera position for {name} in {end - start:.2f} seconds")
        return float(lon), float(lat), float(zcam)
    except Exception as e:
        logging.error(f"Error retrieving camera position for {name}: {e}")
        raise

def plot_angles(df_merged, lon, lat, zcam, path, file_name):
    """
    Plot viewing angles for the merged data.
    
    Args:
        df_merged: Polars DataFrame with angle data
        lon: Camera longitude
        lat: Camera latitude  
        zcam: Camera altitude
        path: Output path for plots
        file_name: Name of the file being processed
    """
    import matplotlib.pyplot as plt
    import os
    
    # Create necessary directories
    top_down_dir = os.path.join(path, "top_down")
    side_view_dir = os.path.join(path, "side_view")
    view_3d_dir = os.path.join(path, "3d_view")
    
    for dir_path in [top_down_dir, side_view_dir, view_3d_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Convert to pandas for plotting
    df_pandas = df_merged.to_pandas()
    
    # Top-down view
    plt.figure(figsize=(10, 8))
    plt.scatter(df_pandas['Xw'], df_pandas['Yw'], c=df_pandas['vza'], cmap='viridis', s=1)
    plt.colorbar(label='View Zenith Angle (degrees)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'View Zenith Angle - {file_name}')
    plt.savefig(os.path.join(top_down_dir, f"angle_data_{file_name}.png"), dpi=200)
    plt.close()
    
    # Side view
    plt.figure(figsize=(10, 8))
    plt.scatter(df_pandas['Xw'], df_pandas['elev'], c=df_pandas['vza'], cmap='viridis', s=1)
    plt.colorbar(label='View Zenith Angle (degrees)')
    plt.xlabel('X (m)')
    plt.ylabel('Elevation (m)')
    plt.title(f'View Zenith Angle vs Elevation - {file_name}')
    plt.savefig(os.path.join(side_view_dir, f"angle_data_{file_name}.png"), dpi=200)
    plt.close()
    
    # 3D view
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_pandas['Xw'], df_pandas['Yw'], df_pandas['elev'], 
                        c=df_pandas['vza'], cmap='viridis', s=1)
    plt.colorbar(scatter, label='View Zenith Angle (degrees)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title(f'3D View Zenith Angle - {file_name}')
    plt.savefig(os.path.join(view_3d_dir, f"angle_data_{file_name}.png"), dpi=200)
    plt.close()

