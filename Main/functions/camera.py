import logging
import os
import traceback
from timeit import default_timer as timer
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from rasterio.warp import transform

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




def get_camera_position(cam_path, name, target_crs=None ):
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

def plot_angles(df_merged, xcam, ycam, zcam, path, file_name):
    """
    Plot the angles and camera position.
    """

    # --------------------
    # TOP-DOWN VIEW
    # --------------------

    plt.figure(figsize=(8, 8))
    plt.scatter(df_merged["Xw"], df_merged["Yw"], s=10, alpha=0.5, label="Ground Points")
    plt.scatter([xcam], [ycam], c='red', label="Drone")

    # Draw view vectors
    #for i in range(0, len(df_merged), 1000):  # Use step to avoid clutter
    #    plt.plot([xcam, df_merged["Xw"][i]], [ycam, df_merged["Yw"][i]], alpha=0.3)

    plt.legend()
    plt.title("Top-Down Projection of Drone to Ground Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.savefig(os.path.join(path, f"top_down/angle_data_{file_name}.png"), dpi=200)

    # --------------------
    # SIDE VIEW
    # --------------------

    plt.figure(figsize=(10, 5))
    plt.scatter(df_merged["Yw"], df_merged["elev"], label="Ground Elevation", alpha=0.5)

    #for i in range(0, len(df_merged), 1000):
    #    plt.plot([0, df_merged["distance_xy"][i]], [zcam, df_merged["elev"][i]], alpha=0.3)

    plt.scatter([ycam], [zcam], c='red', label="Drone")
    plt.xlabel("Horizontal Distance")
    plt.ylabel("Elevation")
    plt.title("Side View: Drone Viewing Geometry")
    plt.legend()
    plt.savefig(os.path.join(path, f"side_view/angle_data_{file_name}.png"), dpi=200)

    # --------------------
    # 3D VIEW
    # --------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Ground points
    ax.scatter(df_merged["Xw"], df_merged["Yw"], df_merged["elev"], s=5, alpha=0.6, label="Ground Points")

    # Drone position
    ax.scatter([xcam], [ycam], [zcam], c='red', label="Drone")

    # Viewing rays
    for i in range(0, len(df_merged), 10000):
        ax.plot(
            [xcam, df_merged["Xw"][i]],
            [ycam, df_merged["Yw"][i]],
            [zcam, df_merged["elev"][i]],
            alpha=0.2
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Elevation (Z)")
    ax.set_title("3D Visualization of Drone Viewing Geometry")
    plt.savefig(os.path.join(path, f"3d_view/angle_data_{file_name}.png"), dpi=200)

    ax.legend()

