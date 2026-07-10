"""Visualize the viewing angle distribution from a single ground point's perspective.

Picks a ground point (center of all plots), finds which drone cameras capture it
within their field of view, and shows:
  - Panel A: 3D isometric view with the ground point and rays to all capturing cameras.
  - Panel B: hemispherical / fish-eye polar plot of VZA vs VAA from the point.
  - Panel C: VZA histogram showing the angular coverage distribution.

VZA = view zenith angle (0° = nadir/overhead, 90° = horizon), measured from the
ground point up to the camera. VAA = view azimuth angle (clockwise from north).
These follow the same convention as src/extract/camera.py:calculate_angles.

Camera FOV: MicaSense Altum multispectral bands have ~48° x 37° HxV FOV
(diagonal ~60°). A point is considered "captured" when the angle between the
camera boresight and the vector camera->point is less than half the diagonal FOV.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/runs/diagnostics/drone_positions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

TARGET_CRS = "EPSG:32632"
SOURCE_CRS = "EPSG:4326"

POLYGON_PATH = Path(
    "/run/media/davidem/data/ONCERCO/metadata/polygons/2024_oncerco_plot_polygons.gpkg"
)
CAMERAS_PATH = Path(
    "/run/media/davidem/data/ONCERCO/data/processed/2024/20240624_week3/"
    "metashape/20241206_week3_products_uav_data/20241206_week3_cameras.txt"
)
DEM_PATH = Path(
    "/run/media/davidem/data/ONCERCO/data/processed/2024/20240624_week3/"
    "metashape/20241206_week3_products_uav_data/20241206_week3_dem.tif"
)

YEAR = 2024
WEEK = "week3"
LABEL = f"{YEAR}_{WEEK}"

TRT_COLORS = {"trt": "#4C9F70", "no_trt": "#C44E52"}
TRT_LABELS = {"trt": "Treated (trt)", "no_trt": "Untreated (no_trt)"}

# MicaSense Altum multispectral: diagonal FOV ~60°, half-diagonal ~30°
HALF_FOV_DEG = 30.0

# Specific plot to analyze (ifz_id from the 2024 polygon file)
TARGET_PLOT_IFZ_ID = 90020


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"plot_point_viewing_angles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return log_path


def load_plots() -> gpd.GeoDataFrame:
    t0 = time.time()
    gdf = gpd.read_file(POLYGON_PATH).to_crs(TARGET_CRS)
    logging.info(f"[PHASE] load_plots: {time.time() - t0:.2f}s | {len(gdf)} plots")
    return gdf


def load_cameras() -> pl.DataFrame:
    t0 = time.time()
    df = pl.read_csv(CAMERAS_PATH, separator="\t", skip_rows=2, has_header=False)
    df.columns = [
        "PhotoID", "X", "Y", "Z", "Omega", "Phi", "Kappa",
        "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33",
    ]
    df = df.with_columns(
        [pl.col(c).cast(pl.Float64) for c in
         ["X", "Y", "Z", "Omega", "Phi", "Kappa",
          "r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"]]
    )
    before = df.height
    df = df.unique(subset=["PhotoID"], keep="first")
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    xs, ys = transformer.transform(df["X"].to_numpy(), df["Y"].to_numpy())
    df = df.with_columns(
        [pl.Series("X_utm", xs, dtype=pl.Float64), pl.Series("Y_utm", ys, dtype=pl.Float64)]
    )
    logging.info(f"[PHASE] load_cameras: {time.time() - t0:.2f}s | {before} -> {df.height} unique")
    return df


def ground_reference() -> float:
    t0 = time.time()
    with rasterio.open(DEM_PATH) as src:
        arr = src.read(1, masked=True)
        valid = arr.compressed()
        ground_z = float(np.nanmedian(valid)) if valid.size else float(np.nanmin(arr))
    logging.info(f"[PHASE] ground_reference: {time.time() - t0:.2f}s | ground_z={ground_z:.2f} m")
    return ground_z


def find_capturing_cameras(cams: pl.DataFrame, px: float, py: float, ground_z: float) -> pl.DataFrame:
    """For a ground point (px, py, ground_z), find cameras whose FOV contains it.

    Returns the subset of cameras that capture the point, with VZA and VAA
    columns computed from the point's perspective.
    """
    t0 = time.time()

    cam_x = cams["X_utm"].to_numpy()
    cam_y = cams["Y_utm"].to_numpy()
    cam_z = cams["Z"].to_numpy()

    # Vector from camera to ground point
    dx = px - cam_x
    dy = py - cam_y
    dz = ground_z - cam_z
    dist_3d = np.sqrt(dx**2 + dy**2 + dz**2)

    # Camera boresight direction in world: R @ [0, 0, -1]
    r31 = cams["r31"].to_numpy()
    r32 = cams["r32"].to_numpy()
    r33 = cams["r33"].to_numpy()
    bore_x = -r31
    bore_y = -r32
    bore_z = -r33

    # Normalized vector from camera to point
    norm_dx = dx / dist_3d
    norm_dy = dy / dist_3d
    norm_dz = dz / dist_3d

    # Angle between boresight and camera->point direction (dot product)
    cos_angle = bore_x * norm_dx + bore_y * norm_dy + bore_z * norm_dz
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    off_axis = np.degrees(np.arccos(cos_angle))

    # Capture mask: point within half-FOV cone
    captured = off_axis < HALF_FOV_DEG
    n_captured = int(captured.sum())
    logging.info(f"[PHASE] find_capturing_cameras: {time.time() - t0:.2f}s | "
                 f"{n_captured}/{len(cams)} cameras capture the point")

    # VZA from point perspective: arctan2(horizontal_dist, height_above_point)
    delta_z = cam_z - ground_z  # height of camera above ground point
    delta_x = cam_x - px        # camera minus ground point
    delta_y = cam_y - py
    horiz_dist = np.sqrt(delta_x**2 + delta_y**2)
    vza = np.degrees(np.arctan2(horiz_dist, delta_z))

    # VAA from point perspective: arctan2(delta_x, delta_y) -> clockwise from north
    vaa = np.degrees(np.arctan2(delta_x, delta_y))
    vaa = (vaa + 360.0) % 360.0

    cams = cams.with_columns([
        pl.Series("off_axis_deg", off_axis),
        pl.Series("vza", vza),
        pl.Series("vaa", vaa),
        pl.Series("dist_3d", dist_3d),
    ])
    result = cams.filter(pl.col("off_axis_deg") < HALF_FOV_DEG)
    return result


def plot_viewing_distribution(
    plots: gpd.GeoDataFrame,
    all_cams: pl.DataFrame,
    cap_cams: pl.DataFrame,
    px: float,
    py: float,
    ground_z: float,
) -> Path:
    t0 = time.time()

    # Coordinates
    cx_all = all_cams["X_utm"].to_numpy()
    cy_all = all_cams["Y_utm"].to_numpy()
    cz_all = all_cams["Z"].to_numpy() - ground_z

    cx_cap = cap_cams["X_utm"].to_numpy()
    cy_cap = cap_cams["Y_utm"].to_numpy()
    cz_cap = cap_cams["Z"].to_numpy() - ground_z
    vza_cap = cap_cams["vza"].to_numpy()
    vaa_cap = cap_cams["vaa"].to_numpy()

    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_polar = fig.add_subplot(gs[0, 1], projection="polar")

    # ---- Panel A: 3D isometric ----
    # Plot polygons (target plot highlighted, others faint)
    for _, row in plots.iterrows():
        xs_, ys_ = row.geometry.exterior.xy
        is_target = int(row["ifz_id"]) == TARGET_PLOT_IFZ_ID
        face = TRT_COLORS.get(row["trt"], "#cccccc")
        ax3d.add_collection3d(
            Poly3DCollection(
                [list(zip(xs_, ys_, [0.0] * len(xs_)))],
                facecolor=face,
                edgecolor="red" if is_target else "black",
                linewidth=2.5 if is_target else 0.6,
                alpha=0.85 if is_target else 0.35,
            )
        )

    # All cameras as small grey dots
    ax3d.scatter(cx_all, cy_all, cz_all, c="grey", s=2, alpha=0.15, edgecolors="none")

    # Capturing cameras: nadir (green) vs off-nadir (coral)
    nadir_mask = vza_cap < 15.0
    ax3d.scatter(
        cx_cap[nadir_mask], cy_cap[nadir_mask], cz_cap[nadir_mask],
        c="#4C9F70", s=18, alpha=0.9, edgecolors="black", linewidths=0.3,
        label=f"Nadir <15° (n={int(nadir_mask.sum())})", zorder=5,
    )
    ax3d.scatter(
        cx_cap[~nadir_mask], cy_cap[~nadir_mask], cz_cap[~nadir_mask],
        c="#C44E52", s=12, alpha=0.7, edgecolors="black", linewidths=0.3,
        label=f"Off-nadir >=15° (n={int((~nadir_mask).sum())})", zorder=4,
    )
    # Rays: nadir green, off-nadir faint coral
    for i in range(len(cx_cap)):
        ray_color = "#4C9F70" if vza_cap[i] < 15.0 else "#C44E52"
        ax3d.plot(
            [cx_cap[i], px], [cy_cap[i], py], [cz_cap[i], 0.0],
            color=ray_color, alpha=0.35 if vza_cap[i] < 15.0 else 0.18,
            linewidth=0.6 if vza_cap[i] < 15.0 else 0.4,
        )

    # Mark the ground point
    ax3d.scatter([px], [py], [0], color="red", s=80, marker="*", zorder=10,
                 edgecolors="black", linewidths=1.0)

    ax3d.set_xlabel("Easting (m)", fontsize=12)
    ax3d.set_ylabel("Northing (m)", fontsize=12)
    ax3d.set_zlabel("Altitude AGL (m)", fontsize=12)
    ax3d.view_init(elev=22, azim=45)
    ax3d.set_box_aspect((1, 1, 0.6))
    ax3d.set_title("3D view: cameras capturing the point", fontsize=13)
    ax3d.legend(loc="upper left", fontsize=11, framealpha=0.9)

    # ---- Panel B: polar / fish-eye plot ----
    # VZA as radial (0° at center = nadir, 90° at edge = horizon)
    # VAA as angular (0° = north, clockwise)
    theta = np.radians(vaa_cap)
    r = vza_cap  # radial = VZA in degrees

    # Nadir / off-nadir zones on the polar plot
    # Nadir disk: 0-15° (filled green), off-nadir ring: 15-90° (filled light red)
    n_theta = 180
    theta_ring = np.linspace(0, 2 * np.pi, n_theta)
    # Nadir disk (0 -> 15°)
    ax_polar.fill_between(
        theta_ring, 0, 15, color="#4C9F70", alpha=0.18, zorder=0,
    )
    # Off-nadir ring (15 -> 90°)
    ax_polar.fill_between(
        theta_ring, 15, 90, color="#C44E52", alpha=0.08, zorder=0,
    )
    # Boundary circles
    ax_polar.plot(theta_ring, np.full(n_theta, 15), color="#4C9F70",
                  linewidth=1.8, linestyle="--", zorder=1)
    ax_polar.plot(theta_ring, np.full(n_theta, 90), color="#5C677D",
                  linewidth=1.0, linestyle=":", alpha=0.6, zorder=1)

    sc_polar = ax_polar.scatter(theta, r, c=vza_cap, cmap="plasma", s=15, alpha=0.8,
                                edgecolors="black", linewidths=0.3, zorder=5)
    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)  # clockwise
    ax_polar.set_ylim(0, 90)
    ax_polar.set_yticks([0, 15, 30, 45, 60, 75, 90])
    ax_polar.set_yticklabels(["0°", "15°", "30°", "45°", "60°", "75°", "90°"], fontsize=10)
    ax_polar.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax_polar.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=11)
    ax_polar.set_title("Hemispherical view\n(from point looking up)", fontsize=13, pad=15)
    # Grid styling
    ax_polar.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.patch.set_alpha(0.0)
    for ax in (ax3d, ax_polar):
        ax.set_facecolor("none")
    ax_polar.set_facecolor("none")
    out_path = FIGURES_DIR / f"point_viewing_angles_{LABEL}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    logging.info(f"[PHASE] plot_viewing_distribution: {time.time() - t0:.2f}s | saved {out_path}")
    return out_path


def write_summary(
    figure_path: Path,
    all_cams: pl.DataFrame,
    cap_cams: pl.DataFrame,
    px: float,
    py: float,
    ground_z: float,
    elapsed_total: float,
) -> Path:
    vza = cap_cams["vza"].to_numpy()
    vaa = cap_cams["vaa"].to_numpy()
    summary_path = REPORTS_DIR / "point_viewing_angles_summary.md"
    lines = [
        f"# Viewing angle distribution from a ground point — {LABEL}",
        "",
        "## Inputs",
        f"- Polygons: `{POLYGON_PATH}`",
        f"- Cameras:  `{CAMERAS_PATH}`",
        f"- DEM (ground reference): `{DEM_PATH}`",
        f"- Target CRS: `{TARGET_CRS}`",
        f"- FOV half-cone threshold: `{HALF_FOV_DEG}°` (MicaSense Altum diagonal)",
        "",
        "## Ground point",
        f"- Easting: {px:.2f} m, Northing: {py:.2f} m (UTM 32N)",
        f"- Elevation: {ground_z:.2f} m (DEM median)",
        f"- Plot: ifz_id={TARGET_PLOT_IFZ_ID} (centroid of this plot polygon).",
        "",
        "## Results",
        f"- Total camera stations: **{all_cams.height}**",
        f"- Cameras capturing the point: **{cap_cams.height}** "
        f"({100*cap_cams.height/all_cams.height:.1f}%)",
        f"- VZA: min={np.nanmin(vza):.1f}°, median={np.nanmedian(vza):.1f}°, "
        f"mean={np.nanmean(vza):.1f}°, max={np.nanmax(vza):.1f}°",
        f"- VAA: min={np.nanmin(vaa):.1f}°, median={np.nanmedian(vaa):.1f}°, "
        f"max={np.nanmax(vaa):.1f}° (full range={np.nanmax(vaa)-np.nanmin(vaa):.1f}°)",
        "",
        "## Figure",
        f"![viewing angles]({figure_path.relative_to(ROOT)})",
        "",
        f"**Interpretation**: Of {all_cams.height} drone camera stations, "
        f"{cap_cams.height} capture this ground point. The hemispherical plot "
        f"shows the angular distribution of those views from the point's "
        f"perspective: VZA ranges from {np.nanmin(vza):.0f}° to {np.nanmax(vza):.0f}° "
        f"(median {np.nanmedian(vza):.0f}°), with {np.nanmax(vaa)-np.nanmin(vaa):.0f}° "
        f"azimuthal spread. This illustrates the multiangular coverage that "
        f"off-nadir imagery provides over a single ground location.",
        "",
        "## Outputs",
        f"- Figure: `{figure_path}`",
        f"- This report: `{summary_path}`",
        f"- Log: `outputs/archive/legacy_unscoped/logs/plot_point_viewing_angles_*.log`",
        "",
        "## Reproducibility",
        f"- Year/week: {LABEL}",
        f"- Source CRS: `{SOURCE_CRS}`, target CRS: `{TARGET_CRS}`",
        "- Cameras deduplicated by PhotoID (keep first row per image).",
        "- Ground reference = median of DEM valid pixels.",
        "- Point captured if angle(camera boresight, camera->point) < half-FOV.",
        f"- Boresight = R @ [0, 0, -1] (Metashape rotation convention).",
        f"- Total runtime: {elapsed_total:.2f}s",
        "",
    ]
    summary_path.write_text("\n".join(lines))
    return summary_path


def main() -> None:
    t_start = time.time()
    log_path = setup_logging()
    logging.info(f"=== plot_point_viewing_angles :: {LABEL} ===")
    logging.info(f"log: {log_path}")

    plots = load_plots()
    cams = load_cameras()
    ground_z = ground_reference()

    # Ground point = centroid of a single target plot
    target_row = plots[plots["ifz_id"] == TARGET_PLOT_IFZ_ID]
    if target_row.empty:
        raise ValueError(f"Plot ifz_id={TARGET_PLOT_IFZ_ID} not found in polygon file")
    target_plot = target_row.iloc[0]
    px = float(target_plot.geometry.centroid.x)
    py = float(target_plot.geometry.centroid.y)
    logging.info(
        f"Ground point: ({px:.2f}, {py:.2f}) UTM, z={ground_z:.2f} m | "
        f"plot ifz_id={TARGET_PLOT_IFZ_ID} trt={target_plot['trt']} cult={target_plot['cult']}"
    )

    cap_cams = find_capturing_cameras(cams, px, py, ground_z)

    figure_path = plot_viewing_distribution(plots, cams, cap_cams, px, py, ground_z)
    summary_path = write_summary(figure_path, cams, cap_cams, px, py, ground_z, time.time() - t_start)

    logging.info(f"summary: {summary_path}")
    logging.info(f"TOTAL runtime: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
