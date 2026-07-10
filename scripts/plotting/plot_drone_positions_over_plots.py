"""Plot drone camera positions over the ONCERCO field plots.

Builds a three-panel figure for a given week:
  - Panel A: top-down map with plot polygons colored by treatment, overlaid
    with the drone camera centers colored by flight altitude.
  - Panel B: side profile (X-Z) with plot footprints at ground level and the
    drone camera positions shown above ground at their true flight height.
  - Panel C: isometric 3D view with plot polygons at ground level and the
    full drone camera point cloud hovering above in real metric coordinates.

This gives the literal "plots at the bottom, drone point cloud above" view
requested for exploring the dataset.

Inputs (defaults target 2024 week3):
  - 2024 plot polygons (EPSG:4326):  metadata/polygons/2024_oncerco_plot_polygons.gpkg
  - Metashape cameras export (EPSG:4326): data/processed/2024/20240624_week3/.../20241206_week3_cameras.txt
  - DEM (EPSG:4326): used only to anchor the ground reference elevation.

Both plots and cameras are reprojected to EPSG:32632 (UTM 32N, meters) so
distances and altitudes are in real metric units.
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

# Treatment color scheme
TRT_COLORS = {"trt": "#4C9F70", "no_trt": "#C44E52"}
TRT_LABELS = {"trt": "Treated (trt)", "no_trt": "Untreated (no_trt)"}

# MicaSense Altum 5-band viewing-ray colors (band index 1..5)
BAND_COLORS = {
    1: "#1f77b4",  # Blue
    2: "#2ca02c",  # Green
    3: "#d62728",  # Red
    4: "#9467bd",  # Red Edge
    5: "#ff7f0e",  # NIR
}
BAND_NAMES = {1: "Blue", 2: "Green", 3: "Red", 4: "Red Edge", 5: "NIR"}
# Number of sample captures whose 5-band viewing rays are drawn
N_SAMPLE_CAPTURES = 6


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"plot_drone_positions_over_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return log_path


def load_plots() -> gpd.GeoDataFrame:
    t0 = time.time()
    gdf = gpd.read_file(POLYGON_PATH)
    gdf = gdf.to_crs(TARGET_CRS)
    logging.info(
        f"[PHASE] load_plots: {time.time() - t0:.2f}s | {len(gdf)} plots, CRS={gdf.crs}"
    )
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
    # Dedupe by PhotoID: each image (5 bands) appears multiple times. Keep first.
    before = df.height
    df = df.unique(subset=["PhotoID"], keep="first")
    # Parse base capture id and band index from PhotoID (e.g. IMG_0468_3 -> IMG_0468, 3)
    df = df.with_columns(
        pl.col("PhotoID").str.replace(r"_\d+$", "").alias("base"),
        pl.col("PhotoID").str.extract(r"_(\d+)$", 1).cast(pl.Int32).alias("band"),
    )
    logging.info(
        f"[PHASE] load_cameras: {time.time() - t0:.2f}s | "
        f"{before} rows -> {df.height} unique PhotoIDs"
    )
    return df


def reproject_cameras(df: pl.DataFrame) -> pl.DataFrame:
    t0 = time.time()
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    xs, ys = transformer.transform(df["X"].to_numpy(), df["Y"].to_numpy())
    df = df.with_columns(
        [pl.Series("X_utm", xs, dtype=pl.Float64), pl.Series("Y_utm", ys, dtype=pl.Float64)]
    )
    logging.info(f"[PHASE] reproject_cameras: {time.time() - t0:.2f}s")
    return df


def ground_reference() -> float:
    """Return a representative ground elevation (m) from the DEM median."""
    t0 = time.time()
    with rasterio.open(DEM_PATH) as src:
        arr = src.read(1, masked=True)
        # Filter nodata
        valid = arr.compressed()
        ground_z = float(np.nanmedian(valid)) if valid.size else float(np.nanmin(arr))
    logging.info(f"[PHASE] ground_reference: {time.time() - t0:.2f}s | ground_z={ground_z:.2f} m")
    return ground_z


def plot_figure(plots: gpd.GeoDataFrame, cams: pl.DataFrame, ground_z: float) -> Path:
    t0 = time.time()
    cx = cams["X_utm"].to_numpy()
    cy = cams["Y_utm"].to_numpy()
    cz = cams["Z"].to_numpy()
    cz_agl = cz - ground_z  # altitude above ground

    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1.2])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])
    ax_iso = fig.add_subplot(gs[0, 2], projection="3d")
    fig.suptitle(
        f"Drone camera positions over ONCERCO plots — {LABEL}\n"
        f"{cams.height} camera stations | ground elevation ~{ground_z:.0f} m",
        fontsize=13,
    )

    # ---------- Panel A: top-down ----------
    for trt_val, color in TRT_COLORS.items():
        sub = plots[plots["trt"] == trt_val]
        sub.plot(ax=ax_top, color=color, edgecolor="black", linewidth=1.2, alpha=0.75)
        for _, row in sub.iterrows():
            c = row.geometry.centroid
            ax_top.text(
                c.x, c.y, str(int(row["ifz_id"])), ha="center", va="center",
                fontsize=6, color="black", weight="bold",
            )

    sc = ax_top.scatter(
        cx, cy, c=cz_agl, cmap="viridis", s=6, alpha=0.55, edgecolors="none", zorder=3
    )
    cbar = fig.colorbar(sc, ax=ax_top, fraction=0.046, pad=0.04)
    cbar.set_label("Altitude above ground (m)")

    ax_top.set_title("Top-down view (UTM 32N)")
    ax_top.set_xlabel("Easting (m)")
    ax_top.set_ylabel("Northing (m)")
    ax_top.set_aspect("equal")
    ax_top.grid(True, linestyle=":", alpha=0.4)

    trt_handles = [
        Patch(facecolor=TRT_COLORS["trt"], edgecolor="black", label=TRT_LABELS["trt"]),
        Patch(facecolor=TRT_COLORS["no_trt"], edgecolor="black", label=TRT_LABELS["no_trt"]),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=6, label="Drone camera", linestyle="None"),
    ]
    ax_top.legend(handles=trt_handles, loc="upper right", fontsize=8, framealpha=0.9)

    # ---------- Panel B: side profile (X-Z) ----------
    # Plot footprints drawn as a filled strip at ground level, colored by trt.
    strip_h = 1.0  # visual thickness of the ground strip (meters)
    for trt_val, color in TRT_COLORS.items():
        sub = plots[plots["trt"] == trt_val]
        for _, row in sub.iterrows():
            poly = row.geometry
            xs = np.array(poly.exterior.xy[0])
            # Draw the plot footprint as a vertical band in X at ground level
            xmin, xmax = xs.min(), xs.max()
            ax_side.fill_between(
                [xmin, xmax], 0, strip_h, color=color, edgecolor="black",
                linewidth=0.8, alpha=0.75, zorder=2,
            )
            ax_side.text(
                (xmin + xmax) / 2, strip_h * 0.5, str(int(row["ifz_id"])),
                ha="center", va="center", fontsize=5.5, color="black", weight="bold",
            )

    # Ground line
    ax_side.axhline(0, color="#5C677D", linewidth=1.0, zorder=1)

    # Drone points above ground
    sc2 = ax_side.scatter(
        cx, cz_agl, c=cz_agl, cmap="viridis", s=6, alpha=0.55,
        edgecolors="none", zorder=3,
    )
    cbar2 = fig.colorbar(sc2, ax=ax_side, fraction=0.046, pad=0.04)
    cbar2.set_label("Altitude above ground (m)")

    ax_side.set_title("Side profile (X vs altitude)")
    ax_side.set_xlabel("Easting (m)")
    ax_side.set_ylabel("Altitude above ground (m)")
    ax_side.grid(True, linestyle=":", alpha=0.4)
    # Pad y so points sit clearly above the ground strip
    y_max = np.nanmax(cz_agl) * 1.1
    ax_side.set_ylim(-y_max * 0.05, y_max)
    ax_side.legend(handles=trt_handles, loc="upper right", fontsize=8, framealpha=0.9)

    # ---------- Panel C: isometric 3D view ----------
    # Plot polygons at ground level (z=0) as 3D faces.
    for trt_val, color in TRT_COLORS.items():
        sub = plots[plots["trt"] == trt_val]
        polys = []
        for _, row in sub.iterrows():
            xs, ys = row.geometry.exterior.xy
            verts = [list(zip(xs, ys, [0.0] * len(xs)))]
            polys.append(verts[0])
        ax_iso.add_collection3d(
            Poly3DCollection(
                polys, facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.75
            )
        )

    # Drone point cloud above ground.
    sc3 = ax_iso.scatter(
        cx, cy, cz_agl, c=cz_agl, cmap="viridis", s=5, alpha=0.5, edgecolors="none"
    )
    cbar3 = fig.colorbar(sc3, ax=ax_iso, fraction=0.046, pad=0.10, shrink=0.7)
    cbar3.set_label("Altitude above ground (m)")

    ax_iso.set_title("Isometric 3D view")
    ax_iso.set_xlabel("Easting (m)")
    ax_iso.set_ylabel("Northing (m)")
    ax_iso.set_zlabel("Altitude AGL (m)")
    # Isometric-like viewing angle.
    ax_iso.view_init(elev=22, azim=45)
    # Equalize XY aspect so plot blocks keep their real shape.
    ax_iso.set_box_aspect((1, 1, 0.6))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = FIGURES_DIR / f"drone_positions_over_plots_{LABEL}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[PHASE] plot_figure: {time.time() - t0:.2f}s | saved {out_path}")
    return out_path


def compute_viewing_rays(cams: pl.DataFrame, plots: gpd.GeoDataFrame, ground_z: float, n: int) -> list:
    """Select n sample captures near the plot center and compute 5-band viewing rays.

    Returns a list of dicts: {base, band, cam_xyz (np.array), ground_xyz (np.array)}.
    Each ray goes from the camera position down to where it intersects the ground
    plane (z_agl = 0) along the camera viewing direction (Metashape convention:
    camera local -Z, world_dir = R @ [0,0,-1]).
    """
    t0 = time.time()
    # Plot center in UTM
    cx0 = float(plots.geometry.centroid.x.mean())
    cy0 = float(plots.geometry.centroid.y.mean())

    # Distance of each camera to plot center
    dist = np.hypot(cams["X_utm"].to_numpy() - cx0, cams["Y_utm"].to_numpy() - cy0)
    cams = cams.with_columns(pl.Series("dist_center", dist))

    # Keep only bands 1..5 (Altum also has a 6th thermal band we skip).
    cams5 = cams.filter(pl.col("band").is_in([1, 2, 3, 4, 5]))

    # For each base capture, count bands present and mean distance to center.
    agg = (
        cams5.group_by("base")
        .agg(
            pl.len().alias("n_bands"),
            pl.col("dist_center").mean().alias("mean_dist"),
        )
        .filter(pl.col("n_bands") == 5)
        .sort("mean_dist")
    )
    sample_bases = agg["base"].head(n).to_list()
    logging.info(f"[RAYS] selected {len(sample_bases)} sample captures: {sample_bases}")

    rays = []
    sub = cams5.filter(pl.col("base").is_in(sample_bases))
    for row in sub.iter_rows(named=True):
        R = np.array([
            [row["r11"], row["r12"], row["r13"]],
            [row["r21"], row["r22"], row["r23"]],
            [row["r31"], row["r32"], row["r33"]],
        ])
        view_dir = R @ np.array([0.0, 0.0, -1.0])  # pointing down in world
        cam_xyz = np.array([row["X_utm"], row["Y_utm"], row["Z"] - ground_z])
        # Intersect with ground plane z_agl = 0
        if view_dir[2] >= 0:
            continue
        t_hit = -cam_xyz[2] / view_dir[2]
        ground_xyz = cam_xyz + t_hit * view_dir
        rays.append({
            "base": row["base"],
            "band": int(row["band"]),
            "cam_xyz": cam_xyz,
            "ground_xyz": ground_xyz,
        })
    logging.info(f"[PHASE] compute_viewing_rays: {time.time() - t0:.2f}s | {len(rays)} rays")
    return rays


def plot_isometric_hires(plots: gpd.GeoDataFrame, cams: pl.DataFrame, ground_z: float) -> Path:
    """Standalone high-resolution isometric 3D view."""
    t0 = time.time()
    cx = cams["X_utm"].to_numpy()
    cy = cams["Y_utm"].to_numpy()
    cz = cams["Z"].to_numpy()
    cz_agl = cz - ground_z

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle(
        f"Drone camera positions over ONCERCO plots — {LABEL} (isometric)\n"
        f"{cams.height} camera stations | ground elevation ~{ground_z:.0f} m | "
        f"altitude AGL {np.nanmin(cz_agl):.1f}-{np.nanmax(cz_agl):.1f} m",
        fontsize=14,
    )

    # Plot polygons at ground level (z=0) as 3D faces.
    label_z = 2.5
    for trt_val, color in TRT_COLORS.items():
        sub = plots[plots["trt"] == trt_val]
        polys = []
        for _, row in sub.iterrows():
            xs, ys = row.geometry.exterior.xy
            polys.append(list(zip(xs, ys, [0.0] * len(xs))))
            c = row.geometry.centroid
            ax.text(
                c.x, c.y, label_z, str(int(row["ifz_id"])), ha="center", va="center",
                fontsize=10, color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.75, edgecolor="none"),
                zorder=10,
            )
        ax.add_collection3d(
            Poly3DCollection(
                polys, facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.85
            )
        )

    # Drone point cloud above ground.
    sc = ax.scatter(
        cx, cy, cz_agl, c=cz_agl, cmap="viridis", s=8, alpha=0.6, edgecolors="none"
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.10, shrink=0.6)
    cbar.set_label("Altitude above ground (m)", fontsize=11)

    ax.set_xlabel("Easting (m)", fontsize=11)
    ax.set_ylabel("Northing (m)", fontsize=11)
    ax.set_zlabel("Altitude AGL (m)", fontsize=11)
    ax.view_init(elev=22, azim=45)
    ax.set_box_aspect((1, 1, 0.7))

    trt_handles = [
        Patch(facecolor=TRT_COLORS["trt"], edgecolor="black", label=TRT_LABELS["trt"]),
        Patch(facecolor=TRT_COLORS["no_trt"], edgecolor="black", label=TRT_LABELS["no_trt"]),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=7, label="Drone camera", linestyle="None"),
    ]
    ax.legend(handles=trt_handles, loc="upper left", fontsize=9, framealpha=0.9)

    out_path = FIGURES_DIR / f"drone_positions_isometric_{LABEL}.png"
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"[PHASE] plot_isometric_hires: {time.time() - t0:.2f}s | saved {out_path}")
    return out_path


def write_summary(
    figure_path: Path,
    iso_path: Path,
    plots: gpd.GeoDataFrame,
    cams: pl.DataFrame,
    ground_z: float,
    elapsed_total: float,
) -> Path:
    cz_agl = cams["Z"].to_numpy() - ground_z
    summary_path = REPORTS_DIR / "drone_positions_over_plots_summary.md"
    lines = [
        f"# Drone positions over ONCERCO plots — {LABEL}",
        "",
        "## Inputs",
        f"- Polygons: `{POLYGON_PATH}`",
        f"- Cameras:  `{CAMERAS_PATH}`",
        f"- DEM (ground reference): `{DEM_PATH}`",
        f"- Target CRS: `{TARGET_CRS}`",
        "",
        "## Dataset summary",
        f"- Plots: **{len(plots)}** (treated={(plots['trt']=='trt').sum()}, "
        f"untreated={(plots['trt']=='no_trt').sum()})",
        f"- Drone camera stations (unique PhotoIDs): **{cams.height}**",
        f"- Ground elevation (DEM median): **{ground_z:.2f} m**",
        f"- Drone altitude above ground: "
        f"min={np.nanmin(cz_agl):.1f} m, median={np.nanmedian(cz_agl):.1f} m, "
        f"max={np.nanmax(cz_agl):.1f} m",
        "",
        "## Figures",
        f"### 3-panel overview",
        f"![drone positions]({figure_path.relative_to(ROOT)})",
        "",
        f"### High-resolution isometric view (400 dpi)",
        f"![isometric]({iso_path.relative_to(ROOT)})",
        "",
        f"**Interpretation**: The top-down panel shows the two plot blocks "
        f"(treated vs untreated, 24 plots total) with the dense drone flight "
        f"grid of {cams.height} camera stations overlaid. The side profile "
        f"shows the drone point cloud hovering above the plot footprints at "
        f"~{np.nanmedian(cz_agl):.0f} m median altitude above ground. The "
        f"isometric 3D panel combines both into a single perspective view, "
        f"illustrating the multiangular viewing geometry of the acquisition.",
        "",
        "## Outputs",
        f"- 3-panel figure: `{figure_path}`",
        f"- Isometric high-res figure: `{iso_path}`",
        f"- This report: `{summary_path}`",
        f"- Log: `{LOGS_DIR}/plot_drone_positions_over_plots_*.log`",
        "",
        "## Reproducibility",
        f"- Year/week: {LABEL}",
        f"- Source CRS: `{SOURCE_CRS}`, target CRS: `{TARGET_CRS}`",
        "- Cameras deduplicated by PhotoID (keep first row per image).",
        "- Ground reference = median of DEM valid pixels.",
        f"- Total runtime: {elapsed_total:.2f}s",
        "",
    ]
    summary_path.write_text("\n".join(lines))
    return summary_path


def main() -> None:
    t_start = time.time()
    log_path = setup_logging()
    logging.info(f"=== plot_drone_positions_over_plots :: {LABEL} ===")
    logging.info(f"log: {log_path}")

    plots = load_plots()
    cams = load_cameras()
    cams = reproject_cameras(cams)
    ground_z = ground_reference()

    figure_path = plot_figure(plots, cams, ground_z)
    iso_path = plot_isometric_hires(plots, cams, ground_z)
    summary_path = write_summary(figure_path, iso_path, plots, cams, ground_z, time.time() - t_start)

    logging.info(f"summary: {summary_path}")
    logging.info(f"TOTAL runtime: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
