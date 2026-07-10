"""Render slide-ready VZA bin sphere diagrams for nadir and multiangular views."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Wedge
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/deliverables/presentation/assets/vza_bin_spheres"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"
IMPORTANCE_PATH = (
    ROOT
    / "outputs/runs/analysis/severity/current/curve_only_offnadir_2024_to_2025/results/curve_only_offnadir_selected_features.csv"
)
FALLBACK_IMPORTANCE_PATH = (
    ROOT
    / "outputs/runs/analysis/severity/current/curve_only_functional_2024_to_2025/results/curve_only_functional_selected_features.csv"
)

VZA_CLASSES = [
    (0, 15),
    (15, 20),
    (20, 25),
    (25, 30),
    (30, 35),
    (35, 40),
    (40, 45),
    (45, 50),
    (50, 55),
]
NADIR_CLASSES = [(0, 15)]

PALETTE = {
    "navy": "#0B132B",
    "teal": "#00A6A6",
    "coral": "#FF6B6B",
    "gold": "#F6C85F",
    "background": "#F7F9FC",
    "grey": "#5C677D",
    "light_grey": "#E7ECF3",
}
BIN_COLORS = [
    "#C8CED8",
    "#49BDBD",
    PALETTE["gold"],
    "#F8D783",
    PALETTE["coral"],
    "#FF8B8B",
    PALETTE["navy"],
    "#35415E",
    PALETTE["grey"],
]
SOLID_BIN_COLORS = ["#C8CED8"] + [PALETTE["teal"]] * (len(VZA_CLASSES) - 1)
GRADIENT_BIN_COLORS = ["#C8CED8"] + [
    "#2166AC",
    "#3B82F6",
    "#73A9F5",
    "#B7D6F7",
    "#F7B7B7",
    "#F27E7E",
    "#E34A4A",
    "#B2182B",
]


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"plot_vza_bin_spheres_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    logging.info("Log file: %s", log_path)
    return log_path


def log_phase(name: str, started: float) -> None:
    logging.info("[PHASE] %s: %.1fs", name, time.perf_counter() - started)


def sphere_band(theta_low: float, theta_high: float, n_theta: int = 12, n_phi: int = 160):
    theta = np.deg2rad(np.linspace(theta_low, theta_high, n_theta))
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    return x, y, z


def ring(theta_deg: float, n_phi: int = 220):
    theta = np.deg2rad(theta_deg)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.full_like(phi, np.cos(theta))
    return x, y, z


def configure_axis(ax) -> None:
    ax.set_box_aspect((1, 1, 0.72))
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    ax.set_xlim(-0.92, 0.92)
    ax.set_ylim(-0.92, 0.92)
    ax.set_zlim(0.48, 1.05)
    ax.set_proj_type("ortho")


def configure_axis_top(ax) -> None:
    ax.set_box_aspect((1, 1, 0.72))
    ax.view_init(elev=90, azim=-90)
    ax.set_axis_off()
    ax.set_xlim(-0.92, 0.92)
    ax.set_ylim(-0.92, 0.92)
    ax.set_zlim(0.48, 1.05)
    ax.set_proj_type("ortho")


def configure_axis_isometric(ax) -> None:
    ax.set_box_aspect((1, 1, 0.72))
    ax.view_init(elev=34, azim=-45)
    ax.set_axis_off()
    ax.set_xlim(-0.92, 0.92)
    ax.set_ylim(-0.92, 0.92)
    ax.set_zlim(0.48, 1.05)
    ax.set_proj_type("ortho")


def draw_reference_dome(ax) -> None:
    x, y, z = sphere_band(0, 55, n_theta=28, n_phi=180)
    ax.plot_surface(
        x,
        y,
        z,
        color=PALETTE["light_grey"],
        alpha=0.18,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    for theta in [10, 20, 30, 40, 50, 55]:
        rx, ry, rz = ring(theta)
        ax.plot(rx, ry, rz, color="#B8C2CF", linewidth=0.7, alpha=0.75)

    # Central view vector.
    ax.plot([0, 0], [0, 0], [1.05, 0.55], color=PALETTE["navy"], linewidth=1.4, alpha=0.9)
    ax.scatter([0], [0], [1.055], color=PALETTE["navy"], s=18, depthshade=False)


def draw_bands(
    ax,
    classes: list[tuple[int, int]],
    colors: list[str],
    alpha: float = 0.82,
    bottom_to_top: bool = False,
) -> None:
    band_specs = list(zip(classes, colors, strict=True))
    if bottom_to_top:
        band_specs = band_specs[::-1]
    for (theta_low, theta_high), color in band_specs:
        x, y, z = sphere_band(theta_low, theta_high)
        ax.plot_surface(
            x,
            y,
            z,
            color=color,
            alpha=alpha,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        for theta in [theta_low, theta_high]:
            rx, ry, rz = ring(theta)
            ax.plot(rx, ry, rz, color="white", linewidth=1.15, alpha=0.95)


def load_angle_importance() -> dict[tuple[int, int], float]:
    path = IMPORTANCE_PATH if IMPORTANCE_PATH.exists() else FALLBACK_IMPORTANCE_PATH
    if not path.exists():
        logging.warning("Missing importance table: %s", path)
        return {}
    import pandas as pd

    table = pd.read_csv(path)
    feature_set = (
        "curve_only_vza_log_offnadir_no_10_15"
        if path == IMPORTANCE_PATH
        else "curve_only_vza_log"
    )
    table = table[
        (table["model"] == "current_hurdle_stability_top50_raw_positive")
        & (table["feature_set"] == feature_set)
        & (table["selected_for_final_model"].astype(bool))
    ]
    scores: dict[float, float] = {}
    for _, row in table.iterrows():
        match = re.search(r"__vza_(\d+\.\d+)$", str(row["feature"]))
        if not match:
            continue
        angle = float(match.group(1))
        score = float(row["selection_frequency"]) * float(row["mean_abs_elasticnet_coef"])
        scores[angle] = scores.get(angle, 0.0) + score
    if not scores:
        return {}
    total_score = sum(scores.values())
    out: dict[tuple[int, int], float] = {}
    for low, high in VZA_CLASSES:
        midpoint = (low + high) / 2
        out[(low, high)] = scores.get(midpoint, 0.0) / total_score
    logging.info("Loaded angle importance from %s", path)
    return out


def add_bin_labels(
    ax,
    classes: list[tuple[int, int]],
    colors: list[str],
    side: str,
    importance: dict[tuple[int, int], float] | None = None,
) -> None:
    x_anchor = 0.08 if side == "left" else 0.84
    ha = "right" if side == "left" else "left"
    y_positions = np.linspace(0.68, 0.30, len(classes)) if len(classes) > 1 else [0.62]
    if side == "right" and importance:
        ax.text2D(
            x_anchor,
            0.725,
            "VZA       imp.",
            transform=ax.transAxes,
            ha=ha,
            va="center",
            fontsize=8.6,
            color=PALETTE["grey"],
            fontweight="bold",
        )
    for y_pos, ((theta_low, theta_high), color) in zip(y_positions, zip(classes, colors, strict=True), strict=True):
        label = f"{theta_low}-{theta_high}°"
        if importance and side == "right" and theta_low == 0 and theta_high == 15:
            label = f"{label}   excl."
        elif importance:
            label = f"{label}   {importance.get((theta_low, theta_high), 0.0):.2f}"
        label_color = PALETTE["grey"] if (side == "right" and theta_low == 10 and theta_high == 15) else color
        ax.text2D(
            x_anchor,
            y_pos,
            label,
            transform=ax.transAxes,
            ha=ha,
            va="center",
            fontsize=9.5,
            color=label_color if label_color != PALETTE["gold"] else "#9A6A00",
            fontweight="bold",
        )


def write_plot_variant(
    suffix: str,
    colors: list[str],
    bottom_note: str,
    surface_alpha: float = 0.82,
) -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    importance = load_angle_importance()

    fig = plt.figure(figsize=(12.0, 6.4))
    fig.patch.set_facecolor("white")
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]

    for ax in axes:
        configure_axis(ax)
        draw_reference_dome(ax)

    draw_bands(axes[0], NADIR_CLASSES, [PALETTE["teal"]])
    draw_bands(axes[1], VZA_CLASSES, colors, alpha=surface_alpha)
    add_bin_labels(axes[0], NADIR_CLASSES, [PALETTE["teal"]], side="left")
    add_bin_labels(axes[1], VZA_CLASSES, colors, side="right", importance=importance)

    axes[0].set_title("Nadir-only bin", fontsize=22, fontweight="bold", color=PALETTE["navy"], pad=8)
    axes[1].set_title("Multiangular bins", fontsize=22, fontweight="bold", color=PALETTE["navy"], pad=8)

    fig.suptitle(
        "View Zenith Angle Bins Used for Reflectance Features",
        fontsize=22,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.965,
    )
    fig.text(
        0.5,
        0.055,
        bottom_note,
        ha="center",
        va="center",
        fontsize=12.5,
        color=PALETTE["grey"],
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.08, wspace=0.04)

    paths = [
        FIGURES_DIR / f"vza_bin_spheres_nadir_vs_multiangular{suffix}.png",
        FIGURES_DIR / f"vza_bin_spheres_nadir_vs_multiangular{suffix}.pdf",
        FIGURES_DIR / f"vza_bin_spheres_nadir_vs_multiangular{suffix}.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=300, transparent=path.suffix == ".png")
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    log_phase("write VZA bin sphere plot", started)
    return paths


def write_top_view_plot() -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    importance = load_angle_importance()

    fig = plt.figure(figsize=(12.0, 6.4))
    fig.patch.set_facecolor("white")
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]

    for ax in axes:
        configure_axis_top(ax)
        draw_reference_dome(ax)

    draw_bands(axes[0], NADIR_CLASSES, [PALETTE["teal"]], alpha=1.0)
    draw_bands(axes[1], VZA_CLASSES, GRADIENT_BIN_COLORS, alpha=1.0, bottom_to_top=True)
    add_bin_labels(axes[0], NADIR_CLASSES, [PALETTE["teal"]], side="left")
    add_bin_labels(axes[1], VZA_CLASSES, GRADIENT_BIN_COLORS, side="right", importance=importance)

    axes[0].set_title("Nadir-only bin", fontsize=22, fontweight="bold", color=PALETTE["navy"], pad=8)
    axes[1].set_title("Multiangular bins", fontsize=22, fontweight="bold", color=PALETTE["navy"], pad=8)

    fig.suptitle(
        "Top View of VZA Bins Used for Reflectance Features",
        fontsize=22,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.965,
    )
    fig.text(
        0.5,
        0.055,
        "Top-view 3D projection: VZA bins appear as concentric rings; 0-15° is excluded from the selected multiangular model.",
        ha="center",
        va="center",
        fontsize=12.0,
        color=PALETTE["grey"],
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.08, wspace=0.04)

    paths = [
        FIGURES_DIR / "vza_bin_spheres_topview_nadir_vs_multiangular_gradient_opaque_share.png",
        FIGURES_DIR / "vza_bin_spheres_topview_nadir_vs_multiangular_gradient_opaque_share.pdf",
        FIGURES_DIR / "vza_bin_spheres_topview_nadir_vs_multiangular_gradient_opaque_share.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=300, transparent=path.suffix == ".png")
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    log_phase("write top-view VZA bin sphere plot", started)
    return paths


def iso_project_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    azimuth = np.deg2rad(-45)
    screen_x = 0.88 * (x * np.cos(azimuth) + y * np.sin(azimuth))
    depth = -x * np.sin(azimuth) + y * np.cos(azimuth)
    screen_y = 0.13 + 0.36 * z + 0.12 * depth
    return screen_x, screen_y


def iso_projected_ring(theta_deg: float, n_phi: int = 260) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(theta_deg)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.full_like(phi, np.cos(theta))
    return iso_project_xyz(x, y, z)


def iso_band_polygon(theta_low: float, theta_high: float) -> np.ndarray:
    outer_x, outer_y = iso_projected_ring(theta_high)
    inner_x, inner_y = iso_projected_ring(theta_low)
    return np.column_stack(
        [
            np.concatenate([outer_x, inner_x[::-1]]),
            np.concatenate([outer_y, inner_y[::-1]]),
        ]
    )


def configure_iso_layer_axis(ax) -> None:
    ax.set_aspect("auto")
    ax.set_xlim(-0.98, 1.50)
    ax.set_ylim(0.12, 0.66)
    ax.axis("off")


def draw_iso_layer_reference(ax) -> None:
    for theta in [10, 20, 30, 40, 50, 55]:
        x, y = iso_projected_ring(theta)
        ax.plot(x, y, color="#D0DAE6", linewidth=0.9, zorder=1)
    x55, y55 = iso_projected_ring(55)
    ax.plot(x55, y55, color="#C8D2DF", linewidth=1.1, zorder=1)
    z_axis = np.linspace(np.cos(np.deg2rad(55)), 1.05, 80)
    axis_x, axis_y = iso_project_xyz(np.zeros_like(z_axis), np.zeros_like(z_axis), z_axis)
    ax.plot(axis_x, axis_y, color=PALETTE["navy"], linewidth=1.4, zorder=20)
    dot_x, dot_y = iso_project_xyz(np.array([0.0]), np.array([0.0]), np.array([1.08]))
    ax.scatter(dot_x, dot_y, color=PALETTE["navy"], s=20, zorder=21)


def draw_iso_layer_bands(ax, classes: list[tuple[int, int]], colors: list[str]) -> None:
    band_specs = list(zip(classes, colors, strict=True))[::-1]
    for index, ((theta_low, theta_high), color) in enumerate(band_specs, start=2):
        ax.add_patch(
            Polygon(
                iso_band_polygon(theta_low, theta_high),
                closed=True,
                facecolor=color,
                edgecolor="none",
                linewidth=0,
                alpha=1.0,
                zorder=index,
            )
        )

def add_iso_layer_labels(
    ax,
    classes: list[tuple[int, int]],
    colors: list[str],
) -> None:
    ax.text(
        0.77,
        0.80,
        "VZA bins",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=12.0,
        color=PALETTE["grey"],
        fontweight="bold",
    )
    y_positions = np.linspace(0.69, 0.28, len(classes))
    for y_pos, (theta_low, theta_high), color in zip(y_positions, classes, colors, strict=True):
        label_color = PALETTE["grey"] if theta_low == 0 and theta_high == 15 else color
        if theta_low == 0 and theta_high == 15:
            label = f"{theta_low}-{theta_high}° excl."
        else:
            label = f"{theta_low}-{theta_high}°"
        ax.text(
            0.77,
            y_pos,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=12.4,
            color=label_color,
            fontweight="bold",
        )


def write_isometric_plot() -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.4))
    fig.patch.set_facecolor("white")
    for ax in axes:
        configure_iso_layer_axis(ax)
        draw_iso_layer_reference(ax)

    draw_iso_layer_bands(axes[0], NADIR_CLASSES, [PALETTE["teal"]])
    draw_iso_layer_bands(axes[1], VZA_CLASSES, GRADIENT_BIN_COLORS)
    add_iso_layer_labels(axes[1], VZA_CLASSES, GRADIENT_BIN_COLORS)
    axes[0].text(
        -0.80,
        0.43,
        "0-15°",
        ha="right",
        va="center",
        fontsize=14.5,
        color=PALETTE["teal"],
        fontweight="bold",
    )

    axes[0].text(
        0.0,
        0.62,
        "Nadir-only bin",
        ha="center",
        va="center",
        fontsize=23,
        fontweight="bold",
        color=PALETTE["navy"],
    )
    axes[1].text(
        0.0,
        0.62,
        "Multiangular bins",
        ha="center",
        va="center",
        fontsize=23,
        fontweight="bold",
        color=PALETTE["navy"],
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04, wspace=0.04)

    paths = [
        FIGURES_DIR / "vza_bin_spheres_isometric_nadir_vs_multiangular_gradient_opaque_share.png",
        FIGURES_DIR / "vza_bin_spheres_isometric_nadir_vs_multiangular_gradient_opaque_share.pdf",
        FIGURES_DIR / "vza_bin_spheres_isometric_nadir_vs_multiangular_gradient_opaque_share.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=300, transparent=path.suffix == ".png", facecolor="none" if path.suffix == ".png" else None)
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    log_phase("write isometric VZA bin sphere plot", started)
    return paths


def angle_to_xy(angle_deg: float, radius: float = 1.0) -> tuple[float, float]:
    angle_rad = np.deg2rad(angle_deg)
    return radius * np.cos(angle_rad), radius * np.sin(angle_rad)


def add_2d_reference(ax) -> None:
    reference_angles = list(range(15, 60, 10)) + [55]
    for theta in reference_angles:
        right_angle = 90 - theta
        left_angle = 90 + theta
        for angle in [right_angle, left_angle]:
            x, y = angle_to_xy(angle, radius=1.02)
            ax.plot([0, x], [0, y], color="#D5DDE8", linewidth=0.7, zorder=0)
    ax.plot([0, 0], [0, 1.08], color=PALETTE["navy"], linewidth=1.6, zorder=4)
    ax.scatter([0], [1.08], s=22, color=PALETTE["navy"], zorder=5)


def draw_2d_vza_bins(ax, classes: list[tuple[int, int]], colors: list[str]) -> None:
    for (theta_low, theta_high), color in zip(classes, colors, strict=True):
        right = Wedge(
            (0, 0),
            1.0,
            90 - theta_high,
            90 - theta_low,
            facecolor=color,
            edgecolor="white",
            linewidth=1.2,
            alpha=1.0,
            zorder=2,
        )
        left = Wedge(
            (0, 0),
            1.0,
            90 + theta_low,
            90 + theta_high,
            facecolor=color,
            edgecolor="white",
            linewidth=1.2,
            alpha=1.0,
            zorder=2,
        )
        ax.add_patch(right)
        ax.add_patch(left)


def add_2d_importance_labels(
    ax,
    classes: list[tuple[int, int]],
    colors: list[str],
    importance: dict[tuple[int, int], float],
) -> None:
    ax.text(
        1.14,
        0.94,
        "VZA      imp.",
        ha="left",
        va="center",
        fontsize=9.2,
        color=PALETTE["grey"],
        fontweight="bold",
    )
    y_positions = np.linspace(0.82, -0.02, len(classes))
    for y_pos, (theta_low, theta_high), color in zip(y_positions, classes, colors, strict=True):
        label_color = PALETTE["grey"] if theta_low == 0 and theta_high == 15 else color
        if theta_low == 0 and theta_high == 15:
            label = f"{theta_low}-{theta_high}°   excl."
        else:
            label = f"{theta_low}-{theta_high}°   {importance.get((theta_low, theta_high), 0.0):.2f}"
        ax.text(
            1.14,
            y_pos,
            label,
            ha="left",
            va="center",
            fontsize=10.2,
            color=label_color,
            fontweight="bold",
        )


def configure_2d_axis(ax) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.12, 1.72)
    ax.set_ylim(-0.06, 1.18)
    ax.axis("off")


def write_2d_plot() -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    importance = load_angle_importance()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.4))
    fig.patch.set_facecolor("white")
    for ax in axes:
        configure_2d_axis(ax)
        add_2d_reference(ax)

    draw_2d_vza_bins(axes[0], NADIR_CLASSES, [PALETTE["teal"]])
    draw_2d_vza_bins(axes[1], VZA_CLASSES, GRADIENT_BIN_COLORS)
    add_2d_importance_labels(axes[1], VZA_CLASSES, GRADIENT_BIN_COLORS, importance)

    axes[0].set_title("Nadir-only bin", fontsize=21, fontweight="bold", color=PALETTE["navy"], pad=6)
    axes[1].set_title("Multiangular bins", fontsize=21, fontweight="bold", color=PALETTE["navy"], pad=6)
    axes[0].text(
        -0.02,
        0.58,
        "0-15°",
        ha="center",
        va="center",
        fontsize=14,
        color="white",
        fontweight="bold",
        zorder=6,
    )

    fig.suptitle(
        "2D View Zenith Angle Bins Used for Reflectance Features",
        fontsize=22,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.965,
    )
    fig.text(
        0.5,
        0.055,
        "Side-view schematic: 0-15° is excluded from the selected multiangular model; off-nadir bins are colored blue to red.",
        ha="center",
        va="center",
        fontsize=12.0,
        color=PALETTE["grey"],
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.83, bottom=0.10, wspace=0.10)

    paths = [
        FIGURES_DIR / "vza_bins_2d_nadir_vs_multiangular_gradient_opaque_share.png",
        FIGURES_DIR / "vza_bins_2d_nadir_vs_multiangular_gradient_opaque_share.pdf",
        FIGURES_DIR / "vza_bins_2d_nadir_vs_multiangular_gradient_opaque_share.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=300)
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    log_phase("write 2D VZA bin plot", started)
    return paths


def write_2d_strip_plot() -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    importance = load_angle_importance()

    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 2.2)
    ax.axis("off")

    ax.text(
        27.5,
        2.05,
        "2D View Zenith Angle Bins Used for Reflectance Features",
        ha="center",
        va="center",
        fontsize=22,
        color=PALETTE["navy"],
        fontweight="bold",
    )

    bar_height = 0.32
    nadir_y = 1.42
    multi_y = 0.78
    ax.text(0, nadir_y + 0.38, "Nadir-only bin", ha="left", va="center", fontsize=18, color=PALETTE["navy"], fontweight="bold")
    ax.text(0, multi_y + 0.38, "Multiangular bins", ha="left", va="center", fontsize=18, color=PALETTE["navy"], fontweight="bold")

    ax.add_patch(plt.Rectangle((0, nadir_y), 15, bar_height, color=PALETTE["teal"], ec="white", lw=1.4))
    ax.text(7.5, nadir_y + bar_height / 2, "0-15°", ha="center", va="center", fontsize=12.5, color="white", fontweight="bold")
    ax.plot([0, 55], [nadir_y - 0.12, nadir_y - 0.12], color="#D5DDE8", linewidth=1.1)

    for (low, high), color in zip(VZA_CLASSES, GRADIENT_BIN_COLORS, strict=True):
        ax.add_patch(plt.Rectangle((low, multi_y), high - low, bar_height, color=color, ec="white", lw=1.4))
        label = "excl." if low == 0 and high == 15 else f"{importance.get((low, high), 0.0):.2f}"
        text_color = PALETTE["grey"] if low == 0 and high == 15 else ("white" if high >= 40 or low == 15 else PALETTE["navy"])
        ax.text(
            (low + high) / 2,
            multi_y + bar_height / 2,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            color=text_color,
            fontweight="bold",
        )

    for tick in [0, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
        ax.plot([tick, tick], [multi_y - 0.06, multi_y - 0.12], color=PALETTE["grey"], linewidth=0.9)
        ax.text(tick, multi_y - 0.20, f"{tick}°", ha="center", va="top", fontsize=10.5, color=PALETTE["grey"])
    ax.text(
        27.5,
        0.16,
        "Numbers inside multiangular bins are normalized importance shares; 0-15° is excluded.",
        ha="center",
        va="center",
        fontsize=12,
        color=PALETTE["grey"],
    )

    paths = [
        FIGURES_DIR / "vza_bins_2d_strip_nadir_vs_multiangular_gradient_opaque_share.png",
        FIGURES_DIR / "vza_bins_2d_strip_nadir_vs_multiangular_gradient_opaque_share.pdf",
        FIGURES_DIR / "vza_bins_2d_strip_nadir_vs_multiangular_gradient_opaque_share.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=300)
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    log_phase("write 2D VZA strip plot", started)
    return paths


def dome_band_points(theta_low: float, theta_high: float, n_points: int = 80) -> np.ndarray:
    y_top = np.cos(np.deg2rad(theta_low))
    y_bottom = np.cos(np.deg2rad(theta_high))
    y_down = np.linspace(y_top, y_bottom, n_points)
    x_right = np.sqrt(np.clip(1.0 - y_down**2, 0.0, 1.0))
    y_up = y_down[::-1]
    x_left = -np.sqrt(np.clip(1.0 - y_up**2, 0.0, 1.0))
    return np.column_stack([np.concatenate([x_right, x_left]), np.concatenate([y_down, y_up])])


def configure_2d_dome_axis(ax) -> None:
    ax.set_aspect("auto")
    ax.set_xlim(-1.18, 1.70)
    ax.set_ylim(0.42, 1.12)
    ax.axis("off")


def draw_2d_dome_reference(ax) -> None:
    theta = np.deg2rad(np.linspace(-55, 55, 220))
    x = np.sin(theta)
    y = np.cos(theta)
    ax.plot(x, y, color="#C8D2DF", linewidth=1.2, zorder=1)
    ax.plot([-np.sin(np.deg2rad(55)), np.sin(np.deg2rad(55))], [np.cos(np.deg2rad(55)), np.cos(np.deg2rad(55))], color="#D8E0EA", linewidth=1.0, zorder=1)
    ax.plot([0, 0], [0.57, 1.05], color=PALETTE["navy"], linewidth=1.5, zorder=5)
    ax.scatter([0], [1.055], s=22, color=PALETTE["navy"], zorder=6)


def draw_2d_dome_bins(ax, classes: list[tuple[int, int]], colors: list[str]) -> None:
    for (theta_low, theta_high), color in zip(classes, colors, strict=True):
        polygon = Polygon(
            dome_band_points(theta_low, theta_high),
            closed=True,
            facecolor=color,
            edgecolor="white",
            linewidth=1.4,
            alpha=1.0,
            zorder=3,
        )
        ax.add_patch(polygon)


def add_2d_dome_importance_labels(
    ax,
    classes: list[tuple[int, int]],
    colors: list[str],
    importance: dict[tuple[int, int], float],
) -> None:
    x_anchor = 0.82
    ax.text(
        x_anchor,
        0.82,
        "VZA      imp.",
        ha="left",
        va="center",
        fontsize=9.2,
        color=PALETTE["grey"],
        fontweight="bold",
        transform=ax.transAxes,
    )
    y_positions = np.linspace(0.75, 0.28, len(classes))
    for y_pos, (theta_low, theta_high), color in zip(y_positions, classes, colors, strict=True):
        label_color = PALETTE["grey"] if theta_low == 0 and theta_high == 15 else color
        if theta_low == 0 and theta_high == 15:
            label = f"{theta_low}-{theta_high}°   excl."
        else:
            label = f"{theta_low}-{theta_high}°   {importance.get((theta_low, theta_high), 0.0):.2f}"
        ax.text(
            x_anchor,
            y_pos,
            label,
            ha="left",
            va="center",
            fontsize=10.0,
            color=label_color,
            fontweight="bold",
            transform=ax.transAxes,
        )


def write_2d_dome_plot() -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    importance = load_angle_importance()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.4))
    fig.patch.set_facecolor("white")
    for ax in axes:
        configure_2d_dome_axis(ax)
        draw_2d_dome_reference(ax)

    draw_2d_dome_bins(axes[0], NADIR_CLASSES, [PALETTE["teal"]])
    draw_2d_dome_bins(axes[1], VZA_CLASSES, GRADIENT_BIN_COLORS)
    add_2d_dome_importance_labels(axes[1], VZA_CLASSES, GRADIENT_BIN_COLORS, importance)

    axes[0].set_title("Nadir-only bin", fontsize=21, fontweight="bold", color=PALETTE["navy"], pad=8)
    axes[1].set_title("Multiangular bins", fontsize=21, fontweight="bold", color=PALETTE["navy"], pad=8)
    axes[0].text(
        -0.78,
        0.965,
        "0-15°",
        ha="right",
        va="center",
        fontsize=13.0,
        color=PALETTE["teal"],
        fontweight="bold",
        zorder=7,
    )

    fig.suptitle(
        "2D Dome View of VZA Bins Used for Reflectance Features",
        fontsize=22,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.965,
    )
    fig.text(
        0.5,
        0.070,
        "Flat dome projection: 0-15° is excluded from the selected multiangular model; off-nadir bins are colored blue to red.",
        ha="center",
        va="center",
        fontsize=12.0,
        color=PALETTE["grey"],
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.82, bottom=0.12, wspace=0.08)

    paths = [
        FIGURES_DIR / "vza_bins_2d_dome_nadir_vs_multiangular_gradient_opaque_share.png",
        FIGURES_DIR / "vza_bins_2d_dome_nadir_vs_multiangular_gradient_opaque_share.pdf",
        FIGURES_DIR / "vza_bins_2d_dome_nadir_vs_multiangular_gradient_opaque_share.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=300)
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    log_phase("write 2D VZA dome plot", started)
    return paths


def configure_polar_axis(ax) -> None:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-60)
    ax.set_thetamax(60)
    ax.set_ylim(0, 1.0)
    ax.set_yticklabels([])
    ax.set_xticks(np.deg2rad([-55, -45, -35, -25, -15, 0, 15, 25, 35, 45, 55]))
    ax.set_xticklabels(["55°", "45°", "35°", "25°", "15°", "0°", "15°", "25°", "35°", "45°", "55°"])
    ax.grid(color="#D5DDE8", linewidth=0.8)
    ax.spines["polar"].set_color("#C8D2DF")
    ax.spines["polar"].set_linewidth(1.0)


def draw_polar_bins(ax, classes: list[tuple[int, int]], colors: list[str]) -> None:
    for (theta_low, theta_high), color in zip(classes, colors, strict=True):
        for sign in [-1, 1]:
            center = sign * (theta_low + theta_high) / 2
            width = theta_high - theta_low
            ax.bar(
                np.deg2rad(center),
                1.0,
                width=np.deg2rad(width),
                bottom=0.0,
                align="center",
                color=color,
                edgecolor="white",
                linewidth=1.2,
                alpha=1.0,
            )
    ax.plot([0, 0], [0, 1.0], color=PALETTE["navy"], linewidth=1.6, zorder=5)


def write_2d_polar_plot() -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    importance = load_angle_importance()

    fig = plt.figure(figsize=(12.0, 6.4))
    fig.patch.set_facecolor("white")
    axes = [
        fig.add_subplot(1, 2, 1, projection="polar"),
        fig.add_subplot(1, 2, 2, projection="polar"),
    ]
    for ax in axes:
        configure_polar_axis(ax)

    draw_polar_bins(axes[0], NADIR_CLASSES, [PALETTE["teal"]])
    draw_polar_bins(axes[1], VZA_CLASSES, GRADIENT_BIN_COLORS)

    axes[0].set_title("Nadir-only bin", fontsize=21, fontweight="bold", color=PALETTE["navy"], pad=18)
    axes[1].set_title("Multiangular bins", fontsize=21, fontweight="bold", color=PALETTE["navy"], pad=18)
    axes[0].text(
        0,
        0.58,
        "0-15°",
        ha="center",
        va="center",
        fontsize=13.5,
        color="white",
        fontweight="bold",
    )

    label_text = "\n".join(
        f"{low}-{high}°   {'excl.' if low == 0 and high == 15 else f'{importance.get((low, high), 0.0):.2f}'}"
        for low, high in VZA_CLASSES
    )
    axes[1].text(
        np.deg2rad(63),
        0.78,
        "VZA      imp.\n" + label_text,
        ha="left",
        va="top",
        fontsize=9.6,
        color=PALETTE["grey"],
        fontweight="bold",
    )

    fig.suptitle(
        "Polar View of VZA Bins Used for Reflectance Features",
        fontsize=22,
        fontweight="bold",
        color=PALETTE["navy"],
        y=0.965,
    )
    fig.text(
        0.5,
        0.060,
        "Polar schematic: angle from nadir is mirrored left and right; 0-15° is excluded from the selected multiangular model.",
        ha="center",
        va="center",
        fontsize=12.0,
        color=PALETTE["grey"],
    )
    fig.subplots_adjust(left=0.04, right=0.94, top=0.82, bottom=0.12, wspace=0.28)

    paths = [
        FIGURES_DIR / "vza_bins_2d_polar_nadir_vs_multiangular_gradient_opaque_share.png",
        FIGURES_DIR / "vza_bins_2d_polar_nadir_vs_multiangular_gradient_opaque_share.pdf",
        FIGURES_DIR / "vza_bins_2d_polar_nadir_vs_multiangular_gradient_opaque_share.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=300)
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    log_phase("write 2D VZA polar plot", started)
    return paths


def write_plot() -> list[Path]:
    paths: list[Path] = []
    paths.extend(
        write_plot_variant(
            "",
            BIN_COLORS,
            "Right-side importance shares sum to 1.00 after excluding the 0-15° bin from the selected model.",
            surface_alpha=0.82,
        )
    )
    paths.extend(
        write_plot_variant(
            "_solid_share",
            SOLID_BIN_COLORS,
            "Solid-color version: right-side importance shares sum to 1.00 after excluding the 0-15° bin.",
            surface_alpha=0.95,
        )
    )
    paths.extend(
        write_plot_variant(
            "_transparent_share",
            SOLID_BIN_COLORS,
            "Transparent version: right-side importance shares sum to 1.00 after excluding the 0-15° bin.",
            surface_alpha=0.36,
        )
    )
    paths.extend(
        write_plot_variant(
            "_gradient_share",
            GRADIENT_BIN_COLORS,
            "Blue-to-red version: right-side importance shares sum to 1.00 after excluding the 0-15° bin.",
            surface_alpha=0.88,
        )
    )
    paths.extend(
        write_plot_variant(
            "_gradient_opaque_share",
            GRADIENT_BIN_COLORS,
            "Opaque blue-to-red version: right-side importance shares sum to 1.00 after excluding the 0-15° bin.",
            surface_alpha=1.0,
        )
    )
    paths.extend(write_2d_plot())
    paths.extend(write_2d_strip_plot())
    paths.extend(write_2d_dome_plot())
    paths.extend(write_2d_polar_plot())
    paths.extend(write_top_view_plot())
    paths.extend(write_isometric_plot())
    return paths


def write_report(paths: list[Path], log_path: Path) -> Path:
    started = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = f"""## VZA Bin Sphere Diagram

This figure visualizes the VZA classes present in the cached distribution-feature tables. The selected compact multiangular models use VZA-binned reflectance only; RAA and phase-angle bins were tested separately and are not part of this selected feature family.

| Representation | VZA classes shown |
| --- | --- |
| Nadir-only | {", ".join(f"{lo}-{hi}°" for lo, hi in NADIR_CLASSES)} near-nadir interval; implemented as closest observed VZA class per plot/week/band/metric |
| Multiangular | {", ".join(f"{lo}-{hi}°" for lo, hi in VZA_CLASSES)}; the near-nadir 0-15° bin is grayed out and excluded on the multiangular sphere |

**Outputs**:
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in paths)}

**Reproducibility**:

- Source for observed classes: `outputs/runs/analysis/severity/future/compact_distribution_feature_family/results/distribution_features_long_2024.csv`
- Nadir implementation: closest available observed VZA class, representing the near-nadir `0-15°` interval.
- Right-side importance: per-angle share of `selection_frequency * mean_abs_elasticnet_coef` from `outputs/runs/analysis/severity/current/curve_only_offnadir_2024_to_2025/results/curve_only_offnadir_selected_features.csv` for `current_hurdle_stability_top50_raw_positive` with `curve_only_vza_log_offnadir_no_10_15`; values sum to 1.00 across included off-nadir bins and the 0-15° bin is excluded.
- Selected model geometry: VZA only. RAA and phase-angle bins are excluded from this figure because they are not used by the selected compact multiangular feature set.
- Palette: Navy `{PALETTE["navy"]}`, Teal `{PALETTE["teal"]}`, Coral `{PALETTE["coral"]}`, Gold `{PALETTE["gold"]}`
- Log: `{log_path.relative_to(ROOT)}`
"""
    path = REPORTS_DIR / "vza_bin_spheres_summary.md"
    path.write_text(report, encoding="utf-8")
    logging.info("Wrote report: %s", path)
    log_phase("write VZA bin sphere report", started)
    return path


def main() -> None:
    log_path = setup_logging()
    paths = write_plot()
    write_report(paths, log_path)


if __name__ == "__main__":
    main()
