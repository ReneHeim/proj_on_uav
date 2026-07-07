"""Render slide-ready VZA bin sphere diagrams for nadir and multiangular views."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs/presentation_assets/vza_bin_spheres"
FIGURES_DIR = OUTPUT_ROOT / "figures"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/logs"

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
    PALETTE["teal"],
    "#49BDBD",
    PALETTE["gold"],
    "#F8D783",
    PALETTE["coral"],
    "#FF8B8B",
    PALETTE["navy"],
    "#35415E",
    PALETTE["grey"],
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


def draw_bands(ax, classes: list[tuple[int, int]], colors: list[str]) -> None:
    for (theta_low, theta_high), color in zip(classes, colors, strict=True):
        x, y, z = sphere_band(theta_low, theta_high)
        ax.plot_surface(
            x,
            y,
            z,
            color=color,
            alpha=0.82,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
        for theta in [theta_low, theta_high]:
            rx, ry, rz = ring(theta)
            ax.plot(rx, ry, rz, color="white", linewidth=1.15, alpha=0.95)


def add_bin_labels(ax, classes: list[tuple[int, int]], colors: list[str], side: str) -> None:
    x_anchor = 0.08 if side == "left" else 0.84
    ha = "right" if side == "left" else "left"
    y_positions = np.linspace(0.68, 0.30, len(classes)) if len(classes) > 1 else [0.62]
    for y_pos, ((theta_low, theta_high), color) in zip(
        y_positions, zip(classes, colors, strict=True), strict=True
    ):
        ax.text2D(
            x_anchor,
            y_pos,
            f"{theta_low}-{theta_high}°",
            transform=ax.transAxes,
            ha=ha,
            va="center",
            fontsize=9.5,
            color=color if color != PALETTE["gold"] else "#9A6A00",
            fontweight="bold",
        )


def write_plot() -> list[Path]:
    started = time.perf_counter()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
    draw_bands(axes[1], VZA_CLASSES, BIN_COLORS)
    add_bin_labels(axes[0], NADIR_CLASSES, [PALETTE["teal"]], side="left")
    add_bin_labels(axes[1], VZA_CLASSES, BIN_COLORS, side="right")

    axes[0].set_title(
        "Nadir-only bin", fontsize=22, fontweight="bold", color=PALETTE["navy"], pad=8
    )
    axes[1].set_title(
        "Multiangular bins", fontsize=22, fontweight="bold", color=PALETTE["navy"], pad=8
    )

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
        "Selected models use VZA-binned reflectance only; RAA and phase-angle bins are not included here.",
        ha="center",
        va="center",
        fontsize=12.5,
        color=PALETTE["grey"],
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.08, wspace=0.04)

    paths = [
        FIGURES_DIR / "vza_bin_spheres_nadir_vs_multiangular.png",
        FIGURES_DIR / "vza_bin_spheres_nadir_vs_multiangular.pdf",
        FIGURES_DIR / "vza_bin_spheres_nadir_vs_multiangular.svg",
    ]
    for path in paths:
        fig.savefig(path, dpi=300)
        logging.info("Wrote figure: %s", path)
    plt.close(fig)
    log_phase("write VZA bin sphere plot", started)
    return paths


def write_report(paths: list[Path], log_path: Path) -> Path:
    started = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = f"""## VZA Bin Sphere Diagram

This figure visualizes the VZA classes present in the cached distribution-feature tables. The selected compact multiangular models use VZA-binned reflectance only; RAA and phase-angle bins were tested separately and are not part of this selected feature family.

| Representation | VZA classes shown |
| --- | --- |
| Nadir-only | {", ".join(f"{lo}-{hi}°" for lo, hi in NADIR_CLASSES)} near-nadir interval; implemented as closest observed VZA class per plot/week/band/metric |
| Multiangular | {", ".join(f"{lo}-{hi}°" for lo, hi in VZA_CLASSES)} |

**Outputs**:
{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in paths)}

**Reproducibility**:

- Source for observed classes: `outputs/multiangular_distribution_feature_family/results/distribution_features_long_2024.csv`
- Nadir implementation: closest available observed VZA class, representing the near-nadir `0-15°` interval.
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
