import logging
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d


def _kde1d_fast(
    v: np.ndarray,
    x_grid: np.ndarray,
    bw: float | None = None,
    bins: int = 1024,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """
    Approximate 1D KDE efficiently via histogram + Gaussian smoothing.

    Steps:
    1) Bin values into a fine histogram.
    2) Smooth counts with gaussian_filter1d using sigma derived from bandwidth.
    3) Interpolate smoothed density to x_grid and normalize to integrate to ~1.

    Args:
        v: 1D array of finite samples.
        x_grid: Points where the PDF should be evaluated.
        bw: Bandwidth in data units. If None, use Scott's rule.
        bins: Number of histogram bins for the smoothing grid.
        vmin, vmax: Optional clipping range. If None, inferred from data.

    Returns:
        y_pdf evaluated at x_grid (approximately normalized).
    """
    v = v[np.isfinite(v)]
    if v.size < 5:
        return np.zeros_like(x_grid)

    # Range and histogram grid
    lo = np.min(v) if vmin is None else float(vmin)
    hi = np.max(v) if vmax is None else float(vmax)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x_grid)

    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(v, bins=edges)

    # Bandwidth: Scott's rule if not provided
    if bw is None:
        s = np.std(v)
        n = v.size
        # Scott's rule: bw = 1.06 * s * n^(-1/5); fallback if s==0
        bw = 1.06 * (s if s > 0 else (hi - lo) / 6.0) * (n ** (-1.0 / 5.0))
        if bw <= 0 or not np.isfinite(bw):
            bw = max((hi - lo) / 100.0, 1e-12)

    # Convert bandwidth to sigma in bins
    bin_width = centers[1] - centers[0]
    sigma_bins = max(bw / bin_width, 1e-6)

    # Smooth counts
    smooth = gaussian_filter1d(counts.astype(float), sigma=sigma_bins, mode="nearest")

    # Convert to density (divide by N and bin width)
    density_centers = smooth / (v.size * bin_width)

    # Interpolate to requested x_grid
    y_pdf = np.interp(x_grid, centers, density_centers, left=0.0, right=0.0)

    # Normalize lightly to ensure area ≈ 1 over [lo, hi]
    area = np.trapezoid(y_pdf, x_grid)
    if area > 0 and np.isfinite(area):
        y_pdf = y_pdf / area

    return y_pdf


def angle_kde_plot(
    df,
    band: str,
    bins: List[Tuple[int, int]],
    angle: str,
    xlim: Optional[Tuple[float, float]],
    points: int,
    linewidth: float,
    colors: Optional[List[str]],
    dpi: int,
    out=None,

) -> None:
    try:
        df = df.drop_nulls().drop_nans()
        if xlim is not None:
            x_min, x_max = xlim
        else:
            x_min = df.select(pl.col(band).quantile(0.01)).item()
            x_max = df.select(pl.col(band).quantile(0.98)).item()
        x_grid = np.linspace(x_min, x_max, int(points))
        fig_k, ax_k = plt.subplots(figsize=(10, 6), dpi=dpi)

        cycle = colors or plt.rcParams["axes.prop_cycle"].by_key().get("color", None)

        i = 0
        for bin in bins:

            filtered_df = df.filter(
                pl.col(angle) > bin[0],
                pl.col(angle) < bin[1],
            )

            v = filtered_df[band].to_numpy()
            # Restrict to finite values within [x_min, x_max] to match chart limits
            v = v[np.isfinite(v)]
            v = v[(v >= x_min) & (v <= x_max)]
            if v.size < 5:
                continue

            # Fast KDE via histogram smoothing (orders of magnitude faster than gaussian_kde on big data)
            y_pdf = _kde1d_fast(
                v,
                x_grid,
                bw=None,  # or set a float bandwidth in data units (e.g., 0.01)
                bins=5128,  # can lower to 512 for even faster
                vmin=x_min,
                vmax=x_max,
            )

            color = None
            if cycle and i < len(cycle):
                color = cycle[i]
            ax_k.plot(x_grid, y_pdf, label=f"{bin[0]},{bin[1]}", linewidth=linewidth, color=color)
            i += 1

        ax_k.set_xlim(x_min, x_max)
        ax_k.set_ylim(bottom=0.0)
        ax_k.set_xlabel("Reflectance")
        ax_k.set_ylabel("Density (KDE)")
        ax_k.set_title(f"{band} value distributions(KDE) by different {angle} bins ")
        ax_k.grid(True, alpha=0.3)
        ax_k.legend(ncol=2)
        fig_k.tight_layout()
        if out == None:
            plt.show()
        else:
            fig_k.savefig(out, dpi=dpi)
            plt.close(fig_k)
            logging.info(f"[plotting_raster] Saved band KDE chart to: {out}")
    except Exception as e:
        logging.error(f"[plotting_raster] Failed to create band KDE chart: {e}")
