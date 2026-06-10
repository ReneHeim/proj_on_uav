import logging
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
from matplotlib import pyplot as plt

from src.extract.raster import _kde1d_fast


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
