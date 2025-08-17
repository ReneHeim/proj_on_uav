
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from src.Common.raster import plotting_raster
import logging



def plot_df(week, gdf, out_dir):

    for path in list(gdf["paths"]):


        try:
            name = Path(path).stem
            if (out_dir / "bands_data" / f"panels_{name}.png").exists():
                logging.info(f"Skipping {name} as it already exists")
                continue
        except Exception as e:
                continue

        try:
            df = pl.read_parquet(path)
        except Exception as e:
            print(e)
            print(path)
            continue

        plotting_raster(
            df,
            out_dir,
            name,
            debug=True,
            fill_empty=False,
            ny=380,
            nx=630,
            plot_density=True,
            dpi=500,
            auto_figsize=True,
            density_discrete=True,
            density_mode="kde",
        )
