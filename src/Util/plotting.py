import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from Common.raster import plotting_raster


def plot_df(week, gdf, band):
    for path in list(gdf["paths"]):
        df = pl.read_parquet(path)
        plotting_raster(df,"/run/media/mak/Heim/RPV_Results/V8",
                        path.split("/")[-1].replace(".png","")+f"_{week}"
                        ,debug=True,fill_empty=False,
                        ny=300,
                        nx=600,
                        plot_density=True,
                        dpi=500,
                        auto_figsize = True, density_discrete=True
)
