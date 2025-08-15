import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from Common.raster import plotting_raster


def plot_df(week, gdf, band):
    for path in list(gdf["paths"]):
        df = pl.read_parquet(path)
        plotting_raster(df,"/run/media/mak/Heim/RPV_Results/V8",path.split("/")[-1].replace(".png",""),debug=True)
