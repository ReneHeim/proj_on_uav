from datetime import datetime

import pandas as pd
import polars as pl
from colorama import Fore, Style, init
from tqdm import tqdm
import time

from src.Common.rpv import *
from src.Util.logging import logging_config
from src.Util.processing import process_weekly_data
from src.Util.search import optimized_recursive_search, order_path_list
from Common.config_object import  *
import geopandas as gpd
import re

IGNORE_DIRS = {"System Volume Information"}
PATTERN_TMPL = "*{obj}*.parquet"

init(autoreset=True)




def main():
    config = config_object("config_file.yml")
    logging_config()


    # Search data
    folders = ['', 'metashape', 'products_uav_data', 'output', 'extract', 'polygon_df']
    objective = "plot_"
    base_dir = r'/run/media/mak/Heim'
    plots_group = optimized_recursive_search(folders, objective, start_dir=base_dir)

    #Search geometry plot data
    gdf = pd.DataFrame(gpd.read_file(config.main_polygon_path))
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom.wkt if geom else None)
    gdf = pl.from_pandas(pd.DataFrame(gdf))

    weeks_dics = {}
    for key,group in plots_group.items():
        ordered_group = order_path_list(group)

        week_id = re.search(r'week\d+', group[0]).group()

        gdf_tmp = gdf
        gdf_tmp = gdf_tmp.with_columns([
            pl.Series("paths",ordered_group)
        ])
        weeks_dics[week_id] = gdf_tmp


    # Create rpvs for each
    for week, gdf in weeks_dics.items():
        result = process_weekly_data({week: gdf})
        result.drop("geometry").write_csv(f"{base_dir}/RPV_Results/V5/rpv_{week}_results.csv")


if __name__ == "__main__":
    main()
