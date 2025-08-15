import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl
from colorama import Fore, Style, init
from tqdm import tqdm

from src.Common.config_object import config_object
from src.Common.rpv import *
from src.Util.logging import logging_config
from src.Util.processing import process_weekly_data
from src.Util.search import optimized_recursive_search, order_path_list

IGNORE_DIRS = {"System Volume Information"}
PATTERN_TMPL = "*{obj}*.parquet"

init(autoreset=True)


def main():
    logging_config()
    parser = argparse.ArgumentParser(description="Fit RPV to weekly per-plot datasets")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "config_file_example.yml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--band", type=str, default=0, help="Band to fit")
    parser.add_argument(
        "--base-dir", type=str, default=r"/run/media/mak/Heim", help="Search base dir"
    )
    args = parser.parse_args()

    config = config_object(args.config)

    if args.band == 0:
        bands = [f"band{i}" for i in range(1, config.bands + 1)]
    else:
        bands = [args.band]

    # Search data
    folders = ["", "metashape", "products_uav_data", "output", "extract", "polygon_df"]
    objective = "plot_"
    base_dir = args.base_dir
    plots_group = optimized_recursive_search(folders, objective, start_dir=base_dir)

    # Search geometry plot data
    gdf = pd.DataFrame(gpd.read_file(config.main_polygon_path))
    gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.wkt if geom else None)
    gdf = pl.from_pandas(pd.DataFrame(gdf))

    weeks_dics = {}
    for key, group in plots_group.items():
        ordered_group = order_path_list(group)

        week_id = re.search(r"week\d+", group[0]).group()

        gdf_tmp = gdf
        gdf_tmp = gdf_tmp.with_columns([pl.Series("paths", ordered_group)])
        weeks_dics[week_id] = gdf_tmp

    # Create rpvs for each
    for week, gdf in weeks_dics.items():
        for band in bands:
            out_dir = Path(base_dir) / "RPV_Results" / "V8"
            out_dir.mkdir(parents=True, exist_ok=True)

            if (out_dir / f"rpv_{week}_{band}_results.csv").exists():
                continue

            result = process_weekly_data({week: gdf}, band=band)
            result.drop("geometry").write_csv(str(out_dir / f"rpv_{week}_{band}_results.csv"))


if __name__ == "__main__":
    main()
