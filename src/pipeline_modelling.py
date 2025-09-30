import argparse
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from colorama import init

from Utils.stats.processing import process_weekly_data_stats
from src.Common.config_object import config_object
from src.Common.logging import logging_config
from src.Common.search import optimized_recursive_search, order_path_list
from src.Utils.RPV_modelling.processing import process_weekly_data_rpv

from src.Utils.RPV_modelling.rpv import *

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
        out_dir = Path(base_dir) / "RPV_Results" / "V12" / week
        out_dir.mkdir(parents=True, exist_ok=True)
        # plot_df(week, gdf, out_dir)

        for band in bands:
            if (out_dir / f"rpv_{week}_{band}_results.csv").exists():
                continue
            result = process_weekly_data_rpv(
                {week: gdf}, band=band, sample_total_dataset=500_000, filter={}
            )
            result.drop("geometry").write_csv(str(out_dir / f"rpv_{week}_{band}_results.csv"))

    # Create Plots of RPV results
    df_all_rpv = pd.DataFrame()
    for week, gdf in weeks_dics.items():
        out_dir = Path(base_dir) / "RPV_Results" / "V12" / week
        for band in bands:
            df_rpv = pd.read_csv(str(out_dir / f"rpv_{week}_{band}_results.csv"))
            df_rpv["week"] = week
            df_rpv["band"] = band
            df_all_rpv = pd.concat([df_all_rpv, df_rpv])
    df_all_rpv.reset_index(inplace=True)

    df_all_rpv.to_csv(str(Path(base_dir) / "RPV_Results" / "V12" / "rpv_results.csv"))


    for week, gdf in weeks_dics.items():
        out_dir = Path(base_dir) / "stats" / "V1" / week
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / f"stats_{week}_{band}.csv").exists():
            continue
        result = process_weekly_data_stats({week: gdf}, filter={}, out=out_dir)


if __name__ == "__main__":
    main()
