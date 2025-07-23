import time

from src.Util.logging import logging_config
from src.Util.search import optimized_recursive_search
from Common.config_object import  *
import geopandas as gpd
IGNORE_DIRS = {"System Volume Information"}
PATTERN_TMPL = "*{obj}*.parquet"
def main():
    config = config_object("config_file.yml")
    logging_config()


    folders = ['metashape','products_uav_data','output','extract','polygon_df']
    objective = "plot_"
    base_dir = r'D:\\'



    or_result = optimized_recursive_search(folders, objective, start=base_dir)

    # Load the GPKG file

    gdf = gpd.read_file(config_object.main_extract_out_polygons_df)

    # Display the first few rows
    print(gdf.head())
    print(gdf.crs)


if __name__ == "__main__":
    main()
