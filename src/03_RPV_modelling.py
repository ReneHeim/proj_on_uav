import logging

from src.Common.data_loader import load_by_polygon
from src.Common.filters import add_mask_and_plot
from Common.filters import OSAVI_index_filtering, excess_green_filter, plot_heatmap, plot_spectrogram
from Common.config_object import config_object
import polars as pl
import os
import glob
from pathlib import Path



def logging_config():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("process.log", encoding='utf-8'),  # Note the encoding parameter
            logging.StreamHandler()
        ]
    )

def recursive_search(folders, objective,start= r'D:\**' , result = []):

    if start.count(r'D:\**') != 0:
        all_paths = glob.glob(r'D:\**')
    else:
         all_paths = glob.glob(os.path.join(start,'*'))

    directories = [path for path in all_paths
                   if os.path.isdir(path)
                   and path.count('$') == 0
                   and path.count('System Volume Information') == 0
                   or any(sub in path for sub in folders)]

    for directory in directories:
        dfs = glob.glob(os.path.join(directory,'*'))
        result.append([x for x in dfs if dfs.count(objective) != 0 and dfs.count('.parquet') > 0 ])
        print(len(result))
        recursive_search(folders=folders, objective = objective, start = directory, result = result)


def find_parquets(base_dir, objective):
    results = []
    for p in Path(base_dir).rglob('*.parquet'):
        if objective in p.name:
            results.append(str(p))
    return results



def main():
    config = config_object("config_file.yml")
    logging_config()

    all_paths = glob.glob(r'D:\**')
    directories = [path for path in all_paths if
                   os.path.isdir(path) and path.count('$') == 0 and path.count('System Volume Information') == 0]

    for dir in directories:
        dirs = glob.glob(os.path.join(dir, '*'))
        meta = [x for x in dirs if x.count('metashape') != 0 and os.path.isdir(x)]

        if len(meta) > 0:
            dirs = glob.glob(os.path.join(meta[0], '*'))
            prod_uav = [x for x in dirs if x.count('products_uav_data') != 0 and os.path.isdir(x)]

        else:
            continue

        if len(prod_uav) > 0:
            dirs = glob.glob(os.path.join(prod_uav[0], '*'))
            prod_uav = [x for x in dirs if x.count('output') != 0 and os.path.isdir(x)]

    #recursive_search(folders = ['metashape','products_uav_data','output','extract','polygon_df'], objective="plot_")

    print(len(find_parquets(r'D:\\', objective="plot_")))

if __name__ == "__main__":
    main()
