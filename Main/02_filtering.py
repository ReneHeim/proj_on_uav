import logging

from functions.filters_functions import OSAVI_index_filtering, excess_green_filter
from functions.config_object import config_object
import polars as pl
import os
import glob

def logging_config():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("process.log", encoding='utf-8'),  # Note the encoding parameter
            logging.StreamHandler()
        ]
    )



def main():
    config = config_object("config_file.yaml")
    logging_config()

    ## Import dataframe


    paths = glob.glob(os.path.join(config.main_extract_out, "*.parquet"))

    df = pl.read_parquet(paths[3])


    ## Apply OSAVI index filtering
    df = OSAVI_index_filtering(df)
    df = excess_green_filter(df)


    print(df["OSAVI"].min())




    print(df)


    ##glob files
if __name__ == "__main__":
    main()