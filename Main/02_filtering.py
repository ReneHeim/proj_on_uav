from functions.filters_functions import  OSAVI_index_filtering
from functions.config_object import config_object
import polars as pl
import os
import glob
def main():
    config = config_object("config_file.yaml")

    ## Import dataframe

    print(config.main_extract_out)

    paths = glob.glob(os.path.join(config.main_extract_out, "*.parquet"))

    df = pl.read_parquet(paths[3])

    print(f"Dataframe shape before filtering: {df.shape}")

    ## Apply OSAVI index filtering
    df = OSAVI_index_filtering(df)

    print(f"Dataframe shape after filtering: {df.shape}")


    print(df)


    ##glob files




    print(paths)

if __name__ == "__main__":
    main()