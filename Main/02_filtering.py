import logging

from Main.functions.data_loader import data_loader_from_file_lists, load_by_polygon
from Main.functions.filters import add_mask_and_plot
from functions.filters import OSAVI_index_filtering, excess_green_filter, plot_heatmap, plot_spectrogram
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

    df = pl.read_parquet(paths[12])



    ## Apply OSAVI index filtering
    df = OSAVI_index_filtering(df)
    df = excess_green_filter(df)
    print(df)

    #Plot heatmaps
    # plot_heatmap(df, "OSAVI", config.main_extract_out)
    # plot_heatmap(df, "ExcessGreen", config.main_extract_out)


    #Plot Spectrograms
    bands_wavelength_list = [475, 560, 668, 717, 842]

    # plot_spectrogram(df,bands_wavelength_list=bands_wavelength_list,n_bands=5)

    # add_mask_and_plot(df,"OSAVI",0.4)
    # add_mask_and_plot(df,"ExcessGreen",0.03)

    dfs =  load_by_polygon(config.main_extract_out)
    print(dfs)



    ##glob files
if __name__ == "__main__":
    main()