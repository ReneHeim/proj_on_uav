import argparse
import glob
import logging
import os

import polars as pl

from src.Common.config_object import config_object
from src.Common.data_loader import load_by_polygon
from src.Common.filters import (
    OSAVI_index_filtering,
    add_mask_and_plot,
    excess_green_filter,
    plot_heatmap,
    plot_spectrogram,
)
from src.Util.logging import logging_config


def main():
    logging_config()
    parser = argparse.ArgumentParser(description="Apply filtering and polygon splitting")
    parser.add_argument(
        "--config",
        type=str,
        default=str(os.path.join(os.path.dirname(__file__), "config_file_example.yml")),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = config_object(args.config)

    ## Import dataframe

    paths = glob.glob(os.path.join(config.main_extract_out, "*.parquet"))

    if not paths:
        raise RuntimeError(f"No parquet files found in {config.main_extract_out}")
    df = pl.read_parquet(paths[0])

    ## Apply OSAVI index filtering
    df = OSAVI_index_filtering(df)
    df = excess_green_filter(df)
    print(df.columns)

    # Plot heatmaps
    # plot_heatmap(df, "OSAVI", config.main_extract_out)
    # plot_heatmap(df, "ExcessGreen", config.main_extract_out)

    # Plot Spectrograms
    bands_wavelength_list = [475, 560, 668, 717, 842]

    # plot_spectrogram(df,bands_wavelength_list=bands_wavelength_list,n_bands=5)

    # add_mask_and_plot(df,"OSAVI",0.4)
    # add_mask_and_plot(df,"ExcessGreen",0.03)

    dfs = load_by_polygon(str(config.main_extract_out), str(config.main_extract_out_polygons_df))


if __name__ == "__main__":
    main()
