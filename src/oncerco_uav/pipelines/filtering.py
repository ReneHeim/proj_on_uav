import argparse
import glob
import os

from oncerco_uav.core.config_object import config_object
from oncerco_uav.core.logging import logging_config
from oncerco_uav.filter.data_loader import load_by_polygon
from oncerco_uav.filter.filters import spectral_index_expressions


def main():
    logging_config()
    parser = argparse.ArgumentParser(description="Apply filtering and polygon splitting")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/config_file_example.yml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = config_object(args.config)

    ## Import dataframe

    paths = glob.glob(os.path.join(config.main_extract_out, "*.parquet"))

    if not paths:
        raise RuntimeError(f"No parquet files found in {config.main_extract_out}")
    load_by_polygon(
        str(config.main_extract_out),
        str(config.main_extract_out_polygons_df),
        derived_columns=spectral_index_expressions(),
    )


if __name__ == "__main__":
    main()
