#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# Version      : 1.0
# ---------------------------------------------------------------------------
"""
This script filters the full pixel data to retain only sample locations within specified radii
around sampling points. The aim is to limit the dataset to relevant points around specified plot
coordinates for focused analysis.

This version is updated to read and write Parquet files instead of Feather files, ensuring
compatibility with outputs from previously run code.

"""

import pandas as pd
import dask.dataframe as dd
import glob
from scipy import spatial
from tqdm import tqdm
from timeit import default_timer as timer
import os
from config_object import config

# Directories from config
input_dir = [config.filter_input_dir]  # Directory containing Parquet files from previous step
out = config.filter_out                 # Output directory for filtered results

# Load plot coordinates (sampling points) with columns for ID, x, y positions
plots_path = config.filter_groung_truth_coordinates
plots = pd.read_csv(plots_path, names=['id', 'x', 'y'])
plots_round = round(plots, 1)  # Round coordinates for more consistent spatial matching

# Loop through specified radii (e.g., 50 cm, 1 m, 2 m) around sampling points
for radius_id, radius in {'My_radius': config.filter_radius}.items():

    # Loop through directories containing Parquet files
    for directory in input_dir:
        # Find all Parquet files in the directory
        parquet_list = glob.glob(os.path.join(directory, "*.parquet"))

        # Process each Parquet file
        for number, parquet_file in tqdm(enumerate(parquet_list), desc='Processing Files'):
            # Step 1: Load Parquet file
            start = timer()
            df = pd.read_parquet(parquet_file)
            end = timer()
            print('Time to load Parquet file with pandas: ', end - start, 'seconds')

            # Step 2: Clean data by removing unnecessary columns
            # Adjust the columns you drop based on which columns are not needed.
            # Ensure these columns exist in your current dataset.
            df_cleaned = df.drop(columns=['elev', 'vaa_rad'], errors='ignore')

            # Step 3: Prepare data for spatial querying
            # Create coordinate lists for KDTree
            plotlist = list(zip(plots_round['x'], plots_round['y']))   # Plot (x, y)
            coordlist = list(zip(df_cleaned['Xw'], df_cleaned['Yw']))  # Pixel (Xw, Yw)

            # Build a KDTree for spatial queries
            tree = spatial.KDTree(coordlist)

            # Step 4: Filter pixels within the specified radius around each sampling point
            plot_box = []
            for count, i in tqdm(enumerate(plotlist), desc='Filtering Points'):
                closest = tree.query(plotlist[count])
                allwithin_radius = tree.query_ball_point(coordlist[closest[1]], radius, p=2, workers=-1)

                plot_df = df_cleaned.iloc[allwithin_radius]
                plot_df["plot"] = plots_round["id"][count]
                plot_box.append(plot_df)

            # Step 5: Concatenate filtered data from all plots and save as a Parquet file
            result = pd.concat(plot_box, ignore_index=True)

            # Create output directory if it doesn't exist
            if not os.path.isdir(out):
                os.makedirs(out)

            # Generate output file name based on the original file name
            file_name = os.path.basename(parquet_file)  # Keep the same name, just filtered

            # Save the filtered result as Parquet
            result.to_parquet(os.path.join(out, file_name), compression='zstd', compression_level=2)
