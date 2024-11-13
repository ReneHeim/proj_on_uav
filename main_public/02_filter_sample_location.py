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
"""

# Import necessary libraries
import pandas as pd
import dask.dataframe as dd
import glob
from scipy import spatial
from tqdm import tqdm
from timeit import default_timer as timer
import pyarrow as pa
import os
from config_object import config

# Define input and output directories and paths for configuration
input_dir = [config.filter_input_dir]  # Input directory containing feather files
out = config.filter_out                 # Output directory for filtered results

# Load plot coordinates (sampling points) with columns for ID, x, y positions
plots_path = config.filter_groung_truth_coordinates
plots = pd.read_csv(plots_path, names=['id', 'x', 'y'])
plots_round = round(plots, 1)  # Round coordinates for more consistent matching
plots_round.head()

# Loop through specified radii (e.g., 50 cm, 1 m, 2 m) around sampling points
for radius_id, radius in {'My_radius': config.filter_radius}.items():

    # Loop through dates in the input directory (e.g., data collected on specific dates)
    for directory in input_dir:
        # Find all feather files in the directory
        csv_list = glob.glob(directory + r'\*.feather')

        # Process each feather file in chunks (here assumed to be 15 chunks for parallel processing)
        for number, csv in tqdm(enumerate(csv_list), desc='Processing Files'):
            # Step 1: Load feather file with pixel data
            start = timer()
            df = pd.read_feather(csv)
            end = timer()
            print('Time to load feather file with pandas: ', end - start, 'seconds')

            # Step 2: Clean data by removing unnecessary columns
            df_cleaned = df.drop(columns=['elev', 'vaa_rad'])  # Drop columns not needed for filtering

            # Step 3: Prepare data structures for spatial querying
            # - Create coordinate list from plots and pixel data to build a KDTree
            plotlist = list(zip(plots_round['x'], plots_round['y']))  # List of plot coordinates (x, y)
            coordlist = list(zip(df_cleaned['Xw'], df_cleaned['Yw'])) # List of pixel coordinates (Xw, Yw)
            tree = spatial.KDTree(coordlist)  # KDTree for fast spatial lookup

            # Step 4: Filter pixels within the specified radius around each sampling point
            plot_box = []  # To hold data for each plot within the specified radius
            for count, i in tqdm(enumerate(plotlist), desc='Filtering Points'):
                # Find the closest pixel to the sample point
                closest = tree.query(plotlist[count])

                # Get all pixels within 'radius' of the sample point using KDTree's query_ball_point
                allwithin_radius = tree.query_ball_point(coordlist[closest[1]], radius, p=2, workers=-1)

                # Filter dataframe rows for pixels within the specified radius
                plot = df_cleaned.iloc[allwithin_radius]
                plot["plot"] = plots_round["id"][count]  # Tag pixels with the corresponding plot ID
                plot_box.append(plot)

            # Step 5: Concatenate filtered data from all plots and save as feather file
            result = pd.concat(plot_box)

            # Create output directory if it doesn't exist
            path_to_output = out
            if not os.path.isdir(path_to_output):
                os.makedirs(path_to_output)

            # Generate file name based on the original feather file
            file_name = csv.split('\\')[-1]

            # Reset index, remove old index, and save filtered data as a new feather file
            result = result.reset_index().drop('index', axis=1)
            result.to_feather(os.path.join(path_to_output, file_name))
