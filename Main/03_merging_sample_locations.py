#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# Version      : 1.0
# ---------------------------------------------------------------------------
"""
This script consolidates the results produced in previous scripts into a single database.
The resulting database will serve as input for machine learning modeling.
"""

# Import necessary libraries
import glob
import pandas as pd
import numpy as np
import os
from config_object import config

# Define input and output paths from configuration
out = config.merging_out  # Output directory for merged results
input_dir = config.merging_input_dir  # Directory containing feather files to merge
ground_truth_coordinates = config.merging_groung_truth_coordinates  # Path to ground truth coordinates
file_name = config.main_extract_name  # Output file name prefix

# Step 1: Load all feather files to be merged
# Find all feather files in the input directory
path_list = glob.glob(input_dir + '\\*.feather')
dfs = []  # List to store each loaded feather file

# Load each feather file and append it to the list of DataFrames
for each in path_list:
    dfs.append(pd.read_feather(each))

# Concatenate all loaded DataFrames into a single DataFrame
df_pix = pd.concat(dfs)
df_pix = df_pix.rename(columns={'plot': 'plot_pix'})  # Rename 'plot' column to avoid conflicts

# Step 2: Load ground truth coordinates
# Load the ground truth coordinates file, which contains plot identifiers and their x, y positions
df_main = pd.read_csv(ground_truth_coordinates, names=['plot', 'x', 'y'])

# Step 3: Match pixels to plot values
# For each pixel, assign the values from the nearest sampling point (plot) if it falls within the specified radius
main_plots = pd.unique(df_main["plot"])  # Unique plot IDs in ground truth data
pix_plots = pd.unique(df_pix["plot_pix"])  # Unique plot IDs in pixel data
rows = []  # List to store DataFrames for each plot's ground truth data
pixplots = []  # List to store DataFrames for each plot's pixel data

# Separate the DataFrames for each plot ID to match with corresponding pixels
for plot_id in main_plots:
    rows.append(df_main[df_main["plot"] == plot_id])  # Extract ground truth data for each plot

for pix_id in pix_plots:
    pixplots.append(df_pix[df_pix["plot_pix"] == pix_id])  # Extract pixel data for each plot ID

# Step 4: Repeat ground truth data for each matching pixel
# Match each plot's ground truth data with its corresponding pixel data
res = []
for ground_truth_df, pixel_df in zip(rows, pixplots):
    # Repeat ground truth data for each pixel to align with pixel data structure
    repeated_df = pd.DataFrame(np.repeat(ground_truth_df.values, len(pixel_df.index), axis=0))
    repeated_df.columns = ground_truth_df.columns
    # Concatenate repeated ground truth data with pixel data
    res.append(pd.concat([repeated_df.reset_index(), pixel_df.reset_index()], sort=False, axis=1))

# Concatenate all plot-pixel merged DataFrames into a single final DataFrame
df_fin = pd.concat(res)
df = df_fin.drop(columns=['index'])  # Drop unnecessary index column

# Step 5: Save the final consolidated DataFrame
# Create the output directory if it does not already exist
path_to_output_data_set = out
if not os.path.isdir(path_to_output_data_set):
    os.makedirs(path_to_output_data_set)

# Reset index and save the consolidated DataFrame as a feather file
df = df.reset_index().drop('index', axis=1)
df.to_feather(os.path.join(path_to_output_data_set, f'{file_name}_for_classification.feather'))
