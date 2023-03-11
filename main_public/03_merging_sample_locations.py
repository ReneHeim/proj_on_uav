#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# version ='1.0'
# ---------------------------------------------------------------------------

''' This script puts together results produced in the last script in a single database which will
then be used as input fosr machine learning modelling'''

# Import libs
import glob
import pandas as pd
import numpy as np
import os
from config_object import config


# Define input and output
out = config.merging_out
input_dir = config.merging_input_dir
ground_truth_coordinates = config.merging_groung_truth_coordinates
file_name = config.main_extract_name

# List files to merge         
path_list = glob.glob(input_dir + '\*.feather')
dfs = list()
# Loop through each of our 15 chuncks and concatenate them
for each in path_list:
    dfs.append(pd.read_feather(each))
df_pix = pd.concat(dfs)
df_pix=df_pix.rename(columns = {'plot':'plot_pix'})

# open coordinate file

df_main = pd.read_csv(ground_truth_coordinates, names=['plot', 'x', 'y'])

# Attribute each pixel the value of the point whose radius it falls in (delete plots with no reference value)
main = pd.unique(df_main["plot"])
pix = pd.unique(df_pix["plot_pix"])
pix_clean = pix
rows = list()
for i in main:
    rows.append(df_main[df_main["plot"] == i])
pixplots = list()
for i in pix_clean:
    pixplots.append(df_pix[df_pix["plot_pix"] == i])
res = list()
for (i, j) in zip(rows, pixplots):
    newdf = pd.DataFrame(np.repeat(i.values, len(j.index), axis=0))
    newdf.columns = i.columns
    res.append(pd.concat([newdf.reset_index(), j.reset_index()], sort=False, axis=1))
df_fin = pd.concat(res)
df = df_fin.drop(columns=['index']) #no column named level_0

# Create folders if not existing and save file
path_to_output_data_set=out 
if not os.path.isdir(path_to_output_data_set):
    os.makedirs(path_to_output_data_set)
    
#df.to_csv(os.path.join(path_to_output_data_set,'on_uav_for_classification.csv'))

df=df.reset_index().drop('index', axis=1)
df.to_feather(os.path.join(path_to_output_data_set,f'{file_name}_for_classification.feather'))
