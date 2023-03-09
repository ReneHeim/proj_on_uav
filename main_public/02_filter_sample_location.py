#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# version ='1.0'
# ---------------------------------------------------------------------------
"""Using the full pixel data to filter only sample locations"""

# Import libs
import pandas as pd
import dask.dataframe as dd
import glob
from scipy import spatial
from tqdm import tqdm
from timeit import default_timer as timer
import pyarrow as pa
import os
from config_object import config


# Define input and output
input_dir=[config.filter_input_dir]
out = config.filter_out

# load plot coordinates
plots_path = config.filter_groung_truth_coordinates
plots = pd.read_csv(plots_path, names=['id', 'x', 'y'])
plots_round = round(plots, 1)
plots_round.head()

# Loop through 3 radii aroung sampling points (50 cm, 1 m, 2 m)
# for radius_id,radius in {'50cm_radius':0.5, '1m_radius':1, '2m_radius':2}.items():
for radius_id,radius in {'My_radius': config.filter_radius}.items():
    # Loop through dates (06.09.2022, 07.09.2022)
    for directory in input_dir:
        csv_list = glob.glob(directory+r'\*.feather')
        # Loop through each of our 15 chuncks
        for number, csv in tqdm(enumerate(csv_list), desc='loop 1'):
            start = timer()
            df = pd.read_feather(csv)
            end = timer()
            print('Pandas Load: ', end - start, 'seconds')
            df_cleaned = df.drop(columns=['elev', 'vaa_rad']) 
            
            # Create coordinate lists to build a kd tree
            plotlist = list(zip(plots_round['x'], plots_round['y']))
            coordlist = list(zip(df_cleaned['Xw'], df_cleaned['Yw']))
            tree = spatial.KDTree(coordlist)
        
            # Iterate to find row id for sample point of each plot">
            plot_box = []
            for count, i in tqdm(enumerate(plotlist), desc='loop 2'):
                closest = tree.query(plotlist[count])  # row id for pixel closest to the sample point
                allwithin2m = tree.query_ball_point(coordlist[closest[1]], radius, p=2, workers=-1)  # all pixels within 'radius' from the sample point
                plot = df_cleaned.iloc[allwithin2m]                    
                plot["plot"] = plots_round["id"][count]  # all pixel of plot 'count' that are part of the imported csv
                plot_box.append(plot)
            result = pd.concat(plot_box)
            
            # Create folders if not existing and save file
            path_to_output=out
            if not os.path.isdir(path_to_output):
                os.makedirs(path_to_output)
                
            file_name=csv.split('\\')[-1]
            
            result=result.reset_index().drop('index', axis=1)
            result.to_feather(os.path.join(path_to_output,file_name))



