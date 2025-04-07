#!/usr/bin/env python3
#----------------------------------------------------------------------------
# Created By  : Nathan Okole and Rene HJ Heim
# Created Date: 2023/02/07
# version ='1.2'
# ---------------------------------------------------------------------------

''' This script puts together results produced in the last script in a single database which will
then be used as input for subsequent modelling'''

# Import libs
import glob
import pandas as pd
import numpy as np
import itertools
import os

# Define input and output
out = r"D:\on_uav_data\proc\merge"
input_dir=r"D:\on_uav_data\proc\filter"


# Loop through 3 radii aroung sampling points (50 cm, 1 m, 2 m)
for radius in os.listdir(input_dir):
    path_to_input_radius=os.path.join(input_dir,radius)
    # Loop through dates (06.09.2022, 07.09.2022)
    for date in os.listdir(path_to_input_radius):
        path_to_input_date=os.path.join(path_to_input_radius,date)
        # Loop through ground samplind distances (20 cm, 50 cm, 100 cm)
        for resolution in os.listdir(path_to_input_date):
            path_to_input_resolution=os.path.join(path_to_input_date,resolution)
            for data_set in os.listdir(path_to_input_resolution):
                path_to_input_data_set=os.path.join(path_to_input_resolution,data_set)            
                path_list = glob.glob(path_to_input_data_set + '\*.feather')
                dfs = list()
                # Loop through each of our 15 chuncks and concatenate them
                for each in path_list:
                    dfs.append(pd.read_feather(each))
                df_pix = pd.concat(dfs)
                df_pix=df_pix.rename(columns = {'plot':'plot_pix'})
                
                # Cut target (ground truth) variable according to different levels (3 or 5 levels) for LAI and Chl
                df_main = pd.read_csv(r"D:\on_uav_data\raw\ground truth\ref_data.csv")
                df_main["lai_cat_a"] = pd.qcut(df_main['lai'], 3, labels=["low", "medium", "high"])
                df_main["chl_cat_a"] = pd.qcut(df_main['chl'], 3, labels=["low", "medium", "high"])
                df_main["glai_cat_a"] = pd.qcut(df_main['glai'], 3, labels=["low", "medium", "high"])
                df_main["nbi_cat_a"] = pd.qcut(df_main['nbi'], 3, labels=["low", "medium", "high"])
                
                # Attribute each pixel the value of the point whose radius it falls in (delete plots with no reference value)
                main = pd.unique(df_main["plot"])
                pix = pd.unique(df_pix["plot_pix"])
                pix_clean = np.delete(pix, [17, 18, 20, 27])
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
                path_to_output_radius=os.path.join(out,radius)
                if not os.path.isdir(path_to_output_radius):
                    os.mkdir(path_to_output_radius)
                path_to_output_date=os.path.join(path_to_output_radius,date)
                if not os.path.isdir(path_to_output_date):
                    os.mkdir(path_to_output_date)
                path_to_output_resolution=os.path.join(path_to_output_date,resolution)
                if not os.path.isdir(path_to_output_resolution):
                    os.mkdir(path_to_output_resolution)
                path_to_output_data_set=os.path.join(path_to_output_resolution,data_set)  
                if not os.path.isdir(path_to_output_data_set):
                    os.mkdir(path_to_output_data_set)
                    
                #df.to_csv(os.path.join(path_to_output_data_set,'on_uav_for_classification.csv'))
                
                df=df.reset_index().drop('index', axis=1)
                df.to_feather(os.path.join(path_to_output_data_set,'on_uav_for_classification.feather'))
