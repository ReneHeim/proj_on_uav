#!/usr/bin/env python3
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim and Nathan Okole
# Created Date: 2023/02/07
# version ='1.2'
# ---------------------------------------------------------------------------
"""Using the full pixel data (01) to filter only sample locations"""

# Import libs
import pandas as pd
import dask.dataframe as dd
import glob
from scipy import spatial
from tqdm import tqdm
from timeit import default_timer as timer
import pyarrow as pa
import os

# Define input and output
input_dir=[r"D:\on_uav_data\proc\extract\20200906_Bot", r"D:\on_uav_data\proc\extract\20200907_Bot"]
out = r"D:\on_uav_data\proc\filter"

# load plot coordinates
plots = pd.read_csv(r"D:\on_uav_data\raw\ground truth\gpscoords_plots_bottelare_raw.csv", names=['id', 'x', 'y'])
plots_round = round(plots, 1)
plots_round.head()

# Loop through 3 radii aroung sampling points (50 cm, 1 m, 2 m)
# for radius_id,radius in {'50cm_radius':0.5, '1m_radius':1, '2m_radius':2}.items():
for radius_id,radius in {'2m_radius':2}.items():
    # Loop through dates (06.09.2022, 07.09.2022)
    for directory in input_dir:
        data_pixel_size=os.listdir(directory)
        # Loop through ground samplind distances (10 cm, 50 cm, 100 cm)
        for each_pixel_size in data_pixel_size:
            data_set_list = os.listdir(os.path.join(directory,each_pixel_size))
            # Loop through datasets
            for each_data_set in data_set_list:
                csv_list = glob.glob(os.path.join(directory,each_pixel_size,each_data_set)+r'\*.feather')
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
                    path_to_output_radius=os.path.join(out,radius_id)
                    if not os.path.isdir(path_to_output_radius):
                        os.mkdir(path_to_output_radius)     
                    path_to_output_date=os.path.join(path_to_output_radius,directory.split('\\')[-1])
                    if not os.path.isdir(path_to_output_date):
                        os.mkdir(path_to_output_date)    
                    path_to_output_resolution=os.path.join(path_to_output_date,each_pixel_size)
                    if not os.path.isdir(path_to_output_resolution):
                        os.mkdir(path_to_output_resolution)
                    path_to_output_data_set=os.path.join(path_to_output_resolution,each_data_set)
                    if not os.path.isdir(path_to_output_data_set):
                        os.mkdir(path_to_output_data_set)
                        
                    file_name=csv.split('\\')[-1]
                    
                    result=result.reset_index().drop('index', axis=1)
                    result.to_feather(os.path.join(path_to_output_data_set,file_name))



