#!/usr/bin/env python3
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2023/02/07
# version ='1.2'
# ---------------------------------------------------------------------------

"""
This script does with the orthomosaic what has been previously done for orthophotos. It breaks the orthomosaic into pixels
which are put in a tabular form. Then a KD Tree is used to search for pixels that fall in a given radius around points 
sampled on the ground. 
"""

# Load libs
import pandas as pd
import glob
import os
from smac_functions import *
from functools import reduce, partial
import numpy as np
from joblib import Parallel, delayed

# Define all necessary input directories in a list of dictionnaries for 2 dates and 3 resolutions (6 elements)
# Each Dictionnary is comprised of 3 elements: 
#   the directory to the orthomosaic
#   the directory to the dem
#   the directory to the output file

sources = [{'ortho_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_20cm\All_flight\20200906_bot_corn_comb_20cm_mosaic.tif",
           'dem_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_20cm\All_flight\20200906_bot_corn_comb_20cm_dem.tif", 
           'out': r"\20200906_Bot\20200906_bot_corn_comb_20cm\All_flight\on_uav_for_classification_mosaic.csv"}, 
           
           {'ortho_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_50cm\All_flight\20200906_bot_corn_comb_50cm_mosaic.tif",
            'dem_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_50cm\All_flight\20200906_bot_corn_comb_50cm_dem.tif", 
            'out': r"\20200906_Bot\20200906_bot_corn_comb_50cm\All_flight\on_uav_for_classification_mosaic.csv"}, 
 
           {'ortho_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\All_flight\20200906_bot_corn_comb_1m_mosaic.tif",
            'dem_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\All_flight\20200906_bot_corn_comb_1m_dem.tif", 
            'out': r"\20200906_Bot\20200906_bot_corn_comb_1m\All_flight\on_uav_for_classification_mosaic.csv"}, 
           
           {'ortho_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_20cm\Nadir_flight\20200906_bot_corn_comb_20cm_mosaic.tif",
            'dem_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_20cm\Nadir_flight\20200906_bot_corn_comb_20cm_dem.tif", 
            'out': r"\20200906_Bot\20200906_bot_corn_comb_20cm\Nadir_flight\on_uav_for_classification_mosaic.csv"}, 
            
            {'ortho_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_50cm\Nadir_flight\20200906_bot_corn_comb_50cm_mosaic.tif",
             'dem_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_50cm\Nadir_flight\20200906_bot_corn_comb_50cm_dem.tif", 
             'out': r"\20200906_Bot\20200906_bot_corn_comb_50cm\Nadir_flight\on_uav_for_classification_mosaic.csv"}, 
  
            {'ortho_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\Nadir_flight\20200906_bot_corn_comb_1m_mosaic.tif",
             'dem_path': r"D:\on_uav_data\raw\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\Nadir_flight\20200906_bot_corn_comb_1m_dem.tif", 
             'out': r"\20200906_Bot\20200906_bot_corn_comb_1m\Nadir_flight\on_uav_for_classification_mosaic.csv"}, 
            
            {'ortho_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_20cm\All_flight\20200907_bot_corn_comb_20cm_mosaic.tif",
            'dem_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_20cm\All_flight\20200907_bot_corn_comb_20cm_dem.tif", 
            'out': r"\20200907_Bot\20200907_bot_corn_comb_20cm\All_flight\on_uav_for_classification_mosaic.csv"}, 
            
            {'ortho_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_50cm\All_flight\20200907_bot_corn_comb_50cm_mosaic.tif",
             'dem_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_50cm\All_flight\20200907_bot_corn_comb_50cm_dem.tif", 
             'out': r"\20200907_Bot\20200907_bot_corn_comb_50cm\All_flight\on_uav_for_classification_mosaic.csv"}, 
  
            {'ortho_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_1m\All_flight\20200907_bot_corn_comb_1m_mosaic.tif",
             'dem_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_1m\All_flight\20200907_bot_corn_comb_1m_dem.tif", 
             'out': r"\20200907_Bot\20200907_bot_corn_comb_1m\All_flight\on_uav_for_classification_mosaic.csv"}, 
            
            {'ortho_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_20cm\Nadir_flight\20200907_bot_corn_comb_20cm_mosaic.tif",
             'dem_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_20cm\Nadir_flight\20200907_bot_corn_comb_20cm_dem.tif", 
             'out': r"\20200907_Bot\20200907_bot_corn_comb_20cm\Nadir_flight\on_uav_for_classification_mosaic.csv"}, 
             
             {'ortho_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_50cm\Nadir_flight\20200907_bot_corn_comb_50cm_mosaic.tif",
              'dem_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_50cm\Nadir_flight\20200907_bot_corn_comb_50cm_dem.tif", 
              'out': r"\20200907_Bot\20200907_bot_corn_comb_50cm\Nadir_flight\on_uav_for_classification_mosaic.csv"}, 
   
             {'ortho_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_1m\Nadir_flight\20200907_bot_corn_comb_1m_mosaic.tif",
              'dem_path': r"D:\on_uav_data\raw\20200907_Bot\agisoft\20200907_bot_corn_comb_1m\Nadir_flight\20200907_bot_corn_comb_1m_dem.tif", 
              'out': r"\20200907_Bot\20200907_bot_corn_comb_1m\Nadir_flight\on_uav_for_classification_mosaic.csv"}, ]



# Define a function that does the main part of the job. It takes a dictionnary as input
def process_orthomosaic(source):
    # Import lib
    from tqdm import tqdm
    import pandas as pd
    import glob
    import os
    import rasterio as rio
    from functools import reduce, partial
    import math
    import numpy as np
    import exiftool as exif
    from tqdm import tqdm
    from pathlib import Path, PureWindowsPath
    from timeit import default_timer as timer
    import pyarrow
    from scipy import spatial
    
    # assign dic entries to variables
    out = source['out']
    dem_path = source['dem_path']
    ortho_path = source['ortho_path']
    
    # Turn DEM to list
    start = timer()
    with rio.open(dem_path) as dem:
        d1 = dem.read(1)
        arr_dem = np.array(d1)
    Xp_dem, Yp_dem, val_dem = xyval(arr_dem)
    res_dem = []
    with rio.open(dem_path) as dem_layer:
        for i, j in zip(Xp_dem, Yp_dem):
            res_dem.append(dem_layer.xy(i, j))
        df = pd.DataFrame(res_dem)
        df_dem = pd.concat([pd.Series(val_dem), df], axis=1)
        df_dem.columns = ['elev', 'Xw', 'Yw']
        df_dem = df_dem.round({'elev': 2, 'Xw': 2, 'Yw': 2})
    end = timer()
    print('Break DEM into pixel: ', end - start, 'seconds')
    
    # Turn DEM to list
    start4 = timer()
    bands = {}
    with rio.open(ortho_path) as rst:
        # Loops through bands
        for counter in range(1, 11, 1):
            res = []
            b1 = rst.read(counter)
            arr = np.array(b1)
            Xp, Yp, val = xyval(arr) # this function is loaded from raster_functions.py
            for i, j in zip(Xp, Yp):
                res.append(rst.xy(i, j))  # input px, py -> see rasterio documentation
            df = pd.DataFrame(res)
            df_ortho = pd.concat([pd.Series(val), df], axis=1)
            df_ortho.columns = ['band'+str(counter), 'Xw', 'Yw']
            df_ortho = df_ortho.round({'Xw': 2, 'Yw': 2})  # rounding coords for matches; careful when aligning raster
            bands[f"band{counter}"] = df_ortho
    my_reduce = partial(pd.merge, on=["Xw", "Yw"], how='outer') # define a function to merge bands together
    df_ortho = reduce(my_reduce, bands.values()) # iterates functions for all 11 bands
    end4 = timer()
    print('Break orthomosaic into pixel: ', end4 - start4, 'seconds')
    
    # Combine ortho values and DEM values while setting NaN's
    start5 = timer()
    dfs = [df_dem, df_ortho]  #
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=["Xw", "Yw"]), dfs)
    end5 = timer()
    print('Combine DEM and orthomosaic', end5 - start5, 'seconds')
    
    # Filter points around sampling points
    start6 = timer()
    plots = pd.read_csv(r"D:\on_uav_data\raw\ground truth\gpscoords_plots_bottelare_raw.csv", names=['id', 'x', 'y'])
    plots_round = round(plots, 2)
    plots_round.head()
    # Loop through different radius sizes
    # for radius_id,radius in tqdm({'50cm_radius':0.5, '1m_radius':1, '2m_radius':2}.items()):
    for radius_id,radius in tqdm({'2m_radius':2}.items()):
        path_to_save = f'D:\\on_uav_data\\proc\\merge\\{radius_id}'
        df = df_merged
        df_cleaned = df.drop(columns=['elev']) 
        
        plotlist = list(zip(plots_round['x'], plots_round['y']))
        coordlist = list(zip(df_cleaned['Xw'], df_cleaned['Yw']))
        tree = spatial.KDTree(coordlist)
        # iterate to find row id for sample point of each plot">
        plot_box = []
        for count, i in tqdm(enumerate(plotlist), desc='loop 2'):
            closest = tree.query(plotlist[count])  # row id for pixel closest to the sample point
            allwithin2m = tree.query_ball_point(coordlist[closest[1]], radius, p=2, workers=-1)  # all pixels within 'radius' from the sample point
            plot = df_cleaned.iloc[allwithin2m]
            plot["plot"] = plots_round["id"][count]  # this is all pixel of plot 'count' that are part of the imported csv  
            plot_box.append(plot)
        # Clean table
        result = pd.concat(plot_box)
        df_pix = result.rename(columns = {'plot':'plot_pix'})
        
        # Add target data (classified)
        df_main = pd.read_csv(r"D:\on_uav_data\raw\ground truth\ref_data.csv")
        df_main["lai_cat_a"] = pd.qcut(df_main['lai'], 3, labels=["low", "medium", "high"])
        df_main["nbi_cat_a"] = pd.qcut(df_main['nbi'], 3, labels=["low", "medium", "high"])
        df_main["chl_cat_a"] = pd.qcut(df_main['chl'], 3, labels=["low", "medium", "high"])
        df_main["glai_cat_a"] = pd.qcut(df_main['glai'], 3, labels=["low", "medium", "high"])
        main = pd.unique(df_main["plot"])
        pix = pd.unique(df_pix["plot_pix"])
        pix_clean = np.delete(pix, [17, 18, 20, 27])
        
        end6 = timer()
        print('Generate table ' , end6 - start6, 'seconds')
        
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
        df.to_csv(path_to_save+out)

# Loop trough dictionnaries using the above-defined function
Parallel(n_jobs=4)(delayed(process_orthomosaic)(i) for i in sources)
#for source in sources:
#    process_orthomosaic(source)
