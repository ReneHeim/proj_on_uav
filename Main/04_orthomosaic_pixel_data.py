#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# Version      : 1.0
# ---------------------------------------------------------------------------
"""
This script processes an orthomosaic image similarly to how individual orthophotos were handled.
It breaks the orthomosaic into individual pixels in a tabular form and uses a KD Tree to locate
pixels within a given radius around specific ground-sampled points.
"""

# Import required libraries
from Main.functions.raster_functions import *  # Import custom functions
from joblib import Parallel, delayed
from config_object import config

# Define the output directory for processed results
output = config.merging_out

# Define input files and directories
ground_truth_coordinates = config.filter_groung_truth_coordinates  # Path to ground truth coordinates

# Define the search radius for pixels around sampling points
radii = config.orthomosaic_radius

# Specify input directories for orthomosaic and DEM files
# Each dictionary entry contains paths for orthomosaic, DEM, and output
sources = [
    {
        'ortho_path': config.orthomosaic_ortho_path,
        'dem_path': config.orthomosaic_dem_path,
        'out': config.orthomosaic_name
    },
]

# Define the main function for processing the orthomosaic file
def process_orthomosaic(source):
    # Import necessary libraries inside the function for parallel processing
    from tqdm import tqdm
    import pandas as pd
    import os
    import rasterio as rio
    from functools import reduce, partial
    import numpy as np
    from scipy import spatial
    from timeit import default_timer as timer

    # Assign paths from the dictionary to local variables
    out = source['out']
    dem_path = source['dem_path']
    ortho_path = source['ortho_path']

    # Step 1: Load the DEM file and prepare a DataFrame of pixel coordinates and elevation
    start = timer()
    with rio.open(dem_path) as dem:
        d1 = dem.read(1)  # Read elevation data
        arr_dem = np.array(d1)
    Xp_dem, Yp_dem, val_dem = xyval(arr_dem)  # Extract pixel values and coordinates using custom function
    res_dem = []
    with rio.open(dem_path) as dem_layer:
        for i, j in zip(Xp_dem, Yp_dem):
            res_dem.append(dem_layer.xy(i, j))  # Convert pixel indices to coordinates
        df = pd.DataFrame(res_dem)
        df_dem = pd.concat([pd.Series(val_dem), df], axis=1)
        df_dem.columns = ['elev', 'Xw', 'Yw']
        df_dem = df_dem.round({'elev': 2, 'Xw': 2, 'Yw': 2})  # Round coordinates for consistency
    end = timer()
    print('Break DEM into pixels: ', end - start, 'seconds')

    # Step 2: Load the orthomosaic file and extract pixel values for each band
    start4 = timer()
    bands = {}
    with rio.open(ortho_path) as rst:
        for counter in range(1, 11):
            res = []
            b1 = rst.read(counter)
            arr = np.array(b1)
            Xp, Yp, val = xyval(arr)  # Extract pixel values and coordinates
            for i, j in zip(Xp, Yp):
                res.append(rst.xy(i, j))
            df = pd.DataFrame(res)
            df_ortho = pd.concat([pd.Series(val), df], axis=1)
            df_ortho.columns = ['band'+str(counter), 'Xw', 'Yw']
            df_ortho = df_ortho.round({'Xw': 2, 'Yw': 2})
            bands[f"band{counter}"] = df_ortho
    my_reduce = partial(pd.merge, on=["Xw", "Yw"], how='outer')
    df_ortho = reduce(my_reduce, bands.values())  # Combine all bands into a single DataFrame
    end4 = timer()
    print('Break orthomosaic into pixels: ', end4 - start4, 'seconds')

    # Step 3: Merge DEM and orthomosaic data to combine elevation and band information
    start5 = timer()
    dfs = [df_dem, df_ortho]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=["Xw", "Yw"]), dfs)
    end5 = timer()
    print('Combine DEM and orthomosaic: ', end5 - start5, 'seconds')

    # Step 4: Filter pixels within a specified radius around sampling points
    start6 = timer()
    plots = pd.read_csv(ground_truth_coordinates, names=['id', 'x', 'y'])
    plots_round = round(plots, 2)

    # Loop through specified radii
    for radius_id, radius in tqdm({'2m_radius': radii}.items()):
        path_to_save = output
        df = df_merged

        plotlist = list(zip(plots_round['x'], plots_round['y']))  # Sampling point coordinates
        coordlist = list(zip(df['Xw'], df['Yw']))  # Pixel coordinates
        tree = spatial.KDTree(coordlist)  # KDTree for fast spatial search

        # Find pixels within the radius for each sampling point
        plot_box = []
        for count, point in tqdm(enumerate(plotlist), desc='Processing Points'):
            closest = tree.query(point)  # Find closest pixel
            all_within_radius = tree.query_ball_point(coordlist[closest[1]], radius, p=2, workers=-1)
            plot = df.iloc[all_within_radius]
            plot["plot"] = plots_round["id"][count]  # Label pixels with plot ID
            plot_box.append(plot)

        # Combine results and prepare for saving
        result = pd.concat(plot_box)
        df_pix = result.rename(columns={'plot': 'plot_pix'})

        # Step 5: Add ground truth data to each pixel entry
        df_main = pd.read_csv(ground_truth_coordinates, names=['plot', 'x', 'y'])
        main_plots = pd.unique(df_main["plot"])
        pix_plots = pd.unique(df_pix["plot_pix"])

        # Match each plot’s ground truth data with pixel data
        rows = [df_main[df_main["plot"] == plot] for plot in main_plots]
        pixplots = [df_pix[df_pix["plot_pix"] == plot] for plot in pix_plots]
        res = []
        for ground_truth_df, pixel_df in zip(rows, pixplots):
            repeated_df = pd.DataFrame(np.repeat(ground_truth_df.values, len(pixel_df.index), axis=0))
            repeated_df.columns = ground_truth_df.columns
            res.append(pd.concat([repeated_df.reset_index(), pixel_df.reset_index()], sort=False, axis=1))

        # Final consolidated DataFrame
        df_fin = pd.concat(res).drop(columns=['index', 'x', 'y'])
        df_fin.to_csv(os.path.join(path_to_save, f"{out}.csv"))
        end6 = timer()
        print('Generate table: ', end6 - start6, 'seconds')

# Process each dictionary entry in parallel using the defined function
Parallel(n_jobs=4)(delayed(process_orthomosaic)(source) for source in sources)
