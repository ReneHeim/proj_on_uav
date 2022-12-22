#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/04
# version ='1.0'
# ---------------------------------------------------------------------------
"""loading a multi-band raster and perform soil segmentation by spectral vegetation index thresholding"""

# <editor-fold desc="01_importing libs">
import matplotlib
#matplotlib.use('TkAgg')

# Load modules
import glob
import os
import rasterio as rio
from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

# Set error to avoid 0 division errors
np.seterr(divide='ignore', invalid='ignore')  

# to clone your directory to create the same data structure for images with and without segmentation
# use robocopy "D:\on_uav_data" "D:\on_uav_data_soil_seg" /e /xf * in the cmd shell to have a folder with the same 
#   subfolders as another one

# Define output directory
saving_directory = r"D:\on_uav_data_soil_seg"

# Define all necessary input directories in a list of dictionnaries for 2 dates and 3 resolutions (6 elements)
# Each Dictionnary is comprised of 3 elements: 
#   the directory to the orthomosaic
#   the directory to the dem
#   the directory to orthophotos
sources = [{'mosaic_path': r"D:\on_uav_data\20200906_Bot\agisoft\20200906_bot_corn_comb_10cm\20200906_bot_corn_comb_10cm_mosaic.tif",
           'dem_path': r"D:\on_uav_data\20200906_Bot\agisoft\20200906_bot_corn_comb_10cm\20200906_bot_corn_comb_10cm_dem.tif", 
           'photos_path': r"D:\on_uav_data\20200906_Bot\agisoft\20200906_bot_corn_comb_10cm\20200906_bot_corn_comb_10cm_orthophotos"}, 
           
           {'mosaic_path': r"D:\on_uav_data\20200906_Bot\agisoft\20200906_bot_corn_comb_50cm\20200906_bot_corn_comb_50cm_mosaic.tif",
           'dem_path': r"D:\on_uav_data\20200906_Bot\agisoft\20200906_bot_corn_comb_50cm\20200906_bot_corn_comb_50cm_dem.tif", 
           'photos_path': r"D:\on_uav_data\20200906_Bot\agisoft\20200906_bot_corn_comb_50cm\20200906_bot_corn_comb_50cm_orthophotos"}, 
           
           {'mosaic_path': r"D:\on_uav_data\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\20200906_bot_corn_comb_1m_mosaic.tif",
           'dem_path': r"D:\on_uav_data\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\20200906_bot_corn_comb_1m_dem.tif", 
           'photos_path': r"D:\on_uav_data\20200906_Bot\agisoft\20200906_bot_corn_comb_1m\20200906_bot_corn_comb_1m_orthophotos"},
           
           {'mosaic_path': r"D:\on_uav_data\20200907_Bot\agisoft\20200907_bot_corn_comb_10cm\20200907_bot_corn_comb_10cm_mosaic.tif",
           'dem_path': r"D:\on_uav_data\20200907_Bot\agisoft\20200907_bot_corn_comb_10cm\20200907_bot_corn_comb_10cm_dem.tif", 
           'photos_path': r"D:\on_uav_data\20200907_Bot\agisoft\20200907_bot_corn_comb_10cm\20200907_bot_corn_comb_10cm_orthophotos"},
           
           {'mosaic_path': r"D:\on_uav_data\20200907_Bot\agisoft\20200907_bot_corn_comb_50cm\20200907_bot_corn_comb_50cm_mosaic.tif",
           'dem_path': r"D:\on_uav_data\20200907_Bot\agisoft\20200907_bot_corn_comb_50cm\20200907_bot_corn_comb_50cm_dem.tif", 
           'photos_path': r"D:\on_uav_data\20200907_Bot\agisoft\20200907_bot_corn_comb_50cm\20200907_bot_corn_comb_50cm_orthophotos"},
           
           {'mosaic_path': r"D:\on_uav_data\20200907_Bot\agisoft\20200907_bot_corn_comb_1m\20200907_bot_corn_comb_1m_mosaic.tif",
           'dem_path': r"D:\on_uav_data\20200907_Bot\agisoft\20200907_bot_corn_comb_1m\20200907_bot_corn_comb_1m_dem.tif", 
           'photos_path': r"D:\on_uav_data\20200907_Bot\agisoft\20200907_bot_corn_comb_1m\20200907_bot_corn_comb_1m_orthophotos"}]


# Loop trough the list of input elements
for source in sources:
    image_path = source['photos_path']
    
    # Specify output subdirectory
    out = "\\".join(image_path.split(sep="\\")[2:])
    
    # Store all orthohotos in a ist
    orthos = os.listdir(image_path)
    orthos = [ortho for ortho in orthos if ortho.endswith("tif")]
    
    # Loop through the list of orthophotos
    for image in orthos:
    
        # Open image, copy all metadata and assign each band to a variable
        img = os.path.join(image_path,image)
        src = rio.open(img)
        h = src.height
        w = src.width
        crs = src.crs
        transform = src.transform
        count = src.count
        blue444, blue475, green531, green560, red650, red668, re705, re717, re740, nir = src.read([1,2,3,4,5,6,7,8,9,10])/65535
        src.close()  
        
        # Use the EVI and NDVI index for soil segmentation
        EVI = 2.5 * ((nir - red668) / ((nir + 6 * red668 - 7.5 * blue475) + 1)).astype(np.float32)
        EVI[EVI == 0] = np.nan
        EVI = np.nan_to_num(EVI, posinf=np.nan, neginf=np.nan)
        NDVI = (nir-red668)/(nir+red668).astype(np.float32)
        NDVI[NDVI == 0] = np.nan
        NDVI = np.nan_to_num(NDVI, posinf=np.nan, neginf=np.nan)
    
        # Otsu thresholding
        otsu_soil_evi=threshold_otsu(EVI[~np.isnan(EVI)])
        print("Found automatic threshold t = {}.".format(otsu_soil_evi))
        otsu_soil_ndvi=threshold_otsu(NDVI[~np.isnan(NDVI)])
        print("Found automatic threshold t = {}.".format(otsu_soil_ndvi))
        
        # Creating and applying mask to each band
        mask = np.logical_and(NDVI > otsu_soil_ndvi, EVI > otsu_soil_evi)
        blue444[~mask] = np.nan
        blue475[~mask] = np.nan
        green531[~mask] = np.nan
        green560[~mask] = np.nan
        red650[~mask] = np.nan
        red668[~mask] = np.nan
        re705[~mask] = np.nan
        re717[~mask] = np.nan
        re740[~mask] = np.nan
        nir[~mask] = np.nan
    
        # Visualize segmentation results in true colors
        plt.imshow(np.stack([red668,green560,blue475], axis = 2)/0.3)
        plt.show()
        
        # Re-creating multi-band raster and using original georef params to export GTiff
        multi = np.dstack((blue444, blue475, green531, green560, red650, red668, re705, re717, re740, nir))
        multi2 = np.array(np.moveaxis(multi, 2, 0))  # numpy axes must be in the correct order (kind os transposition)    
        with rio.open(os.path.join(saving_directory,out,image), "w",
                      driver="GTiff",
                      height=h,
                      width=w,
                      count=10,
                      dtype=blue444.dtype,
                      crs=crs,
                      transform=transform) as dst:
            dst.write(multi2)
        # </editor-fold>
    



