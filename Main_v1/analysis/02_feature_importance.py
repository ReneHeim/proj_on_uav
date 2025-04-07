#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# version ='1.0'
# ---------------------------------------------------------------------------

"""
This code performs the classification and puts the results in a tabular form irregular binning.  
It considers as factors the date (20200906 and 20200907), the resolution (10 cm, 50 cm, 1m), 
different parameters (chl, lai, nbi), different models (RF, Extra trees) , and 
different approaches ( orthomosaic or vza classes).
Two evaluation metrics are computed. Namely the overall accuracy (mean and sd) and 
the f1 score (mean and sd) using a repeated stratified cross-validation approach.
"""

# Loading libs 
import os
import pandas as pd
import numpy as np
from smac_functions import *
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm as tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


# Define cross-validation objects. One cross-validation object with 5 folds and 2 repeats for 10 cm resolution 
# dataset (memory constrraint) and another one with 5 folds and 3 repeats for all other data sets
cv_class = [RepeatedStratifiedKFold(n_splits=5, n_repeats=3),
            RepeatedStratifiedKFold(n_splits=5, n_repeats=2)]

# Define output path
out_path = r"D:\on_uav_data\proc\Classification results"

# Define input path (2 dates and 3 resolution = 6 paths). Each element of the list is another list containing two elements:
# the first element is relative to orthomphotos (vza) and the second to the orthomosaic  


paths=[[r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_20cm\All_flight\on_uav_for_classification.feather",
       r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_20cm\All_flight\on_uav_for_classification_mosaic.csv"],
       
       [r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_50cm\All_flight\on_uav_for_classification.feather",
        r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_50cm\All_flight\on_uav_for_classification_mosaic.csv"],
       
       [r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_1m\All_flight\on_uav_for_classification.feather",
        r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_1m\All_flight\on_uav_for_classification_mosaic.csv"],
       
       [r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_20cm\Nadir_flight\on_uav_for_classification.feather",
        r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_20cm\Nadir_flight\on_uav_for_classification_mosaic.csv"],
        
        [r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_50cm\Nadir_flight\on_uav_for_classification.feather",
         r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_50cm\Nadir_flight\on_uav_for_classification_mosaic.csv"],
        
        [r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_1m\Nadir_flight\on_uav_for_classification.feather",
         r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_1m\Nadir_flight\on_uav_for_classification_mosaic.csv"],
        
        
        [r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_20cm\All_flight\on_uav_for_classification.feather",
        r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_20cm\All_flight\on_uav_for_classification_mosaic.csv"],
        
        [r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_50cm\All_flight\on_uav_for_classification.feather",
         r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_50cm\All_flight\on_uav_for_classification_mosaic.csv"],
        
        [r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_1m\All_flight\on_uav_for_classification.feather",
         r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_1m\All_flight\on_uav_for_classification_mosaic.csv"],
        
        [r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_20cm\Nadir_flight\on_uav_for_classification.feather",
         r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_20cm\Nadir_flight\on_uav_for_classification_mosaic.csv"],
         
         [r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_50cm\Nadir_flight\on_uav_for_classification.feather",
          r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_50cm\Nadir_flight\on_uav_for_classification_mosaic.csv"],
         
         [r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_1m\Nadir_flight\on_uav_for_classification.feather",
          r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_1m\Nadir_flight\on_uav_for_classification_mosaic.csv"]] 


# Define models to use
models = {'RF': RandomForestClassifier(),
          'Etra trees': ExtraTreesClassifier()}

# Define lists that will contain results
dates = []
resolution =[]
parameter = []
model_list = []
approaches = []
dataset_list = []
vza_angle_list = []
vaa_angle_list = []
feature_imp = []

#Define the binning strategy
labels_vza_angle={'All_flight' : ['0-14','14-19','19-25','25-50'],
                  'Nadir_flight' : ['0-12','12-17','17-21','21-35']}

labels_vaa_angle={'All_flight' : ['back scattering','side scattering','forward scattering'],
                  'Nadir_flight' : ['back scattering','side scattering','forward scattering']}


# Start with the generation of results
# Loop through paths
for path in tqdm(paths):
    path_ortho = path[0]
    path_mosaic = path[1]
    print(f'current path : {path_ortho}')
    dataset = path_ortho.split('\\')[-2]
    
    # Loop trough different parameters
    for i, target in tqdm({'lai':'lai_cat_a', 'chl':'chl_cat_a', 'nbi':'nbi_cat_a', 'glai':'glai_cat_a'}.items()):
        print(f'current variable : {i}')
        
        # Fit the 2 different models to the orthomosaic first
        approach = 'mosaic'
        vza_angle  = 'mosaic'
        vaa_angle= 'mosaic'
        X, Y = calculate_bands_mosaic(path[1], method='classification')
        
        # Loop through the 2 models
        for j, model in tqdm(models.items()):
            print(f'current model : {j}')
            # Fit the model
            res = path_ortho.split('\\')[6][23:]
            mod = model
            mod.fit(X,Y[target])
            feature_importance = mod.feature_importances_
            # Save results in predifined lists
            dates += [path_ortho.split('\\')[5][:-4]]
            resolution += [path_ortho.split('\\')[6][23:]]
            parameter += [i]
            model_list += [j]
            approaches += [approach]
            dataset_list += [dataset]
            vza_angle_list += [vza_angle]
            vaa_angle_list += [vaa_angle]
            feature_imp += [feature_importance]
        
        # Fit the 2 different models to the all ortho together, regardless of the anglle
        approach = 'orthophoto'
        vza_angle  = 'all'
        vaa_angle= 'all'
        X, Y, _ , _ = calculate_bands(path[0], method='classification')
        for j, model in tqdm(models.items()):
            print(f'current model : {j}')
            # Fit the model
            res = path_ortho.split('\\')[6][23:]
            mod = model
            mod.fit(X,Y[target])
            feature_importance = mod.feature_importances_
            # Save results in predifined lists
            dates += [path_ortho.split('\\')[5][:-4]]
            resolution += [path_ortho.split('\\')[6][23:]]
            parameter += [i]
            model_list += [j]
            approaches += [approach]
            dataset_list += [dataset]
            vza_angle_list += [vza_angle]
            vaa_angle_list += [vaa_angle]
            feature_imp += [feature_importance]
            
            
        # Then Fit the 2 different models to different classes of vza angles
        X, Y, vza_class, vaa_class = calculate_bands(path[0], method='classification')
        vza_list= labels_vza_angle[dataset]
        vaa_list= labels_vaa_angle[dataset]
        # Loop through different vza and subset
        for vza_angle in tqdm(vza_list):
            print(f'current vza angle : {vza_angle}')
            for vaa_angle in tqdm(vaa_list):
                print(f'current vaa angle : {vaa_angle}')
                approach = 'orthophoto'
                X_vza, Y_vza = X[np.logical_and(vza_class==vza_angle, vaa_class==vaa_angle)], Y[np.logical_and(vza_class==vza_angle, vaa_class==vaa_angle)]
                # Loop through the 2 models
                for j, model in tqdm(models.items()):
                    print(f'current model : {j}')
                    # Fit the model
                    mod = model
                    mod.fit(X_vza,Y_vza[target])
                    feature_importance = mod.feature_importances_
                    # Save results in predifined lists
                    dates += [path_ortho.split('\\')[5][:-4]]
                    resolution += [path_ortho.split('\\')[6][23:]]
                    parameter += [i]
                    model_list += [j]
                    approaches += [approach]
                    dataset_list += [dataset]
                    vza_angle_list += [vza_angle]
                    vaa_angle_list += [vaa_angle]
                    feature_imp += [feature_importance]
            
    # Store results in a panda table and save after each of the 6 paths iteration in case something goes wrong afterwards
    feature_imp_frame = np.array(feature_imp)
    feature_imp_frame = pd.DataFrame(feature_imp_frame, columns= X.columns)
    data_base_ind = np.transpose(np.array([dates, resolution, parameter, model_list, approaches, dataset_list, vza_angle_list, vaa_angle_list]))
    data_base_ind = pd.DataFrame(data_base_ind, columns = ['dates', 'resolution', 'parameter', 'model', 'approaches', 'dataset_list', 'vza_angle_list', 'vaa_angle_list'])
    data_base = pd.concat([data_base_ind, feature_imp_frame], axis = 1)
    data_base.to_csv(os.path.join(out_path, 'feature_importance.csv'))


