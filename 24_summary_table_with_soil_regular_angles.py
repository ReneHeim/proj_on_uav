#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# version ='1.0'
# ---------------------------------------------------------------------------

"""
This code performs the classification and puts the results in a tabular form regular binning.  
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
import json
# Install tpot
# Install torch
from smac_functions import *
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from tqdm import tqdm as tqdm

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split


# Define cross-validation objects. One cross-validation object with 5 folds and 2 repeats for 10 cm resolution 
# dataset (memory constrraint) and another one with 5 folds and 3 repeats for all other data sets

cv_class = [RepeatedStratifiedKFold(n_splits=5, n_repeats=3),
            RepeatedStratifiedKFold(n_splits=5, n_repeats=2)]

# Define output path
out_path = r"Q:\on_uav_data"

# Define input path (2 dates and 3 resolution = 6 paths). Each element of the list is another list containing two elements:
# the first element is relative to orthomphotos (vza) and the second to the orthomosaic  
paths=[[r"Q:\on_uav_data\out_merged\2m_radius\20200906_Bot\20200906_bot_corn_comb_10cm\on_uav_for_classification.feather",
       r"Q:\on_uav_data\out_merged\2m_radius\20200906_Bot\20200906_bot_corn_comb_10cm\on_uav_for_classification_mosaic.csv"],
       
       [r"Q:\on_uav_data\out_merged\2m_radius\20200906_Bot\20200906_bot_corn_comb_50cm\on_uav_for_classification.feather",
        r"Q:\on_uav_data\out_merged\2m_radius\20200906_Bot\20200906_bot_corn_comb_50cm\on_uav_for_classification_mosaic.csv"],
       
       [r"Q:\on_uav_data\out_merged\2m_radius\20200906_Bot\20200906_bot_corn_comb_1m\on_uav_for_classification.feather",
        r"Q:\on_uav_data\out_merged\2m_radius\20200906_Bot\20200906_bot_corn_comb_1m\on_uav_for_classification_mosaic.csv"],
       
       [r"Q:\on_uav_data\out_merged\2m_radius\20200907_Bot\20200907_bot_corn_comb_10cm\on_uav_for_classification.feather",
       r"Q:\on_uav_data\out_merged\2m_radius\20200907_Bot\20200907_bot_corn_comb_10cm\on_uav_for_classification_mosaic.csv"],
       
       [r"Q:\on_uav_data\out_merged\2m_radius\20200907_Bot\20200907_bot_corn_comb_50cm\on_uav_for_classification.feather",
        r"Q:\on_uav_data\out_merged\2m_radius\20200907_Bot\20200907_bot_corn_comb_50cm\on_uav_for_classification_mosaic.csv"],
       
       [r"Q:\on_uav_data\out_merged\2m_radius\20200907_Bot\20200907_bot_corn_comb_1m\on_uav_for_classification.feather",
       r"Q:\on_uav_data\out_merged\2m_radius\20200907_Bot\20200907_bot_corn_comb_1m\on_uav_for_classification_mosaic.csv"]
       ]


# Define models to use
models = {'RF': RandomForestClassifier(),
          'Etra trees': ExtraTreesClassifier()}

# Define lists that will contain results
dates = []
resolution =[]
parameter = []
model_list = []
approaches = []
mean_accuracy = []
sd_accuracy = []
mean_f1 = []
sd_f1 = []

#Define the binning strategy
#labels_angle={'20200906' : ['0-13','13-18','18-23','23-30','30-60'],
#              '20200907' : ['0-11','11-16','16-19','19-23','23-60']}
labels_angle={'20200906' : ['0-10','10-20','20-30','30-40','40-50','50-60'],
              '20200907' : ['0-10','10-20','20-30','30-40','40-50','50-60']}

# Start with the gearation of results
# Loop through paths
for path in tqdm(paths):
    path_ortho = path[0]
    path_mosaic = path[1]
    print(f'current path : {path_ortho}')
    datepath = path_ortho.split('\\')[4][:-4]
    
    # Loop trough different parameters
    for i, target in tqdm({'lai':'lai_cat_a', 'chl':'chl_cat_a', 'nbi':'nbi_cat_a'}.items()):
        print(f'current variable : {i}')
        
        # Fit the 2 different models to the orthomosaic first
        approach = 'mosaic'
        X, Y = calculate_bands_mosaic(path[1], method='classification', date = datepath)
        
        # Loop through the 2 models
        for j, model in tqdm(models.items()):
            print(f'current model : {j}')
            # Fit the model
            res = path_ortho.split('\\')[5][23:]
            cv_score_accuracy = cross_val_score(model, X = X, y = Y[target], cv = cv_class[res=='10cm'], n_jobs= -1, scoring = 'accuracy')
            cv_score_f1 = cross_val_score(model, X = X, y = Y[target], cv = cv_class[res=='10cm'], n_jobs= -1, scoring = 'f1_weighted')
            # Conpute evalution metrics
            accuracy_mean = cv_score_accuracy.mean()
            accuracy_sd = cv_score_accuracy.std()
            f1_mean = cv_score_accuracy.mean()
            f1_sd = cv_score_accuracy.std()
            # Save results in predifined lists
            dates += [path_ortho.split('\\')[4][:-4]]#radius += [path.split('\\')[3][:-7]]
            resolution += [path_ortho.split('\\')[5][23:]]
            parameter += [i]
            model_list += [j]
            approaches += [approach]
            mean_accuracy += [accuracy_mean]
            sd_accuracy += [accuracy_sd]
            mean_f1 += [f1_mean]
            sd_f1 += [f1_sd]
            
        # Then Fit the 2 different models to different classes of vza angles
        X, Y, vza_class = calculate_bands(path[0], method='classification', date = datepath, binning = 'regular')
        vza_list= labels_angle[datepath]
        # Loop through different vza and subset
        for vza_angle in tqdm(vza_list):
            print(f'current angle : {vza_angle}')
            approach = vza_angle
            X_vza, Y_vza = X[vza_class==vza_angle], Y[vza_class==vza_angle]
            # Loop through the 2 models
            for j, model in tqdm(models.items()):
                if len(X_vza)>0:
                    print(f'current model : {j}')
                    # Fit the model
                    cv_score_accuracy = cross_val_score(model, X = X_vza, y = Y_vza[target], cv = cv_class[res=='10cm'], n_jobs= -1, scoring = 'accuracy')
                    cv_score_f1 = cross_val_score(model, X = X_vza, y = Y_vza[target], cv = cv_class[res=='10cm'], n_jobs= -1, scoring = 'f1_weighted')
                    # Conpute evalution metrics
                    accuracy_mean = cv_score_accuracy.mean()
                    accuracy_sd = cv_score_accuracy.std()
                    f1_mean = cv_score_f1.mean()
                    f1_sd = cv_score_f1.std()
                    # Save results in predifined lists
                    dates += [path_ortho.split('\\')[4][:-4]]
                    resolution += [path_ortho.split('\\')[5][23:]]
                    parameter += [i]
                    model_list += [j]
                    approaches += [approach]
                    mean_accuracy += [accuracy_mean]
                    sd_accuracy += [accuracy_sd]
                    mean_f1 += [f1_mean]
                    sd_f1 += [f1_sd]
            
    # Store results in a panda table and save after each of the 6 paths iteration in case something goes wrong afterwards
    data_base = np.transpose(np.array([dates, resolution, parameter, model_list, approaches, mean_accuracy, sd_accuracy, mean_f1, sd_f1]))
    data_base = pd.DataFrame(data_base, columns = ['dates', 'resolution', 'parameter', 'model', 'approaches', 'mean_accuracy', 'sd_accuracy', 'mean_f1', 'sd_f1'])
    data_base.to_csv(os.path.join(out_path, 'data_summary_table_with_soil_regular_bin.csv'))





