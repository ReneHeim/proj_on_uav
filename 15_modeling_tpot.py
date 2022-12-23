#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# version ='1.0'
# ---------------------------------------------------------------------------

''' This script uses the TPOT approach to determine which models are the best ones for our classification problem
This algorithm iterates over different models, preprocessing approaches and hyperparameters.
The two best performing models from this algorithm will be further used to compare the performances of different
vza, which is the main objective of our study
Evaluation metrics (accuracy and f1) as well as the best performing models are stored in a table and in a dictionnary
that is exported as a Json file and can be accessed later'''

# Import libs
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
from tqdm import tqdm as tqdm


param={'n_components':range(2,15)}

cv_class = RepeatedStratifiedKFold(n_splits=4, n_repeats=2)
cv_regr = RepeatedKFold(n_splits=4, n_repeats=2)

out_path = r"D:\on_uav_data"

# Define all necessary input directories in a list  for 2 dates and 3 resolutions and 3 radii (18 elements)
paths=[r"D:\on_uav_data\out_merged\50cm_radius\20200906_Bot\20200906_bot_corn_comb_10cm\on_uav_for_classification.csv",
       r"D:\on_uav_data\out_merged\50cm_radius\20200906_Bot\20200906_bot_corn_comb_50cm\on_uav_for_classification.csv",
       r"D:\on_uav_data\out_merged\50cm_radius\20200906_Bot\20200906_bot_corn_comb_1m\on_uav_for_classification.csv",
       
       r"D:\on_uav_data\out_merged\1m_radius\20200906_Bot\20200906_bot_corn_comb_10cm\on_uav_for_classification.csv",
       r"D:\on_uav_data\out_merged\1m_radius\20200906_Bot\20200906_bot_corn_comb_50cm\on_uav_for_classification.csv",
       r"D:\on_uav_data\out_merged\1m_radius\20200906_Bot\20200906_bot_corn_comb_1m\on_uav_for_classification.csv",
       
       r"D:\on_uav_data\out_merged\2m_radius\20200906_Bot\20200906_bot_corn_comb_10cm\on_uav_for_classification.csv",
       r"D:\on_uav_data\out_merged\2m_radius\20200906_Bot\20200906_bot_corn_comb_50cm\on_uav_for_classification.csv",
       r"D:\on_uav_data\out_merged\2m_radius\20200906_Bot\20200906_bot_corn_comb_1m\on_uav_for_classification.csv",      
       
       
       r"D:\on_uav_data\out_merged\50cm_radius\20200907_Bot\20200907_bot_corn_comb_10cm\on_uav_for_classification.csv",
       r"D:\on_uav_data\out_merged\50cm_radius\20200907_Bot\20200907_bot_corn_comb_50cm\on_uav_for_classification.csv",
       r"D:\on_uav_data\out_merged\50cm_radius\20200907_Bot\20200907_bot_corn_comb_1m\on_uav_for_classification.csv",
       
       r"D:\on_uav_data\out_merged\1m_radius\20200907_Bot\20200907_bot_corn_comb_10cm\on_uav_for_classification.csv",
       r"D:\on_uav_data\out_merged\1m_radius\20200907_Bot\20200907_bot_corn_comb_50cm\on_uav_for_classification.csv",
       r"D:\on_uav_data\out_merged\1m_radius\20200907_Bot\20200907_bot_corn_comb_1m\on_uav_for_classification.csv",
       
       r"D:\on_uav_data\out_merged\2m_radius\20200907_Bot\20200907_bot_corn_comb_10cm\on_uav_for_classification.csv",
       r"D:\on_uav_data\out_merged\2m_radius\20200907_Bot\20200907_bot_corn_comb_50cm\on_uav_for_classification.csv",
       r"D:\on_uav_data\out_merged\2m_radius\20200907_Bot\20200907_bot_corn_comb_1m\on_uav_for_classification.csv"      
       ]

# Define variable which will contain different columns of the database
dates = []
radius = []
resolution =[]
parameter = []
vza_list_col = []
best_model_list = []
accuray_list = []
kappa_list = []

labels_angle={'20200906' : ['0-13','13-18','18-23','23-30','30-60'],
              '20200907' : ['0-11','11-16','16-19','19-23','23-60']}

# Loop through databases
for path in tqdm(paths):
    print(f'current path : {path}')
    datepath = path.split('\\')[4][:-4]
    X, Y, vza_class = calculate_indices(path, method='classification', date = datepath)
    vza_list= labels_angle[datepath]
    # Loop through target variables
    for i, target in tqdm({'lai':'lai_cat_a', 'chl':'chl_cat_a', 'nbi':'nbi_cat_a'}.items()):
        print(f'current variable : {i}')
        classification_results={}
        X_no_nan, Y_no_nan = X[~(vza_class.astype(str)=='nan')], Y[~(vza_class.astype(str)=='nan')]
        X_train, X_test, Y_train, Y_test = split_train_test_data(X_no_nan, Y_no_nan) # This function is imported from smac
        # Define at TPOT object and fit the model to search for the best model on the whole database
        model = TPOTClassifier(generations=50, population_size=50, scoring='accuracy', cv=cv_class, n_jobs=-1)
        model.fit(X_train,Y_train[target])
        best_model=model.fitted_pipeline_
        best_model.fit(X_train,Y_train[target])
        Y_pred=best_model.predict(X_test)
        # Compute evaluation metrics
        accuracy=accuracy_score(Y_test[target],Y_pred)
        kappa=cohen_kappa_score(Y_test[target],Y_pred)
        confusion_mat=confusion_matrix(Y_test[target],Y_pred, labels=['low','medium','high'])
        # Store evaluation metrics in dictionnary
        classification_results['all'] = {}
        classification_results['all']['bes_model'] = str(best_model)
        classification_results['all']['accuracy'] = accuracy
        classification_results['all']['kappa'] = kappa
        classification_results['all']['confusion_mat'] = confusion_mat.tolist()
        # Store evaluation metrics in list which will serve to form the final table
        dates += [path.split('\\')[4][:-4]]
        radius += [path.split('\\')[3][:-7]]
        resolution += [path.split('\\')[5][23:]]
        parameter += [i]
        vza_list_col += ['all']
        best_model_list += [str(best_model)]
        accuray_list += [accuracy]
        kappa_list += [kappa]
        # Loop through vza classes
        for vza_angle in tqdm(vza_list):
            print(f'current angle : {vza_angle}')
            X_vza, Y_vza = X[vza_class==vza_angle], Y[vza_class==vza_angle]
            X_train, X_test, Y_train, Y_test = split_train_test_data(X_vza, Y_vza)
            # Fit the previously obtained best model for each vza
            best_model=model.fitted_pipeline_
            best_model.fit(X_train,Y_train[target])
            Y_pred=best_model.predict(X_test)
            # Compute evaluation matrix
            accuracy=accuracy_score(Y_test[target],Y_pred)
            kappa=cohen_kappa_score(Y_test[target],Y_pred)
            confusion_mat=confusion_matrix(Y_test[target],Y_pred, labels=['low','medium','high'])
            # Store evaluation metrics in dictionnary
            classification_results[vza_angle] = {}
            classification_results[vza_angle]['bes_model'] = str(best_model)
            classification_results[vza_angle]['accuracy'] = accuracy
            classification_results[vza_angle]['kappa'] = kappa
            classification_results[vza_angle]['confusion_mat'] = confusion_mat.tolist()
            # Store evaluation metrics in list which will serve to form the final table
            dates += [path.split('\\')[4][:-4]]
            radius += [path.split('\\')[3][:-7]]
            resolution += [path.split('\\')[5][23:]]
            parameter += [i]
            vza_list_col += [vza_angle]
            best_model_list += [str(best_model)]
            accuray_list += [accuracy]
            kappa_list += [kappa]
        # Save the dictionnary to a json file
        file_name = os.path.splitext(path)[0] + "_" + i + ".json"
        json.dump(classification_results, open(file_name, 'w')) 
# Save datebase to csv file            
data_base = np.transpose(np.array([dates, radius, resolution, parameter, vza_list_col, best_model_list, accuray_list, kappa_list]))
data_base = pd.DataFrame(data_base, columns = ['dates', 'radius', 'resolution', 'parameter', 'vza_list', 'best_model_list', 'accuray_list', 'kappa_list'])
data_base.to_csv(os.path.join(out_path, 'data_all.csv'))