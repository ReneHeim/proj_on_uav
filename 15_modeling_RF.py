# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:51:04 2022

@author: hp
"""


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
from sklearn.pipeline import Pipeline
from tqdm import tqdm as tqdm

from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param={'n_components':range(2,15)}

cv_class = RepeatedStratifiedKFold(n_splits=4, n_repeats=2)
cv_regr = RepeatedKFold(n_splits=5, n_repeats=3)

out_path = r"D:\on_uav_data"

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

for path in tqdm(paths):
    print(f'current path : {path}')
    datepath = path.split('\\')[4][:-4]
    X, Y, vza_class = calculate_indices(path, method='classification', date = datepath)
    vza_list= labels_angle[datepath]
    
    for i, target in tqdm({'lai':'lai_cat_a', 'chl':'chl_cat_a', 'nbi':'nbi_cat_a'}.items()):
        print(f'current variable : {i}')
        classification_results={}
        for vza_angle in tqdm(vza_list):
            print(f'current angle : {vza_angle}')
            X_vza, Y_vza = X[vza_class==vza_angle], Y[vza_class==vza_angle]
            X_train, X_test, Y_train, Y_test = split_train_test_data(X_vza, Y_vza)
            
                    
            model = RandomForestClassifier()
            model.fit(X_train, Y_train[target])
            Y_pred = model.predict(X_test)
            
            
            accuracy=accuracy_score(Y_test[target],Y_pred)
            kappa=cohen_kappa_score(Y_test[target],Y_pred)
            confusion_mat=confusion_matrix(Y_test[target],Y_pred, labels=['low','medium','high'])
            
            classification_results[vza_angle] = {}
            classification_results[vza_angle]['bes_model'] = str('RF')
            classification_results[vza_angle]['accuracy'] = accuracy
            classification_results[vza_angle]['kappa'] = kappa
            classification_results[vza_angle]['confusion_mat'] = confusion_mat.tolist()
            
            dates += [path.split('\\')[4][:-4]]
            radius += [path.split('\\')[3][:-7]]
            resolution += [path.split('\\')[5][23:]]
            parameter += [i]
            vza_list_col += [vza_angle]
            best_model_list += [str('RF')]
            accuray_list += [accuracy]
            kappa_list += [kappa]
        
        file_name = os.path.splitext(path)[0] + "_" + i + "_pls.json"
        json.dump(classification_results, open(file_name, 'w')) 
            
data_base = np.transpose(np.array([dates, radius, resolution, parameter, vza_list_col, best_model_list, accuray_list, kappa_list]))
data_base = pd.DataFrame(data_base, columns = ['dates', 'radius', 'resolution', 'parameter', 'vza_list', 'best_model_list', 'accuray_list', 'kappa_list'])
data_base.to_csv(os.path.join(out_path, 'data_rf.csv'))


