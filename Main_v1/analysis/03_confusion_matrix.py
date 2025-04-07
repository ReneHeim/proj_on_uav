# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 15:23:38 2022

@author: hp
"""
# Loading libs 
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from smac_functions import *
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from tqdm import tqdm as tqdm

out_dir = r"D:\on_uav_data\proc\Confusion matrix"


paths=[
       [r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_1m\All_flight\on_uav_for_classification.feather",
        r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_1m\All_flight\on_uav_for_classification_mosaic.csv"],
       
        
        [r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_1m\Nadir_flight\on_uav_for_classification.feather",
         r"D:\on_uav_data\proc\merge\2m_radius\20200906_Bot\20200906_bot_corn_comb_1m\Nadir_flight\on_uav_for_classification_mosaic.csv"],
        
        
        
        [r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_1m\All_flight\on_uav_for_classification.feather",
         r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_1m\All_flight\on_uav_for_classification_mosaic.csv"],
        
         
         [r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_1m\Nadir_flight\on_uav_for_classification.feather",
          r"D:\on_uav_data\proc\merge\2m_radius\20200907_Bot\20200907_bot_corn_comb_1m\Nadir_flight\on_uav_for_classification_mosaic.csv"],
       ]

# Define models to use
models = {'RF': RandomForestClassifier(),
          'Etra trees': ExtraTreesClassifier()}

#Define the binning strategy
labels_vza_angle={'All_flight' : ['0-14','14-19','19-25','25-50'],
                  'Nadir_flight' : ['0-12','12-17','17-21','21-35']}

labels_vaa_angle={'All_flight' : ['back scattering','side scattering','forward scattering'],
                  'Nadir_flight' : ['back scattering','side scattering','forward scattering']}

# Loop trough different paths
for path in tqdm(paths):
    path_ortho = path[0]
    path_mosaic = path[1]
    print(f'current path : {path_ortho}')
    dataset = path_ortho.split('\\')[-2]
    date = path_ortho.split('\\')[5][:-4]
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
            confusion_mat_list = []
            for rs in range(0,75,5):
                X_i, Y_i = X, Y
                X_train, X_test, Y_train, Y_test = train_test_split(X_i, Y_i, test_size= 0.2, random_state= rs )
                mod = model
                mod.fit(X_train, Y_train[target])
                Y_pred = mod.predict(X_test)
                
                confusion_mat = confusion_matrix(Y_test[target],Y_pred, labels=['low','medium','high'])
                confusion_mat_list += [confusion_mat]
            confusion_mat = np.round(np.mean(confusion_mat_list, axis=0), 0)
            
            df_confusion_mat = pd.DataFrame(confusion_mat, columns= ['low','medium','high'], index = ['low','medium','high'])
            oa = np.sum([df_confusion_mat.loc[i,i] for i in df_confusion_mat.columns])/np.sum(confusion_mat)
            for ii in df_confusion_mat.columns:
                ua=df_confusion_mat.loc[ii,ii]/df_confusion_mat.loc[:,ii].sum()
                df_confusion_mat.loc['user accuracy', ii] = ua
                pa=df_confusion_mat.loc[ii,ii]/df_confusion_mat.loc[ii,:].sum()
                df_confusion_mat.loc[ii, 'producer accuracy'] = pa
            df_confusion_mat.loc['user accuracy', 'producer accuracy'] = oa
            
            # Save results as csv file
            df_confusion_mat.to_csv(os.path.join(out_dir, date +"_"+ dataset +"_"+ i +"_"+ j +"_"+ approach +"_"+ vza_angle +"_"+ vaa_angle +".csv"))

           

        
        # Fit the 2 different models to the all ortho together, regardless of the anglle
        approach = 'orthophoto'
        vza_angle  = 'all'
        vaa_angle= 'all'
        X, Y, _ , _ = calculate_bands(path[0], method='classification')
        for j, model in tqdm(models.items()):
            print(f'current model : {j}')
            # Fit the model
            confusion_mat_list = []
            for rs in range(0,75,5):
                X_i, Y_i = X, Y
                X_train, X_test, Y_train, Y_test = train_test_split(X_i, Y_i, test_size= 0.2, random_state= rs )
                mod = model
                mod.fit(X_train, Y_train[target])
                Y_pred = mod.predict(X_test)
                
                confusion_mat = confusion_matrix(Y_test[target],Y_pred, labels=['low','medium','high'])
                confusion_mat_list += [confusion_mat]
            confusion_mat = np.round(np.mean(confusion_mat_list, axis=0), 0)
            
            df_confusion_mat = pd.DataFrame(confusion_mat, columns= ['low','medium','high'], index = ['low','medium','high'])
            oa = np.sum([df_confusion_mat.loc[i,i] for i in df_confusion_mat.columns])/np.sum(confusion_mat)
            for ii in df_confusion_mat.columns:
                ua=df_confusion_mat.loc[ii,ii]/df_confusion_mat.loc[:,ii].sum()
                df_confusion_mat.loc['user accuracy', ii] = ua
                pa=df_confusion_mat.loc[ii,ii]/df_confusion_mat.loc[ii,:].sum()
                df_confusion_mat.loc[ii, 'producer accuracy'] = pa
            df_confusion_mat.loc['user accuracy', 'producer accuracy'] = oa
            
            # Save results as csv file
            df_confusion_mat.to_csv(os.path.join(out_dir, date +"_"+ dataset +"_"+ i +"_"+ j +"_"+ approach +"_"+ vza_angle +"_"+ vaa_angle +".csv"))
            
            
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
                    confusion_mat_list = []
                    for rs in range(0,75,5):
                        X_i, Y_i = X_vza, Y_vza
                        X_train, X_test, Y_train, Y_test = train_test_split(X_i, Y_i, test_size= 0.2, random_state= rs )
                        mod = model
                        mod.fit(X_train, Y_train[target])
                        Y_pred = mod.predict(X_test)
                        
                        confusion_mat = confusion_matrix(Y_test[target],Y_pred, labels=['low','medium','high'])
                        confusion_mat_list += [confusion_mat]
                    confusion_mat = np.round(np.mean(confusion_mat_list, axis=0), 0)
                    
                    df_confusion_mat = pd.DataFrame(confusion_mat, columns= ['low','medium','high'], index = ['low','medium','high'])
                    oa = np.sum([df_confusion_mat.loc[i,i] for i in df_confusion_mat.columns])/np.sum(confusion_mat)
                    for ii in df_confusion_mat.columns:
                        ua=df_confusion_mat.loc[ii,ii]/df_confusion_mat.loc[:,ii].sum()
                        df_confusion_mat.loc['user accuracy', ii] = ua
                        pa=df_confusion_mat.loc[ii,ii]/df_confusion_mat.loc[ii,:].sum()
                        df_confusion_mat.loc[ii, 'producer accuracy'] = pa
                    df_confusion_mat.loc['user accuracy', 'producer accuracy'] = oa
                    
                    # Save results as csv file
                    df_confusion_mat.to_csv(os.path.join(out_dir, date +"_"+ dataset +"_"+ i +"_"+ j +"_"+ approach +"_"+ vza_angle +"_"+ vaa_angle +".csv"))
                    