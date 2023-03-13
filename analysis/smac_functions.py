#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/07/08
# version ='1.0'
# ---------------------------------------------------------------------------

'''
This code copiles all functions that will be called later by other codes. For the sake of clarity
these functions are defined in this separated piece of code.
'''





def xyval(A):
    """
    Function to list all pixel coords including their associated value starting from the  upper left corner (?)
    :param A: raster band as numpy array
    :return: x and y of each pixel and the associated value
    """
    import numpy as np
    x, y = np.indices(A.shape)
    return x.ravel(), y.ravel(), A.ravel()



  
def calculate_bands_mosaic(database_path,method='classification'):
    import pandas as pd
    import numpy as np
    if database_path.endswith(".csv"):
        data=pd.read_csv(database_path)
    else:
        data=pd.read_feather(database_path)
    data[["band1","band2","band3","band4","band5","band6","band7","band8","band9","band10"]]=data[["band1","band2","band3","band4","band5","band6","band7","band8","band9","band10"]]/65535
    data=data.rename(columns={"band1": "blue444",
                                "band2": "blue475",
                                "band3": "green531",
                                "band4": "green560",
                                "band5": "red650",
                                "band6": "red668",
                                "band7": "re705",
                                "band8": "re717",
                                "band9": "re740",
                                "band10": "nir842"})

    blue444=data['blue444']
    blue475=data['blue475']
    green531=data['green531']
    green560=data['green560']
    red650=data['red650']
    red668=data['red668']
    re705=data['re705']
    re717=data['re717']
    re740=data['re740']
    nir842=data['nir842']
    
    X=pd.concat([blue444, blue475, green531, green560, red650, red668, re705, re717, re740, nir842], 
                 axis=1)
    X.columns=['blue444', 'blue475', 'green531', 'green560', 'red650', 'red668', 're705', 're717', 're740', 'nir842']
    
    for VI in X.columns:
        X[VI]=np.where(X[VI]==np.inf, 
                       np.max(X[VI][~ (X[VI]==np.inf)]),
                       X[VI])
        X[VI]=np.where(X[VI]==-np.inf, 
                       np.min(X[VI][~ (X[VI]==-np.inf)]),
                       X[VI])
        X[VI]=np.where(np.isnan(X[VI]), 
                       np.nanmean(X[VI]),
                       X[VI])
        
    data['nbi_cat_a'] = pd.qcut(data['nbi'], 3, labels=["low", "medium", "high"])
    
    if method == 'classification':
        Y=data[['lai_cat_a','chl_cat_a', 'nbi_cat_a', 'glai_cat_a']]
    else :
        Y=data[['lai','chl','nbi', 'glai']]  
    
    return X, Y
  
def calculate_bands(database_path, method='classification', dataset='All_flight'):
    import pandas as pd
    import numpy as np
    if database_path.endswith(".csv"):
        data=pd.read_csv(database_path)
    else:
        data=pd.read_feather(database_path)
    data[["band1","band2","band3","band4","band5","band6","band7","band8","band9","band10"]]=data[["band1","band2","band3","band4","band5","band6","band7","band8","band9","band10"]]/65535
    data=data.rename(columns={"band1": "blue444",
                                "band2": "blue475",
                                "band3": "green531",
                                "band4": "green560",
                                "band5": "red650",
                                "band6": "red668",
                                "band7": "re705",
                                "band8": "re717",
                                "band9": "re740",
                                "band10": "nir842"})

    blue444=data['blue444']
    blue475=data['blue475']
    green531=data['green531']
    green560=data['green560']
    red650=data['red650']
    red668=data['red668']
    re705=data['re705']
    re717=data['re717']
    re740=data['re740']
    nir842=data['nir842']

    X=pd.concat([blue444, blue475, green531, green560, red650, red668, re705, re717, re740, nir842], 
                 axis=1)
    X.columns=['blue444', 'blue475', 'green531', 'green560', 'red650', 'red668', 're705', 're717', 're740', 'nir842']
    
    for VI in X.columns:
        X[VI]=np.where(X[VI]==np.inf, 
                       np.max(X[VI][~ (X[VI]==np.inf)]),
                       X[VI])
        X[VI]=np.where(X[VI]==-np.inf, 
                       np.min(X[VI][~ (X[VI]==-np.inf)]),
                       X[VI])
        X[VI]=np.where(np.isnan(X[VI]), 
                       np.nanmean(X[VI]),
                       X[VI])       
    
    if method == 'classification':
        Y=data[['lai_cat_a','chl_cat_a', 'nbi_cat_a', 'glai_cat_a']]
    else :
        Y=data[['lai','chl','nbi']]
    
    dataset = database_path.split('\\')[-2]
    labels_vza_angle={'All_flight' : ['0-14','14-19','19-25','25-50'],
                      'Nadir_flight' : ['0-12','12-17','17-21','21-35']}
    cut_angle_vza={'All_flight' : [0, 14, 19, 25, 50],
                   'Nadir_flight' : [0, 12, 17, 21, 35]}
    vza_class=pd.cut(data.vza, cut_angle_vza[dataset], labels=labels_vza_angle[dataset])
    
    
    data['vaa'] = np.abs(data['vaa'])
    labels_vaa_angle={'All_flight' : ['back scattering','side scattering','forward scattering'],
                      'Nadir_flight' : ['back scattering','side scattering','forward scattering']}
    cut_angle_vaa={'All_flight' : [0, 60, 120, 180],
                   'Nadir_flight' : [0, 60, 120, 180]}
    vaa_class=pd.cut(data.vaa, cut_angle_vaa[dataset], labels=labels_vaa_angle[dataset])
    
    return X, Y, vza_class, vaa_class
