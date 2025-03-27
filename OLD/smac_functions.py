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



def linreg(x,y, figpath):
    
    '''
    This function is used to define a linear regression object. Obtain the R2, slope, RMSE and intercept from it.
    Then plot the fitted curve against the sample points
    '''
    
    # imports
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # sckit-learn implementation

    # reshape x and y
    x = x.values.reshape(-1,1)
    y = y.values.reshape(-1,1)



    # Model initialization
    regression_model = LinearRegression()
    # Fit the data(train the model)
    regression_model.fit(x, y)
    # Predict
    y_predicted = regression_model.predict(x)

    # model evaluation
    rmse = mean_squared_error(y, y_predicted)
    r2 = r2_score(y, y_predicted)

    # printing values
    print('Slope:' ,regression_model.coef_)
    print('Intercept:', regression_model.intercept_)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)

    # plotting values

    # data points
    plt.scatter(x, y, s=10)
    plt.title('Linear Regression:')
    plt.xlabel('x')
    plt.ylabel('y')

    # predicted values
    plt.plot(x, y_predicted, color='r')
    plt.savefig(figpath)
    plt.close()

    return dict(slope = regression_model.coef_[0][0],
                intercept = regression_model.intercept_[0])





def getdualex(filepath, seperator):

    '''
    This function imports a Dualex excel spreadsheed, cleans the output, removes summary statistics, removes lon, lat, alt,
    sat_qual and calib columns, and renames remaining columns.

    filepath: absolute path to a Dualx .csv file
    seperator: usually a ';'
    '''

    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")

    df = pd.read_csv(filepath, seperator, skiprows=4)
    #df = df.dropna()


    df = df[~((df['#yyyy/mm/dd'] == '#Group') | (df['#yyyy/mm/dd'].shift(1) == '#Group'))]

    df.columns = ['date', 'time', 'lon', 'lat', 'alt', 'sat_qual', 'temp', 'group', 'measure', 'side', 'chl', 'flav', 'anth', 'nbi', 'calib']

    df = df.drop(['lon', 'lat', 'alt', 'sat_qual', 'calib'], axis=1)

    cols = ['chl', 'flav', 'anth', 'nbi', 'temp', 'group']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)


    df_clean = df[~((df['date'] == '#yyyy/mm/dd'))]

    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['time'] = pd.to_datetime(df_clean['time'], format='%H:%M:%S').dt.time


    return(df_clean)
    #print("Dualex file converted. These are the pandas data types of each column:\n", df_clean.dtypes, '\n', df_clean.describe())

def pixelToWorldCoords(pX, pY, geoTransform):
    ''' Input image pixel coordinates and get world coordinates according to geotransform using gdal
    '''

    def applyGeoTransform(inX, inY, geoTransform):
        outX = geoTransform[0] + inX * geoTransform[1] + inY * geoTransform[2]
        outY = geoTransform[3] + inX * geoTransform[4] + inY * geoTransform[5]
        return outX, outY

    mX, mY = applyGeoTransform(pX, pY, geoTransform)
    return mX, mY

def worldToPixelCoords(wX, wY, geoTransform, dtype='int'):
    ''' Input world coordinates and get pixel coordinates according to reverse geotransform using gdal
    '''
    reverse_transform = ~ affine.Affine.from_gdal(*geoTransform)
    px, py = reverse_transform * (wX, wY)
    if dtype == 'int':
        px, py = int(px + 0.5), int(py + 0.5)
    else:
        px, py = px + 0.5, py + 0.5
    return px, py

def getlicorlai(dir, sep):
    '''
    Export LAI values for the Licor 2200C per txt file.
    '''
    # sep = ';'
    # dir = r'/Volumes/2021UGent/proj_smac/2021/lai/'

    import pandas as pd
    import os
    import re
    #import warnings
    #warnings.filterwarnings("ignore")

    lai_lst = []

    for filename in os.listdir(dir):
        if filename.endswith(".TXT") or filename.endswith(".txt"):
            df = pd.read_csv(os.path.join(dir, filename), sep)
            lai_str = df.iloc[8,0]


            lai_lst.append([float(s) for s in re.findall(r"[-+]?\d*\.\d+|\d+", lai_str)])

    flat_list = [item for sublist in lai_lst for item in sublist]
    numlist = list(map(float, flat_list))


    return(numlist)

def minmax(x):
    '''
    Function to apply min/max scaling
    :param x: numpy array
    :return: min max scaled array
    '''
    import nupy as np
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

def xyval(A):
    """
    Function to list all pixel coords including their associated value starting from the  upper left corner (?)
    :param A: raster band as numpy array
    :return: x and y of each pixel and the associated value
    """
    import numpy as np
    x, y = np.indices(A.shape)
    return x.ravel(), y.ravel(), A.ravel()


def getsunscan(dir, sep):
    """
        Export all values for the SunScan per txt file.

        :param dir: raster band as numpy array
        :param sep: seperator as string (e.g., '\t')
        :return df: values recorded using the sun scan
        :return name: the filename including the recording date

    """
    # sep = '\t'
    # dir = r'/Users/reneheim/Dropbox/2022/projects/proj_smac/data/raw/2021_mel_soy_referencedata/ilvo_data/'

    import pandas as pd
    import os

    for filename in os.listdir(dir):
        if filename.endswith(".TXT"):  # or filename.endswith(".txt"):
            df = pd.read_csv(os.path.join(dir, filename), sep=sep, skiprows=13, header=None)
            df.columns = ['time', 'plot', 'sample', 'transmitted', 'spread', 'incident',
                          'beam_frac', 'zenith', 'lai', 'notes']

    name = os.path.splitext(filename)[1]

    return (df, name)


def split_train_test_data(X,Y,test_size=0.3,n_states=50):
    '''
    This function finds the best way to split the training and testing sets in a large number possible ways
    by mininizing the l2 distance between their mean values. This function is only useful when not using cross
    validation and makes sure the splitting strategy does not bias the performance of the model.
    '''
    # import 
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    # Define container for performance metrics for different splitting strategies
    columns=X.columns
    distance=[]
    # define a number of random states, iterate splitting over random states, and compute the distance
    for nb in range(0,n_states):
        X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=test_size, random_state=nb)
        scale=StandardScaler().fit(X)
        X_train=scale.transform(X_train)
        X_test=scale.transform(X_test)
        X_train=pd.DataFrame(X_train, columns=columns)
        X_test=pd.DataFrame(X_test, columns=columns)
        dist=np.sum((X_train.mean()-X_test.mean())**2)/len(X_train.mean())
        distance+=[dist]
    # take the random state minimizing the distance
    randomstate=np.argmin(distance)
    return train_test_split(X,Y, test_size=test_size, random_state=randomstate)
     
  
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
