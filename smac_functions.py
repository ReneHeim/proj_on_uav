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

'''
The functions "calculate_indices", "calculate_indices_mosaic", "calculate_bands", and "calculate_bands_mosaic",
are defined to clean, and subset a dataframe according to vza. The output of these functions is directly used for
classification or regression according to the aim of the use. In the case of the fist two functions, not only raw
spectral bands are calculated, but also some vegetation indices which have proven to be relevant for LAI and 
Chlorophyll content. In the case of the first and the last functions, which don't have the "_mosaic" ending, they
are intended to be applied on a dataframe derived from orthophotos, which has vza information. A new column is created,
which uses this information to subset the dateframe in a regular or irreguular binnig according to the user-defined value.
Typically, these functions take 4 inputs: The path to the dataframe, the method (classification or regression), the 
binning (regular or irregular) and the date (20200906 or 20200907).
'''

def calculate_indices(database_path,method='classification', date='20200907'):
    # import 
    import pandas as pd
    import numpy as np
    
    # Load dataframe
    if database_path.endswith(".csv"):
        data=pd.read_csv(database_path)
    else:
        data=pd.read_feather(database_path)
        
    # Rename columns
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
    
    # assign bands to variables for rapid processing and easy readabily of formulae
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
    
    # Compute some VI
    ci = (nir842/re717) - 1 
    evi = 2.5 * (nir842 - red668) / (nir842 + 6*red668 - 7.5*blue475 + 1)
    gndvi = (nir842 - green560) / (nir842 + green560)
    mcari = ((re705 - red668) - 0.2 * (re705 - green560)) * (re705/ red668)  # thenkabail
    mdnblue = (blue444 - re717) / (blue444 + nir842)  # https://doi.org/10.1016/j.rse.2017.06.008
    mtci = (re740 - re705) / (re705 - red668)  # thenkabail
    ndre740 = (re740 - re717) / (re740 + re717)  # https://www.indexdatabase.de
    ndre842 = (nir842 - re717) / (nir842 + re717)  # https://www.indexdatabase.de
    ndvi = (nir842 - red668) / (nir842 +red668)  # thenkabail
    pri = (green560 - green531) / (green560 + green531)  # thenkabail
    rep = re705 + 40 * (((red668 + re740 / 2) - re705) / (re740 - re705))  # https://www.indexdatabase.de
    re750_700 = re740 / re705  # https://doi.org/10.1016/S0176-1617(96)80285-9
    tcari = 3 * ((re705 - red668) - 0.2 * (re705 - green560) * (re705 / red668))    # https://www.indexdatabase.de
    osavi = (1 + 0.16) * ((nir842 - red668) / (nir842 + red668 + 0.16))
    tcari_osavi = tcari/osavi
    sccci = ndre842 / ndvi  # Barnes 2000
    
    re_mean = (re705 + re717 + re740) / 3
    green_mean = (green531 + green560)/2
    red_mean=(red650 + red668)/2
    re_ndvi = (nir842 - re_mean)/(nir842 + re_mean)
    msr = (nir842 / red668 -1) / np.sqrt(nir842 / red668 +1)
    re_msr = (nir842 / re_mean - 1)/ np.sqrt(nir842 / re_mean +1)
    ci_green = (nir842 / green_mean) - 1
    ci_re = (nir842 / re_mean) - 1 # 10.1109/JSTARS.2018.2813281
    mtvi = (1.2*(nir842 - green560) - 2.5*(red668 - green560)) # https://www.l3harrisgeospatial.com/docs/narrowbandgreenness.html
    mtvi2 = 1.5*mtvi/np.sqrt((2*nir842 + 1)**2 - (6*nir842 - 5*np.sqrt(red668) - 0.5))
    
    msavi2 = 0.5 * (2*nir842 + 1 - np.sqrt((2*nir842 + 1)**2 - 8*(nir842 - red668))) #https://www.isprs.org/proceedings/xxxv/congress/comm7/papers/21.pdf
    ndvi_re= (re_mean - red_mean) / (re_mean + red_mean)
    cccci= ndre740/ndvi
    
    # Form the final table for predictor variables and assign names to columns
    X=pd.concat([blue444, blue475, green531, green560, red650, red668, re705, re717, re740, nir842,
                 ci, evi, gndvi, mcari, mdnblue, mtci, ndre740, ndre842, ndvi, pri, rep, re750_700,
                 tcari, osavi, tcari_osavi, sccci, re_ndvi, msr, re_msr, ci_green, ci_re, mtvi, mtvi2,
                 msavi2,ndvi_re,cccci], 
                 axis=1)
    X.columns=['blue444', 'blue475', 'green531', 'green560', 'red650', 'red668', 're705', 're717', 're740', 'nir842',
                 'ci', 'evi', 'gndvi', 'mcari', 'mdnblue', 'mtci', 'ndre740', 'ndre842', 'ndvi', 'pri', 'rep', 're750_700',
                 'tcari', 'osavi', 'tcari_osavi', 'sccci', 're_ndvi', 'msr', 're_msr', 'ci_green', 'ci_re', 'mtvi', 'mtvi2',
                 'msavi2','ndvi_re','cccci']
    
    # Remove infite values and NA
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
    
    # Subset for NBI
    data['nbi_cat_a'] = pd.qcut(data['nbi'], 3, labels=["low", "medium", "high"])
    
    # Define the target variable
    if method == 'classification':
        Y=data[['lai_cat_a','chl_cat_a', 'nbi_cat_a']]
    else :
        Y=data[['lai','chl','nbi']]
    
    # define the binning limits for intervals
    labels_angle={'20200906' : ['0-13','13-18','18-23','23-30','30-60'],
                  '20200907' : ['0-11','11-16','16-19','19-23','23-60']}
    cut_angle={'20200906' : [0, 13, 18, 23, 30, 60],
               '20200907' : [0, 11, 16, 19, 23, 60]}
    vza_class=pd.cut(data.vza, cut_angle[date], labels=labels_angle[date])
    
    # Return values
    return X, Y, vza_class

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

def get_components_pls(X, Y, n_repeats=5):
    '''
    This function finds the best value for the number of latent variables to consider when modelling with PLSDA
    by keeping the number of latent variables that maximizes the overall accuracy within a range of possible values.
    '''
    # Import
    import numpy as np
    import pandas as pd
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    # Define container for performance metrics for different numbers of latent variables
    local_acc_1 = []
    # Define numbers of values to consider for the number of latent variables, iterate over them and compute the accuracy
    for comp in range(2,15):
        local_acc_2 = []
        # Do this a number of times to avoid stochastic errors (here 5 times)
        for  nb in range(0, 5*n_repeats, 5):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=nb)
            mod = PLSRegression(n_components = comp)
            mod.fit(X_train,pd.get_dummies(Y_train))
            Y_pred_1 = np.argmax(mod.predict(X_test), axis = 1)
            Y_pred = Y_pred_1.copy().astype(str)
            for ind, val in enumerate(pd.get_dummies(Y).columns):
                Y_pred[Y_pred_1==ind]=val
            acc = accuracy_score(Y_test, Y_pred)
            local_acc_2 += [acc]
        local_acc_1 += [local_acc_2]
    # Take the value that maximises the accuracy
    comp = range(2,15)[np.argmax( np.mean(local_acc_1, axis=1))]
    # Return the best accuracy and the best number of latent variables
    return local_acc_1, comp
     

     

def calculate_indices_mosaic(database_path,method='classification', date='20200907'):
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

    ci = (nir842/re717) - 1 
    evi = 2.5 * (nir842 - red668) / (nir842 + 6*red668 - 7.5*blue475 + 1)
    gndvi = (nir842 - green560) / (nir842 + green560)
    mcari = ((re705 - red668) - 0.2 * (re705 - green560)) * (re705/ red668)  # thenkabail
    mdnblue = (blue444 - re717) / (blue444 + nir842)  # https://doi.org/10.1016/j.rse.2017.06.008
    mtci = (re740 - re705) / (re705 - red668)  # thenkabail
    ndre740 = (re740 - re717) / (re740 + re717)  # https://www.indexdatabase.de
    ndre842 = (nir842 - re717) / (nir842 + re717)  # https://www.indexdatabase.de
    ndvi = (nir842 - red668) / (nir842 +red668)  # thenkabail
    pri = (green560 - green531) / (green560 + green531)  # thenkabail
    rep = re705 + 40 * (((red668 + re740 / 2) - re705) / (re740 - re705))  # https://www.indexdatabase.de
    re750_700 = re740 / re705  # https://doi.org/10.1016/S0176-1617(96)80285-9
    tcari = 3 * ((re705 - red668) - 0.2 * (re705 - green560) * (re705 / red668))    # https://www.indexdatabase.de
    osavi = (1 + 0.16) * ((nir842 - red668) / (nir842 + red668 + 0.16))
    tcari_osavi = tcari/osavi
    sccci = ndre842 / ndvi  # Barnes 2000
    
    re_mean = (re705 + re717 + re740) / 3
    green_mean = (green531 + green560)/2
    red_mean=(red650 + red668)/2
    re_ndvi = (nir842 - re_mean)/(nir842 + re_mean)
    msr = (nir842 / red668 -1) / np.sqrt(nir842 / red668 +1)
    re_msr = (nir842 / re_mean - 1)/ np.sqrt(nir842 / re_mean +1)
    ci_green = (nir842 / green_mean) - 1
    ci_re = (nir842 / re_mean) - 1 # 10.1109/JSTARS.2018.2813281
    mtvi = (1.2*(nir842 - green560) - 2.5*(red668 - green560)) # https://www.l3harrisgeospatial.com/docs/narrowbandgreenness.html
    mtvi2 = 1.5*mtvi/np.sqrt((2*nir842 + 1)**2 - (6*nir842 - 5*np.sqrt(red668) - 0.5))
    
    msavi2 = 0.5 * (2*nir842 + 1 - np.sqrt((2*nir842 + 1)**2 - 8*(nir842 - red668))) #https://www.isprs.org/proceedings/xxxv/congress/comm7/papers/21.pdf
    ndvi_re= (re_mean - red_mean) / (re_mean + red_mean)
    cccci= ndre740/ndvi
    
    X=pd.concat([blue444, blue475, green531, green560, red650, red668, re705, re717, re740, nir842,
                 ci, evi, gndvi, mcari, mdnblue, mtci, ndre740, ndre842, ndvi, pri, rep, re750_700,
                 tcari, osavi, tcari_osavi, sccci, re_ndvi, msr, re_msr, ci_green, ci_re, mtvi, mtvi2,
                 msavi2,ndvi_re,cccci], 
                 axis=1)
    X.columns=['blue444', 'blue475', 'green531', 'green560', 'red650', 'red668', 're705', 're717', 're740', 'nir842',
                 'ci', 'evi', 'gndvi', 'mcari', 'mdnblue', 'mtci', 'ndre740', 'ndre842', 'ndvi', 'pri', 'rep', 're750_700',
                 'tcari', 'osavi', 'tcari_osavi', 'sccci', 're_ndvi', 'msr', 're_msr', 'ci_green', 'ci_re', 'mtvi', 'mtvi2',
                 'msavi2','ndvi_re','cccci']
    
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
        Y=data[['lai_cat_a','chl_cat_a', 'nbi_cat_a']]
    else :
        Y=data[['lai','chl','nbi']]  
    
    return X, Y
  
def calculate_bands_mosaic(database_path,method='classification', date='20200907'):
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
        Y=data[['lai_cat_a','chl_cat_a', 'nbi_cat_a']]
    else :
        Y=data[['lai','chl','nbi']]  
    
    return X, Y
  
def calculate_bands(database_path,method='classification', date='20200907', binning='regular'):
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
        Y=data[['lai_cat_a','chl_cat_a', 'nbi_cat_a']]
    else :
        Y=data[['lai','chl','nbi']]
    
    if binning == 'regular':
        labels_angle={'20200906' : ['0-10','10-20','20-30','30-40','40-50', '50-60'],
                      '20200907' : ['0-10','10-20','20-30','30-40','40-50', '50-60']}
        
        cut_angle={'20200906' : [0, 10, 20, 30, 40, 50, 60],
                   '20200907' : [0, 10, 20, 30, 40, 50, 60]}
    else :
        labels_angle={'20200906' : ['0-13','13-18','18-23','23-30','30-60'],
                      '20200907' : ['0-11','11-16','16-19','19-23','23-60']}
        
        cut_angle={'20200906' : [0, 13, 18, 23, 30, 60],
                   '20200907' : [0, 11, 16, 19, 23, 60]}
        
    vza_class=pd.cut(data.vza, cut_angle[date], labels=labels_angle[date])
    
    return X, Y, vza_class
