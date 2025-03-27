# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 23:42:45 2023

@author: Sensoriki
"""

# Load modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import seaborn as sns


def save_polar(date, dataset):
    data = pd.read_feather(f"D:\\on_uav_data\\proc\\merge\\2m_radius\\{date}_Bot\\{date}_bot_corn_comb_1m\\{dataset}\\on_uav_for_classification.feather")
    out_dir = r"D:\on_uav_data\proc\Sample BDRF3"
    
    plots = data['plot'].unique()
    
    # Open file containing vza and vaa data
    for plot_Id in tqdm(plots):
        print(f'current plot : {plot_Id}')
        plot = data[data['plot']==plot_Id]
        plot = plot[~np.isnan(plot['vza'])]
        plot['vaa'] = np.where(plot['vaa']< 0, plot['vaa'] + 360, plot['vaa'])
        
        # Define the convolution for 
        kern = np.ones((5,5))
        kern = kern/np.sum(kern)
        
        # Define objects that will contain the figures
        fig = plt.figure()
        ax = Axes3D(fig)
        
        
        # Bet all spectral bands in the image
        bands = [i for i in plot.columns if i.startswith('band')]
        
        # Loop through bands
        for band in bands:
            plt.subplot(projection="polar")
            plt.title(band)
            sns.set_style("whitegrid")
            sns.scatterplot(data = plot,
                            x = plot['vaa']/180*np.pi,
                            y = plot['vza'],
                            hue = plot[band]/65535,
                            s = 8,
                            legend = 'brief',
                            palette = "RdYlBu_r")
            plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
            plt.tight_layout()
            plt.ylabel('')
            plt.savefig(os.path.join(out_dir, date, dataset, str(plot_Id) + '_' + band + '.png'), dpi = 350)
            plt.show()

# Specify directories
dates = ['20200906', '20200907']
datasets = ['Nadir_flight', 'All_flight']

Parallel(n_jobs=4)(delayed(save_polar)(date, dataset) for date in dates for dataset in datasets)
        