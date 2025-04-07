#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2023/01/13
# version ='1.1'
# ---------------------------------------------------------------------------
"""
Note: This had to be coded differently for the first and all other plots as we changed the sampling 
strategy after plot 1.

All upcoming parameters are only plot based, this, I can aggregate the data frame to plot level.

Note: The following changes can only happen on a plot level as Cm, Cw, LAI, and chlorophyll data 
was not recorded on a leaf level.
"""

# <editor-fold desc="01_loading libs and paths">
import os
import pandas as pd

dat = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\raw'
out = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\proc'

df_plot = pd.read_csv(os.path.join(out, "04_plotlevel_after_lidf.csv"), index_col=[0])
# </editor-fold>

# <editor-fold desc="02_adding lai for different lai method in plot 1">
# Adding the LAI for the first plot where the sampling differed from the remaining plots

soil = float(df_plot.loc[df_plot["plot"] == 1, "row_width"] * df_plot.loc[df_plot["plot"] == 1, "spacing"]/2) # area for 1 plant
leaf = float(df_plot.loc[df_plot["plot"] == 1, "la_plot_mean_corr[cm2]"]) #leaf area in plot 1

p1_lai = leaf/soil*3 #LAI in plot 1 for 3 plants sampled
# </editor-fold>

# <editor-fold desc="03_adding lai for all other plots than the first">
lai = list()

for i in df_plot["plot"][1:df_plot['plot'].iloc[-1]]:

    soilx = float((df_plot.loc[df_plot["plot"] == i, "row_width"]*200))/int(df_plot.loc[df_plot["plot"] == i, "plants_2m"])*3

    leafx =  float(df_plot.loc[df_plot["plot"] == i, "la_plot_sum[cm2]"])

    lai.append(leafx/soilx)
    

lai.insert(0, p1_lai)
df_plot['lai'] = lai
# </editor-fold>

# <editor-fold desc="04_calculation green leaf area">
# Calculating the green leaf area in cm^2
df_plot['green_la[cm2]'] = (df_plot['green_per_corr[%]']/100)*df_plot['la_plot_mean_corr[cm2]']

leafg = float(df_plot.loc[df_plot["plot"] == 1, "green_la[cm2]"]) #leaf area in plot 1
p1_laig = leafg/soil*3 #LAI in plot 1 for 3 plants sampled


df_plot['gla_plot_sum[cm2]'] = (df_plot['la_plot_sum[cm2]'])*df_plot['green_per_corr[%]']/100

laig = list()

for i in df_plot["plot"][1:df_plot['plot'].iloc[-1]]:

    soilx = float((df_plot.loc[df_plot["plot"] == i, "row_width"]*200))/int(df_plot.loc[df_plot["plot"] == i, "plants_2m"])*3

    leafx =  float(df_plot.loc[df_plot["plot"] == i, "gla_plot_sum[cm2]"])

    laig.append(leafx/soilx)

laig.insert(0, p1_laig)
df_plot['glai'] = laig

df_plot = df_plot.drop(columns=['plants_2m', 'row_width', 'spacing'])
# </editor-fold>

df_plot.to_csv(os.path.join(out, "05_after_lai.csv"))