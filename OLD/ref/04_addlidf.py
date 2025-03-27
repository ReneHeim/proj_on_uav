#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2023/01/13
# version ='1.1'
# ---------------------------------------------------------------------------
"""
Leaf inclinations determine to a great deal the reflectance of vegetation in the near-infrared 
region of the electromagnetic spectrum. They also affect the self-shading in the canopy, and hereby 
the light distribution with the vegetation. In the ratiative transfer model SAIL, the leaf 
inclination distribution is an important parameter. It determines the scattering coefficients, 
gap fractions and the projections of the leaf towards the sun.


The way we measured our leaf angles, we yielded angle values above 90° when measuring the hanging 
leaf tips segments. To generate leaf angle distributions and their functions, we need to restrict 
this between 0° and 90°. 
"""

# <editor-fold desc="01_loading libs, paths, and data">

import matplotlib
matplotlib.use('TkAgg')

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dtw import *


dat = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\raw'
out = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\proc'

df_d = pd.read_csv(os.path.join(out, "03_addgreenness.csv"), index_col=[0])
# </editor-fold>

# <editor-fold desc="02_restricting leaf angles between 0° and 90°">
"""
The way we measured our leaf angles, we yielded angle values above 90° when measuring the hanging leaf tips segments.
To generate leaf angle distributions and their functions, we need to restrict this between 0° and 90°. 
"""

df_e1 = df_d.loc[:, ['plot',
                     'ang_1',
                     'ang_2',
                     'ang_1_per',
                     'la',
                     'strata']]


df_e1['ang_1']= df_e1['ang_1'].map(lambda x: 180-x if x > 90 else x)
df_e1['ang_2']= df_e1['ang_2'].map(lambda x: 180-x if x > 90 else x)

df_e1['ang_1_per'] = df_e1['ang_1_per'] / 100
# </editor-fold>

# <editor-fold desc="03_associate leaf angle 1 (main) and 2 (tip) with leaf area">
'''
As it is, leaf tip angle and leaf angle are split into two different columns. We need to find the 
angle for each leaf segment and the associated leaf area.
'''

area1 = []
angle = []
plots = []
strata = []

for i in df_e1.index:
    
    angle.append(df_e1['ang_1'][i])
    plots.append(df_e1['plot'][i])
    strata.append(df_e1['strata'][i])
    angle.append(df_e1['ang_2'][i])
    plots.append(df_e1['plot'][i])
    strata.append(df_e1['strata'][i])
    
    area1.append(df_e1['la'][i]*df_e1['ang_1_per'][i])

    if df_e1['ang_2'][i] != 'NaN':
    
        area1.append(df_e1['la'][i]*(1-df_e1['ang_1_per'][i]))
    
    else:
        pass
    
ar2 = pd.DataFrame({'area':area1})
an1 = pd.DataFrame({'angle':angle})
#an1 = pd.DataFrame({'angle':angle})
plta = pd.DataFrame({'plot':plots})
stra = pd.DataFrame({'strata':strata})

df_e2 = pd.concat([an1.reset_index(drop=True),
                  ar2.reset_index(drop=True),
                  plta.reset_index(drop=True), 
                  stra.reset_index(drop=True)],
                  axis=1)

df_e3 = df_e2.dropna()

df_e4 = df_e3.sort_values(by=['angle'])
df_e4["angle"] = df_e4["angle"].replace(df_e4["angle"].min(), 0)

n = df_e4['plot'].unique() # looping guide
n.sort()
# </editor-fold>

# <editor-fold desc="04_calculating relative angle frequency and relative area frequency">
#Calculate relative angle frequency per plot

dfs = []

for i in n:

    dfcum =df_e4.loc[(df_e4['plot'] == i)].sort_values(by=['angle']) #plot-wise sorting of angle (low to high)

    new = pd.DataFrame(dfcum.angle.value_counts()).reset_index() # count angles

    new = new.rename(columns={"index": "angle", "angle": "freq"}).sort_values(by=['angle']) #rename columns

    new['cum_freq'] = new['freq'].cumsum() #calculate cumulative frequency

    new['rel_freq'] = new['cum_freq'].map(lambda x: 1/new['cum_freq'].iloc[-1]*x)

    new['plot'] = [i] * len(new['angle'])

    dfs.append(new)
    
df_freq = pd.concat(dfs)

#Calculate relative area frequency per plot

dfs_area = []

for i in n:

    dfcum = df_e4.loc[(df_e4['plot'] == i)].sort_values(by=['angle']) #plot-wise sorting of angle (low to high)

    #new = pd.DataFrame(dfcum.angle.value_counts()).reset_index() # count angles
    
    m = dfcum.loc[(dfcum['plot'] == i)].groupby(['angle'])[['area']].median()

    #new = new.rename(columns={"index": "angle", "angle": "freq"}).sort_values(by=['angle']) #rename columns

    m['cum_area'] = m['area'].cumsum() #calculate cumulative frequency

    m['rel_area'] = m['cum_area'].map(lambda x: 1/m['cum_area'].iloc[-1]*x)

    m['plot'] = [i] * len(m['area'])

    dfs_area.append(m)
    
df_area = pd.concat(dfs_area)

df_area.reset_index(level=0, inplace=True)
# </editor-fold>

# <editor-fold desc="05_plotting relative frequency and area">
#Plotting

fr = df_freq.reset_index()
fig, ax = plt.subplots()

ax = sns.lineplot(data=fr, x="angle", y="rel_freq", hue="plot")
ax.set_title("Leaf Angle vs Relative Angle Frequency")
ax.set_xlabel("Leaf Angle")
ax.set_ylabel("Rel. Freq.")
plt.savefig(os.path.join(out, "e_test.png"))

#plt.show()

ar = df_area.reset_index()
fig2, ax2 = plt.subplots()

ax2 = sns.lineplot(data=ar, x="angle", y="rel_area", hue="plot")
ax2.set_title("Leaf Angle vs Relative Leaf Area")
ax2.set_xlabel("Leaf Angle")
ax2.set_ylabel("Rel. Area")

plt.savefig(os.path.join(out, "e_test2.png"))

#plt.show()

# Compare relative frequency and relative area

# freq = df_freq[(df_freq["plot"] == 3)]
# area = df_area[(df_area["plot"] == 3)]

# x = freq['rel_freq']
# y = area['rel_area']

# alignment = dtw(x, y, keep_internals=True)
# alignment.plot(type="threeway").get_figure().savefig(os.path.join(outpath,'visualizations/dtw.png'))
# plt.close()
# </editor-fold>

# <editor-fold desc="06_loading matlab generated lidf lut">
'''
We are using a LUT approach to find the LIDF for our LADs. Using SCOPE in MATLAB, we are creating a 
LUT of LIDF a/b combination and the according cumulative frequency values. The we can calculate the 
cumulative frequencies of our data plot by plot. Now, we can iterate over the plot level functions 
and LUT functions to find the best match for each plot and note LIDF a/b for each plot.
'''

range_df = pd.read_csv(os.path.join(dat, "LIDFData_Range_analysis.csv"), sep=',', header=None)
lidf_df = pd.read_csv(os.path.join(dat, "LIDFData_Combo_analysis.csv"), sep=',', header=None, names = ['lidfa', 'lidfb'])
lad_df = pd.read_csv(os.path.join(dat, "LIDFData_LAD_analysis.csv"), sep=',', header=None, names = range_df.iloc[:,0])

df_lidf = pd.concat([lidf_df, lad_df], axis=1)
# </editor-fold>

# <editor-fold desc="07_dynamic time warping to find best lidf match">
# Scan through all LUT for Plot 1¶

#help(DTW)
distances = []
#indx = []

for j in n:

    lut = df_lidf[df_area[(df_area["plot"] == j)]['angle'].values.astype(int)] # get LUT data based on my field filter

    for i in np.arange(0, len(lut), 1):

        alignment = dtw(df_area[(df_area["plot"] == j)]['rel_area'], lut.iloc[i, :], keep_internals=True)

        distances.append(alignment.distance)

        #indx.append(lut.index[i])
        

print('Loop successful.')

arr = np.array_split(distances, 26)

dfs = []

for i in arr:

    dfs.append(pd.DataFrame({'dtw_dist':i}))# results in a list with a df for each plot


#This section produces the best LIDFab combination for each plot.

lidfplt = []

for i in dfs:

    best = df_lidf.iloc[i.nsmallest(1, 'dtw_dist').index] # getting the row from the LUT based on the smallest distance 

    # store best LIDF combination

    lidfplt.append(best.loc[:,['lidfa','lidfb']].values)


for j,i in zip(n, range(len(dfs))):
    
    relplt = df_area[(df_area["plot"] == j)]['rel_area']
    
    lut = df_lidf[df_area[(df_area["plot"] == j)]['angle'].values.astype(int)] # get LUT data based on my field filter

    bestlut = lut.iloc[dfs[i].nsmallest(1, 'dtw_dist').index]

    alignment2 = dtw(relplt, bestlut.values[0], keep_internals=True)

#plots
    print('Plot:',j, 'dfid', i)

    alignment2.plot(type="threeway").get_figure().savefig(os.path.join(out,'dtw' + str(j) + '.png'))

    #plt.show()
# </editor-fold>

# <editor-fold desc="08_build df including plot level lidf">
flattened = [val for sublist in lidfplt for val in sublist]

lidfa = []
lidfb = []

for i in np.arange(0,len(flattened),1):
    lidfa.append(flattened[i][0])
    lidfb.append(flattened[i][1])


# remove id, pic_no, plant_no, leaf_no as meaningless after aggregation
df_plot = df_d.groupby(['plot', 'time'],as_index=False).mean()

df_plot['la_plot_sum[cm2]'] = df_d.groupby(['plot'],as_index=False).sum()['la']

df_plot = df_plot.rename(columns={"la": 'la_plot_mean_corr[cm2]', 'green_per' : 'green_per_corr[%]'})


#Adding LIDF now as it is a plot level parameter

df_plot['lidfa'] = pd.DataFrame(lidfa)
df_plot['lidfb'] = pd.DataFrame(lidfb)

df_plot = df_plot.drop(columns=['ang_1', 'ang_1_per', 'ang_2'])
# </editor-fold>

df_plot.to_csv(os.path.join(out, "04_plotlevel_after_lidf.csv"))