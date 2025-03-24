#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2023/01/13
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Leaf greenness was estimated by 4 different human raters. To correct for the rater bias, we created 
a reference rating. This was done by taking the average rating value for each of the two days that 
we had to collect the data. Using this reference, we build a calibration model for each rater and 
corrected the rating towards the daily mean for each leaf.
"""

# <editor-fold desc="01_loading libs and paths">

import pandas as pd
import os
import numpy as np
from smac_functions import linreg

dat = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\proc'
out = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\proc'
# </editor-fold>

# <editor-fold desc="02_creating data based on field notes">
# Data Day1

a = (np.arange(1, 19, 1).tolist())
l = [70, 55, 25,65,15,70,75,65,75,60,65,70,60,75,75,45,35,5]
w = [85,70,35,85,20,70,85,65,90,35,80,85,35,90,85,50,50,10]
r = [90,75,30,90,10,70,90,50,95,30,90,90,30,95,90,50,50,5]
d = {'leaf_no': a, 'louis': l, 'wouter': w, 'rene': r}
df = pd.DataFrame(data=d)

df['mean'] = df.iloc[:, 1:3].mean(axis=1)

# Data Day2

a = (np.arange(1, 16, 1).tolist())
ar = [55,90,80,75,30,55,60,80,60,5,65,80,85,75,55]
w = [55,90,75,80,40,60,70,85,70,5,80,80,95,85,65]
r = [60,95,80,80,45,65,70,90,80,5,80,80,90,80,70]
d = {'leaf_no': a, 'arne': ar, 'wouter': w, 'rene': r}
df2 = pd.DataFrame(data=d)

df2['mean'] = df2.iloc[:, 1:3].mean(axis=1)
# </editor-fold>

# <editor-fold desc="03_creating regression paras using daily average as reference">
# Simple linear regression; Day 1; Rene

modd1r = linreg(df['rene'], df['mean'], os.path.join(out, "linregd1rene.png"))

# Simple linear regression; Day 1; Wouter

modd1w = linreg(df['wouter'], df['mean'], os.path.join(out, "linregd1wouter.png"))

# Simple linear regression; Day 1; Louis

modd1l = linreg(df['louis'], df['mean'], os.path.join(out, "linregd1louis.png"))

# Simple linear regression; Day 2; Arne

modd2a = linreg(df2['arne'], df2['mean'], os.path.join(out, "linregd2arne.png"))

# Simple linear regression; Day 2; Wouter¶

modd2w = linreg(df2['wouter'], df2['mean'], os.path.join(out, "linregd2wouter.png"))

# Simple linear regression; Day 2; Rene

modd2r = linreg(df2['rene'], df2['mean'], os.path.join(out, "linregd2rrene.png"))
# </editor-fold>

# <editor-fold desc="04_using regression parameters to correct each rating towards the mean">
#Split df in day 1 and day 2

df_c = pd.read_csv(os.path.join(out, "02_after_leafarea.csv"),
                   index_col=[0])  # this is the full data after adding leaf area

d1 = df_c.loc[df_c.date == '09.09.2020']
d2 = df_c.loc[df_c.date == '10.09.2020']


#correct louis day1

new = d1.loc[d1.measure == 'louis', 'green_per'].map(lambda x: modd1l['slope']*x + modd1l['intercept']).astype(float)
d1.loc[d1.measure == 'louis', 'green_per'] = new


#correct wouter day1

new2 = d1.loc[d1.measure == 'wouter', 'green_per'].map(lambda x: modd1w['slope']*x + modd1w['intercept']).astype(float)
d1.loc[d1.measure == 'wouter', 'green_per'] = new2

#correct rene day1

new3 = d1.loc[d1.measure == 'rene', 'green_per'].map(lambda x: modd1r['slope']*x + modd1r['intercept']).astype(float)
d1.loc[d1.measure == 'rene', 'green_per'] = new3

# DAY 2

#correct arne day2

new4 = d2.loc[d2.measure == 'arne', 'green_per'].map(lambda x: modd2a['slope']*x + modd2a['intercept']).astype(float)
d2.loc[d2.measure == 'arne', 'green_per'] = new4

#correct wouter day1

new5 = d2.loc[d2.measure == 'wouter', 'green_per'].map(lambda x: modd2w['slope']*x + modd2w['intercept']).astype(float)
d2.loc[d2.measure == 'wouter', 'green_per'] = new5

#correct rene day2

new6 = d2.loc[d2.measure == 'rene', 'green_per'].map(lambda x: modd2r['slope']*x + modd2r['intercept']).astype(float)
d2.loc[d2.measure == 'rene', 'green_per'] = new6

df_d = pd.concat([d1, d2])
# </editor-fold>

df_d.to_csv(os.path.join(out, "03_addgreenness.csv"))
