#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2023/01/13
# version ='1.1'
# ---------------------------------------------------------------------------
"""calculating dry matter content (Cm) and leaf water equivalent (Cw)"""

# <editor-fold desc="01_loading libs and paths">
import pandas as pd
import os

dat = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\raw'
out = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\proc'


df_plot = pd.read_csv(os.path.join(out, "05_after_lai.csv"), index_col=[0])
# </editor-fold>

# <editor-fold desc="02_calculating fresh weight, dry weight, leaf area">
# Calculate the mean of FW, LA and DW (all are the sum of three individuell leaves)

df2 = pd.read_csv(os.path.join(dat, "fw_la_dw_analysis.csv"), sep=",", na_values='na')

df2['FW_g_mean'] = df2['FW_g']/3
df2['LA_cm2_mean'] = df2['LA_cm2']/3

df2 = df2.drop(columns=['FW_g', 'LA_cm2', 'DW1_g', 'DW2_g', 'DWtot_g', 'empty bag', 'mean', 'LA_cm2_mean'])

df2['Sample'] = df2['Sample'].str.replace(r'P', '')
df2.columns = ["plot", 'strata', 'la_mm2', 'dw', 'fw']

df2['la_cm2'] = df2['la_mm2']/100
df2['strata'] = df2['strata'].str.replace(r'm', 'middle')
df2['strata'] = df2['strata'].str.replace(r't', 'top')
df2['strata'] = df2['strata'].str.replace(r'b', 'bottom')

df2['Cm[g/cm2]'] = df2['dw']/df2['la_cm2']
df2['Cw[cm]'] = (df2['fw']-df2['dw'])/(df2['la_cm2']*1)
df2['fw/la_cm2'] = df2['fw']/df2['la_cm2']
df2['sla[cm2/g]'] = df2['la_cm2']/df2['dw']
df2['%h2o'] = 100-(100/df2['fw']*df2['dw'])

#drop bottom strata

df2= df2[df2.strata != 'bottom']

df2_plot = df2.groupby(['plot'],as_index=False).mean() #calculating sum to aggregate all strata for plot values

df3_plot = df2_plot.drop(columns = ['la_mm2', 'dw', 'fw', 'la_cm2', 'fw/la_cm2'])
# </editor-fold>

# <editor-fold desc="03_merging Cw and Cm with main df">
#Add Cm, Cw and fw/la to main df

df3_plot['plot'] = df3_plot['plot'].astype('int64')
dfnew = pd.merge(df_plot, df3_plot, how='inner', on=['plot'])
# </editor-fold>

dfnew.to_csv(os.path.join(out, "06_after_cmcw.csv"))

