#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2022/01/13
# version ='1.1'
# ---------------------------------------------------------------------------
"""adding dualex data and correcting dualex chlorophyll values"""

# <editor-fold desc="01_loading libs and paths">
import matplotlib
matplotlib.use('TkAgg')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
from smac_functions import getdualex

dat = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\raw'
out = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\proc'

df_new = pd.read_csv(os.path.join(out, "06_after_cmcw.csv"), index_col=[0])
# </editor-fold>

# <editor-fold desc="02_loading and cleaning dualex data for field day 1 and 2">
### Day 1

chl1 = getdualex(os.path.join(dat, "DX20200909_analysis.csv"), ';')

chl1 = chl1.dropna()

chl1_p = chl1.groupby(['group'], as_index=False).mean()

chl1_p['plot'] = np.arange(1, len(np.unique(chl1['group']))+1, 1)

### Day 2

chl2 = getdualex(os.path.join(dat, "DX20200910_analysis.csv"), ';')

chl2_p = chl2.groupby(['group'], as_index=False).mean()

#chl2_p['group'] = np.arange(np.unique(chl1_p['group'])[-1]+1, 11+len(np.unique(chl2['group']))+1,1)

chl2_p['plot'] = pd.DataFrame([13, 14, 30, 29, 27, 26, 25, 24, 23, 22, 20, 15, 16, 17])
dfchl = pd.concat([chl1_p, chl2_p])

dfchl.reset_index(inplace=True)

df_sorted = dfchl.sort_values(by=['plot'])
df_sorted2 = df_sorted.drop(columns=['measure', 'group', 'index'])
# </editor-fold>

# <editor-fold desc="03_merging and correcting dualex data">
'''
The merge below will keep the order of the plots. This is important as some plot had been skipped during data 
collection!
'''

result = pd.merge(df_new, df_sorted2, how="inner", on=["plot"])

#with open(os.path.join(out, "dlx_reg.txt")) as json_file:
#    model = json.load(json_file)

#result['chl_corr'] = result['chl'].map(lambda y: (y-model['intercept'])/model['slope']) # chl correction
# </editor-fold>

result.to_csv(os.path.join(out, "07_after_dualex_finaldf.csv"), index=False)
