#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2023/01/13
# version ='1.1'
# ---------------------------------------------------------------------------
""" 2020 true leaf area data for referencing leaf width/length data collected for corn 2020 Bottelare"""

# <editor-fold desc="01_loading libs and paths">
import matplotlib
matplotlib.use('TkAgg')

import os
import matplotlib.pyplot as plt
from scipy import stats
from smac_functions import linreg
import pandas as pd

dat = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\raw'
out = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\proc'


# Import calibration data to establish the allometric relationship between LA and the LW/LL product

df_reg = pd.read_csv(os.path.join(dat, '20201025_allo_lai.csv'), sep=",")
# </editor-fold>

# <editor-fold desc="02_testing whether data is suitable for regression">
# Show summary statistics of this data

fig2 = plt.figure(figsize=(20, 10))
spec2 = plt.GridSpec(ncols=3, nrows=2, figure=fig2, wspace=0.3, hspace=0.3)
ax1 = fig2.add_subplot(spec2[0, 0])
ax2 = fig2.add_subplot(spec2[0, 1])
ax3 = fig2.add_subplot(spec2[0, 2])

ax4 = fig2.add_subplot(spec2[1, 0])
ax5 = fig2.add_subplot(spec2[1, 1])
ax6 = fig2.add_subplot(spec2[1, 2])

ax1.boxplot(df_reg.iloc[:,1])
ax1.set_title('Boxplots and Probability plots for regression evaluation')
ax1.set_xticklabels([])
ax1.set_xlabel('Leaf Length')
ax1.set_ylabel('cm')

ax2.boxplot(df_reg.iloc[:,2])
ax2.set_xticklabels([])
ax2.set_xlabel('Leaf Width')
ax2.set_ylabel('cm')

ax3.boxplot(df_reg.iloc[:,3])
ax3.set_xticklabels([])
ax3.set_xlabel('Leaf Area')
ax3.set_ylabel('cm^2')

stats.probplot(df_reg.iloc[:,1], dist="norm", plot=ax4)
stats.probplot(df_reg.iloc[:,2], dist="norm", plot=ax5)
stats.probplot(df_reg.iloc[:,3], dist="norm", plot=ax6)

#plt.savefig(os.path.join(out, "probplots_greenness.png"))
plt.show()

'''
Both, LL and LW look gaussian. The associated probability plots (QQ-Plots) 
emphasize gaussian behaviour. Now, run regression.

'''
# </editor-fold>

# <editor-fold desc="03_calculate regression parameters to transform LL and LW area">
x = df_reg["LL"]*df_reg["LW"]
y = df_reg["LA"]

model = linreg(x,y, os.path.join(out, "linreg_greeness.png"))
# </editor-fold>

# <editor-fold desc="04_adding leaf area to original df01">
# Edit stratified df

df_b = pd.read_csv(os.path.join(out, "01_after_strati.csv"))

df_b["llxlw"] = df_b["ll"] * df_b["lw"]

df_b['la'] = df_b['llxlw'].map(lambda x: model['slope']*x + model['intercept']) # leaf area correction

df_la = df_b.drop(columns=['Unnamed: 0', 'll', 'lw', 'llxlw'])

df_la.to_csv(os.path.join(out, "02_after_leafarea.csv"))
# </editor-fold>