#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Rene HJ Heim
# Created Date: 2023/01/13
# version ='1.1'
# ---------------------------------------------------------------------------
""" Outlier removal + stratification of 2020 Bottelare Corn Data"""

# <editor-fold desc="01_loading libs and paths">
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dat = 'D:\\OneDrive - Institut für Zuckerrübenforschung\\projects\\proj_on_uav\\data\\raw\\'
out = 'D:\\OneDrive - Institut für Zuckerrübenforschung\\projects\proj_on_uav\\data\\proc\\'
# </editor-fold>

# <editor-fold desc="02_loading field data">
df = pd.read_csv(f"{dat}20201022_laicab_fielddata_corn_bottelare.csv",
                 sep=';',
                 na_values='na')

# </editor-fold>

# <editor-fold desc="03_adding stratification level">
'''
add stratification levels (bottom, middle, top) by using the length of a plant number vector and 
dividing it by 3. This can be done as we measured leaf parameters always in the upper, middle and 
lower third of a corn plant.
'''

string = []

for i in pd.unique(df["plot"]):
    
    a, b, c = np.array_split(range(len(df[(df["plot"] == i) & (df["plant_no"] == 1)])), 3)
    d, e, f = np.array_split(range(len(df[(df["plot"] == i) & (df["plant_no"] == 2)])), 3)
    g, h, i = np.array_split(range(len(df[(df["plot"] == i) & (df["plant_no"] == 3)])), 3)

    string.extend(np.concatenate((np.repeat("top", len(a)), 
                  np.repeat("middle", len(b)), 
                  np.repeat("bottom", len(c)),
                  np.repeat("top", len(d)),
                  np.repeat("middle", len(e)),
                  np.repeat("bottom", len(f)),
                  np.repeat("top", len(g)),
                  np.repeat("middle", len(h)),
                  np.repeat("bottom", len(i))))
                  )

if len(string) == len(df["id"]):
    df["strata"] = string
# </editor-fold>

# <editor-fold desc="04_removing bottom strata">

df = df[df.strata != 'bottom']
# </editor-fold>

# <editor-fold desc="05_finding and handling outlier">
# Found outlier in ll (max value extreme)
fig, ax = plt.subplots()

ax.boxplot(df["ll"])
ax.set_title('Boxplot showing outlier')
ax.set_xlabel('Leaf length across all plots')
ax.set_ylabel('[cm]')

#fig.savefig(f"{out}b_outlier.png")
plt.show()

print(df.loc[df['ll'].idxmax()])

# Correct outlier by adding decimal point

df["ll"] = df["ll"].replace(df["ll"].max(), 47.8)

fig2, ax2 = plt.subplots()

ax2.boxplot(df["ll"])
ax2.set_title('Boxplot showing that the outlier was removed')
ax2.set_xlabel('Leaf length across all plots')
ax2.set_ylabel('[cm]')
#fig2.savefig(f"{out}b_nooutlier.png")
plt.show()
# </editor-fold>

df.to_csv(f"{out}01_after_strati.csv")  # export final df
