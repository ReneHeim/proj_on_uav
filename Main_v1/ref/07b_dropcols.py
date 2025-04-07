import pandas as pd
import os

dat = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\raw'
out = r'D:\OneDrive - Institut für Zuckerrübenforschung\projects\proj_on_uav\data\proc'

df = pd.read_csv(os.path.join(out, "07_after_dualex_finaldf.csv"))

df_cl = df.drop(columns=['time', 'id', 'pic_no', 'plant_no', 'cano_height', 'dua_grp', 'leaf_no', 'green_per_corr[%]',
                         'la_plot_mean_corr[cm2]', 'la_plot_sum[cm2]', 'lidfa', 'lidfb', 'green_la[cm2]',
                         'gla_plot_sum[cm2]', 'Cm[g/cm2]', 'Cw[cm]', 'sla[cm2/g]', '%h2o', 'temp', 'flav', 'anth'])

df_cl.to_csv(os.path.join(out, "ref_data.csv"), index=False)