# Parameter recovery can be run through the code in predator_rbm_model/run_rbm_model_parameter_recovery.py
# Plot parameter recovery for predator task


import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from functions.util_functions import cm2inch, calculate_spearman_corr
import matplotlib.ticker as ticker
import numpy as np


# ---------------------------------
# Functions to analyze recovery
# ---------------------------------
def analyze_recovery(true_params, recov_params):
    df_merged = true_params.merge(recov_params, on='subjectID')

    corr_arr = [calculate_spearman_corr(df_merged['beta_1_x'], df_merged['beta_1_y']),
                calculate_spearman_corr(df_merged['beta_4_x'], df_merged['beta_4_y']),
                calculate_spearman_corr(df_merged['beta_7_x'], df_merged['beta_7_y']),
                calculate_spearman_corr(df_merged['beta_5_x'], df_merged['beta_5_y']),
                calculate_spearman_corr(df_merged['beta_6_x'], df_merged['beta_6_y']),
                calculate_spearman_corr(df_merged['beta_8_x'], df_merged['beta_8_y']),
                calculate_spearman_corr(df_merged['omikron_0_x'], df_merged['omikron_0_y']),
                calculate_spearman_corr(df_merged['omikron_1_x'], df_merged['omikron_1_y']),
                ]

    return df_merged, corr_arr


# ---------------------------------
# Load Data
# ---------------------------------
target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

true_params = pd.read_csv(os.path.join(target_dir,
                                       'supplementary_data/predator_task/parameter_recovery'
                                       '/model_paramrec3_predator_exp4_actual_params_seed125.csv'))
recov_params = pd.read_csv(os.path.join(target_dir,
                                        'supplementary_data/predator_task/parameter_recovery'
                                        '/model_paramrec3_predator_exp4_recovered_params_seed125.csv'))

# ----------------------------------
# If recovery data has a column named 'ID', rename it to 'subjectID'
if 'ID' in recov_params.columns:
    recov_params = recov_params.rename(columns={'ID': 'subjectID'})

# ----------------------------------
# Analyze Recovery
# ----------------------------------

df_params, recov_corr = analyze_recovery(true_params, recov_params)

# ---------------------------------
# Plot Parameter Recovery
# ---------------------------------

# calculate recovery correlation
param_names = ['Fixed LR', 'Adaptive LR', 'Valence', '$h$', '$s$', r'$h \cdot s$', r'$o_0$', r'$o_1$', ]
param_cols = ['beta_1', 'beta_4', 'beta_7', 'beta_5', 'beta_6', 'beta_8', 'omikron_0', 'omikron_1']

# Plot parameter recovery

# setup of figure
# Size of figure
fig_height = 9
fig_width = 15
fontsize = 7

# plot colors
colors = ["#80cdc1", '#de77ae', "#018571", "#dfc27d", '#d492c8', '#AA4499', '#808080', "#77AADD", "#3576b8"]
sns.set_palette(sns.color_palette(colors))

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(2, 4, wspace=0.65, hspace=0.9, top=0.87, bottom=0.1, left=0.15, right=0.98)

for i in range(len(param_cols)):
    col_no = i % 4
    row_no = np.floor(i / 4).astype(int)

    ax = plt.Subplot(f, gs_0[row_no, col_no])
    f.add_subplot(ax)

    # data index
    xind = param_cols[i] + '_x'
    yind = param_cols[i] + '_y'

    sns.regplot(x=df_params[xind].astype('float'), y=df_params[yind].astype('float'),  # color='#3576b8'
                color=colors[-1], robust=True, ax=ax,
                scatter_kws=dict(alpha=0.3, s=10, edgecolor="none", color=colors[-2]),
                line_kws=dict(linewidth=2))

    ax.set_xlabel('Ground Truth', fontsize=fontsize)
    ax.set_ylabel('Recovered', fontsize=fontsize)

    title = param_names[i] + '\n' + "$Spearman \ \it{ρ}$ = " + str(round(recov_corr[i], 2))
    ax.set_title(title, fontsize=fontsize)

    # have 3 ticks per axis
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

sns.despine()

# save figure
name = 'figure_s7.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')

plt.show()
