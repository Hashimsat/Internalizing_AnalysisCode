# Plot split-half reliability of model parameters

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from functions.util_functions import cm2inch

# ---------------------------------
# Load data
# ---------------------------------
target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

model_even = pd.read_csv(
    os.path.join(target_dir, 'supplementary_data/predator_task/df_predator_4exp_model_PEAlphaValence_split=even.csv'))
model_odd = pd.read_csv(
    os.path.join(target_dir, 'supplementary_data/predator_task/df_predator_4exp_model_PEAlphaValence_split=odd.csv'))

# ----------------------------------
# Analyze split-half reliability
# ----------------------------------
# Extract Fixed LR, Adaptive LR etc. from dataframes

column_names = ['beta_1', 'beta_4', 'beta_7', 'beta_5', 'beta_6', 'beta_8']
columns_of_interest_df1 = model_even[column_names]
columns_of_interest_df2 = model_odd[column_names]

# Find correlations
correlation_matrix = columns_of_interest_df1.corrwith(columns_of_interest_df2, method='spearman')

# --------------
# Setup figure
# --------------

# Size of figure
fig_width = 10
fig_height = 8
fontsize = 7

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create grid
gs_0 = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.6, top=0.85, bottom=0.2, left=0.2, right=0.90)
colors = ['#77AADD', '#009988', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))

# Plot - Bar plots showing correlations
ax1 = plt.Subplot(f, gs_0[0, 0])
f.add_subplot(ax1)

ax1.bar(correlation_matrix.index, correlation_matrix.values, color=colors[0], edgecolor='black')

ax1.set_ylabel('Split-Half Correlations ($\\it{ρ}$)', fontsize=fontsize)
ax1.set_xlabel('Parameters', fontsize=fontsize)

ax1.set_xticks(ax1.get_xticks())
ax1.set_xticklabels(['Fixed LR', 'Adaptive LR', 'Valence', 'h', 's', r'$h\cdot s$'], fontsize=fontsize, rotation=45)
ax1.xaxis.set_tick_params(labelsize=fontsize)
ax1.yaxis.set_tick_params(labelsize=fontsize)

# save figure
sns.despine(f)

name = 'figure_s8.pdf'

savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()

# save data in dataframe
stats = {
    'Statistic': ['r_FixedLR', 'r_AdaptiveLR', 'r_Success', 'r_h', 'r_s', 'r_h_s'],

    'Values': [correlation_matrix['beta_1'], correlation_matrix['beta_4'], correlation_matrix['beta_7'],
               correlation_matrix['beta_5'], correlation_matrix['beta_6'], correlation_matrix['beta_8']]
}

df_stats = pd.DataFrame(stats)
df_stats.set_index('Statistic', inplace=False)
df_stats['Values'] = df_stats['Values'].round(2)
