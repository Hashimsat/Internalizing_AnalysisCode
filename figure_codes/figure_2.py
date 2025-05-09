# Plot showing correlations between questionnaire items
# Also shows results from factor analysis
# Environment: predator_task_env

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from functions.util_functions import qns_factor_preprocessing, cm2inch, label_subplots
from scipy.stats import pearsonr


# -----------------
# 1. Load data
# -----------------

qns_itemdata = pd.read_csv('../data/factor_analysis/questionnaires_itemdata.csv', sep=';')
qns_totalscore = pd.read_csv('../data/factor_analysis/questionnaires_totalscores_subscales.csv')
factor_scores = pd.read_csv('../data/factor_analysis/factor_scores.csv')
factor_loadings = pd.read_csv('../data/factor_analysis/factor_loadings.csv')

figure_folder = '../figures/'
# -----------------
# 2. Preprocess qns and factor data
# -----------------

df_qns, df_fs, df_merged = qns_factor_preprocessing(qns_totalscore, factor_scores)

# Remove IG09 questionnaire from raw questionnaire data
qns_pure = qns_itemdata.drop(columns=['REF','Unnamed: 0'])
columns_to_remove = qns_pure.filter(like='IG').columns
qns_pure = qns_pure.drop(columns=columns_to_remove)

# -----------------
# 3. Compute correlations
# -----------------

# Correlation between qns items
correlation_matrix = qns_pure.corr()

# Number of items per questionnaire
num_items_per_questionnaire = np.array([20,20,21,21,27,39,16])  # AI01,AI02,BDI,IC02,IU27,MASQ,PSWQ
# Indices to plot lines
indices = np.cumsum(num_items_per_questionnaire)

# Correlation between factor scores and questionnaire total scores
df_subscale_score = df_merged[['AI01','STAI_anx','STAI_dep','STICSA_cognitive','STICSA_somatic','BD','IU02','PW','MASQ_Anh','MASQ_AnxAr','g','F1.','F2.',]]

correlation_matrix_fs = df_subscale_score.corr()
pval = df_subscale_score.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*correlation_matrix_fs.shape)

# Select only the relevant part of the correlation matrix
correlation_subset = correlation_matrix_fs.loc[['g', 'F1.', 'F2.'], df_subscale_score.columns[0:10]]
pval_subset = pval.loc[['g', 'F1.', 'F2.'], df_subscale_score.columns[0:10]]

# -----------------
# 4. Prepare figure
# -----------------

fig_width = 15
fig_height = 14
fontsize = 7

# Setup Figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
gs_0 = gridspec.GridSpec(7, 6, wspace=2.4, hspace=0.8, top=0.95, bottom=0.15, left=0.09, right=.97)

# -----------------
# 5. Plot correlations between qns items
# -----------------

# Plot correlations between qns items
gs_00 = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs_0[0:4,0:3])
ax_0 = plt.Subplot(f, gs_00[0:4, 0:3])
f.add_subplot(ax_0)

questionnaire_labels = ['STAI-S', 'STAI-T', 'BDI', 'STICSA-T', 'IUS-27', 'MASQ', 'PSWQ']
heatmap = sns.heatmap(correlation_matrix, cmap='coolwarm',cbar=False, annot=False, ax=ax_0)

# Add a colorbar with a smaller size
colorbar = ax_0.figure.colorbar(heatmap.get_children()[0], ax=ax_0, shrink=0.4)
colorbar.set_ticks([0, 0.5, 1])  # Optional: set specific ticks
colorbar.ax.tick_params(labelsize=fontsize)  # Optional: adjust colorbar tick label size
colorbar.set_label('Pearson Correlation', fontsize=fontsize)  # Add label to colorbar

# Adjust colorbar size
colorbar.ax.set_aspect(10, adjustable='box')  # Adjust the aspect ratio (height/width) of the colorbar

# Add the lines to indicate boundaries of each questionnaire
for index in indices:
    ax_0.axhline(index, color='black', linewidth=2)
    ax_0.axvline(index, color='black', linewidth=2)

# Compute the center of each questionnaire to place qns names
center_indices = (indices - num_items_per_questionnaire / 2).astype(int)

# Set the ticks at the center of each questionnaire subset
ax_0.set_xticks(center_indices)
ax_0.set_yticks(center_indices)

# Set the tick labels to the questionnaire labels
ax_0.set_xticklabels(questionnaire_labels, rotation=45, fontsize=fontsize)
ax_0.set_yticklabels(questionnaire_labels, rotation=0, fontsize=fontsize)
# ax_0.set_xlabel('Questionnaire', fontsize=fontsize)


# -----------------
# 6. Plot factor loadings
# -----------------

gs_01 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs_0[0:4,3:6],hspace=1)
ax_1 = plt.Subplot(f, gs_01[0, 0:3])
ax_2 = plt.Subplot(f, gs_01[1, 0:3])
ax_3 = plt.Subplot(f, gs_01[2, 0:3])
f.add_subplot(ax_1)
f.add_subplot(ax_2)
f.add_subplot(ax_3)

# Define colors for each questionnaire
# colors from https://personal.sron.nl/~pault/
colors = ['#77AADD','#99ddff', '#44BB99', '#BBCC33', '#EEDD88', '#EE8866', '#FFAABB']  # qualitative light scheme

# Create a color array corresponding to the questionnaire items
color_array = []
for idx, num_items in enumerate(num_items_per_questionnaire):
    color_array.extend([colors[idx]] * num_items)

factor_loadings['g'].plot(kind='bar', ax=ax_1, color=color_array, legend=False)
tick_positions = np.arange(0, len(factor_loadings.index), 20)
ax_1.set_title('General Factor', fontsize=fontsize, y=1.0, pad=1)
ax_1.set_ylabel('Loading', fontsize=fontsize)
ax_1.set_ylim([-0.3,0.7])
ax_1.tick_params(axis='y', labelsize=fontsize)
ax_1.set_xticks(tick_positions)
ax_1.set_xticklabels(factor_loadings.index[tick_positions],rotation=45, fontsize=fontsize)

factor_loadings['F1.'].plot(kind='bar', ax=ax_2, color=color_array, legend=False)
ax_2.set_title('Factor 1', fontsize=fontsize, y=1.0, pad=-3)
ax_2.set_ylabel('Loading', fontsize=fontsize)
ax_2.set_ylim([-0.3,0.7])
ax_2.tick_params(axis='y', labelsize=fontsize)
ax_2.set_xticks(tick_positions)
ax_2.set_xticklabels(factor_loadings.index[tick_positions],rotation=45, fontsize=fontsize)

factor_loadings['F2.'].plot(kind='bar', ax=ax_3, color=color_array, legend=False)
ax_3.set_title('Factor 2', fontsize=fontsize , y=1.0, pad=-3)
ax_3.set_ylabel('Loading', fontsize=fontsize)
ax_3.set_ylim([-0.3,0.7])
ax_3.set_xlabel('Item Number', fontsize=fontsize)
ax_3.set_xticks(tick_positions)
ax_3.set_xticklabels(factor_loadings.index[tick_positions],rotation=45, fontsize=fontsize)
ax_3.tick_params(axis='y', labelsize=fontsize)

# Create custom legend
handles = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(len(colors))]
ax_3.legend(handles, questionnaire_labels, fontsize=fontsize-1, loc='upper center', bbox_to_anchor=(0.5, -1.05), ncol=4, handlelength=1)


# -----------------
# 7. Plot correlation between factor scores and questionnaire total scores
# -----------------
questionnaire_subscale_labels = ['STAI-S', 'STAI-anx','STAI-dep','STICSA-cog','STICSA-som', 'BDI', 'IUS-27', 'PSWQ', 'MASQ-ad','MASQ-aa']

gs_03 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[5:7,0:6])
ax_4 = plt.Subplot(f, gs_03[0, 0])
f.add_subplot(ax_4)

heatmap_ts = sns.heatmap(correlation_subset, annot=True,fmt = '.2f', annot_kws={'size':fontsize},
                         cmap=sns.diverging_palette(220, 10, as_cmap=False),cbar=False, ax=ax_4,
                        linewidths=0.5, linecolor='black', clip_on=False,)

# Add a colorbar with a smaller size
colorbar = ax_4.figure.colorbar(heatmap_ts.get_children()[0], ax=ax_4, shrink=1)
colorbar.ax.set_aspect(10, adjustable='box')
colorbar.ax.tick_params(labelsize=fontsize)  # Optional: adjust colorbar tick label size
colorbar.set_label('Pearson Correlation', fontsize=fontsize, labelpad=5)  # Add label to colorbar

ax_4.set_xticklabels(questionnaire_subscale_labels, rotation=45, fontsize=fontsize)
ax_4.set_yticklabels(['G','F1','F2'], rotation=0, fontsize=fontsize)

# Add labels
texts = ['a',' ' ,'b',' ',' ','c','']
label_subplots(f, texts, x_offset=0.085, y_offset=0.02,fontsize=fontsize)

sns.despine()

# -----------------
# 8. Save figure
# -----------------

name='figure_2_fa'+'.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=True)

plt.show()
