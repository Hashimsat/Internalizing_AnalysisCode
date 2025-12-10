# Plot a figure for the control study with single predator condition (SPC) which shows descriptive
# as well as model based results

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from functions.util_functions import cm2inch, label_subplots, medianprops, qns_factor_preprocessing
from functions.plotting_functions import plot_x_vs_y_FactorScores_robust, FDR_correction_regression, create_subplots
from functions.predator_descriptive_functions import EstimationError_overall, SingleTrialLR_overall, zscore_columns

# -----------------
# 1. Load Data
# -----------------
target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

ml_data = pd.read_csv(os.path.join(target_dir, 'supplementary_data/predator_task/df_model_exp_SPC.csv'))
df_SPC = pd.read_csv(os.path.join(target_dir, 'supplementary_data/predator_task/df_behav_exp_SPC.csv'))

qns_totalscore = pd.read_csv(os.path.join(target_dir, 'supplementary_data/factor_analysis/qnsdata_exp_SPC.csv'),
                             sep=';')
factor_scores = pd.read_csv(os.path.join(target_dir, 'supplementary_data/factor_analysis/Predator_FS_SPC.csv'))

# -----------------
# 2. Preprocess data
# -----------------
df_qns, df_fs, df_merged = qns_factor_preprocessing(qns_totalscore, factor_scores)

# keep only subjects in df_SPC that are also in df_qns
df_SPC = df_SPC[df_SPC['subjectID'].isin(pd.unique(df_merged['subjectID']))]

# Merge with model data
ml_data = ml_data.merge(df_merged, on='subjectID')

# -----------------
# 3. Calculate Descriptive Measures (EE & LR)
# -----------------
Subjects = pd.unique(df_SPC['subjectID'])
df_EE = EstimationError_overall(df_SPC, Subjects)
df_EE = df_EE.merge(df_merged, on='subjectID')

df_LR = SingleTrialLR_overall(df_SPC, Subjects, HitMissSeparation=False)
df_LR = df_LR.merge(df_merged, on='subjectID')

# -----------------
# 4. Setup Figure
# -----------------
fig_width = 9
fig_height = 9
fontsize = 7
medianprops = medianprops()

colors = ['#77AADD', '#009988', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

gs_0 = gridspec.GridSpec(2, 2, wspace=0.5, hspace=0.9, top=0.90, bottom=0.1, left=0.15, right=0.96)

# Create subplots
axes = create_subplots(f, gs_0, [(0, 0), (0, 1), (1, slice(0, 2))])
ax_0, ax_1, ax_2 = axes

# Plot descriptive and model plots
r_EE, p_EE, t_EE = plot_x_vs_y_FactorScores_robust(df_EE, 'g', 'EE', ax_0, title=True, tstat=True, fontsize=7,
                                                   xlabel='General Factor', ylabel='Estimation Error',
                                                   color_index=-2, line_color_index=-1)

r_LR, p_LR, t_LR = plot_x_vs_y_FactorScores_robust(df_LR, 'g', 'LR', ax_1, title=True, tstat=True, fontsize=7,
                                                   xlabel='General Factor', ylabel='Learning Rate',
                                                   color_index=-2, line_color_index=-1)

r, p, t_FLR = plot_x_vs_y_FactorScores_robust(ml_data, 'g', 'beta_1', ax_2, title=False, tstat=True,
                                              legend_txt='Fixed LR', fontsize=7, xlabel='General Factor',
                                              ylabel='Coefficient', color_index=-2, line_color_index=-1)

r_ad, p_ad, t_ad_LR = plot_x_vs_y_FactorScores_robust(ml_data, 'g', 'beta_4', ax_2, title=False, tstat=True,
                                                      legend_txt='Adaptive LR', fontsize=7, xlabel='General Factor',
                                                      ylabel='Coefficient', color_index=-5, line_color_index=-4)

# ------------------
# 5. Apply FDR Correction
# ------------------
# Z-score columns
columns_to_zscore = ['g', 'F1.', 'F2.', 'Age', 'beta_1', 'beta_4', 'beta_7']
ml_data = zscore_columns(ml_data, columns_to_zscore)

# Apply fdr correction
exog = ['g_z', 'F1_z', 'F2_z', 'Age_z', 'Gender']
endog = ['beta_1_z', 'beta_4_z', 'beta_7_z']

p_corrected, p_b1, p_b4 = FDR_correction_regression(ml_data, endog, exog)
print(p_corrected)

title_params = f"$r_{{fixed}}={r}, p_{{fixed}}={p_b1}$\n$r_{{adaptive}}={r_ad}, p_{{adaptive}}={round(p_b4, 2)}$"
ax_2.set_title(title_params, fontsize=fontsize)
ax_2.legend(loc='upper right', fontsize=fontsize - 1, handlelength=0.5)

# Add labels
texts = ['a', 'b', 'c']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.05)

# add number of subjects
texts = ['', '', f"$N$={len(ml_data)}"]
label_subplots(f, texts, x_offset=-0.03, y_offset=-0.025)
sns.despine(f)

name = 'figure_s13.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()

#  -----------------
# 6. Combine stats into dataframes
# -----------------
EE_IQR = np.percentile(df_EE['EE'], [25, 75])
LR_IQR = np.percentile(df_LR['LR'], [25, 75])

stats_SPC = {'Statistic': ['mean_EE', 'std_EE', 'mean_LR', 'std_LR',
                           'median_EE', 'median_EE_IQIlow', 'median_EE_IQIhigh',
                           'median_LR', 'median_LR_IQIlow', 'median_LR_IQIhigh',
                           'r_EE', 'r_LR', 'r_b1', 'r_b4',
                           'p_EE', 'p_LR', 'p_b1', 'p_b4',
                           't_EE', 't_LR', 't_b1', 't_b4',
                           'n_total'],
             'Values': [np.nanmean(df_EE['EE']), np.nanstd(df_EE['EE']), np.nanmean(df_LR['LR']),
                        np.nanstd(df_LR['LR']),
                        np.nanmedian(df_EE['EE']), EE_IQR[0], EE_IQR[1],
                        np.nanmedian(df_LR['LR']), LR_IQR[0], LR_IQR[1],
                        r_EE, r_LR, r, r_ad,
                        p_EE, p_LR, p_b1, p_b4,
                        t_EE, t_LR, t_FLR, t_ad_LR,
                        int(len(ml_data))]}

df_stats_model_overall = pd.DataFrame(stats_SPC)
df_stats_model_overall.set_index('Statistic', inplace=False)
df_stats_model_overall['Values'] = df_stats_model_overall['Values'].round(2)
df_stats_model_overall.name = 'Supplementary_SPC_results'
