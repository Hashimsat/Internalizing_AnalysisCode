# Plot model parameter boxplot and model parameters against general factor scores
# Environment: predator_task_env

import numpy as np
import pandas as pd
from scipy.stats import zscore
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functions.util_functions import cm2inch, label_subplots, medianprops, qns_factor_preprocessing
from functions.plotting_functions import plot_x_vs_y_FactorScores_robust, FDR_correction_regression
from functions.predator_descriptive_functions import zscore_columns

# -----------------
# 1. Load data
# -----------------
target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

ml_data = pd.read_csv(os.path.join(base_dir, 'data/predator_task/df_predator_4exp_modelresults.csv'))

qns_totalscore = pd.read_csv(os.path.join(base_dir, 'data/factor_analysis/questionnaires_totalscores_subscales.csv'))
factor_scores = pd.read_csv(os.path.join(base_dir, 'data/factor_analysis/factor_scores.csv'))

# -----------------
# 2. Preprocess data
# -----------------
df_qns, df_fs, df_merged = qns_factor_preprocessing(qns_totalscore, factor_scores)

# merge with model data
ml_data = ml_data.merge(df_merged, on='subjectID')

# zscore g, F1 and F2
columns_to_zscore = ['g', 'F1.', 'F2.', 'Age', 'beta_1', 'beta_4', 'beta_5', 'beta_6', 'beta_7', 'beta_8']
ml_data = zscore_columns(ml_data, columns_to_zscore)
ml_data['int'] = 1

# -------------------
# 3. Set up the figure
# -------------------
# Size of figure

fig_width = 15
fig_height = 10
fontsize = 7
medianprops = medianprops()

colors = ['#77AADD', '#009988', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.6, top=0.85, bottom=0.1, left=0.12, right=0.99)

# -----------
# Plot the data
# -----------

# Boxplots of all model parameters

gs_01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[0:1, 0:1])
ax1 = plt.Subplot(f, gs_01[0, 0])
f.add_subplot(ax1)

data_boxplot = ml_data[['beta_1', 'beta_4', 'beta_7', 'beta_5', 'beta_6', 'beta_8']]

sns.boxplot(data=data_boxplot, fliersize=0, linewidth=0.5, ax=ax1, width=0.5, color='#77AADD',
            medianprops=medianprops,
            meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markeredgecolor': 'black', 'markersize': 10},
            boxprops=dict(alpha=.7), zorder=7, )

ax1.hlines(0, xmin=-1, xmax=6, color='k', linewidth=1, linestyle='--')
ax1.set_xticklabels([r'$\beta_1$', r'$\beta_2$', r'$\beta_3$', r'$\beta_4$', r'$\beta_5$', r'$\beta_6$'],
                    fontsize=fontsize)
ax1.set_ylabel('Coefficients', fontsize=fontsize)
ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
ax1.tick_params(axis='y', labelsize=fontsize)
ax1.set_ylim([-1, 1.5])

# Plot model parameters vs general factor scores
gs_02 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[0:1, 1:2], wspace=0.65)

# Plot Fixed LR (beta_1) vs g
ax2 = plt.Subplot(f, gs_02[0, 0])
f.add_subplot(ax2)

r_b1, p_b1_g, t_b1 = plot_x_vs_y_FactorScores_robust(ml_data, 'g', 'beta_1', ax2, tstat=True, title=False,
                                                     fontsize=fontsize, xlabel='General Factor',
                                                     ylabel=r'Fixed LR ($\beta_1$)', color_index=-2,
                                                     line_color_index=-1)

# Plot Adaptive LR (beta_2) vs g
ax3 = plt.Subplot(f, gs_02[0, 1])
f.add_subplot(ax3)

r_b4, p_b4_g, t_b4 = plot_x_vs_y_FactorScores_robust(ml_data, 'g', 'beta_4', ax3, tstat=True, title=False,
                                                     fontsize=fontsize, xlabel='General Factor',
                                                     ylabel=r'Adaptive LR ($\beta_2$)', color_index=-2,
                                                     line_color_index=-1)

# Plot Valence (beta_3) vs g
gs_03 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_0[1:2, 0:2], wspace=0.65)
ax4 = plt.Subplot(f, gs_03[0, 0])
f.add_subplot(ax4)

r_b7, p_b7_g, t_b7 = plot_x_vs_y_FactorScores_robust(ml_data, 'g', 'beta_7', ax4, tstat=True, title=False,
                                                     fontsize=fontsize, xlabel='General Factor',
                                                     ylabel=r'Valence ($\beta_3$)', color_index=-2, line_color_index=-1)

# Plot HR (beta_4) vs g
ax5 = plt.Subplot(f, gs_03[0, 1])
f.add_subplot(ax5)
r_b5, p_b5_g, t_b5 = plot_x_vs_y_FactorScores_robust(ml_data, 'g', 'beta_5', ax5, tstat=True, title=False,
                                                     fontsize=fontsize, xlabel='General Factor',
                                                     ylabel=r'Hazard Rate ($\beta_4$)', color_index=-2,
                                                     line_color_index=-1)

# Plot Variability (beta_5) vs g
ax6 = plt.Subplot(f, gs_03[0, 2])
f.add_subplot(ax6)
r_b6, p_b6_g, t_b6 = plot_x_vs_y_FactorScores_robust(ml_data, 'g', 'beta_6', ax6, tstat=True, title=False,
                                                     fontsize=fontsize, xlabel='General Factor',
                                                     ylabel=r'Variability ($\beta_5$)', color_index=-2,
                                                     line_color_index=-1)

# Plot Variability*HR (beta_6) vs g
ax7 = plt.Subplot(f, gs_03[0, 3])
f.add_subplot(ax7)

r_b8, p_b8_g, t_b8 = plot_x_vs_y_FactorScores_robust(ml_data, 'g', 'beta_8', ax7, tstat=True, title=False,
                                                     fontsize=fontsize, xlabel='General Factor',
                                                     ylabel=r'Var $\cdot$ HR ($\beta_6$)', color_index=-2,
                                                     line_color_index=-1)

# -----------
# 4. Apply robust regression with FDR correction to all model parameters
# -----------

df_reg = ml_data.copy()

# Setup variables for regression
exog = ['g_z', 'F1_z', 'F2_z', 'Age_z', 'Gender']
endog = ['beta_1_z', 'beta_4_z', 'beta_5_z', 'beta_6_z', 'beta_7_z', 'beta_8_z']

# Apply FDR correction
p_corrected, p_b1_fdr, p_b4_fdr, rlm_results, rlm_results_t = FDR_correction_regression(df_reg, endog, exog,
                                                                                        rlm_out=True)
print(p_corrected)

# -----------
# 5. Set correct titles, label subplots and save figure
# -----------

# Set titles for all axes with corrected p-vals
p_index = 1
for (ax, beta) in zip([ax2, ax3, ax4, ax5, ax6, ax7], endog):
    title = '$\it{r}$ = ' + str(round(rlm_results[beta]['g_z'], 2)) + ', $\it{p}$ = ' + str(p_corrected[p_index])
    ax.set_title(title, fontsize=fontsize)
    p_index = p_index + 6

# Add labels
texts = ['a', 'b', 'c', 'd', 'e', 'f', 'g']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

sns.despine(f)
plt.tight_layout()

# Save figure as pdf
name = "figure_s9.pdf"
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()

# -----------
# 6. Save statistics in dataframe
# -----------

stats_model_regression = {'Statistic': ['r_b1', 'r_b4', 'r_b5', 'r_b6', 'r_b7', 'r_b8',
                                        'p_b1', 'p_b4', 'p_b5', 'p_b6', 'p_b7', 'p_b8',
                                        't_b1', 't_b4', 't_b5', 't_b6', 't_b7', 't_b8',
                                        'n_total'],
                          'Values': [r_b1, r_b4, r_b5, r_b6, r_b7, r_b8,
                                     p_b1_g, p_b4_g, p_b5_g, p_b6_g, p_b7_g, p_b8_g,
                                     t_b1, t_b4, t_b5, t_b6, t_b7, t_b8,
                                     len(ml_data)]
                          }

df_stats_model_regression = pd.DataFrame(stats_model_regression)
df_stats_model_regression.set_index('Statistic', inplace=False)
df_stats_model_regression.name = 'PredatorTask_AllModelParams_GeneralFactorRegression'

# save all t-statistics in a df for then calculating bayes factors
stats_model_regression_t = {'Statistic': ['t_b1_int', 't_b1_g', 't_b1_F1', 't_b1_F2', 't_b1_Age', 't_b1_Gender',
                                          't_b4_int', 't_b4_g', 't_b4_F1', 't_b4_F2', 't_b4_Age', 't_b4_Gender',
                                          't_b5_int', 't_b5_g', 't_b5_F1', 't_b5_F2', 't_b5_Age', 't_b5_Gender',
                                          't_b6_int', 't_b6_g', 't_b6_F1', 't_b6_F2', 't_b6_Age', 't_b6_Gender',
                                          't_b7_int', 't_b7_g', 't_b7_F1', 't_b7_F2', 't_b7_Age', 't_b7_Gender',
                                          't_b8_int', 't_b8_g', 't_b8_F1', 't_b8_F2', 't_b8_Age', 't_b8_Gender',
                                          'n_total', ],
                            'Values': [round(rlm_results_t['beta_1_z']['const'], 2),
                                       round(rlm_results_t['beta_1_z']['g_z'], 2),
                                       round(rlm_results_t['beta_1_z']['F1_z'], 2),
                                       round(rlm_results_t['beta_1_z']['F2_z'], 2),
                                       round(rlm_results_t['beta_1_z']['Age_z'], 2),
                                       round(rlm_results_t['beta_1_z']['Gender'], 2),
                                       round(rlm_results_t['beta_4_z']['const'], 2),
                                       round(rlm_results_t['beta_4_z']['g_z'], 2),
                                       round(rlm_results_t['beta_4_z']['F1_z'], 2),
                                       round(rlm_results_t['beta_4_z']['F2_z'], 2),
                                       round(rlm_results_t['beta_4_z']['Age_z'], 2),
                                       round(rlm_results_t['beta_4_z']['Gender'], 2),
                                       round(rlm_results_t['beta_5_z']['const'], 2),
                                       round(rlm_results_t['beta_5_z']['g_z'], 2),
                                       round(rlm_results_t['beta_5_z']['F1_z'], 2),
                                       round(rlm_results_t['beta_5_z']['F2_z'], 2),
                                       round(rlm_results_t['beta_5_z']['Age_z'], 2),
                                       round(rlm_results_t['beta_5_z']['Gender'], 2),
                                       round(rlm_results_t['beta_6_z']['const'], 2),
                                       round(rlm_results_t['beta_6_z']['g_z'], 2),
                                       round(rlm_results_t['beta_6_z']['F1_z'], 2),
                                       round(rlm_results_t['beta_6_z']['F2_z'], 2),
                                       round(rlm_results_t['beta_6_z']['Age_z'], 2),
                                       round(rlm_results_t['beta_6_z']['Gender'], 2),
                                       round(rlm_results_t['beta_7_z']['const'], 2),
                                       round(rlm_results_t['beta_7_z']['g_z'], 2),
                                       round(rlm_results_t['beta_7_z']['F1_z'], 2),
                                       round(rlm_results_t['beta_7_z']['F2_z'], 2),
                                       round(rlm_results_t['beta_7_z']['Age_z'], 2),
                                       round(rlm_results_t['beta_7_z']['Gender'], 2),
                                       round(rlm_results_t['beta_8_z']['const'], 2),
                                       round(rlm_results_t['beta_8_z']['g_z'], 2),
                                       round(rlm_results_t['beta_8_z']['F1_z'], 2),
                                       round(rlm_results_t['beta_8_z']['F2_z'], 2),
                                       round(rlm_results_t['beta_8_z']['Age_z'], 2),
                                       round(rlm_results_t['beta_8_z']['Gender'], 2),
                                       len(ml_data)
                                       ]
                            }
df_stats_model_regression_t = pd.DataFrame(stats_model_regression_t)
df_stats_model_regression_t.set_index('Statistic', inplace=False)
df_stats_model_regression_t.name = 'PredatorTask_AllModelParams_FactorRegression_tstats'

# For FDR corrected values
stats_model_regression_fdr = {'Statistic': ['r_b1_int', 'r_b1_g', 'r_b1_F1', 'r_b1_F2', 'r_b1_Age', 'r_b1_Gender',
                                            'r_b4_int', 'r_b4_g', 'r_b4_F1', 'r_b4_F2', 'r_b4_Age', 'r_b4_Gender',
                                            'r_b5_int', 'r_b5_g', 'r_b5_F1', 'r_b5_F2', 'r_b5_Age', 'r_b5_Gender',
                                            'r_b6_int', 'r_b6_g', 'r_b6_F1', 'r_b6_F2', 'r_b6_Age', 'r_b6_Gender',
                                            'r_b7_int', 'r_b7_g', 'r_b7_F1', 'r_b7_F2', 'r_b7_Age', 'r_b7_Gender',
                                            'r_b8_int', 'r_b8_g', 'r_b8_F1', 'r_b8_F2', 'r_b8_Age', 'r_b8_Gender',
                                            'p_b1_int', 'p_b1_g', 'p_b1_F1', 'p_b1_F2', 'p_b1_Age', 'p_b1_Gender',
                                            'p_b4_int', 'p_b4_g', 'p_b4_F1', 'p_b4_F2', 'p_b4_Age', 'p_b4_Gender',
                                            'p_b5_int', 'p_b5_g', 'p_b5_F1', 'p_b5_F2', 'p_b5_Age', 'p_b5_Gender',
                                            'p_b6_int', 'p_b6_g', 'p_b6_F1', 'p_b6_F2', 'p_b6_Age', 'p_b6_Gender',
                                            'p_b7_int', 'p_b7_g', 'p_b7_F1', 'p_b7_F2', 'p_b7_Age', 'p_b7_Gender',
                                            'p_b8_int', 'p_b8_g', 'p_b8_F1', 'p_b8_F2', 'p_b8_Age', 'p_b8_Gender',
                                            'n_total',
                                            ],
                              'Values': [rlm_results['beta_1_z']['const'], rlm_results['beta_1_z']['g_z'],
                                         rlm_results['beta_1_z']['F1_z'], rlm_results['beta_1_z']['F2_z'],
                                         rlm_results['beta_1_z']['Age_z'], rlm_results['beta_1_z']['Gender'],
                                         rlm_results['beta_4_z']['const'], rlm_results['beta_4_z']['g_z'],
                                         rlm_results['beta_4_z']['F1_z'], rlm_results['beta_4_z']['F2_z'],
                                         rlm_results['beta_4_z']['Age_z'], rlm_results['beta_4_z']['Gender'],
                                         rlm_results['beta_5_z']['const'], rlm_results['beta_5_z']['g_z'],
                                         rlm_results['beta_5_z']['F1_z'], rlm_results['beta_5_z']['F2_z'],
                                         rlm_results['beta_5_z']['Age_z'], rlm_results['beta_5_z']['Gender'],
                                         rlm_results['beta_6_z']['const'], rlm_results['beta_6_z']['g_z'],
                                         rlm_results['beta_6_z']['F1_z'], rlm_results['beta_6_z']['F2_z'],
                                         rlm_results['beta_6_z']['Age_z'], rlm_results['beta_6_z']['Gender'],
                                         rlm_results['beta_7_z']['const'], rlm_results['beta_7_z']['g_z'],
                                         rlm_results['beta_7_z']['F1_z'], rlm_results['beta_7_z']['F2_z'],
                                         rlm_results['beta_7_z']['Age_z'], rlm_results['beta_7_z']['Gender'],
                                         rlm_results['beta_8_z']['const'], rlm_results['beta_8_z']['g_z'],
                                         rlm_results['beta_8_z']['F1_z'], rlm_results['beta_8_z']['F2_z'],
                                         rlm_results['beta_8_z']['Age_z'], rlm_results['beta_8_z']['Gender'],
                                         p_corrected[0], p_corrected[1], p_corrected[2], p_corrected[3], p_corrected[4],
                                         p_corrected[5],
                                         p_corrected[6], p_corrected[7], p_corrected[8], p_corrected[9],
                                         p_corrected[10], p_corrected[11],
                                         p_corrected[12], p_corrected[13], p_corrected[14], p_corrected[15],
                                         p_corrected[16], p_corrected[17],
                                         p_corrected[18], p_corrected[19], p_corrected[20], p_corrected[21],
                                         p_corrected[22], p_corrected[23],
                                         p_corrected[24], p_corrected[25], p_corrected[26], p_corrected[27],
                                         p_corrected[28], p_corrected[29],
                                         p_corrected[30], p_corrected[31], p_corrected[32], p_corrected[33],
                                         p_corrected[34], p_corrected[35],
                                         len(ml_data),
                                         ]

                              }

df_stats_model_regression_fdr = pd.DataFrame(stats_model_regression_fdr)
df_stats_model_regression_fdr.set_index('Statistic', inplace=False)
df_stats_model_regression_fdr['Values'] = df_stats_model_regression_fdr['Values'].round(3)
df_stats_model_regression_fdr.name = 'PredatorTask_AllModelParams_GeneralFactorRegression_FDR'

# save mean and std of model params

b1_iqr = np.percentile(ml_data['beta_1'], [25, 75])
b4_iqr = np.percentile(ml_data['beta_4'], [25, 75])
b5_iqr = np.percentile(ml_data['beta_5'], [25, 75])
b6_iqr = np.percentile(ml_data['beta_6'], [25, 75])
b7_iqr = np.percentile(ml_data['beta_7'], [25, 75])
b8_iqr = np.percentile(ml_data['beta_8'], [25, 75])

# stats for difference from zero
w_b1, p_b1 = stats.wilcoxon(ml_data['beta_1'])
w_b4, p_b4 = stats.wilcoxon(ml_data['beta_4'])
w_b5, p_b5 = stats.wilcoxon(ml_data['beta_5'])
w_b6, p_b6 = stats.wilcoxon(ml_data['beta_6'])
w_b7, p_b7 = stats.wilcoxon(ml_data['beta_7'])
w_b8, p_b8 = stats.wilcoxon(ml_data['beta_8'])

# FDR correction
p_arr = np.array([p_b1, p_b4, p_b5, p_b6, p_b7, p_b8])
p_arr_fdr = sm.stats.fdrcorrection(p_arr, alpha=0.05, method='indep', is_sorted=False)

stats_model_overall = {'Statistic': ['b1_mean', 'b1_std', 'b1_median', 'b1_IQILow', 'b1_IQIHigh', 'w_b1', 'p_b1',
                                     'b4_mean', 'b4_std', 'b4_median', 'b4_IQILow', 'b4_IQIHigh', 'w_b4', 'p_b4',
                                     'b5_mean', 'b5_std', 'b5_median', 'b5_IQILow', 'b5_IQIHigh',
                                     'b6_mean', 'b6_std', 'b6_median', 'b6_IQILow', 'b6_IQIHigh',
                                     'b7_mean', 'b7_std', 'b7_median', 'b7_IQILow', 'b7_IQIHigh',
                                     'b8_mean', 'b8_std', 'b8_median', 'b8_IQILow', 'b8_IQIHigh'],

                       'Values': [round(np.mean(ml_data['beta_1']), 2), round(np.std(ml_data['beta_1']), 2),
                                  round(np.median(ml_data['beta_1']), 2), b1_iqr[0], b1_iqr[1], w_b1, p_b1,
                                  round(np.mean(ml_data['beta_4']), 2), round(np.std(ml_data['beta_4']), 2),
                                  round(np.median(ml_data['beta_4']), 2), b4_iqr[0], b4_iqr[1], w_b4, p_b4,
                                  round(np.mean(ml_data['beta_5']), 2), round(np.std(ml_data['beta_5']), 2),
                                  round(np.median(ml_data['beta_5']), 2), b5_iqr[0], b5_iqr[1],
                                  round(np.mean(ml_data['beta_6']), 2), round(np.std(ml_data['beta_6']), 2),
                                  round(np.median(ml_data['beta_6']), 2), b6_iqr[0], b6_iqr[1],
                                  round(np.mean(ml_data['beta_7']), 2), round(np.std(ml_data['beta_7']), 2),
                                  round(np.median(ml_data['beta_7']), 2), b7_iqr[0], b7_iqr[1],
                                  round(np.mean(ml_data['beta_8']), 2), round(np.std(ml_data['beta_8']), 2),
                                  round(np.median(ml_data['beta_8']), 2), b8_iqr[0], b8_iqr[1]
                                  ]
                       }

df_stats_model_overall = pd.DataFrame(stats_model_overall)
df_stats_model_overall.set_index('Statistic', inplace=False)
df_stats_model_overall['Values'] = df_stats_model_overall['Values'].round(2)
df_stats_model_overall.name = 'PredatorTask_AllModelParams_OverallStats'
