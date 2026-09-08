# Fig S25: Total score, performance, and percentage of switches in the binary reversal learning task with
# reward magnitudes

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from functions.util_functions import cm2inch, medianprops, label_subplots, qns_factor_preprocessing
from functions.plotting_functions import plot_x_vs_y_FactorScores_robust, create_subplots
from functions.prl_descriptive_functions import calculate_score_performance_switch_rates, separate_low_high_groups
from functions.prl_plotting_functions import plot_descriptive_boxplots, set_subplot_title

# ----------------
# 1. Setup Data Paths
# ----------------

target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

df_prl = pd.read_csv(os.path.join(base_dir, 'data/reversal_task/df_prl_rewardmag_AllData.csv'))
qns_totalscore = pd.read_csv(os.path.join(base_dir, 'data/factor_analysis/questionnaires_totalscores_subscales.csv'))
factor_scores = pd.read_csv(os.path.join(base_dir, 'data/factor_analysis/factor_scores.csv'))

# --------------
# 2. Preprocess data
# --------------
df_qns, df_fs, df_merged = qns_factor_preprocessing(qns_totalscore, factor_scores, drop_non_binary=True)

# remove participant who clicked on the fractal showing on the left side on most trials (not understanding the task)
df_prl = df_prl[df_prl['subjectID'] != 'SinglePredator_DifferingVolatility_GagneWithRewardMag_Predator5c5225d8a163ac0001721f01']

# Remove participants who failed to pass attention checks
df_prl = df_prl.merge(df_merged, on='subjectID')
Subjects = pd.unique(df_prl['subjectID'])

# -----------------
# 3. Calculate Switch Rates and Performance
# -----------------
df_descriptive = calculate_score_performance_switch_rates(df_prl, df_merged, Subjects, block_name='BlockVersion',
                                                          merge=True)

# Separate into low and high G groups
df_descriptive = separate_low_high_groups(df_descriptive, col_name='g')


# -----------------
# 4. Setup Figure
# -----------------
fig_width = 15
fig_height = 10
fontsize = 7
medianprops = medianprops()

# colors = ['#77AADD', '#009988', "#dfc27d", "#018571"]
colors = ["#80cdc1", '#de77ae', "#018571", "#dfc27d", '#d492c8', '#AA4499', '#808080', "#77AADD", "#3576b8"]

sns.set_palette(sns.color_palette(colors))

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

f = plt.figure(figsize=cm2inch(fig_width, fig_height))
gs_0 = gridspec.GridSpec(2, 4, wspace=0.5, hspace=0.7, top=0.90, bottom=0.1, left=0.1, right=0.98)

# create subplots
positions = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
axes = create_subplots(f, gs_0, positions)

# -----------------
# 5. Plot Boxplots of Descriptive Measures
# -----------------
# Boxplots Between Stable and Volatile Blocks for Low and High G Groups

# Plot Performance in Stable and Volatile Blocks
stats_ts = plot_descriptive_boxplots(df_descriptive, axes[0], colors,
                                              prefix='TotalScore', ylabel='Total Score', xlabel=None,
                                              Legend=True, title=True,
                                              min_val=0, max_val=3500, fontsize=fontsize, stat='ttest_ind')

stats_performance = plot_descriptive_boxplots(df_descriptive, axes[2], colors,
                                              prefix='Performance', ylabel='P(Correct)', xlabel='Task Phase',
                                              Legend=False, title=True,
                                              min_val=0, max_val=1, fontsize=fontsize, stat='ttest_ind')

# Plot Switch Rates After Hits and Misses in Stable and Volatile Blocks
stats_switch_hit = plot_descriptive_boxplots(df_descriptive, axes[4], colors,
                                             prefix='SwitchPercentageHit', ylabel='% Switches After Wins',
                                             xlabel='Task Phase',
                                             Legend=False, title=True, fontsize=fontsize, stat='ttest_ind')

stats_switch_miss = plot_descriptive_boxplots(df_descriptive, axes[6], colors,
                                              prefix='SwitchPercentageMiss', ylabel='% Switches After Losses',
                                              xlabel='Task Phase',
                                              Legend=False, title=True, fontsize=fontsize, stat='ttest_ind')

# Caluclate statistics for switches after hits and misses
t_stable, p_stable = stats.ttest_rel(df_descriptive['SwitchPercentageHit_B0'],
                                     df_descriptive['SwitchPercentageMiss_B0'])
t_volatile, p_volatile = stats.ttest_rel(df_descriptive['SwitchPercentageHit_B1'],
                                         df_descriptive['SwitchPercentageMiss_B1'])
dof_stat = len(df_descriptive['SwitchPercentageHit_B0']) - 1

stats_U = {
    'Statistic': ['t_stable', 't_volatile', 'p_stable', 'p_volatile', 'dof_stat'],
    'Value': [round(t_stable, 2), round(t_volatile, 2), round(p_stable, 2), round(p_volatile, 2), dof_stat]
}

# ------------------
# 6. Plot Regression between Descriptive Measures and Internalizing
# ------------------
# Plot Total Score vs G
r_ts_stable, p_ts_stable, t_ts_stable = plot_x_vs_y_FactorScores_robust(df_descriptive, 'g', 'TotalScore_B0',
                                                                              axes[1], title=True, tstat=True,
                                                                              fontsize=7,
                                                                              xlabel='General Factor',
                                                                              ylabel='Total Score',
                                                                              color_index=-2, line_color_index=-1)

r_ts_volatile, p_ts_volatile, t_ts_volatile = plot_x_vs_y_FactorScores_robust(df_descriptive, 'g',
                                                                                    'TotalScore_B1',
                                                                                    axes[1], title=True, tstat=True,
                                                                                    fontsize=7,
                                                                                    xlabel='General Factor',
                                                                                    ylabel='Total Score',
                                                                                    color_index=-5, line_color_index=-4)




# Plot Performance vs G
r_perf_stable, p_perf_stable, t_perf_stable = plot_x_vs_y_FactorScores_robust(df_descriptive, 'g', 'Performance_B0',
                                                                              axes[3], title=True, tstat=True,
                                                                              fontsize=7,
                                                                              xlabel='General Factor',
                                                                              ylabel='Performance (P(Correct))',
                                                                              color_index=-2, line_color_index=-1)

r_perf_volatile, p_perf_volatile, t_perf_volatile = plot_x_vs_y_FactorScores_robust(df_descriptive, 'g',
                                                                                    'Performance_B1',
                                                                                    axes[3], title=True, tstat=True,
                                                                                    fontsize=7,
                                                                                    xlabel='General Factor',
                                                                                    ylabel='P(Correct)',
                                                                                    color_index=-5, line_color_index=-4)

# Plot Switch Rate After Hits vs G
r_switch_hit_stable, p_switch_hit_stable, t_switch_hit_stable = plot_x_vs_y_FactorScores_robust(df_descriptive, 'g',
                                                                                                'SwitchPercentageHit_B0',
                                                                                                axes[5], title=True,
                                                                                                tstat=True, fontsize=7,
                                                                                                xlabel='General Factor',
                                                                                                ylabel='% Switches After Wins',
                                                                                                color_index=-2,
                                                                                                line_color_index=-1)

r_switch_hit_volatile, p_switch_hit_volatile, t_switch_hit_volatile = plot_x_vs_y_FactorScores_robust(df_descriptive,
                                                                                                      'g',
                                                                                                      'SwitchPercentageHit_B1',
                                                                                                      axes[5],
                                                                                                      title=True,
                                                                                                      tstat=True,
                                                                                                      fontsize=7,
                                                                                                      xlabel='General Factor',
                                                                                                      ylabel='% Switches After Wins',
                                                                                                      color_index=-5,
                                                                                                      line_color_index=-4)

# Plot Switch Rate After Misses vs G
r_switch_miss_stable, p_switch_miss_stable, t_switch_miss_stable = plot_x_vs_y_FactorScores_robust(df_descriptive, 'g',
                                                                                                   'SwitchPercentageMiss_B0',
                                                                                                   axes[7], title=True,
                                                                                                   tstat=True,
                                                                                                   fontsize=7,
                                                                                                   xlabel='General Factor',
                                                                                                   ylabel='% Switches After Losses',
                                                                                                   color_index=-2,
                                                                                                   line_color_index=-1)

r_switch_miss_volatile, p_switch_miss_volatile, t_switch_miss_volatile = plot_x_vs_y_FactorScores_robust(df_descriptive,
                                                                                                         'g',
                                                                                                         'SwitchPercentageMiss_B1',
                                                                                                         axes[7],
                                                                                                         title=True,
                                                                                                         tstat=True,
                                                                                                         fontsize=7,
                                                                                                         xlabel='General Factor',
                                                                                                         ylabel='% Switches After Losses',
                                                                                                         color_index=-5,
                                                                                                         line_color_index=-4)

# -----------------
# 7. Apply FDR Correction
# -----------------
p_fdr = sm.stats.fdrcorrection(
    [p_ts_stable, p_ts_volatile, p_perf_stable, p_perf_volatile, p_switch_hit_stable, p_switch_hit_volatile, p_switch_miss_stable,
     p_switch_miss_volatile],
    alpha=0.05, method='indep', is_sorted=False)
p_corrected = np.round(p_fdr[1], 2)

# -----------------
# 8. Add titles, labels and save figure
# -----------------
# Set titles for each subplot
set_subplot_title(axes[1], r_ts_stable, p_corrected[0], r_ts_volatile, p_corrected[1], fontsize)
set_subplot_title(axes[3], r_perf_stable, p_corrected[2], r_perf_volatile, p_corrected[3], fontsize)
set_subplot_title(axes[5], r_switch_hit_stable, p_corrected[4], r_switch_hit_volatile, p_corrected[5], fontsize)
set_subplot_title(axes[7], r_switch_miss_stable, p_corrected[6], r_switch_miss_volatile, p_corrected[7], fontsize)

# Add labels
texts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']  # label letters
label_subplots(f, texts, x_offset=0.05, y_offset=0.03)

sns.despine(f)
name = 'figure_s25.pdf'
savename = os.path.join(figure_folder, name)
# plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()

# ------------------
# 9. Combine stats into dataframes
# -----------------

stats_regression = {
    'Statistic': ['r_ts_stable', 'r_ts_volatile', 'p_ts_stable', 'p_ts_volatile', 't_ts_stable', 't_ts_volatile',
                'r_perf_stable', 'r_perf_volatile', 'p_perf_stable', 'p_perf_volatile', 't_perf_stable',
                  't_perf_volatile',
                  'r_hit_stable', 'r_hit_volatile', 'p_hit_stable', 'p_hit_volatile', 't_hit_stable', 't_hit_volatile',
                  'r_miss_stable', 'r_miss_volatile', 'p_miss_stable', 'p_miss_volatile', 't_miss_stable',
                  't_miss_volatile'],
    'Value': [r_ts_stable, r_ts_volatile, p_corrected[0], p_corrected[1], t_ts_stable, t_ts_volatile,
             r_perf_stable, r_perf_volatile, p_corrected[0], p_corrected[1], t_perf_stable, t_perf_volatile,
              r_switch_hit_stable, r_switch_hit_volatile, p_corrected[2], p_corrected[3], t_switch_hit_stable,
              t_switch_hit_volatile,
              r_switch_miss_stable, r_switch_miss_volatile, p_corrected[4], p_corrected[5], t_switch_miss_stable,
              t_switch_miss_volatile]}

# combine all dictionaries together
all_stats = {
    'Statistic': stats_ts['Statistic'] + stats_performance['Statistic'] + stats_switch_hit['Statistic'] + stats_switch_miss['Statistic'] +
                 stats_U['Statistic'] + stats_regression['Statistic'],
    'Value': stats_ts['Value'] + stats_performance['Value'] + stats_switch_hit['Value'] + stats_switch_miss['Value'] + stats_U['Value'] +
             stats_regression['Value']
}

# Save to csv
df_stats = pd.DataFrame(all_stats)
df_stats.set_index('Statistic', inplace=True)
df_stats.name = 'Supplementary_PRLRewardMag_PerformanceAndSwitchRates_FDRCorr_tstat'
print(df_stats)

