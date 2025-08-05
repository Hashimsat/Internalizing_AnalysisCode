# Compare factor scores from our factor analysis with the factor scores obtained using loadings from Gagne et al., 2019
# environment: predator_task_env

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from functions.util_functions import cm2inch, label_subplots, qns_factor_preprocessing
from functions.plotting_functions import plot_x_vs_y_robust

# ----------------
# 1. Load data
# ----------------

# Get the directory of the current script
target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)

factor_scores_using_Gagne_clinic = pd.read_csv(os.path.join(target_dir, 'supplementary_data/factor_analysis'
                                                                        '/Predator_FS_using_GagneClinicalLoadings.csv'))

factor_score_using_Gagne_conf = pd.read_csv(
    os.path.join(target_dir, 'supplementary_data/factor_analysis/Predator_FS_using_GagneConfLoadings.csv'))

factor_score_predator = pd.read_csv(os.path.join(base_dir, 'data/factor_analysis/factor_scores.csv'), sep=',')

qns_totalscore = pd.read_csv(os.path.join(base_dir, 'data/factor_analysis/questionnaires_totalscores_subscales.csv'),
                             sep=',')

figure_folder = target_dir + "/supplementary_figures"

# ---------------------
# 2. Preprocess data
# ---------------------

# Rename columns in factor scores
factor_scores_using_Gagne_clinic = factor_scores_using_Gagne_clinic.rename(columns={'V1': 'subjectID'})
factor_score_using_Gagne_conf = factor_score_using_Gagne_conf.rename(columns={'V1': 'subjectID'})

# Merge factor score with qna data to ensure inattentive subjects are removed
df_qns, df_fs, df_merged = qns_factor_preprocessing(qns_totalscore, factor_score_predator, drop_non_binary=False)

# Merge predator factor scores with Gagne clinical and confirmatory factor scores
df_merged_clinic = df_merged.merge(factor_scores_using_Gagne_clinic, on='subjectID')
df_merged_conf = df_merged.merge(factor_score_using_Gagne_conf, on='subjectID')

# -------------------
# 3. Setup Figure
# -------------------

# Size of figure
fig_height = 12
fig_width = 15
fontsize = 7

# plot colors
colors = ["#80cdc1", '#de77ae', "#dfc27d", '#77AADD', "#018571", ]
sns.set_palette(sns.color_palette(colors))

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# set up the grid
# Create plot grid
gs_0 = gridspec.GridSpec(2, 1, wspace=0.5, hspace=0.85, top=0.87, bottom=0.1, left=0.15, right=0.98)

# FOR CLINICAL DATASET
# ---------------------
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_0[0:1, 0:1], wspace=0.5)
ax1 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax1)

# plot correlations: general factors
clin_r_g, clin_p_g = plot_x_vs_y_robust(df_merged_clinic, 'g_x', 'g_y', ax=ax1, title=True, xlabel='G-Our Dataset',
                                        ylabel='G-Gagne et al.', color_index=-2, line_color_index=-1)

# depression-related factors
ax2 = plt.Subplot(f, gs_00[0, 1])
f.add_subplot(ax2)

clin_r_dep, clin_p_dep = plot_x_vs_y_robust(df_merged_clinic, 'F2._x', 'F1._y', ax=ax2, title=True,
                                            xlabel='Dep Factor-Our Dataset', ylabel='Dep Factor-Gagne et al.',
                                            color_index=-2, line_color_index=-1)

# anxiety-related factors
ax3 = plt.Subplot(f, gs_00[0, 2])
f.add_subplot(ax3)

clin_r_anx, clin_p_anx = plot_x_vs_y_robust(df_merged_clinic, 'F1._x', 'F2._y', ax=ax3, title=True,
                                            xlabel='Anx Factor-Our Dataset', ylabel='Anx Factor-Gagne et al.',
                                            color_index=-2, line_color_index=-1)

# FOR CONFIRMATORY DATASET
# ---------------------
gs_01 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_0[1:2, 0:1], wspace=0.5)
ax4 = plt.Subplot(f, gs_01[0, 0])
f.add_subplot(ax4)

# plot correlations: general factors
conf_r_g, conf_p_g = plot_x_vs_y_robust(df_merged_conf, 'g_x', 'g_y', ax=ax4, title=True, xlabel='G-Our Dataset',
                                        ylabel='G-Gagne et al.', color_index=-2, line_color_index=-1)

# depression-related factors
ax5 = plt.Subplot(f, gs_01[0, 1])
f.add_subplot(ax5)

conf_r_dep, conf_p_dep = plot_x_vs_y_robust(df_merged_conf, 'F2._x', 'F1._y', ax=ax5, title=True,
                                            xlabel='Dep Factor-Our Dataset', ylabel='Dep Factor-Gagne et al.',
                                            color_index=-2, line_color_index=-1)

# anxiety-related factors
ax6 = plt.Subplot(f, gs_01[0, 2])
f.add_subplot(ax6)

conf_r_anx, conf_p_anx = plot_x_vs_y_robust(df_merged_conf, 'F1._x', 'F2._y', ax=ax6, title=True,
                                            xlabel='Anx Factor-Our Dataset', ylabel='Anx Factor-Gagne et al.',
                                            color_index=-2, line_color_index=-1)

# Add super titles for each row
f.text(0.55, 0.97, 'Gagne et al. Clinical dataset', ha='center', fontsize=fontsize + 1, fontweight='bold')
f.text(0.55, 0.47, 'Gagne et al. Online dataset', ha='center', fontsize=fontsize + 1, fontweight='bold')

plt.tight_layout()
sns.despine(f)

texts = ['a', 'b', 'c', 'd', 'e', 'f']
#
# # Add labels
label_subplots(f, texts, x_offset=0.07, y_offset=0.06, fontsize=fontsize)

# ------------------
# 4. Save figure
# ------------------

name = 'figure_s2.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format="pdf", dpi=700, transparent=True)
plt.show()
