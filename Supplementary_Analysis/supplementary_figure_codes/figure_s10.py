# Regression of model parameters against general-factor scores across experiments

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functions.util_functions import cm2inch, label_subplots, medianprops, qns_factor_preprocessing
from functions.plotting_functions import plot_x_vs_y_FactorScores_robust


# Helper function for plots for each experiments
def plot_experiment(ax, ml_data, experiment_num, fontsize):
    r_b1, p_b1 = plot_x_vs_y_FactorScores_robust(ml_data, 'g', 'beta_1', ax, title=False, legend_txt='Fixed LR',
                                                 fontsize=fontsize, xlabel='General Factor', ylabel='Learning Rate',
                                                 color_index=-2, line_color_index=-1)
    r_b4, p_b4 = plot_x_vs_y_FactorScores_robust(ml_data, 'g', 'beta_4', ax, title=False, legend_txt='Adaptive LR',
                                                 fontsize=fontsize, xlabel='General Factor', ylabel='Learning Rate',
                                                 color_index=-5, line_color_index=-4)
    ax.set_title(
        f"Experiment {experiment_num}\n$r_{{fixed}}={r_b1}, p_{{fixed}}={p_b1}$\n$r_{{adaptive}}={r_b4}, p_{{adaptive}}={p_b4}$",
        fontsize=fontsize)
    return len(ml_data)

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

# Merge with model data
ml_data = ml_data.merge(df_merged, on='subjectID')

# Extract model data for each experiment
ml_data_exp2 = ml_data[ml_data['subjectID'].str.contains('SinglePredator_DifferingVolatility_GagnePredator')]
ml_data_exp3 = ml_data[ml_data['subjectID'].str.contains('SinglePredator_DifferingVolatility_GagneWithRewardMag')]
ml_data_exp4 = ml_data[ml_data['subjectID'].str.contains('SinglePredator_DifferingVolatility_GagneWithRewardLossMag')]

# Extract subject IDs from the three arrays
subjects_in_other_arrays = np.concatenate([
    ml_data_exp2['subjectID'].values,
    ml_data_exp3['subjectID'].values,
    ml_data_exp4['subjectID'].values
])

# Get the remaining subjects in ml_data
ml_data_exp1 = ml_data[~ml_data['subjectID'].isin(subjects_in_other_arrays)]

# -------------------
# 3. Set up the figure
# -------------------
# Size of figure

fig_width = 12
fig_height = 15
fontsize = 7
medianprops = medianprops()

colors = ['#77AADD', '#009988', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(2, 2, wspace=0.44, hspace=0.7, top=0.90, bottom=0.1, left=0.15, right=0.96)

axes = [plt.Subplot(f, gs_0[i // 2, i % 2]) for i in range(4)]
for ax in axes:
    f.add_subplot(ax)

# Plot Fixed and Adaptive LR regression against general factor for each experiment

subject_counts = [
        plot_experiment(axes[0], ml_data_exp1, 1, fontsize),
        plot_experiment(axes[1], ml_data_exp2, 2, fontsize),
        plot_experiment(axes[2], ml_data_exp3, 3, fontsize),
        plot_experiment(axes[3], ml_data_exp4, 4, fontsize)
    ]

# ------------------
# 4. Add labels and save figure
# ------------------
# Add labels
texts = ['a', 'b', 'c', 'd']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)

# add number of subjects
label_subplots(f, [f"$N$={count}" for count in subject_counts], x_offset=-0.03, y_offset=-0.025)
sns.despine(f)

name='figure_s10.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=500, transparent=False, bbox_inches='tight')
plt.show()