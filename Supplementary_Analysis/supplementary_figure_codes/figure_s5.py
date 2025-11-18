# Plot relationship between internalizing and EE and LR in different variability conditions

import numpy as np
import pandas as pd
from scipy.stats import zscore
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
from functions.util_functions import cm2inch, qns_factor_preprocessing, label_axes
from scipy import stats
from functions.predator_descriptive_functions import EstimationError, SingleTrialLR
from functions.plotting_functions import plot_descriptive_boxplots, plot_x_vs_y_FactorScores_robust


def create_subplot_and_plot(f, grid_spec, row, col, plot_type, data, x, y, xlabel, ylabel, fontsize, **kwargs):
    """
    Helper function to create a subplot and plot data.
    """
    ax = plt.Subplot(f, grid_spec[row, col])
    f.add_subplot(ax)

    if plot_type == "boxplot":
        result = plot_descriptive_boxplots(data, x=x, y=y, ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize,
                                           **kwargs)
    elif plot_type == "regression":
        result = plot_x_vs_y_FactorScores_robust(data, x=x, y=y, ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize,
                                                 **kwargs)
    else:
        raise ValueError("Invalid plot_type. Use 'boxplot' or 'regression'.")

    return ax, result


# -----------------
# 1. Load data
# -----------------
target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

df_predator = pd.read_csv(os.path.join(base_dir, 'data/predator_task/df_predator_4expdata_combined.csv'))
qns_totalscore = pd.read_csv(os.path.join(base_dir, 'data/factor_analysis/questionnaires_totalscores_subscales.csv'))
factor_scores = pd.read_csv(os.path.join(base_dir, 'data/factor_analysis/factor_scores.csv'))


# -----------------
# 2. Preprocess data
# -----------------

df_qns, df_fs, df_merged = qns_factor_preprocessing(qns_totalscore, factor_scores)

# get subjects that completed the predator task
Subjects_predator_init = pd.unique(df_predator['subjectID'])
# Remove elements that are NaN or 'nan'
Subjects_predator_init = [subj for subj in Subjects_predator_init if subj is not None and subj != 'nan' and subj == subj]

# Filter the factor score df based on the list of subjects
df_merged = df_merged[df_merged['subjectID'].isin(Subjects_predator_init)]

# standardize age, g and f scores
df_merged['g_z'] = zscore(df_merged['g'])
df_merged['Age_z'] = zscore(df_merged['Age'])
df_merged['F1_z'] = zscore(df_merged['F1.'])
df_merged['F2_z'] = zscore(df_merged['F2.'])

# Divide participants into low- and high-internalizing groups
mean_val = df_merged['g_z'].mean()
std_val = df_merged['g_z'].std()

# Create G_Category column based on conditions
df_merged['G_Category'] = pd.np.where(df_merged['g_z'] > mean_val, 'High',
                                pd.np.where(df_merged['g_z'] < mean_val, 'Low', 'Normal'))

# merge with predator task data
df_predator_merge = df_predator.merge(df_merged, on='subjectID')
BlockVersion_predator = np.sort(pd.unique(df_predator_merge['BlockVersion']))
Subjects_predator = pd.unique(df_predator_merge['subjectID'])


# -----------------
# 3. Calculate Estimation Error (EE) and Learning Rate (LR)
# -----------------
df_EE = EstimationError(df_predator_merge, Subjects_predator)
df_EE_merged = df_EE.merge(df_merged,on='subjectID')
df_EE_merged_LowHighAnx = df_EE_merged[df_EE_merged['G_Category'].isin(['High', 'Low'])]

df_LR = SingleTrialLR(df_predator_merge,Subjects_predator,BlockName='BlockVersion')
df_LR_merged = df_LR.merge(df_merged,on='subjectID')
df_LR_merged_LowHighAnx = df_LR_merged[df_LR_merged['G_Category'].isin(['High', 'Low'])]

# -----------------
# 4. Plot Estimation Error and Learning Rate
# -----------------
# Set figure size
fig_width = 15
fig_height = 18
fontsize = 7

# Initialize figure and grid
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
gs_0 = gridspec.GridSpec(4, 4, wspace=0.8, hspace=0.85, top=0.9, bottom=0.1, left=0.1, right=.99)

# Define blocks and parameters
blocks = [0, 1, 2, 3]
EE_params = {
    "boxplot": {"data": df_EE_merged_LowHighAnx, "x": "G_Category", "xlabel": "General Factor", "ylabel": "Estimation Error", "fontsize": fontsize, "stat": "ttest_ind"},
    "regression": {"data": df_EE_merged, "x": "g", "xlabel": "General Factor", "ylabel": "Estimation Error", "fontsize": fontsize, "tstat": True, "color_index": -2, "line_color_index": -1}
}
LR_params = {
    "boxplot": {"data": df_LR_merged_LowHighAnx, "x": "G_Category", "xlabel": "General Factor", "ylabel": "Learning Rate", "fontsize": fontsize, "stat": "ttest_ind"},
    "regression": {"data": df_LR_merged, "x": "g", "xlabel": "General Factor", "ylabel": "Learning Rate", "fontsize": fontsize, "tstat": True, "color_index": -2, "line_color_index": -1}
}

# Iterate over blocks and create subplots
results = {}
axes = {}
for i, block in enumerate(blocks):
    # EE boxplot
    ax, result = create_subplot_and_plot(f, gs_0, 0, i, "boxplot", y=f"EE_B{block}", min_val=10 if block == 0 else 15,
                                         max_val=70, **EE_params["boxplot"])
    results[f"EE_B{block}_boxplot"] = result
    axes[f"EE_B{block}_boxplot"] = ax

    # EE scatter
    ax, result = create_subplot_and_plot(f, gs_0, 1, i, "regression", y=f"EE_B{block}", **EE_params["regression"])
    results[f"EE_B{block}_regression"] = result
    axes[f"EE_B{block}_regression"] = ax

    # LR boxplot
    ax, result = create_subplot_and_plot(f, gs_0, 2, i, "boxplot", y=f"LR_B{block}", min_val=-0.1, max_val=1.2,
                                         **LR_params["boxplot"])
    results[f"LR_B{block}_boxplot"] = result
    axes[f"LR_B{block}_boxplot"] = ax

    # LR scatter
    ax, result = create_subplot_and_plot(f, gs_0, 3, i, "regression", y=f"LR_B{block}", **LR_params["regression"])
    results[f"LR_B{block}_regression"] = result
    axes[f"LR_B{block}_regression"] = ax


# ------------------
# 5. FDR Correction
# ------------------

# Collect p-values from results
p_values = []
r_values = []

for key, result in results.items():
    if isinstance(result, tuple) and len(result) > 1:
        r_values.append(result[0])
        p_values.append(result[1])

# Apply FDR correction
p_corrected = sm.stats.multipletests(p_values, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
p_corr_rounded = np.round(p_corrected[1],2)

# ---------------
# 6. Add stats values to subplots
# ---------------

ax_arr = []
for i, (key, ax) in enumerate(axes.items()):
    r = np.round(r_values[i], 2) if i < len(r_values) else None
    p = p_corr_rounded[i]
    ax_arr.append(ax)
    if "boxplot" in key:
        title = '$\it{p}$ = ' + str(np.round(p, 2))
    else:
        title = '$\it{r}$ = ' + str(r) + ', $\it{p}$ = ' + str(np.round(p, 2))

    ax.set_title(title, fontsize=fontsize)

# -----------------
# 7. Add labels to subplots
# -----------------

# Add vertical separation lines across the full figure (not just within subplots)
for x_pos in [0.26, 0.51, 0.76]:  # Normalized figure positions (0 = left, 1 = right)
    f.lines.append(plt.Line2D([x_pos, x_pos], [0, 1], transform=f.transFigure, color='#808080', linestyle='--', linewidth=0.75, alpha=0.8))

#
# Define column titles
column_titles = [f"Low Variability \nLow Hazard Rate",
                 f"High Variability \nLow Hazard Rate",
                 f"Low Variability \nHigh Hazard Rate",
                 f"High Variability \nHigh Hazard Rate",]

# Add supertitles for each column (positioned at the top)
for i, title in enumerate(column_titles):
    f.text(
        x=(i + 0.65) / 4,  # Centers the text within each column (normalized)
        y=0.95,  # Position slightly above the figure
        s=title,  # Text label
        fontsize=fontsize, fontweight="bold",
        ha="center"  # Center-align text
    )

# Add labels
texts = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']  # label letters
label_axes(f,ax_arr, texts, x_offset=0.08, y_offset=0.02, fontsize=fontsize)

sns.despine()
plt.tight_layout()

# -----------------
# 8. Save figure
# -----------------
name = 'figure_s5.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()






