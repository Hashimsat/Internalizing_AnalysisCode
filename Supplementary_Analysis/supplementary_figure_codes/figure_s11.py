# Plot fixed and adaptive LRs against questionnaire scores

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
from functions.plotting_functions import plot_x_vs_y_robust

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
ml_data = ml_data.merge(df_qns, on='subjectID')

# Set up questionnaire and model data arrays
qns_list = ['AI01', 'AI02', 'IC02', 'IU02', 'BD', 'MA01', 'PW', 'STICSA_somatic', 'STICSA_cognitive',
            'MASQ_AnxAr', 'MASQ_Anh']

qns_names = ['STAI-Y1', 'STAI-Y2', 'STICSA-T', 'IUS-27', 'BDI', 'MASQ', 'PSWQ', 'STICSA-somatic', 'STICSA-cognitive',
             'MASQ-AnxAr', 'MASQ-Anh']

model_param = ['beta_1', 'beta_4']

label_array = [r'Fixed LR ($\beta_1$)', r'Adaptive LR ($\beta_2$)']

# -------------------
# 3. Set up the figure
# -------------------
# Size of figure

fig_width = 15
fig_height = 20
fontsize = 7
medianprops = medianprops()

colors = ['#77AADD', '#009988', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create a grid of subplots
gs_0 = gridspec.GridSpec(6, 4, wspace=0.9, hspace=1.1, top=0.935, bottom=0.1, left=0.1, right=0.95)

# -----------------
# 4. Plot fixed and adaptive LRs against qns scores
# -----------------
r_array = []
p_array = []
t_array = []
ax_array = []
param_name_array = []

line_index = [-1, -4]
color_index = [-2, -5]

for param_no, param_name in enumerate(model_param):

    for q_no, q_name in enumerate(qns_list):
        # Create a subplot
        gs_01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[(param_no * 3) + (q_no // 4), q_no % 4])
        ax1 = plt.Subplot(f, gs_01[0, 0])
        f.add_subplot(ax1)

        # Plot fixed and adaptive LRs against qns scores

        r_b1, p_b1, t_b1 = plot_x_vs_y_robust(ml_data, q_name, param_name, ax1, tstat=True, title=False,
                                              fontsize=fontsize, xlabel=qns_names[q_no],
                                              ylabel=label_array[param_no], color_index=color_index[param_no],
                                              line_color_index=line_index[param_no])

        # Save p_vals for fdr correction
        p_array.append(p_b1)
        ax_array.append(ax1)
        r_array.append(r_b1)
        t_array.append(t_b1)

        param_label = param_name + '_' + q_name
        param_name_array.append(param_label)

# -----------------
# 5. FDR correction
# -----------------

# Apply FDR correction
p_fdr = sm.stats.fdrcorrection(p_array, alpha=0.05, method='indep', is_sorted=False)
p_corrected = np.round(p_fdr[1], 3)

# -----------------
# 6. Add title and label to the different axes
# -----------------

for i, ax in enumerate(ax_array):
    title = "$\it{r}$ = " + str(r_array[i]) + ', ' + "$\it{p}$ = " + str(p_corrected[i])
    ax.set_title(title, fontsize=fontsize)

# Add labels
texts = [chr(i) for i in range(ord('a'), ord('v') + 1)]
label_subplots(f, texts, x_offset=0.08, y_offset=0.022)

sns.despine()
plt.tight_layout()

# Save figure
name = "figure_s11.pdf"
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()

# -----------------
# Save data in a dataframe
# -----------------
p_param_array = ['p_' + s for s in param_name_array]
r_param_array = ['r_' + s for s in param_name_array]
t_param_array = ['t_' + s for s in param_name_array]

stats_model_regression = {'Statistic': r_param_array + p_param_array + t_param_array + ['n_total'],
                          'Values': r_array + p_corrected.tolist() + t_array + [len(ml_data)]}

df_stats = pd.DataFrame(stats_model_regression)
df_stats.set_index('Statistic', inplace=False)
df_stats['Values'] = df_stats['Values'].round(2)
df_stats.name = 'Supplementary_diffVol_ModelParamsVsQnsScores_FDRCorrected'
