# Model comparison for the predator task data


import numpy as np
import pandas as pd
from scipy.stats import zscore
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from functions.util_functions import cm2inch

# BIC function
def calculate_bic(log_likelihood, num_params, num_trials):
    return np.log(num_trials) * num_params - 2 * log_likelihood

def merge_n_trials_return_sumBIC(model,df_ntrials,n_ext):
    model['N_params'] = len(model.columns) - n_ext
    model = model.merge(df_ntrials, on='subjectID')
    model['BIC'] = calculate_bic(-model['llh'],model['N_params'],model['N_Trials'])
    sum_BIC = model['BIC'].sum()

    return sum_BIC


# -----------------
# 1. Load data
# -----------------
target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

df_predator = pd.read_csv(os.path.join(base_dir, 'data/predator_task/df_predator_4expdata_combined.csv'))

# Load models
model_PEAlphaValence = pd.read_csv(os.path.join(base_dir,"data/predator_task/df_predator_4exp_modelresults.csv"))

model_PEAlpha_NoValence = pd.read_csv(os.path.join(target_dir,
                                            'supplementary_data/predator_task/df_4exp_model_PEAlphaOnly.csv'))

model_PEValence = pd.read_csv(os.path.join(target_dir,
                                            'supplementary_data/predator_task/df_4exp_model_PEValenceOnly.csv'))
model_PEOnly = pd.read_csv(os.path.join(target_dir,
                                            'supplementary_data/predator_task/df_4exp_model_PEOnly.csv'))
# --------------------
# 2. Process data
# --------------------
n_ext = 5  # no of extra parameters in the model dataframes

# Extract llh for each model
llh_values = [np.sum(model_PEAlpha_NoValence['llh'].values), np.sum(model_PEAlphaValence['llh'].values),
               np.sum(model_PEValence['llh'].values),
              np.sum(model_PEOnly['llh'].values), ]

# calculate number of trials for each participant
n_trials = df_predator['subjectID'].value_counts()

# Convert the result into a DataFrame
n_trials_df = n_trials.reset_index()
n_trials_df.columns = ['subjectID', 'N_Trials']

# calculate summed BIC for each model
BIC_sum_PEAlpha_NoValence = merge_n_trials_return_sumBIC(model_PEAlpha_NoValence,n_trials_df,n_ext)
BIC_sum_PEAlphaValence = merge_n_trials_return_sumBIC(model_PEAlphaValence,n_trials_df,n_ext)
BIC_sum_PEValence = merge_n_trials_return_sumBIC(model_PEValence,n_trials_df,n_ext)
BIC_sum_PEOnly = merge_n_trials_return_sumBIC(model_PEOnly,n_trials_df,n_ext)

# -------------------------
# 3. Prepare data for plotting
# -------------------------
BIC_values = [BIC_sum_PEOnly,
              BIC_sum_PEValence,
              BIC_sum_PEAlpha_NoValence,
              BIC_sum_PEAlphaValence,
               ]

BIC_values = [round(value, 2) for value in BIC_values]

model_names_full = ['PEOnly',
                    'PEValence',
                    'PEAlpha_NoValence',
                    'PEAlphaValence',]

model_names = ['Fixed LR', 'Fixed LR \n Valence', 'Fixed LR \n Adaptive LR', 'Fixed LR \n Adaptive LR \n Valence']

# -------------------------
# 4. Plotting
# -------------------------

# setup figure
fig_width = 10
fig_height = 8
fontsize = 7

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

gs_0 = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.6, top=0.85, bottom=0.2, left=0.2, right=0.90)
ax = plt.Subplot(f, gs_0[0, 0])
f.add_subplot(ax)

# plot bar plot to compare BIC values
colors = ['#77AADD', '#77AADD', '#77AADD', '#77AADD']

# Plot bar plot
bar_width = 0.4
bar_positions = [0,1.25,2.5,3.75]
bars = ax.bar(bar_positions, BIC_values, color=colors, edgecolor='black')


# Set x-tick labels
ax.set_xticks([0, 1.25, 2.5, 3.75])
ax.set_xticklabels(model_names, rotation=0)

# Set labels and title
ax.set_ylabel('Summed BIC', fontsize=fontsize)
ax.set_xlabel('Models', fontsize=fontsize)

# Add the legend to the plot
ax.tick_params(axis='both', which='major', labelsize=fontsize)

# Save the plot
sns.despine()
plt.tight_layout()

name = "figure_s6.pdf"
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()

# -------------------------
# 5. Save data
# -------------------------
# save the BIC values in a dictionary
stats_BIC = {'Statistic':model_names_full,
                     'Values':BIC_values
                     }

df_stats_model_regression = pd.DataFrame(stats_BIC)
df_stats_model_regression.set_index('Statistic', inplace=False)
df_stats_model_regression.name = 'Supplementary_Predator_ModelComparisonBIC'

