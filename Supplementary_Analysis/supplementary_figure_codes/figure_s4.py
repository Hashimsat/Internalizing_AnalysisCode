# Plot predator task estimation errors for different blocks of each differing variability experiment

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import ttest_rel
from functions.util_functions import cm2inch, label_subplots, qns_factor_preprocessing
from functions.predator_descriptive_functions import Estimation_Error
from functions.plotting_functions import plot_EE_across_blocks


# -----------------
# Custom functions for this script
# -------------------
def extract_experiment_df(df, df_factor, prefix=None):
    """
    :param df: DataFrame containing combined predator task data for all experiments
    :param prefix: Prefix to filter subjects by experiment type (e.g., 'diffVol', 'ReversalLearningPure', etc.)
    :param df_factor: DataFrame containing factor scores (filtered for inattentive participants)
    :return: df_exp: DataFrame containing only data from specific required experiment
    :return: Subjects_exp: Array of unique subject IDs from the filtered DataFrame
    """

    # Ensure subjectID is a string and filter rows based on the prefix
    filtered_df = df[df['subjectID'].str.contains(prefix, na=False)]

    # Only keep subjects for which we also have factor scores (filtering for inattentive subjects)
    filtered_df = filtered_df[filtered_df['subjectID'].isin(df_factor['subjectID'])]

    # Extract unique subject IDs
    subjects_exp = filtered_df['subjectID'].unique()

    return filtered_df, subjects_exp


# -----------------
# 1. Load Data
# -----------------
# Get the directory of the current script
target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

df_predator = pd.read_csv(os.path.join(base_dir, 'data/predator_task/df_predator_4expdata_combined.csv'))
factor_scores = pd.read_csv(os.path.join(base_dir, 'data/factor_analysis/factor_scores.csv'))
qns_totalscore = pd.read_csv(os.path.join(base_dir, 'data/factor_analysis/questionnaires_totalscores_subscales.csv'))

# -----------------
# 2. Extract Data for Each Experiment
# -----------------

# Merge factor score with qna data to ensure inattentive subjects are removed
df_qns, df_fs, df_merged = qns_factor_preprocessing(qns_totalscore, factor_scores, drop_non_binary=True)

# Extract data for each experiment type
df_diffVol_reversal_pure, subjects_diffVol_reversal_pure = extract_experiment_df(df_predator, df_merged,
                                                                                 prefix='SinglePredator_DifferingVolatility_GagnePredator')
df_diffVol_reversal_mag, subjects_diffVol_reversal_mag = extract_experiment_df(df_predator, df_merged,
                                                                               prefix='SinglePredator_DifferingVolatility_GagneWithRewardMag_')
df_diffVol_reversal_lossreward, subjects_diffVol_reversal_lossreward = extract_experiment_df(df_predator, df_merged,
                                                                                             prefix='SinglePredator_DifferingVolatility_GagneWithRewardLossMag')

df_diffVol, _ = extract_experiment_df(df_predator, df_merged, prefix='SinglePredator_DifferingVolatility_')

# filter df_diffVol to remove participants that are in the other 3 experiments
subjects_excluded = np.concatenate(
    (subjects_diffVol_reversal_pure, subjects_diffVol_reversal_mag, subjects_diffVol_reversal_lossreward))
df_diffVol = df_diffVol[~df_diffVol['subjectID'].isin(subjects_excluded)]
subjects_diffVol = df_diffVol['subjectID'].unique()

# -----------------
# 3. Calculate Estimation Errors
# -----------------
EE_diffVol_reversal_pure = Estimation_Error(df_diffVol_reversal_pure, subjects_diffVol_reversal_pure, block_name='BlockVersion')
EE_diffVol_reversal_mag = Estimation_Error(df_diffVol_reversal_mag, subjects_diffVol_reversal_mag, block_name='BlockVersion')
EE_diffVol_reversal_lossreward = Estimation_Error(df_diffVol_reversal_lossreward, subjects_diffVol_reversal_lossreward, block_name='BlockVersion')
EE_diffVol = Estimation_Error(df_diffVol, subjects_diffVol, block_name='BlockVersion')

# -----------------
# 4. Plot Estimation Errors
# -----------------

# Size of figure
fig_height = 12
fig_width = 15
fontsize = 7

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(2, 2, wspace=0.44, hspace=0.4, top=0.90, bottom=0.1, left=0.15, right=0.96)

ax_0 = plt.Subplot(f, gs_0[0, 0])
ax_1 = plt.Subplot(f, gs_0[0, 1])
ax_2 = plt.Subplot(f, gs_0[1, 0])
ax_3 = plt.Subplot(f, gs_0[1, 1])
f.add_subplot(ax_0)
f.add_subplot(ax_1)
f.add_subplot(ax_2)
f.add_subplot(ax_3)

# Plot estimation errors across blocks
plot_EE_across_blocks(EE_diffVol, ax=ax_0, title='Experiment 1', fontsize=fontsize, Legend=True)
plot_EE_across_blocks(EE_diffVol_reversal_pure, ax=ax_1, title='Experiment 2', fontsize=fontsize)
plot_EE_across_blocks(EE_diffVol_reversal_mag, ax=ax_2, title='Experiment 3', fontsize=fontsize)
plot_EE_across_blocks(EE_diffVol_reversal_lossreward, ax=ax_3, title='Experiment 4', fontsize=fontsize)

# Add labels
texts = ['a', 'b', 'c', 'd']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)
sns.despine(f)

# -----------------
# 5. Save figure
# -----------------
name = 'figure_s4.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()

# --------------------
# Extract optional data
# --------------------
Subjects_allcombined = np.concatenate((subjects_excluded, subjects_diffVol))

# extract EE for low noise and high noise conditions
df_low_noise = df_predator[df_predator['BlockVersion'].isin([0, 2])]
EE_low_noise = Estimation_Error(df_low_noise, Subjects_allcombined)
median_low_noise = round(np.median(EE_low_noise['EE']), 2)
IQR_low_noise = np.percentile(EE_low_noise['EE'], [25, 75])

df_high_noise = df_predator[df_predator['BlockVersion'].isin([1, 3])]
EE_high_noise = Estimation_Error(df_high_noise, Subjects_allcombined)
median_high_noise = round(np.median(EE_high_noise['EE']), 2)
IQR_high_noise = np.percentile(EE_high_noise['EE'], [25, 75])

# extract EE for low HR and high HR conditions
df_low_HR = df_predator[df_predator['BlockVersion'].isin([0, 1])]
EE_low_HR = Estimation_Error(df_low_HR, Subjects_allcombined)
median_low_HR = round(np.median(EE_low_HR['EE']), 2)
IQR_low_HR = np.percentile(EE_low_HR['EE'], [25, 75])

df_high_HR = df_predator[df_predator['BlockVersion'].isin([2, 3])]
EE_high_HR = Estimation_Error(df_high_HR, Subjects_allcombined)
median_high_HR = round(np.median(EE_high_HR['EE']), 2)
IQR_high_HR = np.percentile(EE_high_HR['EE'], [25, 75])

# do a paired t-test between low and high noise conditions
res_noise = ttest_rel(EE_low_noise['EE'], EE_high_noise['EE'])
t_noise = res_noise.statistic
p_noise = res_noise.pvalue
dof_noise = res_noise.df

res_HR = ttest_rel(EE_low_HR['EE'], EE_high_HR['EE'])
t_HR = res_HR.statistic
p_HR = res_HR.pvalue
dof_HR = res_HR.df

print('done')

# save in dataframe
stats = {
    'Statistic': ['median_EE_low_noise', 'IQR_low_noise_low', 'IQR_low_noise_high', 'median_EE_high_noise',
                  'IQR_high_noise_low', 'IQR_high_noise_high',
                  'median_EE_low_HR', 'IQR_low_HR_low', 'IQR_low_HR_high', 'median_EE_high_HR', 'IQR_high_HR_low',
                  'IQR_high_HR_high',
                  't_noise', 'p_noise', 'dof_noise', 't_HR', 'p_HR', 'dof_HR'],

    'Values': [median_low_noise, IQR_low_noise[0], IQR_low_noise[1], median_high_noise, IQR_high_noise[0],
               IQR_high_noise[1],
               median_low_HR, IQR_low_HR[0], IQR_low_HR[1], median_high_HR, IQR_high_HR[0], IQR_high_HR[1],
               t_noise, p_noise, dof_noise, t_HR, p_HR, dof_HR]
}

df_stats = pd.DataFrame(stats)
df_stats.set_index('Statistic', inplace=False)
df_stats['Values'] = df_stats['Values'].round(2)
df_stats.name = 'Supplementary_EE_DiffVol_NoiseHRComparison_tstat'
