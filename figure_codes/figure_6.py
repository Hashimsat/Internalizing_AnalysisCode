# plot for lab study and SCR data
# Environment: lab_study_env

import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lab_scr_analysis.utils import merge_with_exclusions
from lab_scr_analysis.config import game_data_variables, excluded_participants
from lab_scr_analysis.scr_data_analysis_updated import analyse_scr_data
from lab_scr_analysis.dataframe_functions import create_average_epochs_dataframe
import scipy.stats as stats

from functions.predator_descriptive_functions import EstimationError,SingleTrialLR
from functions.plotting_functions import boxplots_lab, plot_x_vs_y_FactorScores_robust
from functions.util_functions import qns_factor_preprocessing, cm2inch, label_subplots

# -----------------
# 1. Load data
# -----------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
figure_folder = base_dir + '/figures/'

# Load model data and behavioral data
model_data = pd.read_csv(os.path.join(base_dir, 'data/lab_study/model_PEAlphaValence_Overall.csv'))
model_data_shocks = pd.read_csv(os.path.join(base_dir, 'data/lab_study/model_PEAlphaValence_Shock.csv'))
model_data_screams = pd.read_csv(os.path.join(base_dir, 'data/lab_study/model_PEAlphaValence_Scream.csv'))

df_predator = pd.read_csv(os.path.join(base_dir, 'data/lab_study/df_predator_exp7.csv'))

# Load questionnaire data
factor_scores = pd.read_csv(os.path.join(base_dir, 'data/lab_study/factor_scores_exp7.csv'))
qns_totalscore = pd.read_csv(os.path.join(base_dir, 'data/lab_study/qns_data_exp7.csv'), sep=';')


# Load scr data
df_all = pd.read_csv(os.path.join(base_dir, 'data/lab_study/all_data_zscored_exp7.csv'))
df_all_epochs = create_average_epochs_dataframe(df_all, '128', *game_data_variables)
df_all_epochs_miss = df_all_epochs[df_all_epochs['hit_miss'] == 'miss']

# -----------------
# 2. Preprocess behavioral and model data
# -----------------
# Preprocess qns and factor score data
df_qnstotal_subset_unmerged, df_factor, df_qnstotal_subset = qns_factor_preprocessing(qns_totalscore, factor_scores, drop_non_binary=True)
model_data = merge_with_exclusions(model_data, df_qnstotal_subset, excluded_participants, study=2)
model_data_shocks = merge_with_exclusions(model_data_shocks, df_qnstotal_subset, excluded_participants, study=2)
model_data_screams = merge_with_exclusions(model_data_screams, df_qnstotal_subset, excluded_participants, study=2)

# merge qns scores with mega df

df_predator_merged = merge_with_exclusions(df_predator,df_qnstotal_subset,excluded_participants, study=2)
df_predator_merged['ShockBlock'] = df_predator_merged['ShockBlock'].fillna(method='ffill')


# ----------------  
# 3. Extract Descriptive results
# ----------------

# calculate EE and LR
Subjects = pd.unique(df_predator_merged['subjectID'])
df_EE = EstimationError(df_predator_merged,Subjects,BlockName='ShockBlock')
df_EE = df_EE.merge(df_qnstotal_subset,on='subjectID')

df_LR = SingleTrialLR(df_predator_merged,Subjects,BlockName='ShockBlock',HitMissSeparation=False)
df_LR = df_LR.merge(df_qnstotal_subset,on='subjectID')


# ------------- 
# 2. Analyze SCR data
# -------------
result_hit_miss,pvals_hit_miss = analyse_scr_data(df_all_epochs, 'hit_miss', ['miss', 'hit'])
result_shock_block_miss,pvals_shock_miss = analyse_scr_data(df_all_epochs_miss, 'shock_block', ['scream block', 'shock block'])

# -----------------
# 3. Prepare figure
# -----------------

# size of figure
fig_height = 12
fig_width = 15
fontsize = 7

medianprops = dict(linestyle='-', linewidth=1, color='k')
colors = ["#80cdc1",'#de77ae', "#818589", "#018571"]
sns.set_palette(sns.color_palette(colors))

# create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# create plot grid
gs_0 = gridspec.GridSpec(2, 4, wspace=0.75, hspace=0.55, top=0.90, bottom=0.1, left=0.1, right=0.96)

# --------------------------------------------
# 4. Plot Descriptive and model results
# --------------------------------------------
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[0, 0])
ax_01 = plt.Subplot(f, gs_00[0])
f.add_subplot(ax_01)
t_EE,p_EE,dof_EE,n1_EE,n2_EE,median_EE_Scream,EE_Scream_IQR,median_EE_Shock,EE_Shock_IQR = boxplots_lab(df_EE,ax_01,fontsize=fontsize,abbr='EE',ylabel='Estimation Error',stat='ttest_rel')

ax_02 = plt.Subplot(f,gs_0[0,1])
f.add_subplot(ax_02)
t_LR,p_LR,dof_LR,n1_LR,n2_LR,median_LR_Scream,LR_Scream_IQR,median_LR_Shock,LR_Shock_IQR = boxplots_lab(df_LR,ax_02,fontsize=fontsize,abbr='LR',ylabel='Learning Rate',stat='ttest_rel')

# Plot model based results vs general factor scores

# Fixed LR
ax_03 = plt.Subplot(f, gs_0[0,2])
f.add_subplot(ax_03)

r_sc1,p_sc1,t_sc1 = plot_x_vs_y_FactorScores_robust(model_data_screams,'g','beta_1',ax_03,tstat=True,title=False,legend_txt='Screams',fontsize=7,xlabel='General Factor',ylabel='Fixed LR',color_index=0,line_color_index=0)
r_sh1,p_sh1,t_sh1 = plot_x_vs_y_FactorScores_robust(model_data_shocks,'g','beta_1',ax_03,tstat=True,title=False,legend_txt='Shocks',fontsize=7,xlabel='General Factor',ylabel='Fixed LR',color_index=1,line_color_index=1)

title_params = f"$r_{{sc}}={r_sc1}, p_{{sc}}={p_sc1}$\n$r_{{sh}}={r_sh1}, p_{{sh}}={p_sh1}$"
ax_03.set_title(title_params,fontsize=fontsize)

# Adaptive LR
ax_04 = plt.Subplot(f, gs_0[0,3])
f.add_subplot(ax_04)

r_sc4,p_sc4,t_sc4 = plot_x_vs_y_FactorScores_robust(model_data_screams,'g','beta_4',ax_04,tstat=True,title=False,legend_txt='Screams',fontsize=7,xlabel='General Factor',ylabel='Adaptive LR',color_index=0, line_color_index=0)
r_sh4,p_sh4,t_sh4 = plot_x_vs_y_FactorScores_robust(model_data_shocks,'g','beta_4',ax_04,tstat=True,title=False,legend_txt='Shocks',fontsize=7,xlabel='General Factor',ylabel='Adaptive LR',color_index=1, line_color_index=1)

title_params = f"$r_{{sc}}={r_sc4}, p_{{sc}}={p_sc4}$\n$r_{{sh}}={r_sh4}, p_{{sh}}={p_sh4}$"
ax_04.set_title(title_params,fontsize=fontsize)
ax_04.legend(loc='upper right', fontsize=fontsize-1, handlelength=0.75)


# --------------------------------------------
# 5. Plot SCR amplitude for hit and miss
# --------------------------------------------

# create subplot grid and axis for the first SCR plot

gs_10 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[1, 0:2])
ax_10 = plt.Subplot(f, gs_10[0])
f.add_subplot(ax_10)

# define variable and prepare dataset
hue_variable = 'hit_miss'
hue_order = ['miss','hit']
df = df_all_epochs.sort_values(by=[var for var in [hue_variable] if var is not None])

# plot lineplot
sns.lineplot(x='times', y='data', errorbar='se', hue=hue_variable, hue_order=hue_order,  data=df, ax=ax_10)
sns.despine()
ax_10.set_xlabel('Time since outcome onset (s)', fontsize=fontsize)
ax_10.set_ylabel('Amplitude (a.u.)', fontsize=fontsize)
ax_10.set_xlim(-2.0, 9.0)
ax_10.set_ylim(-0.05, 0.16)
ax_10.set_yticks([0.000, 0.05, 0.1, 0.15])
ax_10.xaxis.set_tick_params(labelsize=fontsize)
ax_10.yaxis.set_tick_params(labelsize=fontsize)
ax_10.set_title('Skin Conductance Response',fontsize=fontsize, pad=10)
legend = ax_10.legend(loc='upper right', fontsize=fontsize-1, handlelength=0.75)
legend.set_title('')
legend_colors = {text.get_text(): line.get_color() for text, line in zip(legend.texts, legend.legendHandles)}

# add lines for significant results after permutation testing
bar_vars = [result_hit_miss[key] for key in result_hit_miss]

# Convert list of lists to list of tuples
bar_vars = [item[0] for item in bar_vars]

for i, (start, end) in enumerate(bar_vars):
    if i == 0:
        color = colors[2]
    else:
        hue_value = list(result_hit_miss.keys())[i]
        color = legend_colors.get(hue_value, colors[i % len(colors)])
    ax_10.hlines(y=-0.03 - i * 0.005, xmin=start, xmax=end, color=color, linewidth=1)

# add vertical and horizontal lines at x = 0 and y = 0
ax_10.axvline(x=0, color='black', linestyle='--', linewidth=0.7)
ax_10.axhline(y=0, color='black', linestyle='--', linewidth=0.7)

for t in legend.texts:
    if t.get_text() == 'hit':
        t.set_text('Success')
    elif t.get_text() == 'miss':
        t.set_text('Failure')

# --------------------------------------------
# 4. Plot SCR amplitude for shocks and screams for misses only
# --------------------------------------------

# create subplot grid and axis for the second plot
gs_11 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[1, 2:4])
ax_11 = plt.Subplot(f, gs_11[0])
f.add_subplot(ax_11)

# define variable and prepare dataset
hue_variable = 'shock_block'
hue_order = ['shock block', 'scream block',]
df = df_all_epochs_miss.sort_values(by=[var for var in [hue_variable] if var is not None])

# plot lineplot
sns.lineplot(x='times', y='data', errorbar='se', hue=hue_variable, hue_order=hue_order,  data=df, ax=ax_11)
sns.despine()
ax_11.set_xlabel('Time since outcome onset (s)', fontsize=fontsize)
ax_11.set_ylabel('Amplitude (a.u.)', fontsize=fontsize)
ax_11.set_xlim(-2.0, 9.0)
ax_11.set_ylim(-0.05, 0.25)
ax_11.set_yticks([0.000, 0.05, 0.1, 0.15, 0.2])
ax_11.xaxis.set_tick_params(labelsize=fontsize)
ax_11.yaxis.set_tick_params(labelsize=fontsize)
ax_11.set_title('Skin Conductance Response',fontsize=fontsize,  pad=10)
legend = ax_11.legend(loc='upper right', fontsize=fontsize-1, handlelength=0.75)
legend.set_title('')
legend_colors = {text.get_text(): line.get_color() for text, line in zip(legend.texts, legend.legendHandles)}

# add lines for significant results after permutation testing
bar_vars = [result_shock_block_miss[key] for key in result_shock_block_miss]

# Convert list of lists to list of tuples
bar_vars = [item[0] for item in bar_vars]

for i, (start, end) in enumerate(bar_vars):
    if i == 0:
        color = colors[2]
    else:
        hue_value = list(result_shock_block_miss.keys())[i]
        color = legend_colors.get(hue_value, colors[i % len(colors)])
    ax_11.hlines(y=-0.02 - i * 0.01, xmin=start, xmax=end, color=color, linewidth=1)

# add vertical and horizontal lines at x = 0 and y = 0
ax_11.axvline(x=0, color='black', linestyle='--', linewidth=0.7)
ax_11.axhline(y=0, color='black', linestyle='--', linewidth=0.7)

# change legend texts
for t in legend.texts:
    if t.get_text() == 'shock block':
        t.set_text('Shocks')
    elif t.get_text() == 'scream block':
        t.set_text('Screams')

# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# label subplots
texts = ['a', 'b', 'c', 'd','e','f']  # label letters
label_subplots(f, texts, x_offset=0.08, y_offset=0.0)


# save figure
sns.despine(f)
plt.tight_layout()

name='figure_6_labstudy.pdf'
# name='DescriptiveFigure_Block0.png'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()



# -----------------
# 6. Save Data
# -----------------

# Extract stats for fixede and adaptive LR
median_FixedLR_screams = np.nanmedian(model_data_screams['beta_1'])
median_FixedLR_shocks = np.nanmedian(model_data_shocks['beta_1'])
t_FixedLR,p_FixedLR = stats.ttest_rel(model_data_screams['beta_1'],model_data_shocks['beta_1'])
n_FixedLR_shock = len(model_data_shocks['beta_1'])
n_FixedLR_scream = len(model_data_screams['beta_1'])

median_AdaptiveLR_screams = np.nanmedian(model_data_screams['beta_4'])
median_AdaptiveLR_shocks = np.nanmedian(model_data_shocks['beta_4'])
t_AdaptiveLR,p_AdaptiveLR = stats.ttest_rel(model_data_screams['beta_4'],model_data_shocks['beta_4'])
n_AdaptiveLR_shock = len(model_data_shocks['beta_4'])
n_AdaptiveLR_scream = len(model_data_screams['beta_4'])


stats = {
    'Statistic':['shockscream_t_EE','p_EE','dof_EE','n_shock','n_scream','median_EE_Screams','EE_Screams_IQIlow','EE_Screams_IQIhigh', 'median_EE_Shocks','EE_Shocks_IQIlow','EE_Shocks_IQIhigh',
                 'shockscream_t_LR','p_LR','dof_LR','median_LR_Screams','LR_Screams_IQIlow','LR_Screams_IQIhigh', 'median_LR_Shocks','LR_Shocks_IQIlow','LR_Shocks_IQIhigh',
                 'median_FixedLR_screams','median_FixedLR_shocks','shockscream_t_FixedLR','p_FixedLR',
                 'median_AdaptiveLR_screams','median_AdaptiveLR_shocks','shockscream_t_AdaptiveLR','p_AdaptiveLR',
                 'r_FixedLR_g_screams','p_FixedLR_g_screams','t_FixedLR_g_screams','r_FixedLR_g_shocks','p_FixedLR_g_shocks','t_FixedLR_g_shocks',
                 'r_AdaptiveLR_g_screams','p_AdaptiveLR_g_screams','t_AdaptiveLR_g_screams','r_AdaptiveLR_g_shocks','p_AdaptiveLR_g_shocks','t_AdaptiveLR_g_shocks',
                 'n_subj'
                 ],
    'Values':[t_EE,p_EE,dof_EE,n1_EE,n2_EE,median_EE_Scream,EE_Scream_IQR[0],EE_Scream_IQR[1],median_EE_Shock,EE_Shock_IQR[0],EE_Shock_IQR[1],
              t_LR,p_LR,dof_LR,median_LR_Scream,LR_Scream_IQR[0],LR_Scream_IQR[1],median_LR_Shock,LR_Shock_IQR[0],LR_Shock_IQR[1],
              median_FixedLR_screams,median_FixedLR_shocks,t_FixedLR,round(p_FixedLR,3),
              median_AdaptiveLR_screams,median_AdaptiveLR_shocks,t_AdaptiveLR,round(p_AdaptiveLR,3),
              r_sc1,round(p_sc1,3),t_sc1,r_sh1,round(p_sh1,3),t_sh1,
              r_sc4,round(p_sc4,3),t_sc4,r_sh4,round(p_sh4,3),t_sh4,
              len(df_EE),
              ],}

df_stats = pd.DataFrame(stats)
df_stats.set_index('Statistic', inplace=False)
df_stats['Values'] = df_stats['Values'].round(2)
df_stats.name = 'LabStudy_sp_descriptive_modelbased_results_tstat'

print(df_stats)

print('pvals_hitmiss',pvals_hit_miss)
print('pvals_shockmiss',pvals_shock_miss)

print('n1_EE,n2_EE',n1_EE,n2_EE)
print('n1_LR,n2_LR',n1_LR,n2_LR)
print('n_fixed_shock,n_fixed_scream',n_FixedLR_shock,n_FixedLR_scream)
print('n_adaptive_shock,n_adaptive_scream',n_AdaptiveLR_shock,n_AdaptiveLR_scream)