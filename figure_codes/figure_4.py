# Plot descriptive plots for predator task results
# Environment: predator_task_env

import pandas as pd
from scipy.stats import zscore
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from functions.util_functions import cm2inch, label_subplots, qns_factor_preprocessing

from functions.predator_descriptive_functions import EstimationError_overall, PerseverationRate_overall, SingleTrialLR_overall, RT_InitConf_overall, combine_descriptive_with_factor_scores
from functions.LR_bins_internalizing import learning_rate_descriptive_internalizing
from functions.plotting_functions import plot_x_vs_y_FactorScores_robust, plot_descriptive_boxplots
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# -----------------
# 1. Load data
# -----------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
figure_folder = base_dir + '/figures/'

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

# merge with predator data
df_predator_merge = df_predator.merge(df_merged, on='subjectID')
Subjects = pd.unique(df_predator_merge['subjectID'])

# -----------------
# 3. Compute descriptive statistics
# -----------------

# Compute mean estimation errors for each subject across blocks
df_EE = EstimationError_overall(df_predator_merge, Subjects)
df_EE_merged, df_EE_merged_LowHighAnx = combine_descriptive_with_factor_scores(df_EE, df_merged, lowhighanx=True)

# Compute median LR for each subject
df_LR = SingleTrialLR_overall(df_predator_merge, Subjects)
df_LR_merged, df_LR_merged_LowHighAnx = combine_descriptive_with_factor_scores(df_LR, df_merged, lowhighanx=True)

# Compute perseveration percentage
df_pers = PerseverationRate_overall(df_predator_merge, Subjects)
df_pers_merged, df_pers_merged_LowHighAnx = combine_descriptive_with_factor_scores(df_pers, df_merged, lowhighanx=True)

# Compute median initiaiton RT for each subject
df_RT_init,_ = RT_InitConf_overall(df_predator_merge, Subjects)
df_RT_merged, df_RT_merged_LowHighAnx = combine_descriptive_with_factor_scores(df_RT_init, df_merged, lowhighanx=True)

n_lowG = len(pd.unique(df_LR_merged_LowHighAnx[df_LR_merged_LowHighAnx['G_Category'] == 'Low']['subjectID']))
n_highG = len(pd.unique(df_LR_merged_LowHighAnx[df_LR_merged_LowHighAnx['G_Category'] == 'High']['subjectID']))

# -----------------
# 4. Plotting
# -----------------
# Create figure

fig_width = 15
fig_height = 10
fontsize = 7

colors = ["#80cdc1",'#de77ae', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))

f = plt.figure(figsize=cm2inch(fig_width, fig_height))
gs_0 = gridspec.GridSpec(2, 4, wspace=0.8, hspace=0.5, top=0.92, bottom=0.1, left=0.1, right=.99)

# TOP ROW

# Plot LR across bins, separated by internalizing group
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[0,0:2])
ax0 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax0)

learning_rate_descriptive_internalizing(df_predator_merge, n_lowG=n_lowG, n_highG=n_highG, ax=ax0,
                                        Subjects=Subjects, colors=colors, fontsize=fontsize, n_bins=20)

# EE vs HighLowInternalizing
ax1 = plt.Subplot(f, gs_0[0, 2])
f.add_subplot(ax1)

(r_EE,p_EE,dof_EE,n1_EE,n2_EE,median_EE_Low,
 median_EE_Low_iqr,median_EE_High,median_EE_High_iqr) = plot_descriptive_boxplots(df_EE_merged_LowHighAnx, x='G_Category',
                                                                                  y='EE', ax=ax1,title=True,fontsize=fontsize,xlabel='General Factor',
                                                                                  ylabel='Estimation Error',min_val=20,stat='ttest_ind')

# EE vs general factor regression plot
ax2 = plt.Subplot(f, gs_0[0, 3])
f.add_subplot(ax2)

r_EE_reg,p_EE_reg,t_EE_reg = plot_x_vs_y_FactorScores_robust(df_EE_merged, 'g', 'EE', ax2,title=True,tstat=True,
                                                             fontsize=fontsize,color_index=-2,xlabel='General Factor',ylabel='Estimation Error',line_color_index=-1)


# BOTTOM ROW
# LR vs HighLowInternalizing
ax3 = plt.Subplot(f, gs_0[1, 0])
f.add_subplot(ax3)

(r_LR,p_LR,dof_LR,n1_LR,n2_LR,median_LR_Low,
 median_LR_Low_iqr,median_LR_High,median_LR_High_iqr) = plot_descriptive_boxplots(df_LR_merged_LowHighAnx, x='G_Category',
                                                                                  y='LR', ax=ax3,title=True,fontsize=fontsize,xlabel='General Factor',
                                                                                  ylabel='Learning Rate',min_val=-0.1,max_val=1,stat='ttest_ind')

# LR vs general factor regression plot
ax4 = plt.Subplot(f, gs_0[1, 1])
f.add_subplot(ax4)

r_LR_reg,p_LR_reg,t_LR_reg = plot_x_vs_y_FactorScores_robust(df_LR_merged, 'g', 'LR', ax4,title=True,tstat=True,
                                                             fontsize=fontsize,color_index=-2,xlabel='General Factor',ylabel='Learning Rate',line_color_index=-1)

# Perseveration vs HighLowInternalizing
ax5 = plt.Subplot(f, gs_0[1, 2])
f.add_subplot(ax5)

(r_pers,p_pers,dof_pers,n1_pers,n2_pers,median_pers_Low,
 median_pers_Low_iqr,median_pers_High,median_pers_High_iqr) = plot_descriptive_boxplots(df_pers_merged_LowHighAnx, x='G_Category',
                                                                                        y='Pers', ax=ax5,title=True,fontsize=fontsize,xlabel='General Factor',
                                                                                        ylabel='Perseveration Probability',min_val=-0.1,max_val=1,stat='ttest_ind')

# RT initiaion vs HighLowInternalizing
ax6 = plt.Subplot(f, gs_0[1, 3])
f.add_subplot(ax6)

(r_rt,p_rt,dof_rt,n1_rt,n2_rt,median_rt_Low,
 median_rt_Low_iqr,median_rt_High,median_rt_High_iqr) = plot_descriptive_boxplots(df_RT_merged_LowHighAnx, x='G_Category',
                                                                                  y='RT', ax=ax6,title=True,fontsize=fontsize,xlabel='General Factor',
                                                                                  ylabel='Reaction Time (ms)',min_val=100,max_val=1600,stat='ttest_ind')


# Add labels
texts = ['b', 'd', 'c', 'e', 'a', 'f','g']  # label letters
label_subplots(f, texts, x_offset=0.07, y_offset=0.03)

sns.despine(f)

# Save figure
name='figure_4_predator_descriptive.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()


#  -----------------
# 5. Combine stats into dataframes
# -----------------

regression_stats = {
    'Statistic':['B_EE','t_EE','p_EE',
                 'B_LR','t_LR','p_LR',
                 'n_subj'],
    'Values':[r_EE_reg,t_EE_reg,p_EE_reg,
             r_LR_reg,t_LR_reg,p_LR_reg,
              len(df_EE_merged)],
}

df_regression = pd.DataFrame(regression_stats)
df_regression.set_index('Statistic', inplace=False)
df_regression['Values'] = df_regression['Values'].round(2)
df_regression.name = 'Descriptive_predator_regression'

stats = {
    'Statistic':['t_EE_lowhigh','p_EE_lowhigh','n_Low','n_High','dof_EE','median_EE_Low','median_EE_Low_IQIlow','median_EE_Low_IQIhigh', 'median_EE_High','median_EE_High_IQIlow','median_EE_High_IQIhigh',
                 't_LR_lowhigh','p_LR_lowhigh','dof_LR','median_LR_Low','median_LR_Low_IQIlow','median_LR_Low_IQIhigh', 'median_LR_High','median_LR_High_IQIlow','median_LR_High_IQIhigh',
                 't_pers_lowhigh','p_pers_lowhigh','dof_pers','median_pers_Low','median_pers_Low_IQIlow','median_pers_Low_IQIhigh', 'median_pers_High','median_pers_High_IQIlow','median_pers_High_IQIhigh',
                 't_rt_lowhigh','p_rt_lowhigh','dof_rt','median_rt_Low','median_rt_Low_IQIlow','median_rt_Low_IQIhigh', 'median_rt_High','median_rt_High_IQIlow','median_rt_High_IQIhigh'],

    'Values':[r_EE,round(p_EE,2),n1_EE,n2_EE,dof_EE,median_EE_Low,median_EE_Low_iqr[0],median_EE_Low_iqr[1],median_EE_High,median_EE_High_iqr[0],median_EE_High_iqr[1],
              r_LR,round(p_LR,2),dof_LR,median_LR_Low,median_LR_Low_iqr[0],median_LR_Low_iqr[1],median_LR_High,median_LR_High_iqr[0],median_LR_High_iqr[1],
              r_pers,round(p_pers,2),dof_pers,median_pers_Low,median_pers_Low_iqr[0],median_pers_Low_iqr[1],median_pers_High,median_pers_High_iqr[0],median_pers_High_iqr[1],
              r_rt,round(p_rt,2),dof_rt,median_rt_Low,median_rt_Low_iqr[0],median_rt_Low_iqr[1],median_rt_High,median_rt_High_iqr[0],median_rt_High_iqr[1]],
}

df_stats = pd.DataFrame(stats)
df_stats.set_index('Statistic', inplace=False)
df_stats['Values'] = df_stats['Values'].round(2)
df_stats.name = 'Descriptive_predator_LowHighG_tstat'

print(df_stats)



