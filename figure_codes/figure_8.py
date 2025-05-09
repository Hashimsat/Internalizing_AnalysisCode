# Plot descriptive and model results from the reversal learning task
# Python Environment: reversal_task_env


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from functions.util_functions import cm2inch, label_subplots,label_axes, compute_median_iqr, qns_factor_preprocessing, compute_test_statistic
import seaborn as sns
from scipy.stats import zscore
import pickle
from functions.prl_plotting_functions import plot_param_posterior_distribution_onesubplot,plot_factor_errorbar, param_by_factor_score,extract_distribution_mean_hdpis
from functions.prl_descriptive_functions import performance_prl

# -----------------
# 1. Load data
# -----------------

figure_folder = '../figures/'
qns_totalscore = pd.read_csv('../data/factor_analysis/questionnaires_totalscores_subscales.csv')
factor_scores = pd.read_csv('../data/factor_analysis/factor_scores.csv')

pickle_filepath = f"../data/reversal_task/prl_nomag_model6_covariate=Bi3itemCDM_date=2025_1_14_samples=2500tune=1200_seed=3_exp=3.pkl"
data_path = "../data/reversal_task/prl_nomag_model_data.pkl"
df_prl = pd.read_csv("../data/reversal_task/df_prl_NoMag_AllData.csv")

# -------------------
# 2. Preprocess data
# -------------------
df_qns, df_fs, df_merged = qns_factor_preprocessing(qns_totalscore, factor_scores)

with open(data_path,'rb') as f:
    data = pickle.load(f)

# load model and model trace from provided file path
with open(pickle_filepath , 'rb') as buff:
    model_dict = pickle.load(buff)

trace = model_dict['trace']
model = model_dict['model']

# extract subjects that completed the reversal learning task
Subjects_reversal = pd.unique(df_prl['subjectID'])
df_merged = df_merged[df_merged['subjectID'].isin(Subjects_reversal)]

# add G category
# standardize g and f scores
df_merged['g_z'] = zscore(df_merged['g'])

mean_val = df_merged['g_z'].mean()
std_val = df_merged['g_z'].std()

# Create G_Category column based on conditions
df_merged['G_Category'] = pd.np.where(df_merged['g_z'] < mean_val, 'Low',
                                pd.np.where(df_merged['g_z'] > mean_val, 'High', 'Normal'))

# ---------------------
# 3. Compute P(Correct)
# ---------------------
Subjects = pd.unique(df_prl['subjectID'])
Blocks=[0,1]

df_performance = performance_prl(df_prl,Subjects,Blocks)

# merge with factor scores
df_performance = df_performance.merge(df_merged, on='subjectID')

# Reshape the DataFrame to a long format
df_long = pd.melt(df_performance, id_vars=['G_Category'],
                  value_vars=['Performance_B0', 'Performance_B1'],
                  var_name='PerformanceBlock', value_name='Performance')

# Extract unique categories and performance blocks
categories = df_long['G_Category'].unique()
categories = categories[::-1]
performance_blocks = df_long['PerformanceBlock'].unique()

# -------------
# 4. Set up the figure and plot
# -------------
fig_width = 15
fig_height = 13
fontsize = 7
medianprops = dict(linestyle='-', linewidth=1, color='k')
boxprops = dict(alpha=0.7)
meanprops = None
whiskerprops = dict(color='k', linewidth=0.5)
colors_bp = {'Low': "#80cdc1",'High': '#de77ae'}

colors = ["#80cdc1",'#de77ae', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))

f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(2, 2, wspace=0.5, hspace=0.7, top=0.95, bottom=0.2, left=0.13, right=0.99)

# ----------
# 5. Plot P(Correct)
# ----------

# Plot P(Correct) vs low and high internalizing, separated by task phase (stable or volatile)
gs_01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[0:1,0:1])
ax1 = plt.Subplot(f, gs_01[0, 0:2])
f.add_subplot(ax1)

# Loop over each PerformanceBlock and G_Category to plot the boxplots
for i, perf_block in enumerate(performance_blocks):
    for j, category in enumerate(categories):
        # Filter the data
        data_bp = df_long[(df_long['PerformanceBlock'] == perf_block) & (df_long['G_Category'] == category)]['Performance']

        # Plot the boxplot
        ax1.boxplot(data_bp, positions=[i + (j * 0.25) - 0.125], widths=0.2, patch_artist=True,
                   boxprops=dict(alpha=0.5, linewidth=0.5,facecolor=colors_bp[category], color='k'),
                    whiskerprops=whiskerprops,capprops=whiskerprops,
                    medianprops=medianprops, meanprops=meanprops,
                    showmeans=False,showfliers=False)

        # Plot data points
        # Add jitter to the x-position of each point
        jitter_strength = 0.03
        jittered_x = (i + (j * 0.25) - 0.125) + np.random.normal(0, jitter_strength, size=len(data_bp))
        ax1.scatter(jittered_x, data_bp,s=10, color=colors_bp[category],marker='o', alpha=0.3, facecolor=None,edgecolors='none')

# Calculate test statistics (Welch's t test between low and high internalizing)
# Calculate p-values
t_stable, p_stable, dof_stable, n_lowG, n_highG = compute_test_statistic(df_performance, 'G_Category', 'Performance_B0', 'Low', 'High', test='ttest_ind')
t_volatile, p_volatile, dof_volatile,_,_ = compute_test_statistic(df_performance, 'G_Category', 'Performance_B1', 'Low', 'High', test='ttest_ind')

# set the title, axes properties and legend
title = "$\it{p_{\mathrm{stable}}}$ = " + str(round(p_stable, 2)) + ", $\it{p_{\mathrm{volatile}}}$ = " + str(round(p_volatile, 2))
ax1.set_title(title, fontsize=fontsize)
ax1.set_ylabel('P(Correct)', fontsize=fontsize)
ax1.set_xlabel('Task Phase', fontsize=fontsize)
ax1.set_ylim([0,1])
ax1.set_xticks([0.0,1.0])
ax1.set_xticklabels(labels=['Stable', 'Volatile'])
ax1.xaxis.set_tick_params(labelsize=fontsize)
ax1.yaxis.set_tick_params(labelsize=fontsize)
ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

# Create legend handles
low_patch = mpatches.Patch(color=colors_bp['Low'], label='Low G')
high_patch = mpatches.Patch(color=colors_bp['High'], label='High G')

# Add the legend to the plot
ax1.legend(handles=[low_patch, high_patch], title='',fontsize=fontsize-1,handlelength=1,handleheight=0.3, loc=4)

# ----------------
# 6. Plot model parameter distibutions
# ----------------
gs_02 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[0:1,1:2])
ax2 = plt.Subplot(f, gs_02[0, 0])
f.add_subplot(ax2)

# Plot population means posterior distribution
pal_dark = ['#77AADD','#77AADD','#77AADD','#77AADD']
plot_param_posterior_distribution_onesubplot(trace=trace,  # data
                                             params=model.params,  # model parameter names
                                             gp='u',  # group parameter
                                             param='lr',  # readable name
                                             offset=-0.15,
                                             fontsize=fontsize,
                                             bp_width=0.2,
                                             ax=ax2,  # plot characteristics
                                             colors=pal_dark,
                                             legend=False,
                                             legendlabel='posterior mean (with 95% HDI)',
                                             ylabel='Group mean ($μ_o$) for \n learning rate components \n(in logit space)',
                                             s_bar=1,
                                             elinewidth=1,
                                             ebar_offset=-0.05,
                                             legend_anchor=[0.6, 0.02],
                                             boxplot=False)

# extract group level effects for LR and inv_temp
df_group_lr = extract_distribution_mean_hdpis(trace=trace, model=model, factor='u', param='lr')
df_group_beta = extract_distribution_mean_hdpis(trace=trace, model=model, factor='u', param='Binv')


# Plot general factor effects
gs_03 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_0[1:,0:], wspace = 0.6)
ax3 = plt.Subplot(f, gs_03[0, 0])
f.add_subplot(ax3)

plot_factor_errorbar(trace=trace, params=model.params,
                         ax=ax3,
                         factor='u_PC1',
                         ylabel='Effect of general factor ($β_g$) \n on learning rate components \n(in logit space)',
                         legendlabel='posterior mean \n (with 95% HDI)',
                         legendloc='best',
                         fontsize=7,
                         legend=True,
                         color='black',
                         elinewidth=1,
                         legend_anchor=[1, -0.55])

# separate general factor effects for low and high internalizing
ax4_1 = plt.Subplot(f, gs_03[0, 1])
ax4_2= plt.Subplot(f, gs_03[0, 2])
f.add_subplot(ax4_1)
f.add_subplot(ax4_2)

for i, (ax, split) in enumerate(zip([ax4_1, ax4_2],
                                    ['low', 'high', ])):

    param_by_factor_score(trace, data, model,
                          param='lr',
                          pc='u_PC1',
                          ax=ax,
                          median=False,
                          split=split,
                          transform='invlogit',
                          scatter_offset=0.05,
                          legendloc='upper right',
                          s=2,
                          markersize=4, elinewidth=1.5,
                          include_errorbar=True,
                          ebar_offset=-0.05, fontsize=fontsize,
                          legend_anchor=[1, -0.55])


# Add labels
texts = ['a', '', '', 'd', 'e']  # label letters
label_subplots(f, texts, x_offset=0.07, y_offset=0.03,fontsize=fontsize)

# Label letters
texts = ['b']
label_axes(f, [ax2],texts, x_offset=0.13, y_offset=0.03,fontsize=fontsize)

texts = ['c']
label_axes(f, [ax3],texts, x_offset=0.10, y_offset=0.03,fontsize=fontsize)

sns.despine(f)

# -----------------
# 7. Save figure
# -----------------
name='figure_8_prl_model.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()

# ---------------
# 9. Data Stats
# ---------------
# Stats for P(Correct)

def compute_performance_stats(arr):
    mean_val = np.round(np.nanmean(arr),2)
    std_val = np.round(np.nanstd(arr),2)
    median_val, iqr_val = compute_median_iqr(arr)
    return mean_val, std_val, median_val, iqr_val

# stable phase

mean_stable_LowG, std_stable_LowG, median_stable_LowG, stable_LowG_IQI = compute_performance_stats(df_performance[df_performance['G_Category'] == 'Low']['Performance_B0'])
mean_stable_HighG, std_stable_HighG, median_stable_HighG, stable_HighG_IQI = compute_performance_stats(df_performance[df_performance['G_Category'] == 'High']['Performance_B0'])

# volatile phase
mean_volatile_LowG, std_volatile_LowG, median_volatile_LowG, volatile_LowG_IQI = compute_performance_stats(df_performance[df_performance['G_Category'] == 'Low']['Performance_B1'])
mean_volatile_HighG, std_volatile_HighG, median_volatile_HighG, volatile_HighG_IQI = compute_performance_stats(df_performance[df_performance['G_Category'] == 'High']['Performance_B1'])


stats_PCorrect = {'Statistic':['mean_stable_lowG','std_stable_lowG','median_stable_lowG','stable_lowG_25','stable_lowG_75',
                                  'mean_stable_highG','std_stable_highG','median_stable_highG','stable_highG_25','stable_highG_75',
                                  'stable_t','stable_p','stable_dof',
                                  'mean_volatile_lowG','std_volatile_lowG','median_volatile_lowG','volatile_lowG_25','volatile_lowG_75',
                                  'mean_volatile_highG','std_volatile_highG','median_volatile_highG','volatile_highG_25','volatile_highG_75',
                                  'volatile_t','volatile_p','volatile_dof',
                                  'n_total','n_lowG','n_highG','g_threshold', 'g_range_max','g_range_min',
                                  ],
                     'Values':[mean_stable_LowG,std_stable_LowG,median_stable_LowG,stable_LowG_IQI[0],stable_LowG_IQI[1],
                               mean_stable_HighG,std_stable_HighG,median_stable_HighG,stable_HighG_IQI[0],stable_HighG_IQI[1],
                               round(t_stable,2),round(p_stable,2),dof_stable,
                               mean_volatile_LowG,std_volatile_LowG,median_volatile_LowG,volatile_LowG_IQI[0],volatile_LowG_IQI[1],
                               mean_volatile_HighG,std_volatile_HighG,median_volatile_HighG,volatile_HighG_IQI[0],volatile_HighG_IQI[1],
                               round(t_volatile,2),round(p_volatile,2),dof_volatile,
                               int(len(df_merged)),n_lowG,n_highG,round(mean_val,2), round(np.max(df_merged['g']),2), round(np.min(df_merged['g']),2)]
                     }

df_stats = pd.DataFrame(stats_PCorrect,dtype=int)
df_stats.set_index('Statistic', inplace=False)

print(df_stats)