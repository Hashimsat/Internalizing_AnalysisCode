# Plot example of fixed and adaptive LRs, along with the model computations
# Environment: predator_task_env

import numpy as np
import pandas as pd
from scipy.stats import zscore
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
from functions.util_functions import cm2inch, label_subplots,medianprops, qns_factor_preprocessing, compute_median_iqr
from scipy import stats
from functions.plotting_functions import plot_x_vs_y_FactorScores_robust, plot_boxplots

# -----------------
# 1. Load data
# -----------------
figure_folder = '../figures/'

ml_data = pd.read_csv("../data/predator_task/df_predator_4exp_modelresults.csv")
df_fdr = pd.read_csv("../data/predator_task/predator_AllModelParams_internalizingRegression_FDR.csv")

qns_totalscore = pd.read_csv('../data/factor_analysis/questionnaires_totalscores_subscales.csv')
factor_scores = pd.read_csv('../data/factor_analysis/factor_scores.csv')

# Data for example participant figure
df_example = pd.read_csv("../data/predator_task/example_participant_data.csv")
df_normative = pd.read_csv("../data/predator_task/simulated_normative_learning_fig5.csv")


# -----------------
# 2. Preprocess data
# -----------------
df_qns, df_fs, df_merged = qns_factor_preprocessing(qns_totalscore, factor_scores)

# merge with model data
ml_data = ml_data.merge(df_merged,on='subjectID')

# Separate data by low and high factor scores
# zscore g, F1 and F2
ml_data['int'] = 1
ml_data['g_z'] = zscore(ml_data['g'])
ml_data['F1_z'] = zscore(ml_data['F1.'])
ml_data['F2_z'] = zscore(ml_data['F2.'])
ml_data['Age_z'] = zscore(ml_data['Age'])

# Split data into high and low general factor
g_threshold = np.nanmean(ml_data['g_z'])
ml_data['G_Category'] = pd.np.where(ml_data['g_z'] > g_threshold, 'High',
                                pd.np.where(ml_data['g_z'] < g_threshold, 'Low', 'Normal'))

nLow = len(ml_data[ml_data['G_Category'] == 'Low'])
nHigh = len(ml_data[ml_data['G_Category'] == 'High'])

# -------------------
# 3. Set up the figure
# -------------------
# Size of figure

fig_width = 15
fig_height = 10
fontsize = 7
medianprops = medianprops()

colors = ['#77AADD','#009988',"#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.6, top=0.85, bottom=0.1, left=0.12, right=0.99)

# -----------------
# 4. Plot the data
# -----------------

# Plot 1 - Example Participant LR
# Plot example of what adaptive LR looks like

gs_01 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[0:2,0:1])
ax0 = plt.Subplot(f, gs_01[0, 0])
f.add_subplot(ax0)

ylim = [-100,101]
xlim= [-100,100]

# Create a boolean mask for rows within the range -100 to 100 for both 'update' and 'PE'
mask = (df_example['Update'].between(-100, 100)) & (df_example['delta_t'].between(-100, 100))

# Use the mask to filter rows
filtered_example = df_example[mask]

ax0.plot(ylim, xlim, color='k', linewidth=1, linestyle='--')

ax0.hlines(0,xmin=xlim[0], xmax=xlim[1], color='k', linewidth=1, linestyle='--')
ax0.hlines(0,xmin=xlim[0], xmax=xlim[1], color='k', linewidth=1, linestyle='--')
sns.scatterplot(x='delta_t', y='Update',data=filtered_example, alpha=0.7,
                color='#808080',edgecolor=None,s=5, ax=ax0, zorder=2)

slope = pd.unique(filtered_example['fixed_LR'])
intercept = pd.unique(filtered_example['fixed_LR_int'])

ax0.axline((0, 0), slope=slope[0], color=colors[0], linewidth=3,alpha=0.9, label='Fixed LR')

# Plot adaptive learning curve
ax0.plot(df_normative['PE'],df_normative['PE']*df_normative['alpha_i'],'-', color=colors[1],linewidth = 3,alpha = 0.85,label='Adaptive LR')

ax0.set_ylim(ylim)
ax0.set_xlim(xlim)
ax0.set_yticks(range(ylim[0], ylim[1], 50))
ax0.set_xticks(range(xlim[0], xlim[1], 50))
ax0.set_xlabel('Prediction Error', fontsize=fontsize,labelpad=0.5)
ax0.set_ylabel('Update', fontsize=fontsize)


ax0.xaxis.set_tick_params(labelsize=fontsize)
ax0.yaxis.set_tick_params(labelsize=fontsize)

title = "Update = β0 + β1·PE + β2·⍺·PE + ..."
ax0.set_title(title, fontsize=fontsize, pad=10)

ax0.annotate('Fixed \nLR', xy=(0.51, 1.09), xytext=(0.51, 1.12), xycoords='axes fraction',
            fontsize=fontsize,color=colors[0],weight='bold', ha='center', va='bottom',annotation_clip=False,
            arrowprops=dict(arrowstyle='-[, widthB=1.47, lengthB=0.28', lw=0.5, color='k'))

ax0.annotate('Adaptive \nLR', xy=(0.71, 1.09), xytext=(0.71, 1.12), xycoords='axes fraction',
            fontsize=fontsize,color=colors[1],weight='bold', ha='center', va='bottom',annotation_clip=False,
            arrowprops=dict(arrowstyle='-[, widthB=2.0, lengthB=0.28', lw=0.5, color='k'))

ax0.legend(fontsize=fontsize, handlelength=1, loc=4)

# Plot boxplots showing the distribution of fixed learning rates for low and high internalizing

gs_02 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[0:1,1:2], wspace=0.65)
ax1 = plt.Subplot(f, gs_02[0, 0])
f.add_subplot(ax1)

colors2 = ["#80cdc1",'#de77ae', "#dfc27d", '#77AADD',"#018571",]
sns.set_palette(sns.color_palette(colors2))

plot_boxplots(df=ml_data, x='G_Category', y='beta_1', ax=ax1,
             order=['Low', 'High'], ylabel='Fixed LR', xlabel='General Factor', fontsize=fontsize)

res_b1 = stats.ttest_ind(ml_data[ml_data['G_Category'] == 'Low']['beta_1'].astype('float'),
                      ml_data[ml_data['G_Category'] == 'High']['beta_1'].astype('float'), equal_var=False)

t_b1 = res_b1.statistic
p_b1 = res_b1.pvalue
dof_b1 = round(res_b1.df,2)

median_b1_Low, median_b1_Low_iqr = compute_median_iqr(ml_data[ml_data['G_Category'] == 'Low']['beta_1'])
median_b1_High, median_b1_High_iqr = compute_median_iqr(ml_data[ml_data['G_Category'] == 'High']['beta_1'])

title = "$\it{p}$ = " + str(round(p_b1, 2))
ax1.set_title(title, fontsize=fontsize)
ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
ax1.xaxis.set_tick_params(labelsize=fontsize)
ax1.yaxis.set_tick_params(labelsize=fontsize)

# Show regression of fixed LR vs internalizing
ax2 = plt.Subplot(f, gs_02[0, 1])
f.add_subplot(ax2)

r1_reg,p1_reg=plot_x_vs_y_FactorScores_robust(ml_data,'g','beta_1',ax2,title=False,fontsize=fontsize,color_index=-2,line_color_index=-1,xlabel='General Factor',ylabel='Fixed LR')

title_b1 = "$\it{r}$ = " + str(np.round(df_fdr[df_fdr['Statistic']=='r_b1_g']['Values'].values[0], 2)) + ", $\it{p}$ = " + str(np.round(df_fdr[df_fdr['Statistic']=='p_b1_g']['Values'].values[0], 2))
ax2.set_title(title_b1, fontsize=fontsize)

# Plot boxplots showing the distribution of adaptive learning rates for low and high internalizing
gs_03 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_0[1:2,1:2], wspace = 0.65)
ax3 = plt.Subplot(f, gs_03[0, 0])
f.add_subplot(ax3)

colors = ["#80cdc1",'#de77ae', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))

plot_boxplots(df=ml_data, x='G_Category', y='beta_4', ax=ax3,
             order=['Low', 'High'], ylabel='Adaptive LR', xlabel='General Factor', fontsize=fontsize)

res_b4 = stats.ttest_ind(ml_data[ml_data['G_Category'] == 'Low']['beta_4'].astype('float'),
                      ml_data[ml_data['G_Category'] == 'High']['beta_4'].astype('float'), equal_var=False)
t_b4 = res_b4.statistic
p_b4 = res_b4.pvalue
dof_b4 = round(res_b4.df,2)

median_b4_Low, median_b4_Low_iqr = compute_median_iqr(ml_data[ml_data['G_Category'] == 'Low']['beta_4'])
median_b4_High, median_b4_High_iqr = compute_median_iqr(ml_data[ml_data['G_Category'] == 'High']['beta_4'])

title = "$\it{p}$ = " + str(round(p_b4, 2))
ax3.set_title(title, fontsize=fontsize)
ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
ax3.xaxis.set_tick_params(labelsize=fontsize)
ax3.yaxis.set_tick_params(labelsize=fontsize)

# Show regression of adaptive LR vs internalizing
ax4 = plt.Subplot(f, gs_03[0, 1])
f.add_subplot(ax4)


r4_reg,p4_reg=plot_x_vs_y_FactorScores_robust(ml_data,'g','beta_4',ax4,title=False,fontsize=fontsize,color_index=-2,line_color_index=-1,xlabel='General Factor',ylabel='Adaptive LR')

title_b4 = "$\it{r}$ = " + str(np.round(df_fdr[df_fdr['Statistic']=='r_b4_g']['Values'].values[0], 2)) + ", $\it{p}$ = " + str(np.round(df_fdr[df_fdr['Statistic']=='p_b4_g']['Values'].values[0], 2))
ax4.set_title(title_b4, fontsize=fontsize)

plt.tight_layout()
sns.despine(f)

# Label subplots
texts = ['a','b','c','d', 'e']
label_subplots(f, texts, x_offset=0.07, y_offset=0.06,fontsize=fontsize)

# -----------------
# 5. Save figure
# -----------------

name='figure_5_predator_model.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format="pdf", dpi=700, transparent=True)
plt.show()

# -----------------
# 6. Make dataframe of stats for low and high internalizing boxplots
# -----------------

stats = {
    'Statistic':['t_b1','p_b1','n_Low','n_High','dof_b1','median_b1_Low','median_b1_Low_IQIlow','median_b1_Low_IQIhigh', 'median_b1_High','median_b1_High_IQIlow','median_b1_High_IQIhigh',
                 't_b4','p_b4','dof_b4','median_b4_Low','median_b4_Low_IQIlow','median_b4_Low_IQIhigh', 'median_b4_High','median_b4_High_IQIlow','median_b4_High_IQIhigh'],

    'Values':[t_b1,round(p_b1,2),nLow,nHigh,dof_b1,median_b1_Low,median_b1_Low_iqr[0],median_b1_Low_iqr[1],median_b1_High,median_b1_High_iqr[0],median_b1_High_iqr[1],
              t_b4,round(p_b4,2),dof_b4,median_b4_Low,median_b4_Low_iqr[0],median_b4_Low_iqr[1],median_b4_High,median_b4_High_iqr[0],median_b4_High_iqr[1]]
}

df_stats = pd.DataFrame(stats)
df_stats.set_index('Statistic', inplace=False)
df_stats['Values'] = df_stats['Values'].round(2)
df_stats.name = 'ModelParams_predator_LowHighG_tstat'

print(df_stats)