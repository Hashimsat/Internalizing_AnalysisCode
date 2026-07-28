# Figure s37: The inverse temperature parameter

import sys
sys.path.append("../../reversal_task_model/")    #path to model_code directory

import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from functions.util_functions import cm2inch, medianprops, label_subplots
from functions.prl_plotting_functions import (plot_param_posterior_distribution_onesubplot, plot_factor_errorbar,
                                              extract_distribution_mean_hdpis)

# ----------------
# 1. Setup Data Paths
# ----------------

target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

# Initialize which model to load
model_number = 13

if model_number == 12:
    path = 'data/reversal_task/prl_rewardloss_priorstd1_meanstd10_targetaccept0.99_model=12_covariate=Bi3itemCDM_date=2025_1_24_samples=3000_tuning1000_seed=123_exp=3.pkl'
    fig_num = 's37'
elif model_number == 13:
    path = "data/reversal_task/prt_rewardloss_priorstd1_meanstd10_targetaccept0.95_model=13_covariate=Bi3itemCDM_date=2025_1_26_samples=3000_tuning1000_seed=123_exp=3.pkl"
    fig_num = 's40'

model_path = os.path.join(base_dir, path)
actual_data_path = os.path.join(base_dir, 'data/reversal_task/prl_rewardloss_data_model_alligned.pkl')

# ----------------
# 2. Load Model and Actual Data
# ----------------

with open(model_path, 'rb') as f:
    model_dict = pickle.load(f)

trace = model_dict['trace']
model = model_dict['model']

# Load actual data
with open(actual_data_path, 'rb') as f:
    data = pickle.load(f)

# ----------------
# Setup Figure
# ----------------
fig_width = 13
fig_height = 10
fontsize = 7
medianprops = medianprops()
colors = ["#80cdc1",'#de77ae', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(2, 2, wspace=0.7, hspace=0.05, top=0.85, bottom=0.1, left=0.12, right=0.99)

# ------------------
#  Plot inverse-temperature group parameters
# ----------------
ax0 = plt.subplot(gs_0[0, 0])
f.add_subplot(ax0)

pal_dark = ['#77AADD','#77AADD','#77AADD','#77AADD']
plot_param_posterior_distribution_onesubplot(trace=trace,  # data
                                             params=model.params,  # model parameter names
                                             gp='u',  # group parameter
                                             param='Binv',  # readable name
                                             offset=-0.15,
                                             fontsize=fontsize,
                                             bp_width=0.2,
                                             ax=ax0,  # plot characteristics
                                             colors=pal_dark,
                                             legend=True,
                                             xlabel=False,
                                             legendlabel='posterior mean (with 95% HDI)',
                                             legendloc='upper right',
                                             ylabel='Group mean ($μ_o$) for \n B components \n(in logarithmic space)',
                                             rotation=90,
                                             s_bar=1,
                                             elinewidth=1,
                                             ebar_offset=-0.05,
                                             legend_anchor=[1, -0.6],
                                             boxplot=False)

# ------------------
# Plot inverse-temperature for G
# ----------------
ax1 = plt.subplot(gs_0[0, 1])
f.add_subplot(ax1)

plot_factor_errorbar(trace=trace, params=model.params,
                         ax=ax1,
                         factor='u_PC1',
                         param='Binv',
                         ylabel='Effect of general factor ($β_g$) \n on B components \n(in logarithmic space)',
                         legendlabel='posterior mean \n (with 95% HDI)',
                         legendloc='upper left',
                         taskVersion='rewardLoss',
                         xlabel=False,
                         fontsize=fontsize,
                         rotation=90,
                         legend=False,
                         color='black',
                         elinewidth=1,
                         legend_anchor=[0.5, 1.1])


# ------------------
# Plot inverse-temperature for F1
# ----------------
ax2 = plt.subplot(gs_0[1, 0])
f.add_subplot(ax2)

plot_factor_errorbar(trace=trace, params=model.params,
                         ax=ax2,
                         factor='u_PC2',
                         param='Binv',
                         ylabel='Effect of F1 ($β_1$) \n on B components \n(in logarithmic space)',
                         legendlabel='posterior mean \n (with 95% HDI)',
                         legendloc='upper left',
                         taskVersion='rewardLoss',
                         fontsize=fontsize,
                         rotation=90,
                         legend=False,
                         color='black',
                         elinewidth=1,
                         legend_anchor=[0.5, 1.1])

# ------------------
# Plot inverse-temperature for F2
# ----------------
ax3 = plt.subplot(gs_0[1, 1])
f.add_subplot(ax3)

plot_factor_errorbar(trace=trace, params=model.params,
                         ax=ax3,
                         factor='u_PC3',
                         param='Binv',
                         ylabel='Effect of F2 ($β_1$) \n on B components \n(in logarithmic space)',
                         legendlabel='posterior mean \n (with 95% HDI)',
                         legendloc='upper left',
                         taskVersion='rewardLoss',
                         fontsize=fontsize,
                         rotation=90,
                         legend=True,
                         color='black',
                         elinewidth=1,
                         legend_anchor=[-0.58, -0.3])


# ---------
# Add labels and save figure
# ----------
# Despine figure
plt.tight_layout()
sns.despine(f)

# Add labels
texts = ['a','b', 'c','d']
label_subplots(f, texts, x_offset=0.09, y_offset=0.0,fontsize=fontsize)

# Save figure
name = f'figure_{fig_num}.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=600, transparent=False, bbox_inches='tight')
plt.show()

# ------- Extract data ---------
df_group = extract_distribution_mean_hdpis(trace=trace, model=model, factor='u', param='Binv')
# add 'group' to every param item
df_group['param'] = df_group['param'].apply(lambda x: x + '_group')

df_g = extract_distribution_mean_hdpis(trace=trace, model=model, factor='u_PC1', param='Binv')
df_g['param'] = df_g['param'].apply(lambda x: x + '_GF')

df_F1 = extract_distribution_mean_hdpis(trace=trace, model=model, factor='u_PC2', param='Binv')
df_F1['param'] = df_F1['param'].apply(lambda x: x + '_F1')

df_F2 = extract_distribution_mean_hdpis(trace=trace, model=model, factor='u_PC3', param='Binv')
df_F2['param'] = df_F2['param'].apply(lambda x: x + '_F2')

# combine all dataframes
df = pd.concat([df_group, df_g, df_F1, df_F2], axis=0)
df.name = 'Suppl_reversal_learning_with_rewardLoss_model_params_Binv_Model12_GagneLoadings'