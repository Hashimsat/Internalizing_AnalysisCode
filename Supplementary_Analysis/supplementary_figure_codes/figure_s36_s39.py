# Figure s36, s39: Effect of the anxiety-related factor F1 and the depression-related factor F2 on learning-rate (
# Models 12 and 13) components

import sys
# path to model_code directory
sys.path.append("../../reversal_task_model/")

import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from functions.util_functions import cm2inch, medianprops, label_subplots
from functions.prl_plotting_functions import (plot_factor_errorbar,
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
    fig_num = 's36'
elif model_number == 13:
    path = "data/reversal_task/prt_rewardloss_priorstd1_meanstd10_targetaccept0.95_model=13_covariate=Bi3itemCDM_date=2025_1_26_samples=3000_tuning1000_seed=123_exp=3.pkl"
    fig_num = 's39'

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
fig = plt.figure(figsize=cm2inch(fig_width, fig_height), dpi=500)
fig.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(1, 2, wspace=0.7, hspace=0.6, top=0.85, bottom=0.6, left=0.12, right=0.99)

# ------------------
# Plot F1 and learning rate association
# ----------------
ax0 = plt.subplot(gs_0[0, 0])
fig.add_subplot(ax0)

plot_factor_errorbar(trace=trace, params=model.params,
                         ax=ax0,
                         factor='u_PC2',
                         ylabel='Effect of F1 ($β_1$) \n on learning rate components \n(in logit space)',
                         legendlabel='posterior mean \n (with 95% HDI)',
                         legendloc='upper left',
                         taskVersion='rewardLoss',
                         fontsize=7,
                         rotation=90,
                         legend=True,
                         color='black',
                         elinewidth=1,
                         legend_anchor=[0.5, 1.1])

# ------------------
# Plot F2 and learning rates association
# ----------------
ax1 = plt.subplot(gs_0[0, 1])
fig.add_subplot(ax1)

plot_factor_errorbar(trace=trace, params=model.params,
                         ax=ax1,
                         factor='u_PC3',
                         ylabel='Effect of F2 ($β_2$) \n on learning rate components \n(in logit space)',
                         legendlabel='posterior mean \n (with 95% HDI)',
                         legendloc='upper left',
                         taskVersion='rewardLoss',
                         fontsize=7,
                         rotation=90,
                         legend=False,
                         color='black',
                         elinewidth=1,
                         legend_anchor=[0.5, 1.1])

# ---------
# Add labels and save figure
# ----------
# Despine figure
plt.tight_layout()
sns.despine(fig)

# Add labels
texts = ['a','b']
label_subplots(fig, texts, x_offset=0.07, y_offset=0.08,fontsize=fontsize)

# Save figure
name=f'figure_{fig_num}.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=600, transparent=False, bbox_inches='tight')
plt.show()

# ------- Extract data ---------
df_F1 = extract_distribution_mean_hdpis(trace=trace, model=model, factor='u_PC2', param='lr')
df_F1.name = 'Suppl_reversal_learning_with_rewardLoss_model_F1_params_lr'

df_F2 = extract_distribution_mean_hdpis(trace=trace, model=model, factor='u_PC3', param='lr')
df_F2.name = 'Suppl_reversal_learning_with_rewardLoss_model_F2_params_lr'
