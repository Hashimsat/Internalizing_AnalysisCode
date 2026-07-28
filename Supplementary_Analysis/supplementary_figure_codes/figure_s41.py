
# Figure s41: Effect of internalizing on learning rate parameters, using model 12 and loadings from Gagne et al. (2019)

import sys
# path to model_code directory
sys.path.append("../../reversal_task_model/")

import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from functions.util_functions import cm2inch, medianprops
from functions.prl_plotting_functions import (plot_factor_errorbar, plot_param_separated_by_domain,
                                              extract_distribution_mean_hdpis, label_panel, add_legend, despine)

# ----------------
# 1. Setup Data Paths
# ----------------

target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

# Initialize which model to load
path = 'data/reversal_task/prl_rewardloss_model=12_gagneLoadings_priorstd1_meanstd10_covariate=Bi3itemCDM_date=2026_2_25_samples=3000_seed=3_exp=3.pkl'

model_path = os.path.join(base_dir, path)

actual_data_path = os.path.join(base_dir, 'data/reversal_task/prl_rewardloss_data_model_alligned_gagneLoadings.pkl')

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
fig_width = 16.51
fig_height = 10.16
fontsize = 7
medianprops = medianprops()
colors = ["#80cdc1",'#de77ae', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))

# Manually set axes
ax0 = plt.axes([0.05, 0.4, 0.2, 0.5])
ax1 = plt.axes([0.4, 0.65, 0.23, 0.23])
ax2 = plt.axes([0.4, 0.15, 0.23, 0.23])
ax3 = plt.axes([0.7, 0.65, 0.23, 0.23])
ax4 = plt.axes([0.7, 0.15, 0.23, 0.23])

# Create figure
fig = plt.figure(figsize=cm2inch(fig_width, fig_height), dpi=500)

# ------------------
# Plot group posterior distribution
# ----------------
plot_factor_errorbar(trace=trace, params=model.params,
                         ax=ax0,
                         factor='u_PC1',
                         ylabel='Effect of general factor ($β_g$) \n on learning rate components \n(in logit space)',
                         legendlabel='posterior mean \n (with 95% HDI)',
                         legendloc='upper left',
                         taskVersion='rewardLoss',
                         fontsize=7,
                         rotation=90,
                         legend=True,
                         color='black',
                         elinewidth=1,
                         legend_anchor=[0.5, 1.1])


# Plot learning rates in reward and aversive phases
for i, (ax, task, split) in enumerate(zip([ax1, ax2, ax3, ax4],
                                              ['reward', 'loss', 'reward', 'loss'],
                                              ['low', 'low', 'high', 'high'])):

    legend = True if i == 1 else False
    plot_param_separated_by_domain(trace, data, model,
                          param='lr',
                          pc='u_PC1',
                          ax=ax,
                          task=task,
                          median=False,
                          split=split,
                          transform='invlogit',
                          scatter_offset=0.05,
                          title=False,
                          legend=legend,
                          legendloc='upper right',
                          s=2,
                          rotation=90,
                          markersize=4, elinewidth=1.5,
                          include_errorbar=True,
                          ebar_offset=-0.05, fontsize=fontsize,
                          legend_anchor=[1, -0.55])

# ----------
# Add labels and align legends
# -----------
# --- Panel labels ---
label_panel(ax0, 'a', -0.45, 1.1, fontsize)
label_panel(ax1, 'b', -0.3, 1.3, fontsize)
label_panel(ax2, 'c', -0.3, 1.3, fontsize)

# --- Task labels ---
ax1.text(0.82, 1.15, 'Reward domain', fontsize=fontsize, transform=ax1.transAxes)
ax2.text(0.82, 1.15, 'Loss domain', fontsize=fontsize, transform=ax2.transAxes)

# --- Legends ---
add_legend(ax0, loc='lower center', ncol=1, bbox_to_anchor=(0.5, -1.02), fontsize=fontsize-1)
add_legend(ax2, loc='lower center', ncol=1, bbox_to_anchor=(0.3, -1.2), fontsize=fontsize-1)
add_legend(ax4, loc='lower center', ncol=1, bbox_to_anchor=(0.45, -1.2), fontsize=fontsize-1)

# --- Despine ---
despine(ax0, ax1, ax2, ax3, ax4)

# ----- Save Figure ----
name='figure_s41.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=600, transparent=False, bbox_inches='tight')
plt.show()

# ------ Extract Data ------
# extract data and save in dropbox
df_gf = extract_distribution_mean_hdpis(trace=trace, model=model, factor='u_PC1', param='lr')
df_gf.name = 'Suppl_reversal_learning_with_rewardLoss_model_GF_params_lr'
