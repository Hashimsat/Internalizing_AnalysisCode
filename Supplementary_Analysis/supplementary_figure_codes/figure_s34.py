# Figure s34: Group-level learning rate parameter


import sys
# path to model_code directory
sys.path.append("../../reversal_task_model/")

import pickle
import os
import matplotlib.pyplot as plt
from functions.util_functions import cm2inch, medianprops
from functions.prl_plotting_functions import (plot_param_posterior_distribution_onesubplot,
                                              plot_param_separated_by_domain, extract_distribution_mean_hdpis,
                                              label_panel, add_legend, despine)

# ----------------
# 1. Setup Data Paths
# ----------------

target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

model_path = os.path.join(base_dir, 'data/reversal_task/'
                                    '/prl_rewardloss_priorstd1_meanstd10_targetaccept0.99_model=12_covariate'
                                    '=Bi3itemCDM_date=2025_1_24_samples=3000_tuning1000_seed=123_exp=3.pkl')

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
fig_width = 16.51
fig_height = 6.35
fontsize = 7
medianprops = medianprops()

# Create figure
fig = plt.figure(figsize=cm2inch(fig_width, fig_height), dpi=500)

# manually setting up the axes
ax0 = plt.axes([0.05, 0.3, 0.2, 0.6])
ax1 = plt.axes([0.4, 0.3, 0.23, 0.6])
ax2 = plt.axes([0.7, 0.3, 0.23, 0.6])
axes = [ax0, ax1, ax2]

# ------------------
# Plot group posterior distribution
# ----------------

pal_dark = ['#77AADD','#77AADD','#77AADD','#77AADD']
plot_param_posterior_distribution_onesubplot(trace=trace,  # data
                                             params=model.params,  # model parameter names
                                             gp='u',  # group parameter
                                             param='lr',  # readable name
                                             offset=-0.15,
                                             fontsize=fontsize,
                                             bp_width=0.2,
                                             ax=ax0,  # plot characteristics
                                             colors=pal_dark,
                                             taskVersion='rewardLoss',
                                             legend=True,
                                             legendlabel='posterior mean (with 95% HDI)',
                                             legendloc='upper right',
                                             ylabel='Group mean ($μ_o$) for \n learning rate components \n(in logit space)',
                                             rotation=90,
                                             s_bar=1,
                                             elinewidth=1,
                                             ebar_offset=-0.05,
                                             legend_anchor=[1, -0.6],
                                             boxplot=False)


# Plot learning rates in reward and loss domains
for i, (ax, task) in enumerate(zip([ax1, ax2],
                                    ['reward', 'loss', ])):

    legend = True if i == 1 else False
    plot_param_separated_by_domain(trace, data, model,
                          param='lr',
                          pc='u_PC1',
                          ax=ax,
                          task=task,
                          median=False,
                          split='mean',
                          transform='invlogit',
                          scatter_offset=0.05,
                          title=True,
                          legend=legend,
                          legendloc='upper right',
                          s=2,
                          rotation=90,
                          markersize=4, elinewidth=1.5,
                          include_errorbar=True,
                          ebar_offset=-0.05, fontsize=fontsize,
                          legend_anchor=[1, -0.55])

# -----------------
# Add labels and adjust legends
# -----------------
# --- Panel labels ---
label_panel(ax0, 'a', -0.45, 1.1, fontsize)
label_panel(ax1, 'b', -0.3, 1.1, fontsize)

# --- Legends ---
add_legend(ax0, loc='lower center', ncol=1, bbox_to_anchor=(0.15, -1.12), fontsize=fontsize-1)
add_legend(ax1, loc='lower center', ncol=1, bbox_to_anchor=(0.45, -1.05), fontsize=fontsize-1)
add_legend(ax2, loc='lower center', ncol=1, bbox_to_anchor=(0.45, -1.05), fontsize=fontsize-1)

# --- Despine ---
despine(ax0, ax1, ax2)

# ------------
# Save figure
# ------------
name='figure_s34.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=600, transparent=False, bbox_inches='tight')
plt.show()


# ------------
# Extract data
# -----------
df_group = extract_distribution_mean_hdpis(trace=trace, model=model, factor='u', param='lr')
df_group.name = 'Suppl_reversal_learning_with_rewardLoss_model_Group_params_lr'