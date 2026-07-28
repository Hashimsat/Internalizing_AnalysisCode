# Plot the effect of G, F1 and F2 on the inverse temperature parameter(ß) for PRL RewardMag Task

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functions.util_functions import cm2inch, label_subplots
import seaborn as sns
import pickle
from functions.prl_plotting_functions import (plot_param_posterior_distribution_onesubplot,
                                              param_by_factor_score, plot_factor_errorbar)

# ----------------
# 1. Setup Data Paths
# ----------------

target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

pickle_filepath = base_dir + '/data/reversal_task/prl_rewardmag_priorstd1_model=9_covariate=Bi3itemCDM_date=2025_1_17_samples=2000_seed=3_exp=3.pkl'


actual_data_path = os.path.join(base_dir, 'data/reversal_task/prl_rewardmag_data_model_alligned.pkl')


# -------------------
# 2. Preprocess data
# -------------------
# load model and model trace from provided file path
with open(pickle_filepath , 'rb') as buff:
    model_dict = pickle.load(buff)

trace = model_dict['trace']
model = model_dict['model']

# Load actual data
with open(actual_data_path, 'rb') as f:
    data = pickle.load(f)

# ----------------
# 3. Setup Figure
# ----------------
fig_width = 15
fig_height = 13
fontsize = 7
medianprops = dict(linestyle='-', linewidth=1, color='k')
colors = ["#80cdc1",'#de77ae', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))


# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()


# Create plot grid
gs_0 = gridspec.GridSpec(2, 2, wspace=0.65, hspace=0.9, top=0.9, bottom=0.28, left=0.23, right=0.95)

# ------------------
# 4. Plot group posterior distribution
# ----------------
ax0 = f.add_subplot(gs_0[0, 0])
f.add_subplot(ax0)

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
                                             legend=False,
                                             legendlabel='posterior mean (with 95% HDI)',
                                             ylabel='Group mean ($μ_o$) for \n learning rate components \n(in logit space)',
                                             s_bar=1,
                                             elinewidth=1,
                                             ebar_offset=-0.05,
                                             legend_anchor=[0.6, 0.02],
                                             boxplot=False)


# ------------------
# 5. Plot effect of internalizing factor (G)
# ----------------
ax1 = f.add_subplot(gs_0[0, 1])
f.add_subplot(ax1)

plot_factor_errorbar(trace=trace, params=model.params,
                         ax=ax1,
                         param='lr',
                         factor='u_PC1',
                         ylabel='Effect of general factor ($β_g$) \n on learning rate components \n(in logit space)',
                         legendlabel='posterior mean \n (with 95% HDI)',
                         legendloc='best',
                         xlabel=True,
                         fontsize=7,
                         legend=False,
                         color='black',
                         elinewidth=1,
                         legend_anchor=[0.5, 1.1])


# ------------------
# 6. Separate Internalizing effects into low and high groups
# ----------------
# setup plot
ax4_1 = plt.Subplot(f, gs_0[1, 0])
ax4_2= plt.Subplot(f, gs_0[1, 1])
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
                          legend_anchor=[0.8, -0.55])


# -----------
# 7. Add labels and save figure
# -----------
# Add labels to subplots
texts = ['a', 'b', 'c', 'd']
label_subplots(f, texts, x_offset=0.08, y_offset=0.05)
sns.despine(f)

# save figure
name='figure_s28.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()

