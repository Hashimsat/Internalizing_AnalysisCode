# Fig_S30: Plot the effect of G, F1 and F2 on the inverse temperature parameter(ß) in PRL Task with Reward Mag

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functions.util_functions import cm2inch, label_subplots
import seaborn as sns
import pickle
from functions.prl_plotting_functions import plot_param_posterior_distribution_onesubplot, plot_factor_errorbar

# ----------------
# 1. Setup Data Paths
# ----------------

target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

pickle_filepath = base_dir + '/data/reversal_task/prl_rewardmag_priorstd1_model=9_covariate=Bi3itemCDM_date=2025_1_17_samples=2000_seed=3_exp=3.pkl'

# -------------------
# 2. Preprocess data
# -------------------
# load model and model trace from provided file path
with open(pickle_filepath , 'rb') as buff:
    model_dict = pickle.load(buff)

trace = model_dict['trace']
model = model_dict['model']


# ----------------
# 3. Setup Figure
# ----------------
fig_width = 15
fig_height = 13
fontsize = 7
medianprops = dict(linestyle='-', linewidth=1, color='k')


# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()
# f.canvas.tostring_argb()

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
                                             param='Binv',  # readable name
                                             offset=-0.15,
                                             fontsize=fontsize,
                                             bp_width=0.2,
                                             ax=ax0,  # plot characteristics
                                             colors=pal_dark,
                                             legend=False,
                                             legendlabel='posterior mean (with 95% HDI)',
                                             ylabel='Group mean ($μ_o$) for \n B components \n(in logarithmic space)',
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
                         param='Binv',
                         factor='u_PC1',
                         ylabel='Effect of general factor ($β_g$) \n on B components \n(in logarithmic space)',
                         legendlabel='posterior mean \n (with 95% HDI)',
                         legendloc='best',
                         xlabel=True,
                         fontsize=7,
                         legend=False,
                         color='black',
                         elinewidth=1,
                         legend_anchor=[1, -0.55])


# ------------------
# 6. Plot for anxiety-related factor F1
# ----------------
# setup plot
ax2 = f.add_subplot(gs_0[1, 0])
f.add_subplot(ax2)

plot_factor_errorbar(trace=trace, params=model.params,
                         ax=ax2,
                         param='Binv',
                         factor='u_PC2',
                         ylabel='Effect of F1 ($β_1$) \n on B components \n(in logarithmic space)',
                         legendlabel='posterior mean \n (with 95% HDI)',
                         xlabel=True,
                         legendloc='lower left',
                         fontsize=7,
                         legend=True,
                         color='black',
                         elinewidth=1,
                         legend_anchor=[1, -0.85])


# --------------------
# 6. Plot for depression-related factor F2
# --------------------
ax3 = f.add_subplot(gs_0[1, 1])
f.add_subplot(ax3)

plot_factor_errorbar(trace=trace, params=model.params,
                         ax=ax3,
                         param='Binv',
                         factor='u_PC3',
                         ylabel='Effect of F2 ($β_2$) \n on B components \n(in logarithmic space)',
                         legendlabel='posterior mean \n (with 95% HDI)',
                         legendloc='lower left',
                         fontsize=7,
                         legend=False,
                         color='black',
                         elinewidth=1,
                         legend_anchor=[1, 1])


# -----------
# 7. Add labels and save figure
# -----------
# Add labels to subplots
texts = ['a', 'b', 'c', 'd']
label_subplots(f, texts, x_offset=0.08, y_offset=0.05)
sns.despine(f)

# save figure
name='figure_s30.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()

