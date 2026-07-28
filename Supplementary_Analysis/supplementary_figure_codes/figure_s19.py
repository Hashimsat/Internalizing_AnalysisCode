# Plot effects of F1 and F2 on model based parameters for the probabilistic reversal learning task without magnitudes

import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from functions.util_functions import cm2inch, medianprops, label_subplots

from functions.prl_plotting_functions import plot_factor_errorbar

# ----------------
# 1. Setup Data Paths
# ----------------

target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

pickle_filepath = base_dir + '/data/reversal_task/prl_nomag_model6_covariate=Bi3itemCDM_date=2025_1_14_samples=2500tune=1200_seed=3_exp=3.pkl'

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
fig_width = 14
fig_height = 8
fontsize = 7
medianprops = medianprops()

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(1, 2, wspace=0.5, hspace=0.7, top=0.95, bottom=0.28, left=0.2, right=0.95)

# ------------------
# 4. Plot F1 and F2
# ----------------
# Plot F1
ax1 = f.add_subplot(gs_0[0, 0])
f.add_subplot(ax1)

plot_factor_errorbar(trace=trace, params=model.params,
                         ax=ax1,
                         factor='u_PC2',
                         ylabel='Effect of F1 ($β_1$) \n on learning rate components \n(in logit space)',
                         legendlabel='posterior mean \n (with 95% HDI)',
                         legendloc='best',
                         fontsize=7,
                         legend=False,
                         color='black',
                         elinewidth=1,
                         legend_anchor=[1, -0.55])

# Plot F2
ax2 = f.add_subplot(gs_0[0, 1])
f.add_subplot(ax2)

plot_factor_errorbar(trace=trace, params=model.params,
                         ax=ax2,
                         factor='u_PC3',
                         ylabel='Effect of F2 ($β_2$) \n on learning rate components \n(in logit space)',
                         legendlabel='posterior mean \n (with 95% HDI)',
                         legendloc='best',
                         fontsize=7,
                         legend=True,
                         color='black',
                         elinewidth=1,
                         legend_anchor=[0.1, -0.3])


# -----------
# 5. Add labels and save figure
# -----------
# add labels to subplots
texts = ['a', 'b']
label_subplots(f, texts, x_offset=0.08, y_offset=0.05)
sns.despine(f)

# save figure
name='figure_s19.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()