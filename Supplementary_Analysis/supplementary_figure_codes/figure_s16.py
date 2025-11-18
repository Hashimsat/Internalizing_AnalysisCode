# Plot parameter recovery for the winning model for the probabilistic reversal learning task

import pickle
import matplotlib.ticker as ticker
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats
from functions.util_functions import cm2inch, medianprops

# -----------------
# 1. Load Data
# -----------------
target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

param_rec_path = os.path.join(base_dir, 'reversal_task_model/fitting_behavioral_model/model_fits'
                                        '/ParamRec_prl_NoMagVersion_model=6try_one_task_True_covariate'
                                        '=Bi3itemCDM_date=2025_11_14_samples=2000_chains=2_seed=3_exp=1.pkl')


with open(param_rec_path, 'rb') as f:
    param_rec_model_dict = pickle.load(f)

# extract true and recovered parameters
trace = param_rec_model_dict['trace']
theta_gen = param_rec_model_dict['Theta_gen']
beta_independent = np.mean(trace['Theta_both'], axis=0)

# -----------------
# 2. Calculate Correlations Between beta_independent and Theta_gen
# -----------------
corr = [stats.spearmanr(beta_independent[:, i], theta_gen[:, i]).correlation for i in range(beta_independent.shape[1])]
print(corr)

# -----------------
# 3. Plot Parameter Recovery
# -----------------
# Set up figure
fig_width = 15
fig_height = 15
fontsize = 7
medianprops = medianprops()

colors = ["#80cdc1", '#de77ae', "#018571", "#dfc27d", '#d492c8', '#AA4499', '#808080', "#77AADD", "#3576b8"]

sns.set_palette(sns.color_palette(colors))

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(3, 4, wspace=0.65, hspace=0.9, top=0.87, bottom=0.1, left=0.15, right=0.98)

# Define parameter names
params_recovered = [r'$\alpha_{Baseline}$', r'$\alpha_{Good-bad}$', r'$\alpha_{Volatile-stable}$',
                    r'$\alpha_{(Good-bad)x(Volatile-stable)}$',
                    r'$\alpha_{ck}$',
                    r'$B_{Baseline}$', r'$B_{Good-bad}$', r'$B_{Volatile-stable}$',
                    r'$B_{(Good-bad)x(Volatile-stable)}$',
                    r'$B_{ck}$']

# Plot correlations between generated and recovered parameters
for i in range(len(params_recovered)):
    # Determine subplot position
    col_no = i % 4
    row_no = np.floor(i / 4).astype(int)

    # Create subplot
    ax = plt.Subplot(f, gs_0[row_no, col_no])
    f.add_subplot(ax)

    sns.regplot(x=theta_gen[:, i].astype('float'), y=beta_independent[:, i].astype('float'),  # color='#3576b8'
                color=colors[-1], robust=True, ax=ax,
                scatter_kws=dict(alpha=0.3, s=10, edgecolor="none", color=colors[-2]),
                line_kws=dict(linewidth=2))

    ax.set_xlabel('Ground Truth', fontsize=fontsize)
    ax.set_ylabel('Recovered', fontsize=fontsize)

    # set title with correlation
    title = params_recovered[i] + '\n' + "$Spearman \ \it{ρ}$ = " + str(round(corr[i], 2))
    ax.set_title(title, fontsize=fontsize)

    # have 3 ticks per axis
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

# De-spine the figure
sns.despine()

# save figure
name = 'figure_s16.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()
