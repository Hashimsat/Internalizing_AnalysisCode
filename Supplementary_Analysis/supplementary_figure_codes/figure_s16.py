# Plot parameter recovery for the winning model for the probabilistic reversal learning task

import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from functions.prl_plotting_functions import plot_param_rec

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
# 2. Plot Parameter Recovery
# -----------------
# Set up figure
fig_width = 15
fig_height = 15
fontsize = 7

params_recovered = [r'$\alpha_{Baseline}$', r'$\alpha_{Good-bad}$', r'$\alpha_{Volatile-stable}$',
                    r'$\alpha_{(Good-bad)x(Volatile-stable)}$',
                    r'$\alpha_{ck}$',
                    r'$B_{Baseline}$', r'$B_{Good-bad}$', r'$B_{Volatile-stable}$',
                    r'$B_{(Good-bad)x(Volatile-stable)}$',
                    r'$B_{ck}$']


f = plot_param_rec(params_recovered, theta_gen, beta_independent, fig_width=fig_width, fig_height=fig_height,
               fontsize=fontsize, n_rows=3)

# save figure
name = 'figure_s16.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()
