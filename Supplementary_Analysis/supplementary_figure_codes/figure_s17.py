# Posterior-predictive checks for the winning model of the reversal learning task

import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from functions.util_functions import cm2inch, medianprops, label_subplots
from functions.prl_descriptive_functions import calculate_switches_PPC, calculate_p_correct_PPC
from functions.plotting_functions import create_subplots
from functions.prl_plotting_functions import plot_ppc, PPC_ax_setup

# ----------------
# 1. Setup Data Paths
# ----------------

target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_dir = os.path.dirname(target_dir)
figure_folder = target_dir + "/supplementary_figures"

model_path = os.path.join(base_dir, 'data/reversal_task/'
                                    '/prl_nomag_model6_covariate=Bi3itemCDM_date=2025_1_14_samples=2500tune'
                                    '=1200_seed=3_exp=3.pkl')

actual_data_path = os.path.join(base_dir, 'data/reversal_task/prl_nomag_data_model_alligned.pkl')

# ----------------
# 2. Load Model and Actual Data
# ----------------

with open(model_path, 'rb') as f:
    model_dict = pickle.load(f)

trace = model_dict['trace']
model = model_dict['model']
ppc = model_dict['ppc']
ppc_samples = ppc['observed_choice']

# Load actual data
with open(actual_data_path, 'rb') as f:
    actual_data_dict = pickle.load(f)

actual_choices = actual_data_dict['participants_choice']
outcome = actual_data_dict['outcomes_c_flipped']
stabvol = actual_data_dict['stabvol']
dominant_fractal = actual_data_dict['dominant_fractal']
subjects = actual_data_dict['subjectID']

# ----------------
# 3. Calculate Switch Rates and P(Correct)
# ----------------
df_switch = pd.DataFrame()
df_p_correct = pd.DataFrame()
p_corr_combined_list = []

for i in range(len(subjects)):
    ppc_subj = np.transpose(ppc_samples[:, :, i])
    stabvol_subj = stabvol[:, i]

    # Create a dataframe for the current subject
    df = pd.DataFrame(ppc_subj)
    df['stabvol'] = stabvol_subj
    df['dominant_fractal'] = dominant_fractal[:, i]
    df['outcome'] = outcome[:, i]
    df['observed'] = actual_choices[:, i]

    # Calculate switches
    switch_stats = calculate_switches_PPC(df, observed_col='observed')
    df_subj = pd.DataFrame([switch_stats])

    # Concatenate switch stats
    df_switch = pd.concat([df_switch, df_subj], axis=0)

    # Calculate P(Correct)
    df_subj_perf, p_correct_ppc = calculate_p_correct_PPC(df, dominant_col='dominant_fractal', observed_col='observed')

    # Concatenate P(Correct) stats
    df_p_correct = pd.concat([df_p_correct, df_subj_perf], axis=0)
    p_corr_combined_list.append(p_correct_ppc)

# Combine P(Correct) arrays
p_corr_combined_arr = np.vstack(p_corr_combined_list)

# ------------
# 4. Figure Setup
# ------------
fig_width = 10
fig_height = 14
fontsize = 7
medianprops = medianprops()

sns.set_palette("tab10")

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(2, 2, wspace=0.5, hspace=0.5, top=0.98, bottom=0.1, left=0.15, right=0.96)

# create subplots
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
axes = create_subplots(f, gs_0, positions)

# ---------------
# 5. Plot PPC
# ---------------

# Plot number of switches in stable block (PPC vs actual data)
plot_ppc(df_switch, 'num_switches_stable_orig', 'mean_switches_stable_sim', 'std_switches_stable_sim',
         axes[0], ax_subt=2, xlabel='Actual # of Switches', ylabel=f"Model Generated \n# of Switches",
         title=True, title_str=f"Stable Block \n" + "Spearman $\\it{ρ}$ = ", fontsize=fontsize)

# Plot number of switches in volatile block (PPC vs actual data)
plot_ppc(df_switch, 'num_switches_volatile_orig', 'mean_switches_volatile_sim', 'std_switches_volatile_sim',
         axes[1], ax_subt=2, xlabel='Actual # of Switches', ylabel=f"Model Generated \n# of Switches",
         title=True, title_str=f"Volatile Block \n" + "Spearman $\\it{ρ}$ = ", fontsize=fontsize)

# Plot overall P(Correct) (PPC vs actual data)
plot_ppc(df_p_correct, 'p_correct_orig', 'mean_p_correct',
         'std_p_correct', axes[2], line_limits=1, ax_subt=0.02,
         xlabel='Actual P(Correct)', ylabel=f"Model Generated \nP(Correct)",
         title=True, title_str="Spearman $\\it{ρ}$ = ", fontsize=fontsize)

# Plot the distribution of P(Correct) across simulations
sns.kdeplot(np.mean(p_corr_combined_arr, axis=0), fill=True, label='Model', ax=axes[3])
axes[3].axvline(np.mean(df_p_correct['p_correct_orig']), color='r', linestyle='-', linewidth=1.5, label='Data')

axes[3].legend(fontsize=fontsize - 1, handlelength=0.75)
PPC_ax_setup(axes[3], xlabel='P(Correct)', ylabel='Posterior Density', fontsize=fontsize)

# ----------
# 6. Add labels and save figure
# -------------
texts = ['a', 'b', 'c', 'd']
label_subplots(f, texts, x_offset=0.09, y_offset=0.03)

sns.despine(f)
plt.tight_layout()

# save figure
name = 'figure_s17.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()
