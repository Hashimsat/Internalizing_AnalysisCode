# Figure S33: PPC for winning model for PRL With Reward Mag- Reward and Loss Domains

import sys
# path to model_code directory
sys.path.append("../../reversal_task_model/")

import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from functions.util_functions import cm2inch, medianprops, label_subplots
from functions.plotting_functions import create_subplots
from functions.prl_plotting_functions import ppc_calculate_measures, plot_ppc_allPlots

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
df_switch, df_p_correct, p_corr_combined_arr = ppc_calculate_measures(actual_data_dict, ppc_samples)

# ------------
# 4. Figure Setup
# ------------
fig_width = 15
fig_height = 13
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
plot_ppc_allPlots(df_switch, df_p_correct, p_corr_combined_arr, axes, fontsize=fontsize)

# ----------
# 6. Add labels and save figure
# -------------
texts = ['a', 'b', 'c', 'd']
label_subplots(f, texts, x_offset=0.09, y_offset=0.03)

sns.despine(f)
plt.tight_layout()

# save figure
name = 'figure_s33.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()
