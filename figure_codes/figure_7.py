# Plot task schematic for the reversal learning task, along with VKF simulations
# Environment: predator_task_env

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
from functions.util_functions import cm2inch, label_subplots, plot_image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -----------------
# 1. Load data
# -----------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
figure_folder = base_dir + '/figures/'
df_vkf_sim = pd.read_csv(os.path.join(base_dir, 'data/reversal_task/VKF_Lambda0.1_v0.1_w0.05_Sim.csv'))

# Picture paths
path = [base_dir + '/figures/generated_anims/Stimulus_grey_noText.png', base_dir + '/figures/generated_anims/Choice_grey_noText.png',
        base_dir + '/figures/generated_anims/Result_grey_noText.png']

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_width = 15
fig_height = 11
fontsize = 7

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(2, 2, wspace=0.25, hspace=0.25, top=0.95, bottom=0.1, left=0.15, right=0.98)

# Set plot colors
colors = ["#80cdc1",'#a17ab1', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))


# ----------------------------
# 3. Plot task trial schematic
# ----------------------------

# Create subplot grid and axis

ax_0 = plt.Subplot(f, gs_0[0, 0])
f.add_subplot(ax_0)

# Figure text and font size
text = ['Stimuli', 'Choice', 'Feedback']

# Initialize image coordinates
cell_x0 = 0.0
cell_x1 = 0.2
image_y = 0.85

# Initialize text coordinates
text_y_dist = [0.20, 0.20, 0.26]
text_pos = 'left'

text_kwargs = {'fontsize':fontsize-3.72, 'color': 'white',
               'horizontalalignment': 'center',
               'verticalalignment': 'center'
               }

text_kwargs_rew = {'fontsize':fontsize-3.85, 'color': '#00f857',
               'horizontalalignment': 'center',
               'verticalalignment': 'center'
               }

# Cycle over images
for i in range(0, len(path)):

    # Plot images and text
    ax,bbox,ab = plot_image(f, path[i], cell_x0, cell_x1, image_y, ax_0, text_y_dist[i], text[i], text_pos, fontsize, zoom=0.05)
    if (i==len(path)-1):
        text_add = f'Trials Left: 79\nScore: 10'
        ax.text(bbox.x0 + 0.25, bbox.y1 - 0.055, text_add, zorder=5, **text_kwargs)

        # green text for reward
        text_rew = f'Reward\n +10'
        ax.text(bbox.x0 + 0.255, bbox.y1-0.13, text_rew, zorder=5, **text_kwargs_rew)
    else:
        text_add=f'Trials Left: 80\nScore: 0'
        ax.text(bbox.x0 + 0.25, bbox.y1 - 0.055, text_add, zorder=5, **text_kwargs)



    # Update coordinates
    cell_x0 += 0.25
    cell_x1 += 0.25
    image_y += -0.35

# Delete unnecessary axes
ax_0.axis('off')

# -----------------
# 4. Prepare reward probability figure
# -----------------
ax_1 = plt.Subplot(f, gs_0[1, 0])
f.add_subplot(ax_1)

stable_rp = 0.75
volatile_rp = 0.8
num_trials = 320

# Define the reward probabilities for each segment
reward_probs = np.zeros(num_trials)
reward_probs[:180] = stable_rp  # First 180 trials

# Switch reward probability between 0.8 and 0.2 every 20 trials for the last 140 trials
for i in range(180, 320, 20):
    if (i // 20) % 2 == 0:
        reward_probs[i:i+20] = 1 - volatile_rp
    else:
        reward_probs[i:i+20] = volatile_rp

# Shade the first 180 trials
ax_1.axvspan(0, 180, facecolor='lightgrey', alpha=0.5)

# Plot the reward probabilities
ax_1.plot(reward_probs, drawstyle='steps-post', color='black', linestyle='--')

# Add vertical line to separate the blocks
ax_1.axvline(x=180, color='#FFAABB', linestyle='--')

ax_1.set_ylim(0, 1)
ax_1.set_xlim(-15, 320)

# Add labels and title
ax_1.set_xlabel('Trials', fontsize=fontsize)
ax_1.set_ylabel('P(fractal 1 results in reward)', fontsize=fontsize)

# Add text labels for "Stable" and "Volatile"
ax_1.text(90, 0.85, 'Stable', fontsize=fontsize, ha='center')
ax_1.text(250, 0.85, 'Volatile', fontsize=fontsize, ha='center')
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# -----------------
# 5. Plot VKF simulations
# -----------------
gs_00 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_0[0:2,1])
ax2_1 = plt.Subplot(f, gs_00[1, 0])
f.add_subplot(ax2_1)
color = '#77AADD'

# Plot LR
ax2_1.plot(df_vkf_sim['lr'], color=color, linestyle='-', linewidth=1.25, zorder=5)
ax2_1.axvline(x=180, color='#FFAABB', linestyle='--', zorder=2)

# create a vertical line whenever reward probability changes
df_vkf_sim['reversal'] = 0
for i in range(181,len(df_vkf_sim)):
    if (df_vkf_sim['reward_probability'][i] != df_vkf_sim['reward_probability'][i - 1]):
        ax2_1.axvline(x=i, color='#808080', linestyle='--',linewidth=0.5, alpha=0.5, zorder=1)
        df_vkf_sim['reversal'][i] = 1

ax2_1.axvspan(0, 180, facecolor='lightgrey', alpha=0.5)

ax2_1.set_ylabel('Learning Rate', fontsize=fontsize)
ax2_1.xaxis.set_tick_params(labelsize=fontsize)
ax2_1.yaxis.set_tick_params(labelsize=fontsize)
ax2_1.set_ylim(0.25, 0.4)
ax2_1.set_xlim(-15, 320)
ax2_1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
ax2_1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

# Plot Volatility
ax2_2 = plt.Subplot(f, gs_00[0, 0])
f.add_subplot(ax2_2)

reversal_indices = np.where(df_vkf_sim['reversal'] == 1)[0]
ax2_2.plot(df_vkf_sim['vol'], c=color, linewidth=1.25, zorder=5)
ax2_2.set_ylabel('Volatility', fontsize=fontsize)
ax2_2.set_ylim(0.0, 0.15)
ax2_2.set_xlim(-15, 320)
ax2_2.axvline(x=180, color='#FFAABB', linestyle='--', zorder=2)

# plot vertical lines where reversal = 1
for idx in reversal_indices:
    ax2_2.axvline(x=idx, color='#808080', linestyle='--', linewidth=0.5, zorder=1, alpha=0.5)

# Shade the first 180 trials
ax2_2.axvspan(0, 180, facecolor='lightgrey', alpha=0.5)
ax2_2.xaxis.set_tick_params(labelsize=fontsize)
ax2_2.yaxis.set_tick_params(labelsize=fontsize)
ax2_2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
ax2_2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

# Plot predictions and outcomes and reward probabilities
ax2_3 = plt.Subplot(f, gs_00[2, 0])
f.add_subplot(ax2_3)

ax2_3.plot(df_vkf_sim['Predictions'], c=color, label='Predicted', linewidth=1.25,alpha = 1, zorder=5)
ax2_3.plot(df_vkf_sim['reward_probability'].iloc[0:319], label='True', drawstyle = 'steps-post', color = '#404040', linestyle = '--', alpha=0.6, zorder=4)

ax2_3.scatter(range(len(df_vkf_sim)), df_vkf_sim['Outcomes'], c='g', s=0.75)
ax2_3.axvline(x=180, color='#FFAABB', linestyle='--', zorder=2)

# Shade the first 180 trials
ax2_3.axvspan(0, 180, facecolor='lightgrey', alpha=0.5)

ax2_3.set_ylabel('Reward Probability', fontsize=fontsize)
ax2_3.set_xlabel('Trials', fontsize=fontsize)
ax2_3.xaxis.set_tick_params(labelsize=fontsize)
ax2_3.yaxis.set_tick_params(labelsize=fontsize)
ax2_3.legend(loc='upper left', fontsize=fontsize-1, handlelength=0.75, bbox_to_anchor=(0.025, 0.4))
ax2_3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
ax2_3.set_xlim(-15, 320)
ax2_3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

# -----------------
# 6. Label plots and save figure
# -----------------
sns.despine()

# Label letters
texts = ['a','b', '', 'c', '']

# Add labels
label_subplots(f, texts, x_offset=0.13, y_offset=0.03,fontsize=fontsize+1)

sns.despine()
name = 'figure_7_reversal_task'+'.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename,format='pdf',dpi=700,transparent=True)
plt.show()


