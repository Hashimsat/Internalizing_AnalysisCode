# Hypotheses figure
# Environment: predator_task_env

import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from functions.util_functions import cm2inch, label_axes, plot_opened_image_with_text


# -----------------
# 1. Load data and figures
# -----------------

# Load normative learning simulation
df_norm = pd.read_csv("../data/predator_task/simulation_normative_learning.csv")
figure_folder = "../figures/"

# Load path to animated figures

path = ['figures/generated_anims/Illustration_Left_noText.png',
        'figures/generated_anims/Illustration_Right_noText.png']

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_width = 14
fig_height = 11
fontsize = 7

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))

# Create plot grid
gs_0 = gridspec.GridSpec(2, 6, wspace=0.5, hspace=0.3, top=0.73, bottom=0.1, left=0.1, right=.91)

# Set up color palette
colors = ["#92e0a9", "#69b0c1", "#6d6192", "#352d4d"]
sns.set_palette(sns.color_palette(colors))

# -----------------
# 3. Plot exmaple illustrations
# -----------------

gs_00 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs_0[0, 0:6], wspace=0.5)
ax_0 = plt.Subplot(f, gs_00[0, 0:3])
ax_1 = plt.Subplot(f, gs_00[0, 3:])
f.add_subplot(ax_0)
f.add_subplot(ax_1)

text1 = 'Bears\n expected,\n stay cautious'
text2 = 'Unexpected\n bear, get more\n cautious here'
text_kwargs = {'fontsize': fontsize - 1, 'color': 'white',
               'horizontalalignment': 'center',
               'verticalalignment': 'center'
               }

plot_opened_image_with_text(path[0], ax_0, 0.42, 0.985, zoom=0.115, text=text1, text_x=0.27, text_y=1.53,
                            text_kwargs=text_kwargs)

plot_opened_image_with_text(path[1], ax_1, 0.52, 0.938, zoom=0.115, text=text2, text_x=0.81, text_y=1.42,
                            text_kwargs=text_kwargs)

# -----------------
# 4. Plot learning-rate simulations
# -----------------

# Plot H1: Normative Learning vs Over-learning

ax_20 = plt.subplot(gs_0[1, 0:2])
plt.axhline(y=0.9, linestyle='-', c="#de77ae", linewidth=2)
ax_20.plot(df_norm['Prediction Error'], df_norm['Learning Rate'], color="#249886", linewidth=2)

ax_20.set_ylim([-0.02, 1.1])
ax_20.set_yticks(np.arange(0, 1.3, 0.5))
ax_20.set_ylabel('Learning Rate', fontsize=fontsize)
ax_20.set_xlabel('Prediction Error', fontsize=fontsize)
title = 'H1: High Learning Rate'
ax_20.set_title(title, fontsize=fontsize - 1, pad=7)
ax_20.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)

# Plot H2: Normative Learning vs impaired adaptive learning

ax_21 = plt.subplot(gs_0[1, 2:4])
ax_21.plot(df_norm['Prediction Error'], 0.5 * df_norm['Learning Rate'], color="#de77ae", linewidth=2)
ax_21.plot(df_norm['Prediction Error'], df_norm['Learning Rate'], color="#249886", linewidth=2)

ax_21.set_ylim([-0.02, 1.1])
ax_21.set_yticks(np.arange(0, 1.3, 0.5))
ax_21.set_xlabel('Prediction Error', fontsize=fontsize)
title = 'H2: Impaired Adaptation to Change'
ax_21.set_title(title, fontsize=fontsize - 1, pad=7)

ax_21.set_yticklabels([])
ax_21.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
plt.xticks(fontsize=fontsize)

# Plot H3: No difference in learning

ax_22 = plt.subplot(gs_0[1, 4:6])
ax_22.plot(df_norm['Prediction Error'], 0.98 * df_norm['Learning Rate'], color="#de77ae", linewidth=2, alpha=0.8)
ax_22.plot(df_norm['Prediction Error'], df_norm['Learning Rate'], color="#249886", linewidth=2, alpha=0.8)

ax_22.set_ylim([-0.02, 1.1])
ax_22.set_yticks(np.arange(0, 1.3, 0.5))
ax_22.set_xlabel('Prediction Error', fontsize=fontsize)
ax_22.legend(["High Internalizing", "Normative Agent"], bbox_to_anchor=(0.35, 0), loc="lower left", framealpha=0,
             fontsize=fontsize - 1, handlelength=1)
ax_22.set_yticklabels([])
plt.xticks(fontsize=fontsize)
title = 'H3: No Change in Learning Rate'
ax_22.set_title(title, fontsize=fontsize - 1, pad=7)
ax_22.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

# Add labels to axes

texts = ['a']
label_axes(f, [ax_0], texts, x_offset=0.05, y_offset=0.16, fontsize=fontsize)

texts = ['b', ]

label_axes(f, [ax_1], texts, x_offset=0.04, y_offset=0.16, fontsize=fontsize)

texts = ['c', 'd', 'e']

label_axes(f, [ax_20, ax_21, ax_22], texts, x_offset=0.03, y_offset=0.07, fontsize=fontsize)

# -----------------
# 5. Save figure
# -----------------
sns.despine()
name = 'fig_1_hypothesis' + '.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format="pdf", dpi=500, transparent=True)
plt.show()
