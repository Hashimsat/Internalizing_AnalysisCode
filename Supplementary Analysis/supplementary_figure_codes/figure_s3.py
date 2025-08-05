# Histogram of training quiz scores for participants in the predator task (Different variability, hazard rate version)

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from functions.util_functions import cm2inch

# -----------------
# 1. Load data
# -----------------
target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

training_quiz_df = pd.read_csv(os.path.join(target_dir,
                                            'supplementary_data/predator_task/predator_4exp_training_quiz_data.csv'))

figure_folder = target_dir + "/supplementary_figures"

# -------------------
# 2. Setup figure
# -------------------

# Size of figure
fig_width = 7
fig_height = 6
fontsize = 7

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create grid
gs_0 = gridspec.GridSpec(1, 1, wspace=0.3, hspace=0.6, top=0.85, bottom=0.2, left=0.2, right=0.90)
colors = ['#77AADD', '#009988', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))

# Plot 1 - Histogram of training quiz scores
ax1 = plt.Subplot(f, gs_0[0, 0])
f.add_subplot(ax1)

ax1.hist(training_quiz_df['total_score'], align='mid', bins=[-0.5, 0.5, 1.5, 2.5, 3.5], color='#77AADD',
         edgecolor='black', alpha=0.7)
ax1.set_ylabel('Count', fontsize=fontsize)
ax1.set_xlabel('Training Quiz Score', fontsize=fontsize)
ax1.xaxis.set_tick_params(labelsize=fontsize)
ax1.yaxis.set_tick_params(labelsize=fontsize)
ax1.set_xticks([0, 1, 2, 3])

sns.despine(f)

# -----------------
# 3. Save figure
# -----------------

name = 'figure_s3.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=False, bbox_inches='tight')
plt.show()
