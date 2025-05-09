# plot showing predator task and model figure, along with motivation and anxiety questionnaire results
# Environment: predator_task_env

from PIL import Image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import zscore
from functions.util_functions import CircularDistance_Array, cm2inch, label_axes, add_text, compute_median_iqr
from functions.plotting_functions import plot_x_vs_y_robust

# -----------------
# 1. Load data
# -----------------
figure_folder = '../figures/'
img_path = "../figures/generated_anims/Full_figure_noText.png"

df_model = pd.read_csv("../data/predator_task/sim_df_bayesian_model_300tr.csv")
df_endquiz = pd.read_csv("../data/predator_task/predator_endquiz_data.csv")
factor_scores = pd.read_csv('../data/factor_analysis/factor_scores.csv')

# -----------------
# 2. Preprocess data
# -----------------

# Preprocess endquiz scores and merge with factor scores
df_endquiz = df_endquiz.rename(columns={'SD01_01': 'Age','SD02': 'Gender'})
df_factor = factor_scores.rename(columns={'V1': 'subjectID'})

# remove participants with non-binary gender due to insufficient participants for control
df_endquiz = df_endquiz[df_endquiz['Gender'] != 3]

# merge with factor scores
df_endquiz = df_endquiz.merge(df_factor, on='subjectID')
df_endquiz['g_z'] = zscore(df_endquiz['g'])

# -----------------
# 2. Prepare figure
# -----------------

# Size of figure
fig_height = 12
fig_width = 15

# Create figure
f = plt.figure(figsize=cm2inch(fig_width, fig_height))
f.canvas.draw()

# Create plot grid
gs_0 = gridspec.GridSpec(3, 2, wspace=0.44, hspace=0.7, top=0.90, bottom=0.1, left=0.15, right=0.96)

# Define plot colors
colors = ["#80cdc1",'#a17ab1', "#dfc27d", "#018571"]
sns.set_palette(sns.color_palette(colors))

# ----------------------------
# 3. Plot task trial schematic
# ----------------------------

# Create subplot grid and axis
gs_00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_0[0:1,0:1])
ax_0 = plt.Subplot(f, gs_00[0])
f.add_subplot(ax_0)

# Figure text and font size
text = ['Prediction', 'Outcome\n(1.4s)', 'Prediction\nError', 'Update (max. 5s)']
fontsize = 7

# Initialize image coordinates
cell_x0 = 0.0
cell_x1 = 0.2
image_y = 0.3
image_x = 0.42

# Initialize text coordinates
text_y_dist = [0.1, 0.22, 0.22, 0.1]
text_pos = 'left_below'

# Open image
img = Image.open(img_path)

# Image zoom factor and axis and coordinates
imagebox = OffsetImage(img, zoom=0.042)
imagebox.image.axes = ax_0
ab = AnnotationBbox(imagebox, (image_x, image_y), xybox=None,
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0, frameon=False)
ax_0.add_artist(ab)


# Add the texts ontop
text_kwargs={'fontsize':fontsize-1, 'color': 'black',
             'horizontalalignment': 'center',
             'verticalalignment': 'center'}

text_kwarg2={'fontsize':fontsize-1, 'color': 'black',
             'horizontalalignment': 'left',
             'verticalalignment': 'center'}

add_text(f,imagebox,image_x-0.4,image_y+0.06,ax_0,'Prediction', text_kwargs=text_kwargs)
add_text(f,imagebox,image_x-0.27,image_y-0.12,ax_0,'Outcome\n(1.5s)', text_kwargs=text_kwarg2)
add_text(f,imagebox,image_x+0.1,image_y-0.27,ax_0,'Prediction\nError', text_kwargs=text_kwargs)
add_text(f,imagebox,image_x+0.25,image_y-0.4,ax_0,'Update\n(max. 5s)', text_kwargs=text_kwarg2)

# Delete unnecessary axes
ax_0.axis('off')

# --------------------------------------------
# 4. Plot block example and model computations
# --------------------------------------------

# Create subplot grid
gs_01 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_0[1:3,0:1], hspace=0.5)

# Indicate plot range and x-axis
plot_range = (190, 250) # original 0 t0 45
x = np.linspace(0, plot_range[1]-plot_range[0]-1, plot_range[1]-plot_range[0])

# Mean, outcomes and predictions
ax_10 = plt.Subplot(f, gs_01[0:2, 0])
f.add_subplot(ax_10)
ax_10.plot(x, df_model['pred_mean'][plot_range[0]:plot_range[1]], '--',
           x, df_model['pred_loc'][plot_range[0]:plot_range[1]], '.', color="#090030")
ax_10.plot(x, df_model['sim_b_t'][plot_range[0]:plot_range[1]], linewidth=2, color="#de77ae", alpha=0.8)
ax_10.set_ylabel('Position', fontsize=fontsize)
ax_10.legend(["Predator Mean", "Outcome", "Model"], loc='upper left', bbox_to_anchor=(0.75, 1.40), #(0.6,1.34)
             framealpha=0.8, fontsize=fontsize-1, handlelength=1)
ax_10.set_ylim(0, 350)
ax_10.set_xticklabels([''])

ax_10.xaxis.set_tick_params(labelsize=fontsize)
ax_10.yaxis.set_tick_params(labelsize=fontsize)

# Estimation Errors
ax_11 = plt.Subplot(f, gs_01[2, 0])
f.add_subplot(ax_11)

est_error = CircularDistance_Array(df_model['pred_mean'].astype('float') ,df_model['sim_b_t'].astype('float'))

ax_11.plot(x,df_model['delta_t'][plot_range[0]:plot_range[1]], linewidth=2, color="#090030", alpha=1)
ax_11.set_xticklabels([''])
ax_11.set_ylabel('Prediction \n Error', fontsize=fontsize)

ax_11.xaxis.set_tick_params(labelsize=fontsize)
ax_11.yaxis.set_tick_params(labelsize=fontsize)

# Relative uncertainty, changepoint probability and learning rate
ax_12 = plt.Subplot(f, gs_01[3, 0])
f.add_subplot(ax_12)
ax_12.plot(x, df_model['tau_t'][plot_range[0]:plot_range[1]], linewidth=2, color="#04879c", alpha=1)
ax_12.plot(x, df_model['omega_t'][plot_range[0]:plot_range[1]], linewidth=2, color="#0c3c78", alpha=1)
ax_12.plot(x, df_model['alpha_t'][plot_range[0]:plot_range[1]], linewidth=2, color="#de77ae", alpha=0.8)
ax_12.legend(['RU', 'CPP', 'LR'], loc='upper left', bbox_to_anchor=(0.96, 1.2),
             fontsize=fontsize-1,handlelength=1)
ax_12.set_xlabel('Trial', fontsize=fontsize)
ax_12.set_ylabel('Variable', fontsize=fontsize)

ax_12.xaxis.set_tick_params(labelsize=fontsize)
ax_12.yaxis.set_tick_params(labelsize=fontsize)
f.align_ylabels()

# -------------------------------------
# 5. Add anxiety and predator ratings on 2nd column
# ----------------------------------
# Create subplot grid
gs_11 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_0[0:3,1:], hspace=0.6, wspace=0.8)
ax_13 = plt.Subplot(f, gs_11[0, 1])
ax_14 = plt.Subplot(f, gs_11[0, 0])
ax_15 = plt.Subplot(f, gs_11[1, 0])
ax_16 = plt.Subplot(f, gs_11[1, 1])

f.add_subplot(ax_13)
f.add_subplot(ax_14)
f.add_subplot(ax_15)
f.add_subplot(ax_16)

# Plot histogram of magnitudes on first column
ax_13.hist(df_endquiz['anxiety_rating'], bins=15, color='#77AADD', edgecolor='black', alpha=0.7)

ax_13.set_title("How anxious did the game \n make you feel?", fontsize=fontsize-1,  loc='center')

ax_13.set_ylabel('Count', fontsize=fontsize)
ax_13.set_xlabel('Anxiety Rating', fontsize=fontsize)
ax_13.xaxis.set_tick_params(labelsize=fontsize)
ax_13.yaxis.set_tick_params(labelsize=fontsize)

ax_14.hist(df_endquiz['predator_rating'], bins=15, color='#77AADD', edgecolor = 'black', alpha=0.7)
ax_14.set_title("How much did you want \n to avoid the predator?", fontsize=fontsize-1,  loc='center')

ax_14.set_ylabel('Count', fontsize=fontsize)
ax_14.set_xlabel('Predator Rating', fontsize=fontsize)
ax_14.xaxis.set_tick_params(labelsize=fontsize)
ax_14.yaxis.set_tick_params(labelsize=fontsize)


# Plot correlation with Sticsa
r_IC, P_IC, t_IC = plot_x_vs_y_robust(df_endquiz,x='IC02',y='anxiety_rating',title=False, ax=ax_15,tstat=True, xlabel='STICSA-T', ylabel='Anxiety Rating' ,fontsize=fontsize, color_index=-2,line_color_index=-1)
title = "$\it{r}$ = " + str(r_IC) + ", $\it{p}$ < 0.001 "
ax_15.set_title(title, fontsize=fontsize)

# Correlation with G-score (internalizing)
r_g, P_g, t_g = plot_x_vs_y_robust(df_endquiz,x='g',y='anxiety_rating',title=False, ax=ax_16,tstat=True, xlabel='General Factor', ylabel='Anxiety Rating' ,fontsize=fontsize, color_index=-2,line_color_index=-1)

title = "$\it{r}$ = " + str(r_g) + ", $\it{p}$ < 0.001 "
ax_16.set_title(title, fontsize=fontsize)
ax_16.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))


# Calculate overall stats
# calculate overall stats of questionaires
mean_anxiety = np.mean(df_endquiz['anxiety_rating'])
std_anxiety = np.std(df_endquiz['anxiety_rating'])
median_anxiety, anxiety_iqi = compute_median_iqr(df_endquiz['anxiety_rating'])

mean_predator = np.mean(df_endquiz['predator_rating'])
std_predator = np.std(df_endquiz['predator_rating'])
median_predator, predator_iqi = compute_median_iqr(df_endquiz['predator_rating'])


# Delete unnecessary axes
sns.despine()

# plt.tight_layout()
# -------------------------------------
# 5. Add subplot labels and save figure
# -------------------------------------

# add labels for a and b
texts = ['a']
label_axes(f,[ax_0], texts, x_offset=0.085, y_offset=0.06,fontsize=fontsize)

texts = ['b']
label_axes(f,[ax_10,], texts, x_offset=0.085, y_offset=0.04,fontsize=fontsize)

texts = ['c','d']
label_axes(f,[ax_14,ax_13], texts, x_offset=0.065, y_offset=0.06,fontsize=fontsize)

# add labels to bottom row
texts = ['e','f']
label_axes(f,[ax_15,ax_16], texts, x_offset=0.065, y_offset=0.04,fontsize=fontsize)

name='figure_3_predator_task.pdf'
savename = os.path.join(figure_folder, name)
plt.savefig(savename, format='pdf', dpi=700, transparent=True)
plt.show()

# create a dictionary for stats
stats = {
    'Statistic':['mean_anx','std_anx','median_anx','anx_25','anx_75',
                 'mean_predator','std_predator','median_predator','pred_25','pred_75',
                'r_anx','p_anx','anx_tstatistic',
                 'r_g','p_g','g_tstatistic',
                 'n_subj'],
    'Values':[mean_anxiety,std_anxiety,median_anxiety,anxiety_iqi[0],anxiety_iqi[1],
              mean_predator,std_predator,median_predator,predator_iqi[0],predator_iqi[1],
              r_IC,P_IC,t_IC,
              r_g,P_g,t_g,
              len(df_endquiz)]
}

print(stats)

