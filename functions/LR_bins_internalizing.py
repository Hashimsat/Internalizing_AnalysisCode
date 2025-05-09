# Compute single-trial learning rates for low and high internalizing for 20 prediciton error bins
# Plot the results, to check if there is a difference in learning rates between low and high internalizing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from functions.util_functions import circular_distance, BoundLR

def calculate_lr_stats(LR_quantiles, quantile_ranges, n_group):
    # Calculate mean, std, sem, and confidence interval
    LR_mean = np.mean(LR_quantiles, axis=0)
    LR_std = np.std(LR_quantiles, axis=0)
    LR_sem = LR_std / np.sqrt(n_group)
    CI_LR = stats.t.interval(
        0.95, len(LR_quantiles) - 1, loc=LR_mean, scale=stats.sem(LR_quantiles)
    )
    yerr = np.array([LR_mean - CI_LR[0], CI_LR[1] - LR_mean])

    # Calculate X-axis values (midpoints of quantile ranges)
    quartile_range_means = np.mean(quantile_ranges, axis=0)
    X_PE_val = [(a + b) / 2 for a, b in zip(quartile_range_means[:-1], quartile_range_means[1::])]

    return LR_mean, LR_std, LR_sem, yerr, X_PE_val

def plot_errorbars(x,y,yerr,ax,color,label):

    # plot errorbars for low and high anx groups
    ax.errorbar(x, y, yerr=yerr, capsize=1, c=color, marker='o',
                elinewidth=1, barsabove=False, ecolor='k', alpha=0.7,
                markersize=4,
                markerfacecolor=color,
                markeredgecolor='black', markeredgewidth=0.5,
                label=label,
                # ls='None'
                )


def learning_rate_descriptive_internalizing(df_data,n_lowG,n_highG,ax,Subjects,colors,fontsize,n_bins=19):
    # Gets a dataframe already separated into low and high internalizing, an axis on which to plot the figure.

    # calculate mean LR and std LR for each bin for each subject

    # Initialize arrays for low and high anxiety groups
    lowAnx_data = {
        "std_20quantiles": np.full([n_lowG, n_bins], np.nan),
        "LR_20quantiles": np.full([n_lowG, n_bins], np.nan),
        "quantile_ranges": np.full([n_lowG, n_bins + 1], np.nan),
        "count": 0
    }

    highAnx_data = {
        "std_20quantiles": np.full([n_highG, n_bins], np.nan),
        "LR_20quantiles": np.full([n_highG, n_bins], np.nan),
        "quantile_ranges": np.full([n_highG, n_bins + 1], np.nan),
        "count": 0
    }

    # Loop through each subject and calculate LR for each bin
    for subjIndex,subj in enumerate(Subjects):
        df_subj = df_data[(df_data['subjectID'] == subj)]

        LR_temp = np.array([])
        PE_temp = np.array([])
        Update_temp = np.array([])

        # Get the learning rate for each trial
        for i in range(len(df_subj) - 1):

            # only have PEs > 5 or < -5 to ensure we don't have PE = 0 or very small PE in denominator of LR=Update/PE
            if ((~np.isnan(df_subj['PredictionError'].iloc[i + 1])) and (df_subj['PredictionError'].iloc[i] != 0)
                    and (df_subj['torchMoved'].iloc[i] == 1) and (df_subj['torchMoved'].iloc[i + 1] == 1)
                    and ((df_subj['PredictionError'].iloc[i]>5) or (df_subj['PredictionError'].iloc[i]<-5))):

                Update = circular_distance(df_subj['torchAngle'].iloc[i + 1], df_subj['torchAngle'].iloc[i])
                PE = df_subj['PredictionError'].iloc[i]
                LR = Update / PE

                LR_temp = np.append(LR_temp, LR)
                PE_temp = np.append(PE_temp, PE)
                Update_temp = np.append(Update_temp, Update)

        LR_Bound = BoundLR(LR_temp) # constrain LR between 0 and 1

        # Separate PE and LR into quantiles
        PE_quant, b = pd.qcut(abs(PE_temp), n_bins, retbins=True, precision=3)

        # Create a DataFrame with both arrays and quantile for each value
        df = pd.DataFrame({'PE': PE_temp, 'LR': LR_Bound, 'quantiles': PE_quant})

        # Group by the quantile bins and calculate the median LR for each bin
        median_LR_by_quantile = df.groupby('quantiles')['LR'].median()
        std_by_quantile = df.groupby('quantiles')['LR'].std()

        # save the median LR and std for each bin for each subject
        category_data = {
            "Low": lowAnx_data,
            "High": highAnx_data
        }

        # Update the appropriate category data
        category = df_subj['G_Category'].iloc[0]
        if category in category_data:
            category_data[category]["quantile_ranges"][category_data[category]["count"], :] = b
            category_data[category]["LR_20quantiles"][category_data[category]["count"], :] = median_LR_by_quantile
            category_data[category]["std_20quantiles"][category_data[category]["count"], :] = std_by_quantile
            category_data[category]["count"] += 1


    # Calculate mean, std, sem, and confidence interval for low and high anxiety groups

    LR_20_mean_lowAnx, LR_20_std_lowAnx, LR_20_sem_lowAnx, yerr_low, X_PE_val_lowAnx = calculate_lr_stats(
        lowAnx_data["LR_20quantiles"], lowAnx_data["quantile_ranges"], n_lowG
    )

    LR_20_mean_highAnx, LR_20_std_highAnx, LR_20_sem_highAnx, yerr_high, X_PE_val_highAnx = calculate_lr_stats(
        highAnx_data["LR_20quantiles"], highAnx_data["quantile_ranges"], n_highG
    )

    # Plot the errorbars
    plot_errorbars(X_PE_val_lowAnx, LR_20_mean_lowAnx, yerr_low, ax, colors[0], 'Low G')
    plot_errorbars(X_PE_val_highAnx, LR_20_mean_highAnx, yerr_high, ax, colors[1], 'High G')

    # Set axis limits and labels
    ax = plt.gca()
    ax.grid(which='major', axis='x', linestyle='--')
    ax.set_ylabel('Learning Rate', fontsize=fontsize)
    ax.set_xlabel('Prediction Error', fontsize=fontsize)
    ax.legend(fontsize=fontsize, loc='lower right')
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.set_ylim([0.2, 1])



