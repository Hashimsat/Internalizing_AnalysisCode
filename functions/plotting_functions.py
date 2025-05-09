# functions that help with plotting

import numpy as np
from scipy.stats import zscore
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm
from functions.util_functions import compute_median_iqr, compute_test_statistic

medianprops = dict(linestyle='-', linewidth=1, color='k')
whiskerprops = dict(linewidth=0.5)
boxprops = dict(alpha=0.7)
colors = ["#80cdc1",'#de77ae',"#018571", "#dfc27d",'#d492c8','#AA4499','#808080',"#77AADD","#3576b8"]
# b082a8
sns.set_palette(sns.color_palette(colors))

def robust_regplot(df,x,y,ax,var_x_z,color_index=0,line_color_index=0, legend_txt=None, endog=None, exog=None,):
    # Plot a regression line with robust regression
    # Plots regression line, returns regression params from robust regression

    sns.regplot(x=df[x].astype('float'), y=df[y].astype('float'),  # color='#3576b8'
                color=colors[line_color_index], robust=True, ax=ax,  # label=legend_txt
                scatter_kws=dict(alpha=0.3, s=10, edgecolor="none", color=colors[color_index]),
                line_kws=dict(linewidth=2, label=legend_txt));

    # Fit robust model if endog and exog are not empty
    if endog is not None and exog is not None:
        rlm_model = sm.RLM(endog, exog, M=sm.robust.norms.HuberT())

        rlm_results = rlm_model.fit()
        r = round(rlm_results.params[var_x_z], 2)
        p = round(rlm_results.pvalues[var_x_z], 2)
        t = round(rlm_results.tvalues[var_x_z], 2)

        # return regression coefficients
        return r, p, t
    else:
        # If no endog and exog provided, return None
        return None, None, None

def plot_boxplots(df,x,y,ax,order=None,ylabel=None,xlabel=None,fontsize=7):

    # Function that plots boxplots and stripplots

    sns.boxplot(x=x, y=y, data=df, order=order, fliersize=0,
                linewidth=0.5, ax=ax, width=0.5,
                medianprops=medianprops,
                meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markeredgecolor': 'black', 'markersize': 10},
                boxprops=boxprops, zorder=7, )

    sns.stripplot(x=x, y=y, data=df, order=order, jitter=0.05,
                  alpha=0.3, size=3, zorder=0, ax=ax)

    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)

def boxplots_lab(df,ax,fontsize=7,abbr='EE',ylabel='Estimation Error', stat='ttest_rel'):
    """Plot boxplots of descriptive parameters from lab study, for shocks and screams"""

    df_relevant = df[[abbr + '_B0', abbr + '_B1']]
    df_long = df_relevant.melt(var_name='Block', value_name='Value')

    # Plot boxplots for lab study
    sns.boxplot(data=df_relevant, fliersize=0, linewidth=0.5, ax=ax, width=0.5,
                medianprops=medianprops,
                meanprops={'marker': 'o', 'markerfacecolor': 'red', 'markeredgecolor': 'black', 'markersize': 10},
                boxprops=dict(alpha=.7), zorder=7, )

    sns.stripplot(data=df_relevant, jitter=0.05, alpha=0.3, size=3, zorder=0, ax=ax)

    # Compute test statistics
    r, p, dof, n1, n2 = compute_test_statistic(df_long, group_col='Block', value_col='Value', group1=abbr + '_B0', group2=abbr + '_B1', test=stat)

    # set title
    title = "$\it{p}$ = " + str(round(p, 2))
    ax.set_title(title, fontsize=fontsize)

    # set x and y limits
    ax.set_xticklabels(['Screams', 'Shocks'])
    min = df_relevant.min().min()
    max = df_relevant.max().max()
    ax.set_ylim([min - 0.5, max + 0.5])
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    # extract median and iqr vals
    median_scream, scream_iqr = compute_median_iqr(df[abbr + '_B0'])
    median_shock, shock_iqr = compute_median_iqr(df[abbr + '_B1'])

    if (stat == 'mannU'):
        return r, p, median_scream, scream_iqr, median_shock, shock_iqr

    else:
        return r, p, dof, n1, n2, median_scream, scream_iqr, median_shock, shock_iqr



def plot_x_vs_y_robust(df,x,y,ax,legend_txt=None,xlabel=None,ylabel=None,title=False,tstat=False,fontsize=7,color_index=0,line_color_index=0):
    # plot y vs x regplot along with showing regression coeffs controlled for age and gender, with robust regression

    df = df[df['Gender']!=3]
    df['Age_z'] = zscore(df['Age'])
    df['var_x_z'] = zscore(df[x])
    df['var_y_z'] = zscore(df[y])

    x_temp = x.replace('.','')
    var_x_z = x_temp + '_z'

    exog = df[['var_x_z','Age_z','Gender']]
    exog = sm.add_constant(exog)
    endog = df['var_y_z']

    # Plot a regression line with robust regression
    r,p,t = robust_regplot(df,x,y,ax,var_x_z,color_index,line_color_index,legend_txt,endog,exog)

    if (title):
        title_curr =  '$\it{r}$ = ' + str(r) + ' , $\it{p}$ = ' + str(p)
        ax.set_title(title_curr, fontsize=fontsize)

    min = np.min(df[x])
    max = np.max(df[x])
    ax.set_xlim([min-0.7, max+0.5])
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    if (tstat):
        return (r,p, t)
    else:
        return (r,p)


def plot_x_vs_y_FactorScores_robust(df,x,y,ax,legend_txt=None,xlabel=None,ylabel=None,title=False,tstat=False,fontsize=7,color_index=0,line_color_index=0):
    # plot y vs x regplot along with showing regression coeffs controlled for age and gender, with robust regression


    df = df[df['Gender']!=3]
    df['Age_z'] = zscore(df['Age'])
    df['g_z'] = zscore(df['g'])
    df['F1_z'] = zscore(df['F1.'])
    df['F2_z'] = zscore(df['F2.'])
    df['var_y_z'] = zscore(df[y])

    # take out the . if it exists in x name
    x_temp = x.replace('.','')
    var_x_z = x_temp + '_z'

    exog = df[['g_z','F1_z','F2_z','Age_z','Gender']]
    exog = sm.add_constant(exog)
    endog = df['var_y_z']

    # Plot a regression line with robust regression
    r, p, t = robust_regplot(df, x, y, ax, var_x_z, color_index, line_color_index, legend_txt, endog, exog)

    if (title):
        title_curr =  "$\it{r}$ = " + str(r) +', ' + "$\it{p}$ = " + str(p)
        ax.set_title(title_curr, fontsize=fontsize)

    min = np.min(df[x])
    max = np.max(df[x])
    ax.set_xlim([min-0.5, max+0.5])
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    if (tstat):
        return (r,p, t)
    else:
        return (r,p)


def plot_descriptive_boxplots(df,x,y,ax,ylabel=None,xlabel=None,title=False,order=None,fontsize=7,min_val=None,max_val=None, stat='ttest'): #stats=mannU or ttest

    # Function that plots boxplots between low- and high-internalizing, and calculates test statistics

    plot_boxplots(df,x,y,ax,order=order,ylabel=ylabel,xlabel=xlabel,fontsize=fontsize)

    # Compute test statistics
    r,p,dof,n1,n2 = compute_test_statistic(df,x,y,group1='Low',group2='High',test=stat)

    median_Low, median_Low_iqr = compute_median_iqr(df[df[x] == 'Low'][y])
    median_High, median_High_iqr = compute_median_iqr(df[df[x] == 'High'][y])


    if title:
        title = "$\it{p}$ = " + str(round(p, 2))
        ax.set_title(title, fontsize=fontsize)


    if min_val==None:
        min_val = df[y].min()-0.5

    if max_val==None:
        max_val = df[y].max()+0.5
    ax.set_ylim([min_val, max_val])
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    if (stat == 'ttest_ind'):
        return r, p, dof, n1, n2, median_Low, median_Low_iqr, median_High, median_High_iqr

    else:
        return (r,p,median_Low,median_Low_iqr,median_High,median_High_iqr)