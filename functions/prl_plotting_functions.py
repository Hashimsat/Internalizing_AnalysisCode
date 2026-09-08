# Functions for plotting distributions from the reversal learning task

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import scipy
from scipy import stats
from functions.util_functions import medianprops, compute_median_iqr, compute_test_statistic, cm2inch
from functions.prl_descriptive_functions import calculate_switches_PPC, calculate_p_correct_PPC
import matplotlib.gridspec as gridspec
import pymc3 as pm

name_replace = {
    'lr_baseline': r'Baseline',
    'lr_goodbad': r'Good-bad',
    'lr_stabvol': r'Volatile-stable',
    'lr_goodbad_stabvol': '(Good-bad) x \n (Volatile-stable)',
    'lr_rewpain': r'Reward-loss',
    'lr_rewpain_goodbad': '(Reward-loss) x \n (Good-bad)',
    'lr_rewpain_stabvol': '(Reward-loss) x \n (Volatile-stable)',
    'lr_rewpain_goodbad_stabvol': r'$(reward-loss)x(good-bad)x(volatile-stable)$',
    'lr_c_baseline': r'$\eta_{baseline}$',
    'Amix_baseline': r'$\lambda_{baseline}$',
    'Amix_goodbad': r'$\lambda_{good-bad}$',
    'Amix_stabvol': r'$\lambda_{volatile-stable}$',
    'Amix_goodbad_stabvol': r'$\lambda_{(good-bad)x(volatile-stable)}$',
    'Binv_baseline': r'Baseline',
    'Binv_goodbad': r'Good-bad',
    'Binv_stabvol': r'Volatile-stable',
    'Binv_goodbad_stabvol': '(Good-bad) x \n (Volatile-stable)',
    'Binv_rewpain': r'Reward-loss',
    'Binv_rewpain_goodbad': '(Reward-loss) x \n (Good-bad)}',
    'Binv_rewpain_stabvol': '(Reward-loss) x \n (Volatile-stable)',
    'Bc_baseline': r'$\omega_{(k)baseline}$',
    'mag_baseline': r'$r_{baseline}$',
    'Amix_rewpain': r'$\lambda_{reward-aversive}$',
    'Amix_rewpain_goodbad': r'$\lambda_{(reward-aversive)x(good-bad)}$',
    'Amix_rewpain_stabvol': r'$\lambda_{(reward-aversive)x(volatile-stable)}$',
    'Bc_rewpain': r'$\omega_{(k) reward-aversive}$',
    'mag_rewpain': r'$r_{reward-aversive}$',
}

name_replace_RewardLoss = {
    'lr_baseline': r'Baseline',
    'lr_goodbad': r'Good-bad',
    'lr_stabvol': r'Volatile-stable',
    'lr_goodbad_stabvol': '(Good-bad)x(Volatile-stable)',
    'lr_rewpain': r'Reward-loss',
    'lr_rewpain_goodbad': '(Reward-loss)x(Good-bad)',
    'lr_rewpain_stabvol': '(Reward-loss)x(Volatile-stable)',
    'lr_rewpain_goodbad_stabvol': r'$(reward-loss)x(good-bad)x(volatile-stable)$',
    'lr_c_baseline': r'$\eta_{baseline}$',
    'Amix_baseline': r'$\lambda_{baseline}$',
    'Amix_goodbad': r'$\lambda_{good-bad}$',
    'Amix_stabvol': r'$\lambda_{volatile-stable}$',
    'Amix_goodbad_stabvol': r'$\lambda_{(good-bad)x(volatile-stable)}$',
    'Binv_baseline': r'Baseline',
    'Binv_goodbad': r'Good-bad',
    'Binv_stabvol': r'Volatile-stable',
    'Binv_goodbad_stabvol': '(Good-bad)x(Volatile-stable)',
    'Binv_rewpain': r'Reward-loss',
    'Binv_rewpain_goodbad': '(Reward-loss)x(Good-bad)',
    'Binv_rewpain_stabvol': '(Reward-loss)x(Volatile-stable)',
    'Bc_baseline': r'$\omega_{(k)baseline}$',
    'mag_baseline': r'$r_{baseline}$',
    'Amix_rewpain': r'$\lambda_{reward-aversive}$',
    'Amix_rewpain_goodbad': r'$\lambda_{(reward-aversive)x(good-bad)}$',
    'Amix_rewpain_stabvol': r'$\lambda_{(reward-aversive)x(volatile-stable)}$',
    'Bc_rewpain': r'$\omega_{(k) reward-aversive}$',
    'mag_rewpain': r'$r_{reward-aversive}$',
}


def boxprop_specifics():
    boxprops = dict(linestyle='-', linewidth=0.5, color='k')
    whiskerprops = dict(linestyle='-', linewidth=0.5, color='k')
    medianprops = dict(linestyle='-', linewidth=1, color='k')

    return boxprops, whiskerprops, medianprops


def basecoding(gb, sv, rp):
    basecode = [0, 0, 0]

    if gb == 'good':
        basecode[0] = 1
    else:
        basecode[0] = -1

    if sv == 'stable':
        basecode[1] = -1
    else:
        basecode[1] = 1

    if rp == 'rew':
        basecode[2] = 1
    else:
        basecode[2] = -1

    return (basecode)


def plot_param_posterior_distribution_onesubplot(
        trace=None,  # data
        params=None,  # model parameter names
        gp='u',  # group parameter
        param='lr',  # readable name
        taskVersion='reward',
        offset=0.5,
        ax=None,  # plot characteristics
        colors='k',
        fontsize=7,
        bp_width=0.1,
        color_errbar='k',
        legend=False,
        legendlabel='posterior mean (w/ 95% HDI)',
        ylabel=None,
        xlabel=True,
        legendloc='best',
        s_bar=5,
        rotation=45,
        elinewidth=1,
        ebar_offset=0.15,
        legend_anchor=[0.45, -0.9],
        boxplot=True):
    '''Error bar plot for parameter components for one parameter type (i.e learning rate)
       Inputs:
           ax: for a subplot of a larger figure
    '''

    # set current axis
    plt.sca(ax)

    boxprops, whiskerprops, medianprops = boxprop_specifics()

    # get the indexes for the model parameters
    pis = [pi for pi, p in enumerate(params) if param in p and param + '_c' not in p]
    piis = np.arange(len(pis))

    if (taskVersion == 'rewardLoss'):
        params_tmp = [name_replace_RewardLoss[params[pi]] for pi in pis]
    else:
        params_tmp = [name_replace[params[pi]] for pi in pis]

    trace_params = np.squeeze(trace[gp][:, pis])
    df = pd.DataFrame(trace_params)
    df.columns = params_tmp

    # plot violinplot
    v = ax.violinplot(df, vert=True, positions=np.array(piis) + offset, showextrema=False, widths=1)

    for b_no, b in enumerate(v['bodies']):
        # get the center
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # modify the paths to not go further right than the center
        b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
        b.set_color(colors[0])
        b.set_alpha(0.75)
        b.set_linewidth(0.7)
        b.set_edgecolor('#BBBBBB')

    # Plot boxplots for independent vars for each participant

    if (boxplot):
        beta_independent = np.mean(trace['Theta_both'], axis=0)[:, pis]

        df_beta_independent = pd.DataFrame(beta_independent, columns=params_tmp)
        df_long = pd.melt(df_beta_independent, value_vars=params_tmp,
                          value_name='Estimated value', var_name='Parameter')

        ax.boxplot(x=df_beta_independent, vert=True, positions=pis, showfliers=False, widths=bp_width,
                   boxprops=boxprops,
                   whiskerprops=whiskerprops, capprops=whiskerprops,
                   medianprops=medianprops, )

        strip = sns.stripplot(y='Estimated value', x="Parameter", data=df_long, palette=colors,
                              jitter=0.02, linewidth=0, size=2, alpha=0.25, zorder=1, dodge=False, ax=ax)

    if (boxplot == False):
        # Plot 95% hdi lines
        mu = np.mean(trace[gp][:, pis], axis=0)

        interval = np.squeeze(pm.stats.hpd(trace[gp][:, pis], alpha=0.05))
        lower2p5 = interval[:, 0]
        upper97p5 = interval[:, 1]

        # error bar for group mean and HDI's
        err_val = np.squeeze(np.array([[mu[:, 0] - lower2p5], [upper97p5 - mu[:, 0]]]))

        plt.errorbar(piis + ebar_offset, mu, yerr=err_val,
                     color=color_errbar,
                     marker='o',
                     markersize=s_bar,
                     elinewidth=elinewidth, linestyle='',
                     label=legendlabel)

    if legend:
        ax.legend(ncol=2, loc=legendloc, bbox_to_anchor=legend_anchor, fontsize=fontsize - 1)

    # horizontal line
    plt.axhline(y=0, linestyle='--', color='k', linewidth=0.5, alpha=0.7)

    # set labels
    ax.set_xticks(piis)

    if xlabel:
        ax.set_xticklabels(params_tmp, rotation=rotation, fontsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize)

    else:
        ax.set_xticklabels([])

    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel('')
    ax.set_ylim(np.min(interval) - 0.45, np.max(interval) + 0.45)
    ax.set_xlim(-1, len(pis) + 0.3)


def plot_factor_errorbar(trace=None, params=None,
                         ax=None,
                         factor='g',
                         param='lr',
                         offset=0,
                         ylabel='effect of general factor \n on update',
                         xlabel=True,
                         legend=False,
                         legendlabel=None,
                         legendloc='best',
                         taskVersion='reward',
                         rotation=45,
                         fontsize=6,
                         color='black',
                         elinewidth=1,
                         s_bar=3,
                         legend_anchor=[1, 1.1]):
    # set current axis
    plt.sca(ax)

    # get the indexes for the model parameters
    pis = [pi for pi, p in enumerate(params) if param in p and param + '_c' not in p]
    piis = np.arange(len(pis))

    if (taskVersion == 'rewardLoss'):
        params_tmp = [name_replace_RewardLoss[params[pi]] for pi in pis]
    else:
        params_tmp = [name_replace[params[pi]] for pi in pis]

    for ii, (pii, pi, param) in enumerate(zip(piis, pis, params_tmp)):

        mu = np.mean(trace[factor][:, pi], axis=0)

        # calculate eror bars
        interval = pm.stats.hpd(trace[factor][:, pi].flatten(), alpha=0.05)
        lower2p5 = interval[0]
        upper97p5 = interval[1]

        if ii == (len(piis) - 1):
            legendlabeltmp = legendlabel
        else:
            legendlabeltmp = None

        # error bar for group mean and HDI's
        err_val = np.array([[mu[0] - lower2p5], [upper97p5 - mu[0]]])

        plt.errorbar(pii + offset, mu, yerr=err_val,
                     color=color, label=legendlabeltmp,
                     marker='o',
                     markersize=s_bar,
                     elinewidth=elinewidth)

    # labels
    if xlabel:
        plt.xticks(piis, params_tmp, rotation=rotation, fontsize=fontsize);

    else:
        plt.xticks(ticks=piis, labels=[])
    plt.yticks(fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlim(np.min(piis) - 0.5, np.max(piis) + 0.5)

    # horizontal line
    plt.axhline(y=0, linestyle='--', color='k', linewidth=0.5)

    if legend:
        plt.legend(loc=legendloc, ncol=1, bbox_to_anchor=legend_anchor, fontsize=fontsize - 1)


def get_param_by_subj_by_cond(Theta,
                              index,
                              transform='invlogit',
                              effects=[],
                              domain=['rew'],
                              n_subs=0):
    '''
    Converts params from sampling space to conditions.

    Inputs:
        Theta to be point estimate so 157xK
        index of parameters in Theta like [0,1,2,4] for learning rate

    '''

    total_conditions = 4 * len(domain)
    param = np.zeros((n_subs, total_conditions))
    B_trace = Theta[:, index]

    for subj in range(n_subs):
        conds = []
        ci = 0
        for rp in domain:
            for gb in ['good', 'bad']:
                for sv in ['stable', 'volatile']:
                    block = gb + ' ' + sv
                    basecode = basecoding(gb, sv, rp)

                    code = []  # needs to be size of the number of effects
                    for effect in effects:
                        if effect == 'baseline':
                            code.append(1)
                        elif effect == 'goodbad':
                            code.append(basecode[0])
                        elif effect == 'stabvol':
                            code.append(basecode[1])
                        elif effect == 'goodbad_stabvol':
                            code.append(basecode[0] * basecode[1])
                        elif effect == 'rewpain':
                            code.append(basecode[2])
                        elif effect == 'rewpain_goodbad':
                            code.append(basecode[2] * basecode[0])
                        elif effect == 'rewpain_stabvol':
                            code.append(basecode[1] * basecode[2])
                        elif effect == 'rewpain_goodbad_stabvol':
                            code.append(basecode[2] * basecode[0] * basecode[1])

                    if transform == 'invlogit':
                        try:
                            param[subj, ci] = (scipy.special.expit(np.sum(np.array(code) * B_trace[subj, :])))
                        except:
                            import pdb;
                            pdb.set_trace()
                    elif transform == 'exp':
                        param[subj, ci] = (np.exp(np.sum(np.array(code) * B_trace[subj, :])))
                    elif transform == 'None':
                        param[subj, ci] = ((np.sum(np.array(code) * B_trace[subj, :])))
                    elif transform == 'invlogit5':
                        try:
                            param[subj, ci] = (5 * scipy.special.expit(np.sum(np.array(code) * B_trace[subj, :])))
                        except:
                            import pdb;
                            pdb.set_trace()

                    ci += 1
                    conds.append(block)

    return (param, conds)


def param_by_factor_score(trace, df_data, model,
                          param='lr',
                          pc='u_PC1',
                          ax=None,
                          median=False,
                          split='mean',
                          transform='invlogit',
                          legendloc='best',
                          fontsize=7,
                          color='black',
                          scatter_offset=0,
                          markersize=3,
                          elinewidth=1,
                          s=1,
                          include_errorbar=True,
                          ebar_offset=0,
                          legend_anchor=[0.45, -0.9]
                          ):
    # set current axis
    plt.sca(ax)

    participant_sel = np.ones(len(df_data['Bi1item_w_j_scaled'])).astype('bool')

    if pc == 'u_PC1':
        factor = 'general'
        factor_in_data = 'Bi1item_w_j_scaled'

    if pc == 'u_PC2':
        factor = 'factor1'
        factor_in_data = 'Bi2item_w_j_scaled'
    if pc == 'u_PC3':
        factor = 'factor2'
        factor_in_data = 'Bi3item_w_j_scaled'

    # get average parameter per participant
    Theta = trace['Theta'].mean(axis=0)

    effects = ['baseline', 'goodbad', 'stabvol', 'goodbad_stabvol']

    pis = [i for i, p in enumerate(model.params) if (param in p) and (param + '_c' not in p)]
    piis = np.arange(len(pis))
    params_tmp = [model.params[pi] for pi in pis]

    # individual subject parameters by condition
    lrs, conds = get_param_by_subj_by_cond(Theta,
                                           index=pis,
                                           effects=effects,
                                           transform=transform,
                                           n_subs=len(participant_sel))

    params = params_tmp
    # do a split by factor
    if median == True:
        thresh = np.median(df_data[factor_in_data])
    else:
        thresh = np.mean(df_data[factor_in_data])

    high_idx = np.logical_and(df_data[factor_in_data] >= thresh, participant_sel)
    low_idx = np.logical_and(df_data[factor_in_data] < thresh, participant_sel)

    # index for the parameters
    pos = pis

    # some more specifications based on split
    if split == 'high':
        idx = high_idx
        color = sns.color_palette()[1]
        extra_legend = ', High G'
        extra_legend_scatter1 = ''
        extra_legend_scatter2 = ' (high ' + factor + ' factor scores)'
    elif split == 'low':
        idx = low_idx
        color = sns.color_palette()[0]
        extra_legend = ', Low G'
        extra_legend_scatter1 = ''
        extra_legend_scatter2 = ' (low ' + factor + ' factor scores)'
    elif split == 'mean':
        idx = np.arange(len(df_data[factor_in_data]))
        color = 'k'
        extra_legend = ' for group average'
        extra_legend_scatter1 = 'individual '
        extra_legend_scatter2 = ''

    # scatter individuals
    mean_arr = np.empty(len(pis))
    mean_arr[:] = np.nan
    std_arr = np.empty(len(params))
    std_arr[:] = np.nan

    for j, i in enumerate(params):  # j is 1-4, i can be 4-8
        y = lrs[idx, j]
        x = np.ones_like(y) * j + 0.1

        mean_arr[j] = np.nanmean(y)
        std_arr[j] = np.nanstd(y)
        if i == pos[-1]:
            plt.scatter(x + scatter_offset, y, c=color, marker="x", s=s,
                        label=extra_legend_scatter1 + 'participants' + extra_legend_scatter2)  # +eq+'0 on '+factor+' factor')
        else:
            plt.scatter(x + scatter_offset, y, c=color, marker="x", s=s)

    if include_errorbar:
        # posterior mean estimates
        plt.errorbar(np.arange(len(pos)) - 0.1 + ebar_offset,
                     y=mean_arr,
                     yerr=std_arr,
                     color=color,
                     elinewidth=elinewidth,
                     # label='posterior mean (w/ std)' + extra_legend,
                     label='mean ± std' + extra_legend,
                     linestyle='None', marker='o', markersize=markersize)

    plt.xticks(np.arange(len(conds)), conds, rotation=45,
               fontsize=fontsize);
    plt.ylabel('Learning rate', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc=legendloc, ncol=2, bbox_to_anchor=legend_anchor, fontsize=fontsize - 1)
    plt.xlim(np.min(pos) - 0.6, np.max(pos) + 0.6)

    plt.ylim([0, 1])


def extract_distribution_mean_hdpis(trace=None, model=None,
                                    factor='u',
                                    param='lr'):
    #function to extract mean parameter values and 95% hdpis

    # get the indexes for the model parameters
    pis = [pi for pi, p in enumerate(model.params) if param in p and param + '_c' not in p]
    piis = np.arange(len(pis))

    params_tmp = [name_replace[model.params[pi]] for pi in pis]
    params_model_name = [model.params[pi] for pi in pis]

    mean_array = np.empty(len(pis))
    mean_array[:] = np.nan

    lower_interval = np.empty(len(pis))
    lower_interval[:] = np.nan

    upper_interval = np.empty(len(pis))
    upper_interval[:] = np.nan

    for ii, (pii, pi, param) in enumerate(zip(piis, pis, params_tmp)):
        mu = np.round(np.mean(trace[factor][:, pi], axis=0), 2)

        # calculate eror bars
        interval = pm.stats.hpd(trace[factor][:, pi].flatten(), alpha=0.05)
        lower2p5 = round(interval[0], 2)
        upper97p5 = round(interval[1], 2)

        mean_array[ii] = mu
        lower_interval[ii] = lower2p5
        upper_interval[ii] = upper97p5

    # add to a dataframe
    df_hdi = pd.DataFrame()
    df_hdi['mean_effect'] = mean_array
    df_hdi['lower_hdi'] = lower_interval
    df_hdi['upper_hdi'] = upper_interval
    df_hdi['param'] = params_model_name

    return df_hdi


def plot_param_separated_by_domain(trace, df_data, model,
                                   param='lr',
                                   pc='u_PC1',
                                   ax=None,
                                   task=None,
                                   median=False,
                                   split='mean',
                                   transform='invlogit',
                                   legendloc='best',
                                   legend=True,
                                   fontsize=7,
                                   color='black',
                                   title=True,
                                   scatter_offset=0,
                                   markersize=3,
                                   elinewidth=1,
                                   rotation=45,
                                   s=1,
                                   include_errorbar=True,
                                   ebar_offset=0,
                                   legend_anchor=[0.45, -0.9]
                                   ):
    # set current axis
    plt.sca(ax)

    participant_sel = np.ones(len(df_data['Bi1item_w_j_scaled'])).astype('bool')

    if pc == 'u_PC1':
        factor = 'general'
        factor_in_data = 'Bi1item_w_j_scaled'

    if pc == 'u_PC2':
        factor = 'factor1'
        factor_in_data = 'Bi2item_w_j_scaled'
    if pc == 'u_PC3':
        factor = 'factor2'
        factor_in_data = 'Bi3item_w_j_scaled'

    # get average parameter per participant
    Theta = trace['Theta'].mean(axis=0)

    effects = ['baseline', 'goodbad', 'stabvol', 'goodbad_stabvol', 'rewpain', 'rewpain_goodbad', 'rewpain_stabvol']

    pis = [i for i, p in enumerate(model.params) if (param in p) and (param + '_c' not in p)]
    piis = np.arange(len(pis))
    params_tmp = [model.params[pi] for pi in pis]

    # individual subject parameters by condition
    lrs, conds = get_param_by_subj_by_cond(Theta,
                                           index=pis,
                                           effects=effects,
                                           transform=transform,
                                           domain=['rew', 'pain'],
                                           n_subs=len(participant_sel))

    params = params_tmp
    # do a split by factor
    if median == True:
        thresh = np.median(df_data[factor_in_data])
    else:
        thresh = np.mean(df_data[factor_in_data])

    high_idx = np.logical_and(df_data[factor_in_data] >= thresh, participant_sel)
    low_idx = np.logical_and(df_data[factor_in_data] < thresh, participant_sel)

    # indexes for the parameter
    if task == 'reward':
        pos = np.array([0, 1, 2, 3])
        slicee = slice(0, 4)
    elif task == 'aversive' or task == 'loss':
        pos = np.array([4, 5, 6, 7])
        slicee = slice(4, 8)

    # some more specifications based on split

    if split == 'high':
        idx = high_idx
        color = sns.color_palette()[1]
        extra_legend = ', High G'
        extra_legend_scatter1 = ''
        extra_legend_scatter2 = ' (high ' + factor + ' factor scores)'
    elif split == 'low':
        idx = low_idx
        color = sns.color_palette()[0]
        extra_legend = ', Low G'
        extra_legend_scatter1 = ''
        extra_legend_scatter2 = ' (low ' + factor + ' factor scores)'
    elif split == 'mean':
        idx = np.arange(len(df_data[factor_in_data]))
        color = 'k'
        extra_legend = ' for group average'
        extra_legend_scatter1 = 'individual '
        extra_legend_scatter2 = ''

    # scatter individuals
    mean_arr = np.empty(len(pos))
    mean_arr[:] = np.nan
    yerr_arr = np.empty(len(pos))
    yerr_arr[:] = np.nan
    std_arr = np.empty(len(pos))
    std_arr[:] = np.nan

    for j, i in enumerate(pos):  # j is 1-4, i can be 4-8
        y = lrs[idx, i]
        x = np.ones_like(y) * j + 0.1
        yerr_arr[j] = y.std() / np.sqrt(len(y))
        std_arr[j] = np.nanstd(y)

        mean_arr[j] = np.nanmean(y)
        if i == pos[-1]:
            plt.scatter(x + scatter_offset, y, c=color, marker="x", s=s,
                        label=extra_legend_scatter1 + 'participants' + extra_legend_scatter2)  # +eq+'0 on '+factor+' factor')
        else:
            plt.scatter(x + scatter_offset, y, c=color, marker="x", s=s)

    if include_errorbar:
        # posterior mean estimates
        plt.errorbar(np.arange(len(pos)) - 0.1 + ebar_offset,
                     y=mean_arr,
                     yerr=std_arr,
                     color=color,
                     elinewidth=elinewidth,
                     label='mean ± std' + extra_legend,
                     linestyle='None', marker='o', markersize=markersize)

    if title:
        title = task + ' domain'
        plt.title(title, fontsize=fontsize)

    plt.xticks(np.arange(len(pos)), conds[pos[0]:pos[-1] + 1], rotation=rotation,
               fontsize=fontsize);
    plt.ylabel('Learning rate', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if (legend):
        plt.legend(loc=legendloc, ncol=1, bbox_to_anchor=legend_anchor, fontsize=fontsize - 1)

    plt.xlim(0 - 0.6, len(pos) + 0.3)
    plt.ylim([-0.1, 1.1])


def get_boxplot(ax, data, positions, color, whiskerprops, meanprops, medianprops):
    """Generate a boxplot."""
    return ax.boxplot(data, positions=positions, patch_artist=True, showfliers=False,
                      boxprops=dict(alpha=0.5, linewidth=0.5, facecolor=color),
                      whiskerprops=whiskerprops, capprops=whiskerprops,
                      medianprops=medianprops, meanprops=meanprops, showmeans=False)


def plot_descriptive_boxplots(df, ax, colors, fontsize=7, prefix=None, order=None, title=True, Legend=False,
                              xlabel=None, ylabel=None, min_val=None, max_val=None, stat='mannU'):
    # Plot switch rates in stable vs volatile blocks
    prefix_stable = prefix + '_B0'
    prefix_volatile = prefix + '_B1'

    # separate data into low and high G
    df_lowG = df[df['G_Category'] == 'Low']
    df_highG = df[df['G_Category'] == 'High']

    whiskerprops = dict(color='k', linewidth=0.5)
    meanprops = None
    medianprop = medianprops()

    # initialize data to plot
    data_to_plot = [df_lowG[prefix_stable], df_highG[prefix_stable], df_lowG[prefix_volatile],
                    df_highG[prefix_volatile]]
    ps = [0, 0.5, 2, 2.5]

    # Plot boxplots
    bp1 = get_boxplot(ax, [df_lowG[prefix_stable], df_lowG[prefix_volatile]], [0, 2], colors[0], whiskerprops,
                      meanprops, medianprop)
    bp2 = get_boxplot(ax, [df_highG[prefix_stable], df_highG[prefix_volatile]], [0.5, 2.5], colors[1], whiskerprops,
                      meanprops, medianprop)

    for i in range(len(data_to_plot)):
        y = data_to_plot[i]
        #     # Add some random "jitter" to the x-axis
        x = np.random.normal(ps[i], 0.02, size=len(y))
        ax.scatter(x, y, alpha=0.4, color='#808080', s=3, edgecolors='none')

    # Calculate test statistics
    t_stable, p_stable, dof_stable, n_lowG, n_highG = compute_test_statistic(df, 'G_Category',
                                                                             prefix_stable, 'Low', 'High',
                                                                             test=stat)
    t_volatile, p_volatile, dof_volatile, _, _ = compute_test_statistic(df, 'G_Category', prefix_volatile,
                                                                        'Low', 'High', test=stat)

    stat_name = ['stable_t', 'stable_p', 'stable_dof', 'volatile_t', 'volatile_p', 'volatile_dof']
    stat_value = [round(t_stable, 2), round(p_stable, 2), round(dof_stable, 2), round(t_volatile, 2),
                  round(p_volatile, 2), round(dof_volatile, 2)]

    # calculate median and IQR
    # stable phase

    median_stable_LowG, stable_LowG_IQI = compute_median_iqr(df[df['G_Category'] == 'Low'][prefix_stable])
    median_stable_HighG, stable_HighG_IQI = compute_median_iqr(df[df['G_Category'] == 'High'][prefix_stable])

    # volatile phase
    median_volatile_LowG, volatile_LowG_IQI = compute_median_iqr(df[df['G_Category'] == 'Low'][prefix_volatile])
    median_volatile_HighG, volatile_HighG_IQI = compute_median_iqr(df[df['G_Category'] == 'High'][prefix_volatile])

    # Set plot title and labels
    if title:
        ax.set_title(f"$p_{{st}}={round(p_stable, 2)}$, $p_{{vol}}={round(p_volatile, 2)}$", fontsize=fontsize, y=1,
                     pad=5)
    if Legend:
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Low G', 'High G'], fontsize=fontsize - 1, handlelength=1)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_xticks([0.25, 2.25])
    ax.set_xticklabels(labels=['Stable', 'Volatile'])

    # Set y-axis limits
    min_val = df[[prefix_stable, prefix_volatile]].min().min() - 2 if min_val is None else min_val
    max_val = df[[prefix_stable, prefix_volatile]].max(numeric_only=True).max() + 0.5 if max_val is None else max_val
    ax.set_ylim([min_val, max_val])
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    # Prepare statistics dictionary
    stats_data = {
        'Statistic': ['median_stable_Low', 'median_stable_High', 'median_volatile_Low', 'median_volatile_High',
                      'median_stable_Low_iqr', 'median_stable_High_iqr', 'median_volatile_Low_iqr',
                      'median_volatile_High_iqr'] + stat_name,
        'Value': [median_stable_LowG, median_stable_HighG, median_volatile_LowG, median_volatile_HighG,
                  stable_LowG_IQI, stable_HighG_IQI, volatile_LowG_IQI, volatile_HighG_IQI] + stat_value
    }
    stats_data['Statistic'] = [f"{prefix}_{stat}" for stat in stats_data['Statistic']]

    return stats_data if title else (stats_data, round(p_stable, 2), round(p_volatile, 2))


# PPC plot
def plot_ppc(
        df: pd.DataFrame,
        x: str,
        y: str,
        yerr: str,
        ax: plt.Axes,
        label: str = None,
        line_limits: float = 90,
        ax_subt: float = 2,
        xlabel: str = None,
        ylabel: str = None,
        fontsize: int = 7,
        title: bool = False,
        title_str: str = None
) -> None:
    # Create the error bar plot for PPC

    ax.errorbar(df[x], df[y],
                yerr=df[yerr], fmt='o', capsize=0, alpha=0.7, label=label,
                markersize=3, markerfacecolor='none', elinewidth=0.25, markeredgewidth=0.6)

    # add a straight line showing correlation of 1
    ax.plot([0, line_limits], [0, line_limits], linestyle='-', color='k', linewidth=0.5, alpha=0.5)

    # set the same limits for x- and y- axes
    min_limit = min(min(df[x]), min(df[x])) - ax_subt
    max_limit = max(max(df[x]), max(df[y])) + ax_subt
    ax.set_xlim(min_limit, max_limit)
    ax.set_ylim(min_limit, max_limit)
    PPC_ax_setup(ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize)

    if (title):
        # calculate spearman correlation between ppc and original data
        r_val, _ = stats.spearmanr(df[x], df[y])
        title = title_str + str(np.round(r_val, 2))
        ax.set_title(title, fontsize=fontsize)

def plot_param_rec(params_recovered, theta_gen, beta_independent, fig_width=15, fig_height=15, fontsize=7, n_cols=4, n_rows=None):
    """
    Plot parameter recovery for the winning model for the probabilistic reversal learning task.

    Parameters
    ----------
    params_recovered : list of str
        List of parameter names to be plotted.
    theta_gen : numpy.ndarray
        Ground truth parameter values.
    beta_independent : numpy.ndarray
        Recovered parameter values.
    figure_folder : str
        Directory to save the generated figure.
    fig_width : int, optional
        Width of the figure in cm (default is 15).
    fig_height : int, optional
        Height of the figure in cm (default is 15).
    fontsize : int, optional
        Font size for the plot (default is 7).
    n_cols : int, optional
        Number of columns in the plot grid (default is 4).
    n_rows : int, optional
        Number of rows in the plot grid (default is None, which will be calculated based on the number of parameters and columns).
    """
    # Calculate correlations
    corr = [stats.spearmanr(beta_independent[:, i], theta_gen[:, i]).correlation for i in range(beta_independent.shape[1])]
    print(corr)

    # Determine number of rows based on params_recovered and n_cols
    if n_rows is None:
        n_rows = int(np.ceil(len(params_recovered) / n_cols))

    # Set up figure
    # medianprop = medianprops();
    color = ["#80cdc1", "#de77ae", "#018571", "#dfc27d", '#d492c8', '#AA4499', '#808080', "#77AADD", "#3576b8"]
    sns.set_palette(sns.color_palette(color))

    f = plt.figure(figsize=cm2inch(fig_width, fig_height))
    f.canvas.draw()
    gs_0 = gridspec.GridSpec(n_rows, n_cols, wspace=0.65, hspace=0.9, top=0.87, bottom=0.1, left=0.15, right=0.98)

    # Plot correlations
    for i in range(len(params_recovered)):
        col_no = i % n_cols
        row_no = i // n_cols

        ax = plt.Subplot(f, gs_0[row_no, col_no])
        f.add_subplot(ax)

        sns.regplot(x=theta_gen[:, i].astype('float'), y=beta_independent[:, i].astype('float'),
                    color=color[-1], robust=True, ax=ax,
                    scatter_kws=dict(alpha=0.3, s=10, edgecolor="none", color=color[-2]),
                    line_kws=dict(linewidth=2))

        ax.set_xlabel('Ground Truth', fontsize=fontsize)
        ax.set_ylabel('Recovered', fontsize=fontsize)

        title = params_recovered[i] + '\n' + "$Spearman \ \it{ρ}$ = " + str(round(corr[i], 2))
        ax.set_title(title, fontsize=fontsize)

        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

    sns.despine()
    return f


def ppc_calculate_measures(actual_data_dict, ppc_samples):
    """
    Calculate switch rates and P(Correct) for posterior predictive checks.

    Parameters
    ----------
    actual_data_dict : dict
        Dictionary containing actual data (e.g., participants' choices, outcomes, etc.).
    ppc_samples : numpy.ndarray
        Posterior predictive samples.

    Returns
    -------
    df_switch : pandas.DataFrame
        DataFrame containing switch statistics.
    df_p_correct : pandas.DataFrame
        DataFrame containing P(Correct) statistics.
    p_corr_combined_arr : numpy.ndarray
        Combined array of P(Correct) values across simulations.
    """
    actual_choices = actual_data_dict['participants_choice']
    outcome = actual_data_dict['outcomes_c_flipped']
    stabvol = actual_data_dict['stabvol']
    dominant_fractal = actual_data_dict['dominant_fractal']
    subjects = actual_data_dict['subjectID']

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
        df_subj_perf, p_correct_ppc = calculate_p_correct_PPC(df, dominant_col='dominant_fractal',
                                                              observed_col='observed')

        # Concatenate P(Correct) stats
        df_p_correct = pd.concat([df_p_correct, df_subj_perf], axis=0)
        p_corr_combined_list.append(p_correct_ppc)

    # Combine P(Correct) arrays
    p_corr_combined_arr = np.vstack(p_corr_combined_list)

    return df_switch, df_p_correct, p_corr_combined_arr


def PPC_ax_setup(ax, xlabel=None, ylabel=None, fontsize=7):
    # set the x and y labels
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)


def plot_ppc_allPlots(df_switch, df_p_correct, p_corr_combined_arr, axes, fontsize=7):
    """
    Plot posterior predictive checks (PPC) results.

    Parameters
    ----------
    df_switch : pandas.DataFrame
        DataFrame containing switch statistics.
    df_p_correct : pandas.DataFrame
        DataFrame containing P(Correct) statistics.
    p_corr_combined_arr : numpy.ndarray
        Combined array of P(Correct) values across simulations.
    axes : list of matplotlib.axes.Axes
        List of axes for plotting.
    fontsize : int, optional
        Font size for the plots (default is 7).
    """
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


def set_subplot_title(ax, r_stable, p_stable, r_volatile, p_volatile, fontsize):
    title_params = f"$r_{{stable}}={r_stable}, p_{{stable}}={p_stable}$\n$r_{{volatile}}={r_volatile}, p_{{volatile}}={p_volatile}$"
    ax.set_title(title_params, fontsize=fontsize)

def label_panel(ax, letter, x, y, fontsize):
    ax.text(x, y, letter, fontsize=fontsize, transform=ax.transAxes)

def add_legend(ax, **kwargs):
    ax.legend(**kwargs)

def despine(*axes):
    for ax in axes:
        sns.despine(ax=ax)
