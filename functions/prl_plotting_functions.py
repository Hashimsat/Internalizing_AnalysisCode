# Functions for plotting distributions from the reversal learning task

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pymc3 as pm


name_replace = {
    'lr_baseline':r'Baseline',
    'lr_goodbad':r'Good-bad',
    'lr_stabvol':r'Volatile-stable',
    'lr_goodbad_stabvol':'(Good-bad) x \n (Volatile-stable)',
    'lr_rewpain':r'Reward-loss',
    'lr_rewpain_goodbad':'(Reward-loss) x \n (Good-bad)',
    'lr_rewpain_stabvol':'(Reward-loss) x \n (Volatile-stable)',
    'lr_rewpain_goodbad_stabvol':r'$(reward-loss)x(good-bad)x(volatile-stable)$',
    'lr_c_baseline':r'$\eta_{baseline}$',
    'Amix_baseline':r'$\lambda_{baseline}$',
    'Amix_goodbad':r'$\lambda_{good-bad}$',
    'Amix_stabvol':r'$\lambda_{volatile-stable}$',
    'Amix_goodbad_stabvol':r'$\lambda_{(good-bad)x(volatile-stable)}$',
    'Binv_baseline':r'Baseline',
    'Binv_goodbad':r'Good-bad',
    'Binv_stabvol':r'Volatile-stable',
    'Binv_goodbad_stabvol':'(Good-bad) x \n (Volatile-stable)',
    'Binv_rewpain': r'Reward-loss',
    'Binv_rewpain_goodbad': '(Reward-loss) x \n (Good-bad)}',
    'Binv_rewpain_stabvol': '(Reward-loss) x \n (Volatile-stable)',
    'Bc_baseline':r'$\omega_{(k)baseline}$',
    'mag_baseline':r'$r_{baseline}$',
    'Amix_rewpain':r'$\lambda_{reward-aversive}$',
    'Amix_rewpain_goodbad':r'$\lambda_{(reward-aversive)x(good-bad)}$',
    'Amix_rewpain_stabvol':r'$\lambda_{(reward-aversive)x(volatile-stable)}$',
    'Bc_rewpain':r'$\omega_{(k) reward-aversive}$',
    'mag_rewpain':r'$r_{reward-aversive}$',
}

name_replace_RewardLoss = {
    'lr_baseline':r'Baseline',
    'lr_goodbad':r'Good-bad',
    'lr_stabvol':r'Volatile-stable',
    'lr_goodbad_stabvol':'(Good-bad)x(Volatile-stable)',
    'lr_rewpain':r'Reward-loss',
    'lr_rewpain_goodbad':'(Reward-loss)x(Good-bad)',
    'lr_rewpain_stabvol':'(Reward-loss)x(Volatile-stable)',
    'lr_rewpain_goodbad_stabvol':r'$(reward-loss)x(good-bad)x(volatile-stable)$',
    'lr_c_baseline':r'$\eta_{baseline}$',
    'Amix_baseline':r'$\lambda_{baseline}$',
    'Amix_goodbad':r'$\lambda_{good-bad}$',
    'Amix_stabvol':r'$\lambda_{volatile-stable}$',
    'Amix_goodbad_stabvol':r'$\lambda_{(good-bad)x(volatile-stable)}$',
    'Binv_baseline':r'Baseline',
    'Binv_goodbad':r'Good-bad',
    'Binv_stabvol':r'Volatile-stable',
    'Binv_goodbad_stabvol':'(Good-bad)x(Volatile-stable)',
    'Binv_rewpain': r'Reward-loss',
    'Binv_rewpain_goodbad': '(Reward-loss)x(Good-bad)',
    'Binv_rewpain_stabvol': '(Reward-loss)x(Volatile-stable)',
    'Bc_baseline':r'$\omega_{(k)baseline}$',
    'mag_baseline':r'$r_{baseline}$',
    'Amix_rewpain':r'$\lambda_{reward-aversive}$',
    'Amix_rewpain_goodbad':r'$\lambda_{(reward-aversive)x(good-bad)}$',
    'Amix_rewpain_stabvol':r'$\lambda_{(reward-aversive)x(volatile-stable)}$',
    'Bc_rewpain':r'$\omega_{(k) reward-aversive}$',
    'mag_rewpain':r'$r_{reward-aversive}$',
}



def boxprop_specifics():
    boxprops = dict(linestyle='-', linewidth=0.5, color='k')
    whiskerprops = dict(linestyle='-', linewidth=0.5, color='k')
    medianprops = dict(linestyle='-', linewidth=1, color='k')

    return boxprops, whiskerprops, medianprops

def basecoding(gb,sv, rp):
    basecode=[0,0,0]

    if gb=='good':
        basecode[0]=1
    else:
        basecode[0]=-1

    if sv=='stable':
        basecode[1]=-1
    else:
        basecode[1]=1

    if rp == 'rew':
        basecode[2] = 1
    else:
        basecode[2] = -1

    return(basecode)

def plot_param_posterior_distribution_onesubplot(
        trace=None, # data
        params=None, # model parameter names
        gp='u', # group parameter
        param = 'lr', # readable name
        taskVersion='reward',
        offset=0.5,
        ax=None,# plot characteristics
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
        ebar_offset = 0.15,
        legend_anchor=[0.45,-0.9],
        boxplot=True):

    '''Error bar plot for parameter components for one parameter type (i.e learning rate)
       Inputs:
           ax: for a subplot of a larger figure
    '''

    # set current axis
    plt.sca(ax)

    boxprops, whiskerprops, medianprops = boxprop_specifics()

    # get the indexes for the model parameters
    pis = [pi for pi,p in enumerate(params) if param in p and param+'_c' not in p]
    piis =np.arange(len(pis))

    if (taskVersion == 'rewardLoss'):
        params_tmp = [name_replace_RewardLoss[params[pi]] for pi in pis]
    else:
        params_tmp = [name_replace[params[pi]] for pi in pis]

    trace_params = np.squeeze(trace[gp][:,pis])
    df = pd.DataFrame(trace_params)
    df.columns = params_tmp

    # plot violinplot
    v = ax.violinplot(df, vert=True,positions=np.array(piis)+offset, showextrema=False, widths=1)

    for b_no,b in enumerate(v['bodies']):
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
        beta_independent = np.mean(trace['Theta_both'],axis=0)[:,pis]

        df_beta_independent = pd.DataFrame(beta_independent,columns = params_tmp)
        df_long = pd.melt(df_beta_independent, value_vars=params_tmp,
                          value_name='Estimated value', var_name='Parameter')

        ax.boxplot(x=df_beta_independent,vert=True,positions=pis, showfliers=False, widths=bp_width,
                   boxprops=boxprops,
                   whiskerprops=whiskerprops,capprops=whiskerprops,
                   medianprops=medianprops,)

        strip = sns.stripplot(y='Estimated value', x="Parameter", data=df_long,palette=colors,
                              jitter=0.02, linewidth=0, size=2, alpha=0.25, zorder=1, dodge=False, ax=ax)


    if (boxplot == False):
        # Plot 95% hdi lines
        mu = np.mean(trace[gp][:, pis], axis=0)

        interval = np.squeeze(pm.stats.hpd(trace[gp][:, pis], alpha=0.05))
        lower2p5 = interval[:,0]
        upper97p5 = interval[:,1]

        # error bar for group mean and HDI's
        err_val = np.squeeze(np.array([[mu[:,0] - lower2p5], [upper97p5 - mu[:,0]]]))

        plt.errorbar(piis + ebar_offset, mu, yerr=err_val,
                     color=color_errbar,
                     marker='o',
                     markersize=s_bar,
                     elinewidth=elinewidth, linestyle='',
                     label=legendlabel)

    if legend:
        ax.legend(ncol=2, loc=legendloc,bbox_to_anchor=legend_anchor, fontsize=fontsize-1)

    # horizontal line
    plt.axhline(y=0,linestyle='--',color='k',linewidth=0.5, alpha=0.7);

    # set labels
    ax.set_xticks(piis)

    if xlabel:
        ax.set_xticklabels(params_tmp,rotation=rotation, fontsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize)

    else:
        ax.set_xticklabels([])

    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel('')
    ax.set_ylim(np.min(interval)-0.45, np.max(interval)+0.45)
    ax.set_xlim(-1, len(pis)+0.3)


def plot_factor_errorbar(trace=None, params=None,
                         ax=None,
                         factor='g',
                         param = 'lr',
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
    plt.yticks(fontsize=fontsize);
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlim(np.min(piis)-0.5, np.max(piis)+0.5)

    # horizontal line
    plt.axhline(y=0, linestyle='--', color='k', linewidth=0.5);

    if legend:
        plt.legend(loc=legendloc, ncol=1, bbox_to_anchor=legend_anchor, fontsize=fontsize-1)


def get_param_by_subj_by_cond(Theta,
                              index,
                              transform='invlogit',
                              effects=[],
                              domain = ['rew'],
                              n_subs=0):
    '''
    Converts params from sampling space to conditions.

    Inputs:
        Theta to be point estimate so 157xK
        index of parameters in Theta like [0,1,2,4] for learning rate

    '''

    total_conditions = 4*len(domain)
    param = np.zeros((n_subs, total_conditions))
    B_trace = Theta[:,index]

    for subj in range(n_subs):
        conds = []
        ci=0
        for rp in domain:
            for gb in ['good','bad']:
                for sv in ['stable','volatile']:
                    block = gb+' '+sv
                    basecode=basecoding(gb,sv,rp)

                    code = [] # needs to be size of the number of effects
                    for effect in effects:
                        if effect=='baseline':
                            code.append(1)
                        elif effect=='goodbad':
                            code.append(basecode[0])
                        elif effect=='stabvol':
                            code.append(basecode[1])
                        elif effect=='goodbad_stabvol':
                            code.append(basecode[0]*basecode[1])
                        elif effect == 'rewpain':
                            code.append(basecode[2])
                        elif effect == 'rewpain_goodbad':
                            code.append(basecode[2] * basecode[0])
                        elif effect == 'rewpain_stabvol':
                            code.append(basecode[1] * basecode[2])
                        elif effect == 'rewpain_goodbad_stabvol':
                            code.append(basecode[2] * basecode[0] * basecode[1])



                    if transform=='invlogit':
                        try:
                            param[subj,ci] = (scipy.special.expit(np.sum(np.array(code)*B_trace[subj,:])))
                        except:
                            import pdb; pdb.set_trace()
                    elif transform=='exp':
                        param[subj,ci] = (np.exp(np.sum(np.array(code)*B_trace[subj,:])))
                    elif transform=='None':
                        param[subj,ci] = ((np.sum(np.array(code)*B_trace[subj,:])))
                    elif transform=='invlogit5':
                        try:
                            param[subj,ci] = (5*scipy.special.expit(np.sum(np.array(code)*B_trace[subj,:])))
                        except:
                            import pdb; pdb.set_trace()

                    ci+=1
                    conds.append(block)

    return(param,conds)


def param_by_factor_score(trace,df_data,model,
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

    plt.ylim([0,1])



def extract_distribution_mean_hdpis(trace=None,model=None,
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
        mu = np.round(np.mean(trace[factor][:, pi], axis=0),2)

        # calculate eror bars
        interval = pm.stats.hpd(trace[factor][:, pi].flatten(), alpha=0.05)
        lower2p5 = round(interval[0],2)
        upper97p5 = round(interval[1],2)

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


def plot_param_separated_by_domain(trace,df_data,model,
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
                                           domain=['rew','pain'],
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
        title= task + ' domain'
        plt.title(title, fontsize=fontsize)

    plt.xticks(np.arange(len(pos)), conds[pos[0]:pos[-1]+1], rotation=rotation,
               fontsize=fontsize);
    plt.ylabel('Learning rate', fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if (legend):
        plt.legend(loc=legendloc, ncol=1, bbox_to_anchor=legend_anchor, fontsize=fontsize - 1)

    plt.xlim(0 - 0.6, len(pos) + 0.3)
    plt.ylim([-0.1, 1.1])

