# Function for descriptive analysis of the reversal learning task data

import numpy as np
import pandas as pd
from scipy.stats import zscore

def performance_prl(df_mega,subjects, blocks=None, block_name='BlockVersion'):

    if (blocks is None):
        blocks = np.sort(pd.unique(df_mega[block_name]))

    correct_percentage = np.full([len(subjects), len(blocks)],np.nan)
    subjList = []

    for subjIndex, subj in enumerate(subjects):
        subjList.append(subj)
        df_subj = df_mega[df_mega['subjectID'] == subj]

        for b in blocks:   # separate into stable and volatile phases
            df_subj_b = df_subj[df_subj[block_name] == b]
            TotalLength = len(df_subj_b)

            # find number of trials where participants chose the high reward fractal
            CorrectChosen = np.sum(df_subj_b['ChosenFractalKey'] == df_subj_b['HighRewardFractalKey'])
            correct_percentage[subjIndex, b] = (CorrectChosen / TotalLength)

    # save in a dataframe
    df_performance = pd.DataFrame(correct_percentage)
    df_performance.columns = ['Performance_B' + str(element) for element in blocks]
    df_performance['subjectID'] = subjList
    df_performance = df_performance.dropna()

    return df_performance

def calculate_switch_stay_percentages(df_block, hit_miss):
    """
    Calculate switch and stay percentages for a given block and hit/miss condition.
    """
    same_rows = df_block[(df_block['HitMiss'] == hit_miss) &
                         (df_block['ChosenFractalKey'] == df_block['ChosenFractalKey'].shift(-1))]
    switched_rows = df_block[(df_block['HitMiss'] == hit_miss) &
                             (df_block['ChosenFractalKey'] != df_block['ChosenFractalKey'].shift(-1))]

    num_switch = switched_rows.shape[0]
    num_hits_or_misses = df_block[df_block['HitMiss'] == hit_miss].shape[0]

    if num_hits_or_misses == 0:
        return np.nan, np.nan

    switch_percentage = num_switch / num_hits_or_misses
    stay_percentage = same_rows.shape[0] / num_hits_or_misses

    return switch_percentage, stay_percentage

def initialize_percentages(num_subjects, num_blocks):
    """
    Initialize arrays for switch and stay percentages.
    """
    shape = (num_subjects, num_blocks)
    return (
        np.full(shape, np.nan),  # switch_hit
        np.full(shape, np.nan),  # switch_miss
        np.full(shape, np.nan),  # stay_hit
        np.full(shape, np.nan),  # stay_miss
    )


def calculate_switch_rates(df, Subjects, block_name='BlockVersion'):
    # Calculate swithc rates after hits and misses for PRL different block versions

    # initialize variables
    block_version = np.sort(pd.unique(df[block_name]))

    # Initialize switch and stay percentages
    switch_hit, switch_miss, stay_hit, stay_miss = initialize_percentages(len(Subjects), len(block_version))

    for subj_index, subj in enumerate(Subjects):
        df_subj = df[df['subjectID'] == subj]

        for block in block_version:
            df_subj_b = df_subj[df_subj[block_name] == block]

            # Calculate switch and stay percentages for hits and misses
            switch_hit[subj_index, block], stay_hit[subj_index, block] = calculate_switch_stay_percentages(
                df_subj_b, hit_miss=1
            )
            switch_miss[subj_index, block], stay_miss[subj_index, block] = calculate_switch_stay_percentages(
                df_subj_b, hit_miss=0
            )
    # Save in a pandas dataframe
    df_switch = pd.DataFrame()

    for b in block_version:
        # save such that if b=0, name has stable in it, otherwise it has volatile
        # block_type = 'Stable' if b == 0 else 'Volatile'
        block_type = str(b)

        df_switch['SwitchPercentageHit_B{}'.format(block_type)] = switch_hit[:, b]
        df_switch['SwitchPercentageMiss_B{}'.format(block_type)] = switch_miss[:, b]
        df_switch['StayPercentageHit_B{}'.format(block_type)] = stay_hit[:, b]
        df_switch['StayPercentageMiss_B{}'.format(block_type)] = stay_miss[:, b]

    # multiply all columns by 100 to get percentages
    df_switch = df_switch * 100
    df_switch['subjectID'] = Subjects

    # drop participants with nan values
    df_switch = df_switch.dropna()

    return df_switch

def calculate_total_score(df, Subjects, BlockName='BlockVersion'):

    # calculate total score of participants in each block version
    blocks = np.sort(pd.unique(df[BlockName]))

    # Within each block version, find the total score of eachblock for each participant
    total_score = np.empty([len(Subjects), len(blocks)])
    total_score[:] = np.nan

    for subjIndex, subj in enumerate(Subjects):
        df_subj = df[df['subjectID'] == subj]

        for b in blocks:
            df_subj_b = df_subj[df_subj[BlockName] == b]
            score = pd.unique(df_subj_b['ScoreNormalized'])

            if len(score) > 1:
                # average the total score across blocks of same type for each participant
                score = np.mean(score)

            total_score[subjIndex, b] = score

    # Save in a pandas dataframe
    df_score = pd.DataFrame(total_score, columns=[f'TotalScore_B{int(block)}' for block in blocks])
    df_score['subjectID'] = Subjects

    # drop participants with nan in performance
    df_score = df_score.dropna()

    return df_score

def calculate_score_performance_switch_rates(df, df_merged, Subjects, blocks=None, block_name='BlockVersion', merge=True):

    """Calculate performance and switch rates for given subjects and blocks."""
    df_score = calculate_total_score(df, Subjects, BlockName=block_name)
    df_performance = performance_prl(df, Subjects, blocks, block_name)
    df_switch = calculate_switch_rates(df, Subjects, block_name)

    if merge:
        df_performance = df_performance.merge(df_merged, on='subjectID')
        df_switch = df_switch.merge(df_score, on='subjectID')
        df_descriptive = df_performance.merge(df_switch, on='subjectID')

        return df_descriptive

    else:
        return df_score, df_performance, df_switch


def separate_low_high_groups(df, col_name='g_z', category_name='G_Category'):

    """Separate subjects into low and high groups based on the specified column."""

    col_name_z = col_name + '_z'
    df[col_name_z] = zscore(df[col_name])

    mean_val = df[col_name_z].mean()

    df[category_name] = pd.np.where(df[col_name_z] > mean_val, 'High',
                                                 pd.np.where(df[col_name_z] < mean_val, 'Low', 'Normal'))

    return df


# Functions for PPC

def calculate_switches_PPC(df, observed_col, num_simulations=500):
    """Calculate the number of switches and their statistics for stable and volatile blocks."""
    df_stable = df[df['stabvol'] == -1]
    df_volatile = df[df['stabvol'] == 1]

    num_switches_stable_sim = np.sum(np.abs(np.diff(df_stable.iloc[:, :num_simulations], axis=0)), axis=0)
    num_switches_volatile_sim = np.sum(np.abs(np.diff(df_volatile.iloc[:, :num_simulations], axis=0)), axis=0)

    mean_switches_stable_sim = np.mean(num_switches_stable_sim)
    std_switches_stable_sim = np.std(num_switches_stable_sim)
    mean_switches_volatile_sim = np.mean(num_switches_volatile_sim)
    std_switches_volatile_sim = np.std(num_switches_volatile_sim)

    num_switches_stable_orig = np.sum(np.abs(np.diff(df_stable[observed_col], axis=0)), axis=0)
    num_switches_volatile_orig = np.sum(np.abs(np.diff(df_volatile[observed_col], axis=0)), axis=0)

    return {
        'mean_switches_stable_sim': mean_switches_stable_sim,
        'std_switches_stable_sim': std_switches_stable_sim,
        'mean_switches_volatile_sim': mean_switches_volatile_sim,
        'std_switches_volatile_sim': std_switches_volatile_sim,
        'num_switches_stable_orig': num_switches_stable_orig,
        'num_switches_volatile_orig': num_switches_volatile_orig
    }

def calculate_p_correct_PPC(df, dominant_col, observed_col, num_simulations=500):
    """Calculate P(Correct) for simulations and original data."""
    df_correct = (df.iloc[:, :num_simulations] == df[dominant_col].values[:, None]).astype(int)
    p_correct_ppc = np.sum(df_correct, axis=0) / df.shape[0]

    mean_p_correct = np.mean(p_correct_ppc)
    std_p_correct = np.std(p_correct_ppc)

    df_correct_orig = (df[observed_col].values == df[dominant_col].values).astype(int)
    p_correct_orig = np.sum(df_correct_orig) / df.shape[0]

    # return as a dataframe
    df_result = pd.DataFrame([{
        'mean_p_correct': mean_p_correct,
        'std_p_correct': std_p_correct,
        'p_correct_orig': p_correct_orig
    }])

    return df_result, p_correct_ppc
