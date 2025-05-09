# Function for descriptive analysis of the reversal learning task data

import numpy as np
import pandas as pd

def performance_prl(df_mega,subjects, blocks=None, block_name='BlockVersion'):

    if (blocks is None):
        blocks = np.sort(pd.unique(df_mega[block_name]))

    correct_percentage = np.full([len(subjects), len(blocks)],np.nan)
    subjList = []

    for subjIndex, subj in enumerate(subjects):
        subjList.append(subj)
        df_subj = df_mega[df_mega['subjectID'] == subj]

        for b in blocks:   # separate into stable and volatile phases
            df_subj_b = df_subj[df_subj['BlockVersion'] == b]
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