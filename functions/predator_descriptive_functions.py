# Functions for descriptive analysis of predator task data

import numpy as np
import pandas as pd
from scipy.stats import zscore
from functions.util_functions import circular_distance, CircularDistance_Array, BoundLR, safe_div_list

# todo: can't we use the circ stats toolbox?
def calculate_estimation_error(df: pd.DataFrame) -> float:
    """Calculate the estimation error for a given block of data.

    Parameters
    ----------
    df : pd.DataFrame
        Task data.

    Returns
    -------
    float
        Computed estimation error.
    """

    # Select valid trials
    df_torchmoved = df[df['torchMoved'] == 1]

    # Compute estimation error
    ee = CircularDistance_Array(
        df_torchmoved['PredatorMean'].to_numpy(),
        df_torchmoved['torchAngle'].to_numpy()
    )
    return np.nanmean(np.abs(ee))

def estimation_error_overall(df: pd.DataFrame, subjects: np.ndarray) -> pd.DataFrame:
    """Calculate mean estimation error for each subject across blocks.

    Parameters
    ----------
    df : pd.DataFrame
        Predator data.
    subjects : np.ndarray
        Subject IDs.

    Returns
    -------
    pd.DataFrame
        Computed estimation errors.
    """

    # Initialize variables
    subj_list = []
    ee_overall = np.full(len(subjects), np.nan)  # 4 blocks in total for each task

    # Cycle over subjects
    for subjIndex, subj in enumerate(subjects):

        subj_list.append(subj)
        df_block = df[df['subjectID'] == subj]
        ee_overall[subjIndex] = calculate_estimation_error(df_block)

    # Create a dataframe of Estimation Error across all blocks
    df_ee = create_dataframe(ee_overall, 'EE', subj_list)
    df_ee = df_ee.dropna()

    return df_ee

# Todo: can this be merged with estimation_error_overall?
def EstimationError(df, Subjects, BlockName='BlockVersion'):
    """Calculate mean Estimation Error (EE) for each subject in each block"""

    Blocks = np.sort(pd.unique(df[BlockName])).astype(int)
    EE_overall = np.full([len(Subjects), len(Blocks)], np.nan)  # 4 blocks in total for each task
    subj_list = []

    for subjIndex, subj in enumerate(Subjects):
        subj_list.append(subj)
        for b in Blocks:
            df_block = df[(df['subjectID'] == subj) & (df[BlockName] == b)]
            EE_overall[subjIndex, b] = calculate_estimation_error(df_block)

    # Create a dataframe of Estimation Error across all blocks
    df_EE = create_dataframe(EE_overall, ['EE_B' + str(b) for b in Blocks],subj_list)
    df_EE = df_EE.dropna()

    return df_EE

def create_dataframe(data: np.ndarray, column_name: str, subj_list: list) -> pd.DataFrame:
    """Create a DataFrame for the given data and subject list.

    Parameters
    ----------
    data : np.ndarray
        Data to turn into the DataFrame.
    column_name : str
        ColumnName for the DataFrame.
    subj_list : list
        Subject IDs as a standard column of the DataFrame.

    Returns
    -------
    pd.DataFrame
        Newly created DataFrame.

    """

    # Create a DataFrame for the given data
    df = pd.DataFrame(data=data, columns=[column_name])
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Add subject ID as a standard column
    df['subjectID'] = subj_list

    return df

def process_updates_and_pe(df_subj):
    """Process updates and prediction errors for hit, miss, and overall for LR calculation."""
    Update = {"Miss": [], "Hit": [], "Overall": []}
    PE = {"Miss": [], "Hit": [], "Overall": []}

    for i in range(len(df_subj) - 1):
        if df_subj['HitMiss'].iloc[i] == 0:
            Update["Miss"].append(circular_distance(df_subj['torchAngle'].iloc[i + 1], df_subj['torchAngle'].iloc[i]))
            PE["Miss"].append(df_subj['PredictionError'].iloc[i])

        if df_subj['HitMiss'].iloc[i] == 1:
            Update["Hit"].append(circular_distance(df_subj['torchAngle'].iloc[i + 1], df_subj['torchAngle'].iloc[i]))
            PE["Hit"].append(df_subj['PredictionError'].iloc[i])

        if df_subj['torchMoved'].iloc[i] == 1 and df_subj['torchMoved'].iloc[i + 1] == 1:
            Update["Overall"].append(circular_distance(df_subj['torchAngle'].iloc[i + 1], df_subj['torchAngle'].iloc[i]))
            PE["Overall"].append(df_subj['PredictionError'].iloc[i])

    return Update, PE

def calculate_learning_rates(Update, PE):
    """Calculate bounded learning rates for hit, miss, and overall."""
    # convert to numpy arrays
    Update = {key: np.array(Update[key]) for key in Update}
    PE = {key: np.array(PE[key]) for key in PE}
    LR = {key: BoundLR(safe_div_list(Update[key], PE[key])) for key in Update}
    return LR

def SingleTrialLR(df, Subjects, Block=None, BlockName='BlockVersion', HitMissSeparation=False):
    """Calculate single trial learning rates for each subject for each block separately block."""

    if (Block is None):
        Block = np.sort(pd.unique(df[BlockName])).astype(int)

    # Initialize arrays and subject list
    LR_medians = {"Hit": np.full([len(Subjects), len(Block)], np.NaN),
                      "Miss": np.full([len(Subjects), len(Block)], np.NaN),
                      "Overall": np.full([len(Subjects), len(Block)], np.NaN)}
    subj_list = []

    for subjIndex, subj in enumerate(Subjects):
        df_subj = df.loc[(df['subjectID'] == subj)]
        subj_list = np.append(subj_list, subj)

        for b, blockV in enumerate(Block):
            df_subj_b = df_subj[df_subj[BlockName] == blockV]

            # Process updates and prediction errors
            Update, PE = process_updates_and_pe(df_subj_b)

            # Calculate learning rates
            LR = calculate_learning_rates(Update, PE)

            # Store medians
            LR_medians["Hit"][subjIndex, b] = np.nanmedian(LR["Hit"])
            LR_medians["Miss"][subjIndex, b] = np.nanmedian(LR["Miss"])
            LR_medians["Overall"][subjIndex, b] = np.nanmedian(LR["Overall"])

    # Create DataFrames
    df_hitmedian = create_dataframe(LR_medians["Hit"], ['HB' + str(element) for element in Block], subj_list)
    df_missmedian = create_dataframe(LR_medians["Miss"], ['MB' + str(element) for element in Block], subj_list)
    df_LR = create_dataframe(LR_medians["Overall"], ['LR_B' + str(element) for element in Block], subj_list)

    if (HitMissSeparation):
        df_merged = df_hitmedian.merge(df_missmedian, on='subjectID')
        return df_merged, df_LR
    else:
        return df_LR



def SingleTrialLR_overall(df,Subjects, HitMissSeparation = False):
    """Calculate overall median single trial learning rate across all blocks for each subject."""

    # Initialize arrays and subject list
    LR_medians = {"Hit": np.full(len(Subjects), np.NaN),
                  "Miss": np.full(len(Subjects), np.NaN),
                  "Overall": np.full(len(Subjects), np.NaN)}
    subj_list = []

    for subjIndex, subj in enumerate(Subjects):
        df_subj = df.loc[(df['subjectID'] == subj)]
        subj_list = np.append(subj_list, subj)

        # Process updates and prediction errors
        Update, PE = process_updates_and_pe(df_subj)

        # Calculate learning rates
        LR = calculate_learning_rates(Update, PE)

        # Store medians
        LR_medians["Hit"][subjIndex] = np.nanmedian(LR["Hit"])
        LR_medians["Miss"][subjIndex] = np.nanmedian(LR["Miss"])
        LR_medians["Overall"][subjIndex] = np.nanmedian(LR["Overall"])

    # Create DataFrames
    df_hitmedian = create_dataframe(LR_medians["Hit"], 'HitLR', subj_list)
    df_missmedian = create_dataframe(LR_medians["Miss"], 'MissLR', subj_list)
    df_LR = create_dataframe(LR_medians["Overall"], 'LR', subj_list)


    if (HitMissSeparation):
        df_merged = df_hitmedian.merge(df_missmedian, on='subjectID')
        return df_merged, df_LR
    else:
        return df_LR


def PerseverationRate_overall(df, Subjects):
    """Compute perseveration rate for each subject."""

    # Initialize arrays and subject list
    PerseverationRate = np.full(len(Subjects), np.NaN)
    TotalTrials = np.full(len(Subjects), np.NaN)
    subjList = []

    for subjIndex, subj in enumerate(Subjects):
        subjList = np.append(subjList, subj)
        df_subj = df.loc[(df['subjectID'] == subj) ]
        PresNumber = 0

        for i in range(len(df_subj) - 1):

            if (
                    (df_subj['torchAngle'].iloc[i + 1] <= df_subj['torchAngle'].iloc[i] + 2.5)
                    & (df_subj['torchAngle'].iloc[i + 1] >= df_subj['torchAngle'].iloc[i] - 2.5)
                    & (df_subj['torchMoved'].iloc[i + 1] == 1)
            ):  # (does not include no movement trials, and provides a range of 5 degrees around previous location)

                PresNumber += 1 # count number of perseveration trials

        PerseverationRate[subjIndex] = PresNumber
        TotalTrials[subjIndex] = len(df_subj)

    # calculate percentage perseveration
    PercentagePerseveration = (PerseverationRate / (TotalTrials - 1))

    df_pers = create_dataframe(PercentagePerseveration, 'Pers', subjList)

    return df_pers


def RT_InitConf_overall(df: pd.DataFrame, Subjects: np.ndarray) -> tuple:
    """Calculate median reaction time for initiation and confirmation for each subject.

    This function processes a DataFrame of reaction times for multiple subjects,
    calculating the median reaction time for both initiation and confirmation actions.
    The results are stored in separate DataFrames for initiation and confirmation
    reaction times.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing reaction times and subject IDs.
    Subjects : np.ndarray
        Array of unique subject IDs for which the reaction times are to be calculated.

    Returns
    -------
    tuple of pd.DataFrame
        A tuple containing two DataFrames:
        - The first DataFrame contains median initiation reaction times for each subject.
        - The second DataFrame contains median confirmation reaction times for each subject.
    """


    # Initialize arrays and subject list
    RT_init = np.full(len(Subjects), np.NaN)
    RT_conf = np.full(len(Subjects), np.NaN)
    subjList = []

    # Cycle over subjects
    for subjIndex, subj in enumerate(Subjects):
        subjList = np.append(subjList, subj)

        df_subj = df.loc[(df['subjectID'] == subj) ]

        RT_init[subjIndex] = np.nanmedian(df_subj['RTInitiation'])
        RT_conf[subjIndex] = np.nanmedian(df_subj['RTConfirmation'])

    # Create DataFrames
    df_init = create_dataframe(RT_init, 'RT', subjList)
    df_conf = create_dataframe(RT_conf, 'RT', subjList)

    return df_init, df_conf


def combine_descriptive_with_factor_scores(df: pd.DataFrame, df_fs: pd.DataFrame, lowhighanx: bool=False) -> pd.DataFrame:
    """Combine descriptive statistics with questionnaire data.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the descriptive statistics.
    df_fs : pd.DataFrame
        A DataFrame containing the factor scores.
    lowhighanx : bool, default=False
        A flag indicating whether to extract and return an additional
        subset of the merged DataFrame for subjects categorized as
        "High" and "Low" in anxiety.

    Returns
    -------
    pd.DataFrame or tuple of pd.DataFrame
        Returns the merged DataFrame. If `lowhighanx` is True, returns a
        tuple where the first element is the full merged DataFrame and the
        second is the subset for "High" and "Low" anxiety categories.
    """


    # Combine the two DataFrames
    df_combined = df.merge(df_fs, on='subjectID')

    # Convert to DataFrame
    df_combined = pd.DataFrame(df_combined)  # todo: is this necessary?
    df_combined_LowHighAnx = df_combined[df_combined['G_Category'].isin(['High', 'Low'])]

    if lowhighanx:
        return df_combined, df_combined_LowHighAnx

    else:
        return df_combined

# Helper function to z-score columns
def zscore_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Compute z-scores for specified columns and append them to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data for which Z-scores are computed.
    columns : list
        List of column names (as strings) in the DataFrame for which to compute
        the z-scores.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with additional columns containing the calculated
        z-scores for the specified columns.

    """
    for col in columns:
        # Replace '.' in column names with '_' for the z-scored column
        col_z = col.replace('.', '') + '_z'
        df[col_z] = zscore(df[col])
    return df
