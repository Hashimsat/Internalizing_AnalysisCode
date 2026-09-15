# Functions for descriptive analysis of predator task data

import numpy as np
import pandas as pd
from scipy.stats import zscore
from functions.util_functions import circular_distance, CircularDistance_Array, BoundLR, safe_div_list, remove_nans_from_array

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

    # Add a warning if number of nans in EE is more than 10% of the total number of trials
    ee = remove_nans_from_array(ee)

    return np.mean(np.abs(ee))


def Estimation_Error(df, subjects, block_name=None):
    """Calculate mean Estimation Error for each subject, optionally split by block."""

    subjects = list(subjects)

    if block_name is None:
        # Overall EE
        EE = [
            calculate_estimation_error(df[df['subjectID'] == subj]) for subj in subjects
        ]

        result = create_dataframe(EE, 'EE', subjects)

    else:
        # Block-wise EE
        blocks = np.sort(pd.unique(df[block_name])).astype(int)

        EE = np.full((len(subjects), len(blocks)), np.nan)

        for subj_idx, subj in enumerate(subjects):
            for block_idx, block in enumerate(blocks):
                df_block = df[(df['subjectID'] == subj) & (df[block_name] == block)]

                EE[subj_idx, block_idx] = calculate_estimation_error(df_block)

        result = create_dataframe(EE, [f'EE_B{block}' for block in blocks], subjects)

    return result.dropna()
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
    PercentagePerseveration = (PerseverationRate / (TotalTrials - 1));

    df_pers = create_dataframe(PercentagePerseveration, 'Pers', subjList)

    return df_pers


def Single_Trial_LR(
    df,
    subjects,
    block_name=None,
    blocks=None,
    hit_miss_separation=False
):
    """Calculate median single-trial learning rates for each subject.

    If block_name is provided, learning rates are calculated separately
    for each block. Otherwise, they are calculated across all trials.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing subject and trial information.
    subjects : iterable
        Subject IDs to include.
    block_name : str, optional
        Column containing block identifiers. If None, calculate LR
        across all blocks.
    blocks : iterable, optional
        Specific blocks to include. If None, use all blocks found in df.
    hit_miss_separation : bool
        If True, return separate Hit and Miss DataFrames in addition
        to the overall LR DataFrame.

    Returns
    -------
    pd.DataFrame
        Overall median learning rates.
    tuple of pd.DataFrame
        If hit_miss_separation=True, returns Hit/Miss and Overall DataFrames.
    """

    subjects = list(subjects)

    # Determine whether we are calculating across blocks or within blocks
    by_block = block_name is not None

    if by_block:
        if blocks is None:
            blocks = np.sort(df[block_name].dropna().unique()).astype(int)
        else:
            blocks = list(blocks)

        n_blocks = len(blocks)

        lr_medians = {
            "Hit": np.full((len(subjects), n_blocks), np.nan),
            "Miss": np.full((len(subjects), n_blocks), np.nan),
            "Overall": np.full((len(subjects), n_blocks), np.nan),
        }

    else:
        lr_medians = {
            "Hit": np.full(len(subjects), np.nan),
            "Miss": np.full(len(subjects), np.nan),
            "Overall": np.full(len(subjects), np.nan),
        }

    # Calculate learning rates
    for subj_idx, subj in enumerate(subjects):

        df_subj = df[df["subjectID"] == subj]

        if by_block:

            for block_idx, block in enumerate(blocks):

                df_subj_block = df_subj[
                    df_subj[block_name] == block
                ]

                update, pe = process_updates_and_pe(df_subj_block)
                lr = calculate_learning_rates(update, pe)

                for lr_type in lr_medians:
                    values = remove_nans_from_array(lr[lr_type])
                    lr_medians[lr_type][subj_idx, block_idx] = np.median(values)

        else:

            update, pe = process_updates_and_pe(df_subj)
            lr = calculate_learning_rates(update, pe)

            for lr_type in lr_medians:
                values = remove_nans_from_array(lr[lr_type])
                lr_medians[lr_type][subj_idx] = np.median(values)

    # Create DataFrames
    if by_block:

        df_hit = create_dataframe(
            lr_medians["Hit"],
            [f"HB{block}" for block in blocks],
            subjects
        )

        df_miss = create_dataframe(
            lr_medians["Miss"],
            [f"MB{block}" for block in blocks],
            subjects
        )

        df_overall = create_dataframe(
            lr_medians["Overall"],
            [f"LR_B{block}" for block in blocks],
            subjects
        )

    else:

        df_hit = create_dataframe(
            lr_medians["Hit"],
            "HitLR",
            subjects
        )

        df_miss = create_dataframe(
            lr_medians["Miss"],
            "MissLR",
            subjects
        )

        df_overall = create_dataframe(
            lr_medians["Overall"],
            "LR",
            subjects
        )

    if hit_miss_separation:
        df_hit_miss = df_hit.merge(df_miss, on="subjectID")
        return df_hit_miss, df_overall

    return df_overall

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

        # Remove nans and claculate median
        RT_init[subjIndex] = np.median(remove_nans_from_array(df_subj['RTInitiation'].to_numpy()))
        RT_conf[subjIndex] = np.median(remove_nans_from_array(df_subj['RTConfirmation'].to_numpy()))

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
