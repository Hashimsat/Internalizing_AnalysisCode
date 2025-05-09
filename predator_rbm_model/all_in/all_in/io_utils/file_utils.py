import os
import re
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd


def get_df_subj(df: pd.DataFrame, i: int) -> pd.DataFrame:
    """This function creates a subject-specific data frame with adjusted index.

    Parameters
    ----------
    df : pd.DataFrame
        Subject data frame.
    i : int
        Subject number.

    Returns
    -------
    pd.DataFrame
        Index-adjusted subject-specific data frame (df_subj).
    """

    df_subj = df[(df["subj_num"] == i + 1)].copy()
    df_subj = df_subj.reset_index(drop=True)  # adjust index

    return df_subj


def load_data(f_names: list[Path], expected_n_trials: int = 400) -> pd.DataFrame:
    """This function loads the adaptive learning BIDS data and checks if they are complete.

    Parameters
    ----------
    f_names : list[Path]
        List with all file names.
    expected_n_trials : int
        Expected number of trials.

    Returns
    -------
    pd:DataFrame
        Data frame that contains all data.
    """

    # Initialize arrays
    n_trials = np.full(len(f_names), np.nan)  # number of trials

    # Put data in data frame
    all_data = np.nan
    for i in range(0, len(f_names)):

        if i == 0:

            # Load data of participant 0
            all_data = pd.read_csv(f_names[0], sep="\t", header=0)
            new_data = all_data
        else:

            # Load data of participant 1,..,N
            new_data = pd.read_csv(f_names[i], sep="\t", header=0)

        # Count number of respective trials
        n_trials[i] = len(new_data)
        new_data["trial"] = np.int64(np.arange(n_trials[i]))

        # Indicate if less than expected N trials
        if n_trials[i] < expected_n_trials:
            print("Only %i trials found" % n_trials[i])

        # Append data frame
        if i > 0:
            all_data = pd.concat([all_data, new_data], ignore_index=True)

    return all_data


def sorted_nicely(input_list: list) -> list:
    """This function sorts the given iterable in the way that is expected.

    Obtained from:
    https://arcpy.wordpress.com/2012/05/11/sorting-alphanumeric-strings-in-python

    Parameters
    ----------
    input_list : list
        The iterable to be sorted.

    Returns
    -------
    list
        Sorted iterable.
    """

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(input_list, key=alphanum_key)


def get_file_paths(folder_path: Path, identifier: str) -> Path:
    """This function extracts the file path.

    Parameters
    ----------
    folder_path : Path
        Relative path to current folder.
    identifier : Path
        Identifier for file of interest.

    Returns
    -------
    Path
        Absolute path to file (file_paths).
    """

    file_paths = []
    for path, subdirs, files in os.walk(folder_path):

        for name in files:
            if fnmatch(name, identifier):
                file_paths.append(os.path.join(path, name))

    return file_paths
