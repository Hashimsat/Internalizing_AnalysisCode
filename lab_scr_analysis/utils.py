import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append(os.path.abspath('/Users/hashim/PhD/PhD Project/Code and Data/Trait Anxiety Paper Figures/LabStudy/SCR_Analysis/'))
from config import settings_study_1, settings_study_2


def create_directories(directories):
    if isinstance(directories, str):
        directories = [directories]
    for directory in directories:
        if not os.path.isdir(directory):
            os.makedirs(directory)


def set_study_settings(study_number):
    if study_number == 1:
        return settings_study_1
    else:
        return settings_study_2


def standardize_sub_id(sub_id):
    sub_id_std = re.sub(r'Participant-', 'Participant_', sub_id)
    sub_id_std = re.sub(r'Participant_(\d)(\D|$)', r'Participant_0\1\2', sub_id_std)
    return sub_id_std

def exclude_participants(df, excluded_participants, study=2):
    if study == 1:
        study_exclusions = [participant for participant in excluded_participants if participant.startswith('study1')]
        study_exclusions = [participant.replace('study1_', '') for participant in study_exclusions]
    elif study == 2:
        study_exclusions = [participant for participant in excluded_participants if participant.startswith('study2')]
        study_exclusions = [participant.replace('study2_', '') for participant in study_exclusions]

    # Align subjectids

    df['subjectID'] = df['subjectID'].apply(standardize_sub_id)
    df = df[~df['subjectID'].isin(study_exclusions)]
    return df

def merge_with_exclusions(df, df_qnstotal_subset, excluded_participants, study=2):
    df = exclude_participants(df, excluded_participants, study)
    df = df.merge(df_qnstotal_subset, on='subjectID')
    return df

def set_plot_meta_data(title, label_x, label_y, path):
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.savefig(path)
    plt.close()


def plot_data(data, title, label_x, label_y, path):
    fig, ax = plt.subplots()
    ax.plot(data.times, data.get_data().T)
    set_plot_meta_data(title, label_x, label_y, path)


def plot_hist_data(data, bins, title, label_x, label_y, path):
    plt.figure()
    sns.histplot(data, bins=bins, kde=True)
    set_plot_meta_data(title, label_x, label_y, path)


def align_qns_factor_data(qns_totalscore,factor_scores,Gender_3=False):
    """ This function aligns the data from questionnaires and factor scores"""

    df_qnstotal_subset = qns_totalscore.rename(columns={'REF': 'subjectID', 'SD01_01': 'Age', 'SD02': 'Gender'})
    df_factor = factor_scores.rename(columns={'V1': 'subjectID'})
    df_qnstotal_subset.loc[df_qnstotal_subset['Gender'] == 1, 'Gender'] = -1
    df_qnstotal_subset.loc[df_qnstotal_subset['Gender'] == 2, 'Gender'] = 1

    # drop participants who have value of gender apart from 2 or 1
    if (Gender_3 == False):
        df_qnstotal_subset = df_qnstotal_subset[df_qnstotal_subset['Gender'] != 3]

    # merge with qns score and factor scores
    df_merged = df_qnstotal_subset.merge(df_factor, on='subjectID')

    return df_merged, df_qnstotal_subset, df_factor

def cm2inch(*tupl):
    """ This function convertes cm to inches

    Obtained from: https://stackoverflow.com/questions/14708695/
    specify-figure-size-in-centimeter-in-matplotlib/22787457

    :param tupl: Size of plot in cm
    :return: Converted image size in inches
    """

    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def label_subplots(f, texts, x_offset=-0.07, y_offset=0.015, fontsize=8, bold=False):
    """ This function labels the subplots

     Obtained from: https://stackoverflow.com/questions/52286497/
     matplotlib-label-subplots-of-different-sizes-the-exact-same-distance-from-corner

    :param f: Figure handle
    :param x_offset: Shifts labels on x-axis
    :param y_offset: Shifts labels on y-axis
    :param texts: Subplot labels
    """

    # Get axes
    axes = f.get_axes()

    # Cycle over subplots and place labels
    for a, l in zip(axes, texts):
        x = a.get_position().x0
        y = a.get_position().y1

        if (bold):
            f.text(x - x_offset, y + y_offset, l, size=fontsize,fontweight='bold')

        else:
            f.text(x - x_offset, y + y_offset, l, size=fontsize)


def label_axes(f,ax, texts, x_offset=-0.07, y_offset=0.015, fontsize=8, bold=False):
    """ This function labels the subplots

     Obtained from: https://stackoverflow.com/questions/52286497/
     matplotlib-label-subplots-of-different-sizes-the-exact-same-distance-from-corner

    :param f: Figure handle
    :param x_offset: Shifts labels on x-axis
    :param y_offset: Shifts labels on y-axis
    :param texts: Subplot labels
    """


    # Cycle over subplots and place labels
    for a, l in zip(ax, texts):
        x = a.get_position().x0
        y = a.get_position().y1

        if (bold):
            f.text(x - x_offset, y + y_offset, l, size=fontsize,fontweight='bold')

        else:
            f.text(x - x_offset, y + y_offset, l, size=fontsize)
