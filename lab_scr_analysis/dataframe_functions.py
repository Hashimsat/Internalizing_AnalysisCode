import itertools
import pandas as pd
import numpy as np
from scr_data_processing_functions import construct_epoch_with_df, baseline_correct


def create_dataframe(current_study, scr_filtered, df_game_data, path):
    sampling_rate_downsampled = scr_filtered.info['sfreq']

    # create dataframe for SCR times and data
    df = pd.DataFrame({
        'times': scr_filtered.times.flatten(),
        'data': scr_filtered._data.flatten(),
        'sampling_rate': sampling_rate_downsampled
    })

    # read marker file
    markers = pd.read_csv(path, comment=';', header=None, usecols=[0, 1, 2], names=['marker', 'stimulus', 'times'])
    markers = markers[markers['marker'].str.endswith('Stimulus')]

    # remove non-numeric characters
    markers['stimulus'] = markers['stimulus'].str.replace(r'\D', '', regex=True)

    # prepare columns for unique stimuli
    for value in markers['stimulus'].unique():
        df[value] = 0

    # set stimuli = 1 in df for corresponding markers
    for index, row in markers.iterrows():
        df_index = round(row['times'] * (sampling_rate_downsampled / current_study['sampling_rate_original']))
        df.loc[df_index, row['stimulus']] = 1

    # load game data
    df_game_data = df_game_data.copy()
    df_game_data['index'] = (df_game_data['block_number'] - 1) * current_study['num_trials'] + df_game_data['trial_number'] - 1

    # check number of triggers and reset trial numbers in case of missing triggers
    num_events = (df[str(current_study['result_trigger'])] == 1).sum()
    if num_events < current_study['num_trials'] * current_study['num_blocks']:  # for Participant 4, 10 and 11
        df_game_data['index'] = df_game_data['index'] - (current_study['num_trials'] * current_study['num_blocks'] - num_events)
        df_game_data = df_game_data[df_game_data['index'] >= 0]

    # add game data to dataframe
    df['index'] = df[str(current_study['result_trigger'])].cumsum() - 1
    df = df.join(df_game_data.set_index('index'), on='index', rsuffix='_game_data_sub')
    df['sub_id'] = df_game_data['sub_id'].iloc[0]

    return df


def create_epochs_dataframe(df, stimulus_column, apply_baseline_correct, *variables):
    var_dict = {'shock_block': {0: 'scream block', 1: 'shock block'},
                'hit_miss': {0: 'miss', 1: 'hit'},
                'predator_type': {1: 'cheetah', 2: 'leopard', 3: 'panther'},
                'change_point': {0: 'no change point', 1: 'change point'}}
    combinations = list(itertools.product(*[var_dict[var].keys() for var in variables]))
    epochs_data_list = []

    for sub_id in df['sub_id'].unique():
        df_sub = df[df['sub_id'] == sub_id]
        sampling_rate = df_sub['sampling_rate'].iloc[0]

        for i, combination in enumerate(combinations):
            conditions = [(df_sub[var] == val) for var, val in zip(variables, combination)]
            df_filtered = df_sub[np.logical_and.reduce(conditions)]
            events = df_filtered[df_filtered[stimulus_column] == 1]['times']

            if len(events) == 0:
                continue

            for event in events:
                epoch = construct_epoch_with_df(df_sub, event, sampling_rate)
                if epoch is None:
                    continue

                if apply_baseline_correct:
                    epoch, baseline_mean = baseline_correct(epoch, 'data')
                    epoch['percentage_change'] = 100 * epoch / baseline_mean
                else:
                    epoch['percentage_change'] = None

                # add epoch data to list
                for row in epoch.itertuples():
                    epochs_data_list.append({
                        'times': row.Index,
                        'data': row.data,
                        'percentage_change': row.percentage_change,
                        'sub_id': sub_id,
                        **{var: var_dict[var][val] for var, val in zip(variables, combination)}
                    })

    epochs_df = pd.DataFrame(epochs_data_list)
    return epochs_df


def create_average_epochs_dataframe(df, stimulus_column, *variables):
    epochs_df = create_epochs_dataframe(df, stimulus_column, True, *variables)

    # calculate average over all epochs for each combination of times, sub_id and variables
    average_epochs_df = epochs_df.groupby(['times', 'sub_id', *variables]).agg({
        'data': 'mean',
        'percentage_change': 'mean'
    }).reset_index()

    return average_epochs_df
