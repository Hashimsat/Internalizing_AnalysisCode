from itertools import combinations
from scipy.stats import zscore
from config import *
from data_processing.scr_data_processing_functions import *
from data_processing.dataframe_functions import create_dataframe
from plot_scr_functions import *


def process_file(current_study, sub_id, df_game_data_sub):
    # load and prepare SCR data
    file_path = current_study['data_folder_scr'] + '/' + sub_id.split('_', 1)[-1]
    raw = load_physiological_data(file_path + '.vhdr')
    raw_scr = prepare_scr_data(raw)

    # resample and filter SCR data
    scr_downsampled_before_filtering = resample(raw_scr, 1)
    scr_filtered = filter_data(scr_downsampled_before_filtering)

    # z-score SCR data and reset to correct file format
    scr_data_array = scr_filtered.get_data()
    scr_z_scored_array = zscore(scr_data_array, axis=1)
    scr_z_scored = scr_filtered.copy()
    scr_z_scored._data = scr_z_scored_array

    # downsample z-scored data
    scr_downsampled = resample(scr_z_scored, 2)

    # construct events
    events, event_id = construct_events(scr_downsampled, current_study['trial_trigger'], current_study['result_trigger'])  # event trigger

    # construct epochs and evoked potentials
    epochs = construct_epochs(scr_downsampled, events, event_id, 1)
    evoked_potentials = construct_evoked_potentials(epochs)

    # create own dataframe
    df = create_dataframe(current_study, scr_downsampled, df_game_data_sub, file_path + '.vmrk')

    # plot data
    plot_raw_data(raw, plots_folder_raw, sub_id + '_raw')
    plot_raw_data_psd(raw, plots_folder_raw, sub_id + '_raw_psd')
    plot_raw_scr(raw_scr, plots_folder_scr, sub_id + '_scr_raw')
    plot_scr_filtered(scr_filtered, plots_folder_scr, sub_id + '_scr_filtered')
    plot_scr_filtered(scr_z_scored, plots_folder_scr, sub_id + '_scr_z_scored')
    plot_scr_filtered(scr_downsampled, plots_folder_scr, sub_id + '_scr_downsampled')
    # plot_events(scr_downsampled, events, event_id, plots_folder_epochs, sub_id + '_events')
    plot_epochs(epochs, plots_folder_epochs, sub_id + '_epochs')
    plot_evoked_potentials(evoked_potentials, len(epochs), plots_folder_evoked_potentials,
                           sub_id + '_evoked_potentials')

    # plot data in relation to game data
    for r in range(2, 4):
        for combination in combinations(game_data_variables, r):
            plot_multiple_evoked_potentials(df, plots_folder_evoked_potentials, str(current_study['result_trigger']), *combination)

    return df
