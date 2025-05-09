import mne
import numpy as np
from config import epoch_start, epoch_end, baseline_start, baseline_end


def load_physiological_data(file):
    return mne.io.read_raw_brainvision(file, preload=True)


def prepare_scr_data(raw):
    raw.rename_channels({'GSR_MR_100_xx': 'SCR'})
    return raw.copy().pick_channels(['SCR'])


def filter_data(raw):
    # de Berker 2016: 1st order Butterworth low-pass filter, h_freq=5
    return raw.copy().filter(l_freq=None, h_freq=5, method='iir', iir_params={'order': 1, 'ftype': 'butter'})


def resample(raw, resampling_rate_type):
    if resampling_rate_type == 1:
        rate = 100  # resampling rate before filtering
    else:
        rate = 10  # resampling rate after filtering
    return raw.copy().resample(rate)


def construct_events(scr_filtered, trial_trigger, result_trigger):
    events, event_dict = mne.events_from_annotations(scr_filtered)
    event_id = dict(trial=trial_trigger, result=result_trigger)
    return events, event_id


def construct_epochs(scr_filtered, events, event_id, epochs_range_type):
    if epochs_range_type == 1:
        start, end = epoch_start, epoch_end
    elif epochs_range_type == 2:
        start, end = -5, 25  # Bach 2018
    else:
        start, end = -0.5, 30  # Bach 2010
    return mne.Epochs(scr_filtered, events, event_id, tmin=start, tmax=end, proj=True, baseline=(baseline_start, baseline_end), preload=True)


def construct_evoked_potentials(epochs):
    return epochs.average()


def baseline_correct(df, column, use_times_column=None):
    if use_times_column:
        baseline_mean = df.loc[(df['times'] >= baseline_start) & (df['times'] <= baseline_end), column].mean()
    else:
        baseline_mean = df.loc[(df.index >= baseline_start) & (df.index < baseline_end), column].mean()
    df[column] = df[column] - baseline_mean
    return df, baseline_mean


def construct_epoch_with_df(df_sub, time, sampling_rate):
    # define epoch length
    time_start = time + epoch_start
    time_end = time + epoch_end

    # checking the event length
    if time_start < df_sub['times'].min() or time_end > df_sub['times'].max():
        return None

    # create epoch
    epoch = df_sub.loc[(df_sub['times'] >= time_start) & (df_sub['times'] < time_end), 'data']

    # reset epoch index to time
    epoch_duration = len(epoch) / sampling_rate
    epoch.index = np.arange(epoch_start, epoch_start + epoch_duration, 1 / sampling_rate)

    epoch = epoch.to_frame()
    return epoch
