# definition of directories
directory = '/Users/kw/Documents/Studium2/Bachelorarbeit/data_analysis'
data_directory = '/Users/kw/Downloads/data'
data_folder_preprocessed_scr = data_directory + '/preprocessed_scr_data_both_studies'
plots_directory = directory + '/data_analysis_predator_lab_study/plots_both_studies'
plots_folder_raw = plots_directory + '/1_raw'
plots_folder_scr = plots_directory + '/2_scr'
plots_folder_epochs = plots_directory + '/3_epochs'
plots_folder_evoked_potentials = plots_directory + '/4_evoked_potentials'
plots_folder_scr_game = plots_directory + '/5_scr_and_game_data'

# general settings
game_data_variables = ['shock_block', 'hit_miss', 'predator_type', 'change_point'] # game_data_variables = ['shock_block', 'hit_miss'] #, 'predator_type', 'change_point']
epoch_start, epoch_end = -2, 10
baseline_start, baseline_end = -0.5, 0.0
excluded_participants = ['study1_Participant_03', 'study2_Participant_07', 'study2_Participant_11',
                         'study2_Participant_13', 'study2_Participant_21', 'study2_Participant_24',
                         'study2_Participant_30']

# specific settings for first lab study
settings_study_1 = {
    'num_participants': 41,
    'excluded_participants': [],
    'num_trials': 60,
    'num_blocks': 4,
    'sampling_rate_original': 5000,
    'trial_trigger': 8,
    'result_trigger': 128,
    'data_folder_scr': data_directory + '/physiological_data',
    'data_folder_game': data_directory + '/game_data',
    'data_folder_questionnaires': data_directory + '/questionnaire_data',
    'data_folder_shocks': data_directory + '/shock_data'}

# specific settings for second lab study
settings_study_2 = {
    'num_participants': 35,
    'num_trials': 80,
    'num_blocks': 4,
    'sampling_rate_original': 5000,
    'trial_trigger': 64,
    'result_trigger': 128,
    'data_folder_scr': data_directory + '/physiological_data_single_predator',
    'data_folder_game': data_directory + '/game_data_single_predator',
    'data_folder_questionnaires': data_directory + '/questionnaire_data_single_predator',
    'data_folder_shocks': data_directory + '/shock_data_single_predator'}


# definition of all directories
all_directories = [
    data_directory,
    data_folder_preprocessed_scr,
    settings_study_1['data_folder_scr'],
    settings_study_1['data_folder_game'],
    settings_study_1['data_folder_questionnaires'],
    settings_study_1['data_folder_shocks'],
    settings_study_2['data_folder_scr'],
    settings_study_2['data_folder_game'],
    settings_study_2['data_folder_questionnaires'],
    settings_study_2['data_folder_shocks'],
    plots_directory,
    plots_folder_raw,
    plots_folder_scr,
    plots_folder_epochs,
    plots_folder_evoked_potentials,
    plots_folder_scr_game
]

