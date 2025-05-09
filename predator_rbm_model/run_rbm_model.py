""" Regression analysis for the predator task
First simulates the reduced Bayesian model based on participant prediction errors
Then regression analysis is performed to compute the extent to which participant's behavior is in line with predictions
of the normative model

"""
import numpy as np

if __name__ == '__main__':

    import pandas as pd
    from RegVars_Predator import RegVars
    from RegressionChildPredator import RegressionChildPredator
    from al_simulation_rbm import simulation
    import random
    from functions.util_functions import qns_factor_preprocessing
    import os

    # Control random number generator for reproducible results
    seed_val = 125
    random.seed(seed_val)
    np.random.seed(seed_val)

    # -----------------
    # 1. Load data
    # -----------------
    df = pd.read_csv("../data/predator_task/df_predator_4expdata_combined.csv")
    qns_totalscore = pd.read_csv("../data/factor_analysis/questionnaires_totalscores_subscales.csv", sep=',')
    factor_scores = pd.read_csv('../data/factor_analysis/factor_scores.csv')

    data_folder = "../data/predator_task/"

    # --------------
    # 2. Preprocess data
    # --------------
    # Preprocess qns data
    df_qnstotal_subset,_ = qns_factor_preprocessing(qns_totalscore, factor_scores, merge_both=False, drop_non_binary=True)

    # Merge predator data with qns data,so we only have participants that completed both the questionnaire and the tasks
    df = df.merge(df_qnstotal_subset, on='subjectID')

    # Extract subject IDs
    Subjects = pd.unique(df['subjectID'])

    # sort values by block number and trial number for each participant
    df = df.sort_values(by=['subjectID', 'BlockNumber', 'trialNumber']).reset_index()

    # Exclude participant with nan as subjectID if they exist
    df = df.dropna(subset=['subjectID'])

    # Extract participant information
    all_id = list(set(df['subjectID']))  # ID for each participant
    n_subj = len(all_id)  # number of subjects

    # -----------------
    # 3. Simulate the reduced Bayesian model based on participant prediction errors
    # -----------------
    # Set parameters for simulation
    model_params = pd.DataFrame(columns=['h', 's', 'u', 'q', 'sigma_H', 'd',
                                         'subjId', 'PredatorType'], index=range(n_subj))

    model_params['h'] = 0.1
    model_params['s'] = 1
    model_params['u'] = 0
    model_params['q'] = 0
    model_params['sigma_H'] = 0
    model_params['d'] = 0.0
    model_params['subjId'] = all_id

    # Run simulation
    df_sim = simulation(df, model_params, n_subj, sim=False)

    # Save simulation data
    name = 'normative_predator_sim.csv'
    savename = os.path.join(data_folder, name)
    # df_sim.to_csv(savename)

    # -----------------
    # 4. Extract and preprocess simulation data for running subsequent regression
    # -----------------

    df['omega_t'] = df_sim['omega_t'].copy()
    df['tau_t'] = df_sim['tau_t'].copy()
    df['a_t'] = df_sim['actual_update'].copy()
    df_sim = df_sim.rename(columns={"a_t_hat": "a_t"})

    # Convert PE to radians
    df['PredictionError'] = np.deg2rad(df['PredictionError'])
    df_sim['delta_t'] = np.deg2rad(df_sim['delta_t'])

    # Recode valence to [-1,1] format
    df.loc[df['HitMiss'] == 0, 'HitMiss'] = -1

    # Check if prediction errors are equal in both simulations and actual data
    same_delta = df['PredictionError'].equals(df_sim['delta_t'])
    print("All prediction errors equal: " + str(same_delta))
    print("Make sure that the model parameters are set to normative values")


    # --------------------------
    # 5. Run regression analysis
    # --------------------------

    # Define regression variables

    # add 'subj_num' column
    df['subj_num'] = df.groupby('subjectID').ngroup() + 1
    n_subj = len(np.unique(df['subj_num']))  # number of subjects

    reg_vars = RegVars()
    reg_vars.n_subj = n_subj  # number of subjects
    reg_vars.n_ker = 6  # number of kernels for estimation
    reg_vars.n_sp = 50  # 50  # number of random starting points
    reg_vars.rand_sp = True  # use random starting points
    reg_vars.use_prior = False  # use of prior for estimations
    reg_vars.seed = seed_val  # set seed for reproducibility

    # Run mixture model
    # -----------------

    # Free parameters for model with Fixed Lr, Adaptive LR and Valence
    reg_vars.which_vars = {reg_vars.beta_0: True,  # Intercept
                           reg_vars.beta_1: True,  # delta_t
                           reg_vars.beta_2: False,  # delta_t * tau_t
                           reg_vars.beta_3: False,  # delta_t * omega_t
                           reg_vars.beta_4: True,  # delta_t * alpha (combined learning rate)
                           reg_vars.beta_5: True,  # interaction PE:HazardRateLevel
                           reg_vars.beta_6: True,  # interaction PE:StochasticityLevel
                           reg_vars.beta_7: True,  # interaction PE:HitMiss
                           reg_vars.beta_8: True,  # interaction PE:HR:Stoch

                           reg_vars.omikron_0: True,  # motor noise  #true
                           reg_vars.omikron_1: True,  # learning rate noise
                           reg_vars.lambda_0: False,  # mixture weight, #true
                           reg_vars.lambda_1: False
                           }

    # Select parameters according to selected variables and create data frame
    prior_columns = [reg_vars.beta_0, reg_vars.beta_1, reg_vars.beta_2, reg_vars.beta_3, reg_vars.beta_4,
                     reg_vars.beta_5, reg_vars.beta_6, reg_vars.beta_7, reg_vars.beta_8,
                     reg_vars.omikron_0, reg_vars.omikron_1,
                     reg_vars.lambda_0, reg_vars.lambda_1]

    # Initialize regression object
    predator_regression = RegressionChildPredator(reg_vars)  # regression object instance

    # Drop nans if any exist and initialize group column (group not used in this analysis)
    df_pred = df.dropna(subset=['PredictionError', 'a_t']).reset_index()
    df_pred['group'] = 1

    # Run regression
    # --------------
    results_df = predator_regression.parallel_estimation(df_pred, prior_columns)

    # Save results
    name = 'df_predator_4exp_modelresults_redone.csv'
    savename = os.path.join(data_folder, name)
    results_df.to_csv(savename)


