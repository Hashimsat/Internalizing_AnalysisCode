""" Regression analysis for the predator task
First simulates the reduced Bayesian model based on participant prediction errors
Then regression analysis is performed to compute the extent to which participant's behavior is in line with predictions
of the normative model

"""
import numpy as np

if __name__ == '__main__':

    import pandas as pd
    import random
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from RegVars_Predator import RegVars
    from RegressionChildPredator import RegressionChildPredator
    from functions.util_functions import calculate_spearman_corr

    # Control random number generator for reproducible results
    seed_val = 125
    random.seed(seed_val)
    np.random.seed(seed_val)

    # -----------------
    # 1. Load data
    # -----------------
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    df_pred = pd.read_csv(os.path.join(base_dir, "data/predator_task/df_predator_normative_added.csv"))
    df_params_all = pd.read_csv(os.path.join(base_dir, 'data/predator_task/df_predator_4exp_modelresults.csv'))
    data_folder = base_dir + "/Supplementary_Analysis/supplementary_data/predator_task/parameter_recovery/"

    # --------------
    # 2. Preprocess data
    # --------------
    Subjects = np.unique(df_pred['subjectID'])

    # Number of participants which we use for recovery
    n_subj = len(Subjects)

    # -----------------------------
    # 1. Simulate data for recovery
    # -----------------------------

    # Initialize regression variables
    reg_vars = RegVars()
    reg_vars.n_subj = n_subj
    reg_vars.n_sp = 60  # Number of random starting points for regression estimation
    reg_vars.usePrior = False
    reg_vars.n_ker = 6

    # Determine which parameters should be estimated
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
                           }  # true

    # Create regression-components list
    reg_vars.regressionComponents = [
        reg_vars.which_vars["beta_0"], reg_vars.which_vars["beta_1"],
        reg_vars.which_vars["beta_4"], reg_vars.which_vars["beta_5"],
        reg_vars.which_vars["beta_6"], reg_vars.which_vars["beta_7"],
        reg_vars.which_vars["beta_8"]
    ]

    # Select parameters according to selected variables and create data frame
    prior_columns = [reg_vars.beta_0, reg_vars.beta_1, reg_vars.beta_2, reg_vars.beta_3, reg_vars.beta_4,
                     reg_vars.beta_5, reg_vars.beta_6, reg_vars.beta_7, reg_vars.beta_8,
                     reg_vars.omikron_0, reg_vars.omikron_1,
                     reg_vars.lambda_0, reg_vars.lambda_1]

    # Create regression-object instance
    regression = RegressionChildPredator(reg_vars)

    # Get subject model parameters that we try to recover
    df_params = df_params_all.iloc[:n_subj]
    df_params.drop(columns=['Unnamed: 0'], inplace=True)

    # Simulate updates based on sampled parameters
    samples = regression.sample_data(df_params, allSubBehavData=df_pred)

    # ----------------------------
    # 2. Estimate regression model
    # ----------------------------

    results_w = regression.parallel_estimation(samples, prior_columns)

    # ------------------------------
    # 3. Save Data
    # ------------------------------
    name_recov = 'model_paramrec1_predator_exp4_recovered_params_seed125' + '.csv'
    results_w.to_csv(data_folder + name_recov, index=False)

    name_true = 'model_paramrec1_predator_exp4_actual_params_seed125' + '.csv'
    df_params['subjectID'] = Subjects[:n_subj]
    df_params.to_csv(data_folder + name_true, index=False)

    name_sim = 'model_paramrec1_predator_exp4_simulated_data_seed125' + '.csv'
    samples.to_csv(data_folder + name_sim, index=False)

    # ------------------------------
    # 4. Calculate Spearman correlation
    # ------------------------------
    corr_arr = [calculate_spearman_corr(results_w['beta_1'], df_params['beta_1']),
                calculate_spearman_corr(results_w['beta_4'], df_params['beta_4']),
                calculate_spearman_corr(results_w['beta_7'], df_params['beta_7']),
                calculate_spearman_corr(results_w['beta_5'], df_params['beta_5']),
                calculate_spearman_corr(results_w['beta_6'], df_params['beta_6']),
                calculate_spearman_corr(results_w['beta_8'], df_params['beta_8']),
                calculate_spearman_corr(results_w['omikron_0'], df_params['omikron_0']),
                calculate_spearman_corr(results_w['omikron_1'], df_params['omikron_1']),
                ]

    print(corr_arr)
