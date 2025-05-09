""" Task-Agent Interaction: Interaction between reduced Bayesian model and predictive-inference task
    Uses participant prediction errors to simulate the reduced Bayesian model if sim=False
"""

import numpy as np
import pandas as pd
from functions.util_functions import circular_distance
import math



def task_agent_int(df, agent, agent_vars, sim=False):
    """ This function models the interaction between task and agent (RBM)

    :param df: Data frame with relevant data
    :param agent: Agent-object instance
    :param agent_vars: Agent-variables-object instance
    :param sim: Indicates if function is currently used for simulations or not
    :return: llh_mix, df_data: Negative log-likelihoods of mixture model and data frame with with simulation results
    """

    # Extract and initialize relevant variables
    # -----------------------------------------
    n_trials = len(df)  # number of trials

    mu = np.full([n_trials], np.nan)  # inferred mean of the outcome-generating distribution
    mu_bias = np.full([n_trials], np.nan)  # inferred mean with bucket bias
    a_hat = np.full(n_trials, np.nan)  # predicted update according to reduced Bayesian model
    omega = np.full(n_trials, np.nan)  # changepoint probability
    tau = np.full(n_trials, np.nan)  # relative uncertainty
    alpha = np.full(n_trials, np.nan)  # learning rate
    actual_update = np.full(n_trials, np.nan)  # actual update by participant in the game

    # Prediction error
    if not sim:
        delta = np.deg2rad(df['PredictionError']) # prediction error in radians
    else:
        delta = np.full(len(df), np.nan)

    # Initialize variables related to simulations
    sim_b_t = np.full(n_trials, np.nan)  # simulated prediction
    sim_z_t = np.full(n_trials, np.nan)  # simulated initial fire location
    sim_y_t = np.full(n_trials, np.nan)  # simulated shift of the fire
    sim_a_t = np.full(n_trials, np.nan)  # simulated update

    # -----------------
    # Cycle over trials
    # -----------------

    for t in range(0, n_trials-1):

        # Extract noise value from data for each trial
        agent.sigma = np.deg2rad(df['PredatorStd'][t])

        # Use hazard rate specific for each block if it differs between blocks, otherwise use hazard rate of 0.1
        if 'HazardLevel' in df.columns:
            if (df['HazardLevel'][t] == 0):
                agent.h = 0.10  # using the empirical hazard rate
            elif (df['HazardLevel'][t] == 1):
                agent.h = 0.16

        else:
            agent.h = 0.1

        # compute actual participant update on each trial
        actual_update[t] = np.deg2rad(circular_distance(df['torchAngle'][t + 1], df['torchAngle'][t]))

        # For first trial of new block
        if df['trialNumber'][t] == 1:

            # Initialize estimation uncertainty, relative uncertainty and changepoint probability
            agent.sigma_t_sq = agent_vars.sigma_0
            agent.tau_t = agent_vars.tau_0
            agent.omega_t = agent_vars.omega_0

            if sim:
                # Set initial fire location, prediction, and push
                sim_z_t[t] = agent_vars.mu_0
                sim_b_t[t] = agent_vars.mu_0
                sim_y_t[t] = 0.0

        # For all other trials
        else:
            if sim:
                # For simulations, we take the actual fire location
                sim_z_t[t] = df['torchAngle'][t]

                # We compute Shift as: (y_t := z_t - b_{t-1})
                sim_y_t[t] = sim_z_t[t] - sim_b_t[t]

                # # # Adjust for circular task. This is necessary because the model makes different trial-by-trial
                # # # predictions than participants, where we corrected for this already during preprocessing

                if (not np.isnan(sim_y_t[t])):
                    sim_y_t[t] = sim_y_t[t] % 360

        # Record relative uncertainty of current trial
        tau[t] = agent.tau_t

        # For all but last trials of a block:
        if (df['time'][t + 1] == df['time'][t]):

            # Sequential belief update
            if sim:
                # We calculate prediction error between actual predator location and model belief
                delta[t] = np.deg2rad(circular_distance(df['PredatorAngle'][t],np.rad2deg(sim_b_t[t])))
                agent.learn(delta[t], sim_b_t[t],0, df['PredatorMean'][t], 0)
            else:
                # We take the actual participant prediction error on that trial
                agent.learn(delta[t], df['torchAngle'][t], 0,df['PredatorMean'][t], 0)

            # Record updated belief
            mu[t] = agent.mu_t

            # Record predicted update according to reduced Bayesian model
            a_hat[t] = agent.a_t

            # Record changepoint probability
            omega[t] = agent.omega_t

            # Record learning rate
            alpha[t] = agent.alpha_t

    # Attach model variables to data frame
    df_data = pd.DataFrame(index=range(0, n_trials), dtype='float')
    df_data['a_t_hat'] = a_hat
    df_data['mu_t'] = mu
    df_data['mu_t_bias'] = mu_bias
    df_data['delta_t'] = delta
    df_data['omega_t'] = omega
    df_data['tau_t'] = tau
    df_data['alpha_t'] = alpha
    df_data['actual_update'] = actual_update
    df_data['BlockNumber'] = df['BlockNumber']
    df_data['trialNumber'] = df['trialNumber']
    df_data['HitMiss'] = df['HitMiss']


    if sim:

        # Save simulation-related variables
        df_data['sim_b_t'] = sim_b_t
        df_data['sim_a_t'] = sim_a_t
        df_data['sim_y_t'] = sim_y_t
        df_data['sim_z_t'] = sim_z_t
        df_data['sigma'] = df['PredatorStd']
        df_data['PredatorAngle'] = df['PredatorAngle']
        df_data['PredatorMean'] = df['PredatorMean']
        df_data['TorchActualPlacement'] = df['torchAngle']

    return df_data
