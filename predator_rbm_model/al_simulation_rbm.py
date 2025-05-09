""" Simulations Reduced Bayesian Model: Run simulations across whole data set, e.g., for posterior predictive checks
    Extracts data for each participant and uses it to simulate the rbm model
 """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm
from rbm_analyses.rbm_analyses.agent_rbm.AgentRbm import AlAgent
from rbm_analyses.rbm_analyses.agent_rbm.AgentVarsRbm import AgentVars
from al_task_agent_int_rbm import task_agent_int
import math


def simulation(df_exp, df_model, n_subj, plot_data=False, sim=True):
    """ This function simulates data using the mixture model

    :param df_exp: Data frame containing participant data
    :param df_model: Data frame containing model parameters
    :param n_subj: Number of participants
    :param plot_data: Indicates if single-trial plots for updates and predictions should be generated
    :param sim: Indicates if prediction errors are simulated or not
    :return: sim_est_err, sim_pers_prob, df_sim, true_params: Simulated estimation errors,
             simulated perseveration probability, set of true parameters
    """

    # Inform user
    sleep(0.1)
    print('\nModel simulation:')
    sleep(0.1)

    # Initialize progress bar
    pbar = tqdm(total=n_subj)

    # Agent variables object
    agent_vars = AgentVars()

    # Initialize data frame for data that will be recovered
    df_sim = pd.DataFrame()

    # Cycle over participants
    Subjects = np.sort(pd.unique(df_exp['subjectID']))

    # -----------------------
    for i,subj in enumerate(Subjects):

        # Extract subject-specific data frame
        df_subj = df_exp.loc[(df_exp['subjectID'] == subj)]

        # reset index
        df_subj = df_subj.reset_index(drop=True)

        # Extract model parameters from model data frame
        sel_coeffs = df_model[df_model['subjId'] == subj].copy()

        # Select relevant variables from parameter data frame
        sel_coeffs = sel_coeffs[['h', 's', 'u', 'q', 'sigma_H']].values.tolist()[0]

        # Set agent variables of current participant
        agent_vars.h = sel_coeffs[0]
        agent_vars.s = sel_coeffs[1]
        agent_vars.u = np.exp(sel_coeffs[2])
        agent_vars.q = sel_coeffs[3]
        agent_vars.sigma_H = sel_coeffs[4]

        # Initialize agent variables for the RBM agent
        agent_vars.sigma = np.deg2rad(20)
        agent_vars.sigma_0 = (np.deg2rad(20)) ** 2
        agent_vars.mu_0 = 0
        agent_vars.max_x = 2 * math.pi
        agent_vars.circular = True

        # Agent object
        agent = AlAgent(agent_vars)

        # Run task-agent interaction
        df_data = task_agent_int(df_subj, agent, agent_vars, sim=sim)

        # Record subject number
        df_data['subj_num'] = i+1
        df_data['subjectID'] = subj

        # Add data to data frame
        df_sim = pd.concat([df_sim, df_data], ignore_index=True)

        # Update progress bar
        pbar.update(1)

        # Close progress bar
        if i == n_subj - 1:
            pbar.close()

    return df_sim

