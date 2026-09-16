"""Unit and integration tests for the task-agent interaction."""

import os
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from rbmpy import AgentVars, AlAgent

from predator_rbm_model.al_task_agent_int_rbm import task_agent_int


def mock_learn(self, *args):
    """Mocks out the learning function of the agent.

    Parameters
    ----------
    self : object
        The instance of the class.
    args : tuple
        Additional arguments provided to the function.
    """

    self.tau_t = 0.2
    self.mu_t = 2
    self.omega_t = 0.7
    self.alpha_t = 0.35
    self.a_t = 0.6
    self.sigma_t_sq = 50


@pytest.fixture
def mock_task():
    """Fixture to create a mock task with deterministic behavior."""

    task = MagicMock()

    # Set deterministic values for task attributes
    task.cp = 0
    task.mu = 3.14
    task.x_t = 2.5
    task.catch_trial = 0
    task.new_block = 0
    task.kappa = 16

    # Make the methods do nothing
    task.sample_cp.return_value = None
    task.sample_mu.return_value = None
    task.sample_outcome.return_value = None
    task.sample_catch_trial.return_value = None

    return task


def test_first_trial(monkeypatch):
    """This function implements a unit test of the task-agent interaction
    in the first trial.

    The test covers the first trial with the new_block[t] case,
    where the agent is initialized.
    """

    # Replace the learn method
    monkeypatch.setattr("rbmpy.AlAgent.learn", mock_learn)

    # Load function input
    df, agent, agent_vars = load_default_input()

    # Extract trials for unit test
    df_trial = df[0:3].copy()

    # Run first trial
    df_data = task_agent_int(df_trial, agent, agent_vars)

    # Test function output
    assert df_data["a_t_hat"][0] == 0.6  # update mock function
    assert df_data["mu_t"][0] == 2  # belief mock function
    assert df_data["delta_t"][0] == 2.479061103708933  # actual prediction error
    assert df_data["omega_t"][0] == 0.7  # omega mock function
    assert df_data["tau_t"][0] == 0.5  # new block initialization
    assert df_data["alpha_t"][0] == 0.35  # alpha mock function
    assert df_data["actual_update"][0] == 2.576678135126356  # actual update
    assert df_data["BlockNumber"][0] == 4.0  # actual block number
    assert df_data["trialNumber"][0] == 1  # actual trial number
    assert df_data["HitMiss"][0] == 0.0  # actual miss


def test_first_trial_sim(monkeypatch):
    """This function implements a unit test of the task-agent interaction
    in the first trial with agent simulation.

    The test covers the first trial with the new_block[t] case,
    where the agent is initialized.
    """

    # Replace the learn method
    monkeypatch.setattr("rbmpy.AlAgent.learn", mock_learn)

    # Load function input
    df, agent, agent_vars = load_default_input()

    # Extract trials for unit test
    df_trial = df[0:3].copy()

    # Run first trial
    df_data = task_agent_int(df_trial, agent, agent_vars, sim=True)

    # Test function output
    assert df_data["a_t_hat"][0] == 0.6  # update mock function
    assert df_data["mu_t"][0] == 2  # belief mock function
    assert (
        df_data["delta_t"][0] == 2.478367537831948
    )  # simulated prediction error (computed in task_agent_int)
    assert df_data["omega_t"][0] == 0.7  # omega mock function
    assert df_data["tau_t"][0] == 0.5  # new block initialization
    assert df_data["alpha_t"][0] == 0.35  # alpha mock function
    assert df_data["actual_update"][0] == 2.576678135126356  # actual update
    assert df_data["BlockNumber"][0] == 4.0  # actual block number
    assert df_data["trialNumber"][0] == 1  # actual trial number
    assert df_data["HitMiss"][0] == 0.0  # actual miss

    # Todo: some of this seems redundant.
    #   Let's check if we really need these values in the function.
    assert df_data["sim_b_t"][0] == 0.0
    assert df_data["sigma"][0] == df["PredatorStd"][0]
    assert df_data["PredatorAngle"][0] == df["PredatorAngle"][0]
    assert df_data["PredatorMean"][0] == df["PredatorMean"][0]
    assert df_data["TorchActualPlacement"][0] == df["torchAngle"][0]

    # Todo: the function might still contain irrelevant (nan) output
    #   once cleaned, ensure we test all values
    # df_data['sim_b_t'] = sim_b_t
    # df_data['sim_a_t'] = sim_a_t
    # df_data['sim_z_t'] = sim_z_t
    # df_data['sigma'] = df['PredatorStd']
    # df_data['PredatorAngle'] = df['PredatorAngle']
    # df_data['PredatorMean'] = df['PredatorMean']
    # df_data['TorchActualPlacement'] = df['torchAngle']


def test_second_trial(monkeypatch):
    """This function implements a unit test of the task-agent interaction
    in the second trial.

    The test covers the second trial, which is a regular trial without
    new_block[t] or new_block[t+1].
    """

    # Replace the learn method
    monkeypatch.setattr("rbmpy.AlAgent.learn", mock_learn)

    # Load function input
    df, agent, agent_vars = load_default_input()

    # Extract trials for unit test
    df_trial = df[0:3].copy()

    # Run first trial
    df_data = task_agent_int(df_trial, agent, agent_vars)

    # Test function output
    assert df_data["a_t_hat"][1] == 0.6  # update mock function
    assert df_data["mu_t"][1] == 2  # belief mock function
    assert (
        df_data["delta_t"][1] == 0.1292757713418404
    )  # simulated prediction error (computed in task_agent_int)
    assert df_data["omega_t"][1] == 0.7  # omega mock function
    assert df_data["tau_t"][1] == 0.2  # new block initialization
    assert df_data["alpha_t"][1] == 0.35  # alpha mock function
    assert df_data["actual_update"][1] == 0.0036008776259946274  # actual update
    assert df_data["BlockNumber"][1] == 4.0  # actual block number
    assert df_data["trialNumber"][1] == 2  # actual trial number
    assert df_data["HitMiss"][1] == 1.0  # actual miss


def test_last_trial(monkeypatch):
    """This function implements a unit test of the task-agent interaction
    in the second trial.

    The test covers the second trial, which is a regular trial without
    new_block[t] or new_block[t+1].
    """

    # Replace the learn method
    monkeypatch.setattr("rbmpy.AlAgent.learn", mock_learn)

    # Load function input
    df, agent, agent_vars = load_default_input()

    # Extract trials for unit test
    df_trial = df[0:3].copy()

    # Run first trial
    df_data = task_agent_int(df_trial, agent, agent_vars)

    # Test function output
    assert np.isnan(df_data["a_t_hat"][2])
    assert np.isnan(df_data["mu_t"][2])
    assert df_data["delta_t"][2] == 1.137965859872557  # actual prediction error
    assert np.isnan(df_data["omega_t"][2])
    assert np.isnan(df_data["tau_t"][2])
    assert np.isnan(df_data["alpha_t"][2])
    assert np.isnan(df_data["actual_update"][2])
    assert df_data["BlockNumber"][2] == 4.0  # actual block number
    assert df_data["trialNumber"][2] == 3  # actual trial number
    assert df_data["HitMiss"][2] == 0.0  # actual miss


def test_integration_task_agent_int():
    """This function implements an integration test of the task-agent interaction
    across trials."""

    # Load function input
    df, agent, agent_vars = load_default_input()

    # Run all trials
    df_data = task_agent_int(df, agent, agent_vars, sim=False)

    # savename = 'data/testing/test_task_agent_int.pkl'
    # df_data.to_pickle(savename)

    # Load test data
    test_data = pd.read_pickle("data/testing/test_task_agent_int.pkl")

    # Test function output
    assert test_data.equals(df_data)


# Todo: - figure out if the sim=True is used at all
#       - second trial sim (once output completely figured out)
#       - last block trial (when figured out re: time variable)
#       - last block trial sim (once output completely figured out)


def load_default_input() -> tuple[Any, AlAgent, AgentVars]:
    """This function loads the default input to task_agent_int.py.

    Returns
    -------
    tuple of (pd.DataFrame, AlAgent, AgentVars)
        A tuple containing:
        - df: Preprocessed input data for the task.
        - agent: Initialized agent object based on agent variables.
        - agent_vars: Configuration object for agent-specific parameters.
    """

    # df input argument
    # -----------------

    # Load data
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    df = pd.read_csv(
        os.path.join(base_dir, "data/predator_task/df_predator_4expdata_combined.csv")
    )
    ids = df["subjectID.1"].unique()
    df = df[df["subjectID.1"] == ids[0]]

    # Agent and agent_vars input arguments
    # ------------------------------------

    # Initialize agent based on agent_vars
    agent_vars = AgentVars()
    agent_vars.circular = True
    agent_vars.max_x = 2 * np.pi
    agent_vars.mu_0 = 0
    agent_vars.u = np.exp(0)
    agent = AlAgent(agent_vars)

    return df, agent, agent_vars
