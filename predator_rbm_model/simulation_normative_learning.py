# Simulation of normative learning, used in plotting normative learning example in figure 1 or in figure 5

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from functions.util_functions import CircularDistance_Array, safe_save_dataframe
from rbmpy.agent_rbm.AgentRbm import AlAgent
from rbmpy.agent_rbm.AgentVarsRbm import AgentVars

def simulate_normative_learning(fig_no, save_data=False, plot=False):
    """
    Simulate normative learning for Figure 1 or Figure 5.

    Parameters
    ----------
    fig_no : int
        Figure number. Must be either 1 or 5.
    save_data : bool, default=False
        Whether to save the simulated data as a CSV file.
    plot : bool, default=False
        Whether to plot the normative learning curve.

    Returns
    -------
    pd.DataFrame
        Simulated normative learning data.
    """

    if fig_no not in [1, 5]:
        raise ValueError("fig_no must be either 1 or 5.")

    # --------------------------------
    # Set up figure-specific parameters
    # --------------------------------

    if fig_no == 1:
        n_range = 300
        start = 0
        end = n_range / 2
        tau_t = 0.1

    else:  # Figure 5
        n_range = 600
        start = -180
        end = 180
        tau_t = 0.5

    # ----------------
    # Set up the agent
    # ----------------

    agent_vars = AgentVars()

    # Set agent variables
    agent_vars.h = 0.1
    agent_vars.s = 1
    agent_vars.u = np.exp(0)
    agent_vars.q = 0
    agent_vars.sigma_H = 0
    agent_vars.mu_0 = 0
    agent_vars.max_x = 2 * math.pi
    agent_vars.sigma = np.deg2rad(10)
    agent_vars.sigma_0 = np.deg2rad(20) ** 2
    agent_vars.circular = True

    # ----------------------------
    # Initialize simulation arrays
    # ----------------------------

    pe = np.linspace(start, end, n_range)

    alpha = np.full(n_range, np.nan)
    b_t = np.full(n_range, np.nan)
    a_t = np.full(n_range, np.nan)
    cpp = np.full(n_range, np.nan)

    # -----------------
    # Run simulation
    # -----------------

    for i, prediction_error in enumerate(pe):

        agent = AlAgent(agent_vars)
        agent.tau_t = tau_t

        agent.learn(
            np.deg2rad(prediction_error),
            0,
            False,
            0,
            high_val=0
        )

        alpha[i] = agent.alpha_t
        b_t[i] = np.rad2deg(agent.mu_t)
        a_t[i] = np.rad2deg(agent.a_t)
        cpp[i] = agent.omega_t

    # ---------------------------------------
    # Compute circular update if needed
    # ---------------------------------------

    update = CircularDistance_Array(b_t[1:], b_t[:-1])

    # ----------------
    # Create dataframe
    # ----------------

    if fig_no == 1:
        df_norm = pd.DataFrame({
            'Prediction Error': pe,
            'Learning Rate': alpha,
            'Belief': b_t
        })

    else:  # Figure 5
        df_norm = pd.DataFrame({
            'a_t': a_t,
            'alpha_i': alpha,
            'PE': pe,
            'CPP': cpp
        })

    # ----------------
    # Save data
    # ----------------

    if save_data:
        save_path = (
            f'../data/predator_task/'
            f'simulated_normative_learning_fig{fig_no}.csv'
        )

        safe_save_dataframe(df_norm, save_path)

    # ----------------
    # Plot
    # ----------------

    if plot:
        plt.figure()

        plt.plot(
            pe,
            alpha,
            color="#249886",
            linewidth=2,
            label='Normative\nLearning'
        )

        plt.ylim([-0.02, 1.1])
        plt.ylabel('Learning Rate')
        plt.xlabel('Prediction Error')
        plt.legend(loc=2)
        plt.show()

    return df_norm

# ----------
# Run this function to simulate normative learning for Figure 1 or Figure 5
# ---------
if __name__ == "__main__":
    # Simulate for Figure 1
    df_norm_fig1 = simulate_normative_learning(fig_no=1, save_data=True, plot=False)

    # Simulate for Figure 5
    df_norm_fig5 = simulate_normative_learning(fig_no=5, save_data=True, plot=False)