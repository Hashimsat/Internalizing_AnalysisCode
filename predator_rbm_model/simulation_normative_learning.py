# Simulation of normative learning, used in plotting normative learning example in figure 1 or in figure 5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from functions.util_functions import CircularDistance_Array
from rbmpy.agent_rbm.AgentRbm import AlAgent
from rbmpy.agent_rbm.AgentVarsRbm import AgentVars

# --------------
# Set up the agent
# -------------
# Agent object instance
agent_vars = AgentVars()
agent = AlAgent(agent_vars)

# ------------------
# Initialize simulation params
# -----------------
fig_no = 1
save_csv = True
plot = False

if fig_no == 1:
    n_range = 300
    start = 0
    end = n_range/2
    tau_t = 0.1

else:
    n_range = 600
    start = -180
    end = 180
    tau_t = 0.5

pe = np.linspace(start, end, n_range)
alpha = np.full(n_range, np.nan)
b_t = np.full(n_range, np.nan)
prediction = np.full(n_range, np.nan)
a_t = np.full(n_range, np.nan)
cpp = np.full(n_range, np.nan)
bt = 0

# Cycle over prediction error range
# ---------------------------------

for i in range(0, n_range):

    # Set agent variables
    agent_vars.h = 0.1
    agent_vars.s = 1
    agent_vars.u = np.exp(0)
    agent_vars.q = 0
    agent_vars.sigma_H = 0
    agent_vars.mu_0 = 0
    agent_vars.max_x = 2 * math.pi
    agent_vars.sigma = np.deg2rad(10)
    agent_vars.sigma_0 = (np.deg2rad(20))**2
    agent_vars.circular = True

    # Normative model
    agent = AlAgent(agent_vars)
    agent.tau_t = tau_t
    agent.learn(np.deg2rad(pe[i]), 0, False,0, high_val=0)
    alpha[i] = agent.alpha_t
    b_t[i] = np.rad2deg(agent.mu_t)
    a_t[i] = np.rad2deg(agent.a_t)
    cpp[i] = agent.omega_t


Update = CircularDistance_Array(b_t[1:], b_t[0:len(b_t)-1])

# save in dataframe
if fig_no == 1:
    df_norm = pd.DataFrame({'Prediction Error': pe, 'Learning Rate': alpha, 'Belief': b_t})

else:
    df_norm = pd.DataFrame({'a_t': a_t, 'alpha_i': alpha, 'PE': pe, 'CPP': cpp})

if save_csv:
    # save as csv
    df_norm.to_csv(f'../data/predator_task/simulated_normative_learning_fig{fig_no}.csv',index=False)

# --------------
# Plot PE vs Learning Rate simulation if required
# --------------
if plot:
    plt.figure()
    plt.plot(pe, alpha, color="#249886",linewidth=2, label='Normative\nLearning')
    plt.ylim([-0.02, 1.1])
    plt.ylabel('Learning Rate')
    plt.xlabel('Prediction Error')
    plt.legend(loc=2)
    plt.show()


