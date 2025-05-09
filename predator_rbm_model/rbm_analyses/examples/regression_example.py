"""Circular regression example: This script runs a simple circular regression analysis
with simulated data to illustrate how to use the parent class in combination with
a custom child class and regression variables.

1. Simulate data
2. Run regression analysis
3. Plot the results
"""

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from all_in import latex_plt
    from rbm_analyses.utilities import parameter_summary
    from RegressionChildExample import RegressionChildExample
    from RegVarsExample import RegVars

    # Update matplotlib to use Latex and to change some defaults
    matplotlib = latex_plt(matplotlib)

    try:
        matplotlib.use("Qt5Agg")
    except ImportError:
        pass

    # Turn interactive mode on
    plt.ion()

    # ----------------
    # 1. Simulate data
    # ----------------

    # Simulation variables
    n_subj = 10  # number of subjects
    n_trials = 400  # number of trials
    motor_noise = 10  # motor noise
    fixed_LR = 0.3  # fixed learning rate

    # Define regression variables
    # ---------------------------

    reg_vars = RegVars()
    reg_vars.n_subj = n_subj
    reg_vars.n_ker = 4  # number of kernels for estimation
    reg_vars.n_sp = 5  # number of random starting points
    reg_vars.rand_sp = True  # use random starting points

    # Free parameters
    reg_vars.which_vars = {
        reg_vars.beta_0: True,  # intercept
        reg_vars.beta_1: True,  # prediction error
        reg_vars.omikron_0: True,  # motor noise
        reg_vars.omikron_1: False,  # learning-rate noise
        reg_vars.lambda_0: False,  # perseveration intercept
        reg_vars.lambda_1: False,  # perseveration slope
    }

    # Select parameters according to selected variables and create data frame
    prior_columns = [
        reg_vars.beta_0,
        reg_vars.beta_1,
        reg_vars.omikron_0,
        reg_vars.omikron_1,
        reg_vars.lambda_0,
        reg_vars.lambda_1,
    ]

    # Sample random model parameters that we try to recover
    df_params = pd.DataFrame()
    df_params["beta_0"] = np.zeros(n_subj)
    df_params["beta_1"] = np.repeat(fixed_LR, n_subj)
    df_params["omikron_0"] = np.repeat(motor_noise, n_subj)
    df_params["subj_num"] = np.arange(1, n_subj + 1)

    # Initialize regression object
    regression = RegressionChildExample(reg_vars)

    # Simulate updates based on sampled parameters
    samples = regression.sample_data(df_params, n_trials)

    # --------------------------
    # 2. Run regression analysis
    # --------------------------

    results_df = regression.parallel_estimation(samples, prior_columns)

    # Plot results
    # ------------

    behav_labels = [
        "beta_0",
        "beta_1",
        "omikron_0",
        "omikron_1",
        "lambda_0",
        "lambda_1",
    ]

    # Filter based on estimated parameters
    which_params_vec = list(reg_vars.which_vars.values())
    behav_labels = [label for label, use in zip(behav_labels, which_params_vec) if use]

    grid_size = (2, 3)
    parameter_summary(results_df, behav_labels, grid_size)

    # Show plot
    plt.ioff()
    plt.show()
