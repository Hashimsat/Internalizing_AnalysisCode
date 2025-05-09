"""This class is an example of the regression-variables class for a circular regression analysis"""

import numpy as np


class RegVars:
    """This class specifies the RegVars object for a circular regression analysis

    futuretodo: Consider using a data class here
    """

    def __init__(self):
        """This function defines the instance variables unique to each instance"""

        # Parameter names for data frame
        self.beta_0 = "beta_0"  # intercept
        self.beta_1 = "beta_1"  # coefficient for prediction error (delta)
        self.omikron_0 = "omikron_0"  # noise intercept
        self.omikron_1 = "omikron_1"  # noise slope
        self.lambda_0 = "lambda_0"
        self.lambda_1 = "lambda_1"

        # Variable names of update regressors (independent of noise terms)
        self.which_update_regressors = ["int", "delta_t"]

        # Select staring points (used if rand_sp = False)
        self.beta_0_x0 = 0
        self.beta_1_x0 = 0
        self.omikron_0_x0 = 10
        self.omikron_1_x0 = 0.1
        self.lambda_0_x0 = 0.1
        self.lambda_1_x0 = 0.1

        # Select range of random starting point values
        self.beta_0_x0_range = (-1, 1)
        self.beta_1_x0_range = (0, 1)
        self.omikron_0_x0_range = (5, 50)
        self.omikron_1_x0_range = (0, 1)
        self.lambda_0_x0_range = (0, 1)
        self.lambda_1_x0_range = (0, 1)

        # Select boundaries for estimation
        self.beta_0_bnds = (-2, 2)
        self.beta_1_bnds = (-2, 2)
        self.omikron_0_bnds = (0.0001, 50)
        self.omikron_1_bnds = (0.001, 1)
        self.lambda_0_bnds = (0, 1)
        self.lambda_1_bnds = (0, 1)

        self.bnds = [
            self.beta_0_bnds,
            self.beta_1_bnds,
            self.omikron_0_bnds,
            self.omikron_1_bnds,
            self.lambda_0_bnds,
            self.lambda_1_bnds,
        ]

        # Free parameters
        self.which_vars = {
            self.beta_0: True,
            self.beta_1: True,
            self.omikron_0: True,
            self.omikron_1: True,
            self.lambda_0: False,
            self.lambda_1: True,
        }

        # Fixed parameter values
        self.fixed_coeffs_reg = {
            self.beta_0: 0.0,
            self.beta_1: 0.0,
            self.omikron_0: 10.0,
            self.omikron_1: 0.0,
            self.lambda_0: 0.0,
            self.lambda_1: 0.0,
        }

        # When prior is used: pior mean
        self.beta_0_prior_mean = 0
        self.beta_1_prior_mean = 0
        self.omikron_0_prior_mean = 10
        self.omikron_1_prior_mean = 0.1
        self.lambda_0_prior_mean = 0
        self.lambda_1_prior_mean = 0

        # All prior means
        self.prior_mean = [
            self.beta_0_prior_mean,
            self.beta_1_prior_mean,
            self.omikron_0_prior_mean,
            self.omikron_1_prior_mean,
            self.lambda_0_prior_mean,
            self.lambda_1_prior_mean,
        ]

        # When prior is used: pior width
        # Note these can be tuned for future versions
        self.beta_0_prior_width = 5
        self.beta_1_prior_width = 5
        self.omikron_0_prior_width = 20
        self.omikron_1_prior_width = 5
        self.lambda_0_prior_width = 5
        self.lambda_1_prior_width = 5

        # All prior widths
        self.prior_width = [
            self.beta_0_prior_width,
            self.beta_1_prior_width,
            self.omikron_0_prior_width,
            self.omikron_1_prior_width,
            self.lambda_0_prior_width,
            self.lambda_1_prior_width,
        ]

        # Other attributes
        self.n_subj = np.nan  # number of subjects
        self.n_ker = 4  # number of kernels for estimation
        self.seed = 123  # seed for random number generator
        self.show_ind_prog = True  # Update progress bar for each subject (True, False)
        self.rand_sp = True  # 0 = fixed starting points, 1 = random starting points
        self.n_sp = 5  # number of starting points (if rand_sp = 1)
        self.use_prior = False  # use of prior for estimations
