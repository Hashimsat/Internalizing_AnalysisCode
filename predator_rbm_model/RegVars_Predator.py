""" This class is an example of the regression variables class for a circular regression analysis """

import numpy as np
import math


class RegVars:
    """ This class specifies the RegVars object for a circular regression analysis

        futuretodo: Consider using a data class here
    """

    def __init__(self):
        """ This function defines the instance variables unique to each instance """

        # Parameter names for data frame
        self.beta_0 = 'beta_0'  # intercept
        self.beta_1 = 'beta_1'  # coefficient for prediction error (delta)
        self.beta_2 = 'beta_2'  # interaction coefficient delta and relative uncertainty (tau)
        self.beta_3 = 'beta_3'  # interaction coefficient delta and change-point probability (omega)
        self.beta_4 = 'beta_4'  # delta_t * alpha (alpha = combined learning rate)
        self.beta_5 = 'beta_5'  # interaction coefficient delta and HazardRateLevel (Baseline would be PredatorType1)
        self.beta_6 = 'beta_6'  # interaction coefficient delta and StochasticityLevel
        self.beta_7 = 'beta_7'  # interaction coefficient delta and HitMiss
        self.beta_8 = 'beta_8'  # interaction coefficient delta and interaction between HRLevela nd StochLevel

        self.omikron_0 = 'omikron_0'  # noise intercept
        self.omikron_1 = 'omikron_1'  # noise slope
        self.lambda_0 = 'lambda_0'
        self.lambda_1 = 'lambda_1'

        # Variable names of update regressors (independent of noise terms)
        self.which_update_regressors = ['int', 'delta_t', 'delta_tau_t', 'delta_omega_t', 'delta_alpha_t', 'delta_HazardRateLevel',
                                 'delta_StochasticityLevel', 'delta_HitMiss', 'delta_HRStoch']

        # Select staring points (used if rand_sp = False)
        self.beta_0_x0 = 0
        self.beta_1_x0 = 0
        self.beta_2_x0 = 0
        self.beta_3_x0 = 0
        self.beta_4_x0 = 0
        self.beta_5_x0 = 0
        self.beta_6_x0 = 0
        self.beta_7_x0 = 0
        self.beta_8_x0 = 0

        self.omikron_0_x0 = 0.01
        self.omikron_1_x0 = 0
        self.lambda_0_x0 = 0.1
        self.lambda_1_x0 = 0

        # Select range of random starting point values
        self.beta_0_x0_range = (-2, 2)
        self.beta_1_x0_range = (-1, 1)
        self.beta_2_x0_range = (-1, 1)
        self.beta_3_x0_range = (-1, 1)
        self.beta_4_x0_range = (-1, 1)
        self.beta_5_x0_range = (-1, 1)
        self.beta_6_x0_range = (-1, 1)
        self.beta_7_x0_range = (-1, 1)
        self.beta_8_x0_range = (-1, 1)

        self.omikron_0_x0_range = (0.001, 50)
        self.omikron_1_x0_range = (0, 1)
        self.lambda_0_x0_range = (-4, 4)
        self.lambda_1_x0_range = (-2, 2)

        # Select boundaries for estimation
        self.beta_0_bnds = (-2, 2)
        self.beta_1_bnds = (-2, 2)
        self.beta_2_bnds = (-2, 2)
        self.beta_3_bnds = (-2, 2)
        self.beta_4_bnds = (-2, 2)
        self.beta_5_bnds = (-2, 2)
        self.beta_6_bnds = (-2, 2)
        self.beta_7_bnds = (-2, 2)
        self.beta_8_bnds = (-2, 2)

        self.omikron_0_bnds = (0.0001, 50)
        self.omikron_1_bnds = (0.001, 1)
        self.lambda_0_bnds = (-math.pi, math.pi)
        self.lambda_1_bnds = (-1.5, 1.5)

        self.bnds = [self.beta_0_bnds, self.beta_1_bnds, self.beta_2_bnds, self.beta_3_bnds, self.beta_4_bnds,
                     self.beta_5_bnds, self.beta_6_bnds, self.beta_7_bnds, self.beta_8_bnds,
                     self.omikron_0_bnds, self.omikron_1_bnds, self.lambda_0_bnds, self.lambda_1_bnds]

        # Free parameters
        self.which_vars = {self.beta_0: True,
                           self.beta_1: True,
                           self.beta_2: True,
                           self.beta_3: True,
                           self.beta_4: True,
                           self.beta_5: True,
                           self.beta_6: True,
                           self.beta_7: True,
                           self.beta_8: True,

                           self.omikron_0: True,
                           self.omikron_1: True,
                           self.lambda_0: True,
                           self.lambda_1: True
                           }

        # Fixed parameter values
        self.fixed_coeffs_reg = {self.beta_0: 0.0,
                                 self.beta_1: 0.0,
                                 self.beta_2: 0.0,
                                 self.beta_3: 0.0,
                                 self.beta_4: 0.0,
                                 self.beta_5: 0.0,
                                 self.beta_6: 0.0,
                                 self.beta_7: 0.0,
                                 self.beta_8: 0.0,

                                 self.omikron_0: 10.0,
                                 self.omikron_1: 0.0,
                                 self.lambda_0: 0.0,
                                 self.lambda_1: 0.0
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
        self.n_ker = 5  # number of kernels for estimation
        self.show_ind_prog = True  # Update progress bar for each subject (True, False)
        self.rand_sp = True  # 0 = fixed starting points, 1 = random starting points
        self.n_sp = 5  # number of starting points (if rand_sp = 1)
        self.use_prior = False  # use of prior for estimations
        self.seed = 125  # random seed for reproducibility
