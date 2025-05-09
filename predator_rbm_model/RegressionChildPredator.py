""" This class is a child class specific for the predator task data for a circular regression analysis.
 This class inherits the Regression parent class from the rbm_analyses module.
 Rbm analysis module can be found in: "https://github.com/rasmusbruckner/rbm_analyses"
 """

import numpy as np
import pandas as pd
from rbm_analyses.rbm_analyses.circular_regression.RegressionParent import RegressionParent


class RegressionChildPredator(RegressionParent):
    """ This class specifies the instance variables and methods for regression analysis of the predator task data."""

    def __init__(self, reg_vars: "RegVars"):
        """ This function defines the instance variables unique to each instance

            See RegVarsExample for documentation

        :param reg_vars: Regression-variables-object instance
        """

        # Parameters from parent class
        super().__init__(reg_vars)

        # Extract parameter names for data frame
        self.beta_0 = reg_vars.beta_0
        self.beta_1 = reg_vars.beta_1
        self.beta_2 = reg_vars.beta_2
        self.beta_3 = reg_vars.beta_3
        self.beta_4 = reg_vars.beta_4
        self.beta_5 = reg_vars.beta_5
        self.beta_6 = reg_vars.beta_6
        self.beta_7 = reg_vars.beta_7
        self.beta_8 = reg_vars.beta_8

        self.omikron_0 = reg_vars.omikron_0
        self.omikron_1 = reg_vars.omikron_1
        self.lambda_0 = reg_vars.lambda_0
        self.lambda_1 = reg_vars.lambda_1


        # Extract staring points
        self.beta_0_x0 = reg_vars.beta_0_x0
        self.beta_1_x0 = reg_vars.beta_1_x0
        self.beta_2_x0 = reg_vars.beta_2_x0
        self.beta_3_x0 = reg_vars.beta_3_x0
        self.beta_4_x0 = reg_vars.beta_4_x0
        self.beta_5_x0 = reg_vars.beta_5_x0
        self.beta_6_x0 = reg_vars.beta_6_x0
        self.beta_7_x0 = reg_vars.beta_7_x0
        self.beta_8_x0 = reg_vars.beta_8_x0

        self.omikron_0_x0 = reg_vars.omikron_0_x0
        self.omikron_1_x0 = reg_vars.omikron_1_x0
        self.lambda_0_x0 = reg_vars.lambda_0_x0
        self.lambda_1_x0 = reg_vars.lambda_1_x0

        # Extract range of random starting point values
        self.beta_0_x0_range = reg_vars.beta_0_x0_range
        self.beta_1_x0_range = reg_vars.beta_1_x0_range
        self.beta_2_x0_range = reg_vars.beta_2_x0_range
        self.beta_3_x0_range = reg_vars.beta_3_x0_range
        self.beta_4_x0_range = reg_vars.beta_4_x0_range
        self.beta_5_x0_range = reg_vars.beta_5_x0_range
        self.beta_6_x0_range = reg_vars.beta_6_x0_range
        self.beta_7_x0_range = reg_vars.beta_7_x0_range
        self.beta_8_x0_range = reg_vars.beta_8_x0_range

        self.omikron_0_x0_range = reg_vars.omikron_0_x0_range
        self.omikron_1_x0_range = reg_vars.omikron_1_x0_range
        self.lambda_0_x0_range = reg_vars.lambda_0_x0_range
        self.lambda_1_x0_range = reg_vars.lambda_1_x0_range

        # Extract boundaries for estimation
        self.beta_0_bnds = reg_vars.beta_0_bnds
        self.beta_1_bnds = reg_vars.beta_1_bnds
        self.beta_2_bnds = reg_vars.beta_2_bnds
        self.beta_3_bnds = reg_vars.beta_3_bnds
        self.beta_4_bnds = reg_vars.beta_4_bnds
        self.beta_5_bnds = reg_vars.beta_5_bnds
        self.beta_6_bnds = reg_vars.beta_6_bnds
        self.beta_7_bnds = reg_vars.beta_7_bnds
        self.beta_8_bnds = reg_vars.beta_8_bnds

        self.omikron_0_bnds = reg_vars.omikron_0_bnds
        self.omikron_1_bnds = reg_vars.omikron_1_bnds
        self.lambda_0_bnds = reg_vars.lambda_0_bnds
        self.lambda_1_bnds = reg_vars.lambda_1_bnds

        # Extract free parameters
        self.which_vars = reg_vars.which_vars

        # Extract fixed parameter values
        self.fixed_coeffs_reg = reg_vars.fixed_coeffs_reg

    @staticmethod
    def get_datamat(df):
        """ This function creates the explanatory matrix

        :param df: Data frame containing subset of data
        :return: reg_df: Regression data frame
        """

        reg_df = pd.DataFrame(columns=['delta_t'])

        reg_df['delta_t'] = df['PredictionError']
        reg_df['delta_tau_t'] = df['tau_t'] * df['PredictionError']
        reg_df['delta_omega_t'] = df['omega_t'] * df['PredictionError']
        reg_df['delta_alpha_t'] = df['PredictionError'] * (df['omega_t'] + df['tau_t'] - (df['omega_t'] * df['tau_t']))  # CPP_new + RU_new - (CPP_new.*RU_new);

        # For interaction effect of PE and BlockVersion
        df.loc[df['HazardLevel'] == 0, 'HazardLevel'] = -1
        df.loc[df['StochasticityLevel'] == 0, 'StochasticityLevel'] = -1
        reg_df['delta_HazardRateLevel'] = df['PredictionError'] * df['HazardLevel']
        reg_df['delta_StochasticityLevel'] = df['PredictionError'] * df['StochasticityLevel']
        reg_df['delta_HRStoch'] = df['PredictionError'] * df['HazardLevel'] * df['StochasticityLevel'];

        reg_df['delta_HitMiss'] = df['PredictionError'] * df['HitMiss']
        reg_df['int'] = np.ones(len(df))

        reg_df['a_t'] = df['a_t'].to_numpy()
        reg_df['ID'] = df['subjectID']
        reg_df['group'] = df['group'].to_numpy()
        reg_df['subj_num'] = df['subj_num'].to_numpy()

        # remove nans
        reg_df = reg_df.dropna(axis=0, how='any')


        return reg_df

    def get_starting_point(self):
        """ This function determines the starting points of the estimation process

        :return: x0: List with starting points
        """

        # Put all starting points into list
        if self.rand_sp:

            # Draw random starting points
            x0 = [np.random.uniform(self.beta_0_x0_range[0], self.beta_0_x0_range[1]),
                  np.random.uniform(self.beta_1_x0_range[0], self.beta_1_x0_range[1]),
                  np.random.uniform(self.beta_2_x0_range[0], self.beta_2_x0_range[1]),
                  np.random.uniform(self.beta_3_x0_range[0], self.beta_3_x0_range[1]),
                  np.random.uniform(self.beta_4_x0_range[0], self.beta_4_x0_range[1]),
                  np.random.uniform(self.beta_5_x0_range[0], self.beta_5_x0_range[1]),
                  np.random.uniform(self.beta_6_x0_range[0], self.beta_6_x0_range[1]),
                  np.random.uniform(self.beta_7_x0_range[0], self.beta_7_x0_range[1]),
                  np.random.uniform(self.beta_8_x0_range[0], self.beta_8_x0_range[1]),

                  np.random.uniform(self.omikron_0_x0_range[0], self.omikron_0_x0_range[1]),
                  np.random.uniform(self.omikron_1_x0_range[0], self.omikron_1_x0_range[1]),
                  np.random.uniform(self.lambda_0_x0_range[0], self.lambda_0_x0_range[1]),
                  np.random.uniform(self.lambda_1_x0_range[0], self.lambda_1_x0_range[1])
            ]

        else:
            # Use fixed starting points
            x0 = [self.beta_0_x0,
                  self.beta_1_x0,
                  self.beta_2_x0,
                  self.beta_3_x0,
                  self.beta_4_x0,
                  self.beta_5_x0,
                  self.beta_6_x0,
                  self.beta_7_x0,
                  self.beta_8_x0,

                  self.omikron_0_x0,
                  self.omikron_1_x0,
                  self.lambda_0_x0,
                  self.lambda_1_x0
            ]

        return x0
