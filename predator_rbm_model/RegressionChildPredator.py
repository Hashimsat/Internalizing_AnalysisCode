""" This class is a child class specific for the predator task data for a circular regression analysis.
 This class inherits the Regression parent class from the rbm_analyses module.
 Rbm analysis module can be found in: "https://github.com/rasmusbruckner/rbm_analyses"
 """

import numpy as np
import pandas as pd
from rbmpy.circular_regression.RegressionParent import RegressionParent
from rbmpy.utilities import compute_persprob, residual_fun, normalize_angle


class RegressionChildPredator(RegressionParent):
    """ This class specifies the instance variables and methods for regression analysis of the predator task data."""

    def __init__(self, reg_vars: "RegVars"):
        """
        Defines the instance variables unique to each instance.

        Parameters
        ----------
        reg_vars : RegVars
            Regression-variables-object instance.

        See Also
        --------
        RegVarsExample : Documentation for regression variables.
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
        """
        Creates the explanatory matrix for regression analysis.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame containing a subset of data.

        Returns
        -------
        reg_df : pandas.DataFrame
                Regression data frame with explanatory variables.
        """

        reg_df = pd.DataFrame(columns=['delta_t'])

        reg_df['delta_t'] = df['PredictionError']
        reg_df['delta_tau_t'] = df['tau_t'] * df['PredictionError']
        reg_df['delta_omega_t'] = df['omega_t'] * df['PredictionError']
        reg_df['delta_alpha_t'] = df['PredictionError'] * df['alpha_t']  # CPP_new + RU_new - (CPP_new.*RU_new);

        # For interaction effect of PE and BlockVersion
        df.loc[df['HazardLevel'] == 0, 'HazardLevel'] = -1
        df.loc[df['StochasticityLevel'] == 0, 'StochasticityLevel'] = -1
        reg_df['delta_HazardRateLevel'] = df['PredictionError'] * df['HazardLevel']
        reg_df['delta_StochasticityLevel'] = df['PredictionError'] * df['StochasticityLevel']
        reg_df['delta_HRStoch'] = df['PredictionError'] * df['HazardLevel'] * df['StochasticityLevel']

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
        """
            Determines the starting points of the estimation process.

            Returns
            -------
            x0 : list containing the starting points for the estimation process.
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

    def sample_data(self, df_params, n_trials=None, allSubBehavData=None):
        """
        Samples the data for simulations.

        Parameters
        ----------
        df_params : pandas.DataFrame
                    Regression parameters for simulation.
        n_trials : int, optional
                    Number of trials to simulate.
        allSubBehavData : pandas.DataFrame, optional
                        Subject behavioral data for simulations.

        Returns
        -------
        df_sim: pandas.DataFrame
                Sampled regression updates.
        """

        # Number of simulations
        n_sim = len(df_params.beta_0)

        # Initialize
        df_sim = pd.DataFrame()  # Simulated data

        # Cycle over simulations
        for i in range(0, n_sim):

            # Extract regression coefficients
            coeffs = df_params.iloc[i].to_numpy()

            # Regression variables
            if allSubBehavData is None:

                # Randomly generate data
                datamat = pd.DataFrame({
                    "delta_t": np.random.uniform(-np.pi, np.pi, n_trials),
                    "delta_tau_t": np.random.uniform(-np.pi, np.pi, n_trials),
                    "delta_omega_t": np.random.uniform(-np.pi, np.pi, n_trials),
                    "delta_alpha_t": np.random.uniform(-np.pi, np.pi, n_trials),
                    "delta_HazardRateLevel": np.random.uniform(-np.pi, np.pi, n_trials),
                    "delta_StochasticityLevel": np.random.uniform(-np.pi, np.pi, n_trials),
                    "delta_HRStoch": np.random.uniform(-np.pi, np.pi, n_trials),
                    "visible_dummy": np.random.binomial(1, 0.1, n_trials),
                    "hit_dummy": np.random.binomial(1, 0.5, n_trials),
                    "sigma_dummy": np.concatenate([np.zeros(n_trials // 2), np.ones(n_trials // 2)]),
                    "tau_t": np.random.rand(n_trials),
                    "omega_t": np.random.rand(n_trials),
                    "a_t": np.full(n_trials, np.nan),
                    "group": np.zeros(n_trials)
                })

            else:
                # Optionally based on subject data:

                # Logical index for ID
                subjects = allSubBehavData["subjectID"].unique()
                subj = subjects[i]

                df_data = allSubBehavData.loc[(allSubBehavData['subjectID'] == subj)]

                # Create design matrix
                datamat = self.get_datamat(df_data)

            # Get fixed parameters of regression
            fixed_coeffs = self.fixed_coeffs_reg

            # Initialize coefficient dictionary and counters
            sel_coeffs = dict()  # initialize list with regressor names
            j = 0  # initialize counter

            # futuretodo: maybe as a separate function when used in a different context as well
            # Put selected coefficients in list that is used for the regression
            for key, value in self.which_vars.items():
                if value:
                    sel_coeffs[key] = coeffs[j]
                    j += 1
                else:
                    sel_coeffs[key] = fixed_coeffs[key]

            # Create linear regression matrix
            lr_mat = datamat[self.which_update_regressors].to_numpy()

            # Linear regression parameters
            update_regressors = [value for key, value in sel_coeffs.items() if
                                 key not in ['omikron_0', 'omikron_1', 'lambda_0', 'lambda_1']]

            # Predicted updates
            a_t_hat = np.sum(lr_mat * update_regressors, 1)

            a_t_hat = normalize_angle(a_t_hat)

            # Residuals
            if self.which_vars["omikron_1"]:  # Access dictionary key

                # Compute updating noise based on common function
                kappa_up = residual_fun(abs(a_t_hat), sel_coeffs['omikron_0'], sel_coeffs['omikron_1'])

            else:
                # Motor noise only
                kappa_up = 1 / (
                    sel_coeffs['omikron_0'])  # np.full(len(datamat), sel_coeffs[sum(self.regressionComponents)])

            # Compute update
            a_t_hat_omik = np.random.vonmises(a_t_hat, kappa_up)

            if self.which_vars["lambda_1"]:
                pers_prob = compute_persprob(sel_coeffs["lambda_0"], sel_coeffs["lambda_1"], abs(a_t_hat))

            else:
                pers_prob = sel_coeffs['lambda_0']

            if isinstance(pers_prob, np.ndarray):
                for p in range(len(a_t_hat)):
                    if np.random.rand() < pers_prob[p]:
                        a_t_hat_omik[p] = 0
            else:
                for p in range(len(a_t_hat)):
                    if np.random.rand() < pers_prob:
                        a_t_hat_omik[p] = 0

            # todo: hier von mises nehmen..

            # Store update and ID
            df_data.loc[:, "a_t"] = a_t_hat_omik
            df_data.loc[:, "subj_num"] = i + 1

            # Combine all data
            df_sim = pd.concat([df_sim, df_data], ignore_index=True)

        return df_sim
