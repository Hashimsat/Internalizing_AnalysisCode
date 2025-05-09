"""This class is an example of a child class for a circular regression analysis."""

import numpy as np
import pandas as pd
from rbm_analyses import RegressionParent, get_sel_coeffs, residual_fun


class RegressionChildExample(RegressionParent):
    """This class specifies the instance variables and methods of the example of the regression analysis."""

    def __init__(self, reg_vars: "RegVars"):
        """This function defines the instance variables unique to each instance.

        See RegVarsExample for documentation.

        Parameters
        ----------
        reg_vars : RegVars
            Regression-variables-object instance.
        """

        # Parameters from parent class
        super().__init__(reg_vars)

        # Extract parameter names for data frame
        self.beta_0 = reg_vars.beta_0
        self.beta_1 = reg_vars.beta_1
        self.omikron_0 = reg_vars.omikron_0
        self.omikron_1 = reg_vars.omikron_1
        self.lambda_0 = reg_vars.lambda_0
        self.lambda_1 = reg_vars.lambda_1

        # Extract staring points
        self.beta_0_x0 = reg_vars.beta_0_x0
        self.beta_1_x0 = reg_vars.beta_1_x0
        self.omikron_0_x0 = reg_vars.omikron_0_x0
        self.omikron_1_x0 = reg_vars.omikron_1_x0
        self.lambda_0_x0 = reg_vars.lambda_0_x0
        self.lambda_1_x0 = reg_vars.lambda_1_x0

        # Extract range of random starting point values
        self.beta_0_x0_range = reg_vars.beta_0_x0_range
        self.beta_1_x0_range = reg_vars.beta_1_x0_range
        self.omikron_0_x0_range = reg_vars.omikron_0_x0_range
        self.omikron_1_x0_range = reg_vars.omikron_1_x0_range
        self.lambda_0_x0_range = reg_vars.lambda_0_x0_range
        self.lambda_1_x0_range = reg_vars.lambda_1_x0_range

        # Extract boundaries for estimation
        self.beta_0_bnds = reg_vars.beta_0_bnds
        self.beta_1_bnds = reg_vars.beta_1_bnds
        self.omikron_0_bnds = reg_vars.omikron_0_bnds
        self.omikron_1_bnds = reg_vars.omikron_1_bnds
        self.lambda_0_bnds = reg_vars.lambda_0_bnds
        self.lambda_1_bnds = reg_vars.lambda_1_bnds

        # Extract free parameters
        self.which_vars = reg_vars.which_vars

        # Extract fixed parameter values
        self.fixed_coeffs_reg = reg_vars.fixed_coeffs_reg

    @staticmethod
    def get_datamat(df: pd.DataFrame) -> pd.DataFrame:
        """This function creates the explanatory matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame containing subset of data.

         Returns
        -------
        pd.DataFrame
            Regression data frame.
        """

        # Create custom data matrix for project
        reg_df = pd.DataFrame(columns=["delta_t"])
        reg_df["int"] = np.ones(len(df))
        reg_df["delta_t"] = df["delta_t_rad"].to_numpy()
        reg_df["a_t"] = df["a_t_rad"].to_numpy()
        reg_df["group"] = df["group"].to_numpy()
        reg_df["subj_num"] = df["subj_num"].to_numpy()
        reg_df["ID"] = df["ID"].to_numpy()

        return reg_df

    def get_starting_point(self) -> list:
        """This function determines the starting points of the estimation process.

        Returns
        -------
        list
            List with starting points.
        """

        # Put all starting points into list
        if self.rand_sp:

            # Draw random starting points
            x0 = [
                np.random.uniform(self.beta_0_x0_range[0], self.beta_0_x0_range[1]),
                np.random.uniform(self.beta_1_x0_range[0], self.beta_1_x0_range[1]),
                np.random.uniform(
                    self.omikron_0_x0_range[0], self.omikron_0_x0_range[1]
                ),
                np.random.uniform(
                    self.omikron_1_x0_range[0], self.omikron_1_x0_range[1]
                ),
            ]

        else:

            # Use fixed starting points
            x0 = [self.beta_0_x0, self.beta_1_x0, self.omikron_0_x0, self.omikron_1_x0]

        return x0

    def sample_data(
        self,
        df_params: pd.DataFrame,
        n_trials: int,
        all_sub_behav_data: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """This function samples the data for simulations.

        Parameters
        ----------
        df_params : pd.DataFrame
            Regression parameters for simulation.
        n_trials : int
            Number of trials.
        all_sub_behav_data : pd.DataFrame
             Optional subject behavioral data.

        Returns
        -------
        pd.DataFrame
            Sampled regression updates.
        """

        # Number of simulations
        n_sim = len(df_params["subj_num"])

        # Initialize
        df_sim = pd.DataFrame()  # simulated data

        # Cycle over simulations
        for i in range(0, n_sim):

            # Extract regression coefficients
            coeffs = df_params.iloc[i].to_numpy()

            # Randomly generate data
            df_data = pd.DataFrame(
                {
                    "delta_t_rad": np.random.uniform(-np.pi, np.pi, n_trials),
                    "a_t_rad": np.full(n_trials, np.nan),
                    "subj_num": np.random.randint(1, 100, n_trials),
                    "ID": np.random.randint(1000, 5000, n_trials),
                    "group": np.zeros(n_trials),
                }
            )

            # Create design matrix
            datamat = self.get_datamat(df_data)

            # Extract parameters
            sel_coeffs = get_sel_coeffs(
                self.which_vars.items(), self.fixed_coeffs_reg, coeffs
            )

            # Create linear regression matrix
            lr_mat = datamat[self.which_update_regressors].to_numpy()

            # Linear regression parameters
            update_regressors = [
                value
                for key, value in sel_coeffs.items()
                if key not in ["omikron_0", "omikron_1", "lambda_0", "lambda_1"]
            ]

            # Predicted updates
            a_t_hat = lr_mat @ np.array(update_regressors)

            # Residuals
            if self.which_vars["omikron_1"]:

                # Compute updating noise
                concentration = residual_fun(
                    np.abs(a_t_hat), sel_coeffs["omikron_0"], sel_coeffs["omikron_1"]
                )

            else:
                # Motor noise only
                concentration = np.repeat(sel_coeffs["omikron_0"], len(a_t_hat))

            # Sample updates from von Mises distribution
            a_t_hat = np.random.vonmises(a_t_hat, concentration)

            # Store update and ID
            df_data["a_t_rad"] = a_t_hat
            df_data["subj_num"] = np.full(len(df_data), i + 1)

            # Combine all data
            df_sim = pd.concat([df_sim, df_data], ignore_index=True)

        return df_sim
