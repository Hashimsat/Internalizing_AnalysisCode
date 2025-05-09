"""Parent Regression Class: Parent class for custom circular regressions"""

import sys
from itertools import compress
from multiprocessing import Pool
from time import sleep

import numpy as np
import pandas as pd
from predator_rbm_model.all_in.all_in.utils.utilities import callback
from ..utilities import (compute_persprob, get_sel_coeffs, normalize_angle, residual_fun)
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm, vonmises
from tqdm import tqdm


class RegressionParent:
    """This parent class specifies the instance variables and methods of the common methods of
    circular regression analyses.
    """

    def __init__(self, reg_vars: "RegVars"):
        """This function defines the instance variables unique to each instance.

        See project-specific RegVars in child class for documentation.

        Parameters
        ----------
        reg_vars : RegVars
            Regression-variables-object instance defined in your project.
        """

        # Extract free parameters
        self.which_vars = reg_vars.which_vars

        # Extract fixed parameter values
        self.fixed_coeffs_reg = reg_vars.fixed_coeffs_reg

        # Extract prior mean and width
        self.prior_mean = reg_vars.prior_mean
        self.prior_width = reg_vars.prior_width

        # Extract other attributes
        self.n_subj = reg_vars.n_subj
        self.n_ker = reg_vars.n_ker
        self.seed = reg_vars.seed
        self.show_ind_prog = reg_vars.show_ind_prog
        self.rand_sp = reg_vars.rand_sp
        self.use_prior = reg_vars.use_prior
        self.n_sp = reg_vars.n_sp
        self.bnds = reg_vars.bnds
        self.which_update_regressors = reg_vars.which_update_regressors

    def parallel_estimation(
        self, df: pd.DataFrame, prior_columns: list
    ) -> pd.DataFrame:
        """This function manages the parallel estimation of the regression models.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame containing the data.
        prior_columns : list
            Selected parameters for regression.

        Returns
        -------
        pd.DataFrame
            Data frame containing regression results.
        """

        # Inform user about progress
        pbar = None
        if self.show_ind_prog:
            # Inform user
            sleep(0.1)
            print("\nRegression model estimation:")
            sleep(0.1)

            # Initialize progress bar
            pbar = tqdm(total=self.n_subj)

        # Initialize pool object for parallel processing
        pool = Pool(processes=self.n_ker)

        # Estimate parameters in parallel
        results = [
            pool.apply_async(
                self.estimation,
                args=(df[(df["subj_num"] == i + 1)].copy(),),
                callback=lambda _: callback(self.show_ind_prog, pbar),
            )
            for i in range(0, self.n_subj)
        ]

        output = [p.get() for p in results]
        pool.close()
        pool.join()

        # Close progress bar
        if self.show_ind_prog and pbar:
            pbar.close()

        # Put all results in data frame
        values = self.which_vars.values()
        columns = list(compress(prior_columns, values))
        columns.append("llh")
        columns.append("group")
        columns.append("subj_num")
        columns.append("ID")
        results_df = pd.DataFrame(output, columns=columns)

        return results_df

    def estimation(self, df_subj_input: pd.DataFrame) -> list:
        """This function estimates the coefficients of the mixture model.

        Parameters
        ----------
        df_subj_input : pd.DataFrame
            Data frame containing subject-specific subset of data.

        Returns
        -------
        list
            A list with the regression results.
        """

        # Control random number generator for reproducible results
        np.random.seed(self.seed)

        # Get data matrix that is required for the model from child class
        df_subj = self.get_datamat(df_subj_input)

        # Adjust index of this subset of variables
        df_subj.reset_index().rename(columns={"index": "trial"})

        # Select starting points and boundaries
        # -------------------------------------

        # Extract free parameters
        values = self.which_vars.values()

        # Select boundaries according to free parameters
        bnds = list(compress(self.bnds, values))

        # Initialize with unrealistically high likelihood and no parameter estimate
        min_llh = 100000
        min_x = np.nan

        # Cycle over starting points
        for _ in range(0, self.n_sp):

            # Get project-specific starting points for child class
            x0 = self.get_starting_point()

            # Select starting points according to free parameters
            x0 = np.array(list(compress(x0, values)))

            # Estimate parameters
            res = minimize(
                self.llh,
                x0,
                args=(df_subj,),
                method="L-BFGS-B",
                options={"disp": False, "maxiter": 500},
                bounds=bnds,
            )

            # Parameter values
            x = res.x

            # Negative log-likelihood
            llh_sum = res.fun

            # Check if cumulated negative log likelihood is lower than the previous
            # one and select the lowest
            if llh_sum < min_llh:
                min_llh = llh_sum
                min_x = x

        # Add results to list
        results_list = list()
        for i in range(len(min_x)):
            results_list.append(float(min_x[i]))

        # Extract group and subj_num for output
        group = int(pd.unique(df_subj["group"])[0])
        subj_num = int(pd.unique(df_subj["subj_num"])[0])
        id = pd.unique(df_subj["ID"])[0]

        # Add group and log-likelihood to output
        results_list.append(float(min_llh))
        results_list.append(group)
        results_list.append(subj_num)
        results_list.append(id)

        return results_list

    def llh(self, coeffs: np.ndarray, df: pd.DataFrame) -> float:
        """This function computes the likelihood of participant updates, given the specified parameters.

        Parameters
        ----------
        coeffs : np.ndarray
            Regression coefficients.
        df : pd.DataFrame
            Data frame containing subset of data.

        Returns
        -------
        float
            Summed negative log-likelihood.
        """

        # Initialize small value that replaces zero probabilities for numerical stability
        corrected_0_p = 1e-10

        # Extract parameters
        sel_coeffs = get_sel_coeffs(
            self.which_vars.items(), self.fixed_coeffs_reg, coeffs
        )

        # Linear regression component
        # ---------------------------

        # Create linear regression matrix
        lr_mat = df[self.which_update_regressors].to_numpy()

        # Linear regression parameters
        update_regressors = [
            value
            for key, value in sel_coeffs.items()
            if key not in ["omikron_0", "omikron_1", "lambda_0", "lambda_1"]
        ]

        # Compute predicted update
        a_t_hat = lr_mat @ np.array(update_regressors)  # matrix multiplication

        # Ensure value is in range [-pi, pi]
        a_t_hat = normalize_angle(a_t_hat)

        # Residuals
        if self.which_vars["omikron_1"]:

            # Compute updating noise
            concentration = residual_fun(
                abs(a_t_hat), sel_coeffs["omikron_0"], sel_coeffs["omikron_1"]
            )

        else:
            # Motor noise only
            concentration = np.repeat(sel_coeffs["omikron_0"], len(a_t_hat))

        # Compute probability density of update
        p_a_t = vonmises.pdf(df["a_t"], loc=a_t_hat, kappa=concentration)
        p_a_t[p_a_t == 0] = (
            corrected_0_p  # adjust zeros to small value for numerical stability
        )

        # Check for inf and nan
        if sum(np.isinf(p_a_t)) > 0:
            sys.exit("\np_a_t contains infs")
        elif sum(np.isnan(p_a_t)) > 0:
            sys.exit("\np_a_t contains nans")

        # Compute log-likelihood of linear regression
        llh_reg = np.log(p_a_t)

        # Check for inf and nan
        if sum(np.isinf(llh_reg)) > 0:
            sys.exit("llh_reg contains infs")
        elif sum(np.isnan(llh_reg)) > 0:
            sys.exit("llh_reg contains nans")

        # Identify perseveration trials
        pers = df["a_t"] == 0

        # Adjust for probabilities on the edge
        delta_fun = np.full(len(pers), np.nan)
        delta_fun[pers == 1] = 1 - corrected_0_p
        delta_fun[pers == 0] = corrected_0_p

        # Note: keep in mind that this has not been systematically validated with participant data
        # the setup works for simulated data based on a uniform distribution which has more large-scale PEs
        # than actual data. For actual data, some adjustments might be necessary, such as parameter ranges, parameter
        # combinations, or potentially in degrees instead of radians.
        lambda_t = None
        if self.which_vars["lambda_0"] and not self.which_vars["lambda_1"]:

            # Single average perseveration parameter lambda_0
            lambda_t = np.repeat(sel_coeffs["lambda_0"], len(pers))

        elif (not self.which_vars["lambda_0"] and self.which_vars["lambda_1"]) or (
            self.which_vars["lambda_0"] and self.which_vars["lambda_1"]
        ):

            # Logistic function combining both parameters
            lambda_t = compute_persprob(
                sel_coeffs["lambda_0"], sel_coeffs["lambda_1"], abs(a_t_hat)
            )

        if self.which_vars["lambda_0"] or self.which_vars["lambda_1"]:

            lambda_t[lambda_t == 0] = corrected_0_p
            lambda_t[lambda_t == 1] = 1 - corrected_0_p

            # Compute mixture between linear regression and perseveration model using lambda as weight
            llh_mix = logsumexp(
                [
                    np.log(delta_fun) + np.log(lambda_t),
                    np.log((1 - lambda_t)) + llh_reg,
                ],
                axis=0,
            )

            # Check for inf and nan
            if sum(np.isinf(llh_mix)) > 0:
                sys.exit("llh_mix contains infs")
            elif sum(np.isnan(llh_mix)) > 0:
                sys.exit("llh_mix contains nans")

            # Compute negative log-likelihood
            llh_sum = -1 * np.sum(llh_mix)

        else:
            # Compute negative log-likelihood
            llh_sum = -1 * np.sum(llh_reg)

        # Check for inf and nan
        if np.isinf(llh_sum) or np.isnan(llh_sum):
            sys.exit("\nllh incorrect")

        if self.use_prior:
            # Extract free parameters
            values = self.which_vars.values()

            # Prior mean and variance
            prior_mean = np.array(list(compress(self.prior_mean, values)))
            prior_width = np.array(list(compress(self.prior_width, values)))

            # Compute coefficient probabilites
            prior_prob = norm.pdf(coeffs, loc=prior_mean, scale=prior_width)

            # Set a minimum prior probability threshold before taking log
            prior_prob = np.maximum(prior_prob, corrected_0_p)

            # Adjust the negative log-likelihood
            llh_sum -= np.sum(np.log(prior_prob))

        return llh_sum

    @staticmethod
    def get_datamat(df_subj_input):
        """This function raises an error if the get_datamat function is undefined in the
        project-specific regression.
        """
        raise NotImplementedError("Subclass needs to define this.")

    def get_starting_point(self):
        """This function raises an error if the get_starting_point function is undefined in the
        project-specific regression.
        """
        raise NotImplementedError("Subclass needs to define this.")

    def sample_data(self, df_params, n_trials, all_sub_data=None):
        """This function raises an error if the sample_data function is undefined in the
        project-specific regression.
        """
        raise NotImplementedError("Subclass needs to define this.")
