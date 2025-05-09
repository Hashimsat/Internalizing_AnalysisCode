from typing import ItemsView, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from functions.util_functions import cm2inch
from scipy.special import expit


def residual_fun(
    abs_dist: np.ndarray, motor_noise: float, lr_noise: float
) -> np.ndarray:
    """This function computes updating noise (residuals) as a combination of two noise components.

    The noise variable is returned in terms of von-Mises concentration,
    which is a measure of precision, where variance = 1/concentration.

    Parameters
    ----------
    abs_dist : np.ndarray
        Absolute distance (predicted update or prediction error).
    motor_noise : np.ndarray
        Motor-noise parameter (imprecise motor control).
    lr_noise : np.ndarray
        Learning-rate-noise parameter (more noise for larger updates).

    Returns
    -------
    np.ndarray
        Updating noise expressed as von Mises concentration.

    futuretodo: add to unit tests
    """

    # Compute updating noise expressed as variance
    # (1) motor noise is updating variance due to imprecise motor control and
    # (2) learning-rate noise models more noise for larger updates
    up_noise = motor_noise + lr_noise * (np.rad2deg(abs_dist))

    # Convert std of update distribution to radians and kappa
    up_noise_radians = np.deg2rad(up_noise)
    up_var_radians = up_noise_radians**2
    kappa_up = 1 / up_var_radians

    return kappa_up


def compute_persprob(
    intercept: float, slope: float, abs_pred_up: np.ndarray
) -> np.ndarray:
    """This function computes the perseveration probability.

    Parameters
    ----------
    intercept : float
        Logistic function intercept.
    slope : float
        Logistic function slope.
    abs_pred_up : np.ndarray
        Absolute predicted update or prediction error.

    Returns
    -------
    np.ndarray
        Computed perseveration probability.
    """

    # expit(x) = 1/(1+exp(-x)), i.e., (1/(1+exp(-slope*(abs_pred_up-int))))
    return expit(slope * (abs_pred_up - intercept))


def get_sel_coeffs(items: ItemsView[str, bool], fixed_coeffs: dict, coeffs) -> dict:
    """This function extracts the model coefficients.

    Parameters
    ----------
    items : ItemsView[str, bool]
        Free parameters, specified based on which_vars dict.
    fixed_coeffs : dict
        Dictionary of fixed coefficients.
    coeffs : np.ndarray
        Dictionary of free model coefficients.

    Returns
    -------
    dict
        Dictionary of selected model coefficients.
    """

    # Initialize coefficient dictionary and counters
    sel_coeffs = dict()  # initialize list with regressor names
    i = 0  # initialize counter

    # Put selected coefficients in list that is used for the regression
    for key, value in items:
        if value:
            sel_coeffs[key] = coeffs[i]
            i += 1
        else:
            sel_coeffs[key] = fixed_coeffs[key]

    return sel_coeffs


def parameter_summary(
    parameters: pd.DataFrame, param_labels: list, grid_size: tuple
) -> None:
    """This function creates a simple plot showing parameter values.

    Parameters
    ----------
    parameters : pd.DataFrame
        All parameters.
    param_labels : list
        Labels for the plot.
    grid_size : tuple
        Grid size for subplots (rows, cols).

    Returns
    -------
    None
        This function does not return any value.
    """

    fig_width = 15
    fig_height = 10

    # Create figure
    plt.figure(figsize=cm2inch(fig_width, fig_height))

    # Cycle over parameters
    for i, label in enumerate(param_labels):

        # Create subplot
        plt.subplot(grid_size[0], grid_size[1], i + 1)

        ax = plt.gca()
        sns.boxplot(
            y=label,
            data=parameters,
            notch=False,
            showfliers=False,
            linewidth=0.8,
            width=0.15,
            boxprops=dict(alpha=1),
            ax=ax,
            showcaps=False,
        )
        sns.swarmplot(y=label, data=parameters, color="gray", alpha=0.7, size=3, ax=ax)

        ttest_result = stats.ttest_1samp(parameters[label], 0)
        plt.ylabel(f"{label}")
        plt.title("p = " + str(np.round(ttest_result.pvalue, 3)))
        sns.despine()

    # Adjust layout and show plot
    plt.tight_layout()


def weighted_circular_mean(angles: Tuple[float, float], weights: Tuple[float, float]) -> float:
    """Computes a weighted circular mean.

    Parameters
    ----------
    angles : Tuple[float, float]
        Angles that are averaged.
    weights : Tuple[float, float]
        Weights for averaging.

    Returns
    -------
    float
        Weighted average.
    """
    angles = np.asarray(angles)
    weights = np.asarray(weights)
    z = weights * np.exp(1j * angles)
    mean_angle = np.angle(np.sum(z) / np.sum(weights))
    return mean_angle % (2 * np.pi)


def circ_dist(x, y):
    """Compute the pairwise signed circular distance between angles x and y.

    The distance is the signed shortest distance on the circle between each pair of angles,
    returned in the range (-π, π].

    Parameters
    ----------
    x : array_like
        Sample of linear random variables (in radians).
    y : array_like or float
        Sample of linear random variables (in radians), or a single constant angle.

    Returns
    -------
    r : ndarray
        Matrix or array of pairwise signed circular differences.

    Notes
    -----
    This is a direct translation of the MATLAB function `circ_dist` from
    the Circular Statistics Toolbox for MATLAB (Berens, 2009).
    Reference: Berens, P. (2009). "CircStat: A MATLAB Toolbox for Circular Statistics."

    See Also
    --------
    numpy.angle, numpy.exp

    Examples
    --------
    >>> circ_dist(np.pi/4, np.pi/2)
    -0.7853981633974483
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape and y.size != 1:
        raise ValueError("Input dimensions do not match.")

    r = np.angle(np.exp(1j * x) / np.exp(1j * y))
    return r


def compute_bic(llh: float, n_params: int, n_trials: int) -> float:
    """This function computes the Bayesian information criterion (BIC).

    See Stephan et al. (2009). Bayesian model selection for group studies. NeuroImage.

    Parameters
    ----------
    llh : float
        Negative log-likelihood.
    n_params : int
        Number of free parameters.
    n_trials : int
        Number of trials.

    Returns
    -------
    float
        Computed BIC.

    """

    return (-1 * llh) - (n_params / 2) * np.log(n_trials)


def normalize_angle(angle_rad: np.ndarray) -> np.ndarray:
    """ This function normalizes circular angles (in radians).

    Parameters
    ----------
    angle_rad : np.ndarray
        Raw values in radians.

    Returns
    -------
    np.ndarray
        Normalized values in radians.
    """

    # Translate to degrees
    angle_deg = np.rad2deg(angle_rad)

    # Ensure the angle is within the range -180 to 180 degrees
    # futuretodo: do everything in radians
    corr_angle = (angle_deg + 180) % 360 - 180

    # print('corrected angle deg is:', corr_angle)

    return np.deg2rad(corr_angle)
