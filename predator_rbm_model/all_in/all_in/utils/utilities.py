from typing import Union
from tqdm import tqdm


def safe_div(x: Union[int, float], y: Union[int, float]) -> float:
    """This function divides two numbers and avoids division by zero.

    Obtained from:
        https://www.yawintutor.com/zerodivisionerror-division-by-zero/

    Parameters
    ----------
    x : int or float
        Numerator.
    y : int or float
        Denominator. If zero, returns 0.0 to avoid ZeroDivisionError.

    Returns
    -------
    float
        Result of the division. Returns 0.0 if the denominator is zero.

    Examples
    --------
    safe_div(10, 2)
    5.0

    safe_div(10, 0)
    0.0
    """

    if y == 0:
        return 0.0
    return x / y


def callback(show_ind_prog: bool, pbar: tqdm) -> None:
    """Update the progress bar if enabled.

    Parameters
    ----------
    show_ind_prog : bool
        Flag indicating whether the progress bar should be updated.
    pbar : tqdm.tqdm
         Progress-bar-object instance.

    Returns
    -------
    None
    """

    if show_ind_prog:
        pbar.update()
