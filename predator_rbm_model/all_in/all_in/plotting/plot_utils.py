import types
from typing import Sequence, Tuple, Union

import numpy as np
from matplotlib.figure import Figure


def latex_plt(matplotlib: types.ModuleType) -> types.ModuleType:
    """This function updates the matplotlib library to use Latex and changes some default plot parameters.

    Parameters
    ----------
    matplotlib : module
        The matplotlib module (e.g., `import matplotlib`) to configure.

    Returns
    -------
    module
        The updated matplotlib module with LaTeX and custom settings applied.
    """

    pgf_with_latex = {
        "axes.labelsize": 6,
        "font.size": 6,
        "legend.fontsize": 6,
        "axes.titlesize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.titlesize": 6,
        "pgf.rcfonts": False,
    }
    matplotlib.rcParams.update(pgf_with_latex)

    return matplotlib


def cm2inch(*tupl: Union[float, Tuple[float, ...]]) -> Tuple[float, ...]:
    """This function converts cm to inches.

    Obtained from: https://stackoverflow.com/questions/14708695/
    specify-figure-size-in-centimeter-in-matplotlib/22787457

    Parameters
    ----------
    tupl : float or tuple of float
        Size of the plot in centimeters. Can be provided as individual float arguments (e.g., width, height)
        or as a single tuple of floats.

    Returns
    -------
    tuple of float
        Converted image size in inches.

    """

    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def label_subplots(f: Figure, texts: Sequence[str], x_offset: Union[float, Sequence[float]] = -0.07, y_offset: Union[float, Sequence[float]] = 0.015) -> None:
    """This function labels the subplots.

    Obtained from: https://stackoverflow.com/questions/52286497/
    matplotlib-label-subplots-of-different-sizes-the-exact-same-distance-from-corner

    Parameters
    ----------
    f : matplotlib.figure.Figure
        Matplotlib figure handle containing the subplots.
    texts : sequence of str
        List of labels for each subplot (e.g., ["A", "B", "C"]).
    x_offset : float or sequence of float, optional
        Horizontal offset(s) for the subplot labels.
        If a single float, the same offset is applied to all subplots.
        Default is -0.07.
    y_offset : float or sequence of float, optional
        Vertical offset(s) for the subplot labels.
        If a single float, the same offset is applied to all subplots.
        Default is 0.015.

    Returns
    -------
    None
        This function does not return any value.
    """

    # Get axes
    axes = f.get_axes()

    if isinstance(x_offset, float):
        x_offset = np.repeat(x_offset, len(axes))

    if isinstance(y_offset, float):
        y_offset = np.repeat(y_offset, len(axes))

    axis_counter = 0
    # Cycle over subplots and place labels
    for a, l in zip(axes, texts):
        x = a.get_position().x0
        y = a.get_position().y1
        f.text(x - x_offset[axis_counter], y + y_offset[axis_counter], l, size=12)
        axis_counter += 1
