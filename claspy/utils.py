import numpy as np


def check_input_time_series(time_series):
    """
    Check that the input time series is a 1-dimensional numpy array of numbers.

    Parameters
    ----------
    time_series : array-like
        The input time series.

    Returns
    -------
    time_series : ndarray
        The input time series as a 1-dimensional numpy array of floats or integers.

    Raises
    ------
    TypeError
        If the input time series is not an array-like object or not a 1-dimensional array.
    ValueError
        If the input time series is not composed of numbers.
    """
    if not isinstance(time_series, np.ndarray):
        raise TypeError("Input time series must be a numpy array.")

    if len(time_series.shape) != 1:
        raise ValueError("Input time series must be one-dimensional.")

    if not np.issubdtype(time_series.dtype, np.number):
        raise TypeError("Input time series must contain numeric values.")
