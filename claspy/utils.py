import os
import shutil

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


def check_excl_radius(k_neighbours, excl_radius):
    """
    Check if the exclusion radius is larger than the amount of neighbours used for ClaSP.
    This is conceptually useful, as the excl_radius defines the min_seg_size, which
    should include at least k repetitions of a temporal pattern and itself.

    Parameters
    ----------
    k_neighbours : int
        The number of neighbours used for ClaSP.
    excl_radius : float
        The exclusion radius used for ClaSP.

    Raises
    ------
    ValueError
        If the exclusion radius is smaller than the number of neighbours used.
    """
    if excl_radius <= k_neighbours:
        raise ValueError("Exclusion radius must be larger than the number of neighbours used.")


def numba_cache_safe(func, *args, **kwargs):
    """
    Execute a function safely, handling potential issues with numba's cache. If a
    ReferenceError is caught, indicating a potential issue with the cache, it
    is cleared and the function is attempted to be executed again.

    Parameters
    ----------
    func : callable
        The function to be executed. This function can be any callable but is
        intended for use with functions that use Numba's JIT compilation.
    *args : tuple
        Positional arguments to be passed to the function.
    **kwargs : dict
        Keyword arguments to be passed to the function.

    Returns
    -------
    Any
        The return value of the `func` function.

    Raises
    ------
    ReferenceError
        If the ReferenceError persists even after clearing the Numba cache.
    """
    try:
        return func(*args, **kwargs)
    except ReferenceError as e:
        pycache_path = os.path.join(os.path.dirname(__file__), "__pycache__")

        # delete cache and try again
        if os.path.exists(pycache_path):
            shutil.rmtree(pycache_path)
            return func(*args, **kwargs)

        # other issue
        raise e

