import math
import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf

from claspy.utils import check_input_time_series


def _suss_score(time_series, window_size, stats):
    """
    Calculate the SuSS score for a given window size.

    Parameters
    ----------
    time_series : np.ndarray
        The time series to calculate the SuSS score for.
    window_size : int
        The window size to use in the SuSS score calculation.
    stats : Tuple[float, float, float]
        A tuple of statistics for the time series. The tuple should contain the mean,
        standard deviation, and difference between the maximum and minimum values of
        the time series.

    Returns
    -------
    float
        The SuSS score for the given time series and window size.
    """
    roll = pd.Series(time_series).rolling(window_size)
    ts_mean, ts_std, ts_min_max = stats

    roll_mean = roll.mean().to_numpy()[window_size:]
    roll_std = roll.std(ddof=0).to_numpy()[window_size:]
    roll_min = roll.min().to_numpy()[window_size:]
    roll_max = roll.max().to_numpy()[window_size:]

    X = np.array([
        roll_mean - ts_mean,
        roll_std - ts_std,
        (roll_max - roll_min) - ts_min_max
    ])

    X = np.sqrt(np.sum(np.square(X), axis=0)) / np.sqrt(window_size)

    return np.mean(X)


def suss(time_series, lbound=10, threshold=.89):
    """
    Calculate the optimal window size for a time series comparing summary statistics
    of different-sized subsequences with the ones of the entire time series.

    Parameters
    ----------
    time_series : numpy.ndarray
        The time series to to determine the window size for.
    lbound : int, optional
        Lower bound for the window size search range. Default is 10.
    threshold : float, optional
        The target similarity score between window size and global summary statistics
        between 0 and 1. Default is 0.89.

    Returns
    -------
    int
        The window size for the time series.
    """
    check_input_time_series(time_series)

    if time_series.shape[0] < lbound:
        warnings.warn(
            f"Time series must at least have lbound much data points. Using window_size={time_series.shape[0]}.")
        return time_series.shape[0]

    if time_series.min() == time_series.max():
        warnings.warn(f"Constant time series. Using window_size={lbound}.")
        return lbound

    time_series = (time_series - time_series.min()) / (time_series.max() - time_series.min())

    ts_mean = np.mean(time_series)
    ts_std = np.std(time_series)
    ts_min_max = np.max(time_series) - np.min(time_series)

    stats = (ts_mean, ts_std, ts_min_max)

    max_score = _suss_score(time_series, 1, stats)
    min_score = _suss_score(time_series, time_series.shape[0] - 1, stats)

    if min_score == max_score:
        warnings.warn(f"Found constant SuSS scores. Using window_size={lbound}.")
        return lbound

    exp = 0

    # exponential search (to find window size interval)
    while True:
        window_size = 2 ** exp

        if window_size < lbound:
            exp += 1
            continue

        score = 1 - (_suss_score(time_series, window_size, stats) - min_score) / (max_score - min_score)

        if score > threshold:
            break

        exp += 1

    lbound, ubound = max(lbound, 2 ** (exp - 1)), 2 ** exp + 1

    # binary search (to find window size in interval)
    while lbound <= ubound:
        window_size = int((lbound + ubound) / 2)
        score = 1 - (_suss_score(time_series, window_size, stats) - min_score) / (max_score - min_score)

        if score < threshold:
            lbound = window_size + 1
        elif score > threshold:
            ubound = window_size - 1
        else:
            break

    return 2 * lbound


def dominant_fourier_frequency(time_series, lbound=10, ubound=1000):
    """
    Find the dominant Fourier frequency of the time series within a window size range.

    Parameters
    ----------
    time_series : array-like
        The input time series.
    lbound : int, optional
        The lower bound of the window size range. Default is 10.
    ubound : int, optional
        The upper bound of the window size range. Default is 1000.

    Returns
    -------
    int
        The dominant Fourier frequency's corresponding window size within the specified range.
    """
    check_input_time_series(time_series)

    if time_series.shape[0] < 2 * lbound:
        warnings.warn(
            f"Time series must at least have 2*lbound much data points. Using window_size={time_series.shape[0]}.")
        return time_series.shape[0]

    fourier = np.fft.fft(time_series)
    freq = np.fft.fftfreq(time_series.shape[0], 1)

    magnitudes = []
    window_sizes = []

    for coef, freq in zip(fourier, freq):
        if coef and freq > 0:
            window_size = int(1 / freq)
            mag = math.sqrt(coef.real * coef.real + coef.imag * coef.imag)

            if window_size >= lbound and window_size < ubound:
                window_sizes.append(window_size)
                magnitudes.append(mag)

    if len(magnitudes) == 0:
        warnings.warn(f"Could not extract valid frequencies. Using window_size={lbound}.")
        return lbound

    return window_sizes[np.argmax(magnitudes)]


def highest_autocorrelation(time_series, lbound=10, ubound=1000):
    """
    Computes the highest autocorrelation of a time series within a specified window.

    Parameters
    ----------
    time_series : array-like of shape (n_samples,)
        The input time series.

    lbound : int, optional (default=10)
        The lower bound of the window size to search for the highest autocorrelation.

    ubound : int, optional (default=1000)
        The upper bound of the window size to search for the highest autocorrelation.

    Returns
    -------
    int
        The window size corresponding to the highest autocorrelation within the specified bounds.
        Returns lbound if no peaks are found within the specified bounds.
    """
    check_input_time_series(time_series)

    if time_series.shape[0] < lbound:
        warnings.warn(
            f"Time series must at least have lbound much data points. Using window_size={time_series.shape[0]}.")
        return time_series.shape[0]

    acf_values = acf(time_series, fft=True, nlags=int(time_series.shape[0] / 2))

    peaks, _ = find_peaks(acf_values)
    peaks = peaks[np.logical_and(peaks >= lbound, peaks < ubound)]
    corrs = acf_values[peaks]

    if peaks.shape[0] == 0:
        warnings.warn(f"Could not find any autocorrelation peaks. Using window_size={lbound}.")
        return lbound

    return peaks[np.argmax(corrs)]


_WINDOW_SIZE_MAPPING = {
    "suss": suss,
    "fft": dominant_fourier_frequency,
    "acf": highest_autocorrelation
}


def map_window_size_methods(window_size_method):
    """
    Maps the given window size method to its corresponding implementation.

    Parameters
    ----------
    window_size_method : str
        The name of the window size method to be mapped.

    Returns
    -------
    function
        The implementation of the given window size method.

    Raises
    ------
    ValueError
        If the given window size method is not a valid window size method.
        Valid implementations include: 'suss', 'fft', and 'acf'.

    """
    if window_size_method not in _WINDOW_SIZE_MAPPING:
        raise ValueError(
            f"{window_size_method} is not a valid window size method. Implementations include: {', '.join(_WINDOW_SIZE_MAPPING.keys())}")

    return _WINDOW_SIZE_MAPPING[window_size_method]
