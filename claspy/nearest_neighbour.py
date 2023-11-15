import os

import numpy as np
import numpy.fft as fft
from numba import njit, prange, objmode, get_num_threads, set_num_threads
from numba.typed.typedlist import List

from claspy.distance import map_distances
from claspy.utils import check_input_time_series, numba_cache_safe


@njit(fastmath=True, cache=True)
def _sliding_dot(query, time_series):
    """
    Calculate sliding dot product between a query and a time series.
    The sliding dot product (SDP) is a measure of similarity between two sequences,
    which is calculated as the dot product between a query and a window of the time
    series of the same length as the query, and then shifted by one element in the
    time series at a time.

    Parameters
    ----------
    query : array-like of shape (m,)
        The query sequence.
    time_series : array-like of shape (n,)
        The time series sequence.

    Returns
    -------
    dot_product : ndarray of shape (n - m + 1,)
        The dot product between the query and all windows of the time series.

    Notes
    -----
    This function calculates the SDP between the input query and the input time series.
    It first pads the query and the time series with zeros to the same length, then calculates
    the SDP using the Fast Fourier Transform (FFT). Finally, it trims the result to the valid
    range and returns it.

    Examples
    --------
    >>> query = [1, 2, 3]
    >>> time_series = [1, 2, 3, 4, 5, 6, 7]
    >>> dot_product = _sliding_dot(query, time_series)
    """
    m = len(query)
    n = len(time_series)

    time_series_add = 0
    if n % 2 == 1:
        time_series = np.concatenate((np.array([0]), time_series))
        time_series_add = 1

    q_add = 0
    if m % 2 == 1:
        query = np.concatenate((np.array([0]), query))
        q_add = 1

    query = query[::-1]
    query = np.concatenate((query, np.zeros(n - m + time_series_add - q_add)))
    trim = m - 1 + time_series_add
    with objmode(dot_product="float64[:]"):
        dot_product = fft.irfft(fft.rfft(time_series) * fft.rfft(query))
    return dot_product[trim:]


@njit(fastmath=True, cache=True)
def _argkmin(dist, k):
    """
    Compute the indices of the k smallest elements in a numpy array.
    This function computes the indices of the k smallest elements in a
    numpy array, and returns them in an array.

    Parameters
    ----------
    dist : ndarray of shape (n,)
        The input numpy array of distances.
    k : int
        The number of smallest elements to return.

    Returns
    -------
    args : ndarray of shape (k,)
        The indices of the k smallest elements in the input numpy array.

    Notes
    -----
    This function computes the indices of the k smallest elements in the
    input numpy array using a nested loop approach. It first initializes
    two arrays to store the indices and values of the k smallest elements,
    and then iteratively loops over the input array to find the k smallest
    elements. Each time a new smallest element is found, its index and value
    are stored in the arrays, and the corresponding element in the input array
    is replaced with infinity to prevent it from being selected again. The
    function returns the array of indices of the k smallest elements.

    Examples
    --------
    >>> dist = [3, 2, 1, 4, 5]
    >>> k = 3
    >>> args = _argkmin(dist, k)
    >>> args
    array([2, 1, 0])
    """
    args = np.zeros(shape=k, dtype=np.int64)
    vals = np.zeros(shape=k, dtype=np.float64)

    for idx in range(k):
        min_arg = np.nan
        min_val = np.inf

        for kdx, val in enumerate(dist):
            if val < min_val:
                min_val = val
                min_arg = kdx

        min_arg = np.int64(min_arg)

        args[idx] = min_arg
        vals[idx] = min_val

        dist[min_arg] = np.inf

    dist[args] = vals
    return args


@njit(fastmath=True, cache=True)
def _knn(time_series, start, end, window_size, k_neighbours, tcs, dot_first, dot_ref, distance, distance_preprocessing):
    """
    Perform k-nearest neighbors search between all pairs of subsequences of `time_series`
    of length `window_size`, based on their Euclidean distance after normalization by mean and
    standard deviation. Uses a dot product method for fast calculation.

    Parameters
    ----------
    time_series : ndarray of shape (n,)
        The time series to search over.
    start : int
        The first index to consider (inclusive).
    end : int
        The last index to consider (exclusive).
    window_size : int
        The length of the sliding window for comparison.
    k_neighbours : int
        The number of nearest neighbors to return for each query.
    tcs : ndarray of shape (m, 2), where each row is (start, end)
        Temporal constraints to consider.
    dot_first : ndarray of shape (n - window_size + 1,)
        Dot product of the sliding window at start with itself.
    dot_ref : ndarray of shape (n - window_size + 1,)
        Dot product of the first sliding window with itself.
    distance: callable
        The distance function to be computed.
    distance_preprocessing: callable
        The distance preprocessing function to be computed.

    Returns
    -------
    dists : ndarray of shape (l, m * k_neighbours)
        Array of distances between subsequences, sorted in increasing order.
    knns : ndarray of shape (l, m * k_neighbours)
        Array of indices of k nearest neighbors for each subsequence.
    """
    l = len(time_series) - window_size + 1
    exclusion_radius = np.int64(window_size / 2)

    knns = np.zeros(shape=(end-start, len(tcs) * k_neighbours), dtype=np.int64)
    dists = np.zeros(shape=(end-start, len(tcs) * k_neighbours), dtype=np.float64)

    dot_prev = None
    preprocessing = distance_preprocessing(time_series, window_size)

    for order in range(start, end):
        if order == start:
            dot_rolled = dot_first
        else:
            dot_rolled = np.roll(dot_prev, 1) \
                         + time_series[order + window_size - 1] * time_series[window_size - 1:l + window_size] \
                         - time_series[order - 1] * np.roll(time_series[:l], 1)
            dot_rolled[0] = dot_ref[order]

        dist = distance(order, dot_rolled, window_size, preprocessing)

        # self-join: exclusion zone
        trivialMatchRange = (
            int(max(0, order - np.round(exclusion_radius, 0))),
            int(min(order + np.round(exclusion_radius + 1, 0), l))
        )

        dist[trivialMatchRange[0]:trivialMatchRange[1]] = np.max(dist)

        for kdx, (lbound, ubound) in enumerate(tcs):
            if order < lbound or order >= ubound: continue
            tc_nn = lbound + _argkmin(dist[lbound:ubound - window_size + 1], k_neighbours)

            knns[order-start, kdx * k_neighbours:(kdx + 1) * k_neighbours] = tc_nn
            dists[order-start, kdx * k_neighbours:(kdx + 1) * k_neighbours] = dist[tc_nn]

        dot_prev = dot_rolled

    return dists, knns


@njit(fastmath=True, cache=True, parallel=True)
def _parallel_knn(time_series, window_size, k_neighbours, pranges, tcs, distance, distance_preprocessing):
    """
    Perform k-nearest neighbors search between all pairs of subsequences of `time_series`
    of length `window_size` in parallel with n_jobs threads.

    Parameters
    ----------
    time_series : ndarray of shape (n,)
        The time series to search over.
    window_size : int
        The length of the sliding window for comparison.
    k_neighbours : int
        The number of nearest neighbors to return for each query.
    pranges : ndarray of shape (m, 2), where each row is (start, end)
        Ranges in which the k-NNs are calculated per thread. Infers the number of threads.
    tcs : ndarray of shape (m, 2), where each row is (start, end)
        Temporal constraints to consider.
    distance: callable
        The distance function to be computed.
    distance_preprocessing: callable
        The distance preprocessing function to be computed.

    Returns
    -------
    dists : ndarray of shape (l, m * k_neighbours)
        Array of distances between subsequences, sorted in increasing order.
    knns : ndarray of shape (l, m * k_neighbours)
        Array of indices of k nearest neighbors for each subsequence.
    """
    dot_firsts = np.zeros(shape=(len(pranges), len(time_series) - window_size + 1), dtype=np.float64)
    knns = np.zeros(shape=(len(time_series) - window_size + 1, len(tcs) * k_neighbours), dtype=np.int64)
    dists = np.zeros(shape=(len(time_series) - window_size + 1, len(tcs) * k_neighbours), dtype=np.float64)

    for idx in prange(len(pranges)):
        start, end = pranges[idx]
        dot_firsts[idx] = _sliding_dot(time_series[start:start + window_size], time_series)

    for idx in prange(len(pranges)):
        start, end = pranges[idx]

        dists[start:end, :], knns[start:end, :] = _knn(
            time_series,
            start,
            end,
            window_size,
            k_neighbours,
            tcs,
            dot_firsts[idx],
            dot_firsts[0],
            distance,
            distance_preprocessing
        )

    return dists, knns


@njit(fastmath=True, cache=True)
def cross_val_labels(offsets, split_idx, window_size):
    """
    Generate predicted and true labels for cross-validation based on nearest neighbour distances.

    Parameters
    ----------
    offsets : ndarray of shape (n_timepoints, k_neighbours)
        The indices of the nearest neighbours for each timepoint in the time series. These indices
        are relative to the start of the time series and should be positive integers.
    split_idx : int
        The index at which to split the time series into two potential segments. This index should be
        less than n_timepoints and greater than window_size.
    window_size : int
        The size of the window used to calculate nearest neighbours.

    Returns
    -------
    y_true : ndarray of shape (n_timepoints,)
        The true labels for each timepoint in the time series.
    y_pred : ndarray of shape (n_timepoints,)
        The predicted labels for each timepoint in the time series.
    """
    n_timepoints, k_neighbours = offsets.shape

    y_true = np.concatenate((
        np.zeros(split_idx, dtype=np.int64),
        np.ones(n_timepoints - split_idx, dtype=np.int64),
    ))

    knn_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

    for i_neighbor in range(k_neighbours):
        neighbours = offsets[:, i_neighbor]
        knn_labels[i_neighbor] = y_true[neighbours]

    ones = np.sum(knn_labels, axis=0)
    zeros = k_neighbours - ones
    y_pred = np.asarray(ones > zeros, dtype=np.int64)

    exclusion_zone = np.arange(split_idx - window_size, split_idx)
    y_pred[exclusion_zone] = 1

    return y_true, y_pred


class KSubsequenceNeighbours:
    """
    Class implementing the K-Subsequence Neighbours algorithm.

    Parameters
    ----------
    window_size : int, optional (default=10)
        Length of subsequence window.
    k_neighbours : int, optional (default=3)
        Number of nearest neighbors to return for each time series subsequence.
    distance: str
        The name of the distance function to be computed. Available options are "znormed_euclidean_distance"
        and "euclidean_distance".
    n_jobs : int, optional (default=1)
        Amount of threads used in the k-nearest neighbour calculation.

    Methods
    -------
    fit(time_series)
        Fit the KSN model to the input time series data.
    constrain(self, lbound, ubound)
        Return a constrained KSN model for the given temporal constraint.
    """

    def __init__(self, window_size=10, k_neighbours=3, distance="znormed_euclidean_distance", n_jobs=-1):
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.distance_name = distance
        self.distance_preprocessing, self.distance = map_distances(distance)
        self.n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs

    def fit(self, time_series, temporal_constraints=None):
        """
        Fits the k-subsequence neighbors model to the provided time series.

        Parameters
        ----------
        time_series : array-like
            The time series to fit the model to.

        temporal_constraints : array-like of shape (n, 2), optional (default=None)
            Temporal constraints for the subsequences. Each row defines a temporal constraint as
            a pair of integers (l, u), such that the model is only allowed to match subsequences
            starting at positions that are greater than or equal to l and smaller than u. If None,
            a single constraint spanning the entire length of the time series is used.

        Returns
        -------
        self : KSubsequenceNeighbours
            A reference to the fitted model.
        """
        check_input_time_series(time_series)

        if time_series.shape[0] < self.window_size * self.k_neighbours:
            raise ValueError("Time series must at least have k_neighbours*window_size data points.")

        self.time_series = time_series

        if temporal_constraints is None:
            self.temporal_constraints = np.asarray([(0, time_series.shape[0])], dtype=int)
        else:
            self.temporal_constraints = temporal_constraints

        pranges = List()
        n_jobs = self.n_jobs

        while time_series.shape[0] // n_jobs < self.window_size * self.k_neighbours and n_jobs != 1:
            n_jobs -= 1

        bin_size = time_series.shape[0] // n_jobs

        for idx in range(n_jobs):
            start = idx * bin_size
            end = min((idx + 1) * bin_size, len(time_series)-self.window_size+1)
            if end > start: pranges.append((start, end))

        n_threads = get_num_threads()
        set_num_threads(n_jobs)

        self.distances, self.offsets = numba_cache_safe(_parallel_knn, time_series, self.window_size, self.k_neighbours,
                                                     pranges, List(self.temporal_constraints),
                                                     self.distance, self.distance_preprocessing)

        set_num_threads(n_threads)
        return self

    def constrain(self, lbound, ubound):
        """
        Constrain the k-nearest neighbours search to a specific range.

        Parameters
        ----------
        lbound : int
            The lower bound of the subsequence to constrain the k-nearest neighbours search to.
        ubound : int
            The upper bound of the subsequence to constrain the k-nearest neighbours search to.

        Returns
        -------
        KSubsequenceNeighbours
            A new instance of KSubsequenceNeighbours class with the k-nearest neighbours search constrained to the
            range defined by the lbound and ubound indices.

        Raises
        ------
        ValueError
            If the (lbound, ubound) tuple is not a valid temporal constraint.
        """
        if (lbound, ubound) not in self.temporal_constraints:
            raise ValueError(f"({lbound},{ubound}) is not a valid temporal constraint.")

        for idx, tc in enumerate(self.temporal_constraints):
            if tuple(tc) == (lbound, ubound):
                tc_idx = idx

        ts = self.time_series[lbound:ubound]
        distances = self.distances[lbound:ubound - self.window_size + 1,
                    tc_idx * self.k_neighbours:(tc_idx + 1) * self.k_neighbours]
        offsets = self.offsets[lbound:ubound - self.window_size + 1,
                  tc_idx * self.k_neighbours:(tc_idx + 1) * self.k_neighbours] - lbound

        knn = KSubsequenceNeighbours(
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
            distance=self.distance_name
        )

        knn.time_series = ts
        knn.temporal_constraints = np.asarray([(0, ts.shape[0])], dtype=int)
        knn.distances, knn.offsets = distances, offsets
        return knn
