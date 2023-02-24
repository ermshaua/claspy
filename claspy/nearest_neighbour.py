import numpy as np
import numpy.fft as fft
from numba import njit


def _sliding_dot(query, time_series):
    '''
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
    '''
    m = len(query)
    n = len(time_series)

    time_series_add = 0
    if n % 2 == 1:
        time_series = np.insert(time_series, 0, 0)
        time_series_add = 1

    q_add = 0
    if m % 2 == 1:
        query = np.insert(query, 0, 0)
        q_add = 1

    query = query[::-1]
    query = np.pad(query, (0, n - m + time_series_add - q_add), 'constant')
    trim = m - 1 + time_series_add
    dot_product = fft.irfft(fft.rfft(time_series) * fft.rfft(query))
    return dot_product[trim:]


@njit(fastmath=True, cache=True)
def _sliding_mean_std(time_series, window_size):
    '''
    Calculate sliding mean and standard deviation of a time series.
    The sliding mean and standard deviation are calculated by computing
    the mean and standard deviation over a sliding window of fixed size,
    which is moved over the time series with a stride of one element at
    a time.

    Parameters
    ----------
    time_series : array-like of shape (n,)
        The time series sequence.
    window_size : int
        The size of the sliding window.

    Returns
    -------
    movmean : ndarray of shape (n - window_size + 1,)
        The sliding mean of the time series.
    movstd : ndarray of shape (n - window_size + 1,)
        The sliding standard deviation of the time series.

    Notes
    -----
    This function calculates the sliding mean and standard deviation of
    the input time series using a sliding window approach. It first computes
    the cumulative sum and cumulative sum of squares of the time series, then
    computes the window sum and window sum of squares for each sliding window.
    Finally, it computes the mean and standard deviation over each window and
    returns the results.

    Examples
    --------
    >>> time_series = [1, 2, 3, 4, 5, 6, 7]
    >>> window_size = 3
    >>> movmean, movstd = _sliding_mean_std(time_series, window_size)
    '''
    s = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(time_series)))
    sSq = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(time_series ** 2)))

    segSum = s[window_size:] - s[:-window_size]
    segSumSq = sSq[window_size:] - sSq[:-window_size]

    movmean = segSum / window_size
    # todo: handle this as implemented in sktime
    movstd = np.sqrt(1e-9 + segSumSq / window_size - (segSum / window_size) ** 2)

    return [movmean, movstd]


@njit(fastmath=True, cache=True)
def _argkmin(dist, k):
    '''
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
    '''
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
def _knn(time_series, window_size, k_neighbours, tcs, dot_first):
    '''
    Perform k-nearest neighbors search between all pairs of subsequences of `time_series`
    of length `window_size`, based on their Euclidean distance after normalization by mean and
    standard deviation. Uses a dot product method for fast calculation.

    Parameters
    ----------
    time_series : ndarray of shape (n,)
        The time series to search over.
    window_size : int
        The length of the sliding window for comparison.
    k_neighbours : int
        The number of nearest neighbors to return for each query.
    tcs : ndarray of shape (m, 2), where each row is (start, end)
        Temporal constraints to consider.
    dot_first : ndarray of shape (n - window_size + 1,)
        Dot product of the first sliding window with itself.

    Returns
    -------
    dists : ndarray of shape (l, m * k_neighbours)
        Array of distances between subsequences, sorted in increasing order.
    knns : ndarray of shape (l, m * k_neighbours)
        Array of indices of k nearest neighbors for each subsequence.
    '''
    l = len(time_series) - window_size + 1
    exclusion_radius = np.int64(window_size / 2)

    knns = np.zeros(shape=(l, len(tcs) * k_neighbours), dtype=np.int64)
    dists = np.zeros(shape=(l, len(tcs) * k_neighbours), dtype=np.float64)

    dot_prev = None
    means, stds = _sliding_mean_std(time_series, window_size)

    for order in range(0, l):
        if order == 0:
            dot_rolled = dot_first
        else:
            dot_rolled = np.roll(dot_prev, 1) + time_series[order + window_size - 1] * time_series[
                                                                                       window_size - 1:l + window_size] - \
                         time_series[order - 1] * np.roll(time_series[:l], 1)
            dot_rolled[0] = dot_first[order]

        x_mean = means[order]
        x_std = stds[order]

        # dist is squared
        dist = 2 * window_size * (1 - (dot_rolled - window_size * means * x_mean) / (window_size * stds * x_std))

        # self-join: exclusion zone
        trivialMatchRange = (
            int(max(0, order - np.round(exclusion_radius, 0))),
            int(min(order + np.round(exclusion_radius + 1, 0), l))
        )

        dist[trivialMatchRange[0]:trivialMatchRange[1]] = np.max(dist)

        for kdx, (lbound, ubound) in enumerate(tcs):
            if order < lbound or order >= ubound: continue

            tc_nn = lbound + _argkmin(dist[lbound:ubound-window_size+1], k_neighbours)

            knns[order, kdx * k_neighbours:(kdx + 1) * k_neighbours] = tc_nn
            dists[order, kdx * k_neighbours:(kdx + 1) * k_neighbours] = dist[tc_nn]

        dot_prev = dot_rolled

    return dists, knns


class KSubsequenceNeighbours:
    '''
    Class implementing the K-Subsequence Neighbours algorithm.

    Parameters
    ----------
    window_size : int, optional (default=10)
        Length of subsequence window.
    k_neighbours : int, optional (default=3)
        Number of nearest neighbors to return for each time series subsequence.

    Methods
    -------
    fit(time_series)
        Fit the KSN model to the input time series data.
    constrain(self, lbound, ubound)
        Return a constrained KSN model for the given temporal constraint.
    '''
    def __init__(self, window_size=10, k_neighbours=3):
        self.window_size = window_size
        self.k_neighbours = k_neighbours

    def fit(self, time_series, temporal_constraints=None):
        '''
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
        '''
        self.time_series = time_series

        if temporal_constraints is None:
            self.temporal_constraints = np.asarray([(0, time_series.shape[0])], dtype=np.int64)
        else:
            self.temporal_constraints = temporal_constraints

        dot_first = _sliding_dot(time_series[:self.window_size], time_series)
        self.distances, self.offsets = _knn(time_series, self.window_size, self.k_neighbours, self.temporal_constraints,
                                            dot_first)
        return self

    def constrain(self, lbound, ubound):
        '''
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
        '''
        if (lbound, ubound) not in self.temporal_constraints:
            raise ValueError(f"({lbound},{ubound}) is not a valid temporal constraint.")

        for idx, tc in enumerate(self.temporal_constraints):
            if tuple(tc) == (lbound, ubound):
                tc_idx = idx

        ts = self.time_series[lbound:ubound]
        distances = self.distances[lbound:ubound-self.window_size+1, tc_idx * self.k_neighbours:(tc_idx + 1) * self.k_neighbours]
        offsets = self.offsets[lbound:ubound-self.window_size+1, tc_idx * self.k_neighbours:(tc_idx + 1) * self.k_neighbours] - lbound

        knn = KSubsequenceNeighbours(
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
        )

        knn.time_series = ts
        knn.temporal_constraints = np.asarray([(0, ts.shape[0])], dtype=np.int64)
        knn.distances, knn.offsets = distances, offsets
        return knn
