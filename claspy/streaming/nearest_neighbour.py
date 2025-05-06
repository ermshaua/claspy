import numpy as np
from numba import njit

from claspy.distance import map_distances
from claspy.nearest_neighbour import _sliding_dot, _argkmin, KSubsequenceNeighbours
from claspy.utils import roll_array


@njit(fastmath=True, cache=True)
def _sliding_mean(idx, csum, window_size):
    """
    Computes the mean of a sliding window over a time series using a cumulative sum array.

    This function calculates the mean value of a fixed-size subsequence (window) in a time series,
    starting at a given index. It uses the cumulative sum array for efficient computation, avoiding
    explicit summation over the window elements.

    Parameters
    ----------
    idx : int
        Starting index of the sliding window in the original time series.
    csum : np.ndarray
        Cumulative sum array of the time series, where csum[i] = sum(time_series[:i]).
    window_size : int
        Length of the sliding window.

    Returns
    -------
    mean : float
        The mean value of the subsequence starting at the given index.
    """
    window_sum = csum[idx + window_size] - csum[idx]
    return window_sum / window_size


@njit(fastmath=True, cache=True)
def _sliding_std(idx, csumsq, csum, window_size):
    """
    Computes the standard deviation of a sliding window over a time series using cumulative sums.

    This function efficiently calculates the standard deviation of a fixed-size subsequence
    in a time series by using precomputed cumulative sums and squared cumulative sums. A numerical
    safeguard ensures stability by replacing very small standard deviations with 1 to avoid division
    issues in later normalization steps.

    Parameters
    ----------
    idx : int
        Starting index of the sliding window in the time series.
    csumsq : np.ndarray
        Cumulative sum of squares array, where csumsq[i] = sum(time_series[:i]**2).
    csum : np.ndarray
        Cumulative sum array, where csum[i] = sum(time_series[:i]).
    window_size : int
        Length of the sliding window.

    Returns
    -------
    movstd : float or np.ndarray
        The standard deviation of the subsequence starting at the given index. Values close to zero
        are clipped to 1 for numerical stability.
    """
    window_sum = csum[idx + window_size] - csum[idx]
    window_sum_sq = csumsq[idx + window_size] - csumsq[idx]

    movstd = window_sum_sq / window_size - (window_sum / window_size) ** 2

    # should not happen, but just in case
    if movstd < 0: return 1

    movstd = np.sqrt(movstd)

    # avoid dividing by too small std, like 0
    if abs(movstd) < 1e-3: return 1

    return movstd


@njit(fastmath=True, cache=True)
def _roll_sliding_window(time_series, timepoint, n_filled, csum, csumsq, dcsum, window_size, means, stds):
    """
    Updates and rolls time series statistics for streaming computation.

    This function performs a one-step update for a streaming time series by shifting the arrays
    representing the raw data and its statistics (cumulative sums, means, standard deviations).
    It incorporates the latest data point and updates derived statistics efficiently using
    cumulative sums and previous values.

    Parameters
    ----------
    time_series : np.ndarray
        Array containing the current streaming time series data.
    timepoint : float
        The new data point to be appended to the time series.
    n_filled : int
        Number of time points filled so far; used to decide when to start computing statistics.
    csum : np.ndarray
        Cumulative sum array for the time series.
    csumsq : np.ndarray
        Cumulative sum of squares array for variance calculation.
    dcsum : np.ndarray
        Cumulative squared difference from previous time point (used in cinvariant distance).
    window_size : int
        Length of the window used for mean and standard deviation calculations.
    means : np.ndarray
        Array storing sliding means of the time series.
    stds : np.ndarray
        Array storing sliding standard deviations of the time series.

    Returns
    -------
    time_series : np.ndarray
        Updated and rolled time series array.
    csum : np.ndarray
        Updated cumulative sum array.
    csumsq : np.ndarray
        Updated cumulative sum of squares array.
    dcsum : np.ndarray
        Updated cumulative squared difference array.
    means : np.ndarray
        Updated sliding means array.
    stds : np.ndarray
        Updated sliding standard deviations array.
    """
    time_series = roll_array(time_series, -1, timepoint)

    csum = roll_array(csum, -1, csum[-1] + timepoint)
    csumsq = roll_array(csumsq, -1, csumsq[-1] + timepoint ** 2)

    if n_filled > 1:
        dcsum = roll_array(dcsum, -1, dcsum[-1] + np.square(timepoint - time_series[-2]))

    if n_filled >= window_size:
        means = roll_array(means, -1, _sliding_mean(len(time_series) - window_size, csum, window_size))
        stds = roll_array(stds, -1, _sliding_std(len(time_series) - window_size, csumsq, csum, window_size))

    return time_series, csum, csumsq, dcsum, means, stds


@njit(fastmath=True, cache=True)
def _knn(time_series, idx, n_windows, n_filled, window_size, dot_rolled, first_flag, distance, preprocessing,
         exclusion_radius, k_neighbours, lbound):
    """
    Computes k-nearest neighbors (k-NNs) for a subsequence in a time series using dot product optimization.

    This function calculates the distances between a query window and all valid subsequences in the time
    series. It applies optimizations using rolling dot products to reduce computation and excludes a
    region around the query index to avoid trivial matches. It returns updated dot products, distances,
    and the indices of the k-nearest neighbors.

    Parameters
    ----------
    time_series : np.ndarray
        The full 1D time series from which subsequences are compared.
    idx : int
        The index of the current query subsequence for which neighbors are being found.
    n_windows : int
        Total number of valid subsequences in the time series.
    n_filled : int
        Number of previous k-NN computations used for warm-starting the rolling dot product.
    window_size : int
        Length of each subsequence (sliding window).
    dot_rolled : np.ndarray
        A preallocated array to store rolling dot products used for fast distance computation.
    first_flag : bool
        Indicates whether this is the first invocation, requiring a full dot product computation.
    distance : callable
        Function that transforms dot products into a distance metric.
    preprocessing : tuple
        A tuple containing precomputed arrays required for the distance calculation.
    exclusion_radius : int
        Radius around the query index to exclude from consideration to avoid self-matches.
    k_neighbours : int
        Number of nearest neighbors to return.
    lbound : int
        Lower bound for excluding offsets from earlier iterations.

    Returns
    -------
    dot_rolled : np.ndarray
        Updated rolling dot product array.
    dist : np.ndarray
        A 1D array of shape (n_windows,) containing distances between the query and each subsequence.
    knns : np.ndarray
        Indices of the k-nearest neighbors to the query subsequence.
    """
    start_idx = lbound - 1
    valid_dist = slice(start_idx, n_windows)
    dist = np.full(shape=n_windows, fill_value=np.inf, dtype=np.float64)

    if first_flag:
        dot_rolled[valid_dist] = _sliding_dot(time_series[idx:idx + window_size], time_series[-n_filled:])
    else:
        dot_rolled = dot_rolled + time_series[idx + window_size - 1] * time_series[window_size - 1:]

        # fill sliding window
        if start_idx >= 0:
            dot_rolled[start_idx] = np.dot(time_series[start_idx:start_idx + window_size],
                                           time_series[idx:idx + window_size])

    rolled_dist = distance(idx, dot_rolled, window_size, preprocessing)
    dist[valid_dist] = rolled_dist[valid_dist]

    excl_range = slice(max(0, idx - exclusion_radius), min(idx + exclusion_radius, n_windows))
    dist[excl_range] = np.max(dist)
    knns = _argkmin(dist, k_neighbours, lbound)

    # update dot product
    dot_rolled -= time_series[idx] * time_series[:n_windows]

    return dot_rolled, dist, knns


@njit(fastmath=True, cache=True)
def _roll_knns(dists, offsets, dist, knn, knn_insert_idx, knn_filled, n_windows, k_neighbours, lbound):
    """
    Incrementally updates k-nearest neighbors (k-NNs) using a rolling distance comparison.

    This function inserts a new set of k-nearest neighbors and their distances into existing
    k-NN structures. It efficiently updates the k-NN indices and distances for all windows
    by comparing the new distances against existing ones and adjusting the entries accordingly.

    Parameters
    ----------
    dists : np.ndarray
        A 2D array of shape (n_windows, k_neighbours) storing current nearest neighbor distances.
    offsets : np.ndarray
        A 2D array of shape (n_windows, k_neighbours) storing indices of the k-nearest neighbors.
    dist : np.ndarray
        A 1D array of shape (n_windows,) containing distances between the new candidate and each window.
    knn : np.ndarray
        A 1D array containing indices of the k-nearest neighbors for the new candidate.
    knn_insert_idx : int
        The index where the new k-NN distances and offsets should be inserted.
    knn_filled : int
        Counter indicating how many k-NN sets have been inserted so far.
    n_windows : int
        Total number of windows/time points being considered.
    k_neighbours : int
        Number of nearest neighbors to maintain for each time point.
    lbound : int
        Lower bound index used to limit updates during early filling phase.

    Returns
    -------
    dists : np.ndarray
        Updated k-NN distance array.
    offsets : np.ndarray
        Updated k-NN index array.
    lbound : int
        Updated lower bound index for the next insertion.
    knn_filled : int
        Updated count of how many k-NN sets have been filled.
    """
    dists[knn_insert_idx, :] = dist[knn]
    offsets[knn_insert_idx, :] = knn

    idx = np.arange(lbound, n_windows)
    change_mask = np.full(shape=n_windows - lbound, fill_value=True, dtype=np.bool_)

    # roll k-NNs
    for kdx in range(k_neighbours - 1):
        change_idx = dist[idx] < dists[idx, kdx]
        change_idx = np.logical_and(change_idx, change_mask[idx - lbound])
        change_idx = idx[change_idx]

        change_mask[change_idx - lbound] = False
        offsets[change_idx, kdx + 1:] = offsets[change_idx, kdx:k_neighbours - 1]
        offsets[change_idx, kdx] = knn_insert_idx

        dists[change_idx, kdx + 1:] = dists[change_idx, kdx:k_neighbours - 1]
        dists[change_idx, kdx] = dist[change_idx]

    # decrease lbound
    lbound = max(0, lbound - 1)

    # log how much knn data is ingested
    knn_filled = min(knn_filled + 1, knn_insert_idx)

    return dists, offsets, lbound, knn_filled


class StreamingKSubsequenceNeighbours:
    """
    Class implementing the K-Subsequence Neighbours algorithm in the streaming setting.

    Parameters
    ----------
    n_timepoints : int, optional (default=10_000)
        Length of sliding time series window.
    window_size : int, optional (default=10)
        Length of subsequence window.
    k_neighbours : int, optional (default=3)
        Number of nearest neighbors to return for each time series subsequence.
    distance: str
        The name of the distance function to be computed. Available options are "znormed_euclidean_distance",
        "cinvariant_euclidean_distance", and "euclidean_distance".

    Methods
    -------
    update(timepoint)
        Update the KSN model for a new time series data point.
    transform()
        Returns a fitted KSubsequenceNeighbours model for the current data.
    update_transform(timepoint)
        Updates and returns a KSN for the new data point. 
    """

    def __init__(self, n_timepoints=10_000, window_size=10, k_neighbours=3, distance="znormed_euclidean_distance"):
        self.window_size = window_size
        self.exclusion_radius = window_size // 2

        self.n_timepoints = n_timepoints
        self.k_neighbours = k_neighbours

        self.distance_name = distance
        _, self.distance = map_distances(distance)

        # counter
        self.lbound = 0
        self.n_filled = 0
        self.knn_filled = 0

        self.n_windows = n_timepoints - window_size + 1
        self.knn_insert_idx = self.n_windows - self.exclusion_radius - self.k_neighbours - 1

        # sliding window + statistics
        self.time_series = np.full(shape=n_timepoints, fill_value=np.nan, dtype=np.float64)

        self.csum = np.full(shape=n_timepoints + 1, fill_value=0, dtype=np.float64)
        self.csumsq = np.full(shape=n_timepoints + 1, fill_value=0, dtype=np.float64)
        self.dcsum = np.full(shape=n_timepoints + 1, fill_value=0, dtype=np.float64)

        self.means = np.full(shape=self.n_windows, fill_value=np.nan, dtype=np.float64)
        self.stds = np.full(shape=self.n_windows, fill_value=np.nan, dtype=np.float64)

        self.dists = np.full(shape=(self.n_windows, k_neighbours), fill_value=np.inf, dtype=np.float64)
        self.knns = np.full(shape=(self.n_windows, k_neighbours), fill_value=-1, dtype=np.int64)

        self.dot_rolled = None

    def update(self, timepoint, change_point=0):
        """
        Incorporates a new time point into the streaming k-NN.

        This method updates the internal state of the streaming k-nearest neighbor (k-NN) model
        applied to the sliding window. It maintains rolling statistics (mean, standard deviation,
        cumulative sums), and efficiently computes k-NNs for the sliding window in an online fashion.

        Parameters
        ----------
        timepoint : float
            The latest data point to append to the sliding window.
        change_point : int
            The last observed change point, if available (default=0)

        Returns
        -------
        self : object
            The updated object containing the revised sliding window and associated k-NN structures.
        """
        self.lbound = self.knn_insert_idx - self.knn_filled + 1 + change_point
        self.n_filled = min(self.n_filled + 1, self.n_windows)

        # update sliding window statistics
        self.time_series, self.csum, self.csumsq, self.dcsum, self.means, self.stds \
            = _roll_sliding_window(self.time_series, timepoint,
                                   self.n_filled, self.csum, self.csumsq, self.dcsum,
                                   self.window_size, self.means, self.stds)

        if self.n_filled < self.window_size + self.exclusion_radius + self.k_neighbours:
            return self

        # shift k-NN indices
        if self.knn_filled > 0:
            self.dists = roll_array(self.dists, -1,
                                    np.full(shape=self.dists.shape[1], fill_value=np.inf, dtype=np.float64))

            self.knns = roll_array(self.knns, -1)
            self.knns[self.knn_insert_idx - self.knn_filled:self.knn_insert_idx] -= 1
            self.knns[-1, :] = np.full(shape=self.dists.shape[1], fill_value=-1, dtype=np.float64)

        first_flag = self.dot_rolled is None

        # Package preprocessing for distance calculation
        if self.distance_name == "znormed_euclidean_distance":
            preprocessing = self.means, self.stds
        elif self.distance_name == "euclidean_distance":
            preprocessing = self.csumsq[self.window_size:] - self.csumsq[:-self.window_size]
        elif self.distance_name == "cinvariant_euclidean_distance":
            csumsq = self.csumsq[self.window_size:] - self.csumsq[:-self.window_size]
            ce = self.dcsum[self.window_size:] - self.dcsum[
                                                 :-self.window_size] + 1e-5  # add constant to prevent zero divisons
            preprocessing = csumsq, ce, self.means, self.stds
        else:
            raise ValueError(f"{self.distance_name} is not a supported distance.")

        if first_flag:
            self.dot_rolled = np.full(shape=self.n_windows, fill_value=np.inf, dtype=np.float64)

        # compute new distances and k-NNs
        self.dot_rolled, dist, knns = _knn(self.time_series, self.knn_insert_idx, self.n_windows, self.n_filled,
                                           self.window_size, self.dot_rolled, first_flag, self.distance, preprocessing,
                                           self.exclusion_radius, self.k_neighbours, self.lbound)

        # insert new k-NNs and update existing ones
        self.dists, self.knns, self.lbound, self.knn_filled = _roll_knns(
            self.dists, self.knns, dist, knns, self.knn_insert_idx,
            self.knn_filled, self.n_windows, self.k_neighbours, self.lbound
        )

        return self

    def transform(self):
        """
        Converts the internal streaming k-NN state into a static KSubsequenceNeighbours object.

        Returns
        -------
        ksn : KSubsequenceNeighbours
            A static object containing the k-NN distances and offsets for currently processed subsequences.
        """
        ksn = KSubsequenceNeighbours(window_size=self.window_size, k_neighbours=self.k_neighbours,
                                     distance=self.distance_name, n_jobs=1)

        ksn.time_series = self.time_series
        ksn.temporal_constraints = np.asarray([(0, ksn.time_series.shape[0])], dtype=int)
        ksn.distances, ksn.offsets = self.dists, self.knns

        ksn.distances = ksn.distances[self.lbound:self.knn_insert_idx]
        ksn.offsets = ksn.offsets[self.lbound:self.knn_insert_idx] - self.lbound
        ksn.offsets = np.clip(ksn.offsets, 0, ksn.offsets.shape[0] - 1)

        return ksn

    def update_transform(self, timepoint):
        """
        Updates the streaming model with a new time point and returns the transformed k-NN structure.

        Parameters
        ----------
        timepoint : float
            The new data point to be added to the sliding window.

        Returns
        -------
        ksn : KSubsequenceNeighbours
            A static object representing the updated k-NN distances and offsets after incorporating
            the new time point.
        """
        return self.update(timepoint).transform()
