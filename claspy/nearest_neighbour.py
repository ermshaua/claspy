import numpy as np
import numpy.fft as fft
from numba import njit


def _sliding_dot(query, time_series):
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

    def __init__(self, window_size=10, k_neighbours=3, temporal_constraints=None):
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.temporal_constraints = temporal_constraints

    def fit(self, time_series):
        self.time_series = time_series

        if self.temporal_constraints is None:
            self.temporal_constraints = np.asarray([(0, time_series.shape[0])], dtype=np.int64)

        dot_first = _sliding_dot(time_series[:self.window_size], time_series)
        self.distances, self.offsets = _knn(time_series, self.window_size, self.k_neighbours, self.temporal_constraints,
                                            dot_first)
        return self

    def constrain(self, lbound, ubound):
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
            temporal_constraints=np.asarray([(0, ts.shape[0])], dtype=np.int64)
        )

        knn.time_series = ts
        knn.distances, knn.offsets = distances, offsets
        return knn
