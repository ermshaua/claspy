import numpy as np
from numba import njit

from claspy.nearest_neighbour import KSubsequenceNeighbours
from claspy.scoring import map_scores


@njit(fastmath=True, cache=True)
def cross_val_labels(offsets, split_idx, window_size):
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


@njit(fastmath=True, cache=False)
def _profile(offsets, window_size, score, min_seg_size):
    n_timepoints, _ = offsets.shape
    profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float64)

    for split_idx in range(min_seg_size, n_timepoints - min_seg_size):
        y_true, y_pred = cross_val_labels(offsets, split_idx, window_size)
        profile[split_idx] = score(y_true, y_pred)

    return profile


class ClaSP:

    def __init__(self, window_size=10, k_neighbours=3, score="roc_auc", excl_radius=5):
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.score_name = score
        self.score = map_scores(score)
        self.excl_radius = excl_radius

    def fit(self, time_series, knn=None):
        self.min_seg_size = self.window_size * self.excl_radius

        if time_series.shape[0] < 2 * self.min_seg_size:
            raise ValueError("Time series must at least have 2*min_seg_size data points.")

        self.time_series = time_series

        if knn is None:
            self.knn = KSubsequenceNeighbours(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours
            ).fit(time_series)
        else:
            self.knn = knn

        self.profile = _profile(self.knn.offsets, self.window_size, self.score, self.min_seg_size)
        return self

    def split(self, sparse=True):
        cp = np.argmax(self.profile)

        if sparse is True:
            return cp

        return self.time_series[:cp], self.time_series[cp:]


class ClaSPEnsemble(ClaSP):

    def __init__(self, n_estimators=10, window_size=10, k_neighbours=3, score="roc_auc", excl_radius=5,
                 random_state=2357):
        super().__init__(window_size, k_neighbours, score, excl_radius)
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _calculate_temporal_constraints(self):
        tcs = [(0, self.time_series.shape[0])]
        np.random.seed(self.random_state)

        while len(tcs) < self.n_estimators and self.time_series.shape[0] > 3 * self.min_seg_size:
            lbound, area = np.random.choice(self.time_series.shape[0], 2, replace=True)

            if self.time_series.shape[0] - lbound < area:
                area = self.time_series.shape[0] - lbound

            ubound = lbound + area
            if ubound - lbound < 2 * self.min_seg_size: continue
            tcs.append((lbound, ubound))

        return np.asarray(tcs, dtype=np.int64)

    def fit(self, time_series, knn=None):
        self.min_seg_size = self.window_size * self.excl_radius

        if time_series.shape[0] < 2 * self.min_seg_size:
            raise ValueError("Time series must at least have 2*min_seg_size data points.")

        self.time_series = time_series
        tcs = self._calculate_temporal_constraints()

        if knn is None:
            knn = KSubsequenceNeighbours(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
                temporal_constraints=tcs,
            ).fit(time_series)

        best_score, best_tc, best_clasp = -np.inf, None, None

        for idx, (lbound, ubound) in enumerate(tcs):
            clasp = ClaSP(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
                score=self.score_name,
                excl_radius=self.excl_radius
            ).fit(time_series[lbound:ubound], knn=knn.constrain(lbound, ubound))

            clasp.profile = (2 * clasp.profile + (ubound - lbound) / time_series.shape[0]) / 3

            if clasp.profile.max() > best_score or best_clasp is None and idx == tcs.shape[0]-1:
                best_score = clasp.profile.max()
                best_tc = (lbound, ubound)
                best_clasp = clasp

        self.knn = best_clasp.knn
        self.profile = np.full(shape=time_series.shape[0]-self.window_size+1, fill_value=-np.inf, dtype=np.float64)

        lbound, ubound = best_tc
        self.profile[lbound:ubound-self.window_size+1] = best_clasp.profile

        return self
