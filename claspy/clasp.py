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

    def __init__(self, window_size=10, k_neighbours=3, score="roc_auc", min_seg_size=5):
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.score = map_scores(score)
        self.min_seg_size = min_seg_size

    def fit(self, time_series, knn=None):
        self.min_seg_size *= self.window_size

        if time_series.shape[0] < 2 * self.min_seg_size:
            raise ValueError("Time series must at least have 2*min_seg_size data points.")

        if knn is None:
            self.knn = KSubsequenceNeighbours(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours
            ).fit(time_series)
        else:
            self.knn = knn

        self.profile = _profile(self.knn.offsets, self.window_size, self.score, self.min_seg_size)
        return self


class ClaSPEnsemble:

    def __init__(self, n_estimators=10, window_size=10, k_neighbours=3, score="roc_auc", min_seg_size=5,
                 random_state=2357):
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.score = score
        self.min_seg_size = min_seg_size
        self.random_state = random_state

    def _calculate_temporal_constraints(self, time_series):
        tcs = [(0, time_series.shape[0])]
        np.random.seed(self.random_state)

        while len(tcs) < self.n_estimators and time_series.shape[0] > 3 * self.min_seg_size:
            lbound, area = np.random.choice(time_series.shape[0], 2, replace=True)

            if time_series.shape[0] - lbound < area:
                area = time_series.shape[0] - lbound

            ubound = lbound + area
            if ubound - lbound < 2 * self.min_seg_size: continue
            tcs.append((lbound, ubound))

        return np.asarray(tcs, dtype=np.int64)

    def fit(self, time_series):
        self.min_seg_size *= self.window_size
        tcs = self._calculate_temporal_constraints(time_series)

        knn = KSubsequenceNeighbours(
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
            temporal_constraints=tcs,
        ).fit(time_series)

        best_score, best_clasp = -np.inf, None

        for (lbound, ubound) in tcs:
            clasp = ClaSP(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
                score=self.score,
                min_seg_size=int(self.min_seg_size / self.window_size)
            ).fit(time_series[lbound:ubound], knn=knn.constrain(lbound, ubound))

            clasp.profile = (2 * clasp.profile + (ubound - lbound) / time_series.shape[0]) / 3

            if clasp.profile.max() > best_score:
                best_score = clasp.profile.max()
                best_clasp = clasp

        self.knn = best_clasp.knn
        self.profile = best_clasp.profile

        return self
