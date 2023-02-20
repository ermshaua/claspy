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

    def fit(self, time_series):
        self.min_seg_size *= self.window_size
        self.knn = KSubsequenceNeighbours(window_size=self.window_size, k_neighbours=self.k_neighbours).fit(time_series)
        self.profile = _profile(self.knn.offsets, self.window_size, self.score, self.min_seg_size)
        return self