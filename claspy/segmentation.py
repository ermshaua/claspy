import numpy as np

from claspy.clasp import ClaSP, ClaSPEnsemble


class BinaryClaSPSegmentation:

    def __init__(self, n_estimators=10, window_size=10, k_neighbours=3, score="roc_auc", min_seg_size=5,
                 random_state=2357):
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.score_name = score
        self.score = score
        self.min_seg_size = min_seg_size
        self.random_state = random_state

    def fit(self, time_series):
        pass