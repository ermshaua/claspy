import numpy as np
from queue import PriorityQueue

from claspy.clasp import ClaSP, ClaSPEnsemble


class BinaryClaSPSegmentation:

    def __init__(self, n_segments=1, n_estimators=10, window_size=10, k_neighbours=3, score="roc_auc", excl_radius=5,
                 random_state=2357):
        self.n_segments = n_segments
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.score_name = score
        self.score = score
        self.excl_radius = excl_radius
        self.random_state = random_state

    def fit(self, time_series):
        self.min_seg_size = self.window_size * self.excl_radius
        queue = PriorityQueue()
        clasp_tree = []

        prange = 0, time_series.shape[0]
        clasp = ClaSPEnsemble(
            n_estimators=self.n_estimators,
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
            score=self.score,
            excl_radius=self.excl_radius,
            random_state=self.random_state
        ).fit(time_series)

        clasp_tree.append((prange, clasp))
        queue.put((-clasp.profile[clasp.split()], len(clasp_tree)))

        profile = clasp.profile
        change_points = []
        scores = []

        for idx in range(self.n_segments-1):
            priority, clasp_tree_idx = queue.get()
            (lbound, ubound), clasp = clasp_tree[clasp_tree_idx]

            change_points.append(clasp.split())
            scores.append(-priority)

            profile[lbound:ubound-self.window_size+1] = np.max([profile[lbound:ubound-self.window_size+1], clasp.profile])
