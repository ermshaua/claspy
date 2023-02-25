import unittest

import numpy as np

from claspy.nearest_neighbour import KSubsequenceNeighbours
from claspy.tests.tssb_data_loader import load_tssb_dataset


class KSubsequenceNeighboursTest(unittest.TestCase):

    def test_tssb_benchmark(self):
        tssb = load_tssb_dataset()

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            knn = KSubsequenceNeighbours(window_size=window_size)
            knn.fit(time_series)

            for arr in (knn.distances, knn.offsets):
                assert arr.shape[0] == time_series.shape[0] - knn.window_size + 1
                assert arr.shape[1] == knn.k_neighbours

    def test_constraints(self):
        tssb = load_tssb_dataset()

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            tcs = np.arange(0, time_series.shape[0], int(time_series.shape[0] / 5))
            tcs = [(tcs[idx], tcs[idx + 1]) for idx in range(0, len(tcs) - 1)]

            knn = KSubsequenceNeighbours(window_size=window_size)
            knn.fit(time_series, temporal_constraints=tcs)

            for arr in (knn.distances, knn.offsets):
                assert arr.shape[0] == time_series.shape[0] - knn.window_size + 1
                assert arr.shape[1] == len(tcs) * knn.k_neighbours

            for tc in tcs:
                tc_knn = knn.constrain(*tc)

                for arr in (tc_knn.distances, tc_knn.offsets):
                    assert arr.shape[0] == time_series.shape[0] - tc_knn.window_size + 1
                    assert arr.shape[1] == tc_knn.k_neighbours
