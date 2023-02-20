import unittest

from claspy.tests.tssb_data_loader import load_tssb_dataset
from claspy.nearest_neighbour import KSubsequenceNeighbours


class KSubsequenceNeighboursTest(unittest.TestCase):

    def test_tssb_benchmark(self):
        tssb = load_tssb_dataset()

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            knn = KSubsequenceNeighbours(window_size=window_size)
            knn.fit(time_series)

            for arr in (knn.distances, knn.offsets):
                assert arr.shape[0] == time_series.shape[0]-knn.window_size+1
                assert arr.shape[1] == knn.k_neighbours
