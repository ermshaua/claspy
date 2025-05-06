import unittest
from itertools import product

import numpy as np

from claspy.data_loader import load_tssb_dataset
from claspy.streaming.nearest_neighbour import StreamingKSubsequenceNeighbours


class StreamingKSubsequenceNeighboursTest(unittest.TestCase):

    def test_tssb_benchmark(self):
        tssb = load_tssb_dataset()
        np.random.seed(2357)
        tssb = tssb.sample(10)

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            knn_stream = StreamingKSubsequenceNeighbours(n_timepoints=1000, window_size=window_size)

            for timepoint in time_series:
                knn_stream.update(timepoint)

            knn = knn_stream.transform()

            for arr in (knn.distances, knn.offsets):
                assert arr.shape[0] == knn_stream.knn_insert_idx - knn_stream.lbound
                assert arr.shape[1] == knn.k_neighbours

    def test_param_configs(self):
        tssb = load_tssb_dataset()
        np.random.seed(2357)
        tssb = tssb.sample(3)

        n_timepoints = (100, 500, 1000)
        k_neighbours = (1, 3, 5)
        distances = ("znormed_euclidean_distance", "euclidean_distance", "cinvariant_euclidean_distance")

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            for n, k, dist in product(n_timepoints, k_neighbours, distances):
                knn_stream = StreamingKSubsequenceNeighbours(n_timepoints=n, window_size=window_size,
                                                             k_neighbours=k, distance=dist)

                for timepoint in time_series:
                    knn_stream.update(timepoint)

                knn = knn_stream.transform()

                for arr in (knn.distances, knn.offsets):
                    assert arr.shape[0] == knn_stream.knn_insert_idx - knn_stream.lbound
                    assert arr.shape[1] == knn.k_neighbours
