import unittest
from itertools import product

import numpy as np

from claspy.data_loader import load_tssb_dataset
from claspy.streaming.clasp import ClaSS


class ClaSSTest(unittest.TestCase):

    def test_tssb_benchmark(self):
        tssb = load_tssb_dataset()
        np.random.seed(2357)
        tssb = tssb.sample(10)

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            clasp = ClaSS(window_size=window_size)
            clasp.fit(time_series)
            assert clasp.profile.shape[0] == time_series.shape[0] - clasp.window_size + 1

    def test_param_configs(self):
        tssb = load_tssb_dataset()
        np.random.seed(2357)
        tssb = tssb.sample(3)

        window_sizes = (10, 50, 100)
        scores = ("f1", "accuracy")

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            for window_size, score in product(window_sizes, scores):
                if time_series.shape[0] < 2 * 5 * window_size: continue
                clasp = ClaSS(window_size=window_size, score=score)
                clasp.fit(time_series)
                assert clasp.profile.shape[0] == time_series.shape[0] - clasp.window_size + 1
