import unittest
from itertools import product

import numpy as np

from claspy.clasp import ClaSP, ClaSPEnsemble
from claspy.data_loader import load_tssb_dataset


class ClaSPTest(unittest.TestCase):

    def test_tssb_benchmark(self):
        tssb = load_tssb_dataset()
        np.random.seed(2357)
        tssb = tssb.sample(10)

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            clasp = ClaSP(window_size=window_size)
            clasp.fit(time_series)
            assert clasp.profile.shape[0] == time_series.shape[0] - clasp.window_size + 1

    def test_param_configs(self):
        tssb = load_tssb_dataset()
        np.random.seed(2357)
        tssb = tssb.sample(3)

        window_sizes = (10, 50, 100)
        k_neighbours = (1, 3, 5)
        scores = ("f1", "roc_auc")

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            for window_size, k, score in product(window_sizes, k_neighbours, scores):
                if time_series.shape[0] < 2 * 5 * window_size: continue
                clasp = ClaSP(window_size=window_size, k_neighbours=k, score=score, excl_radius=max(5, k+1))
                clasp.fit(time_series)
                assert clasp.profile.shape[0] == time_series.shape[0] - clasp.window_size + 1


class ClaSPEnsembleTest(unittest.TestCase):

    def test_tssb_benchmark(self):
        tssb = load_tssb_dataset()
        np.random.seed(2357)
        tssb = tssb.sample(10)

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            clasp = ClaSPEnsemble(window_size=window_size)
            clasp.fit(time_series)
            assert clasp.profile.shape[0] == time_series.shape[0] - clasp.window_size + 1

    def test_param_configs(self):
        tssb = load_tssb_dataset()
        np.random.seed(2357)
        tssb = tssb.sample(3)

        window_sizes = (10, 50, 100)
        k_neighbours = (1, 3, 5)
        scores = ("f1", "roc_auc")

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            for window_size, k, score in product(window_sizes, k_neighbours, scores):
                if time_series.shape[0] < 2 * 5 * window_size: continue
                clasp = ClaSPEnsemble(window_size=window_size, k_neighbours=k, score=score, excl_radius=max(5, k+1))
                clasp.fit(time_series)
                assert clasp.profile.shape[0] == time_series.shape[0] - clasp.window_size + 1
