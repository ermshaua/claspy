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
        scores = ("f1", "roc_auc")
        n_jobs = (1, -1)

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            for window_size, score, n_job in product(window_sizes, scores, n_jobs):
                if time_series.shape[0] < 2 * 5 * window_size: continue
                clasp = ClaSP(window_size=window_size, score=score, n_jobs=n_job)
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
        scores = ("f1", "roc_auc")
        early_stopping = (True, False)
        n_jobs = (1, -1)

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            for window_size, score, stop, n_job in product(window_sizes, scores, early_stopping, n_jobs):
                if time_series.shape[0] < 2 * 5 * window_size: continue
                clasp = ClaSPEnsemble(
                    window_size=window_size,
                    score=score,
                    early_stopping=stop,
                    n_jobs=n_job
                )
                clasp.fit(time_series)
                assert clasp.profile.shape[0] == time_series.shape[0] - clasp.window_size + 1
