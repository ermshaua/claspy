import unittest
from itertools import product

import numpy as np

from claspy.clap import CLaP
from claspy.data_loader import load_tssb_dataset


class CLaPTest(unittest.TestCase):

    def test_tssb_benchmark(self):
        tssb = load_tssb_dataset()
        np.random.seed(2357)
        tssb = tssb.sample(10)

        for _, (dataset, window_size, cps, labels, time_series) in tssb.iterrows():
            clap = CLaP(window_size=window_size)

            y_true, y_pred = clap.fit_transform(time_series, labels)
            score = clap.score()

            assert y_true.shape[0] <= labels.shape[0] and y_pred.shape[0] <= labels.shape[0]
            assert isinstance(score, float) and 0 <= score <= 1.0

    def test_param_configs(self):
        tssb = load_tssb_dataset()
        np.random.seed(2357)
        tssb = tssb.sample(3)

        window_sizes = (10, 50, 100)
        classifiers = ("rocket", "weasel")
        splits = (3, 5)
        sample_sizes = (500, 1000)

        for _, (dataset, window_size, cps, labels, time_series) in tssb.iterrows():
            for window_size, classifier, n_splits, sample_size in product(window_sizes, classifiers, splits,
                                                                          sample_sizes):
                if time_series.shape[0] < 2 * 5 * window_size: continue

                clap = CLaP(window_size=window_size, classifier=classifier, n_splits=n_splits,
                            sample_size=sample_size)

                y_true, y_pred = clap.fit_transform(time_series, labels)
                score = clap.score()

                assert y_true.shape[0] <= labels.shape[0] and y_pred.shape[0] <= labels.shape[0]
                assert isinstance(score, float) and 0 <= score <= 1.0
