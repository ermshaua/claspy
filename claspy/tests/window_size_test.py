import unittest

import numpy as np

from claspy.data_loader import load_tssb_dataset
from claspy.window_size import suss, dominant_fourier_frequency, highest_autocorrelation


class WindowSizeTest(unittest.TestCase):

    def test_tssb_benchmark(self):
        tssb = load_tssb_dataset()
        lbound = 10

        for wss_algo in (suss, dominant_fourier_frequency, highest_autocorrelation):
            for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
                assert lbound <= wss_algo(time_series, lbound=lbound)


    def test_edge_cases(self):
        lbound = 10

        for wss_algo in (suss, dominant_fourier_frequency, highest_autocorrelation):
            time_series = np.zeros(1, dtype=np.float64)
            assert wss_algo(time_series, lbound=lbound) == time_series.shape[0]

            time_series = np.zeros(1000, dtype=np.float64)
            assert wss_algo(time_series, lbound=lbound) == lbound





