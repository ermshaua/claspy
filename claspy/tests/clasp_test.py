import unittest

from claspy.tests.tssb_data_loader import load_tssb_dataset
from claspy.clasp import ClaSP


class ClaSPTest(unittest.TestCase):

    def test_tssb_benchmark(self):
        tssb = load_tssb_dataset()

        for _, (dataset, window_size, cps, time_series) in tssb.iterrows():
            clasp = ClaSP(window_size=window_size)
            clasp.fit(time_series)

            assert clasp.profile.shape[0] == time_series.shape[0] - clasp.window_size + 1
