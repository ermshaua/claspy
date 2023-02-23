import unittest
from itertools import product

import numpy as np

from claspy.segmentation import BinaryClaSPSegmentation
from claspy.tests.tssb_data_loader import load_tssb_dataset


class SegmentationTest(unittest.TestCase):

    def test_tssb_benchmark(self):
        tssb = load_tssb_dataset()

        for idx, (dataset, window_size, cps, time_series) in list(tssb.iterrows()):
            clasp = BinaryClaSPSegmentation(n_segments=len(cps)+1, window_size=window_size)
            clasp.fit(time_series)
