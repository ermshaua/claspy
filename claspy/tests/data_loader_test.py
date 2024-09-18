import unittest

import numpy as np

from claspy.data_loader import load_tssb_dataset, load_has_dataset
from claspy.segmentation import BinaryClaSPSegmentation


class DataLoaderTest(unittest.TestCase):

    def test_load_tssb_dataset(self):
        tssb = load_tssb_dataset()
        assert tssb.shape[0] == 75

        assert all(isinstance(w, int) for w in tssb.window_size)
        assert all(isinstance(cps, np.ndarray) for cps in tssb.cps)
        assert all(cps.ndim == 1 for cps in tssb.cps)
        assert all(isinstance(ts, np.ndarray) for ts in tssb.time_series)
        assert all(ts.ndim == 1 for ts in tssb.time_series)

        cps, ts = tssb.iloc[0,2:]
        clasp = BinaryClaSPSegmentation(n_segments=cps.shape[0]+1, validation=None)
        assert cps.shape[0] == clasp.fit_predict(ts).shape[0]

    def test_load_has_dataset(self):
        tssb = load_has_dataset()
        assert tssb.shape[0] == 250

        assert all(isinstance(w, int) for w in tssb.window_size)
        assert all(isinstance(cps, np.ndarray) for cps in tssb.cps)
        assert all(cps.ndim == 1 for cps in tssb.cps)
        assert all(isinstance(ts, np.ndarray) for ts in tssb.time_series)
        assert all(ts.ndim > 1 for ts in tssb.time_series)

        cps, _, ts = tssb.iloc[0,2:]
        clasp = BinaryClaSPSegmentation(n_segments=cps.shape[0]+1, validation=None)
        assert cps.shape[0] == clasp.fit_predict(ts[:,0]).shape[0]
