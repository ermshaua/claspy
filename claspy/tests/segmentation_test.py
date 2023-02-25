import unittest

import numpy as np

from claspy.segmentation import BinaryClaSPSegmentation
from claspy.tests.evaluation import covering
from claspy.tests.tssb_data_loader import load_tssb_dataset

# data sets, with recurring segments
REC_SEG_DS = [
    "Crop",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "FreezerRegularTrain",
    "Ham",
    "MelbournePedestrian",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "ProximalPhalanxOutlineCorrect",
    "Strawberry",
]


class SegmentationTest(unittest.TestCase):

    def test_tssb_benchmark(self):
        tssb = load_tssb_dataset()

        scores = []

        for idx, (dataset, window_size, cps, time_series) in list(tssb.iterrows()):
            clasp = BinaryClaSPSegmentation()
            found_cps = clasp.fit_predict(time_series)
            score = np.round(covering({0: cps}, found_cps, time_series.shape[0]), 2)
            print(f"{dataset}: {score}")
            scores.append(score)

        print(np.mean(scores))
        assert np.mean(scores) > .8
