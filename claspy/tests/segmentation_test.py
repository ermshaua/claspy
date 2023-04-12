import os
import time
import unittest
from itertools import product

import numpy as np

from claspy.data_loader import load_tssb_dataset
from claspy.segmentation import BinaryClaSPSegmentation
from claspy.tests.evaluation import covering

ABS_PATH = os.path.dirname(os.path.abspath(__file__))

# tssb data sets, with recurring segments
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

    def test_tssb_benchmark_accuracy(self):
        tssb = load_tssb_dataset()
        scores = []
        runtime = time.process_time()

        for idx, (dataset, window_size, cps, time_series) in list(tssb.iterrows()):
            clasp = BinaryClaSPSegmentation()
            found_cps = clasp.fit_predict(time_series)
            score = np.round(covering({0: cps}, found_cps, time_series.shape[0]), 2)
            scores.append(score)

        runtime = np.round(time.process_time() - runtime, 3)
        score = np.mean(scores)

        assert score >= .85, f"Covering is only: {np.mean(scores)}"

    def test_param_configs(self):
        tssb = load_tssb_dataset()
        np.random.seed(2357)
        tssb = tssb.sample(1)

        n_segments = (1, "learn")
        window_sizes = (10, "suss", "fft", "acf")
        validations = (None, "significance_test", "score_threshold")
        thresholds = {"significance_test": 1e-15, "score_threshold": .75}
        n_jobs = (1, -1)

        for idx, (dataset, window_size, cps, time_series) in list(tssb.iterrows()):
            for n_seg, window_size, val, n_job in product(n_segments, window_sizes, validations, n_jobs):
                BinaryClaSPSegmentation(
                    n_segments=n_seg,
                    window_size=window_size,
                    validation=val,
                    threshold=thresholds[val] if val is not None else None,
                    n_jobs=n_job
                ).fit(time_series)

    def test_readme(self):
        dataset, window_size, true_cps, time_series = load_tssb_dataset(names=("CricketX",)).iloc[0, :]

        clasp = BinaryClaSPSegmentation()
        change_points = clasp.fit_predict(time_series)
        assert np.array_equal(change_points, np.array([712, 1281, 1933, 2581]))
        clasp.plot(
            gt_cps=true_cps,
            heading="Segmentation of different umpire cricket signals",
            ts_name="ACC",
            file_path=f"{ABS_PATH}/../../segmentation_example.png"
        )

    def test_very_small_ts(self):
        time_series = np.zeros(0)
        clasp = BinaryClaSPSegmentation()
        clasp.fit_predict(time_series)

        time_series = np.zeros(1)
        clasp = BinaryClaSPSegmentation()
        clasp.fit_predict(time_series)
