import unittest, time

import numpy as np

from claspy.data_loader import load_tssb_dataset
from claspy.segmentation import BinaryClaSPSegmentation
from claspy.tests.evaluation import covering
from itertools import product

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
        n_estimators = (1, 10)
        window_sizes = (10, "suss", "fft", "acf")
        k_neighbours = (1, 3, 5)
        scores = ("f1", "roc_auc")
        early_stopping = (True, False)
        validations = (None, "significance_test", "score_threshold")
        thresholds = {"significance_test" : 1e-15, "score_threshold" : .75}

        for idx, (dataset, window_size, cps, time_series) in list(tssb.iterrows()):
            for n_seg, n_est, window_size, k, score, stop, val in product(n_segments, n_estimators, window_sizes, k_neighbours, scores, early_stopping, validations):
                BinaryClaSPSegmentation(
                    n_segments=n_seg,
                    n_estimators=n_est,
                    window_size=window_size,
                    k_neighbours=k,
                    score=score,
                    early_stopping=stop,
                    validation=val,
                    threshold=thresholds[val] if val is not None else None,
                    excl_radius=max(5, k + 1)
                ).fit(time_series)

    def test_readme(self):
        dataset, window_size, true_cps, time_series = load_tssb_dataset(names=("CricketX",)).iloc[0,:]

        clasp = BinaryClaSPSegmentation()
        change_points = clasp.fit_predict(time_series)
        assert np.array_equal(change_points, np.array([712, 1281, 1933, 2581]))
        clasp.plot(gt_cps=true_cps, heading="Segmentation of different umpire cricket signals", ts_name="ACC", file_path="../../segmentation_example.png")

    def test_very_small_ts(self):
        time_series = np.zeros(0)
        clasp = BinaryClaSPSegmentation()
        clasp.fit_predict(time_series)

        time_series = np.zeros(1)
        clasp = BinaryClaSPSegmentation()
        clasp.fit_predict(time_series)

