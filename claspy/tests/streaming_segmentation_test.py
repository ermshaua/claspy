import os
import unittest
from itertools import product

import numpy as np

from claspy.data_loader import load_tssb_dataset
from claspy.streaming.segmentation import StreamingClaSPSegmentation
from claspy.tests.evaluation import covering

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


class StreamingSegmentationTest(unittest.TestCase):

    def test_tssb_benchmark_accuracy(self):
        tssb = load_tssb_dataset()
        scores = []

        for idx, (dataset, window_size, cps, time_series) in list(tssb.iterrows()):
            clasp = StreamingClaSPSegmentation(n_timepoints=min(10_000, time_series.shape[0]), log_cps=True)

            for value in time_series:
                clasp.update(value)

            found_cps = clasp.change_points
            score = np.round(covering({0: cps}, found_cps, time_series.shape[0]), 2)
            scores.append(score)

        score = np.mean(scores)
        assert score >= .83, f"Covering is only: {score}"

    def test_param_configs(self):
        tssb = load_tssb_dataset()
        np.random.seed(2357)
        tssb = tssb.sample(1)

        n_timepoints = (100, 1000, 10_000)
        window_sizes = (10, "suss", "fft", "acf")
        distances = ("znormed_euclidean_distance", "euclidean_distance", "cinvariant_euclidean_distance")
        validations = ("significance_test", "score_threshold")

        for idx, (dataset, window_size, cps, time_series) in list(tssb.iterrows()):
            for n_tp, window_size, distance, val in product(n_timepoints, window_sizes, distances, validations):
                clasp = StreamingClaSPSegmentation(
                    n_timepoints=min(n_tp, time_series.shape[0]),
                    window_size=window_size,
                    distance=distance,
                    validation=val,
                    log_cps=True
                )

                for value in time_series:
                    clasp.update(value)

    def test_readme(self):
        dataset, window_size, true_cps, time_series = load_tssb_dataset(names=("ECG200",)).iloc[0, :]
        clasp = StreamingClaSPSegmentation(n_timepoints=1000)

        for idx, value in enumerate(time_series):
            clasp.update(value)

            if idx >= clasp.n_warmup and clasp.predict() != 0:
                break

        clasp.plot(
            heading="Detection of myocardial infarction in ECG stream",
            stream_name="ECG",
            file_path=f"{ABS_PATH}/../../streaming_segmentation_example.png"
        )

        assert clasp.predict() == 663
