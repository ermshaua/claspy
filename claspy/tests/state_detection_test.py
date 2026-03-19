import os
import time
import unittest

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

from claspy.data_loader import load_tssb_dataset, load_has_dataset
from claspy.state_detection import AgglomerativeCLaPDetection
from claspy.utils import create_state_labels

ABS_PATH = os.path.dirname(os.path.abspath(__file__))


class StateDetectionTest(unittest.TestCase):

    def test_tssb_benchmark_accuracy(self):
        tssb = load_tssb_dataset()
        scores = []
        runtime = time.process_time()

        for idx, (dataset, window_size, cps, labels, time_series) in list(tssb.iterrows()):
            clap = AgglomerativeCLaPDetection()
            found_labels = clap.fit_predict(time_series)
            score = np.round(adjusted_mutual_info_score(labels, found_labels), 2)
            scores.append(score)

        runtime = np.round(time.process_time() - runtime, 3)
        score = np.mean(scores)

        assert score >= .77, f"AMI is only: {score}"

    def test_has_benchmark_accuracy(self):
        has = load_has_dataset()
        scores = []
        runtime = time.process_time()

        for idx, (dataset, window_size, cps, labels, time_series) in list(has.iterrows()):
            clap = AgglomerativeCLaPDetection()
            found_labels = clap.fit_predict(time_series)

            # use numeric labels for comparison
            label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
            labels = np.array([label_mapping[label] for label in labels])
            labels = create_state_labels(cps, labels, time_series.shape[0])

            score = np.round(adjusted_mutual_info_score(labels, found_labels), 2)
            scores.append(score)

        runtime = np.round(time.process_time() - runtime, 3)
        score = np.mean(scores)

        assert score >= .64, f"AMI is only: {score}"

    def test_readme(self):
        dataset, window_size, true_cps, labels, time_series = load_tssb_dataset(names=("Crop",)).iloc[0, :]
        clap = AgglomerativeCLaPDetection()
        state_seq = clap.fit_predict(time_series)

        clap.plot(
            gt_states=labels,
            heading="Detection of crops in satellite image time series",
            ts_name="Sensor",
            file_path=f"{ABS_PATH}/../../state_detection_example.png"
        )

        states, transitions = clap.predict(sparse=True)
        assert states == {1, 2, 3}
        assert transitions == {(1, 2), (2, 3), (3, 1)}

        clap.plot(
            heading="Process of captured crops",
            file_path=f"{ABS_PATH}/../../process_discovery_example.png",
            sparse=True
        )
