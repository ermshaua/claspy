"""Utilities for loading tssb datasets."""

import os
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

__all__ = []

__author__ = [
    "ermshaua",
]

TSSB_URL = (
    "https://raw.githubusercontent.com/ermshaua/"
    "time-series-segmentation-benchmark/main/tssb/datasets/"
)

TSSB_DIRNAME = "data/tssb"
MODULE = os.path.dirname(__file__)


tssb_dataset_names = ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'Chinatown', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'Crop', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxTW', 'DodgerLoopDay', 'ECG200', 'ECGFiveDays', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FreezerRegularTrain', 'Ham', 'Herring', 'GunPoint', 'Haptics', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf', 'Plane', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'ShapesAll', 'ShapeletSim', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'WordSynonyms', 'Yoga']


def load_tssb_dataset(names=None, extract_path=None):
    """Load the Time Series Segmentation Benchmark (TSSB) dataset from GitHub.
    Downloads and extracts the dataset if not already downloaded. Data is assumed to be
    in the standard .txt format: each row is a univariate time series,
    annotated with name, window size and ground truth change points.
    For examples see https://github.com/ermshaua/time-series-segmentation-benchmark.
    Parameters
    ----------
    names : List[str]
        List of names of data sets. If an available dataset is given,
        this function will look in the extract_path first, and if it is not present,
        attempt to download the data from the TSSB GitHub repository,
        saving it to the extract_path.
    extract_path : str, optional (default=None)
        the path to look for the data. If no path is provided, the function
        looks in `claspy/tests/data/tssb/`.
    Returns
    -------
    tssb: pandas DataFrame
        The time series data for the problem with n_cases rows and either
        4 columns. Columns 1 to 3 are the name, window size and ground truth
        change points. Column 4 is the associated time series.
    """
    # Allow user to have non standard extract path
    if extract_path is not None:
        local_module = os.path.dirname(extract_path)
        local_dirname = extract_path
    else:
        local_module = MODULE
        local_dirname = TSSB_DIRNAME

    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))

    desc = []
    desc_file_name = "desc.txt"
    desc_file_path = os.path.join(local_module, local_dirname, desc_file_name)

    if (
        not os.path.exists(desc_file_path)
        or names is not None
        and any(name not in tssb_dataset_names for name in names)
    ):
        url = os.path.join(TSSB_URL, "desc.txt")
        urlretrieve(url, desc_file_path)

    with open(desc_file_path, "r") as file:
        for line in file.readlines():
            line = line.split(",")

            if names is None or line[0] in names:
                desc.append(line)

    tssb = []

    for row in desc:
        (ts_name, window_size), change_points = row[:2], row[2:]

        ts_file_path = os.path.join(local_module, local_dirname, ts_name + ".txt")

        if not os.path.exists(ts_file_path):
            url = os.path.join(TSSB_URL, ts_name + ".txt")
            urlretrieve(url, ts_file_path)

        ts = np.loadtxt(fname=ts_file_path, dtype=np.float64)
        window_size = np.int64(window_size)
        change_points = np.asarray(change_points, dtype=np.int64)

        tssb.append((ts_name, window_size, change_points, ts))

    return pd.DataFrame.from_records(
        tssb, columns=["dataset", "window_size", "cps", "time_series"]
    )