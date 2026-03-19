"""Utilities for loading tssb datasets."""

import os
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from claspy.utils import create_state_labels

__all__ = []

__author__ = [
    "ermshaua",
]

TSSB_URL = (
    "https://raw.githubusercontent.com/ermshaua/"
    "time-series-segmentation-benchmark/main/tssb/datasets/"
)

HAS_URL = (
    "https://raw.githubusercontent.com/patrickzib"
    "/human_activity_segmentation_challenge/main/datasets/has2023_master.csv"
    ".zip"
)

TSSB_DIRNAME = "data/tssb"
HAS_DIRNAME = "data/has"

MODULE = os.path.dirname(__file__)

tssb_dataset_names = ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'Chinatown',
                      'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY',
                      'CricketZ', 'Crop', 'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxTW',
                      'DodgerLoopDay', 'ECG200', 'ECGFiveDays', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'FaceAll',
                      'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FreezerRegularTrain', 'Ham', 'Herring', 'GunPoint',
                      'Haptics', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances',
                      'Lightning2', 'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
                      'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxTW', 'MoteStrain',
                      'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'OliveOil', 'OSULeaf', 'Plane',
                      'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
                      'ShapesAll', 'ShapeletSim', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'Strawberry',
                      'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace',
                      'TwoLeadECG', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY',
                      'UWaveGestureLibraryZ', 'WordSynonyms', 'Yoga']


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
        looks in `claspy/data/tssb/`.

    Returns
    -------
    tssb: pandas DataFrame
        The time series data for the problem with n_cases rows and either
        5 columns. Columns 1 to 4 are the name, window size, ground truth
        change points and labels. Column 5 is the associated time series.

    Raises
    ------
    IOError
        If the dataset cannot be downloaded or accessed.
    """
    # allow user to have non-standard extract path
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
        url = os.path.join(TSSB_URL, desc_file_name)
        urlretrieve(url, desc_file_path)

    with open(desc_file_path, "r") as file:
        for line in file.readlines():
            line = line.split(",")

            if names is None or line[0] in names:
                desc.append(line)

    prop = []
    prop_file_name = "properties.txt"
    prop_file_path = os.path.join(local_module, local_dirname, prop_file_name)

    if (
            not os.path.exists(prop_file_path)
            or names is not None
            and any(name not in tssb_dataset_names for name in names)
    ):
        url = os.path.join(TSSB_URL, prop_file_name)
        urlretrieve(url, prop_file_path)

    with open(prop_file_path, 'r') as file:
        for line in file.readlines():
            line = line.split(",")

            ds_name, interpretable, label_cut, resample_rate, labels = line[0], bool(line[1]), int(line[2]), int(
                line[3]), line[4:]
            labels = np.array([int(l.replace("\n", "")) // (label_cut + 1) for l in labels])

            if names is None or ds_name in names:
                prop.append((ds_name, label_cut, resample_rate, labels))

    tssb = []

    for desc_row, prop_row in zip(desc, prop):
        (ts_name, window_size), change_points = desc_row[:2], desc_row[2:]
        labels = prop_row[3]

        ts_file_path = os.path.join(local_module, local_dirname, ts_name + ".txt")

        if not os.path.exists(ts_file_path):
            url = os.path.join(TSSB_URL, ts_name + ".txt")
            urlretrieve(url, ts_file_path)

        ts = np.loadtxt(fname=ts_file_path, dtype=np.float64)
        window_size = np.int64(window_size)
        change_points = np.asarray(change_points, dtype=np.int64)
        labels = create_state_labels(change_points, labels, ts.shape[0]) + 1

        tssb.append((ts_name, window_size, change_points, labels, ts))

    return pd.DataFrame.from_records(tssb, columns=["dataset", "window_size", "cps", "labels", "time_series"])


def load_has_dataset(extract_path=None):
    """
    Load the Human Activity Segmentation (HAS) dataset.

    Downloads the dataset if not already available locally and constructs
    multivariate time series along with corresponding change points and labels.

    Parameters
    ----------
    extract_path : str, optional (default=None)
        Path to store or load the dataset. If not provided, defaults to
        `claspy/data/has/`.

    Returns
    -------
    has : pandas.DataFrame
        A DataFrame where each row corresponds to a time series instance.
        Columns include dataset name, window size, change points, labels,
        and the multivariate time series data.

    Raises
    ------
    IOError
        If the dataset cannot be downloaded or accessed.
    """
    # allow user to have non-standard extract path
    if extract_path is not None:
        local_module = os.path.dirname(extract_path)
        local_dirname = extract_path
    else:
        local_module = MODULE
        local_dirname = HAS_DIRNAME

    if not os.path.exists(os.path.join(local_module, local_dirname)):
        os.makedirs(os.path.join(local_module, local_dirname))

    has_file_path = os.path.join(local_module, local_dirname, "has.csv.zip")

    if not os.path.exists(has_file_path):
        urlretrieve(HAS_URL, has_file_path)

    # converters to correctly load benchmark
    np_cols = [
        "change_points",
        "activities",
        "x-acc",
        "y-acc",
        "z-acc",
        "x-gyro",
        "y-gyro",
        "z-gyro",
        "x-mag",
        "y-mag",
        "z-mag",
        "lat",
        "lon",
        "speed",
    ]

    converters = {
        col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val))
        for col in np_cols
    }

    df = pd.read_csv(has_file_path, converters=converters)
    has = []

    for _, row in df.iterrows():
        dataset = (
            f"{row.group}_subject{row.subject}_routine{row.routine} "
            f"(id{row.ts_challenge_id})"
        )

        if row.group == "indoor":
            ts = np.hstack(
                (
                    row["x-acc"].reshape(-1, 1),
                    row["y-acc"].reshape(-1, 1),
                    row["z-acc"].reshape(-1, 1),
                    row["x-gyro"].reshape(-1, 1),
                    row["y-gyro"].reshape(-1, 1),
                    row["z-gyro"].reshape(-1, 1),
                    row["x-mag"].reshape(-1, 1),
                    row["y-mag"].reshape(-1, 1),
                    row["z-mag"].reshape(-1, 1),
                )
            )
        elif row.group == "outdoor":
            ts = np.hstack(
                (
                    row["x-acc"].reshape(-1, 1),
                    row["y-acc"].reshape(-1, 1),
                    row["z-acc"].reshape(-1, 1),
                    row["x-mag"].reshape(-1, 1),
                    row["y-mag"].reshape(-1, 1),
                    row["z-mag"].reshape(-1, 1),
                    # row["lat"].reshape(-1, 1),
                    # row["lon"].reshape(-1, 1),
                    # row["speed"].reshape(-1, 1),
                )
            )

        has.append((dataset, 50, row.change_points, row.activities, ts))

    return pd.DataFrame.from_records(
        has, columns=["dataset", "window_size", "cps", "labels", "time_series"]
    )
