import numpy as np
from scipy.stats import distributions

from claspy.nearest_neighbour import cross_val_labels


def _rank_binary_data(data):
    """
    Assigns rank-based scores to binary data for statistical evaluation.

    This function assigns ranks to binary values in an array, distinguishing between `0`s and `1`s.
    The ranks are computed such that all `0` values receive the average rank of their group,
    and all `1` values receive the average rank of theirs.

    Parameters
    ----------
    data : np.ndarray
        A 1D array of shape (n_timepoints,) containing binary values (0 or 1) representing
        class labels or outcomes.

    Returns
    -------
    ranks : np.ndarray
        A 1D array of shape (n_timepoints,) where each entry is the average rank assigned to
        the corresponding class (0 or 1). The ranks start from 1, following the convention of
        statistical ranking.
    """
    zeros = data == 0
    ones = data == 1

    zero_ranks = np.arange(np.sum(zeros))
    one_ranks = np.arange(zero_ranks.shape[0], data.shape[0])

    zero_mean = np.mean(zero_ranks) + 1 if zero_ranks.shape[0] > 0 else 0
    one_mean = np.mean(one_ranks) + 1 if one_ranks.shape[0] > 0 else 0

    ranks = np.full(data.shape[0], fill_value=zero_mean, dtype=np.float64)
    ranks[ones] = one_mean

    return ranks


def _rank_sums_test(x, y):
    """
    Performs a two-sided Wilcoxon rank-sum test (Mann–Whitney U test) on two samples.

    This function tests the null hypothesis that two independent samples `x` and `y` come
    from the same distribution. It computes the test statistic based on the sum of ranks
    and returns the z-score and corresponding two-tailed p-value under the normal approximation.

    Parameters
    ----------
    x : np.ndarray
        A 1D array representing the first sample of observations.
    y : np.ndarray
        A 1D array representing the second sample of observations.

    Returns
    -------
    z : float
        The standardized test statistic (z-score) computed from the rank sums.
    p : float
        The two-sided p-value indicating the significance of the difference between the samples.
    """
    n1, n2 = len(x), len(y)

    alldata = np.concatenate((x, y))
    ranked = _rank_binary_data(alldata)

    x = ranked[:n1]
    s = np.sum(x, axis=0)

    expected = n1 * (n1 + n2 + 1) / 2.0
    z = (s - expected) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    p = 2 * distributions.norm.sf(np.abs(z))

    return z, p


def significance_test(clasp, change_point, threshold=1e-15):
    """
    Perform a significance test on a candidate change point using the provided ClaSP.

    Parameters
    ----------
    clasp : ClaSP
        A fitted ClaSP object to use for the significance test.
    change_point : int
        The candidate change point to test.
    threshold : float, optional (default=1e-15)
        The p-value threshold for significance. The default value is 1e-15.

    Returns
    -------
    bool
        True if the change point is significant, False otherwise.

    Notes
    -----
    This method uses a two-sample rank-sum test with a p-value threshold to determine if a candidate change point
    is statistically significant. The test is performed using the classification labels for the candidate change point.
    If the resulting p-value is less than or equal to the specified threshold, the change point is considered significant
    and the method returns True. Otherwise, the change point is considered not significant and the method returns False.

    """
    _, y_pred = cross_val_labels(clasp.knn.offsets, change_point - clasp.lbound, clasp.window_size)
    _, p = _rank_sums_test(y_pred[:change_point], y_pred[change_point:])
    return p <= threshold


def score_threshold(clasp, change_point, threshold=0.75):
    """
    Returns whether the ClaSP score at the given change point exceeds the specified threshold.

    Parameters
    ----------
    clasp : ClaSP object
        An instance of the ClaSP class that has already been fit to a time series.
    change_point : int
        The index of the change point to be tested.
    threshold : float, optional
        The threshold value to test against the ClaSP score. The default is 0.75.

    Returns
    -------
    bool
        True if the ClaSP score at the given change point is greater than or equal to the specified
        threshold, False otherwise.
    """
    return clasp.profile[change_point] >= threshold


_VALIDATION_MAPPING = {
    "significance_test": significance_test,
    "score_threshold": score_threshold
}


def map_validation_tests(validation_method):
    """
    Maps a validation method name to its corresponding function.

    Parameters
    ----------
    validation_method : str
        The name of the validation method to map.

    Returns
    -------
    function
        The validation function that corresponds to the input method name.

    Raises
    ------
    ValueError
        If the input validation method name is not in the list of implemented methods.
    """
    if validation_method not in _VALIDATION_MAPPING:
        raise ValueError(
            f"{validation_method} is not a valid validation method. Implementations include: {', '.join(_VALIDATION_MAPPING.keys())}")

    return _VALIDATION_MAPPING[validation_method]
