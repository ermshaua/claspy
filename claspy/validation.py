from scipy.stats import ranksums

from claspy.nearest_neighbour import cross_val_labels


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
    _, y_pred = cross_val_labels(clasp.knn.offsets, change_point-clasp.lbound, clasp.window_size)
    _, p = ranksums(y_pred[:change_point], y_pred[change_point:])
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
