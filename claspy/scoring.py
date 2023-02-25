import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def f1_score(y_true, y_pred):
    """
    Compute the F1 score, which is the harmonic mean of precision and recall.
    The F1 score is a measure of a model"s accuracy that takes both precision
    and recall into account. It is commonly used in binary classification problems
    where there are two classes, positive and negative.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    Returns
    -------
    score : float
        F1 score.

    Notes
    -----
    The F1 score is computed as follows:
        F1 = 2 * (precision * recall) / (precision + recall)
    where precision is the number of true positives divided by the total number
    of predicted positives, and recall is the number of true positives divided
    by the total number of actual positives. The F1 score is a value between 0
    and 1, where 1 is the best possible score.

    References
    ----------
    .. [1] "F1 score", Wikipedia, https://en.wikipedia.org/wiki/F1_score
    """
    f1_scores = np.zeros(shape=2, dtype=np.float64)

    for label in (0, 1):
        tp = np.sum(np.logical_and(y_true == label, y_pred == label))
        fp = np.sum(np.logical_and(y_true != label, y_pred == label))
        fn = np.sum(np.logical_and(y_true == label, y_pred != label))

        if (tp + fp) == 0 or (tp + fn) == 0:
            return -np.inf

        pr = tp / (tp + fp)
        re = tp / (tp + fn)

        if (pr + re) == 0:
            return -np.inf

        f1 = 2 * (pr * re) / (pr + re)
        f1_scores[label] = f1

    return np.mean(f1_scores)


@njit(fastmath=True, cache=True)
def roc_auc_score(y_score, y_true):
    """
    Compute the Receiver Operating Characteristic (ROC) area under the curve (AUC).
    The ROC curve is a graphical plot that illustrates the performance of a binary
    classifier system as its discrimination threshold is varied. The area under
    the ROC curve (AUC) is a measure of the classifier"s ability to distinguish between
    positive and negative classes. The AUC ranges from 0.0 to 1.0, with 1.0
    representing perfect classification.

    Parameters
    ----------
    y_score : array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive class,
        confidence values, or binary decisions.
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    Returns
    -------
    auc : float
        Area under the ROC curve.

    Notes
    -----
    Since the AUC represents the degree or measure of separability, it is invariant to
    the class distribution. It can be interpreted as the probability that a randomly selected
    positive example is ranked higher than a randomly selected negative example. Additionally,
    the AUC provides an aggregate measure of performance across all possible classification
    thresholds.

    References
    ----------
    .. [1] "Receiver operating characteristic", Wikipedia, https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """
    # make y_true a boolean vector
    y_true = (y_true == 1)

    # sort scores and corresponding truth values (y_true is sorted by design)
    desc_score_indices = np.arange(y_score.shape[0])[::-1]

    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.concatenate((
        distinct_value_indices,
        np.array([y_true.size - 1])
    ))

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.concatenate((np.array([0]), tps))
    fps = np.concatenate((np.array([0]), fps))

    if fps[-1] <= 0 or tps[-1] <= 0:
        return -np.inf

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    if fpr.shape[0] < 2:
        return np.nan

    direction = 1
    dx = np.diff(fpr)

    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            return -np.inf

    area = direction * np.trapz(tpr, fpr)
    return area


_SCORE_MAPPING = {
    "f1": f1_score,
    "roc_auc": roc_auc_score,
}


def map_scores(score_name):
    """
    Map score names to score functions.

    Parameters
    ----------
    score_name : str
        Name of the score to be mapped. Valid score names are: "f1", "roc_auc".

    Returns
    -------
    score_func : callable
        The score function corresponding to the input score name.

    Raises
    ------
    ValueError
        If the input score name is not a valid score name. Valid score names are: "f1", "roc_auc".

    Notes
    -----
    This function maps the input score name to its corresponding score function from the score
    function dictionary `_SCORE_MAPPING`, which maps score names to score functions. If the input score
    name is not a valid score name, a `ValueError` is raised.

    See Also
    --------
    f1_score : Compute the F1 score, which is the harmonic mean of precision and recall.
    roc_auc_score : Compute the Receiver Operating Characteristic (ROC) area under the curve (AUC).

    Examples
    --------
    >>> score_func = map_scores('f1')
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> score = score_func(y_true, y_pred)

    >>> score_func = map_scores('roc_auc')
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_score = [0, 1, 0, 0, 1]
    >>> score = score_func(y_score, y_true)
    """
    if score_name not in _SCORE_MAPPING:
        raise ValueError(
            f"{score_name} is not a valid score. Implementations include: {', '.join(_SCORE_MAPPING.keys())}")

    return _SCORE_MAPPING[score_name]
