import numpy as np
from numba import njit

from claspy.clasp import ClaSP
from claspy.nearest_neighbour import KSubsequenceNeighbours
from claspy.utils import check_input_time_series


@njit(fastmath=True, cache=True)
def _rnn(knn_offsets):
    """
    Computes the reverse nearest neighbors (RNN) from the k-nearest neighbor (k-NN) offsets.

    The function takes an array of k-nearest neighbor indices and constructs an RNN structure, 
    which allows for efficiently retrieving the indices of all time points that have a given 
    point as one of their nearest neighbors.

    Parameters
    ----------
    knn_offsets : np.ndarray
        A 2D array of shape (n_timepoints, k_neighbors) containing the indices of the k-nearest 
        neighbors for each time point in the time series.

    Returns
    -------
    offsets : np.ndarray
        A 1D array of shape (n_timepoints,) where `offsets[i]` provides the starting index in 
        `values` for retrieving all time points that have `i` as one of their nearest neighbors.
    values : np.ndarray
        A 1D array of shape (n_timepoints * k_neighbors,) containing the indices of time points 
        that have each respective index in `offsets` as their nearest neighbor.
    """
    offsets, values = np.zeros(knn_offsets.shape[0], np.int64), np.zeros(knn_offsets.shape[0] * knn_offsets.shape[1],
                                                                         np.int64)

    counts = np.zeros(offsets.shape[0], np.int64)
    counters = np.zeros(offsets.shape[0], np.int64)

    # count knn occurrences
    for idx in range(knn_offsets.shape[0]):
        for kdx in range(knn_offsets.shape[1]):
            pos = knn_offsets[idx, kdx]
            counts[pos] += 1

    # store rnn pointers
    for idx in range(1, offsets.shape[0]):
        offsets[idx] = offsets[idx - 1] + counts[idx - 1]

    # save rnn values at correct offsets
    for idx, neigh in enumerate(knn_offsets):
        for nn in neigh:
            pos = offsets[nn] + counters[nn]
            values[pos] = idx
            counters[nn] += 1

    return offsets, values


@njit(fastmath=True, cache=True)
def _init_labels(knn_offsets, split_idx):
    """
    Initializes classification labels for evaluating change points in time series.

    This function assigns true labels (`y_true`) based on a predefined split point,
    determines predicted labels (`y_pred`) using k-nearest neighbor (k-NN) voting,
    and computes class counts (zeros and ones) for each time point.

    Parameters
    ----------
    knn_offsets : np.ndarray
        A 2D array of shape (n_timepoints, k_neighbors) containing the indices of the
        k-nearest neighbors for each time point in the time series.
    split_idx : int
        The split point that separates the two segments of the time series. All
        indices below `bound` are assigned label 0, and all indices from `bound`
        onwards are assigned label 1.

    Returns
    -------
    counts : tuple of np.ndarray
        A tuple containing two arrays:
        - `zeros` : A 1D array of shape (n_timepoints,) where each entry represents
          the number of nearest neighbors assigned label 0.
        - `ones` : A 1D array of shape (n_timepoints,) where each entry represents
          the number of nearest neighbors assigned label 1.
    y_true : np.ndarray
        A 1D array of shape (n_timepoints,) containing the true labels for each time
        point, where values are 0 for the first segment (before `split_idx`) and 1 for
        the second segment (after `split_idx`).
    y_pred : np.ndarray
        A 1D array of shape (n_timepoints,) containing the predicted labels based on
        k-NN majority voting. A time point is assigned label 1 if the majority of its
        k-nearest neighbors belong to the second segment; otherwise, it is assigned 0.
    """
    n_timepoints, k_neighbours = knn_offsets.shape

    y_true = np.concatenate((np.zeros(split_idx, dtype=np.int64), np.ones(n_timepoints - split_idx, dtype=np.int64)))
    knn_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

    for i_neighbor in range(k_neighbours):
        neighbours = knn_offsets[:, i_neighbor]
        knn_labels[i_neighbor] = y_true[neighbours]

    ones = np.sum(knn_labels, axis=0)
    zeros = k_neighbours - ones
    y_pred = np.asarray(ones > zeros, dtype=np.int64)

    return (zeros, ones), y_true, y_pred


@njit(fastmath=True, cache=True)
def _init_binary_conf_matrix(y_true, y_pred):
    """
    Initializes a confusion matrix for binary classification.

    This function computes the confusion matrix entries for a binary classification
    problem, specifically storing the values for label `0`. Since the confusion
    matrix is symmetrical for binary labels, the same values can be inferred for label `1`.

    Parameters
    ----------
    y_true : np.ndarray
        A 1D array of shape (n_timepoints,) containing the true class labels (0 or 1)
        for each time point.
    y_pred : np.ndarray
        A 1D array of shape (n_timepoints,) containing the predicted class labels (0 or 1)
        for each time point.

    Returns
    -------
    conf_matrix : np.ndarray
        A 1D array of shape (4,) containing the four confusion matrix components for label `0`:
        - `tp` (True Positives): Number of correctly predicted class `0` instances.
        - `fp` (False Positives): Number of times class `1` was incorrectly predicted as class `0`.
        - `fn` (False Negatives): Number of times class `0` was incorrectly predicted as class `1`.
        - `tn` (True Negatives): Number of correctly predicted class `1` instances.
    """
    tp = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 1) & (y_pred == 0))
    fn = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([tp, fp, fn, tn], dtype=np.int64)


@njit(fastmath=True, cache=True)
def _update_binary_conf_matrix(conf_matrix, old_true, old_pred, new_true, new_pred):
    """
    Updates the binary confusion matrix based on a change in prediction or ground truth.

    This function efficiently updates an existing binary confusion matrix when a single
    data point's true or predicted label changes. It subtracts the contribution of the
    old labels and adds the contribution of the new labels, avoiding a full recomputation
    of the confusion matrix.

    Parameters
    ----------
    conf_matrix : np.ndarray
        A 1D array of shape (4,) representing the current confusion matrix:
        [true_positives, false_positives, false_negatives, true_negatives].
    old_true : int
        The previous true label (0 or 1) of the data point.
    old_pred : int
        The previous predicted label (0 or 1) of the data point.
    new_true : int
        The updated true label (0 or 1) of the data point.
    new_pred : int
        The updated predicted label (0 or 1) of the data point.

    Returns
    -------
    conf_matrix : np.ndarray
        The updated confusion matrix after applying the label changes.
    """
    conf_matrix[0] -= (not old_true and not old_pred) - (not new_true and not new_pred)
    conf_matrix[1] -= (old_true and not old_pred) - (new_true and not new_pred)
    conf_matrix[2] -= (not old_true and old_pred) - (not new_true and new_pred)
    conf_matrix[3] -= (old_true and old_pred) - (new_true and new_pred)

    return conf_matrix


@njit(fastmath=True, cache=True)
def _binary_macro_f1_score(conf_matrix):
    """
    Computes the macro-averaged F1 score for binary classification.

    This function calculates the F1 score for each class (0 and 1) based on the
    provided confusion matrix and returns their average. The F1 score is the harmonic
    mean of precision and recall. If precision or recall is undefined for a class
    (due to division by zero), the function returns negative infinity.

    Parameters
    ----------
    conf_matrix : np.ndarray
        A 1D array of shape (4,) containing the confusion matrix components for label `0`:
        - `tp` (True Positives): Correctly predicted label 0.
        - `fp` (False Positives): Incorrectly predicted label 0.
        - `fn` (False Negatives): Missed label 0.
        - `tn` (True Negatives): Correctly predicted label 1.

    Returns
    -------
    score : float
        The macro-averaged F1 score for binary classification. Returns -inf if the
        score is undefined due to zero division.
    """
    score = 0

    for label in (0, 1):
        # confusion matrix is symmetric
        if label == 0:
            tp, fp, fn, _ = conf_matrix
        else:
            _, fn, fp, tp = conf_matrix

        if (tp + fp) == 0 or (tp + fn) == 0:
            return -np.inf

        pr = tp / (tp + fp)
        re = tp / (tp + fn)

        if (pr + re) == 0:
            return -np.inf

        score += 2 * (pr * re) / (pr + re)

    return score / 2


@njit(fastmath=True, cache=True)
def _binary_balanced_accuracy_score(conf_matrix):
    """
    Computes the balanced accuracy score for binary classification.

    This function calculates the balanced accuracy for each class (0 and 1) based on the
    provided confusion matrix and returns their average. Balanced accuracy is the average
    of sensitivity (recall) and specificity. If the total number of samples is zero, the
    function returns negative infinity.

    Parameters
    ----------
    conf_matrix : np.ndarray
        A 1D array of shape (4,) containing the confusion matrix components for label `0`:
        - `tp` (True Positives): Correctly predicted label 0.
        - `fp` (False Positives): Incorrectly predicted label 0.
        - `fn` (False Negatives): Missed label 0.
        - `tn` (True Negatives): Correctly predicted label 1.

    Returns
    -------
    score : float
        The balanced accuracy score for binary classification. Returns -inf if the
        score is undefined due to division by zero.
    """
    score = 0

    for label in (0, 1):
        # confusion matrix is symmetric
        if label == 0:
            tp, fp, fn, tn = conf_matrix
        else:
            tn, fn, fp, tp = conf_matrix

        if (tp + fp + fn + tn) == 0:
            return -np.inf

        acc = (tp + tn) / (tp + fp + fn + tn)
        score += acc

    return score / 2


@njit(fastmath=True, cache=True)
def _update_labels(split_idx, excl_zone, rnn, knn_counts, y_true, y_pred, conf_matrix):
    """
    Updates predicted and true labels based on a shift in the split index.

    This function adjusts labels after changing the split point that separates two
    segments of a time series. It updates the true label at the split, propagates the
    effect to reverse nearest neighbors (RNN), modifies predicted labels using majority
    voting, and updates the binary confusion matrix accordingly.

    Parameters
    ----------
    split_idx : int
        The new index separating the two segments of the time series.
    excl_zone : tuple of int
        A tuple (excl_start, excl_end) specifying the exclusion zone range.
    rnn : tuple of np.ndarray
        A tuple (offsets, values) representing the reverse nearest neighbor structure.
    knn_counts : tuple of np.ndarray
        A tuple (knn_zeros, knn_ones) holding the counts of 0s and 1s among each time point's neighbors.
    y_true : np.ndarray
        A 1D array of true labels for each time point.
    y_pred : np.ndarray
        A 1D array of predicted labels based on k-NN majority voting.
    conf_matrix : np.ndarray
        A 1D array of shape (4,) representing the current confusion matrix.

    Returns
    -------
    y_true : np.ndarray
        The updated array of true labels.
    y_pred : np.ndarray
        The updated array of predicted labels.
    conf_matrix : np.ndarray
        The updated confusion matrix after label adjustments.
    """
    offsets, values = rnn
    excl_start, excl_end = excl_zone
    knn_zeros, knn_ones = knn_counts

    ind = values[offsets[split_idx]:offsets[split_idx + 1]]

    if ind.shape[0] > 0:
        ind = np.append(ind, split_idx)
    else:
        ind = np.array([split_idx])

    for pos in ind:
        if pos != split_idx:
            knn_zeros[pos] += 1
            knn_ones[pos] -= 1

        in_excl_zone = excl_end > pos >= excl_start
        zeros, ones = knn_zeros[pos], knn_ones[pos]

        # predict majority label
        label = zeros < ones

        if not in_excl_zone:
            conf_matrix = _update_binary_conf_matrix(conf_matrix, y_true[pos], y_pred[pos], y_true[pos], label)

        y_pred[pos] = label

    y_true[split_idx] = 0

    # update exclusion zone range
    conf_matrix = _update_binary_conf_matrix(conf_matrix, y_true[excl_end], y_pred[excl_end], y_true[excl_start],
                                             y_pred[excl_start])

    return y_true, y_pred, conf_matrix


@njit(fastmath=True, cache=True)
def _profile(knn_offsets, window_size, min_seg_size):
    """
    Computes a classification score profile using a k-NN classifier and macro F1 score.

    This function evaluates the quality of binary splits across a time series by computing
    a profile of macro-averaged F1 scores. For each possible split index, it measures how
    well the time series can be classified into two segments using k-nearest neighbor voting,
    excluding a moving window around the current split to reduce transition effects.

    Parameters
    ----------
    knn_offsets : np.ndarray
        A 2D array of shape (n_timepoints, k_neighbors) containing k-nearest neighbor indices
        for each time point.
    window_size : int
        The size of the k-NN subsequences and exclusion zone around each split index, used to
        ignore ambiguous transitions.
    min_seg_size : int
        The initial offset from both ends of the time series to avoid boundary effects.

    Returns
    -------
    profile : np.ndarray
        A 1D array of shape (n_timepoints,) where each value represents the macro F1 score
        at a given split index. Positions that cannot be evaluated are set to -inf.
    """
    n_timepoints = knn_offsets.shape[0]
    profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float64)

    rnn = _rnn(knn_offsets)
    knn_counts, y_true, y_pred = _init_labels(knn_offsets, min_seg_size)
    conf_matrix = _init_binary_conf_matrix(y_true, y_pred)

    excl_start, excl_end = min_seg_size, min_seg_size + window_size
    excl_conf_matrix = _init_binary_conf_matrix(y_true[excl_start:excl_end], y_pred[excl_start:excl_end])
    conf_matrix = conf_matrix - excl_conf_matrix

    for split_idx in range(min_seg_size, n_timepoints - min_seg_size):
        profile[split_idx] = _binary_macro_f1_score(conf_matrix)

        _update_labels(split_idx, (excl_start, excl_end), rnn, knn_counts, y_true, y_pred, conf_matrix)

        excl_start += 1
        excl_end += 1

    return profile


class ClaSS(ClaSP):
    """
    An implementation of the ClaSS algorithm for detecting change points in streaming time series data.

    Parameters
    ----------
    window_size : int, optional
        The size of the window used for computing distances and offsets, by default 10.
    k_neighbours : int, optional
        The number of nearest neighbors to consider when computing distances and offsets, by default 3.
    distance: str
        The name of the distance function to be computed for determining the k-NNs. Available options are
        "znormed_euclidean_distance", "cinvariant_euclidean_distance", and "euclidean_distance".
    score : str or callable, optional
        The name of the classification score to use.
        Available options are "f1" and "accuracy".
    excl_radius : int, optional
        The radius of the exclusion zone around the detected change point, by default 5*window_size.

    Methods
    -------
    fit(time_series)
        Create a ClaSS for the input time series window.
    transform()
        Return the ClaSS for the input time series window.
    fit_transform(time_series)
        Create and return a ClaSS for the input time series window.
    split()
        Split ClaSS into two segments.
    """

    def __init__(self, window_size=10, k_neighbours=3, distance="znormed_euclidean_distance", score="f1",
                 excl_radius=5):
        super().__init__(window_size, k_neighbours, distance, "f1", excl_radius, 1)

        self.score_name = score

        if score == "f1":
            self.score = _binary_macro_f1_score
        elif score == "accuracy":
            self.score = _binary_balanced_accuracy_score
        else:
            raise ValueError(f"{score} is not a valid score. Implementations include: f1, accuracy.")

    def fit(self, time_series, knn=None):
        """
        Fits the ClaSS model to the input time series data.

        Parameters
        ----------
        time_series : numpy.ndarray
            The input time series data to fit the model on.

        knn : KSubsequenceNeighbours, optional
            Pre-computed KSubsequenceNeighbours object to use for fitting the model.
            If None (default), a new KSubsequenceNeighbours object will be created
            and fitted on the input time series data.

        Returns
        -------
        self : ClaSS
            The fitted ClaSS object.

        Raises
        ------
        ValueError
            If the input time series has less than 2*min_seg_size data points.
        """
        time_series = check_input_time_series(time_series)
        self.min_seg_size = self.window_size * self.excl_radius
        self.lbound, self.ubound = 0, time_series.shape[0]

        if time_series.shape[0] < 2 * self.min_seg_size:
            raise ValueError("Time series must at least have 2*min_seg_size data points.")

        self.time_series = time_series

        if knn is None:
            self.knn = KSubsequenceNeighbours(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
                distance=self.distance,
                n_jobs=self.n_jobs
            ).fit(time_series)
        else:
            self.knn = knn

        self.profile = _profile(self.knn.offsets, self.window_size, self.min_seg_size)

        self.is_fitted = True
        return self
