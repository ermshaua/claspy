import numpy as np
from numba import njit

from claspy.nearest_neighbour import KSubsequenceNeighbours
from claspy.scoring import map_scores
from claspy.utils import check_input_time_series


@njit(fastmath=True, cache=True)
def _cross_val_labels(offsets, split_idx, window_size):
    '''
    Generate predicted and true labels for cross-validation based on nearest neighbour distances.

    Parameters
    ----------
    offsets : ndarray of shape (n_timepoints, k_neighbours)
        The indices of the nearest neighbours for each timepoint in the time series. These indices
        are relative to the start of the time series and should be positive integers.
    split_idx : int
        The index at which to split the time series into two potential segments. This index should be
        less than n_timepoints and greater than window_size.
    window_size : int
        The size of the window used to calculate nearest neighbours.

    Returns
    -------
    y_true : ndarray of shape (n_timepoints,)
        The true labels for each timepoint in the time series.
    y_pred : ndarray of shape (n_timepoints,)
        The predicted labels for each timepoint in the time series.
    '''
    n_timepoints, k_neighbours = offsets.shape

    y_true = np.concatenate((
        np.zeros(split_idx, dtype=np.int64),
        np.ones(n_timepoints - split_idx, dtype=np.int64),
    ))

    knn_labels = np.zeros(shape=(k_neighbours, n_timepoints), dtype=np.int64)

    for i_neighbor in range(k_neighbours):
        neighbours = offsets[:, i_neighbor]
        knn_labels[i_neighbor] = y_true[neighbours]

    ones = np.sum(knn_labels, axis=0)
    zeros = k_neighbours - ones
    y_pred = np.asarray(ones > zeros, dtype=np.int64)

    exclusion_zone = np.arange(split_idx - window_size, split_idx)
    y_pred[exclusion_zone] = 1

    return y_true, y_pred


@njit(fastmath=True, cache=False)
def _profile(offsets, window_size, score, min_seg_size):
    '''
    Computes the classification score profile given nearest neighbour offsets.

    Parameters
    ----------
    offsets : np.ndarray
        An array of shape (n_timepoints, k_neighbors) containing the offsets of the k nearest
        neighbors for each timepoint.
    window_size : int
        The size of the window used to calculate nearest neighbours.
    score : callable
        A callable that computes the score of a segmentation given the true and predicted labels.
        The callable must accept two arguments: y_true and y_pred, which are arrays of binary labels
        indicating whether each timepoint belongs to the first or second segment of the segmentation.
    min_seg_size : int
        The minimum length of a segment. Segments shorter than this will be ignored.

    Returns
    -------
    np.ndarray
        An array of shape (n_timepoints,) containing the classification score profile.
    '''
    n_timepoints, _ = offsets.shape
    profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float64)

    for split_idx in range(min_seg_size, n_timepoints - min_seg_size):
        y_true, y_pred = _cross_val_labels(offsets, split_idx, window_size)
        profile[split_idx] = score(y_true, y_pred)

    return profile


class ClaSP:
    '''
    An implementation of the ClaSP algorithm for detecting change points in time series data.

    Parameters
    ----------
    window_size : int, optional
        The size of the window used for computing distances and offsets, by default 10.
    k_neighbours : int, optional
        The number of nearest neighbors to consider when computing distances and offsets, by default 3.
    score : str or callable, optional
        The name of the classification score to use.
        Available options are "roc_auc", "f1", by default "roc_auc".
    excl_radius : int, optional
        The radius of the exclusion zone around the detected change point, by default 5*window_size.

    Methods
    -------
    fit(time_series)
        Create a ClaSP for the input time series data.
    split()
        Split ClaSP into two segments.
    '''

    def __init__(self, window_size=10, k_neighbours=3, score="roc_auc", excl_radius=5):
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.score_name = score
        self.score = map_scores(score)
        self.excl_radius = excl_radius

    def fit(self, time_series, knn=None):
        '''
        Fits the ClaSP model to the input time series data.

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
        self : ClaSP
            The fitted ClaSP object.

        Raises
        ------
        ValueError
            If the input time series has less than 2*min_seg_size data points.
        '''
        check_input_time_series(time_series)
        self.min_seg_size = self.window_size * self.excl_radius

        if time_series.shape[0] < 2 * self.min_seg_size:
            raise ValueError("Time series must at least have 2*min_seg_size data points.")

        self.time_series = time_series

        if knn is None:
            self.knn = KSubsequenceNeighbours(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours
            ).fit(time_series)
        else:
            self.knn = knn

        self.profile = _profile(self.knn.offsets, self.window_size, self.score, self.min_seg_size)
        return self

    def transform(self):
        '''
        Transform the input time series into a ClaSP profile.

        Returns
        -------
        profile : numpy.ndarray
            The ClaSP profile for the input time series.
        '''
        return self.profile

    def fit_transform(self, time_series, knn=None):
        '''
        Fit the ClaSP algorithm to the given time series and return the
        corresponding profile.

        Parameters
        ----------
        time_series : np.ndarray, shape (n_timepoints,)
            The input time series to be segmented.
        knn : KSubsequenceNeighbours, optional
            Pre-computed KSubsequenceNeighbours object to use for fitting the model.
            If None (default), a new KSubsequenceNeighbours object will be created
            and fitted on the input time series data.

        Returns
        -------
        np.ndarray, shape (n_timepoints,)
            The ClaSP scores corresponding to each time point of the input time series.

        '''
        return self.fit(time_series, knn).transform()

    def split(self, sparse=True):
        '''
        Split the time series into two segments using the change point location.

        Parameters
        ----------
        sparse : bool, optional
            If True, returns only the index of the change point. If False, returns the two segments
            separated by the change point. Default is True.

        Returns
        -------
        int or tuple
            If `sparse` is True, returns the index of the change point. If False, returns a tuple
            of the two segments separated by the change point.
        '''
        cp = np.argmax(self.profile)

        if sparse is True:
            return cp

        return self.time_series[:cp], self.time_series[cp:]


class ClaSPEnsemble(ClaSP):
    '''
    An ensemble of ClaSP.

    Parameters
    ----------
    n_estimators : int, optional
        The number of ClaSP models to use in the ensemble. Default is 10.
    window_size : int, optional
        The size of the window used for the k-subsequence neighbours. Default is 10.
    k_neighbours : int, optional
        The number of nearest neighbours to consider in the k-subsequence method. Default is 3.
    score : str or callable, optional
        The scoring method to use in the profile scoring. Must be a string ('f1' 'roc_auc',).
        Default is 'roc_auc'.
    excl_radius : int, optional
        The radius of the exclusion zone in the profile scoring. Default is 5*window_size.
    random_state : int or RandomState, optional
        Seed for the random number generator. Default is 2357.

    Methods
    -------
    fit(time_series)
        Create a ClaSP ensemble for the input time series data.
    '''

    def __init__(self, n_estimators=10, window_size=10, k_neighbours=3, score="roc_auc", excl_radius=5,
                 random_state=2357):
        super().__init__(window_size, k_neighbours, score, excl_radius)
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _calculate_temporal_constraints(self):
        '''
        Calculates a set of random temporal constraints for each ClaSP in the ensemble.

        Returns
        -------
        tcs : ndarray of shape (n_estimators, 2)
            Array of start and end indices for each temporal constraint.
        '''
        tcs = [(0, self.time_series.shape[0])]
        np.random.seed(self.random_state)

        while len(tcs) < self.n_estimators and self.time_series.shape[0] > 3 * self.min_seg_size:
            lbound, area = np.random.choice(self.time_series.shape[0], 2, replace=True)

            if self.time_series.shape[0] - lbound < area:
                area = self.time_series.shape[0] - lbound

            ubound = lbound + area
            if ubound - lbound < 2 * self.min_seg_size: continue
            tcs.append((lbound, ubound))

        return np.asarray(tcs, dtype=np.int64)

    def fit(self, time_series, knn=None):
        '''
        Fits the ClaSP ensemble on the given time series, using temporal constraints to so that
        each ClaSP instance works on different (but possibly overlapping) parts of the time series.

        Parameters
        ----------
        time_series : np.ndarray
            The input time series of shape (n_samples,).
        knn : Optional[KSubsequenceNeighbours], default=None
            The precomputed KSubsequenceNeighbours object to use. If None, it will be computed
            using `KSubsequenceNeighbours` with the `window_size` and `k_neighbours` parameters
            passed during the object initialization.

        Returns
        -------
        ClaSPEnsemble
            The fitted ClaSPEnsemble object.

        Raises
        ------
        ValueError
            If the input time series has less than 2 times the minimum segment size.
        '''
        check_input_time_series(time_series)
        self.min_seg_size = self.window_size * self.excl_radius

        if time_series.shape[0] < 2 * self.min_seg_size:
            raise ValueError("Time series must at least have 2*min_seg_size data points.")

        self.time_series = time_series
        tcs = self._calculate_temporal_constraints()

        if knn is None:
            knn = KSubsequenceNeighbours(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
            ).fit(time_series, temporal_constraints=tcs)

        best_score, best_tc, best_clasp = -np.inf, None, None

        for idx, (lbound, ubound) in enumerate(tcs):
            clasp = ClaSP(
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
                score=self.score_name,
                excl_radius=self.excl_radius
            ).fit(time_series[lbound:ubound], knn=knn.constrain(lbound, ubound))

            clasp.profile = (clasp.profile + (ubound - lbound) / time_series.shape[0])

            if clasp.profile.max() > best_score or best_clasp is None and idx == tcs.shape[0] - 1:
                best_score = clasp.profile.max()
                best_tc = (lbound, ubound)
                best_clasp = clasp

        self.knn = best_clasp.knn
        self.profile = np.full(shape=time_series.shape[0] - self.window_size + 1, fill_value=-np.inf, dtype=np.float64)

        lbound, ubound = best_tc
        self.profile[lbound:ubound - self.window_size + 1] = best_clasp.profile

        return self
