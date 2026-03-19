import numpy as np
from aeon.classification import DummyClassifier
from aeon.classification.convolution_based import MultiRocketHydraClassifier, RocketClassifier
from aeon.classification.dictionary_based import WEASEL_V2
from aeon.classification.distance_based import ProximityForest
from aeon.classification.feature_based import FreshPRINCEClassifier
from aeon.classification.interval_based import QUANTClassifier
from aeon.classification.shapelet_based._rdst import RDSTClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from claspy.utils import check_input_time_series, check_input_labels


class CLaP:
    """
    An implementation of the CLaP algorithm for state classification in time series data.

    Parameters
    ----------
    window_size : int, optional
        The size of the sliding windows used to construct the classification dataset,
        by default 10.
    classifier : str, optional
        The classifier used for cross-validated state prediction. Available options are
        "dummy", "mrhydra", "rocket", "weasel", "quant", "rdst", "proximityforest",
        "freshprince", and "inception". By default "rocket".
    n_splits : int, optional
        The number of folds used for cross-validation, by default 5.
    sample_size : int, optional
        The maximum number of samples per class used for training and evaluation,
        by default 1_000.
    n_jobs : int, optional
        Amount of parallel jobs used by supported classifiers, by default -1.
    random_state : int, optional
        Seed for the random number generator, by default 2357.

    Methods
    -------
    fit(time_series, y_true, y_pred=None)
        Fit the CLaP model on the input time series and labels.
    transform()
        Return the true and predicted labels.
    fit_transform(time_series, y_true, y_pred=None)
        Fit the CLaP model and return the true and predicted labels.
    score()
        Compute the macro-averaged F1 score between true and predicted labels.
    """

    def __init__(self, window_size=10, classifier="rocket", n_splits=5, sample_size=1_000, n_jobs=-1,
                 random_state=2357):
        self.window_size = window_size
        self.classifier = classifier
        self.n_splits = n_splits
        self.sample_size = sample_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.is_fitted = False

    def _check_is_fitted(self):
        """
        Checks if the CLaP object is fitted.

        Raises
        ------
        NotFittedError
            If the CLaP object is not fitted.

        Returns
        -------
        None
        """
        if not self.is_fitted:
            raise NotFittedError("CLaP object is not fitted yet. Please fit the object before using this method.")

    def _create_dataset(self, time_series, state_labels):
        """
        Create a window-based classification dataset from a labeled time series.

        Parameters
        ----------
        time_series : np.ndarray
            The input time series of shape (n_samples,) or (n_samples, n_dimensions).
        state_labels : np.ndarray
            Array of state labels for each time point.

        Returns
        -------
        tuple
            A tuple ``(X, y)`` where ``X`` contains the extracted windows and ``y`` the
            corresponding state labels.
        """
        sample_size = self.window_size
        stride = sample_size // 2

        label_diffs = state_labels[:-1] != state_labels[1:]
        change_points = np.arange(state_labels.shape[0] - 1)[label_diffs] + 1

        X, y = [], []

        excl_zone = np.full(time_series.shape[0], fill_value=False, dtype=bool)

        for cp in change_points:
            excl_zone[cp - sample_size // 2 + 1:cp] = True

        for idx in range(0, time_series.shape[0] - sample_size + 1, stride):
            if not excl_zone[idx]:
                window = time_series[idx:idx + sample_size].T
                if time_series.shape[1] == 1: window = window.flatten()

                X.append(window)
                y.append(state_labels[idx])

        return np.array(X, dtype=float), np.array(y, dtype=int)

    def _subselect_dataset(self, X, y):
        """
        Subsample the dataset such that at most ``sample_size`` samples per class are kept.

        Parameters
        ----------
        X : np.ndarray
            The input samples.
        y : np.ndarray
            The corresponding class labels.

        Returns
        -------
        tuple
            A tuple ``(X_sel, y_sel)`` containing the subsampled and shuffled dataset.
        """
        np.random.seed(self.random_state)
        labels = np.unique(y)

        X_sel, y_sel = [], []

        for label in labels:
            args = y == label
            X_label, y_label = X[args], y[args]

            if X_label.shape[0] > self.sample_size:
                rand_idx = np.random.choice(np.arange(y.shape[0])[args], self.sample_size, replace=False)
                X_label, y_label = X[rand_idx], y[rand_idx]

            X_sel.extend(X_label)
            y_sel.extend(y_label)

        X_sel, y_sel = np.array(X_sel, dtype=float), np.array(y_sel, dtype=int)

        # randomize order
        args = np.random.choice(X_sel.shape[0], X_sel.shape[0], replace=False)
        return X_sel[args], y_sel[args]

    def _cross_val_classifier(self, X, y):
        """
        Perform cross-validated classification on the given dataset.

        Parameters
        ----------
        X : np.ndarray
            The input samples.
        y : np.ndarray
            The corresponding class labels.

        Returns
        -------
        tuple
            A tuple ``(y_true, y_pred)`` containing the true and predicted labels across
            all cross-validation folds.

        Raises
        ------
        ValueError
            If the configured classifier is not supported.
        """
        y_true = np.zeros_like(y)
        y_pred = np.zeros_like(y)

        n_splits = min(X.shape[0], self.n_splits)

        if n_splits < 2:
            return np.copy(y), np.copy([y])

        for train_idx, test_idx in KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state).split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            if self.classifier == "dummy":
                clf = DummyClassifier(strategy="uniform", random_state=self.random_state)
            elif self.classifier == "mrhydra":
                clf = MultiRocketHydraClassifier(n_jobs=self.n_jobs, random_state=self.random_state)
            elif self.classifier == "rocket":
                clf = RocketClassifier(n_jobs=self.n_jobs, random_state=self.random_state)
            elif self.classifier == "weasel":
                clf = WEASEL_V2(random_state=self.random_state, n_jobs=self.n_jobs)
            elif self.classifier == "quant":
                clf = QUANTClassifier(random_state=self.random_state)
            elif self.classifier == "rdst":
                clf = RDSTClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            elif self.classifier == "proximityforest":
                clf = ProximityForest(n_jobs=self.n_jobs, random_state=self.random_state)
            elif self.classifier == "freshprince":
                clf = FreshPRINCEClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            elif self.classifier == "inception":
                from aeon.classification.deep_learning import IndividualInceptionClassifier
                clf = IndividualInceptionClassifier()
            else:
                raise ValueError(f"The classifier {self.classifier} is not supported.")

            y_true[test_idx] = y_test
            y_pred[test_idx] = clf.fit(X_train, y_train).predict(X_test)

        return y_true, y_pred

    def fit(self, time_series, y_true, y_pred=None):
        """
        Fit the CLaP model on the given time series and labels.

        Parameters
        ----------
        time_series : np.ndarray
            The input time series of shape (n_samples,) or (n_samples, n_dimensions).
        y_true : np.ndarray
            The ground-truth state labels for the input time series.
        y_pred : np.ndarray, optional
            Precomputed predicted labels. If None (default), predictions are obtained
            using cross-validated classification.

        Returns
        -------
        self : CLaP
            The fitted CLaP object.
        """
        self.time_series = check_input_time_series(time_series)
        y_true = check_input_labels(y_true)

        if y_pred is None:
            X, y = self._create_dataset(self.time_series, y_true)
            self.y_true, self.y_pred = self._cross_val_classifier(*self._subselect_dataset(X, y))
        else:
            self.y_true, self.y_pred = y_true, check_input_labels(y_pred)

        self.is_fitted = True
        return self

    def transform(self):
        """
        Return the true and predicted labels.

        Returns
        -------
        tuple
            A tuple ``(y_true, y_pred)`` containing the true and predicted labels.

        Raises
        ------
        NotFittedError
            If the CLaP object is not fitted.
        """
        self._check_is_fitted()
        return self.y_true, self.y_pred

    def fit_transform(self, time_series, y_true, y_pred=None):
        """
        Fit the CLaP model and return the true and predicted labels.

        Parameters
        ----------
        time_series : np.ndarray
            The input time series of shape (n_samples,) or (n_samples, n_dimensions).
        y_true : np.ndarray
            The ground-truth state labels for the input time series.
        y_pred : np.ndarray, optional
            Precomputed predicted labels. If None (default), predictions are obtained
            using cross-validated classification.

        Returns
        -------
        tuple
            A tuple ``(y_true, y_pred)`` containing the true and predicted labels.
        """
        return self.fit(time_series, y_true, y_pred).transform()

    def score(self):
        """
        Compute the macro-averaged F1 score between true and predicted labels.

        Returns
        -------
        float
            The macro-averaged F1 score.

        Raises
        ------
        NotFittedError
            If the CLaP object is not fitted.
        """
        self._check_is_fitted()
        return f1_score(self.y_true, self.y_pred, average="macro")
