import hashlib

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix, f1_score

from claspy.clap import CLaP
from claspy.segmentation import BinaryClaSPSegmentation
from claspy.utils import create_state_labels, check_input_time_series, check_input_change_points
from claspy.window_size import map_window_size_methods


class AgglomerativeCLaPDetection:
    """
    Detect recurring states in time series data using agglomerative merging with CLaP.

    Reference:
    Ermshaus, A., Schäfer, P., & Leser, U. (2025). CLaP - State Detection from Time Series.
    Proceedings of the VLDB Endowment, 19 (1), 70 - 83. https://doi.org/10.14778/3772181.3772187

    Parameters
    ----------
    window_size : str or int, default="suss"
        The window size detection method or size of the sliding window used for state
        classification. Valid implementations include: "suss", "fft", and "acf".

    classifier : str, default="rocket"
        The classifier used in CLaP for distinguishing between segments.

    n_splits : int, default=5
        The number of cross-validation folds used in CLaP.

    sample_size : int, default=1_000
        The maximum number of samples per class used for classification.

    n_jobs : int, optional (default=-1)
        Amount of parallel jobs used by supported classifiers.

    random_state : int, default=2357
        Seed for the random number generator.

    Methods
    -------
    fit(time_series, change_points=None)
        Fit the AgglomerativeCLaPDetection model to the input time series.
    predict(sparse=False)
        Predict state labels or the state transition process for the input time series.
    fit_predict(time_series, change_points=None, sparse=False)
        Fit the model and predict states or the state transition process.
    score()
        Compute the classification gain of the fitted state detection.
    get_segment_labels()
        Return the detected state label for each segment.
    get_change_points()
        Return the change points separating detected states.
    get_process()
        Return the detected states and transitions.
    plot(sparse=False)
        Visualize the detected states or the transition process.
    """

    def __init__(self, window_size="suss", classifier="rocket", n_splits=5, sample_size=1_000, n_jobs=-1,
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
        Checks if the AgglomerativeCLaPDetection object is fitted.

        Raises
        ------
        NotFittedError
            If the AgglomerativeCLaPDetection object is not fitted.

        Returns
        -------
        None
        """
        if not self.is_fitted:
            raise NotFittedError(
                "AgglomerativeCLaPDetection object is not fitted yet. Please fit the object before using this method.")

    def fit(self, time_series, change_points=None):
        """
        Fit the agglomerative state detection model to the input time series.

        Parameters
        ----------
        time_series : array-like of shape (n_samples,) or (n_samples, d_dimensions)
            The input time series.
        change_points : array-like of shape (n_cps,), optional
            Precomputed change points. If None (default), change points are estimated
            using BinaryClaSPSegmentation.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        np.random.seed(self.random_state)

        self.time_series = check_input_time_series(time_series)

        if change_points is None:
            self.change_points = BinaryClaSPSegmentation(n_jobs=self.n_jobs).fit_predict(self.time_series)
        else:
            self.change_points = check_input_change_points(change_points)

        W = []

        if isinstance(self.window_size, str):
            wss = map_window_size_methods(self.window_size)

            for dim in range(self.time_series.shape[1]):
                W.append(max(1, wss(self.time_series[:, dim])))

            if len(W) > 0:
                self.window_size = int(np.mean(W))
            else:
                self.window_size = 10

        labels = np.arange(self.change_points.shape[0] + 1)

        merged = True
        ignore_cache = set()

        state_labels = create_state_labels(self.change_points, labels, self.time_series.shape[0])
        clap = CLaP(window_size=self.window_size, classifier=self.classifier, n_splits=self.n_splits,
                    sample_size=self.sample_size, n_jobs=self.n_jobs, random_state=self.random_state)
        y_true, y_pred = clap.fit_transform(self.time_series, state_labels)

        while merged and np.unique(labels).shape[0] > 1:
            unique_labels = np.unique(labels)

            conf_loss = np.zeros(unique_labels.shape[0], dtype=float)
            conf_index = np.zeros(unique_labels.shape[0], dtype=int)

            # calculate confusions
            for idx, conf in enumerate(confusion_matrix(y_true, y_pred)):
                # drop TPs
                tmp = conf.copy()
                tmp[idx] = 0

                # store most confused label
                conf_index[idx] = np.argmax(tmp)
                conf_loss[idx] = np.max(tmp) / np.sum(conf)

            merged = False

            # merge most confused classes (with descending confusion)
            for idx in np.argsort(conf_loss)[::-1]:
                label1, label2 = unique_labels[idx], unique_labels[conf_index[idx]]

                if label1 == label2 or label1 not in labels or label2 not in labels:
                    continue

                test_idx = np.logical_or(y_true == label1, y_true == label2)
                test_key = hashlib.sha256(test_idx.tobytes()).hexdigest()

                if test_key in ignore_cache:
                    continue

                _y_true, _y_pred = y_true.copy(), y_pred.copy()

                _y_true[_y_true == label2] = label1
                _y_pred[_y_pred == label2] = label1

                if self._classification_gain(y_true, y_pred) > self._classification_gain(_y_true, _y_pred):
                    ignore_cache.add(test_key)
                    continue

                if label2 > label1:
                    label1, label2 = label2, label1

                labels[labels == label2] = label1

                y_true[y_true == label2] = label1
                y_pred[y_pred == label2] = label1

                merged = True
                break

        # map labels from 1 to n
        label_mapping = {label: idx + 1 for idx, label in enumerate(np.unique(labels))}

        self.labels = np.array([label_mapping[label] for label in labels], dtype=int)
        self.clap = CLaP(window_size=self.window_size, classifier=self.classifier, n_splits=self.n_splits,
                         sample_size=self.sample_size, n_jobs=self.n_jobs, random_state=self.random_state)
        clap.fit(self.time_series, y_true, y_pred)

        self.is_fitted = True
        return self

    def predict(self, sparse=False):
        """
        Predict the state sequence or the state transition process.

        Parameters
        ----------
        sparse : bool, optional (default=False)
            Whether to return the sparse process representation instead of dense
            state labels.

        Returns
        -------
        If `sparse` is True, returns a tuple containing the detected states and transitions.
        Otherwise, returns an array of state labels for each time point.
        """
        self._check_is_fitted()

        if sparse is True:
            return self.get_process()

        return create_state_labels(self.get_change_points(), self.get_segment_labels(), self.time_series.shape[0])

    def fit_predict(self, time_series, change_points=None, sparse=False):
        """
        Fit the model to the provided time series and predict states.

        Parameters
        ----------
        time_series : array-like of shape (n_samples,) or (n_samples, d_dimensions)
            The input time series.
        change_points : array-like of shape (n_cps,), optional
            Precomputed change points. If None (default), change points are estimated
            automatically.
        sparse : bool, optional (default=False)
            Whether to return the sparse process representation instead of dense
            state labels.

        Returns
        -------
        If `sparse` is True, returns a tuple containing the detected states and transitions.
        Otherwise, returns an array of state labels for each time point.
        """
        return self.fit(time_series, change_points).predict(sparse=sparse)

    def _random_f1_score(self, y_true):
        """
        Compute the expected macro-F1 score of a random classifier.

        Parameters
        ----------
        y_true : np.ndarray
            The ground-truth class labels.

        Returns
        -------
        float
            The expected macro-F1 score under random predictions.
        """
        labels = np.unique(y_true)
        score = 0

        for label in labels:
            pos_instances = np.sum(y_true == label)
            neg_instances = np.sum(y_true != label)

            # likelihood of occurrence
            tp = pos_instances * pos_instances / y_true.shape[0]
            fn = pos_instances * neg_instances / y_true.shape[0]
            fp = neg_instances * pos_instances / y_true.shape[0]

            pre = tp / (tp + fp)
            re = tp / (tp + fn)

            if pre + re > 0:
                score += 2 * (pre * re) / (pre + re)

        return score / labels.shape[0]

    def _classification_gain(self, y_true, y_pred):
        """
        Compute the classification gain over a random baseline.

        Parameters
        ----------
        y_true : np.ndarray
            The ground-truth labels.
        y_pred : np.ndarray
            The predicted labels.

        Returns
        -------
        float
            The difference between macro-F1 and the expected random macro-F1 score.
        """
        cross_val_score = f1_score(y_true, y_pred, average="macro")
        rand_score = self._random_f1_score(y_true)
        return cross_val_score - rand_score

    def score(self):
        """
        Compute the classification gain of the fitted state detection.

        Returns
        -------
        float
            The classification gain of the fitted model.

        Raises
        ------
        NotFittedError
            If the AgglomerativeCLaPDetection object is not fitted.
        """
        self._check_is_fitted()
        return self._classification_gain(*self.clap.transform())

    def get_segment_labels(self):
        """
        Return the detected state label for each segment.

        Returns
        -------
        np.ndarray
            An array containing one state label per segment.
        """
        self._check_is_fitted()
        labels = [self.labels[0]]

        for idx in np.arange(1, self.labels.shape[0]):
            if labels[-1] != self.labels[idx]:
                labels.append(self.labels[idx])

        return np.asarray(labels)

    def get_change_points(self):
        """
        Return the change points separating detected states.

        Returns
        -------
        np.ndarray
            An array containing the change point indices between consecutive states.
        """
        self._check_is_fitted()
        labels = [self.labels[0]]
        change_points = []

        for idx in np.arange(1, self.labels.shape[0]):
            if labels[-1] != self.labels[idx]:
                labels.append(self.labels[idx])
                change_points.append(self.change_points[idx - 1])

        return np.asarray(change_points)

    def get_process(self):
        """
        Return the detected process as states and transitions.

        Returns
        -------
        tuple
            A tuple ``(states, transitions)`` where ``states`` is the set of detected
            states and ``transitions`` is the set of observed state transitions.
        """
        labels = self.get_segment_labels()

        states = set(labels)
        transitions = set([(labels[idx], labels[idx + 1]) for idx in range(0, len(labels) - 1)])

        return states, transitions

    def plot(self, gt_states=None, heading=None, ts_name=None, file_path=None, fig_size=None, font_size=26,
             sparse=False):
        """
        Plot the fitted state detection or the inferred transition process.

        Parameters
        ----------
        gt_states : array-like, optional (default=None)
            Ground-truth state annotations used for visualization.
        heading : str, optional (default=None)
            Title of the plot.
        ts_name : str or list of str, optional (default=None)
            Name of the time series or dimension labels.
        file_path : str, optional (default=None)
            If provided, the resulting plot will be saved to this file.
        fig_size : tuple of int, optional (default=None)
            The size of the resulting figure. Defaults depend on `sparse`.
        font_size : int, optional (default=26)
            The font size used for titles and axis labels.
        sparse : bool, optional (default=False)
            Whether to plot the sparse process representation instead of the dense
            state detection.

        Returns
        -------
        matplotlib.Axes or ndarray of matplotlib.Axes
            The axes of the resulting figure.
        """
        self._check_is_fitted()

        if sparse is True:
            if fig_size is None: fig_size = (10, 10)
            return self.plot_process(heading=heading, file_path=file_path, fig_size=fig_size, font_size=font_size)

        if fig_size is None: fig_size = (20, 10)
        return self.plot_state_detection(gt_states=gt_states, heading=heading, ts_name=ts_name, file_path=file_path,
                                         fig_size=fig_size, font_size=font_size)

    def plot_process(self, heading, file_path=None, fig_size=(10, 10), font_size=26):
        """
        Plot the detected transition process as a directed graph.

        Parameters
        ----------
        heading : str
            The title of the plot.
        file_path : str, optional (default=None)
            If provided, the resulting plot will be saved to this file.
        fig_size : tuple of int, optional (default=(10, 10))
            The size of the resulting figure.
        font_size : int, optional (default=26)
            The font size used for titles and node labels.

        Returns
        -------
        ax : matplotlib.Axes
            The axes of the resulting figure.
        """
        state_seq = self.get_segment_labels()

        unique_states, state_indices = np.unique(state_seq, return_inverse=True)
        n_states = len(unique_states)

        transition_counts = np.zeros((n_states, n_states))
        for (i, j) in zip(state_indices[:-1], state_indices[1:]):
            transition_counts[i, j] += 1

        row_sums = transition_counts.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transition_counts, row_sums, out=np.zeros_like(transition_counts),
                                     where=row_sums != 0)

        G = nx.DiGraph()
        for i, state_from in enumerate(unique_states):
            G.add_node(state_from)
            for j, state_to in enumerate(unique_states):
                prob = transition_probs[i, j]
                if prob > 0:
                    G.add_edge(state_from, state_to, weight=prob)

        node_colors = []

        for node in G.nodes():
            if node == state_seq[0]:
                node_colors.append("lightgreen")
            elif node == state_seq[-1]:
                node_colors.append("lightblue")
            else:
                node_colors.append("lightgrey")

        pos = nx.spring_layout(G, seed=42)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

        _, ax = plt.subplots(figsize=fig_size)
        nx.draw(G, pos, ax=ax, with_labels=True, node_size=1000, node_color=node_colors, arrowsize=20,
                font_size=font_size // 2)

        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_color="black",
                                     font_size=font_size // 2)

        ax.set_title(heading, fontsize=font_size)

        if file_path is not None:
            plt.savefig(file_path, bbox_inches="tight")

        return ax

    def plot_state_detection(self, gt_states=None, heading=None, ts_name=None, file_path=None, fig_size=(20, 10),
                             font_size=26):
        """
        Plot the fitted time series annotated with detected states.

        Parameters
        ----------
        gt_states : array-like
            Ground-truth state annotations to plot alongside the time series.
        heading : str, optional (default=None)
            The title of the first subplot.
        ts_name : str or list of str, optional (default=None)
            The name of the time series or dimension labels.
        file_path : str, optional (default=None)
            If provided, the resulting plot will be saved to this file.
        fig_size : tuple of int, optional (default=(20, 10))
            The size of the resulting figure.
        font_size : int, optional (default=26)
            The font size used for titles and axis labels.

        Returns
        -------
        axes : matplotlib.Axes
            The subplots of the resulting figure.
        """
        if isinstance(ts_name, str):
            ts_name = [ts_name]

        if gt_states.ndim == 1:
            gt_states = gt_states.reshape(-1, 1)

        fig, axes = plt.subplots(
            self.time_series.shape[1] + gt_states.shape[1],
            sharex=True,
            gridspec_kw={'hspace': .05},
            figsize=fig_size
        )

        label_colours = {}
        state_colours = {}

        idx = 0
        labels = self.get_segment_labels()

        for label in labels:
            if label not in label_colours:
                label_colours[label] = f"C{idx}"
                idx += 1

        for label in np.unique(gt_states):
            state_colours[label] = f"C{idx}"
            idx += 1

        for dim, ax in enumerate(axes):
            if dim < self.time_series.shape[1]:
                series = self.time_series[:, dim]
            else:
                series = gt_states[:, dim - self.time_series.shape[1]]

            segments = [0] + self.change_points.tolist() + [series.shape[0]]
            colors = label_colours
            annotation = labels

            if len(series) > 0:
                for idx in np.arange(0, len(segments) - 1):
                    ax.plot(
                        np.arange(segments[idx], segments[idx + 1]),
                        series[segments[idx]:segments[idx + 1]],
                        c=colors[annotation[idx]]
                    )

            if ts_name is not None and dim < len(ts_name):
                ax.set_ylabel(ts_name[dim], fontsize=font_size)

            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(font_size)

            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(font_size)

        axes[0].set_title(heading, fontsize=font_size)

        axes[-1].set_xlabel("time point", fontsize=font_size)
        axes[-1].set_ylabel("State", fontsize=font_size)
        axes[-1].locator_params(axis='y', nbins=np.unique(labels).shape[0])

        if file_path is not None:
            plt.savefig(file_path, bbox_inches="tight")

        return axes
