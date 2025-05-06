import numpy as np
from matplotlib import pyplot as plt
from sklearn.exceptions import NotFittedError

from claspy.streaming.clasp import ClaSS
from claspy.streaming.nearest_neighbour import StreamingKSubsequenceNeighbours
from claspy.utils import roll_array
from claspy.window_size import map_window_size_methods


class StreamingClaSPSegmentation:
    """
    Perform online segmentation of time series using the streaming ClaSP algorithm.

    Reference:
    Ermshaus, A., Schäfer, P., & Leser, U. (2024). Raising the ClaSS of Streaming Time Series Segmentation.
    Proceedings of the VLDB Endowment, 17(8), 1953–1966. https://doi.org/10.14778/3659437.3659450

    Parameters
    ----------
    n_timepoints : int, default=10_000
        The total length of the sliding window used for streaming computation.

    n_warmup : int, default=10000
        Number of initial time points used to warm up the model before change point detection starts.

    window_size : str or int, default="suss"
        The window size detection method or fixed window size used in ClaSP.
        Valid options include: 'suss', 'fft', and 'acf'.

    k_neighbours : int, default=3
        The number of nearest neighbors to use in the ClaSP algorithm.

    distance : str, default="znormed_euclidean_distance"
        The distance function used to compute similarity between subsequences.
        Options: "znormed_euclidean_distance", "cinvariant_euclidean_distance", "euclidean_distance".

    score : str, default="f1"
        The scoring metric used in the classification-based ClaSP model.
        Available options include "f1", "accuracy".

    jump : int, default=5
        The step size in time points between two consecutive change point detection attempts.

    validation : str, default="significance_test"
        The method used to validate candidate change points.
        Options: "significance_test", "score_threshold".

    threshold : str or float, default="default"
        Threshold used by the validation method. For "score_threshold", this is a score (e.g., 0.75).
        For "significance_test", this is a p-value (e.g., 1e-50). Use "default" to choose automatically.

    log_cps : bool, default=False
        Determines if change points are logged as a list of global timestamps. This is useful for
        debugging, benchmarking, or finite data sets.

    excl_radius : int, default=5
        The exclusion radius factor around a potential change point to avoid detecting trivial or nearby changes.
        Actual offset is excl_radius * window_size.

    Methods
    -------
    update(timepoint)
        Update the streaming ClaSP model with a new time point and perform change point detection if possible.

    predict(sparse=True)
        Predict the location of detected change points, or return the segmented time series.

    update_predict(timepoint, sparse=True)
        Update the model with a new point and return the latest change point or segmented time series.
    """

    def __init__(self, n_timepoints=10_000, n_warmup=10_000, window_size="suss", k_neighbours=3,
                 distance="znormed_euclidean_distance", score="f1", jump=5, validation="significance_test",
                 threshold="default", log_cps=False, excl_radius=5):
        self.n_timepoints = n_timepoints
        self.n_warmup = min(n_timepoints, n_warmup)
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.distance = distance
        self.score = score
        self.jump = jump
        self.validation = validation
        self.threshold = threshold
        self.log_cps = log_cps
        self.excl_radius = excl_radius

        self.warmup = np.full(shape=self.n_warmup, fill_value=-np.inf, dtype=np.float64)
        self.profile = np.full(shape=n_timepoints, fill_value=-np.inf, dtype=np.float64)

        self.change_points = list()
        self.last_cp = 0

        self.ingested = 0
        self.knn_stream_lag = 0
        self.lag = -1
        self.warmup_counter = 0

    def _check_is_warmed_up(self):
        """
        Checks if the StreamingClaSPSegmentation object is warmed up.

        Raises
        ------
        NotFittedError
            If the BinaryClaSPSegmentation object is not fitted.

        Returns
        -------
        None
        """
        if self.warmup_counter != self.n_warmup:
            raise NotFittedError(
                "StreamingClaSPSegmentation object is not warmed up yet. Please insert more data points before using this method.")

    def _warmup(self, timepoint):
        """
       Performs the warm-up phase by accumulating initial data points and initializing internal structures.

       This method updates the warm-up buffer with a new time point until the required number
       of warm-up points is reached. Once warmed up, it determines the window size, initializes
       segmentation parameters, and constructs the streaming k-nearest neighbors model using the
       buffered data.

       Parameters
       ----------
       timepoint : float
           The latest data point to append to the warm-up buffer.

       Returns
       -------
       self : object or None
           Returns the updated StreamingClaSPSegmentation instance.
        """
        # update warmup ts
        self.warmup_counter += 1

        self.warmup = roll_array(self.warmup, -1)
        self.warmup[-1] = timepoint

        if self.warmup_counter != self.n_warmup:
            return self

        # compute window size
        if isinstance(self.window_size, str):
            self.window_size = map_window_size_methods(self.window_size)(self.warmup)

        self.min_seg_size = 5 * self.window_size

        # init threshold
        if self.threshold == "default":
            if self.validation == "score_threshold":
                self.threshold = 0.75
            elif self.validation == "significance_test":
                self.threshold = 1e-50

        # create streaming k-NN
        self.knn_stream = StreamingKSubsequenceNeighbours(self.n_timepoints, self.window_size, self.k_neighbours,
                                                          self.distance)
        self.knn_stream_lag = self.knn_stream.window_size + self.knn_stream.exclusion_radius + self.knn_stream.k_neighbours

        # insert warmup buffer
        for timepoint in self.warmup:
            self.update(timepoint)

        return self

    def update(self, timepoint):
        """
        Updates the streaming ClaSP segmentation model with a new data point.

        This method appends a new time point to the internal time series, updates the streaming
        k-nearest neighbor model, and performs change point detection based on the ClaSP score.
        If the model is still in the warm-up phase, it accumulates data until it is ready.

        Parameters
        ----------
        timepoint : float
            The latest data point to append to the time series for online segmentation.

        Returns
        -------
        self : object
            The updated StreamingClaSPSegmentation instance containing revised internal state,
            updated profiles, and detected change points (if any).
        """
        if self.warmup_counter < self.n_warmup:
            return self._warmup(timepoint)

        self.ingested += 1

        # update sliding window and k-NNs
        self.knn_stream.update(timepoint, self.last_cp)
        self.profile = roll_array(self.profile, -1, -np.inf)

        # validate constraints
        if self.ingested < self.min_seg_size * 2:
            return self

        if self.knn_stream.knn_insert_idx - self.knn_stream.knn_filled == 0:
            self.last_cp = max(0, self.last_cp - 1)

        profile_start, profile_end = self.knn_stream.lbound, self.knn_stream.knn_insert_idx

        if profile_end - profile_start < 2 * self.min_seg_size or self.ingested % self.jump != 0:
            return self

        # compute ClaSS
        clasz = ClaSS(window_size=self.window_size, k_neighbours=self.k_neighbours,
                      distance=self.distance, score=self.score, excl_radius=self.excl_radius)

        profile = clasz.fit_transform(self.knn_stream.time_series, self.knn_stream.transform())
        cp = clasz.split(validation=self.validation, threshold=self.threshold)

        if cp is None or cp < self.min_seg_size or profile.shape[0] - cp < self.min_seg_size:
            return self

        if profile[cp:-self.min_seg_size].shape[0] == 0:
            return self

        not_ninf = np.logical_not(profile == -np.inf)
        tc = profile[not_ninf].shape[0] / self.n_timepoints
        profile[not_ninf] = (2 * profile[not_ninf] + tc) / 3

        self.profile[profile_start:profile_end] = np.max([profile, self.profile[profile_start:profile_end]], axis=0)
        self.last_cp += cp

        if self.log_cps is True:
            global_cp = self.ingested - self.knn_stream_lag - (profile_end - profile_start) + cp + self.window_size
            self.change_points.append(global_cp)

        return self

    def predict(self, sparse=True):
        """
        Predicts the location of the last change point in the streaming time series.

        Parameters
        ----------
        sparse : bool, optional (default=True)
            Whether to return a sparse or a dense output.

        Returns
        -------
        If `sparse` is True, returns the change point as an integer.
        Otherwise, returns a list of numpy arrays, where each array corresponds to a segment in the streaming time series.
        """
        self._check_is_warmed_up()

        if sparse is True:
            return self.last_cp

        seg_idx = np.unique([0, self.last_cp, self.knn_stream.time_series.shape[0]])
        return [self.knn_stream.time_series[seg_idx[idx]:seg_idx[idx + 1]] for idx in range(len(seg_idx) - 1)]

    def update_predict(self, timepoint, sparse=True):
        """
        Updates the streaming ClaSP segmentation model with a new data point and predicts last change point.

        Parameters:
        -----------
        timepoint : float
            The latest data point to append to the sliding window.
        sparse : bool, optional (default=True)
            Whether to return the change point index only, or to return the segmented time series.

        Returns:
        --------
        If `sparse` is True, returns the change point as an integer.
        Otherwise, returns a list of numpy arrays, where each array corresponds to a segment in the streaming time series.
        """
        return self.update(timepoint).predict(sparse)

    def plot(self, heading=None, stream_name=None, fig_size=(20, 10), font_size=26, file_path=None):
        """
        Plot the current sliding window annotated with ClaSP and last CP.

        Parameters
        ----------
        heading : str, optional (default=None)
            The title of the first subplot (i.e., the time series plot).
        stream_name : str, optional (default=None)
            The name of the time series stream (used as y-label for the time series subplot).
        fig_size : tuple of int, optional (default=(20, 10))
            The size of the resulting figure.
        font_size : int, optional (default=26)
            The font size used for the title and axis labels.
        file_path : str, optional (default=None)
            If provided, the resulting plot will be saved to this file.

        Returns
        -------
        axes : matplotlib.Axes
            The subplots of the resulting figure (the sliding window and the ClaSP plot).
        """
        self._check_is_warmed_up()
        fig, axes = plt.subplots(2, sharex=True, gridspec_kw={"hspace": .05}, figsize=fig_size)

        ts_ax, profile_ax = axes

        ts_ax.plot(np.arange(self.knn_stream.time_series.shape[0]), self.knn_stream.time_series)
        profile_ax.plot(np.arange(self.profile.shape[0]), self.profile, color="black")

        if heading is not None:
            ts_ax.set_title(heading, fontsize=font_size)

        if stream_name is not None:
            ts_ax.set_ylabel(stream_name, fontsize=font_size)

        profile_ax.set_xlabel("split point", fontsize=font_size)
        profile_ax.set_ylabel("ClaSP Score", fontsize=font_size)

        for ax in axes:
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(font_size)

            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(font_size)

        if self.last_cp != 0:
            profile_ax.axvline(x=self.last_cp, linewidth=2, color="g", label="Predicted Change Point")

        profile_ax.legend(prop={"size": font_size})

        if file_path is not None:
            plt.savefig(file_path, bbox_inches="tight")

        return axes
