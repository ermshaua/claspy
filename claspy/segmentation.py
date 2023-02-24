from queue import PriorityQueue

import matplotlib
import numpy as np
import pandas as pd

from claspy.clasp import ClaSPEnsemble
from claspy.utils import check_input_time_series

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import seaborn as sns

sns.set_theme()
sns.set_color_codes()

import matplotlib.pyplot as plt


class BinaryClaSPSegmentation:
    '''
    Segment time series data using the binary segmentation algorithm with ClaSP.

    Parameters
    ----------
    n_segments : int, default=1
        The number of segments to split the time series into.

    n_estimators : int, default=10
        The number of ClaSPs in the ensemble.

    window_size : int, default=10
        The size of the sliding window used in the ClaSP algorithm.

    k_neighbours : int, default=3
        The number of nearest neighbors to use in the ClaSP algorithm.

    score : str, default="roc_auc"
        The name of the scoring metric to use in ClaSP.

    excl_radius : int, default=5*window_size
        The radius (in multiples of the window size) around each point in the time series to exclude
        when searching for nearest neighbors.

    random_state : int, default=2357
        Seed for the random number generator. Default is 2357.

    Methods
    -------
    fit(time_series)
        Fit the BinaryClaSPSegmentation model to the input time series.
    '''
    def __init__(self, n_segments=1, n_estimators=10, window_size=10, k_neighbours=3, score="roc_auc", excl_radius=5,
                 random_state=2357):
        self.n_segments = n_segments
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.score_name = score
        self.score = score
        self.excl_radius = excl_radius
        self.random_state = random_state

    def _cp_is_valid(self, candidate, change_points):
        '''
        Determine whether a candidate change point is valid given a list of existing change points.

        Parameters
        ----------
        candidate : int
            The index of the candidate change point.
        change_points : list of int
            The indices of existing change points.

        Returns
        -------
        bool
            True if the candidate change point is valid, False otherwise.
        '''
        for change_point in [0] + change_points + [self.n_timepoints]:
            left_begin = max(0, change_point - self.min_seg_size)
            right_end = min(self.n_timepoints, change_point + self.min_seg_size)
            if candidate in range(left_begin, right_end): return False

        return True

    def _local_segmentation(self, lbound, ubound, change_points):
        '''
        Perform local segmentation of the time series within the range [lbound, ubound) using ClaSP algorithm.

        Parameters:
        -----------
        lbound : int
            The left bound of the time series range.
        ubound : int
            The right bound of the time series range.
        change_points : list of int
            A list of current change points in the time series.

        Returns:
        --------
        None
        '''
        if ubound - lbound < 2 * self.min_seg_size: return

        clasp = ClaSPEnsemble(
            n_estimators=self.n_estimators,
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
            score=self.score,
            excl_radius=self.excl_radius,
            random_state=self.random_state
        ).fit(self.time_series[lbound:ubound])

        cp = clasp.split()
        score = clasp.profile[cp]

        if not self._cp_is_valid(lbound + cp, change_points): return

        self.clasp_tree.append(((lbound, ubound), clasp))
        self.queue.put((-score, len(self.clasp_tree) - 1))

    def fit(self, time_series):
        """
        Segments the input time series.

        Parameters
        ----------
        time_series : array-like of shape (n_samples,)
            The input time series.

        Returns
        -------
        self : object
            Returns an instance of self.

        Raises
        ------
        ValueError
            If the input time series has less than 2 times the minimum segment size.
        """
        check_input_time_series(time_series)
        self.min_seg_size = self.window_size * self.excl_radius

        if time_series.shape[0] < 2 * self.min_seg_size:
            raise ValueError("Time series must at least have 2*min_seg_size data points.")

        self.time_series = time_series
        self.n_timepoints = time_series.shape[0]

        self.queue = PriorityQueue()
        self.clasp_tree = []

        prange = 0, time_series.shape[0]
        clasp = ClaSPEnsemble(
            n_estimators=self.n_estimators,
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
            score=self.score,
            excl_radius=self.excl_radius,
            random_state=self.random_state
        ).fit(time_series)

        self.clasp_tree.append((prange, clasp))
        self.queue.put((-clasp.profile[clasp.split()], len(self.clasp_tree) - 1))

        profile = clasp.profile
        change_points = []
        scores = []

        for idx in range(self.n_segments - 1):
            priority, clasp_tree_idx = self.queue.get()
            (lbound, ubound), clasp = self.clasp_tree[clasp_tree_idx]
            cp = lbound + clasp.split()

            profile[lbound:ubound - self.window_size + 1] = np.max(
                [profile[lbound:ubound - self.window_size + 1], clasp.profile], axis=0)

            change_points.append(cp)
            scores.append(-priority)

            if len(change_points) == self.n_segments - 1: break

            lrange, rrange = (lbound, cp), (cp, ubound)

            for prange in (lrange, rrange):
                self._local_segmentation(*prange, change_points)

        sorted_cp_args = np.argsort(change_points)
        self.change_points, self.scores = np.asarray(change_points)[sorted_cp_args], np.asarray(scores)[sorted_cp_args]

        profile[np.isinf(profile)] = np.nan
        self.profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()

        return self

    def predict(self, sparse=True):
        '''
        Predicts the location of the change points in the time series.

        Parameters
        ----------
        sparse : bool, optional (default=True)
            Whether to return a sparse or a dense output.

        Returns
        -------
        If `sparse` is True, returns an array containing the indices of the change points.
        Otherwise, returns a list of numpy arrays, where each array corresponds to a segment in the imput time series.
        '''
        if sparse is True:
            return self.change_points

        seg_idx = np.concatenate(([0], self.change_points, [self.time_series.shape[0]]))
        return [self.time_series[seg_idx[idx]:seg_idx[idx+1]] for idx in range(len(seg_idx)-1)]

    def fit_predict(self, time_series, sparse=True):
        '''
        Fit the ClaSP model to the provided time series and predict change points.

        Parameters:
        -----------
        time_series : numpy.ndarray
            A one-dimensional numpy array containing the time series.
        sparse : bool, optional (default=True)
            Whether to return the change point indices only, or to return the segmented time series.

        Returns:
        --------
        If `sparse` is True, returns an array containing the indices of the change points.
        Otherwise, returns a list of numpy arrays, where each array corresponds to a segment in the imput time series.
        '''
        return self.fit(time_series).predict(sparse)

    def plot(self, gt_cps=None, heading=None, ts_name=None, fig_size=(20, 10), font_size=26, file_path=None):
        '''
        Plot the fitted time series annotated with ClaSP and found change points.

        Parameters
        ----------
        gt_cps : array-like of shape (n_true_cps,), optional (default=None)
            Array of true change points. If provided, vertical lines will be plotted on the graphs to indicate
            the true change points.
        heading : str, optional (default=None)
            The title of the first subplot (i.e., the time series plot).
        ts_name : str, optional (default=None)
            The name of the time series (used as y-label for the time series subplot).
        fig_size : tuple of int, optional (default=(20, 10))
            The size of the resulting figure.
        font_size : int, optional (default=26)
            The font size used for the title and axis labels.
        file_path : str, optional (default=None)
            If provided, the resulting plot will be saved to this file.

        Returns
        -------
        ax1, ax2 : matplotlib.Axes
            The two subplots of the resulting figure (the time series and the ClaSP plot).
        '''
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': .05}, figsize=fig_size)

        if gt_cps is not None:
            segments = [0] + gt_cps.tolist() + [self.time_series.shape[0]]
            for idx in np.arange(0, len(segments) - 1):
                ax1.plot(np.arange(segments[idx], segments[idx + 1]), self.time_series[segments[idx]:segments[idx + 1]])

            ax2.plot(np.arange(self.profile.shape[0]), self.profile, color='b')
        else:
            ax1.plot(np.arange(self.time_series.shape[0]), self.time_series)
            ax2.plot(np.arange(self.profile.shape[0]), self.profile)

        if heading is not None:
            ax1.set_title(heading, fontsize=font_size)

        if ts_name is not None:
            ax2.set_ylabel(heading, fontsize=font_size)

        ax2.set_xlabel('split point', fontsize=font_size)
        ax2.set_ylabel("ClaSP Score", fontsize=font_size)

        for ax in (ax1, ax2):
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(font_size)

            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(font_size)

        if gt_cps is not None:
            for idx, true_cp in enumerate(gt_cps):
                ax1.axvline(x=true_cp, linewidth=2, color='r', label=f'True Change Point' if idx == 0 else None)
                ax2.axvline(x=true_cp, linewidth=2, color='r', label='True Change Point' if idx == 0 else None)

        for idx, found_cp in enumerate(self.change_points):
            ax1.axvline(x=found_cp, linewidth=2, color='g', label='Predicted Change Point' if idx == 0 else None)
            ax2.axvline(x=found_cp, linewidth=2, color='g', label='Predicted Change Point' if idx == 0 else None)

        ax1.legend(prop={'size': font_size})

        if file_path is not None:
            plt.savefig(file_path, bbox_inches="tight")

        return ax1, ax2
