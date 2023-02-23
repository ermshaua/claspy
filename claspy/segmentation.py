from queue import PriorityQueue

import matplotlib
import numpy as np
import pandas as pd

from claspy.clasp import ClaSPEnsemble

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import seaborn as sns

sns.set_theme()
sns.set_color_codes()

import matplotlib.pyplot as plt


class BinaryClaSPSegmentation:

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

    def fit(self, time_series):
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

            print(priority)

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

    def _cp_is_valid(self, candidate, change_points):
        for change_point in [0] + change_points + [self.n_timepoints]:
            left_begin = max(0, change_point - self.min_seg_size)
            right_end = min(self.n_timepoints, change_point + self.min_seg_size)
            if candidate in range(left_begin, right_end): return False

        return True

    def _local_segmentation(self, lbound, ubound, change_points):
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

    def plot(self, gt_cps=None, heading=None, ts_name=None, fig_size=(20, 10), font_size=26, file_path=None):
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
