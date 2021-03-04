from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
from sklearn.utils import check_array


class Binarizer:
    def __init__(self,
                 n_bins: int = 64,
                 top_n: int = 200000,
                 n_jobs: int = 8):
        self.n_bins = n_bins
        self.top_n = top_n
        self.n_jobs = n_jobs
        # self.boundaries = None

    def fit(self, X: np.ndarray) -> 'Binarizer':
        X = check_array(X)

        def f(f_index: int) -> np.ndarray:
            return np.unique(np.percentile(X[:self.top_n, f_index],
                                 np.linspace(0, 100, num=self.n_bins, endpoint=True),
                                 interpolation='lower'))

        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            boundaries = pool.map(f, range(X.shape[1]))
            self.boundaries = list(boundaries)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.boundaries is not None
        X = check_array(X)
        assert X.shape[1] == len(self.boundaries)

        def f(f_index: int) -> np.ndarray:
            return np.digitize(X[:, f_index], self.boundaries[f_index]).astype(np.int64) - 1

        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            df_bins = pool.map(f, range(X.shape[1]))
            df_bins = np.vstack(list(df_bins))

        np.clip(df_bins, 0, self.n_bins - 1, out=df_bins)

        return df_bins

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def sample_columns(self, columns: np.ndarray) -> 'Binarizer':
        assert self.boundaries is not None
        result = Binarizer(self.n_bins)
        result.boundaries = self.boundaries[columns]
        return result


def binarization_plot(x, tr, y, n_bins=32):
    bins_transformer = Binarizer(n_bins=n_bins).fit(x.reshape(-1, 1))
    bins = bins_transformer.transform(x.reshape(-1, 1)).ravel()
    boundaries = bins_transformer.boundaries[0]

    tr_flags = tr.astype(np.bool)

    sums_0 = np.bincount(bins[~tr_flags], y[~tr_flags], minlength=n_bins)
    counts_0 = np.bincount(bins[~tr_flags], minlength=n_bins)
    sums_sq_0 = np.bincount(bins[~tr_flags], y[~tr_flags] ** 2, minlength=n_bins)

    sums_1 = np.bincount(bins[tr_flags], y[tr_flags], minlength=n_bins)
    counts_1 = np.bincount(bins[tr_flags], minlength=n_bins)
    sums_sq_1 = np.bincount(bins[tr_flags], y[tr_flags] ** 2, minlength=n_bins)

    known_bins = (counts_0 > 0) & (counts_1 > 0)

    stds = np.sqrt((sums_sq_0 / counts_0 - (sums_0 / counts_0) ** 2) / counts_0 + (
            sums_sq_1 / counts_1 - (sums_1 / counts_1) ** 2) / counts_1)
    stds = stds[known_bins]

    uplifts = sums_1[known_bins] / counts_1[known_bins] - sums_0[known_bins] / counts_0[known_bins]
    return boundaries[known_bins], uplifts, stds
