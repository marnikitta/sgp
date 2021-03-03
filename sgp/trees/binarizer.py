import numpy as np
from sklearn.utils import check_array


class Binarizer:
    def __init__(self, n_bins: int = 64):
        self.n_bins = n_bins
        self.boundaries = None

    def fit(self, X: np.ndarray) -> 'Binarizer':
        X = check_array(X)
        boundaries = np.zeros((X.shape[1], self.n_bins), np.float64)
        for f_index in np.arange(X.shape[1]):
            # noinspection PyTypeChecker
            bins = np.percentile(X[:, f_index],
                                 np.linspace(0, 100, num=self.n_bins, endpoint=True),
                                 interpolation='lower')
            boundaries[f_index] = bins

        self.boundaries = boundaries
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.boundaries is not None
        X = check_array(X)
        assert X.shape[1] == self.boundaries.shape[0]
        df_bins = np.zeros(X.T.shape, np.int64)
        for f_index in np.arange(df_bins.shape[0]):
            df_bins[f_index] = np.digitize(X[:, f_index], self.boundaries[f_index]).astype(np.int64) - 1

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
