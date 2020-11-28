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
