import numpy as np
import scipy.stats
from sklearn.datasets import load_boston

from sgp.trees.binarizer import Binarizer, binarization_plot


def test_binarizer():
    X, _ = load_boston(return_X_y=True)
    binarizer = Binarizer(n_bins=64)
    df_bins = binarizer.fit_transform(X)

    assert df_bins.T.shape == X.shape
    assert set(df_bins.ravel()) == set(np.arange(0, 64))

    assert df_bins.dtype == np.int64
    assert df_bins.shape[0] < df_bins.shape[1]


def test_binary_arrays():
    x = [0] * 100000 + [1]
    X = np.vstack((x, x)).T
    binarizer = Binarizer(n_bins=64)
    df_bins = binarizer.fit_transform(X)
    assert np.all(df_bins[0] == df_bins[1])
    assert len(set(df_bins.ravel())) == 2


def test_ternary_arrays():
    x = [-1] + [0] * 100000 + [1]
    X = np.vstack((x, x)).T
    df_bins = Binarizer(n_bins=64).fit_transform(X)
    assert len(set(df_bins.ravel())) == 3


def test_plot():
    x = np.asarray([-1] * 10 + [0] * 100000 + [1] * 10)
    tr = scipy.stats.bernoulli(0.5).rvs(len(x))
    y = scipy.stats.norm(0, 1).rvs(len(x))
    x_plot, y_plot, std_plot = binarization_plot(x, tr, y)
    assert len(x_plot) == 3
    assert len(y_plot) == 3
    assert len(std_plot) == 3


# test_binarizer()
# test_binary_arrays()
# test_ternary_arrays()
# test_plot()
