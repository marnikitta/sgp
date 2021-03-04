import numpy as np
from sklearn.datasets import load_boston

from sgp.trees.binarizer import Binarizer


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


# test_binarizer()
# test_binary_arrays()
# test_ternary_arrays()
