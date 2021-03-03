from typing import Tuple

import numpy as np
import scipy.stats

from sgp.trees.loss import MSEUpliftLoss, uplift_at_percents, SigUpliftLoss
from sgp.trees.tree import DecisionTree


def make_dataset(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = scipy.stats.norm(0, 1).rvs((n, 5))
    tr = scipy.stats.bernoulli(0.5).rvs(n)
    y = np.maximum(np.maximum(X[:, 0] + X[:, 1], X[:, 2]), 0) \
        + np.maximum(X[:, 3] + X[:, 4], 0) \
        + (tr - 0.5) * (X[:, 0] + np.log1p(1 + np.exp(X[:, 1])))
    return X, tr, y


def test_uplift():
    X, tr, y = make_dataset(100000)

    scores = []
    sig_scores = []
    for d in range(6):
        tree = DecisionTree(max_depth=d + 2, n_bins=64, verbose=False)

        y_pred = tree.fit(X, MSEUpliftLoss.point_stats(y, tr), MSEUpliftLoss()).predict(X)
        scores.append(uplift_at_percents(y_pred, y, tr, percents=10))

        y_pred_sig = tree.fit(X, SigUpliftLoss.point_stats(y, tr), SigUpliftLoss()).predict(X)
        sig_scores.append(uplift_at_percents(y_pred_sig, y, tr, percents=10))

    scores = np.asarray(scores)
    sig_scores = np.asarray(sig_scores)
    print(scores)
    print(sig_scores)

    assert np.all(np.diff(scores) >= 0)
    assert np.all(np.diff(sig_scores) >= 0)

    assert np.all(sig_scores > scores)

    assert (np.min(scores > 1.4))
    assert (np.max(scores > 2.4))

    assert (np.min(sig_scores > 2.2))
    assert (np.max(sig_scores > 2.5))

# test_uplift()
