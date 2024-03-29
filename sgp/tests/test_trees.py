from typing import List

import numpy as np
from scipy.stats import kendalltau
from sklearn.datasets import load_boston, make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sgp.trees.boosting import Boosting, L2Loss, PairwiseLL
from sgp.trees.forest import RandomForest
from sgp.trees.loss import MSELoss
from sgp.trees.tree import DecisionTree, DecisionTreeModel


def benchmark_tree(n: int = 100000, f_count: int = 500, max_depth: int = 6, n_bins: int = 64):
    X, y = make_regression(n, f_count, 50)
    DecisionTree(max_depth=max_depth, n_bins=n_bins).fit(X, MSELoss.point_stats(y), MSELoss(min_samples_leaf=1))


def test_boston_fit_single_tree():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
    tree = DecisionTree(max_depth=6, n_bins=64).fit(X_train, MSELoss.point_stats(y_train), MSELoss(min_samples_leaf=1))

    score = r2_score(y_test, tree.predict(X_test))
    corr = kendalltau(y_test, tree.predict(X_test)).correlation
    print(score, corr)
    # print(tree.pretty_str(load_boston().feature_names))
    assert score > 0.6


def test_boston_fit_forest():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

    forest = RandomForest(max_depth=6, n_trees=500, n_bins=16, random_state=31, n_jobs=5, verbose=False)
    trees = forest.fit(X_train, MSELoss().point_stats(y_train), MSELoss(min_samples_leaf=1))

    predicts = trees.predict(X_test)
    score = r2_score(y_test, predicts)
    corr = kendalltau(y_test, predicts).correlation
    print(score, corr)
    assert score > 0.8


def test_boston_boosting():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

    boosting = Boosting(max_depth=3, shrinkage=0.01, iterations=2000)

    y_pred = np.zeros(y_test.shape[0])

    def validation_listener(trees: List[DecisionTreeModel]) -> bool:
        new_score = trees[-1].predict(X_test)
        np.add(y_pred, new_score, out=y_pred)
        # print(f'Iter {len(trees)}: {r2_score(y_test, y_pred)}')
        return True

    boosts = boosting.fit(X_train, L2Loss(y_train), listeners=[validation_listener])
    predicts = np.sum(np.vstack([t.predict(X_test) for t in boosts]), axis=0)
    score = r2_score(y_test, predicts)
    corr = kendalltau(y_test, predicts).correlation
    print(score, corr)
    assert score > 0.85


def train_boston_pairwise():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    boosts = Boosting(shrinkage=0.01, iterations=100) \
        .fit(X_train, PairwiseLL(y_train, np.zeros(y_train.shape)))

    predicts = np.sum(np.vstack([t.predict(X_test) for t in boosts]), axis=0)
    corr = kendalltau(y_test, predicts).correlation
    print(corr)
    assert corr > 0.7

# benchmark_tree()
# test_boston_fit_single_tree()
# test_boston_fit_forest()
# test_boston_boosting()
# train_boston_pairwise()
