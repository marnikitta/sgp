from typing import List

import numpy as np
from scipy.stats import kendalltau
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sgp.trees.boosting import Boosting, L2Loss
from sgp.trees.forest import RandomForest
from sgp.trees.loss import MSELoss
from sgp.trees.tree import DecisionTree, DecisionTreeModel


def test_boston_fit_single_tree():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
    tree = DecisionTree(max_depth=6, n_bins=32) \
        .fit_regression(X_train, y_train, min_samples_leaf=50)

    score = r2_score(y_test, tree.predict(X_test))
    corr = kendalltau(y_test, tree.predict(X_test)).correlation
    print(score, corr)
    assert score > 0.6


def test_boston_fit_forest():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

    forest = RandomForest(max_depth=6, n_trees=500, n_bins=16, random_state=31, n_jobs=5, verbose=False)
    trees = forest.fit(X_train, MSELoss().point_stats(y_train), MSELoss(min_samples_leaf=10))

    predicts = np.mean(np.vstack([t.predict(X_test) for t in trees]), axis=0)
    score = r2_score(y_test, predicts)
    corr = kendalltau(y_test, predicts).correlation
    print(score, corr)
    assert score > 0.8


def test_boston_boosting():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

    boosting = Boosting(max_depth=3, shrinkage=0.1, iterations=200)

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
    pass


test_boston_fit_single_tree()
test_boston_fit_forest()
test_boston_boosting()
train_boston_pairwise()
