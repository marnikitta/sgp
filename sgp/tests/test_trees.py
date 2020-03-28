import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from sgp.trees.tree import DecisionTree
from sgp.trees.forest import RandomForest
from sgp.trees.loss import MSELoss


def test_boston_fit_single_tree():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
    tree = DecisionTree(max_depth=6, n_bins=32) \
        .fit_regression(X_train, y_train, min_samples_leaf=50)

    score = r2_score(y_test, tree.predict(X_test))
    assert score > 0.6


def test_boston_fit_forest():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

    forest = RandomForest(max_depth=6, n_trees=500, n_bins=16, feature_samples=0.7, random_state=31)
    trees = forest.fit(X_train, MSELoss().point_stats(y_train), MSELoss(min_samples_leaf=10))

    predicts = np.mean(np.vstack([t.predict(X_test) for t in trees]), axis=0)
    score = r2_score(y_test, predicts)
    assert score > 0.8


test_boston_fit_single_tree()
test_boston_fit_forest()
