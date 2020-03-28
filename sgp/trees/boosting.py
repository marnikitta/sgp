from typing import Optional, List, Callable

import numpy as np
import scipy.stats

from .binarizer import Binarizer
from .loss import MSELoss
from .tree import DecisionTree, DecisionTreeModel


class Boosting:
    def __init__(self,
                 iterations: int = 100,
                 shrinkage: float = 0.1,
                 binarizer: Optional[Binarizer] = None,
                 n_bins: int = 32,
                 random_state: int = 42,
                 min_samples_leaf: int = 50,
                 **kwargs):
        self.iterations = iterations
        self.shrinkage = shrinkage
        self.random_state = random_state
        self.binarizer = binarizer
        self.n_bins = n_bins
        self.tree_params = kwargs
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: np.ndarray,
            loss: 'DifferentiableLoss',
            binarize: bool = True,
            listeners: Optional[List[Callable[[List[DecisionTreeModel]], bool]]] = None) -> List[DecisionTreeModel]:
        df_bins, binarizer = DecisionTree.check_input(X, self.binarizer, self.n_bins, binarize)

        if listeners is None:
            listeners = []
        assert listeners is not None

        current_predict = np.zeros(df_bins.shape[1])
        boosts: List[DecisionTreeModel] = []
        for i in np.arange(self.iterations):
            gradient = -loss.gradient(current_predict) * self.shrinkage

            boot_w = scipy.stats.poisson(1).rvs(df_bins.shape[1], random_state=self.random_state + i)
            m = DecisionTree(binarizer=binarizer, **self.tree_params)
            boost = m.fit(df_bins,
                          MSELoss().point_stats(gradient) * boot_w,
                          MSELoss(self.min_samples_leaf),
                          binarize=False)
            boosts.append(boost)
            current_predict += boost.predict(df_bins, binarize=False)

            ok = True
            for l in listeners:
                ok &= l(boosts)

            if not ok:
                break
        return boosts


class DifferentiableLoss:
    def gradient(self, f: np.ndarray) -> np.ndarray:
        pass


class L2Loss(DifferentiableLoss):
    def __init__(self, y: np.ndarray):
        self.y = y

    def gradient(self, f: np.ndarray) -> np.ndarray:
        return -2 * (self.y - f)
