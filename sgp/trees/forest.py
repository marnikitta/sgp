from typing import List, Optional, Tuple

import numpy as np
import scipy.stats
from joblib import Parallel, delayed

from .binarizer import Binarizer
from .loss import AdditiveLoss
from .tree import DecisionTreeModel, DecisionTree


class RandomForest:
    def __init__(self,
                 n_trees=100,
                 binarizer: Optional[Binarizer] = None,
                 n_bins: int = 32,
                 random_state: int = 42,
                 n_jobs: int = 1,
                 oob_stats: bool = True,
                 verbose: bool = True,
                 **kwargs):
        self.n_trees = n_trees
        self.random_state = random_state
        self.binarizer = binarizer
        self.n_bins = n_bins
        self.n_jobs = n_jobs
        self.oob_stats = oob_stats
        self.verbose = verbose
        self.tree_params = kwargs

    def fit(self, X: np.ndarray,
            point_stats: np.ndarray,
            loss: AdditiveLoss,
            binarize: bool = True) -> 'RandomForestModel':
        df_bins, binarizer = DecisionTree.check_input(X, self.binarizer, self.n_bins, binarize)

        m = DecisionTree(binarizer=binarizer, **self.tree_params)

        def train_ith_tree(i: int) -> DecisionTreeModel:
            boot_w = scipy.stats.poisson(1).rvs(df_bins.shape[1], random_state=self.random_state + i)

            tree = m.fit(df_bins, point_stats * boot_w, loss, binarize=False)

            if self.verbose:
                print(f'{i}-th tree is done')
            return tree

        trees = Parallel(n_jobs=self.n_jobs)(delayed(train_ith_tree)(i) for i in range(self.n_trees))
        return RandomForestModel(trees)


class RandomForestModel:
    def __init__(self, trees: List[DecisionTreeModel]):
        self.trees = trees

    def predict(self, X: np.ndarray, binarize: bool = True):
        return np.mean(np.vstack([t.predict(X, binarize=binarize) for t in self.trees]), axis=0)

    def importances(self) -> np.ndarray:
        importances = np.zeros(self.binarizer.boundaries.shape)
        for t in self.trees:
            importances += t.importances()
        return importances

    def pdp(self) -> Tuple[np.ndarray, np.ndarray]:
        sums = np.zeros(self.binarizer.boundaries.shape)
        sums_sq = np.zeros(self.binarizer.boundaries.shape)
        for t in self.trees:
            pdps = t.pdp()
            sums += pdps
            sums_sq += pdps ** 2

        mean_pdp = sums / self.n_trees
        std_pdp = np.sqrt(sums_sq / self.n_trees - (sums / self.n_trees ** 2))
        return mean_pdp, std_pdp / np.sqrt(self.n_trees)

    @property
    def binarizer(self) -> Binarizer:
        return self.trees[0].binarizer

    @property
    def n_trees(self) -> int:
        return len(self.trees)
