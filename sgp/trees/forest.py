from typing import List, Optional

import numpy as np
import scipy.stats
from joblib import Parallel, delayed
from sklearn.utils.random import sample_without_replacement

from .binarizer import Binarizer
from .tree import DecisionTreeModel, DecisionTree
from .loss import AdditiveLoss


class RandomForest:
    def __init__(self,
                 n_trees=100,
                 feature_samples: float = 0.7,
                 binarizer: Optional[Binarizer] = None,
                 n_bins: int = 32,
                 random_state: int = 42,
                 n_jobs: int = 1,
                 oob_stats: bool = False,
                 **kwargs):
        self.feature_samples = feature_samples
        self.n_trees = n_trees
        self.random_state = random_state
        self.binarizer = binarizer
        self.n_bins = n_bins
        self.n_jobs = n_jobs
        self.oob_stats = oob_stats
        self.tree_params = kwargs

    def fit(self, X: np.ndarray,
            point_stats: np.ndarray,
            loss: AdditiveLoss,
            binarize: bool = True) -> List[DecisionTreeModel]:
        df_bins, binarizer = DecisionTree.check_input(X, self.binarizer, self.n_bins, binarize)

        def train_ith_tree(i: int) -> DecisionTreeModel:
            boot_features = sample_without_replacement(df_bins.shape[0], int(df_bins.shape[0] * self.feature_samples),
                                                       random_state=self.random_state + i)
            boot_w = scipy.stats.poisson(1).rvs(df_bins.shape[1], random_state=self.random_state + i)
            m = DecisionTree(binarizer=binarizer.sample_columns(boot_features), **self.tree_params)

            df_bins_boot = df_bins[boot_features]
            tree = m.fit(df_bins_boot, point_stats * boot_w, loss, binarize=False)

            if self.oob_stats:
                w_oob = (boot_w == 0).astype(np.int64)
                tree.predict(df_bins_boot, point_stats * w_oob, binarize=False)

            tree.binarizer = binarizer
            for n in tree.nodes:
                if n is None or n.is_terminal():
                    continue
                n.f_index = boot_features[n.f_index]

            print(f'{i}-th tree is done')
            return tree

        return Parallel(n_jobs=self.n_jobs)(delayed(train_ith_tree)(i) for i in range(self.n_trees))

    def fit_uplift(self, X: np.ndarray,
                   y: np.ndarray,
                   t: np.ndarray,
                   w: Optional[np.ndarray] = None,
                   binarize: bool = True,
                   **kwargs) -> List[DecisionTreeModel]:
        pass
