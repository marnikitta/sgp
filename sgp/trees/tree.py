# Inspired by [jmll library](https://github.com/spbsu-ml-community/jmll)

from typing import Tuple, List, Optional

import numpy as np
from sklearn.utils import assert_all_finite

from .binarizer import Binarizer
from .loss import KLUpliftLoss, MSEUpliftLoss, MSELoss, AdditiveLoss


class DecisionTree:
    def __init__(self, max_depth: int = 3,
                 binarizer: Optional[Binarizer] = None,
                 n_bins: int = 32,
                 verbose: bool = False):
        self.max_depth = max_depth
        self.verbose = verbose
        self.binarizer = binarizer
        self.n_bins = n_bins

    def fit_uplift(self, X: np.ndarray,
                   y: np.ndarray,
                   t: np.ndarray,
                   w: Optional[np.ndarray] = None,
                   min_samples_leaf: int = 2000,
                   min_treatment_leaf: int = 1000,
                   loss='mse',
                   prior_factor: int = 500,
                   **kwargs) -> 'DecisionTreeModel':
        if loss == 'kl':
            loss = KLUpliftLoss(min_samples_leaf=min_samples_leaf, min_treatment_leaf=min_treatment_leaf,
                                prior_factor=prior_factor)
        elif loss == 'mse':
            loss = MSEUpliftLoss(min_samples_leaf=min_samples_leaf, min_treatment_leaf=min_treatment_leaf,
                                 prior_factor=prior_factor)
        else:
            raise BaseException(f'Unknown loss type {loss}. Use "kl" or "mse"')

        point_stats = KLUpliftLoss.point_stats(y, t)
        if w is not None:
            point_stats = point_stats * w

        return self.fit(X, point_stats, loss, **kwargs)

    def fit_regression(self, X: np.ndarray,
                       y: np.ndarray,
                       w: Optional[np.ndarray] = None,
                       min_samples_leaf: int = 1000,
                       **kwargs) -> 'DecisionTreeModel':
        loss = MSELoss(min_samples_leaf=min_samples_leaf)

        point_stats = MSELoss.point_stats(y)
        if w is not None:
            point_stats = point_stats * w

        return self.fit(X, point_stats, loss, **kwargs)

    @staticmethod
    def check_input(X: np.ndarray,
                    binarizer: Optional[Binarizer],
                    n_bins: int,
                    binarize: bool) -> Tuple[np.ndarray, Binarizer]:
        df_bins = X

        assert (binarize and binarizer is None) or (not binarize and binarizer is not None)
        binarizer = binarizer
        if binarizer is None:
            binarizer = Binarizer(n_bins)
            df_bins = binarizer.fit_transform(X)
        elif binarize:
            binarizer = binarizer
            df_bins = binarizer.transform(X)

        assert binarizer.is_fitted()
        assert df_bins.dtype == np.int64
        assert df_bins.shape[0] < df_bins.shape[1]

        return df_bins, binarizer

    def fit(self, X: np.ndarray,
            point_stats: np.ndarray,
            loss: AdditiveLoss,
            binarize: bool = True) -> 'DecisionTreeModel':
        df_bins, binarizer = self.check_input(X, self.binarizer, self.n_bins, binarize)
        assert_all_finite(X)
        assert_all_finite(point_stats)

        items_leafs = np.zeros(df_bins.shape[1], dtype=np.int64)

        nodes: List[Optional[TreeNode]] = [None] * (2 ** (self.max_depth + 1))
        nodes[0] = TreeNode(point_stats.sum(axis=1))

        for depth in np.arange(self.max_depth):
            stats = self.eval_stats(df_bins, items_leafs, point_stats, depth, binarizer.n_bins)

            for node_id in self.leafs_ids_range(depth):
                current_node = nodes[node_id]

                if current_node is None:
                    continue

                left_stats, right_stats = self.split_stats(stats, node_id, binarizer.n_bins)
                assert left_stats.shape[2] == binarizer.n_bins + 1
                assert right_stats.shape[2] == binarizer.n_bins + 1
                assert np.allclose((left_stats[:, 0, 0] + right_stats[:, 0, 0]), current_node.stats)

                split_scores = loss.score(left_stats) + loss.score(right_stats)
                valid_splits = loss.valid_stats(left_stats) & loss.valid_stats(right_stats)

                if valid_splits.sum() == 0:
                    if self.verbose:
                        print(f'{node_id}: no valid splits')
                    continue

                valid_scores = np.ma.masked_array(split_scores, mask=~valid_splits)
                best_score = valid_scores.min()

                best_f_index, best_bin = np.argwhere(valid_scores == best_score)[0]

                current_node.set_split(best_f_index, best_bin, binarizer.boundaries[best_f_index][best_bin])

                left_node_id = 2 * node_id + 1
                right_node_id = 2 * node_id + 2

                nodes[left_node_id] = TreeNode(left_stats[:, best_f_index, best_bin])
                nodes[right_node_id] = TreeNode(right_stats[:, best_f_index, best_bin])

                current_leaf_items = items_leafs == node_id
                left_flags, right_flags = current_node.process_items(df_bins)
                items_leafs[left_flags & current_leaf_items] = left_node_id
                items_leafs[right_flags & current_leaf_items] = right_node_id

                if self.verbose:
                    print(current_node.pretty_str(loss))

        return DecisionTreeModel(nodes, self.max_depth, binarizer, loss)

    @staticmethod
    def leafs_ids_range(depth: int) -> np.ndarray:
        return np.arange(2 ** depth - 1, 2 ** (depth + 1) - 1)

    @staticmethod
    def eval_stats(df_bins: np.ndarray,
                   items_leafs: np.ndarray,
                   point_stats: np.ndarray,
                   depth: int,
                   n_bins: int) -> np.ndarray:
        layer_n_bins = (2 ** depth + (2 ** depth - 1)) * n_bins
        stats_shape = (point_stats.shape[0], df_bins.shape[0], layer_n_bins)
        result = np.zeros(stats_shape, np.float64)

        bins_offsets = items_leafs * n_bins
        leafs_bins = np.zeros(df_bins.shape[1], dtype=np.int64)

        for f_index in np.arange(result.shape[1]):
            leafs_bins = np.add(df_bins[f_index], bins_offsets, out=leafs_bins)

            for s_index in np.arange(result.shape[0]):
                b_count = np.bincount(leafs_bins, weights=point_stats[s_index], minlength=layer_n_bins)
                result[s_index, f_index] = b_count

        return result

    @staticmethod
    def split_stats(stats: np.ndarray, node_id: int, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        stats = stats[:, :, node_id * n_bins: (node_id + 1) * n_bins]

        left_stats = np.cumsum(stats, axis=2)
        left_stats = np.insert(left_stats, 0, 0, axis=2)
        right_stats = left_stats[:, :, [-1]] - left_stats
        return left_stats, right_stats


class DecisionTreeModel:
    def __init__(self,
                 nodes: List['TreeNode'],
                 depth: int,
                 binarizer: Binarizer,
                 loss: AdditiveLoss):
        self.binarizer = binarizer
        self.nodes = nodes
        self.depth = depth
        self.loss = loss

    def bins_importances(self) -> np.ndarray:
        result = np.zeros(self.binarizer.boundaries.shape)

        for n in self.nodes:
            if n is None or n.is_terminal():
                continue
            result[n.f_index, n.bin_id] += 1

        return result

    def predict(self, X: np.ndarray, point_stats: Optional[np.ndarray] = None, binarize=True):
        if binarize:
            df_bins = self.binarizer.transform(X)
        else:
            df_bins = X

        assert df_bins.dtype == np.int64
        assert df_bins.shape[0] < df_bins.shape[1]

        items_leafs = np.zeros(df_bins.shape[1], np.int64)

        for depth in np.arange(self.depth):
            for node_id in DecisionTree.leafs_ids_range(depth):
                node = self.nodes[node_id]
                if node is None:
                    continue

                if point_stats is not None:
                    node.stats = point_stats[:, items_leafs == node_id].sum(axis=1)

                if not node.is_terminal():
                    current_leaf_items = items_leafs == node_id
                    left_flags, right_flags = node.process_items(df_bins)
                    items_leafs[left_flags & current_leaf_items] = 2 * node_id + 1
                    items_leafs[right_flags & current_leaf_items] = 2 * node_id + 2

        leaf_means = np.repeat(np.nan, len(self.nodes))

        for i, n in enumerate(self.nodes):
            if n is not None:
                leaf_means[i] = self.loss.leaf_predicts(n.stats)

        return leaf_means[items_leafs]

    def pretty_str(self, feature_names: Optional[List[str]] = None) -> str:
        assert self.nodes[0] is not None

        def append(node_id: int, depth):
            node = self.nodes[node_id]

            result = ('    ' * depth + node.pretty_str(self.loss, feature_names))
            if node.f_index is None:
                return result

            result += ('\n' + append(2 * node_id + 1, depth + 1))
            result += ('\n' + append(2 * node_id + 2, depth + 1))
            return result

        return append(0, 0)


class TreeNode:
    def __init__(self, stats: np.ndarray):
        self.stats = stats
        self.f_index: Optional[int] = None
        self.bin_id: Optional[int] = None
        self.f_value: Optional[float] = None

    def pretty_str(self, loss: 'AdditiveLoss', feature_names: Optional[List[str]] = None) -> str:
        condition = ''
        if not self.is_terminal():
            feature_name = f'f{self.f_index}'
            if feature_names is not None:
                assert self.f_index is not None
                feature_name = feature_names[self.f_index]
            condition = f'{feature_name}>={self.f_value} '
        return f'{condition}[{loss.pretty_str(self.stats)}]'

    def set_split(self, f_index: int, bin_id: int, f_value: float):
        self.f_index = f_index
        self.bin_id = bin_id
        self.f_value = f_value

    def is_terminal(self) -> bool:
        return self.f_index is None

    def process_items(self, X: np.ndarray):
        assert (self.f_index is not None)
        assert (self.bin_id is not None)

        right_flags = X[self.f_index] >= self.bin_id
        left_flags = ~right_flags

        return left_flags, right_flags
