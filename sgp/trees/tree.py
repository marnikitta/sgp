# Inspired by [jmll library](https://github.com/spbsu-ml-community/jmll)

from typing import Tuple, List, Optional

import numpy as np
from sklearn.utils import assert_all_finite

from .binarizer import Binarizer
from .loss import AdditiveLoss


class DecisionTree:
    def __init__(self, max_depth: int = 3,
                 binarizer: Optional[Binarizer] = None,
                 n_bins: int = 32,
                 verbose: bool = False):
        self.max_depth = max_depth
        self.verbose = verbose
        self.binarizer = binarizer
        self.n_bins = n_bins

    def fit(self, X: np.ndarray,
            point_stats: np.ndarray,
            loss: AdditiveLoss,
            binarize: bool = True) -> 'DecisionTreeModel':
        assert_all_finite(X)
        assert_all_finite(point_stats)

        df_bins, binarizer = DecisionTree.check_input(X, self.binarizer, self.n_bins, binarize)
        assert binarizer.boundaries is not None
        assert point_stats.shape[1] == df_bins.shape[1]
        assert point_stats.ndim == 2

        # Initially, all items are in the tree root
        items_leafs = np.zeros(df_bins.shape[1], dtype=np.int64)

        nodes: List[Optional[TreeNode]] = [None] * (2 ** (self.max_depth + 1))
        nodes[0] = TreeNode(point_stats.sum(axis=1))

        for depth in np.arange(self.max_depth):
            stats = DecisionTree.eval_stats(df_bins, items_leafs, point_stats, depth, binarizer.n_bins)

            for node_id in DecisionTree.leafs_ids_range(depth):
                current_node = nodes[node_id]

                if current_node is None:
                    continue

                left_stats, right_stats = DecisionTree.split_stats(stats, node_id, binarizer.n_bins)
                assert left_stats.shape[-1] == binarizer.n_bins + 1
                assert right_stats.shape[-1] == binarizer.n_bins + 1
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

                left_flags, right_flags = current_node.process_items(df_bins, items_leafs == node_id)
                items_leafs[left_flags] = left_node_id
                items_leafs[right_flags] = right_node_id

                if self.verbose:
                    print(current_node.pretty_str(loss))

        return DecisionTreeModel(nodes, self.max_depth, binarizer, loss)

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

        return df_bins, binarizer

    @staticmethod
    def leafs_ids_range(depth: int) -> range:
        result = range(2 ** depth - 1, 2 ** (depth + 1) - 1)
        assert len(result) == 2 ** depth, 'There should be 2^depth leafs in binary tree at level `depth`'
        return result

    @staticmethod
    def eval_stats(df_bins: np.ndarray,
                   items_leafs: np.ndarray,
                   point_stats: np.ndarray,
                   depth: int,
                   n_bins: int) -> np.ndarray:
        # Use a single array for two axes to optimize bincount function call
        leafs_shape = (2 ** depth + (2 ** depth - 1))
        leaf_bin_axis_shape = leafs_shape * n_bins
        result = np.zeros((point_stats.shape[0], df_bins.shape[0], leaf_bin_axis_shape), np.float64)

        # Offsets for the leaf-bin axis
        leaf_bin_axis_offsets = items_leafs * n_bins

        # Reuse for an optimization
        leaf_bin_indexes = np.zeros(df_bins.shape[1], dtype=np.int64)

        for f_index in range(df_bins.shape[0]):
            # Coordinates in leaf-bin axis
            leaf_bin_indexes = np.add(df_bins[f_index], leaf_bin_axis_offsets, out=leaf_bin_indexes)

            for s_index in range(point_stats.shape[0]):
                b_count = np.bincount(leaf_bin_indexes, weights=point_stats[s_index], minlength=leaf_bin_axis_shape)
                result[s_index, f_index] = b_count

        return result.reshape((point_stats.shape[0], df_bins.shape[0], leafs_shape, n_bins))

    @staticmethod
    def split_stats(stats: np.ndarray, node_id: int, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
        # Take a subrange that corresponds to the current node
        node_stats = stats[:, :, node_id]
        assert node_stats.shape[-1] == n_bins, 'Subrange should have exactly n_bins elements. One for each bin'

        left_stats = np.cumsum(node_stats, axis=-1)
        # If we split using the first bin all items will belong to the right child
        left_stats = np.insert(left_stats, 0, 0, axis=-1)
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
        assert self.binarizer.boundaries is not None

    def importances(self) -> np.ndarray:
        result = np.zeros(self.binarizer.boundaries.shape)

        for n in self.nodes:
            if n is None or n.is_terminal():
                continue
            result[n.f_index, n.bin_id] += 1

        return result

    def predict(self, X: np.ndarray, binarize=True):
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
                if node is None or node.is_terminal():
                    continue

                left_flags, right_flags = node.process_items(df_bins, items_leafs == node_id)
                items_leafs[left_flags] = 2 * node_id + 1
                items_leafs[right_flags] = 2 * node_id + 2

        leaf_means = np.repeat(np.nan, len(self.nodes))

        for i, n in enumerate(self.nodes):
            if n is not None:
                leaf_means[i] = self.loss.leaf_predicts(n.stats)

        return leaf_means[items_leafs]

    def pdp(self) -> np.ndarray:
        def g(node_index: int, point_weights: np.ndarray):
            n = self.nodes[node_index]
            assert n is not None

            if self.nodes[node_index].is_terminal():
                return self.loss.leaf_predicts(n.stats) * point_weights

            left_n = self.nodes[2 * node_index + 1]
            right_n = self.nodes[2 * node_index + 2]

            w_left = point_weights.copy()
            w_right = point_weights

            w_left[n.f_index, n.bin_id:] = 0
            w_right[n.f_index, :n.bin_id] = 0

            total_weight = self.loss.node_weight(n.stats)
            left_weight = self.loss.node_weight(left_n.stats)
            right_weight = self.loss.node_weight(right_n.stats)
            assert np.allclose(left_weight + right_weight, total_weight)

            ind = np.delete(np.arange(self.binarizer.boundaries.shape[0]), n.f_index)
            w_left[ind] *= left_weight / total_weight
            w_right[ind] *= right_weight / total_weight

            return g(2 * node_index + 1, w_left) + g(2 * node_index + 2, w_right)

        return g(0, np.ones(self.binarizer.boundaries.shape, dtype=np.float64))

    def pretty_str(self, feature_names: Optional[List[str]] = None) -> str:
        assert self.nodes[0] is not None

        def append(node_id: int, prefix: str, beginning: str):
            node = self.nodes[node_id]

            result = beginning + node.pretty_str(self.loss, feature_names)
            if node.f_index is None:
                return result

            result += ('\n' + append(2 * node_id + 2, prefix + '│   ',  prefix + '├── '))
            result += ('\n' + append(2 * node_id + 1, prefix + '    ',  prefix + '└── '))
            return result

        return append(0, '', '')


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

    def process_items(self, df_bins: np.ndarray, current_node_flags: np.ndarray):
        assert (self.f_index is not None)
        assert (self.bin_id is not None)

        right_flags = self.bin_id <= df_bins[self.f_index]
        left_flags = ~right_flags

        return left_flags & current_node_flags, right_flags & current_node_flags
