import numpy as np


class AdditiveLoss:
    def valid_stats(self, stats):
        pass

    def leaf_predicts(self, stats):
        pass

    def score(self, stats, parent_stats=None):
        pass

    def pretty_str(self, stats) -> str:
        pass


class UpliftLoss(AdditiveLoss):
    def __init__(self, min_samples_leaf: int = 100,
                 min_treatment_leaf: int = 20,
                 prior_factor: int = 0):
        self.min_treatment_leaf = min_treatment_leaf
        self.min_samples_leaf = min_samples_leaf
        self.prior_factor = prior_factor

    @staticmethod
    def point_stats(y: np.ndarray, tr: np.ndarray) -> np.ndarray:
        return np.vstack((y,
                          tr * y,
                          np.ones(y.shape),
                          tr))

    def leaf_predicts(self, stats):
        p = stats[1] / stats[3]
        q = (stats[0] - stats[1]) / (stats[2] - stats[3])
        return p - q

    def valid_stats(self, stats: np.ndarray):
        return (stats[2] >= self.min_samples_leaf) \
               & (stats[3] >= self.min_treatment_leaf) \
               & (stats[2] - stats[3] >= self.min_treatment_leaf)

    def pretty_str(self, stats) -> str:
        uplift = f'uplift={self.leaf_predicts(stats):.3}'
        n = f'n={stats[3]:.0f}#{stats[2] - stats[3]:.0f}'
        score = f'score={self.score(stats):.2}'
        return f'{uplift}, {n}, {score}'

    def pq(self, stats, parent_stats=None):
        if parent_stats is not None:
            p_parent, q_parent = self.pq(parent_stats)
            p = (stats[1] + self.prior_factor * p_parent) / (stats[3] + self.prior_factor)
            q = (stats[0] - stats[1] + self.prior_factor * q_parent) / (stats[2] - stats[3] + self.prior_factor)
        else:
            p = stats[1] / stats[3]
            q = (stats[0] - stats[1]) / (stats[2] - stats[3])

        return p, q


class KLUpliftLoss(UpliftLoss):
    def score(self, stats, parent_stats=None):
        p, q = self.pq(stats, parent_stats)
        return -p * np.log(p / q)


class MSEUpliftLoss(UpliftLoss):
    def score(self, stats, parent_stats=None):
        p, q = self.pq(stats, parent_stats)
        return -(p - q) ** 2


class MSELoss(AdditiveLoss):
    def __init__(self, min_samples_leaf: int = 100):
        self.min_samples_leaf = min_samples_leaf

    def leaf_predicts(self, stats):
        return stats[1] / stats[0]

    def valid_stats(self, stats: np.ndarray):
        return stats[0] >= self.min_samples_leaf

    @staticmethod
    def point_stats(y: np.ndarray):
        return np.vstack((np.ones(y.shape), y, y ** 2))

    def score(self, stats, parent_stats=None) -> np.ndarray:
        return (stats[2] / stats[0] - (stats[1] / stats[0]) ** 2) * stats[0]

    def pretty_str(self, stats: np.ndarray) -> str:
        n = f'n={stats[0]:.0f}'
        mean = f'mean={self.leaf_predicts(stats):.2f}'
        var = f'var={self.score(stats):.2f}'
        return f'{mean}, {n}, {var}'
