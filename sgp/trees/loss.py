import numpy as np


class AdditiveLoss:
    def valid_stats(self, stats):
        pass

    def leaf_predicts(self, stats):
        pass

    def node_weight(self, stats):
        return stats[0]

    def score(self, left_stats, right_stats):
        pass

    def pretty_str(self, stats) -> str:
        pass


class MSEUpliftLoss(AdditiveLoss):
    def __init__(self, min_treatment_leaf: int = 20):
        self.min_treatment_leaf = min_treatment_leaf

    def node_weight(self, stats):
        return stats[2]

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
        return (stats[3] >= self.min_treatment_leaf) & (stats[2] - stats[3] >= self.min_treatment_leaf)

    def pretty_str(self, stats) -> str:
        uplift = f'uplift={self.leaf_predicts(stats):.3}'
        n = f'n={stats[3]:.0f}#{stats[2] - stats[3]:.0f}'
        score = f'score={self.__leaf_score(stats):.2}'
        return f'{uplift}, {n}, {score}'

    def score(self, left_stats, right_stats):
        return self.__leaf_score(left_stats) + self.__leaf_score(right_stats)

    def __leaf_score(self, stats):
        p = stats[1] / stats[3]
        q = (stats[0] - stats[1]) / (stats[2] - stats[3])
        return -(p - q) ** 2


class SigUpliftLoss(AdditiveLoss):
    # Radcliffe, N. and Patrick D. Surry. “Real-World Uplift Modelling with Significance-Based Uplift Trees.” (2012).

    def __init__(self, min_treatment_leaf: int = 20):
        self.min_treatment_leaf = min_treatment_leaf

    @staticmethod
    def point_stats(y: np.ndarray, tr: np.ndarray) -> np.ndarray:
        return np.vstack((y,  # 0
                          tr * y,  # 1
                          y ** 2,  # 2
                          tr * (y ** 2),  # 3
                          np.ones(y.shape),  # 4
                          tr))  # 5

    def node_weight(self, stats):
        return stats[4]

    def leaf_predicts(self, stats):
        p = stats[1] / stats[5]
        q = (stats[0] - stats[1]) / (stats[4] - stats[5])
        return p - q

    def score(self, left_stats, right_stats):
        U_l = self.leaf_predicts(left_stats)
        U_r = self.leaf_predicts(right_stats)
        C_44 = 1.0 / left_stats[5] + 1.0 / (left_stats[4] - left_stats[5]) \
               + 1.0 / right_stats[5] + 1.0 / (right_stats[4] - right_stats[5])
        n = left_stats[4] + right_stats[4]

        SSE_left = self.__sum_of_squares(left_stats[5], left_stats[1], left_stats[3]) \
                   + self.__sum_of_squares(left_stats[4] - left_stats[5], left_stats[0] - left_stats[1],
                                           left_stats[2] - left_stats[3])

        SSE_right = self.__sum_of_squares(right_stats[5], right_stats[1], right_stats[3]) \
                    + self.__sum_of_squares(right_stats[4] - right_stats[5], right_stats[0] - right_stats[1],
                                            right_stats[2] - right_stats[3])
        SSE = SSE_left + SSE_right

        return -(n - 4.0) * (U_r - U_l) ** 2 / (C_44 * SSE)

    def __sum_of_squares(self, cnt, sum, sum_sq):
        return (sum_sq / cnt - (sum / cnt) ** 2) * cnt

    def valid_stats(self, stats: np.ndarray):
        return (stats[5] >= self.min_treatment_leaf) & (stats[4] - stats[5] >= self.min_treatment_leaf)

    def pretty_str(self, stats) -> str:
        uplift = f'uplift={self.leaf_predicts(stats):.3}'
        n = f'n={stats[5]:.0f}#{stats[4] - stats[5]:.0f}'
        return f'{uplift}, {n}'


class MSELoss(AdditiveLoss):
    def __init__(self, min_samples_leaf: int = 1):
        self.min_samples_leaf = min_samples_leaf

    def leaf_predicts(self, stats):
        return stats[1] / stats[0]

    def valid_stats(self, stats: np.ndarray):
        return stats[0] >= self.min_samples_leaf

    @staticmethod
    def point_stats(y: np.ndarray):
        return np.vstack((np.ones(y.shape), y, y ** 2))

    def score(self, left_stats, right_stats) -> np.ndarray:
        return self.__leaf_score(left_stats) + self.__leaf_score(right_stats)

    def __leaf_score(self, stats):
        assert stats.shape[0] == 3
        # sum of squares
        return (stats[2] / stats[0] - (stats[1] / stats[0]) ** 2) * stats[0]

    def pretty_str(self, stats: np.ndarray) -> str:
        n = f'n={stats[0]:.0f}'
        mean = f'mean={self.leaf_predicts(stats):.2f}'
        var = f'var={self.__leaf_score(stats):.2f}'
        return f'{mean}, {n}, {var}'


def uplift_at_k(y_predicted: np.ndarray, y: np.ndarray, tr: np.ndarray) -> np.ndarray:
    order = np.argsort(-y_predicted)
    y_0_sums, tr_0_sums = np.cumsum(y[order] * (tr[order] == 0)), np.cumsum(tr[order] == 0)
    y_1_sums, tr_1_sums = np.cumsum(y[order] * (tr[order] == 1)), np.cumsum(tr[order] == 1)
    return y_1_sums / tr_1_sums - y_0_sums / tr_0_sums


def uplift_at_percents(y_pred: np.ndarray, y: np.ndarray, tr: np.ndarray, percents: int = 10) -> float:
    uplifts = uplift_at_k(y_pred, y, tr)
    return uplifts[int(len(uplifts) * percents / 100)] / uplifts[-1]
