import numpy as np
from scipy.spatial.distance import cdist
from sklearn.gaussian_process.kernels import _check_length_scale, RBF as SRBF


class RBF(SRBF):
    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient and Y is not None:
            X = np.atleast_2d(X)

            length_scale = _check_length_scale(X, self.length_scale)
            dists = cdist(X / length_scale, Y / length_scale, metric='sqeuclidean')

            K = np.exp(-.5 * dists)

            if self.hyperparameter_length_scale.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * dists)[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                K_gradient = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2 \
                             / (length_scale ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient

        else:
            return super().__call__(X, Y, eval_gradient)
