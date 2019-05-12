import warnings

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import fmin_l_bfgs_b
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils import check_random_state, check_X_y, check_array

from sgp.kernel import RBF


class VariationalGP(BaseEstimator, RegressorMixin):
    def __init__(self, kernel=None, std=0.1, sparsity=10, random_state=None, fit_std=True):
        self.kernel = kernel
        self.std = std
        self.sparsity = sparsity
        self.random_state = random_state
        self.fit_std = fit_std

    def fit(self, X, y, inducing_points=None):
        if self.kernel is None:
            self.kernel = RBF(2)
        else:
            self.kernel = clone(self.kernel)

        X, y = check_X_y(X, y, multi_output=False, y_numeric=True)

        self.X_f = X
        self.y = y
        self._rng = check_random_state(self.random_state)

        if inducing_points is None:
            inducing_points = self.X_f[
                self._rng.choice(np.arange(X.shape[0]), size=np.minimum(self.sparsity, X.shape[0]), replace=False)]

        def obj_func(params, eval_gradient=True):
            theta = params[:-1]
            std = params[-1]
            if eval_gradient:
                lml, grad = self.log_marginal_likelihood(theta, std, inducing_points, eval_gradient=True)
                return -lml, -grad
            else:
                return -self.log_marginal_likelihood(theta, std, inducing_points, eval_gradient=False)

        initial_params = np.append(self.kernel.theta, [self.std])

        bounds = np.vstack((self.kernel.bounds, np.array([1e-3, 1e5] if self.fit_std else [self.std, self.std])))
        params_opt, func_min, convergence_dict = fmin_l_bfgs_b(obj_func, initial_params, bounds=bounds)
        if convergence_dict["warnflag"] != 0:
            warnings.warn("fmin_l_bfgs_b terminated abnormally with the state: %s" % convergence_dict)

        self.kernel.theta = params_opt[:-1]
        self.std = params_opt[-1]

        # Some day we would greedily optimize inducing points via EM-algorithm
        self.inducing_points = inducing_points
        self.log_marginal_likelihood_value, self.log_marginal_likelihood_grad_value = self.log_marginal_likelihood(
            self.kernel.theta, self.std, inducing_points)

        return self

    def predict(self, X_star, return_cov=False, return_std=False, latent=True):
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        X_star = check_array(X_star)

        mu = self.K_u_u.dot(self.alpha) / self.std ** 2

        K_star_u = self.kernel(X_star, self.inducing_points)
        K_star = self.kernel(X_star, X_star)
        gamma = cho_solve((self.L_u, True), K_star_u.T)
        beta = cho_solve((self.L, True), self.K_u_u.dot(gamma))

        f_star = K_star_u.dot(cho_solve((self.L_u, True), mu))
        K_star_post = K_star - K_star_u.dot(gamma - beta)

        if not latent:
            K_star_post[np.diag_indices_from(K_star_post)] += self.std ** 2

        if return_std:
            return f_star, np.sqrt(np.diag(K_star_post))
        elif return_cov:
            return f_star, K_star_post
        else:
            return f_star

    def sample_y(self, X, n_samples=1, random_state=0, latent=True):
        rng = check_random_state(random_state)
        y_mean, y_cov = self.predict(X, latent=latent, return_cov=True)
        return rng.multivariate_normal(y_mean, y_cov, n_samples).T

    def log_marginal_likelihood(self, theta, std, inducing_points, eval_gradient=True):
        X_u = inducing_points

        kernel_ = self.kernel.clone_with_theta(theta)

        X_f = self.X_f
        y = self.y

        K_u_u, d_u_u_all = kernel_(X_u, eval_gradient=True)
        K_u_f, d_u_f_all = kernel_(X_u, X_f, eval_gradient=True)
        K_u_f_dot = K_u_f.dot(K_u_f.T)

        eps = 1e-7 * np.eye(K_u_u.shape[0])

        try:
            L = cholesky(K_u_u + K_u_f_dot / std ** 2 + eps, lower=True)
            L_u = cholesky(K_u_u + eps, lower=True)
        except np.linalg.LinAlgError:
            warnings.warn('Matrix is not positive semi-definite')
            return (-np.inf, np.zeros(theta.shape[0] + 1)) if eval_gradient else -np.inf

        lml = 0

        alpha = cho_solve((L, True), K_u_f.dot(y))
        lml -= np.sum(np.log(np.diagonal(L))) - np.sum(np.log(np.diagonal(L_u))) + X_f.shape[0] * np.log(std)

        quad_no_y = y / std ** 2 - K_u_f.T.dot(alpha) / (std ** 4)
        lml -= 0.5 * y.dot(quad_no_y)
        lml -= 0.5 * X_f.shape[0] * np.log(2 * np.pi)

        ksi = cho_solve((L_u, True), K_u_f_dot)
        trace = np.sum(kernel_.diag(X_f)) - np.trace(ksi)
        lml -= 0.5 * std ** -2 * trace

        self.K_u_u = K_u_u
        self.alpha = alpha
        self.L = L
        self.L_u = L_u

        if not eval_gradient:
            return lml

        # + 1 is for std gradient, as it is evaluated separately
        lml_grad = np.zeros(theta.shape[0] + 1)

        for i in np.arange(theta.shape[0]):
            d_u_u = d_u_u_all[..., i]
            d_u_f = d_u_f_all[..., i]

            tra = K_u_f.dot(d_u_f.T)
            fi = d_u_u + std ** -2 * (tra + tra.T)
            # Here we need only diagonal elements, may be we can optimize it somehow
            rho = cho_solve((L_u, True), d_u_u)
            lml_grad[i] -= 0.5 * (np.trace(cho_solve((L, True), fi)) - np.trace(rho))

            l = K_u_f.T.dot(cho_solve((L, True), d_u_f.dot(y) - fi.dot(alpha))) + d_u_f.T.dot(alpha)
            lml_grad[i] += 0.5 * y.dot(l) / std ** 4

            lml_grad[i] -= 0.5 * std ** -2 * (np.trace(rho.dot(ksi)) + np.trace(cho_solve((L_u, True), tra + tra.T)))

        lml_grad[-1] -= X_f.shape[0] / std - 1 / (std ** 3) * np.trace(cho_solve((L, True), K_u_f_dot))

        alpha_prime = cho_solve((L, True), K_u_f.dot(quad_no_y))
        lml_grad[-1] += std * y.dot(quad_no_y / std ** 2 - K_u_f.T.dot(alpha_prime) / (std ** 4))

        lml_grad[-1] += std ** -3 * trace

        return lml, lml_grad
