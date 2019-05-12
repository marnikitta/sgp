import numpy as np
from scipy.optimize import approx_fprime
from sklearn.gaussian_process.tests.test_gpr import X, y
from sklearn.utils.testing import assert_true, assert_greater, assert_array_less, assert_almost_equal

from sgp.kernel import RBF
from sgp.variational import VariationalGP as VGP

kernel = RBF(length_scale=1.0)


def test_gpr_interpolation():
    gpr = VGP(kernel=kernel, std=1e-7, fit_std=False).fit(X, y)
    y_pred, y_cov = gpr.predict(X, return_cov=True)

    assert_almost_equal(y_pred, y, decimal=5)
    assert_almost_equal(np.diag(y_cov), 0.0, decimal=5)


def test_lml_improving():
    init_std = 0.1
    gpr = VGP(kernel=kernel, std=init_std).fit(X, y)

    assert_greater(gpr.log_marginal_likelihood(gpr.kernel.theta, gpr.std, gpr.inducing_points, eval_gradient=False),
                   gpr.log_marginal_likelihood(kernel.theta, init_std, gpr.inducing_points, eval_gradient=False))


def test_converged_to_local_maximum():
    gpr = VGP(kernel=kernel).fit(X, y)

    lml, lml_gradient = \
        gpr.log_marginal_likelihood(gpr.kernel.theta, gpr.std, gpr.inducing_points, True)

    assert_true(np.all((np.abs(lml_gradient) < 1e-4) |
                       (gpr.kernel.theta == gpr.kernel.bounds[:, 0]) |
                       (gpr.kernel.theta == gpr.kernel.bounds[:, 1])))


def test_solution_inside_bounds():
    gpr = VGP(kernel=kernel).fit(X, y)

    bounds = gpr.kernel.bounds
    max_ = np.finfo(gpr.kernel.theta.dtype).max
    tiny = 1e-10
    bounds[~np.isfinite(bounds[:, 1]), 1] = max_

    assert_array_less(bounds[:, 0], gpr.kernel.theta + tiny)
    assert_array_less(gpr.kernel.theta, bounds[:, 1] + tiny)


def test_lml_gradient():
    gpr = VGP(kernel=kernel).fit(X, y)

    lml, lml_gradient = gpr.log_marginal_likelihood(kernel.theta, 1, gpr.inducing_points, True)
    lml_gradient_approx = \
        approx_fprime(np.concatenate((kernel.theta, [1])),
                      lambda x: gpr.log_marginal_likelihood(x[:-1], x[-1], gpr.inducing_points, False),
                      1e-10)

    assert_almost_equal(lml_gradient, lml_gradient_approx, 3)


test_gpr_interpolation()
test_lml_gradient()
test_converged_to_local_maximum()
test_lml_improving()
test_solution_inside_bounds()
