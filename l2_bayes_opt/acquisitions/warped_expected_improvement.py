# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0



import numpy as np
from scipy.stats import truncnorm, norm

from GPyOpt.util.general import get_quantiles

from emukit.core.interfaces import IModel, IDifferentiable
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement

from typing import Tuple, Union


class WarpedExpectedImprovement(ExpectedImprovement):

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.
        :param x: points where the acquisition is evaluated.
        """
        # Rename for consistency with paper
        g = self.model.model.warping_function.f
        g_inv = self.model.model.warping_function.f_inv

        # Find minimum in unwarped space
        y_min = self.model.Y.min()
        z_min = g(np.atleast_2d(y_min)).ravel()

        # Get predictive dists in unwarped space
        self.model.model.predict_in_warped_space = False
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)
        self.model.model.predict_in_warped_space = True

        # Get expected improvement in warped space
        pdf, cdf, u = get_quantiles(
            self.jitter, z_min, mean, standard_deviation)
        z_improvement = standard_deviation * (u * cdf + pdf)

        # Map expected improvement to warped space
        y_improvement = g_inv(z_improvement)

        return y_improvement

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.
        :param x: locations where the evaluation with gradients is done.
        """

        # Rename for consistency with paper
        g = self.model.model.warping_function.f
        g_inv = self.model.model.warping_function.f_inv
        dg_dx = self.model.model.warping_function.fgrad_y

        # Find minimum in unwarped space
        y_min = self.model.Y.min()
        z_min = g(np.atleast_2d(y_min)).ravel()

        # Get predictive dists and derivatives in unwarped space
        self.model.model.predict_in_warped_space = False
        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)
        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)
        self.model.model.predict_in_warped_space = True

        # Get expected improvement and derivative in unwarped space
        pdf, cdf, u = get_quantiles(
            self.jitter, y_min, mean, standard_deviation)

        z_improvement = standard_deviation * (u * cdf + pdf)
        dz_improvement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx

        # Map expected improvement and derivative to warped space
        y_improvement = g_inv(z_improvement)
        dy_improvement_dx = dg_dx(z_improvement) * dz_improvement_dx

        return y_improvement, dz_improvement_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False


if __name__ == "__main__":
    from GPy.models import WarpedGP
    from emukit.model_wrappers import GPyModelWrapper
    import matplotlib.pyplot as plt
    import ipdb
    from scipy.stats import norm as scipy_norm

    np.random.seed(0)

    def f_noiseless(x):
        return 5 + np.sin(x)

    def f(x):
        res = f_noiseless(x)
        return res + scipy_norm.rvs(scale=0.1, size=res.shape)

    def d_noiseless(x):
        return (f_noiseless(x) - target)**2

    def d(x):
        return ((f(x) - target)**2).sum(axis=1)


    X0 = np.random.random((10,)) * 2*np.pi
    Y0 = f(X0)
    target = np.array([4.5])
    D0 = (Y0 - target)**2

    model_gpy = WarpedGP(X0[:, None], D0[:, None])
    model_gpy.optimize_restarts(num_restarts=10, max_iters=50, robust=True,
                                verbose=False)
    model = GPyModelWrapper(model_gpy)
    acq = WarpedExpectedImprovement(model=model)

    plt.ion()
    fig, axes = plt.subplots(5, 1, figsize=(8, 12))
    ax0, ax1, ax2, ax3, ax4 = axes
    for ax in axes:
        ax.grid()
    r = np.linspace(0, 2*np.pi, 500)

    color = plt.cm.Set1(1)[:3]

    y_noiseless = f_noiseless(r)
    ax0.plot(r, y_noiseless, 'k--')
    ax0.axhline(y=target, color="r", linestyle=":")
    ax0.plot(X0, Y0, 'ko')

    model_gpy.predict_in_warped_space = False
    l, m, u = model_gpy.predict_quantiles(r[:, None], (25, 50, 75))
    model_gpy.predict_in_warped_space = True
    ax1.plot(r, m, color=color)
    ax1.fill_between(r, l.ravel(), u.ravel(), color=color + (.3,))

    d_noiseless_vals = d_noiseless(r)
    l, m, u = model_gpy.predict_quantiles(r[:, None], (25, 50, 75))
    ax2.plot(r, d_noiseless_vals, 'k--')
    ax2.plot(X0, D0, 'ko')
    ax2.plot(r, m, color=color)
    ax2.fill_between(r, l.ravel(), u.ravel(), color=color + (.3,))

    r = np.linspace(0, 2*np.pi, 100)
    improvement = acq.evaluate(r[:, None])
    ax3.plot(r, improvement, 'k')

    improvement, dimprovement_dx = acq.evaluate_with_gradients(r[:, None])
    ax4.plot(r, dimprovement_dx, 'k')
    
    stationary = np.isclose(dimprovement_dx.ravel(), 0, atol=1e-4)
    for x in r[stationary]:
        ax3.axvline(x=x, color='k', linestyle="--")

    ipdb.set_trace()
    from scipy.optimize import minimize

    def wrapper(x):
        x = np.atleast_2d(x)
        y, dy = acq.evaluate_with_gradients(x)
        return -y[0], -dy[0]

    options = {"gtol": 1e-9}
    res = minimize(wrapper, np.array([5.0]), jac=True)
    ax3.axvline(x=res["x"])
    ipdb.set_trace()
