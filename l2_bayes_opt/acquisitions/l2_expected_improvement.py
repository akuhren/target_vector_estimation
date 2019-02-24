# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

from GPyOpt.util.general import get_quantiles

try:
    import autograd.numpy as np
    from autograd.scipy.stats import norm
    from autograd import elementwise_grad
    AUTOGRAD = True
except ImportError:
    import numpy as np
    from scipy.stats import norm
    AUTOGRAD = False

from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.acquisition import Acquisition

class L2ExpectedImprovement(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], target: np.array,
                 use_grad=True)-> None:
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. For more information see:
        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization
        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """

        self.model = model
        self.target = target
        self.k = target.shape[-1]
        self.grad_fun = None
        self.use_grad = use_grad  # TODO remove!

    def _ncx2_cdf(self, t, k_, nc):
        r1 = k_ + nc
        r2 = 2*(k_ + 2*nc)
        r3 = 8*(k_ + 3*nc)

        m = 1 - r1*r3/(3*r2**2)
        z = (t/(k_ + nc))**m
        alpha = 1 + m*(m - 1)*(r2/(2*r1**2) - \
                    (2 - m)*(1 - 3*m)*r2**2/(8*r1**4))
        rho = m*np.sqrt(r2)/r1 * (1 - (1 - m)*(1 - 3*m)/(4*r1**2)*r2)

        return norm.cdf((z - alpha)/rho)

    def _l2_ei(self, mv, y_min):
        means, variances = mv

        gamma2 = variances.mean(axis=1)
        nc = ((self.target-means)**2).sum(axis=1) / gamma2
        
        def h(k_):
            return self._ncx2_cdf(y_min / gamma2, k_, nc)

        t1 = y_min * self._ncx2_cdf(y_min / gamma2, self.k, nc)
        t2 = gamma2 * (self.k * h(self.k + 2) + nc * h(self.k + 4))

        return t1 - t2

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Improvement.
        :param x: points where the acquisition is evaluated.
        """

        means, variances = self.model.predict(x)
        y_min = ((self.model.Y-self.target)**2).sum(axis=1).min()
        improvement = self._l2_ei((means, variances), y_min)

        return np.atleast_2d(improvement).T

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Improvement and its derivative.
        :param x: locations where the evaluation with gradients is done.
        """
        if not (AUTOGRAD and self.use_grad):
            raise NotImplementedError()
        elif self.grad_fun is None:
            self.grad_fun = elementwise_grad(self._l2_ei)

        means, variances = self.model.predict(x)

        y_min = ((self.model.Y-self.target)**2).sum(axis=1).min()
        dmean_dx, dvariance_dx = self.model.model.predictive_gradients(x)
        if dvariance_dx.ndim == 2:
            dvariance_dx = np.atleast_3d(dvariance_dx)
            dvariance_dx = np.repeat(dvariance_dx, dmean_dx.shape[-1], axis=2)

        improvement = self._l2_ei((means, variances), y_min)
        dei_dmean, dei_dvariance = self.grad_fun((means, variances), y_min)
        
        dmean_dx = np.dot(dmean_dx, dei_dmean.T)[:, :, 0]
        dvariance_dx = np.dot(dvariance_dx, dei_dvariance.T)[:, :, 0]
        dimprovement_dx = dmean_dx + dvariance_dx
        
        return np.atleast_2d(improvement).T, dimprovement_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return AUTOGRAD and self.use_grad


if __name__ == "__main__":
    from GPy.models import GPRegression
    from emukit.model_wrappers import GPyModelWrapper
    import matplotlib.pyplot as plt
    import ipdb
    from scipy.stats import norm as scipy_norm

    np.random.seed(0)

    def f_noiseless(x):
        y1 = np.sin(x).sum(axis=1)
        y2 = np.cos(x).sum(axis=1)
        y3 = np.tan(x).sum(axis=1)
        return np.vstack((y1, y2, y3)).T
    
    def f(x):
        res = f_noiseless(x)
        return res + scipy_norm.rvs(scale=0.1, size=res.shape)

    X0 = np.random.random((10, 3)) * 2*np.pi
    Y0 = f(X0)
    target = np.zeros((3,))

    model_gpy = GPRegression(X0, Y0, normalizer=True)
    model_gpy.optimize_restarts(num_restarts=10, max_iters=50, robust=True,
                                verbose=False)
    model = GPyModelWrapper(model_gpy)
    acq = L2ExpectedImprovement(model=model, target=target)

    # plt.ion()
    # fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    # ax1, ax2, ax3 = axes
    # for ax in axes:
    #     ax.grid()
    # r = np.linspace(0, 2*np.pi, 500)
    # y_noiseless = f_noiseless(r)
    # l, m, u = model.predict_quantiles(r[:, None], (25, 50, 75))
    # ax1.plot(r, y_noiseless, 'k--')
    # ax1.plot(X0, Y0, 'ko')
    # color = plt.cm.Set1(1)[:3]
    # ax1.plot(r, m, color=color)
    # ax1.fill_between(r, l.ravel(), u.ravel(), color=color + (.3,))
    # ax1.axhline(y=target, color="r", linestyle=":")
    #
    # means, variances = model.predict(r[:, None])
    # d_min = ((Y0 - target)**2).min()
    # k = 1
    #
    # improvement, dimprovement_dx = acq.evaluate_with_gradients(r[:, None])
    # ax2.plot(r, improvement, 'k')
    # ax3.plot(r, dimprovement_dx, 'k')

    from scipy.optimize import minimize

    def wrapper(x):
        x = np.atleast_2d(x)
        v, dv = acq.evaluate_with_gradients(x)
        return -v[0], -dv[0]
    
    options = {"gtol": 1e-9}
    res = minimize(wrapper, np.zeros((3,)), jac=True, method="L-BFGS-B")
    # ax2.axvline(x=res["x"])
    ipdb.set_trace()

    
    # improvement = acq.evaluate(r[:, None])
    
    x_good, x_bad = np.array([[0]]), np.array([[np.pi]])

    ei_good = acq.evaluate_with_gradients(np.vstack((x_good, x_bad)))
    ei_bad = acq.evaluate(x_bad)

    x_new = np.array([[3.]])
    acq.evaluate()