# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

try:
    import autograd.numpy as np
    from autograd import elementwise_grad
    AUTOGRAD = True
except ImportError:
    import numpy as np
    AUTOGRAD = False

from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.acquisition import Acquisition


class L2NegativeLowerConfidenceBound(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], target: np.array,
                 beta: np.float64=np.float64(1), use_grad=True) -> None:

        """
        This acquisition computes the negative lower confidence bound for a given input point. This is the same
        as optimizing the upper confidence bound if we would maximize instead of minimizing the objective function.
        For information as well as some theoretical insights see:

        Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design
        Niranjan Srinivas, Andreas Krause, Sham Kakade, Matthias Seeger
        In Proceedings of the 27th International Conference  on  Machine  Learning

        :param model: The underlying model that provides the predictive mean and variance for the given test points
        :param beta: Is multiplied on the standard deviation to control exploration / exploitation
        """
        self.model = model
        self.target = target
        self.k = target.shape[-1]
        self.beta = beta
        self.grad_fun = None
        self.use_grad = use_grad  # TODO remove

    def _l2_bound(self, mv):
        means, variances = mv
        nc = ((means-self.target)**2).sum(axis=1)
        gamma2 = variances.mean(axis=1)

        r1 = self.k + nc
        r2 = 2*(self.k + 2*nc)
        r3 = 8*(self.k + 3*nc)

        m = 1 - r1*r3/(3*r2**2)
        alpha = 1 + m*(m - 1)*(r2/(2*r1**2) - \
                    (2 - m)*(1 - 3*m)*r2**2/(8*r1**4))
        rho = m*np.sqrt(r2)/r1 * (1 - (1 - m)*(1 - 3*m)/(4*r1**2)*r2)

        root = (alpha - self.beta*rho)**(1./m)
        return -root * (self.k + nc)*gamma2

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the negative lower confidence bound

        :param x: points where the acquisition is evaluated.
        """
        means, variances = self.model.predict(x)
        bound = self._l2_bound((means, variances))
        return np.atleast_2d(bound).T

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the negative lower confidence bound and its derivative

        :param x: points where the acquisition is evaluated.
        """
        if not (AUTOGRAD and self.use_grad):
            raise NotImplementedError()
        elif self.grad_fun is None:
            self.grad_fun = elementwise_grad(self._l2_bound)

        means, variances = self.model.predict(x)
        dmean_dx, dvariance_dx = self.model.predictive_gradients(x)
        if dvariance_dx.ndim == 2:
            dvariance_dx = np.atleast_3d(dvariance_dx)
            dvariance_dx = np.repeat(dvariance_dx, dmean_dx.shape[-1], axis=2)

        bound = self._l2_bound((means, variances))
        dbound_dmean, dbound_dvariance = self.grad_fun((means, variances))

        dmean_dx = np.dot(dmean_dx, dbound_dmean.T)[:, :, 0]
        dvariance_dx = np.dot(dvariance_dx, dbound_dvariance.T)[:, :, 0]
        dbound_dx = dmean_dx + dvariance_dx

        return np.atleast_2d(bound).T, dbound_dx

    @property
    def has_gradients(self):
        return AUTOGRAD and self.use_grad

if __name__ == "__main__":
    from GPy.models import GPRegression
    from emukit.bayesian_optimization.acquisitions \
        import NegativeLowerConfidenceBound as LCB
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

    
    model_gpy = GPRegression(X0[:, None], D0[:, None])
    model_gpy.optimize_restarts(num_restarts=10, max_iters=50, robust=True,
                                verbose=False)
    model = GPyModelWrapper(model_gpy)
    acq = LCB(model=model)

    plt.ion()
    fig, axes = plt.subplots(4, 1, figsize=(8, 12))
    ax0, ax1, ax2, ax3 = axes
    for ax in axes:
        ax.grid()
    r = np.linspace(0, 2*np.pi, 500)

    y_noiseless = f_noiseless(r)
    ax0.plot(r, y_noiseless)
    ax0.axhline(y=target, color="r", linestyle=":")
    ax0.plot(X0, Y0, 'ko')
    
    d_noiseless_vals = d_noiseless(r)
    l, m, u = model_gpy.predict_quantiles(r[:, None], (25, 50, 75))
    ax1.plot(r, d_noiseless_vals, 'k--')
    ax1.plot(X0, D0, 'ko')
    color = plt.cm.Set1(1)[:3]
    ax1.plot(r, m, color=color)
    ax1.fill_between(r, l.ravel(), u.ravel(), color=color + (.3,))

    means, variances = model.predict(r[:, None])
    d_min = ((Y0 - target)**2).min()
    k = 1

    improvement, dimprovement_dx = acq.evaluate_with_gradients(r[:, None])
    ax2.plot(r, improvement, 'k')
    ax3.plot(r, dimprovement_dx, 'k')

    from scipy.optimize import minimize

    def wrapper(x):
        x = np.atleast_2d(x)
        v, dv = acq.evaluate_with_gradients(x)
        return -v[0], -dv[0]

    options = {"gtol": 1e-9}
    res = minimize(wrapper, np.array([5.0]), jac=True, method="L-BFGS-B")
    ax2.axvline(x=res["x"])
    ipdb.set_trace()
