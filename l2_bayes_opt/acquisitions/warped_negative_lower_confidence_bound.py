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

from emukit.bayesian_optimization.acquisitions \
    import NegativeLowerConfidenceBound as LCB


class WarpedNegativeLowerConfidenceBound(LCB):
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the negative lower confidence bound

        :param x: points where the acquisition is evaluated.
        """
        self.model.predict_in_warped_space = False
        try:
            sup = super(WarpedNegativeLowerConfidenceBound, self)
            return sup.evaluate(x)
        finally:
            self.model.predict_in_warped_space = True

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the negative lower confidence bound and its derivative

        :param x: points where the acquisition is evaluated.
        """
        self.model.predict_in_warped_space = False
        try:
            sup = super(WarpedNegativeLowerConfidenceBound, self)
            return sup.evaluate_with_gradients(x)
        finally:
            self.model.predict_in_warped_space = True


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
    acq = WarpedNegativeLowerConfidenceBound(model=model)

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
