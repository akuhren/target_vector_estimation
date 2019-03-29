#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class BayesOptPlotter(object):
    """docstring for BayesOptPlotter"""
    def __init__(self, f, target, xmin, xmax, X0=None, Y0=None):
        super(BayesOptPlotter, self).__init__()
        self.f = f
        self.target = target
        self.xmin = xmin
        self.xmax = xmax

        if X0 is None or Y0 is None:
            X0 = np.zeros((0, 1))
            Y0 = np.zeros((0, target.shape[-1]))
        self.X0 = X0
        self.Y0 = Y0
        self.D0 = ((Y0 - target)**2).sum(axis=1)

    def plot_function(self, axes, plot_samples=True):
        r = np.linspace(self.xmin, self.xmax, 500)
        Y = self.f(r[:, None])
        for j, ax in enumerate(axes):
            ax.set_title("Dimension {:d}".format(j + 1))
            ax.plot(r, Y[:, j], color=plt.cm.Set1(0),
                    label="Noiseless function")
            ax.axhline(y=self.target[j], linestyle="--", color="k",
                       label="Target")
            if plot_samples and len(self.X0) > 0:
                ax.plot(self.X0, self.Y0[:, j], 'ko', label="Noisy samples")

    def plot_distance(self, ax, plot_samples=True):
        r = np.linspace(self.xmin, self.xmax, 500)
        Y = self.f(r[:, None])
        D = ((Y-self.target)**2).sum(axis=1)
        ax.plot(r, D, color=plt.cm.Set1(0), label="Noiseless distance")
        if plot_samples and len(self.X0) > 0:
            ax.plot(self.X0, self.D0, 'ko', label="Noisy samples")

    def plot_function_predictions(self, axes, model):
        r = np.linspace(self.xmin, self.xmax, 500)
        l, m, u = model.predict_quantiles(r[:, None], (25, 50, 75))
        color = plt.cm.Set1(1)[:3]
        for j, ax in enumerate(axes):
            self.plot_quantiles(ax, r, l[:, j], m[:, j], u[:, j])

    def plot_quantiles(self, ax, r, l, m, u):
        color = plt.cm.Set1(1)[:3]
        ax.plot(r, m, color=color, label="Predictive distribution")
        ax.fill_between(r, l, u, color=color + (.3,))

    def _quantile_fn_l2(self, model):
        def inner(X, quantiles):
            mean, var = model.predict(X)
            gamma2 = var.mean(axis=1)
            nc = ((mean - self.target)**2).sum(axis=1) / gamma2
            k = mean.shape[-1]

            q = np.array(quantiles) / 100.

            r1 = k + nc
            r2 = 2*(k + 2*nc)
            r3 = 8*(k + 3*nc)

            m = 1 - r1*r3/(3*r2**2)
            alpha = 1 + m*(m - 1)*(r2/(2*r1**2) - \
                        (2 - m)*(1 - 3*m)*r2**2/(8*r1**4))
            rho = m*np.sqrt(r2)/r1 * (1 - (1 - m)*(1 - 3*m)/(4*r1**2)*r2)

            root = (np.outer(norm.ppf(q), rho) + alpha)**(1./m)
            return root * (k + nc)*gamma2
        return inner

    def plot_distance_prediction(self, ax, model, multi=False):
        if multi:
            fn = self._quantile_fn_l2(model)
        else:
            fn = model.predict_quantiles

        r = np.linspace(self.xmin, self.xmax, 500)
        l, m, u = fn(r[:, None], (25, 50, 75))
        self.plot_quantiles(ax, r, l.ravel(), m.ravel(), u.ravel())

    def plot_acquisition(self, ax, acq):
        r = np.linspace(self.xmin, self.xmax, 500)
        improvement, dimprovement_dx = acq.evaluate_with_gradients(r[:, None])
        ax.plot(r, improvement, label="Acquisition utility",
                color=plt.cm.Set1(0))
        ax.plot(r, dimprovement_dx, label="Gradient", color=plt.cm.Set1(2),
                linestyle="--")

    def decorate(self, axes):
        for ax in np.array([axes]).ravel():
            ax.grid()
            ax.set_xlim(self.xmin, self.xmax)
            ax.legend()
