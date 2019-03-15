# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

try:
    import autograd.numpy as np
    from autograd.scipy.special import erf
    from autograd import elementwise_grad as egrad
    AUTOGRAD = True
except ImportError:
    import numpy as np
    from autograd.scipy.special import erf
    AUTOGRAD = False

from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.acquisition import Acquisition

class L2ExpectedImprovement(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], target: np.array) -> None:
        """
        Expected Improvement acquisition function for target vector estimation.
        For more information see:
        Efficient Bayesian Optimization for Target Vector Estimation
        Uhrenholt, Anders K. and Jensen, Bjørn S.
        2019, AISTATS

        :param model: model that is used to compute the improvement.
        :param target: target to be estimated.
        """
        self.model = model
        self.target = target
        self.k = target.shape[-1]
        self.grad_fun = None

    def _ncx2_cdf(self, t, k_, nc):
        """
        Approximation of the cumulative distribution function for a noncentral
        Chi-squared distributed variable.
        """
        r1 = k_ + nc
        r2 = 2*(k_ + 2*nc)
        r3 = 8*(k_ + 3*nc)

        m = 1 - r1*r3/(3*r2**2)
        z = (t/(k_ + nc))**m
        alpha = 1 + m*(m - 1)*(r2/(2*r1**2) - \
                    (2 - m)*(1 - 3*m)*r2**2/(8*r1**4))
        rho = m*np.sqrt(r2)/r1 * (1 - (1 - m)*(1 - 3*m)/(4*r1**2)*r2)

        norm_cdf = 0.5*(1 + erf((z - alpha)/(rho*np.sqrt(2))))
        return norm_cdf

    def _l2_ei(self, mv, y_min):
        """
        Helper function for the expected improvement.
        """
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

        We use autograd [TODO cite] to find the gradient.

        :param x: points where the acquisition is evaluated.
        """
        if not AUTOGRAD:
            raise NotImplementedError()
        elif self.grad_fun is None:
            self.grad_fun = egrad(self._l2_ei)

        y_min = ((self.model.Y-self.target)**2).sum(axis=1).min()
        k = self.target.shape[-1]

        # Values and derivatives for GP mean and variance w.r.t. input
        means, variances = self.model.predict(x)
        if variances.shape[-1] < k:
            variances = np.repeat(variances, k, axis=1)
        dmean_dx, dvariance_dx = self.model.model.predictive_gradients(x)

        if dvariance_dx.ndim == 2:
            dvariance_dx = np.atleast_3d(dvariance_dx)
            dvariance_dx = np.repeat(dvariance_dx, k, axis=2)

        # Values and derivatives for Expected Improvement w.r.t. mean and 
        # variances
        ei = self._l2_ei((means, variances), y_min)
        dei_dmean, dei_dvariance = self.grad_fun((means, variances), y_min)

        # Derivatives for Expected Improvement w.r.t. input
        dei_dx = np.zeros_like(x)
        for i in range(len(dei_dx)):
            dei_dmean_dx = np.dot(dmean_dx[i], dei_dmean[i])
            dei_dvariance_dx = np.dot(dvariance_dx[i], dei_dvariance[i])
            dei_dx[i] = dei_dmean_dx + dei_dvariance_dx

        return np.atleast_2d(ei).T, dei_dx

    @property
    def has_gradients(self) -> bool:
        """Returns whether this acquisition has gradients"""
        return AUTOGRAD
