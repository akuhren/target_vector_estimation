# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

try:
    import autograd.numpy as np
    from autograd import elementwise_grad as egrad
    AUTOGRAD = True
except ImportError:
    import numpy as np
    AUTOGRAD = False

from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.acquisition import Acquisition


class L2NegativeLowerConfidenceBound(Acquisition):
    def __init__(self, model: Union[IModel, IDifferentiable], target: np.array,
                 beta: np.float64=np.float64(1)) -> None:
        """
        Negative Lower Confidence Bound acquisition function for target vector
        estimation. For more information see:
        Efficient Bayesian Optimization for Target Vector Estimation
        Uhrenholt, Anders K. and Jensen, Bjørn S.
        2019, AISTATS

        :param model: model that is used to compute the improvement.
        :param target: target to be estimated.
        :param beta: Exploration / exploitation parameter.
        """
        self.model = model
        self.target = target
        self.k = target.shape[-1]
        self.beta = beta
        self.grad_fun = None

    def _l2_bound(self, mv):
        """
        Approximate bound for a noncentral Chi-squared distributed variable.
        """
        means, variances = mv
        gamma2 = variances.mean(axis=1)
        nc = ((means - self.target)**2).sum(axis=1) / gamma2

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
        Computes the negative lower confidence bound.

        :param x: points where the acquisition is evaluated.
        """
        means, variances = self.model.predict(x)
        bound = self._l2_bound((means, variances))
        return np.atleast_2d(bound).T

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the negative lower confidence bound and its derivative.

        We use autograd [TODO cite] to find the gradient.

        :param x: points where the acquisition is evaluated.
        """
        if not AUTOGRAD:
            raise NotImplementedError()
        elif self.grad_fun is None:
            self.grad_fun = egrad(self._l2_bound)

        # Values and derivatives for GP mean and variance w.r.t. input
        means, variances = self.model.predict(x)
        k = means.shape[-1]
        if variances.shape[-1] < k:
            variances = np.repeat(variances, k, axis=1)
        dmean_dx, dvariance_dx = self.model.model.predictive_gradients(x)

        if dvariance_dx.ndim == 2:
            dvariance_dx = np.atleast_3d(dvariance_dx)
            dvariance_dx = np.repeat(dvariance_dx, k, axis=2)

        # Values and derivatives for confidence bound w.r.t. mean and
        # variances
        bound = self._l2_bound((means, variances))
        dbound_dmean, dbound_dvariance = self.grad_fun((means, variances))

        # Derivatives for confidence bound w.r.t. input
        dbound_dx = np.zeros_like(x)
        for i in range(len(dbound_dx)):
            dbound_dmean_dx = np.dot(dmean_dx[i], dbound_dmean[i])
            dbound_dvariance_dx = np.dot(dvariance_dx[i], dbound_dvariance[i])
            dbound_dx[i] = dbound_dmean_dx + dbound_dvariance_dx

        return np.atleast_2d(bound).T, dbound_dx

    @property
    def has_gradients(self):
        """Returns whether this acquisition has gradients"""
        return AUTOGRAD
