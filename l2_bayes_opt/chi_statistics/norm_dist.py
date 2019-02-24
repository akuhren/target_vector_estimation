#!/usr/bin/env python

from __future__ import print_function

import numpy as np
from scipy.special import erf, erfinv, gamma
from scipy.stats import ncx2, norm, foldnorm


def _add_dim(q):
    return q[:, :, None]


def _folded_norm_moments(mu, sigma2, dm_dx=None, dv_dx=None):
    sigma = np.sqrt(sigma2)
    a = np.exp(-mu**2/(2*sigma2))
    b = erf(mu/np.sqrt(2*sigma2))
    fn_mean = np.sqrt(2/np.pi)*sigma*a + mu*b
    fn_var = mu**2+sigma2-fn_mean**2

    if dm_dx is not None and dv_dx is not None:
        # Add extra dimension for matrix arithmetics
        mu, sigma, sigma2, a, b, fn_mean, fn_var = \
            map(_add_dim, (mu, sigma, sigma2, a, b, fn_mean, fn_var))

        ds_dx = dv_dx/(2*sigma)
        da_dx = -a*mu/sigma**3*(dm_dx*sigma - ds_dx*mu)
        db_dx = 1./sigma2*np.sqrt(2/np.pi)*np.exp(-mu**2/(2*sigma2)) \
            * (dm_dx*sigma - ds_dx*mu)
        dsa_dx = ds_dx*a + sigma*da_dx
        dmb_dx = dm_dx*b + mu*db_dx

        dfn_mean_dx = np.sqrt(2/np.pi)*dsa_dx + dmb_dx
        dfn_var_dx = 2*(mu*dm_dx + sigma*ds_dx - fn_mean*dfn_mean_dx)

        return fn_mean, fn_var, dfn_mean_dx, dfn_var_dx
    else:
        return fn_mean, fn_var


def _nc_chi2_moments(mu, sigma2, dm_dx=None, dv_dx=None):
    nc = (mu**2).sum(axis=1)
    k = mu.shape[-1]

    chi_mean = k + nc
    chi_var = 2*(k + 2*nc)

    if dm_dx is not None and dv_dx is not None:
        raise NotImplementedError()
    else:
        return chi_mean, chi_var


def norm_moments(mean, var, target, norm_ord=2, dm_dx=None, dv_dx=None):
    """
    Parameters
    ----------
    mean : array_like
        2D matrix with containing the mean of the random variables for which
        to calculate the norm moments. An N x M matrix corresponds to N
        variables each with M components.
    var : array_like
        2D matrix with containing the standard deviations of the random
        variables for which to calculate the norm moments. Dimensions should be
        same as `mean`.
    target : array_like
        1D vector containing the target. The size should be equal to the number
        of components for `mean` and `var`.
    norm_ord : int in range [1, 2] (optional)
        The order of the norm. 1 is the absolute norm while 2 is the squared
        Euclidean norm.
    dm_dx : array_like (optional)
        The derivatives of the mean with respect to the input of the function
        creating `mean` and `var`. If this function has K inputs the provided
        3D tensor should have dimensions N x M x K. If `None` is provided the
        norm derivatives are not returned.
    dv_dx : array_like (optional)
        The derivatives of the standard deviations with respect to the input of
        the function creating `mean` and `var`. Should have same format as
        `dm_dx`.

    Returns
    -------
    out : tuple of array_like
        If the derivatives are not provided then only the expected value and
        the variance of the norm distribution is returned. If both derivatives
        are provided then the norm distribution derivaties for the moments
        are returned as well.
    """
    assert (dm_dx is None and dv_dx is None) or \
        (dm_dx is not None and dv_dx is not None)

    assert mean.shape == var.shape

    if dm_dx is not None:
        assert dm_dx.shape == dv_dx.shape
        assert dm_dx.shape[:2] == mean.shape

    if norm_ord == 1:
        f = _folded_norm_moments
    elif norm_ord == 2:
        f = _nc_chi2_moments
    else:
        raise Exception(("Only 1- or 2-norm are supported. Provided norm: %d"
                         % norm_ord))

    Z_mean = mean-target
    res = f(Z_mean, var, dm_dx, dv_dx)

    def _sum(a):
        a = a.sum(axis=1)
        if a.ndim == 1:
            a = a[:, None]
        return a

    return tuple(map(_sum, res))


def folded_norm_cdf(x, loc=0, scale=1):
    t1 = erf((x+loc)/(np.sqrt(2)*scale))
    t2 = erf((x-loc)/(np.sqrt(2)*scale))
    return 0.5*(t1+t2)


def _folded_norm_logpdf(x, deltas, scales):
    if deltas.shape[-1] != 1:
        raise NotImplementedError()

    deltas = np.abs(deltas).mean(axis=1)
    scales = scales.mean(axis=1)

    # Adapted from
    # en.wikipedia.org/wiki/Folded_normal_distribution#Parameter_estimation

    vs = scales**2
    t1 = -0.5 * np.log(2*np.pi*vs)
    t2 = -0.5 * (x-deltas)**2/vs
    t3 = np.log(1. + np.exp(-2*x*deltas/vs))
    return t1 + t2 + t3


def _ncx2_logpdf(x, deltas, scales):
    norms = (deltas**2).sum(axis=1)
    beta_hat = scales.mean(axis=1)
    beta_hat_inv2 = 1./beta_hat**2
    k = deltas.shape[-1]
    nc = np.clip(beta_hat_inv2*norms, 1e-5, None)
    log_pdf = ncx2.logpdf(x.ravel()*beta_hat_inv2, k, nc) - 2*np.log(beta_hat)
    log_pdf[np.isnan(log_pdf)] = 0.0
    log_pdf[np.isinf(log_pdf)] = 0.0
    return log_pdf[:, None]


def _ncx_logpdf(x, deltas, scales):
    log_d = np.log(2*x)[:, None]
    return _ncx2_logpdf(x**2, deltas, scales) + log_d


def norm_logpdf(true_distances, target, means, variances, norm_ord=2,
                normalize=False, root_norm=True):

    q = true_distances.mean()**(1./norm_ord) if normalize else 1.

    deltas = np.atleast_2d(np.abs(means-target))/q
    scales = np.atleast_2d(np.sqrt(variances))/q
    true_distances = true_distances/(q**norm_ord)

    assert deltas.shape == scales.shape
    assert deltas.shape[0] in (1, true_distances.shape[0])

    if norm_ord == 1:
        logpdf = _folded_norm_logpdf(true_distances, deltas, scales)
    elif norm_ord == 2:
        if root_norm:
            logpdf = _ncx_logpdf(true_distances, deltas, scales)
        else:
            logpdf = _ncx2_logpdf(true_distances, deltas, scales)
    else:
        raise NotImplementedError()

    return logpdf - norm_ord*np.log(q)


def _ncx2_cdf(x, k, nc, use_scipy=True):
    if use_scipy:
        return ncx2.cdf(x, k, nc)
    else:
        h = 1-2./3*(k+nc)*(k+3*nc)/(k+2*nc)**2
        p = (k + 2*nc)/(k + nc)**2
        m = (h - 1)*(1 - 3*h)

        g1 = (x/(k + nc))**h
        g2 = 1 + h*p*(h - 1 - 0.5*(2 - h)*m*p)
        g3 = h*np.sqrt(2*p)*(1 + 0.5*m*p)

        return norm.cdf((g1 - g2)/g3)


def _ncx2_ppf(q, t, m, s, use_scipy=False):
    # t: target
    # m: mean(s) at x
    # s: std(s) at x
    # Returns: x positions as rows, quantiles as columns

    q = np.atleast_2d(q)
    if np.ndim(m) < 2:
        m = np.atleast_2d(m).T
    if np.ndim(s) < 2:
        s = np.atleast_2d(s).T

    norms = ((m-t)**2).sum(axis=1)[:, None]
    beta_hat2 = (s**2).mean(axis=1)[:, None]
    beta_hat_inv2 = 1./beta_hat2
    k = s.shape[-1]
    nc = np.clip(norms/beta_hat2, 1e-1, None)

    if use_scipy:
        std_ppf = ncx2.ppf(q, k, nc)
    else:
        # Adapted from Sankaran, M. (1959). "On the non-central chi-squared distribution".
        h = 1-2./3*(k+nc)*(k+3*nc)/(k+2*nc)**2
        p = (k + 2*nc)/(k + nc)**2
        m = (h - 1)*(1 - 3*h)

        g1 = np.sqrt(2)*erfinv(2*q-1)*h*np.sqrt(2*p)*(1 + .5*m*p)
        g2 = 1 + h*p*(h-1-.5*(2-h)*m*p)
        g = g1 + g2

        std_ppf = g**(1/h) * (k + nc)

    return std_ppf * beta_hat2

def _folded_norm_ppf(q, t, m, s, use_scipy=True):
    q = np.atleast_2d(q)
    if np.ndim(m) < 2:
        m = np.atleast_2d(m).T
    if np.ndim(s) < 2:
        s = np.atleast_2d(s).T

    c = np.abs(m-t)/s

    if use_scipy:
        std_ppf = foldnorm.ppf(q, c)
    else:
        raise NotImplementedError()

    return std_ppf*s

if __name__ == "__main__":
    from scipy.stats import multivariate_normal as mvn
    import matplotlib.pyplot as plt
    import ipdb

    np.random.seed(10)

    k = 80
    n = 1000000
    mus = np.random.random(k)*8
    sigmas = np.random.random(k)*4
    X = mvn.rvs(mean=mus, cov=np.diag(sigmas**2), size=n+1)
    if k == 1:
        X = X[:, None]
    X, x_target = X[:-1], X[-1]*0.8

    D = ((X-x_target)**2).sum(axis=1)
    hist, edges = np.histogram(D, bins=100, density=True)
    bin_width = np.diff(edges).mean()
    r = edges[:-1]+bin_width/2

    logpdf = norm_logpdf(r, x_target, mus[None, :], (sigmas**2)[None, :],
                         norm_ord=2, root_norm=False).ravel()
    # logpdf = _ncx_logpdf(r, (mus-x_target)[None, :], sigmas[None, :]).ravel()
    # logpdf = _ncx2_logpdf(r, (mus-x_target)[None, :], sigmas[None, :]).ravel()
    fig, ax = plt.subplots()
    ax.bar(r, hist, width=bin_width)
    ax.plot(r, np.exp(logpdf), 'r')

    # ax.plot(r, pdf, 'g')
    ax.legend()

    plt.show()
