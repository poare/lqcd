################################################################################
# This script saves a list of common utility functions for data fitting        #
# that I may need to use in my python code. To import the script, simply add   #
# the path with sys before importing. For example, if lqcd/ is in              #
# /Users/theoares:                                                             #
#                                                                              #
# import sys                                                                   #
# sys.path.append('/Users/theoares/lqcd/utilities')                            #
# from fittools import *                                                        #
################################################################################

from __main__ import *
n_boot = n_boot

import numpy as np
import h5py
import os
from scipy.special import zeta
import time
import re
import itertools
import io
import random
from scipy import optimize
from scipy.stats import chi2

class Model:

    def __init__(self, f, np):
        """
        Initialize a model. f should be an input function which is a function of np parameters c_i, and
        outputs a function of the fitting parameter t. (TODO extend to multivariate).
        """
        self.n_params = np
        self.F = f

    @staticmethod
    def const_model():
        """
        Initializes a constant fit model, y = c0. Should replicate quadratic_model(0).
        """
        def const_fn(params):
            def model(t):
                return params
            return model
        m = Model(const_fn, 1)
        return m

    @staticmethod
    def power_model(n):
        """
        Returns a power law fit model to the nth power in x, where x is a scalar input.

        Examples:
        quadratic_model(0) : y = c0
        quadratic_model(1) : y = c0 + c1 x
        quadratic_model(2) : y = c0 + c1 x + c2 x^2
        """
        def model_fn(params):
            assert len(params) == n + 1
            def model(x):
                sum = 0.
                for ii, c in enumerate(params):
                    sum += c * (x ** ii)
                return sum
            return model
        m = Model(model_fn, n + 1)
        return m


class BootstrapFitter:

    def __init__(self, fit_region, data, model, corr = True):
        """
        Class to perform fits to data. TODO think about how to implement this to multidimensional fits.

        Parameters
        ----------
        fit_region : np.array (N_x1, ..., N_xm)
            Data range to fit to. N_xi is the number of points in the ith dimension of the fit.
        data : np.array (n_boot, N_x1, ..., N_xm)
            Data to fit to. The first fit_dims dimensions should have this shape
        model : Model
            Fit model to use.
        corr : bool
            True for correlated fit and False for uncorrelated fit.

        Returns
        -------
        """
        self.data = data
        self.mean = np.mean(data, axis = 0)
        self.corr = corr
        if corr:
            self.covar = get_covariance(data)
        else:
            self.covar = np.diag(np.std(data, axis = 0, ddof = 1) ** 2)
        self.fit_region = fit_region
        self.fit_dims = len(fit_region.shape)    # dimensionality of fit data.
        self.model = model
        self.chi2 = self.get_chi2()

    def set_model(self, m):
        self.model = m
        self.chi2 = self.get_chi2()

    def shrink_covar(self, lam):
        if lam == 0:
            self.corr = False
        self.covar = shrinkage(self.covar, lam)
        self.chi2 = self.get_chi2()

    def get_chi2(self):
        def chi2(params, data_f, cov_f):
            """
            Chi^2 goodness of fit for the given model.
            """
            dy = data_f - self.model.F(params)(self.fit_region)
            return np.einsum('i,ij,j->', dy, np.linalg.inv(cov_f), dy)
        return chi2

    def fit(self, params0 = None):
        if params0 is None:
            params0 = np.zeros((self.model.n_params), dtype = np.float64)
        print('Fitting data: ' + str(self.mean) + ' at x positions: ' + str(self.fit_region))
        out = optimize.minimize(self.chi2, params0, args = (self.mean, self.covar), method = 'Powell')
        params_fit = out['x']
        chi2_min = self.chi2(params_fit, self.mean, self.covar)
        ndof = len(self.fit_region) - 1
        return params_fit, chi2_min, ndof

# TODO don't need functions like this since it reduces to a correlated fit with a diagonal covariance.

# # data should be an array of size (n_fits, T) and fit_region gives the times to fit at
# def fit_constant_uncorr(fit_region, data, nfits = n_boot):
#     if type(fit_region) != np.ndarray:
#         fit_region = np.array([x for x in fit_region])
#     if len(data.shape) == 1:        # if data only has one dimension, add an axis
#         data = np.expand_dims(data, axis = 0)
#     sigma_fit = np.std(data[:, fit_region], axis = 0)
#     c_fit = np.zeros((nfits), dtype = np.float64)
#     chi2 = lambda x, data, sigma : np.sum((data - x[0]) ** 2 / (sigma ** 2))     # x[0] = constant to fit to
#     for i in range(nfits):
#         data_fit = data[i, fit_region]
#         x0 = [1]          # guess to start at
#         out = optimize.minimize(chi2, x0, args=(data_fit, sigma_fit), method = 'Powell')
#         c_fit[i] = out['x']
#         # cov_{ij} = 1/2 * D_i D_j chi^2
#     # return the total chi^2 and dof for the fit. Get chi^2 by using mean values for all the fits.
#     c_mu = np.mean(c_fit)
#     data_mu = np.mean(data, axis = 0)
#     chi2_mu = chi2([c_mu], data_mu[fit_region], sigma_fit)
#     ndof = len(fit_region) - 1    # since we're just fitting a constant, n_params = 1
#     return c_fit, chi2_mu, ndof

# # data should be an array of size (n_fits, T). Fits over every range with size >= TT_min and weights
# # by p value of the fit. cut is the pvalue to cut at.
# def fit_constant_allrange(data, TT_min = 4, cut = 0.01):
#     TT = data.shape[1]
#     fit_ranges = []
#     for t1 in range(TT):
#         for t2 in range(t1 + TT_min, TT):
#             fit_ranges.append(range(t1, t2))
#     f_acc = []        # for each accepted fit, store [fidx, fit_region]
#     stats_acc = []    # for each accepted fit, store [pf, chi2, ndof]
#     meff_acc = []     # for each accepted fit, store [meff_f, meff_sigma_f]
#     weights = []
#     print('Accepted fits\nfit index | fit range | p value | meff mean | meff sigma | weight ')
#     for f, fit_region in enumerate(fit_ranges):
#         meff_ens_f, chi2_f, ndof_f = fit_constant(fit_region, data)
#         pf = chi2.sf(chi2_f, ndof_f)
#         if pf > cut:
#             # TODO change so that we store m_eff_mu as an ensemble, want to compute m_eff_bar ensemble
#             meff_mu_f = np.mean(meff_ens_f)
#             meff_sigma_f = np.std(meff_ens_f, ddof = 1)
#             weight_f = pf * (meff_sigma_f ** (-2))
#             print(f, fit_region, pf, meff_mu_f, meff_sigma_f, weight_f)
#             f_acc.append([f, fit_region])
#             stats_acc.append([pf, chi2_f, ndof_f])
#             # meff_acc.append([meff_mu_f, meff_sigma_f])
#             meff_acc.append(meff_ens_f)
#             weights.append(weight_f)
#     print('Number of accepted fits: ' + str(len(f_acc)))
#     weights, meff_acc, stats_acc = np.array(weights), np.array(meff_acc), np.array(stats_acc)
#     # weights = weights / np.sum(weights)    # normalize to 1
#     return f_acc, stats_acc, meff_acc, weights

def get_covariance(R, dof = 1):
    """
    Returns the covariance matrix for correlator data {R_b(t)}. Assumes that R comes in the shape
    (n_boot, n_t), where n_t is some number of time slices to compute the covariance over. The
    covariance for data of this form is defined as:
    $$
        Cov(t_1, t_2) = \frac{1}{n_b - dof} \sum_b (R_b(t_1) - \overline{R}(t_1)) (R_b(t_2) - \overline{R}(t_2))
    $$

    Parameters
    ----------
    R : np.array (n_boot, n_t)
        Correlator data to fit plateau to. The first index should be bootstrap, and the second should be time.
    dof : int
        Number of degrees of freedom to compute the variance with. dof = 1 is the default and gives an unbiased
        estimator of the population from a sample.

    Returns
    -------
    np.array (n_t, n_t)
        Covariance matrix from data.
    """
    nb = R.shape[0]
    mu = np.mean(R, axis = 0)
    cov = np.einsum('bi,bj->ij', R - mu, R - mu) / (nb - dof)
    assert np.max(cov - np.cov(R.T, ddof = 1)) < 1e-10
    return cov

def shrinkage(covar, lam):
    """
    Performs linear shrinkage on a covariance matrix. Returns the regulated covariance matrix
    """
    uncorr_covar = np.zeros(covar.shape, dtype = covar.dtype)
    for i in range(covar.shape[0]):
        uncorr_covar[i, i] = covar[i, i]
    shrunk_cov = lam * covar + (1 - lam) * uncorr_covar
    return shrunk_cov

def fit_const(fit_region, data, cov):
    """
    data should be either an individual bootstrap or the averages \overline{R}(t), i.e. a vector
    of size n_pts, where n_pts = len(fit_range). cov should have shape (n_pts, n_pts) (so it should already be
    restricted to the fit range). This function is what fit_model reduces to when passed a constant model.

    Parameters
    ----------
    fit_region : np.array (n_pts)
        x axis data for the fit.
    data : np.array (n_pts)
        y axis data for the fit.
    cov : np.array (n_pts, n_pts)
        Covariance matrix for the fit. For an uncorrelated fit, this should be diagonal.

    Returns
    -------
    np.float64
        Best fit coefficient c0.
    np.float64
        Minimum value of the chi^2 for the fit.
    int
        Degrees of freedom for the fit.
    """
    if type(fit_region) != np.ndarray:
        fit_region = np.array([x for x in fit_region])
    def chi2(params, data_f, cov_f):
        return np.einsum('i,ij,j->', data_f - params[0], np.linalg.inv(cov_f), data_f - params[0])
    params0 = [1.]
    out = optimize.minimize(chi2, params0, args = (data, cov), method = 'Powell')
    c_fit = out['x'][()]
    chi2_min = chi2([c_fit], data, cov)
    ndof = len(fit_region) - 1    # can change this to more parameters later
    return c_fit, chi2_min, ndof

def fit_const_allrange(data, corr = True, TT_min = 4, cut = 0.01):
    """
    Performs a correlated fit over every range with size >= TT_min and weights by p value of the fit.
    Fit is performed to the mean of the data first, and if the p value of this fit is > cut the fit is
    accepted. A correlated fit is then performed on the bootstraps of each accepted fit to quantify the
    fitting error, and the accepted fit data is combined in a weighted average. Here n_acc is the
    number of accepted fits.

    Parameters
    ----------
    data : np.array (n_boot, T)
        Correlator data to fit plateau to. The first index should be bootstrap, and the second should be time.
    corr : bool
        True if data is correlated, False if data is uncorrelated.
    TT_min : int
        Minimum size to fit plateau to.
    cut : double
        Cut on p values. Only accept fits with pf > cut.

    Returns
    -------
    [n_acc, 2]
        For each accepted fit, stores: [fit_index, fit_range]
    np.array (n_acc, 3)
        For each accepted fit, stores: [p value, chi2, ndof]
    np.array (n_acc, nb)
        For each accepted fit, stores the ensemble of fit parameters from fitting each individual bootstrap.
    np.array (n_acc, nb)
        For each accepted fit, stores the ensemble of chi2 values from fitting each individual bootstrap.
    np.array ()
        For each accepted fit, stores the weight w_f.
    """
    nb, TT = data.shape[0], data.shape[1]
    fit_ranges = []
    for t1 in range(TT):
        for t2 in range(t1 + TT_min, TT):
            # fit_ranges.append(np.array([x for x in range(t1, t2)]))
            fit_ranges.append(range(t1, t2))
    f_acc = []        # for each accepted fit, store [fidx, fit_region]
    stats_acc = []    # for each accepted fit, store [pf, chi2, ndof]
    c_ens_acc = []    # for each accepted fit, store ensemble of best fit coefficients c
    chi2_full = []    # for each accepted fit, store ensemble of chi2
    weights = []
    data_mu = np.mean(data, axis = 0)
    if corr:
        cov = get_covariance(data)
    else:
        cov = np.zeros((TT, TT))   # check using diagonal covariance matrix
        for t in range(TT):
           cov[t, t] = np.std(data[:, t], ddof = 1) ** 2
    print('Accepted fits:\nfit index | fit range | p value | c_fit mean | c_fit sigma | weight ')
    for f, fit_region in enumerate(fit_ranges):
        cov_sub = cov[np.ix_(fit_region, fit_region)]
        c_fit, chi2_fit, ndof = fit_const(fit_region, data_mu[fit_region], cov_sub)
        pf = chi2.sf(chi2_fit, ndof)
        if pf > cut:    # then accept the fit and fit each individual bootstrap
            f_acc.append([f, fit_region])
            stats_acc.append([pf, chi2_fit, ndof])
            c_ens, chi2_ens = np.zeros((nb), dtype = np.float64), np.zeros((nb), dtype = np.float64)
            for b in range(nb):
                c_ens[b], chi2_ens[b], _ = fit_const(fit_region, data[b, fit_region], cov_sub)
            c_ens_acc.append(c_ens)
            chi2_full.append(chi2_ens)
            c_ens_mu = np.mean(c_ens)
            c_ens_sigma = np.std(c_ens, ddof = 1)
            weight_f = pf * (c_ens_sigma ** (-2))
            weights.append(weight_f)
            print(f, fit_region, pf, c_ens_mu, c_ens_sigma, weight_f)
    print('Number of accepted fits: ' + str(len(f_acc)))
    weights, c_ens_acc, chi2_full, stats_acc = np.array(weights), np.array(c_ens_acc), np.array(chi2_full), np.array(stats_acc)
    return f_acc, stats_acc, c_ens_acc, chi2_full, weights
