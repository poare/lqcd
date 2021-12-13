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
        def const_fn(c):
            def model(t):
                return c
            return model
        m = Model(const_fn, 1)
        return m


class Fitter:

    def __init__(self, fit_range, data, corr):
        """
        Class to perform fits to data. TODO think about how to implement this to multidimensional fits.

        Parameters
        ----------
        fit_range : np.array
            Data range to fit to. For something like a plateau fit in t, should be 1D, but can also be
            higher dimensional.
        data : np.array
            Data to fit to. The first fit_dims dimensions should have this shape
        corr : bool
            True for correlated fit and False for uncorrelated fit.

        Returns
        -------
        """
        self.data = data
        self.corr = corr
        if corr:
            self.covar = get_covariance(data)
        self.fit_range = fit_range
        self.fit_dims = len(fit_range.shape)    # dimensionality of fit data.
        assert self.data.shape[:fit_dims] == self.fit_range.shape
        return

    def get_chisq(self, model):
        def chi2(params):
            """
            params should be the same size as model, so that model.F(params) outputs a function on the space
            of fit data.
            """
            assert model.n_params == len(params)
            # TODO implement
            if self.corr:
                s
            else:
                s
        return chi2

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

# data should be an array of size (n_fits, T) and fit_region gives the times to fit at
def fit_constant(fit_region, data, nfits = n_boot):
    if type(fit_region) != np.ndarray:
        fit_region = np.array([x for x in fit_region])
    if len(data.shape) == 1:        # if data only has one dimension, add an axis
        data = np.expand_dims(data, axis = 0)
    sigma_fit = np.std(data[:, fit_region], axis = 0)
    c_fit = np.zeros((nfits), dtype = np.float64)
    chi2 = lambda x, data, sigma : np.sum((data - x[0]) ** 2 / (sigma ** 2))     # x[0] = constant to fit to
    for i in range(nfits):
        data_fit = data[i, fit_region]
        x0 = [1]          # guess to start at
        out = optimize.minimize(chi2, x0, args=(data_fit, sigma_fit), method = 'Powell')
        c_fit[i] = out['x']
        # cov_{ij} = 1/2 * D_i D_j chi^2
    # return the total chi^2 and dof for the fit. Get chi^2 by using mean values for all the fits.
    c_mu = np.mean(c_fit)
    data_mu = np.mean(data, axis = 0)
    chi2_mu = chi2([c_mu], data_mu[fit_region], sigma_fit)
    ndof = len(fit_region) - 1    # since we're just fitting a constant, n_params = 1
    return c_fit, chi2_mu, ndof

# data should be an array of size (n_fits, T). Fits over every range with size >= TT_min and weights
# by p value of the fit. cut is the pvalue to cut at.
def fit_constant_allrange(data, TT_min = 4, cut = 0.01):
    TT = data.shape[1]
    fit_ranges = []
    for t1 in range(TT):
        for t2 in range(t1 + TT_min, TT):
            fit_ranges.append(range(t1, t2))
    f_acc = []        # for each accepted fit, store [fidx, fit_region]
    stats_acc = []    # for each accepted fit, store [pf, chi2, ndof]
    meff_acc = []     # for each accepted fit, store [meff_f, meff_sigma_f]
    weights = []
    print('Accepted fits\nfit index | fit range | p value | meff mean | meff sigma | weight ')
    for f, fit_region in enumerate(fit_ranges):
        meff_ens_f, chi2_f, ndof_f = fit_constant(fit_region, data)
        pf = chi2.sf(chi2_f, ndof_f)
        if pf > cut:
            # TODO change so that we store m_eff_mu as an ensemble, want to compute m_eff_bar ensemble
            meff_mu_f = np.mean(meff_ens_f)
            meff_sigma_f = np.std(meff_ens_f, ddof = 1)
            weight_f = pf * (meff_sigma_f ** (-2))
            print(f, fit_region, pf, meff_mu_f, meff_sigma_f, weight_f)
            f_acc.append([f, fit_region])
            stats_acc.append([pf, chi2_f, ndof_f])
            # meff_acc.append([meff_mu_f, meff_sigma_f])
            meff_acc.append(meff_ens_f)
            weights.append(weight_f)
    print('Number of accepted fits: ' + str(len(f_acc)))
    weights, meff_acc, stats_acc = np.array(weights), np.array(meff_acc), np.array(stats_acc)
    # weights = weights / np.sum(weights)    # normalize to 1
    return f_acc, stats_acc, meff_acc, weights
