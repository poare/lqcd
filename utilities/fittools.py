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
import sympy as sp
from copy import *

# at some point, will want a class Term that represents a single term in a model.

class Model:

    def __init__(self, f, np, str_terms, coeff_names):
        """
        Initialize a model. f should be an input function which is a function of np parameters c_i, and
        outputs a function of the fitting parameter t. (TODO extend to multivariate).
        '_' is a placeholder.

        TODO modify how this works for non-polynomial models, i.e. an exponential fit.
        The code should work but the string representation might not.
        """
        self.n_params = np
        self.F = f
        if str_terms is None:
            self.str_terms = ['_' for ii in range(np)]
        elif type(str_terms) != list:
            self.str_terms = [str_terms]
        else:
            self.str_terms = str_terms
        if coeff_names is None:
            self.coeff_names = ['_' for ii in range(np)]
        elif type(coeff_names) != list:
            self.coeff_names = [coeff_names]
        else:
            self.coeff_names = coeff_names

    def get_symbolic_params(self):
        """
        Returns symbolic parameters c0, c1, ..., c_{n_params - 1} to use when differentiating
        the fit model.
        """
        sstring = ''
        for ii in range(self.n_params):
            sstring += 'c' + str(ii)
            if ii < self.n_params - 1:
                sstring += ' '
        if self.n_params == 1:
            return [sp.symbols(sstring)]         # make it a list
        return list(sp.symbols(sstring))

    def __add__(self, other):
        """
        Overloads the + operation for the Model class. The returned Model will have
        N + M = self.n_params + other.n_params parameters, the first N parameters of which
        will call self.F(...), and the remaining M parameters will call other.F(...)

        Parameters
        ----------
        self : Model
            Summand 1 for addition.
        other : Model
            Summand 2 for addition.

        Returns
        -------
        Model
            Sum of self and other.

        Examples
        --------
        def model1(params):
            return lambda x : params[0] * np.log(x) + params[1] * (x ** 2)
        M1 = Model(model1, 2)
        def model2(params):
            return lambda x : params[0] / x
        M2 = Model(model2, 1)
        M3 = model1 + model2
        M3.n_params             # 3
        M3([2, 3, 4])(1)        # 2 * np.log(1) + 3 * (1 ** 2) + 4 / 1

        def model_4(params):
            return lambda x : params[0] * np.log(x) + params[1] * (x ** 2) + params[2] / x
        M4 = Model(model_4, 3)
        M3 == M4                    # True
        """
        N, M = self.n_params, other.n_params
        def plus_fn(params):
            def model(x):
                return self.F(params[:N])(x) + other.F(params[N:])(x)
            return model
        s_terms, coeffs = self.str_terms.copy(), self.coeff_names.copy()
        s_terms.extend(other.str_terms)
        coeffs.extend(other.coeff_names)
        return Model(plus_fn, N + M, s_terms, coeffs)

    def __mul__(self, c):
        """
        Parameters
        ----------
        self : Model
            Model to scalar multiply.
        other : float, or float-type
            Scalar to multiply by.

        Returns
        -------
        Model
            Scalar multiple of model with c.

        Examples
        --------
        m = Model(lambda params : lambda x : params[0] * (x ** 2), 1)
        mm = m * 3.0
        mm.F([4])(0.3)                  # 1.08
        3.0 * 4 * (0.3 ** 2)            # 1.08
        """
        assert type(c) != Model, 'Can only scalar multiply.'
        def mul_fn(params):
            def model(x):
                return c * self.F(params)(x)
            return model
        s_terms = [str(c) + '*' + self.str_terms[ii] for ii in self.n_params]
        return Model(mul_fn, self.n_params, s_terms, self.coeff_names)

    def __rmul__(self, other):
        """Overload right multiplication as well."""
        return self.__mul__(other)

    def __repr__(self):
        s = ''
        for ii in range(self.n_params):
            if self.coeff_names[ii] == '_':
                s += 'c' + str(ii) + '*' + self.str_terms[ii]        # default coeff name
            else:
                s += self.coeff_names[ii] + '*' + self.str_terms[ii]
            if ii < self.n_params - 1:
                s += ' + '
        return s

    def __str__(self):
        """ tostring method. """
        return self.__repr__()

    @staticmethod
    def zero_model():
        return Model(lambda params : lambda t : 0.0, 0, '', coeff_names = [''])

    @staticmethod
    def const_model():
        """
        Initializes a constant fit model, y = c0. Should replicate power_model(0).
        """
        def const_fn(params):
            def model(t):
                return params
            return model
        m = Model(const_fn, 1, '(x^0)', coeff_names = ['c0'])
        return m

    @staticmethod
    def power_model(n):
        """
        Returns a power law fit model with the powers in n.

        Parameters
        ----------
        n : iterable, or int.
            Powers of x to use for the fit model. If an integer is passed in, uses all possible
            powers up to n.

        Returns
        -------
        Model object with the corersponding power law as its model function.

        Examples:
        power_model(0) : y = c0
        power_model(1) : y = c0 + c1 x
        power_model(2) : y = c0 + c1 x + c2 x^2
        """
        if type(n) == int:
            n = list(range(0, n + 1))
        n_params = len(n)
        def model_fn(params):
            assert len(params) == n_params
            def model(x):
                sum = 0.
                for ii, c in enumerate(params):
                    sum += c * (x ** n[ii])
                return sum
            return model
        m = Model(model_fn, n_params, str_terms = ['(x^' + str(nn) + ')' for nn in n])
        return m

class Fitter:
    """
    Base Fitter class from which all other fitters inhereit and implement instances of
    """

    def __init__(self):
        pass

    def set_model(self, m):
        self.model = m
        self.chi2 = self.get_chi2()
        self.chi2_sym = self.get_chi2_sym()

    def shrink_covar(self, lam):
        if lam == 0:
            self.corr = False
        self.covar = shrinkage(self.covar, lam)
        self.chi2 = self.get_chi2()
        self.chi2_sym = self.get_chi2_sym()

    def get_chi2(self):
        def chi2(params, data_f, cov_f):
            """
            Chi^2 goodness of fit for the given model. Assumes all input is
            numpy arrays.
            """
            dy = data_f - self.model.F(params)(self.fit_region)
            return np.einsum('i,ij,j->', dy, np.linalg.inv(cov_f), dy)# / 2.
        return chi2

    def get_chi2_sym(self):
        def chi2_sym(params, data_f, cov_f):
            """
            Chi^2 goodness of fit for the given model. Assumes that params is a
            list of sympy symbols, so that the covariance can be differentiated.
            Note that this limits the operations we can do.

            Parameters
            ----------
            params : [sp.symbol]
                Symbolic parameters to pass into chi^2 function.
            data_f : np.array [len(self.fit_region)]
                Data to pass into chi^2 function
            cov_f : np.array [len(self.fit_region)]
                Covariance between data points.
            """
            dy = sp.Matrix(data_f - self.model.F(params)(self.fit_region))
            inv_cov = sp.Matrix(np.linalg.inv(cov_f))
            chi2_mat = sp.Transpose(dy) * (inv_cov * dy)# / 2.
            return chi2_mat[0, 0]
        return chi2_sym

    def fit(self, params0 = None, disp_output = True):
        """
        Performs a fit to data self.cvs, with covariance matrix self.covar.

        Parameters
        ----------
        params0 : np.array
            Initial guess for chi^2 minimum to start solver at.

        Returns
        -------
        np.array (self.model.n_params)
            Central values for the best fit coefficients, of size self.model.n_params.
        float
            Minimum value of the chi^2.
        int
            Degrees of freedom for the fit, dof = len(self.fit_region) - self.model.n_params.
        np.array (self.model.n_params, self.model.n_params)
            Covariance matrix for the best fit coefficients.
        """
        if params0 is None:
            params0 = np.zeros((self.model.n_params), dtype = np.float64)
        if disp_output:
            print('Fitting data: ' + str(self.cvs) + ' at x positions: ' + str(self.fit_region))
        out = optimize.minimize(self.chi2, params0, args = (self.cvs, self.covar), method = 'Powell')
        params_fit = out['x']
        chi2_min = self.chi2(params_fit, self.cvs, self.covar)
        fit_covar = self.get_fit_covar(params_fit)
        ndof = len(self.fit_region) - self.model.n_params
        # return params_fit, chi2_min, ndof, fit_covar
        return [params_fit, chi2_min, ndof, fit_covar]

    def get_fit_covar(self, fit_params):
        """
        Gets the covariance matrix for the coefficients from a correlated fit.

        Parameters
        ----------
        fit_params : np.array [self.model.n_params]
            Best fit parameters for the fit. Should be the same size as self.cvs.

        Returns
        -------
        np.array [self.model.n_params, self.model.n_params]
            Covariance matrix for the best fit coefficients.
        """
        sym_params = self.model.get_symbolic_params()
        eval_pt = list(zip(sym_params, fit_params))         # evaluate derivative at p = fit_params
        cov_inv = np.zeros((self.model.n_params, self.model.n_params), dtype = np.float64)
        sym_chi2 = self.chi2_sym(sym_params, self.cvs, self.covar)
        for n in range(self.model.n_params):
            for m in range(self.model.n_params):
                Dnm = sp.diff(sym_chi2, sym_params[n], sym_params[m])
                cov_inv[n, m] = (1/2) * Dnm.subs(eval_pt)
        fit_covar = np.linalg.inv(cov_inv)# / 2.
        return fit_covar

    def gen_fit_band(self, fit_params, fit_covar, xrange):
        """
        Generates a fit band on domain x_range, based on the results from fout.

        Parameters
        ----------
        fit_params : np.array [self.model.n_params]
            Central values for the best fit coefficients.
        fit_covar : np.array [self.model.n_params, self.model.n_params]
            Covariance matrix for the best fit coefficients from the fit.
        x_range : np.array [npts]
            X-axis to generate data from

        Returns
        -------
        np.array [npts]
            Central values, generated point in xrange
        np.array [npts]
            Uncertainties in central values, generated at each point in xrange
        """
        fit_cvs = np.zeros(xrange.shape, dtype = np.float64)
        fit_sigmas = np.zeros(xrange.shape, dtype = np.float64)
        syms = self.model.get_symbolic_params()    # c0, c1, ..., c_{n - 1}
        eval_pt = list(zip(syms, fit_params))
        for ii, x in enumerate(xrange):
            F_sym = self.model.F(syms)(x)
            fit_cvs[ii] = self.model.F(fit_params)(x)
            dF = np.array([sp.diff(F_sym, p) for p in syms], dtype = np.float64)
            fit_sigmas[ii] = np.sqrt(dF.T @ fit_covar @ dF)
            # fit_sigmas[ii] = np.sqrt(dF.T @ fit_covar @ dF / 2.)
        return fit_cvs, fit_sigmas

class BootstrapFitter(Fitter):

    def __init__(self, fit_region, data, model, corr = True):
        """
        Class to perform fits to bootstrapped data. TODO think about how to implement this to multidimensional fits.

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
        self.cvs = np.mean(data, axis = 0)
        self.corr = corr
        if corr:
            self.covar = get_covariance(data)
        else:
            self.covar = np.diag(np.std(data, axis = 0, ddof = 1) ** 2)
        self.fit_region = fit_region
        self.fit_dims = len(fit_region.shape)    # dimensionality of fit data.
        self.set_model(model)

class CorrFitter(Fitter):

    def __init__(self, fit_region, cvs, cov, model):
        """
        Class to perform fits to correlated data, given the central values and the covariance matrix.

        Parameters
        ----------
        fit_region : np.array (npts)
            Data range to fit to. N_xi is the number of points in the ith dimension of the fit.
        cvs : np.array (npts)
            Central values for the fit.
        cov : np.array (npts, npts)
            Uncertainties for the fit.
        model : Model
            Fit model to use.
        corr : bool
            True for correlated fit and False for uncorrelated fit.

        Returns
        -------
        """
        self.cvs = cvs
        self.corr = True
        self.covar = cov
        self.fit_region = fit_region
        self.fit_dims = len(fit_region.shape)    # dimensionality of fit data.
        self.set_model(model)

class UncorrFitter(Fitter):

    def __init__(self, fit_region, cvs, sigmas, model):
        """
        Class to perform fits to uncorrelated data.

        Parameters
        ----------
        fit_region : np.array (N_x1, ..., N_xm)
            Data range to fit to. N_xi is the number of points in the ith dimension of the fit.
        cvs : np.array (N_x1, ..., N_xm)
            Central values for the fit.
        sigmas : np.array (N_x1, ..., N_xm)
            Uncertainties for the fit.
        model : Model
            Fit model to use.
        corr : bool
            True for correlated fit and False for uncorrelated fit.

        Returns
        -------
        """
        self.cvs = cvs
        self.corr = False
        self.covar = np.diag(sigmas ** 2)
        self.fit_region = fit_region
        self.fit_dims = len(fit_region.shape)    # dimensionality of fit data.
        self.set_model(model)

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
    Performs linear shrinkage on a covariance matrix. Returns the regulated covariance matrix,
    $$
        cov(x, y; λ) = λ*cov(x, y) + (1 - λ)*diag(cov)(x, y)
    $$
    where x and y individually index the domain of the data.
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

def process_fit_forms_AIC_best(fit_region, cvs, cov, form_list_full, base = Model.const_model(), A = 0.):#A = 0.5):
    """
    Uses the Akaike Information Criterion (AIC) to choose the optimal fit form for the data, out of the corresponding
    operands in form_list. At each level of the calculation, uses the best new fit form (as determined by the
    minimum chi^2 value) for the next level.

    Parameters
    ----------
    fit_region : np.array (n_pts)
        Domain to fit the data to
    cvs : np.array (n_pts)
        Central values to fit to.
    cov : np.array (n_pts, n_pts)
        Covariance matrix for the data to fit to.
    form_list_full : list (function)
        Array of functional forms to fit to. A functional form will be generated from the operands in
        form_list_full by taking a linear combination sum_i c_i form_list_full[i] over all possible combinations
        of the elements in form_list_full.
    base : Model
        The Model base will be kept in the functional form at all times. For example, for fitting Z(p^2) = Z0 + ...,
        Z0 should always be included in the fit form, so pin_idx = [Model.const_model()].
    A : float
        The proportionality constant for the AIC. Accepts a fit model G over a fit model F if and only if AIC(G) - AIC(F) < -A ndof(G).

    Returns
    -------
    """
    form_list = deepcopy(form_list_full)            # don't mutate form_list_full
    cur_model = base
    fitter = lambda m : CorrFitter(fit_region, cvs, cov, m).fit(disp_output = False)
    [cur_params, cur_chi2, cur_dof, cur_fit_covar] = fitter(cur_model)
    cur_AIC = AIC(cur_chi2, cur_model.n_params)
    while cur_dof > 1:
        print('Current summands to add to model: ' + str(form_list))
        print('Current degrees of freedom: ' + str(cur_dof))
        acc_forms = []
        for tmp_model_idx, form in enumerate(form_list):
            tmp_model = cur_model + form
            # (tmp_params, tmp_chi2, tmp_dof, tmp_fit_covar) = fitter(tmp_model)
            tmp_fit_out = fitter(tmp_model)
            tmp_chi2, tmp_dof = tmp_fit_out[1], tmp_fit_out[2]
            print('Trying model: ' + str(tmp_model) + '. chi^2 / ndof = ' + str(tmp_chi2 / tmp_dof))
            tmp_AIC = AIC(tmp_chi2, tmp_model.n_params)
            tmp_fit_out.extend([tmp_AIC, tmp_model, tmp_model_idx])
            if tmp_AIC - cur_AIC < - A * tmp_dof:                       # then it satisfies the AIC
                print('Accepted model: ' + str(tmp_model))
                acc_forms.append(tmp_fit_out)                           # also store other fit results
        if len(acc_forms) == 0:                                         # then we're done
            break
        best_data = min(acc_forms, key = lambda f : f[1])               # organize by minimum chi^2
        [cur_params, cur_chi2, cur_dof, cur_fit_covar, cur_AIC, cur_model, model_idx] = best_data
        del form_list[model_idx]
        print('Using model ' + str(cur_model) + ' for next iteration.')
    print('Best fit form is: ' + str(cur_model))
    print('Best fit parameters: ' + str(cur_params))
    print('Parameter covariance: ' + str(cur_fit_covar))
    print('Chi^2 / ndof: ' + str(cur_chi2 / cur_dof))
    return cur_params, cur_chi2, cur_dof, cur_fit_covar, cur_model

def powerset(L, min_length = 0):
    """Constructs the power set of L."""
    return list(itertools.chain.from_iterable(itertools.combinations(L, r) for r in range(min_length, len(L) + 1)))

def process_fit_forms_AIC_all(fit_region, cvs, cov, form_list_full, base = Model.const_model()):
    """
    Uses the Akaike Information Criterion (AIC) to choose the optimal fit form for the data, out of the corresponding
    operands in form_list. Iterates over all possible forms that can be constructed from form_list and returns the one
    which minimizes the AIC.

    Parameters
    ----------
    fit_region : np.array (n_pts)
        Domain to fit the data to
    cvs : np.array (n_pts)
        Central values to fit to.
    cov : np.array (n_pts, n_pts)
        Covariance matrix for the data to fit to.
    form_list_full : list (function)
        Array of functional forms to fit to. A functional form will be generated from the operands in
        form_list by taking a linear combination sum_i c_i form_list[i] over all possible combinations
        of the elements in form_list.
    base : list [Model]
        All Models in base will be kept in the functional form at all times. For example, for fitting Z(p^2) = Z0 + ...,
        Z0 should always be included in the fit form, so pin_idx = [Model.const_model()].
    """
    form_list = deepcopy(form_list_full)            # don't mutate form_list_full
    all_models = []
    pwr_set = powerset(form_list, min_length = 1)
    for subset in pwr_set:
        if len(subset) + base.n_params >= len(fit_region):          # too many parameters
            continue
        sum_form = deepcopy(base)
        for summand in subset:
            sum_form += summand
        all_models.append(sum_form)
    # print(len(all_models))
    # print(all_models)
    fitter = lambda m : CorrFitter(fit_region, cvs, cov, m).fit(disp_output = False)
    print('Fitting base model = ' + str(base))
    best_data = fitter(base)
    best_data.extend([base])
    best_AIC = AIC(best_data[1], base.n_params)
    best_chi2_ndof = best_data[1] / best_data[2]
    for model in all_models:
        tmp_data = fitter(model)
        tmp_data.extend([model])
        tmp_AIC = AIC(tmp_data[1], model.n_params)
        tmp_chi2_ndof = tmp_data[1] / tmp_data[2]
        # if tmp_chi2_ndof < best_chi2_ndof:          # minimize chi2 / ndof
        if tmp_AIC < best_AIC:                    # minimize AIC
            print('Using model ' + str(model))                                  # then this is the best fit so far
            best_data = tmp_data
            best_AIC = tmp_AIC
            best_chi2_ndof = tmp_chi2_ndof
    [cur_params, cur_chi2, cur_dof, cur_fit_covar, cur_model] = best_data
    print('Best fit form is: ' + str(cur_model))
    print('Best fit parameters: ' + str(cur_params))
    print('Parameter covariance: ' + str(cur_fit_covar))
    print('Chi^2 / ndof: ' + str(cur_chi2 / cur_dof))
    return cur_params, cur_chi2, cur_dof, cur_fit_covar, cur_model

def process_fit_forms_all(fit_region, cvs, cov, form_list_full, base = Model.const_model(), max_chi2_dof = 50):
    """
    Fits all possible fit forms to the data. Iterates over all possible forms that can be constructed from form_list and
    returns the fit value and error for each one. Returns all fits with chi^2/ndof < max_chi2_dof

    Parameters
    ----------
    fit_region : np.array (n_pts)
        Domain to fit the data to
    cvs : np.array (n_pts)
        Central values to fit to.
    cov : np.array (n_pts, n_pts)
        Covariance matrix for the data to fit to.
    form_list_full : list (function)
        Array of functional forms to fit to. A functional form will be generated from the operands in
        form_list by taking a linear combination sum_i c_i form_list[i] over all possible combinations
        of the elements in form_list.
    base : list [Model]
        All Models in base will be kept in the functional form at all times. For example, for fitting Z(p^2) = Z0 + ...,
        Z0 should always be included in the fit form, so pin_idx = [Model.const_model()].
    """
    form_list = deepcopy(form_list_full)            # don't mutate form_list_full
    all_forms = []
    pwr_set = powerset(form_list)
    for subset in pwr_set:
        if len(subset) + base.n_params >= len(fit_region):          # too many parameters
            continue
        sum_form = deepcopy(base)
        for summand in subset:
            sum_form += summand
        all_forms.append(sum_form)
    fitter = lambda m : CorrFitter(fit_region, cvs, cov, m).fit(disp_output = False)
    all_params, all_chi2, all_dof, all_fit_covar, all_models = [], [], [], [], []
    for model in all_forms:
        tmp_data = fitter(model)
        tmp_data.extend([model])
        tmp_AIC = AIC(tmp_data[1], model.n_params)
        tmp_chi2_ndof = tmp_data[1] / tmp_data[2]
        # if tmp_chi2_ndof < best_chi2_ndof:          # minimize chi2 / ndof
        if tmp_chi2_ndof < max_chi2_dof:                    # minimize AIC
            print('Adding model ' + str(model))                                  # then this is the best fit so far
            all_params.append(tmp_data[0])
            all_chi2.append(tmp_data[1])
            all_dof.append(tmp_data[2])
            all_fit_covar.append(tmp_data[3])
            all_models.append(tmp_data[4])
    print('Number of fits with chi^2 / dof < ' + str(max_chi2_dof) + ': ' + str(len(all_params)))
    return all_params, np.array(all_chi2), np.array(all_dof), all_fit_covar, all_models

def AIC(chi2, nparams):
    """
    Returns the Akaike Information Criterion (AIC) for a given fit with total chi^2 chi2
    (not this is **not** the chi^2 / dof, it is the full chi^2 for the fit).

    Parameters
    ----------
    chi2 : float
        Chi^2 value of the fit.
    nparams : int
        Degrees of freedom of the fit.

    Returns
    -------
    float
        AIC for the fit.
    """
    return 2 * nparams + chi2
