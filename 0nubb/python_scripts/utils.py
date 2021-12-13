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

# STANDARD BOOTSTRAPPED PROPAGATOR ARRAY FORM: [b, cfg, c, s, c, s] where:
  # b is the boostrap index
  # cfg is the configuration index
  # c is a color index
  # s is a spinor index

Nc = 3
Nd = 4
d = 4

g = np.diag([1, 1, 1, 1])
hbarc = .197327

delta = np.identity(Nd, dtype = np.complex64)
gamma = np.zeros((d, Nd, Nd),dtype=complex)
gamma[0] = gamma[0] + np.array([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]])
gamma[1] = gamma[1] + np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
gamma[2] = gamma[2] + np.array([[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]])
gamma[3] = gamma[3] + np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
gamma5 = np.dot(np.dot(np.dot(gamma[0], gamma[1]), gamma[2]), gamma[3])

# gammaListRev = [np.identity(4,dtype=complex), gamma[0], gamma[1], np.matmul(gamma[1], gamma[0]), gamma[2], \
#     np.matmul(gamma[2], gamma[0]), np.matmul(gamma[2], gamma[1]), np.matmul(gamma[2], np.matmul(gamma[1],gamma[0])), \
#     gamma[3], np.matmul(gamma[3], gamma[0]), np.matmul(gamma[3], gamma[1]), np.matmul(gamma[3], np.matmul(gamma[1], gamma[0])), \
#     np.matmul(gamma[3], gamma[2]), np.matmul(gamma[3], np.matmul(gamma[2], gamma[0])), np.matmul(gamma[3], np.matmul(gamma[2], gamma[1])), \
#     np.matmul(np.matmul(gamma[3], gamma[2]), np.matmul(gamma[1], gamma[0]))]
gammaList = [np.identity(4,dtype=complex), gamma[0], gamma[1], np.matmul(gamma[0], gamma[1]), gamma[2], \
    np.matmul(gamma[0], gamma[2]), np.matmul(gamma[1], gamma[2]), np.matmul(gamma[0], np.matmul(gamma[1],gamma[2])), \
    gamma[3], np.matmul(gamma[0], gamma[3]), np.matmul(gamma[1], gamma[3]), np.matmul(gamma[0], np.matmul(gamma[1], gamma[3])), \
    np.matmul(gamma[2], gamma[3]), np.matmul(gamma[0], np.matmul(gamma[2], gamma[3])), np.matmul(gamma[1], np.matmul(gamma[2], gamma[3])), \
    np.matmul(np.matmul(gamma[0], gamma[1]), np.matmul(gamma[2], gamma[3]))]

bvec = [0, 0, 0, 0.5]

# initialize Dirac matrices
gammaMu5 = np.array([gamma[mu] @ gamma5 for mu in range(d)])
sigmaD = np.zeros((Nd, Nd, Nd, Nd), dtype = np.complex64)               # sigma_{mu nu}
gammaGamma = np.zeros((Nd, Nd, Nd, Nd), dtype = np.complex64)           # gamma_mu gamma_nu
for mu in range(d):
    for nu in range(mu + 1, d):
        sigmaD[mu, nu, :, :] = (1 / 2) * (gamma[mu] @ gamma[nu] - gamma[nu] @ gamma[mu])
        sigmaD[nu, mu, :, :] = -sigmaD[mu, nu, :, :]
        gammaGamma[mu, nu, :, :] = gamma[mu] @ gamma[nu]
        gammaGamma[nu, mu, :, :] = - gammaGamma[mu, nu, :, :]

# tree level structures for the operators
tree = np.zeros((5, Nc, Nd, Nc, Nd, Nc, Nd, Nc, Nd), dtype = np.complex64)
for a, b in itertools.product(range(Nc), repeat = 2):
    for alpha, beta, gam, sigma in itertools.product(range(Nd), repeat = 4):
        tree[2, a, alpha, a, beta, b, gam, b, sigma] += 2 * (delta[alpha, beta] * delta[gam, sigma] - gamma5[alpha, beta] * gamma5[gam, sigma])
        tree[2, a, alpha, b, beta, b, gam, a, sigma] -= 2 * (delta[alpha, sigma] * delta[gam, beta] - gamma5[alpha, sigma] * gamma5[gam, beta])
        tree[3, a, alpha, a, beta, b, gam, b, sigma] += 2 * (delta[alpha, beta] * delta[gam, sigma] + gamma5[alpha, beta] * gamma5[gam, sigma])
        tree[3, a, alpha, b, beta, b, gam, a, sigma] -= 2 * (delta[alpha, sigma] * delta[gam, beta] + gamma5[alpha, sigma] * gamma5[gam, beta])
        for mu in range(d):
            tree[0, a, alpha, a, beta, b, gam, b, sigma] += 2 * (gamma[mu, alpha, beta] * gamma[mu, gam, sigma] + gammaMu5[mu, alpha, beta] * gammaMu5[mu, gam, sigma])
            tree[0, a, alpha, b, beta, b, gam, a, sigma] -= 2 * (gamma[mu, alpha, sigma] * gamma[mu, gam, beta] + gammaMu5[mu, alpha, sigma] * gammaMu5[mu, gam, beta])
            tree[1, a, alpha, a, beta, b, gam, b, sigma] += 2 * (gamma[mu, alpha, beta] * gamma[mu, gam, sigma] - gammaMu5[mu, alpha, beta] * gammaMu5[mu, gam, sigma])
            tree[1, a, alpha, b, beta, b, gam, a, sigma] -= 2 * (gamma[mu, alpha, sigma] * gamma[mu, gam, beta] - gammaMu5[mu, alpha, sigma] * gammaMu5[mu, gam, beta])
            for nu in range(mu + 1, d):
                tree[4, a, alpha, a, beta, b, gam, b, sigma] += 2 * (gammaGamma[mu, nu, alpha, beta] * gammaGamma[mu, nu, gam, sigma])
                tree[4, a, alpha, b, beta, b, gam, a, sigma] -= 2 * (gammaGamma[mu, nu, alpha, sigma] * gammaGamma[mu, nu, gam, beta])

# Saves the dimensions of a lattice.
class Lattice:
    def __init__(self, l, t):
        self.L = l
        self.T = t
        self.LL = [l, l, l, t]
        self.vol = (l ** 3) * t

    def to_linear_momentum(self, k, datatype = np.complex64):
        # return np.array([np.complex64(2 * np.pi * k[mu] / self.LL[mu]) for mu in range(4)])
        return np.array([datatype(2 * np.pi * k[mu] / self.LL[mu]) for mu in range(4)])

    def to_lattice_momentum(self, k):
        return np.array([np.complex64(2 * np.sin(np.pi * k[mu] / self.LL[mu])) for mu in range(4)])
    # Converts a wavevector to an energy scale using ptwid. Lattice parameter is a = A femtometers.
    # Shouldn't use this-- should use k_to_mu_p instead and convert at p^2 = mu^2
    def k_to_mu_ptwid(self, k, A = .1167):
        aGeV = fm_to_GeV(A)
        return 2 / aGeV * np.sqrt(sum([np.sin(np.pi * k[mu] / self.LL[mu]) ** 2 for mu in range(4)]))

    def k_to_mu_p(self, k, A = .1167):
        aGeV = fm_to_GeV(A)
        return (2 * np.pi / aGeV) * np.sqrt(sum([(k[mu] / self.LL[mu]) ** 2 for mu in range(4)]))

n_boot = 50
# n_boot = 100

def kstring_to_list(pstring, str):
    def get_momenta(x):
        lst = []
        mult = 1
        for digit in x:
            if digit == '-':
                mult *= -1
            else:
                lst.append(mult * int(digit))
                mult = 1
        return lst
    return get_momenta(pstring.split(str)[1])

# str is the beginning of the string, ex klist_to_string([1, 2, 3, 4], 'k1') gives 'k1_1234'
def klist_to_string(k, prefix):
    return prefix + str(k[0]) + str(k[1]) + str(k[2]) + str(k[3])

# squares a 4 vector.
def square(k):
    return np.dot(k, np.dot(g, k.T))

def slash(k):
    return sum([k[mu] * gamma[mu] for mu in range(4)])

def norm(p):
    return np.sqrt(np.abs(square(p)))

# cfgs should contain a list of paths to hdf5 files. If op_renorm is True, then will also
# read in 3 point functions for operator renormalization; if it is False, then
# will only read in 3 point functions for quark / vector / axial renormalization,
# and will return an empty list for G0
def readfiles(cfgs, q, op_renorm = True):
    props_k1 = np.zeros((len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    props_k2 = np.zeros((len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    props_q = np.zeros((len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    GV = np.zeros((4, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    GA = np.zeros((4, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    GO = np.zeros((16, len(cfgs), 3, 4, 3, 4, 3, 4, 3, 4), dtype = np.complex64)

    for idx, file in enumerate(cfgs):
        f = h5py.File(file, 'r')
        qstr = klist_to_string(q, 'q')
        if idx == 0:            # just choose a specific config to get these on, since they should be the same
            k1 = f['moms/' + qstr + '/k1'][()]
            k2 = f['moms/' + qstr + '/k2'][()]
        props_k1[idx] = np.einsum('ijab->aibj', f['prop_k1/' + qstr][()])
        props_k2[idx] = np.einsum('ijab->aibj', f['prop_k2/' + qstr][()])
        props_q[idx] = np.einsum('ijab->aibj', f['prop_q/' + qstr][()])
        for mu in range(4):
            GV[mu, idx] = np.einsum('ijab->aibj', f['GV' + str(mu + 1) + '/' + qstr][()])
            GA[mu, idx] = np.einsum('ijab->aibj', f['GA' + str(mu + 1) + '/' + qstr][()])
        if op_renorm:
            for n in range(16):
                # TODO be careful about this with the chroma input
                GO[n, idx] = np.einsum('ijklabcd->aibjckdl', f['Gn' + str(n) + '/' + qstr][()])      # for chroma
                # GO[n, idx] = np.einsum('ijklabcd->dlckbjai', f['Gn' + str(n) + '/' + qstr][()])    # for qlua
    return k1, k2, props_k1, props_k2, props_q, GV, GA, GO

# Read an Npt function in David's format
def read_Npt(folder, stem, fnums, N, n_t, start_idx = 2):
    files = [folder + '/' + stem + '.' + str(fnum) for fnum in fnums]
    if N == 2:
        Cnpt = np.zeros((len(fnums), n_t, n_t), dtype = np.complex64)
    else:
        Cnpt = np.zeros((len(fnums), 24, n_t, n_t, n_t), dtype = np.complex64)   # for raw readout without operator structures
        # Cnpt = np.zeros((len(fnums), 5, n_t, n_t, n_t), dtype = np.complex64)   # for non time-averaged data
        # Cnpt = np.zeros((len(fnums), 5, n_t, n_t), dtype = np.complex64)     # for time averaged data
    for f_idx, file in enumerate(files):
        print(file)
        with io.open(file, 'r', encoding = 'windows-1252') as f:    # encoding catches ASCII characters
            lines = [line.rstrip() for line in f]
        for l in lines:
            # print(l)
            x = l.split()
            if N == 2:
                tsrc, tsnk = int(x[0]), int(x[1])
                # entry = np.complex(float(x[2]), float(x[3]))
                entry = np.complex(float(x[start_idx]), float(x[start_idx + 1]))
                Cnpt[f_idx, tsrc, tsnk] = entry
            else:           # read 3-pt contraction structure
                # read out non time-averaged data, without constructing the appropriate operator structures in this function.
                try:
                    tminus, tx, tplus = int(x[0]), int(x[1]), int(x[2])
                    start = 3               # first index with an actual entry
                    for op_idx in range(24):
                        Cnpt[f_idx, op_idx, tminus, tx, tplus] = np.complex(float(x[start + 2 * op_idx]), float(x[start + 2 * op_idx + 1]))
                except (ValueError, UnicodeDecodeError) as e:
                    print('Error: File ' + str(file) + ' at index (' + str(tminus) + ','  + str(tx) + ', ' + str(tplus) + '), op_idx = ' + str(op_idx))
                    print(e)
    return Cnpt

# folds meff over midpoint-- its size is T - 1, so have to be careful about the indexing
def fold_meff(m_eff, T):
    m_eff_folded = np.zeros((n_boot, T // 2 - 1), dtype = np.float64)
    for t in range(1, T // 2):
        t1, t2 = t, T - (t + 1)
        m_eff_folded[:, t - 1] = (m_eff[:, t1] - m_eff[:, t2]) / 2
    return m_eff_folded

# C should be an array of size [num_files, T] where T is the lattice temporal extent.
# Folder is a function of two arguments which returns how to add them, (A + B)
# for sym and (A - B) for antisym. Folding should map 0 to itself, 1 <--> T - 1,
# 2 <--> T - 2, ...
def fold(C, T, sym = True):
    # folded = np.zeros((C.shape[0], T // 2 + 1), dtype = np.complex64)
    folded = np.zeros((C.shape[0], T // 2 + 1), dtype = np.float64)
    folded[:, 0] = C[:, 0]
    for t in range(T // 2):
        if sym:
            folded[:, t + 1] = (C[:, t + 1] + C[:, T - (t + 1)]) / 2
        else:    # fold antisym
            folded[:, t + 1] = (C[:, T - (t + 1)] - C[:, t + 1]) / 2
    if not sym:
        folded[:, T // 2] = np.abs(C[:, T // 2])    # midpoint for asymmetric shouldn't be folded into itself
    return folded

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

def fit_const(fit_region, data, cov):
    """
    data should be either an individual bootstrap or the averages \overline{R}(t), i.e. a vector
    of size T, where T = len(fit_range). cov should have shape (T, T) (so it should already be
    restricted to the fit range).
    """
    if type(fit_region) != np.ndarray:
        fit_region = np.array([x for x in fit_region])
    def chi2(params, data_f, cov_f):
        # return np.einsum('i,ij,j->', (data - np.sum(params * moments), np.linalg.inv(cov), np.sum(params * moments))
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

# data should be an array of size (n_fits, T) and fit_region gives the times to fit at
# TODO this is old code, probably drop it
def fit_constant(fit_region, data, nfits = n_boot):
    if type(fit_region) != np.ndarray:
        fit_region = np.array([x for x in fit_region])
    if len(data.shape) == 1:        # if data only has one dimension, add an axis
        data = np.expand_dims(data, axis = 0)
    sigma_fit = np.std(data[:, fit_region], axis = 0)
    # sigma_fit = np.std(data[:, fit_region], axis = 0, ddof = 1)
    c_fit = np.zeros((nfits), dtype = np.float64)
    # cov = np.zeros((n_boot, n_boot), dtype = np.float64)
    chi2 = lambda x, data, sigma : np.sum((data - x[0]) ** 2 / (sigma ** 2))     # x[0] = constant to fit to
    for i in range(nfits):
        data_fit = data[i, fit_region]
        x0 = [1]          # guess to start at
        # leastsq = lambda x, data : np.sum((data - x[0]) ** 2)        # if we aren't allowed to use the error, do least squares
        # out = optimize.minimize(leastsq, x0, args=(data_fit), method = 'Powell')
        out = optimize.minimize(chi2, x0, args=(data_fit, sigma_fit), method = 'Powell')
        c_fit[i] = out['x']
        # cov_{ij} = 1/2 * D_i D_j chi^2
    # return the total chi^2 and dof for the fit. Get chi^2 by using mean values for all the fits.
    c_mu = np.mean(c_fit)
    data_mu = np.mean(data, axis = 0)
    chi2_mu = chi2([c_mu], data_mu[fit_region], sigma_fit)
    ndof = len(fit_region) - 1    # since we're just fitting a constant, n_params = 1
    return c_fit, chi2_mu, ndof

# data should have length n_boot and data[b] should consist of a list {y_i} with the same size as fit_region.
# We're trying to fit y_i = a x_i + c, where fit_region = {x_i}
# Performs a linear fit to the (x_i, y_i) data for each bootstrap and extrapolates to the point
# x_0.
# @return
# fit_params : array of fit parameters (a, c) for each bootstrap
# chi2 : list of chi2 values for each fit for each bootstrap
# y_extrap : list of extrapolated Lambda values for each bootstrap
def corr_linear_fit(fit_region, data, x_extrap, nfits = n_boot):
    sigma_fit = np.std(data, axis = 0, ddof = 1)
    chi2 = lambda x, dat, sigma : np.sum((dat - (x[0] * fit_region + x[1] * np.ones(len(fit_region)))) ** 2 / (sigma ** 2))
    fit_params = np.zeros((n_boot, 2), dtype = np.float64)    # 2 fit params
    chi2_list = np.zeros((n_boot), dtype = np.float64)
    y_extrap = np.zeros((n_boot), dtype = np.float64)
    for i in range(nfits):
        # print('Fit ' + str(i))
        data_fit = data[i]
        x0 = [0., np.mean(data)]    # guess for fit params (a, c)
        out = optimize.minimize(chi2, x0, args = (data_fit, sigma_fit), method = 'Powell')
        fit_params[i] = out['x']
        chi2_list[i] = chi2(fit_params[i], data_fit, sigma_fit)
        y_extrap[i] = fit_params[i][0] * x_extrap + fit_params[i][1]
    print('Extrapolated Lambda_ij: ' + str(np.mean(y_extrap)) + ' \pm ' + str(np.std(y_extrap, ddof = 1)))
    return fit_params, chi2_list, y_extrap

# performs an uncorrelated fit to superboot objects. This performs (n_boot x n_ens) fits,
# where n_ens is the
# fit form is y_i = c_1 * x_i + c_0
def uncorr_linear_fit(fit_region, superboot_data, x_extrap, label = 'Lambda_ij'):
    n_ens = len(superboot_data)
    nb = superboot_data[0].n_boot
    assert n_ens == superboot_data[0].n_ens
    sigma_fit = np.array([superboot_data[i].std for i in range(n_ens)])
    chi2 = lambda x, dat, sigma : np.sum((dat - (x[0] * fit_region + x[1] * np.ones(len(fit_region)))) ** 2 / (sigma ** 2))
    fit_params = [Superboot(n_ens), Superboot(n_ens)]    # (c_0, c_1)
    chi2_boots = Superboot(n_ens)
    y_extrap = Superboot(n_ens)
    for k in range(n_ens):
        for b in range(nb):
            data_fit = [superboot_data[l].boots[k, b] for l in range(n_ens)]
            x0 = [0., np.mean(data_fit)]    # guess for fit params (a, c)
            out = optimize.minimize(chi2, x0, args = (data_fit, sigma_fit), method = 'Powell')
            c0, c1 = out['x']
            fit_params[0].boots[k, b] = c0
            fit_params[1].boots[k, b] = c1
            chi2_boots.boots[k, b] = chi2(out['x'], data_fit, sigma_fit)
            y_extrap.boots[k, b] = c0 * x_extrap + c1
    # for i in range(n_ens):
    for i in range(2):      # two fit coefficients
        fit_params[i].compute_mean()
        fit_params[i].compute_std()
    chi2_boots.compute_mean()
    chi2_boots.compute_std()
    y_extrap.compute_mean()
    y_extrap.compute_std()
    print('Extrapolated result for ' + label + ': ' + str(y_extrap.mean) + ' \pm ' + str(y_extrap.std))
    return fit_params, chi2_boots, y_extrap

# performs an uncorrelated fit to superboot objects. This performs (n_boot x n_ens) fits,
# where n_ens is the
# fit form is y(x_i) = c_0
def uncorr_const_fit(fit_region, superboot_data, x_extrap): # TODO constant fit and see how the error changes
    n_ens = len(superboot_data)
    nb = superboot_data[0].n_boot
    assert n_ens == superboot_data[0].n_ens
    sigma_fit = np.array([superboot_data[i].std for i in range(n_ens)])
    chi2 = lambda x, dat, sigma : np.sum((dat - (x[0] * np.ones(len(fit_region)))) ** 2 / (sigma ** 2))
    chi2_boots = Superboot(n_ens)
    y_extrap = Superboot(n_ens)   # c_0
    for k in range(n_ens):
        for b in range(nb):
            data_fit = [superboot_data[l].boots[k, b] for l in range(n_ens)]
            x0 = [np.mean(data_fit)]    # guess for fit params (a, c)
            out = optimize.minimize(chi2, x0, args = (data_fit, sigma_fit), method = 'Powell')
            c0 = out['x']
            chi2_boots.boots[k, b] = chi2([c0], data_fit, sigma_fit)
            y_extrap.boots[k, b] = c0
    chi2_boots.compute_mean()
    chi2_boots.compute_std()
    y_extrap.compute_mean()
    y_extrap.compute_std()
    print('Extrapolated result for Lambda_ij: ' + str(y_extrap.mean) + ' \pm ' + str(y_extrap.std))
    return chi2_boots, y_extrap

# Performs a correlated fit in powers of (ap)^2 to the data. Assumes that data
# is a MATRIX of shape [len(fit_range), Superboot.boots], i.e. indexed by (momenta, ens_idx, boot)
# order is what power of \mu^2 to fit to. mu1 = 3 GeV is the matching point
def corr_superboot_fit_apsq(fit_region, data, mu1 = 3.0, order = 1, label = 'Z_ij / Z_V^2'):
    n_pts = len(fit_region)
    assert n_pts == data.shape[0]
    n_ens = data.shape[1]
    nb = data.shape[2]
    sigma_fit = np.zeros((n_pts), dtype = np.float64)
    superboot_data = []
    for i in range(n_pts):
        tmp = Superboot(n_ens)
        tmp.boots = data[i]
        tmp.compute_mean()
        sigma_fit[i] = tmp.compute_std()
        superboot_data.append(tmp)
    moments = np.ones((order + 1, len(fit_region)), dtype = np.float64)         # moments mu_k^{2n} for n = 0, 1, ..., order. Should have len order + 1
    mu1_moments = np.ones((order + 1), dtype = np.float64)
    print('Fitting c0')
    for i in range(order):
        power = 2 * (i + 1)
        print('+ c' + str(i + 1) + ' \mu^' + str(power))
        moments[i + 1] = fit_region ** power
        mu1_moments[i + 1] = mu1 ** power
    # here x = (x[0], x[1], ..., x[order]) are the fit coefficiens, f(\mu) = x[0] + x[1] \mu^2 + x[2] \mu^4 + ...
    chi2 = lambda x, dat, sigma : np.sum((dat - np.einsum('a,ab->b', x, moments)) ** 2 / (sigma ** 2))
    fit_coeffs = np.zeros((order + 1, n_ens, nb), dtype = np.float64)
    chi2_fits = np.zeros((n_ens, nb), dtype = np.float64)
    y_extrap = np.zeros((n_ens, nb), dtype = np.float64)
    for k in range(n_ens):
        for b in range(nb):
            data_fit = data[:, k, b]
            # x0 = [1 for i in range(order + 1)]    # TODO find a better guess
            x0 = [1]    # start with [1, 0, 0, ...]
            for i in range(order):
                x0.append(0)
            contr = np.einsum('a,ab->b', x0, moments)
            out = optimize.minimize(chi2, x0, args = (data_fit, sigma_fit), method = 'Powell')
            fit_coeffs[:, k, b] = out['x']
            chi2_fits[k, b] = chi2(out['x'], data_fit, sigma_fit)
            y_extrap[k, b] = np.sum(out['x'] * mu1_moments)
            print(fit_coeffs[:, k, b])
            print('Chi^2: ' + str(chi2_fits[k, b]))
            print('Extrapolated ' + label + ' = ' + str(y_extrap[k, b]))
    return fit_coeffs, chi2_fits, y_extrap

# TODO deprecated, use fit_const_allrange instead.
# data should be an array of size (n_b, T). Fits over every range with size >= TT_min and weights
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

# Inputs should all be in the same format as above. c_ens = list of c_ens_f ensembles, each of
# shape n_boot, so meff has shape (# accepted fits, n_boot)
# for each accepted fit, weights = pf (\delta c_ens)^-2 for each accepted fit. Note that
# weights need not be normalized, this will make sure it is.
def analyze_accepted_fits(c_ens, weights):
    weights = weights / np.sum(weights)
    c_ens_mu, c_ens_sigma = np.mean(c_ens, axis = 1), np.std(c_ens, axis = 1, ddof = 1)
    cbar = np.sum(weights * c_ens_mu)
    dmeff_stat_sq = np.sum(weights * (c_ens_sigma ** 2))
    dmeff_sys_sq = np.sum(weights * ((c_ens_mu - cbar) ** 2))
    c_sigma = np.sqrt(dmeff_stat_sq + dmeff_sys_sq)
    c_boot = np.einsum('f,fb->b', weights, c_ens)
    c_boot = spread_boots(c_boot, c_sigma)
    return cbar, c_sigma, c_boot

# returns a bootstrap ensemble by summing fits over their weight
def weighted_sum_bootstrap(meff, weights):
    weights = weights / np.sum(weights)
    return np.einsum('fb,f->b', meff, weights)

# spreads a dataset in a correlated way to have standard deviation new_std.
def spread_boots(data, new_std, boot_axis = 0):
    old_std = np.std(data, axis = boot_axis, ddof = 1)
    mu = np.mean(data, axis = boot_axis)
    spread_data = np.full(data.shape, mu, dtype = data.dtype)
    spread_factor = new_std / old_std
    spread_data += spread_factor * (data - spread_data)
    return spread_data

# VarPro code

# Ab = A(b) should already be evaluated of size N_data x N_params, z should be a vector of size N_data
def ahat(Ab, z):
    #return np.sum(Ab * z) / np.sum(Ab * Ab)
    return (np.transpose(Ab) @ z) / np.sum(Ab * Ab)

# m is a N_params length vector, A should be a function which returns a N_data x N_params matrix,
# z a vector of size N_data
def chi2_varpro(m, A, z):
    Ab = A(m)
    # ahatb = np.array([ahat(Ab, z)])
    ahatb = ahat(Ab, z)
    return np.sum((Ab @ ahatb - z) ** 2)

# Generate fake ensemble of data with mean mu and std sigma for one ensemble
def gen_fake_ensemble(val, n_samples = n_boot, s = 20):
    random.seed(s)
    fake_data = np.zeros((n_samples), dtype = np.float64)
    # generate fake data in each ensemble with total std sigma: since we're treating each ensemble
    # independently and add their variances in quadrature, each ensemble should have std = sigma / sqrt(N_ens)
    mu, sigma = val[0], val[1]
    for i in range(n_samples):
        fake_data[i] = random.gauss(mu, sigma)
    return fake_data

# BOOTSTRAP OBJECT: should be an array of size (n_ens, n_boot)
class Superboot:

    # self.avg = approximate average over the active ensemble (so over n_boot values)-- this is demonstrated
    # most clearly with populating the fake ensemble, where self.avg is the mu used to get a sample.
    # self.mean = average over the entire ensemble (so average over card parameters), this is the actual sample mean
    def __init__(self, n_ens, nb = n_boot):
        self.n_ens = n_ens
        self.boots = np.zeros((n_ens, n_boot), dtype = np.float64)
        self.avg = 0           # avg
        self.mean = 0
        self.std = 0
        self.n_boot = nb
        self.cardinal = nb * n_ens    # total cardinality

    # populates ensemble with data on axis (size of data should be n_boots) and avg elsewhere.
    def populate_ensemble(self, data, axis):
        self.avg = np.mean(data)
        for e in range(self.n_ens):
            if e == axis:
                self.boots[e] = data
            else:
                self.boots[e] = self.avg
        self.compute_mean()
        self.compute_std()
        return self

    def gen_fake_ensemble(self, mean, sigma, s = 20):
        random.seed(s)
        ens_sig = sigma / np.sqrt(self.n_ens)
        self.avg = mean
        for e in range(self.n_ens):
            for b in range(self.n_boot):
                self.boots[e, b] = random.gauss(mean, ens_sig)
        self.compute_mean()
        self.compute_std()
        return self

    # generates a fake ensemble along one axis. This should only be used for testing, i.e. for replicating the stats on David's data.
    def gen_fake_ensemble_axis(self, mean, sigma, axis, s = 20):
        random.seed(s)
        self.avg = mean
        fake_data = [random.gauss(mean, sigma) for i in range(self.n_boot)]
        self.populate_ensemble(fake_data, axis)
        self.compute_mean()
        self.compute_std()
        return self

    # mean is the average over all values, not just the active ensemble.
    def compute_mean(self):
        self.mean = np.sum(self.boots) / self.cardinal
        return self.mean

    def compute_std(self):
        vars = np.std(self.boots, axis = 1, ddof = 1) ** 2
        self.std = np.sqrt(np.sum(vars))
        return self.std

    def __add__(self, other):
        sum = Superboot(self.n_ens, self.n_boot)
        sum.boots = self.boots + other.boots
        sum.avg = self.avg + other.avg
        sum.compute_mean()
        sum.compute_std()
        return sum

    def __sub__(self, other):
        diff = Superboot(self.n_ens, self.n_boot)
        diff.boots = self.boots - other.boots
        diff.avg = self.avg - other.avg
        diff.compute_mean()
        diff.compute_std()
        return diff

    def __mul__(self, other):
        prod = Superboot(self.n_ens, self.n_boot)
        prod.boots = self.boots * other.boots
        prod.avg = self.avg * other.avg
        prod.compute_mean()
        prod.compute_std()
        return prod

    def __truediv__(self, other):
        quotient = Superboot(self.n_ens, self.n_boot)
        quotient.boots = self.boots / other.boots
        quotient.avg = self.avg / other.avg
        quotient.compute_mean()
        quotient.compute_std()
        return quotient

    def __pow__(self, x):
        exp = Superboot(self.n_ens, self.n_boot)
        exp.boots = self.boots ** x
        exp.avg = self.avg ** x
        exp.compute_mean()
        exp.compute_std()
        return exp

    def scale(self, c):
        scaled = Superboot(self.n_ens, self.n_boot)
        scaled.boots = self.boots * c
        scaled.avg = self.avg * c
        scaled.compute_mean()
        scaled.compute_std()
        return scaled


# X should be a list of length n_ens, containing arrays of shape (n_ops, n_samples[ens_idx]). Averages over ensemble and boot indices (superboot indices)
def superboot_mean(X):
    N = len(X)
    return sum([np.mean(X[i], axis = 1) for i in range(N)]) / N

# Compute sigma^2 for each ensemble, then add in quadrature
def superboot_std(X):
    N = len(X)
    var = sum([np.std(X[i], axis = 1)**2 for i in range(N)])
    return np.sqrt(var)

def get_effective_mass(ensemble_avg):
    ratios = np.abs(ensemble_avg / np.roll(ensemble_avg, shift = -1, axis = 1))[:, :-1]
    m_eff_ensemble = np.log(ratios)
    return m_eff_ensemble

# Returns the mean and standard deviation of cosh corrected effective mass ensemble.
# def get_cosh_effective_mass(ensemble_avg, TT):
def get_cosh_effective_mass(ensemble_avg):
    TT = ensemble_avg.shape[1]
    ratios = np.abs(ensemble_avg / np.roll(ensemble_avg, shift = -1, axis = 1))[:, :-1]
    m_eff_ensemble = np.log(ratios)
    cosh_m_eff_ensemble = np.zeros(ratios.shape, dtype = np.float64)
    for ens_idx in range(ratios.shape[0]):
        for t in range(ratios.shape[1]):
            m = optimize.root(lambda m : ratios[ens_idx, t] - np.cosh(m * (t - TT / 2)) / np.cosh(m * (t + 1 - TT / 2)), \
                         m_eff_ensemble[ens_idx, t])
            # m = optimize.root(lambda m : ratios[ens_idx, t] - np.cosh(m * t) / np.cosh(m * (t + 1)), \
            #              m_eff_ensemble[ens_idx, t])
            cosh_m_eff_ensemble[ens_idx, t] = m.x
    return cosh_m_eff_ensemble

def get_sinh_effective_mass(ensemble_avg):
    TT = ensemble_avg.shape[1]
    ratios = np.abs(ensemble_avg / np.roll(ensemble_avg, shift = -1, axis = 1))[:, :-1]
    m_eff_ensemble = np.log(ratios)
    sinh_m_eff_ensemble = np.zeros(ratios.shape, dtype = np.float64)
    for ens_idx in range(ratios.shape[0]):
        for t in range(ratios.shape[1]):
            m = optimize.root(lambda m : ratios[ens_idx, t] - np.sinh(m * (t - TT / 2)) / np.sinh(m * (t + 1 - TT / 2)), \
                         m_eff_ensemble[ens_idx, t])
            sinh_m_eff_ensemble[ens_idx, t] = m.x
    return sinh_m_eff_ensemble

# Bootstraps an input tensor. Pass in a tensor with the shape (ncfgs, tensor_shape)
def bootstrap(S, seed = 10, weights = None, data_type = np.complex64, Nb = n_boot):
    num_configs, tensor_shape = S.shape[0], S.shape[1:]
    bootshape = [Nb]
    bootshape.extend(tensor_shape)    # want bootshape = (n_boot, tensor_shape)
    samples = np.zeros(bootshape, dtype = data_type)
    if weights == None:
        weights = np.ones((num_configs))
    weights2 = weights / float(np.sum(weights))
    np.random.seed(seed)
    for boot_id in range(Nb):
        cfg_ids = np.random.choice(num_configs, p = weights2, size = num_configs, replace = True)
        samples[boot_id] = np.mean(S[cfg_ids], axis = 0)
    return samples

# Invert propagators to get S^{-1} required for amputation.
def invert_props(props, Nb = n_boot):
    Sinv = np.zeros(props.shape, dtype = np.complex64)
    for b in range(Nb):
        Sinv[b] = np.linalg.tensorinv(props[b])
    return Sinv

# Amputate legs to get vertex function \Gamma(p). Uses first argument to amputate left-hand side and
# second argument to amputate right-hand side.
def amputate_threepoint(props_inv_L, props_inv_R, threepts, Nb = n_boot):
    Gamma = np.zeros(threepts.shape, dtype = np.complex64)
    for b in range(Nb):
        Sinv_L, Sinv_R, G = props_inv_L[b], props_inv_R[b], threepts[b]
        Gamma[b] = np.einsum('aibj,bjck,ckdl->aidl', Sinv_L, G, Sinv_R)
    return Gamma

# amputates the four point function. Assumes the left leg has momentum p1 and right legs have
# momentum p2, so amputates with p1 on the left and p2 on the right
def amputate_fourpoint(props_inv_L, props_inv_R, fourpoints, Nb = n_boot):
    Gamma = np.zeros(fourpoints.shape, dtype = np.complex64)
    for b in range(Nb):
        Sinv_L, Sinv_R, G = props_inv_L[b], props_inv_R[b], fourpoints[b]
        Gamma[b] = np.einsum('aiem,ckgp,emfngphq,fnbj,hqdl->aibjckdl', Sinv_L, Sinv_L, G, Sinv_R, Sinv_R)
    return Gamma

# q is the lattice momentum that should be passed in
def quark_renorm(props_inv_q, q, Nb = n_boot):
    Zq = np.zeros((Nb), dtype = np.complex64)
    for b in range(Nb):
        Sinv = props_inv_q[b]
        # for mu in range(d):
        #     print(np.einsum('ij,ajai', gamma[mu], Sinv))
        Zq[b] = (1j) * sum([q[mu] * np.einsum('ij,ajai', gamma[mu], Sinv) for mu in range(d)]) / (12 * square(q))
    return Zq

# Returns the adjoint of a bootstrapped propagator object
def adjoint(S):
    return np.conjugate(np.einsum('...aibj->...bjai', S))

def antiprop(S):
    Sdagger = adjoint(S)
    return np.einsum('ij,zajbk,kl->zaibl', gamma5, Sdagger, gamma5)

# key == 'gamma' or key == 'qslash' for the different schemes. If qslash, must enter q, p_1, and p_2
# Along with getF(), seems to return the correct projectors which agree with the paper
def projectors(scheme = 'gamma', q = 0, p1 = 0, p2 = 0):
    P = np.zeros((5, Nc, Nd, Nc, Nd, Nc, Nd, Nc, Nd), dtype = np.complex64)
    if scheme == 'gamma':
        for a, b in itertools.product(range(Nc), repeat = 2):
            for beta, alpha, sigma, gam in itertools.product(range(Nd), repeat = 4):
                P[2, a, beta, a, alpha, b, sigma, b, gam] += delta[beta, alpha] * delta[sigma, gam] - gamma5[beta, alpha] * gamma5[sigma, gam]
                P[3, a, beta, a, alpha, b, sigma, b, gam] += delta[beta, alpha] * delta[sigma, gam] + gamma5[beta, alpha] * gamma5[sigma, gam]
                for mu in range(d):
                    P[0, a, beta, a, alpha, b, sigma, b, gam] += gamma[mu, beta, alpha] * gamma[mu, sigma, gam] + gammaMu5[mu, beta, alpha] * gammaMu5[mu, sigma, gam]
                    P[1, a, beta, a, alpha, b, sigma, b, gam] += gamma[mu, beta, alpha] * gamma[mu, sigma, gam] - gammaMu5[mu, beta, alpha] * gammaMu5[mu, sigma, gam]
                    for nu in range(d):
                        P[4, a, beta, a, alpha, b, sigma, b, gam] += (1 / 2) * sigmaD[mu, nu, beta, alpha] * sigmaD[mu, nu, sigma, gam]
    elif scheme == 'qslash':
        qsq, p1sq, p2sq, qslash, p1p2 = square(q), square(p1), square(p2), slash(q), np.dot(p1, p2)
        qslash5 = qslash @ gamma5
        Pl = (delta - gamma5) / 2
        psigmap = np.zeros((Nd, Nd), dtype = np.complex64)
        for mu, nu in itertools.product(range(Nd), repeat = 2):
            psigmap += p1[mu] * (sigmaD[mu, nu] @ Pl) * p2[nu]
        for a, b in itertools.product(range(Nc), repeat = 2):
            for beta, alpha, sigma, gam in itertools.product(range(Nd), repeat = 4):
                P[0, a, beta, a, alpha, b, sigma, b, gam] += (1 / qsq) * (qslash[beta, alpha] * qslash[sigma, gam] + qslash5[beta, alpha] * qslash5[sigma, gam])
                P[1, a, beta, a, alpha, b, sigma, b, gam] += (1 / qsq) * (qslash[beta, alpha] * qslash[sigma, gam] - qslash5[beta, alpha] * qslash5[sigma, gam])
                P[2, a, beta, b, alpha, b, sigma, a, gam] += (1 / qsq) * (qslash[beta, alpha] * qslash[sigma, gam] - qslash5[beta, alpha] * qslash5[sigma, gam])
                P[3, a, beta, b, alpha, b, sigma, a, gam] += (1 / (p1sq * p2sq - p1p2 * p1p2)) * (psigmap[beta, alpha] * psigmap[sigma, gam])
                P[4, a, beta, a, alpha, b, sigma, b, gam] += (1 / (p1sq * p2sq - p1p2 * p1p2)) * (psigmap[beta, alpha] * psigmap[sigma, gam])
    return P

# gets tree level projections in scheme == 'gamma' or scheme == 'qslash'. Pass in a Lattice instance L so that
# the function knows the lattice dimensions.
def getF(L, scheme = 'gamma'):
    q, p1, p2 = L.to_linear_momentum([1, 1, 0, 0]), L.to_linear_momentum([-1, 0, 1, 0]), L.to_linear_momentum([0, 1, 1, 0])
    P = projectors(scheme, q, p1, p2)
    F = np.einsum('nbjaidlck,maibjckdl->mn', P, tree)
    return F

# Returns a in units of GeV^{-1}
def fm_to_GeV(a):
    return a / hbarc

# returns mu for mode k at lattice spacing A fm, on lattice L
def get_energy_scale(k, a, L):
    return 2 * (hbarc / a) * np.linalg.norm(np.sin(np.pi * k / L.LL))

def load_data_h5(file):
    print('Loading ' + str(file) + '.')
    f = h5py.File(file, 'r')
    k_list = f['momenta'][()]
    p_list = np.array([to_lattice_momentum(k) for k in k_list])
    Z = f['Z'][()]
    Zq = f['Zq'][()]
    cfgnum = f['cfgnum'][()]
    f.close()
    return k_list, p_list, Z, Zq, cfgnum

def load_Zq(file):
    f = h5py.File(file, 'r')
    k_list = f['momenta'][()]
    p_list = np.array([to_lattice_momentum(k) for k in k_list])
    Zq = f['Zq'][()]
    f.close()
    return k_list, p_list, Zq

# partition k_list into orbits by O(3) norm and p[3]
def get_O3_orbits(k_list):
    orbits, psquared_p3_list = [], []    # these should always be the same size
    for p in k_list:
        psquared = p[0] ** 2 + p[1] ** 2 + p[2] ** 2
        if (psquared, p[3]) in psquared_p3_list:    # then add to existing orbit
            idx = psquared_p3_list.index((psquared, p[3]))
            orbits[idx].append(p)
        else:
            psquared_p3_list.append((psquared, p[3]))
            orbits.append([p])
    return orbits, psquared_p3_list

# Z is a (n_momentum x n_boot) matrix
def average_O3_orbits(Z, k_list):
    orbits, psquared_p3_list = get_O3_orbits(k_list)
    k_rep_list, Zavg, sigma = [], [], []
    for coset in orbits:
        # idx_list = [k_list.index(p) for p in coset]
        idx_list = [np.where(k_list == p) for p in coset]
        k_rep_list.append(coset[0])    # take the first representative
        Zavg.append(np.mean(Z[idx_list, :]))
        sigma.append(np.std(Z[idx_list, :]))
    return k_rep_list, Zavg, sigma
