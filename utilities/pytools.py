
################################################################################
# This script saves a list of all common utility functions that I may need to  #
# use in my python code. To import the script, simply add the path with sys    #
# before importing. For example, if lqcd/ is in /Users/theoares:               #
#                                                                              #
# import sys
# sys.path.append('/Users/theoares/lqcd/utilities')
# from pytools import *
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

def set_boots(nb):
    global n_boot
    n_boot = nb
    return n_boot

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

# q is the lattice momentum that should be passed in
def quark_renorm(props_inv_q, q, Nb = n_boot):
    Zq = np.zeros((Nb), dtype = np.complex64)
    for b in range(Nb):
        Sinv = props_inv_q[b]
        Zq[b] = (1j) * sum([q[mu] * np.einsum('ij,ajai', gamma[mu], Sinv) for mu in range(d)]) / (12 * square(q))
    return Zq

# Returns the adjoint of a bootstrapped propagator object
def adjoint(S):
    return np.conjugate(np.einsum('...aibj->...bjai', S))

def antiprop(S):
    Sdagger = adjoint(S)
    return np.einsum('ij,zajbk,kl->zaibl', gamma5, Sdagger, gamma5)

# Returns a in units of GeV^{-1}
def fm_to_GeV(a):
    return a / hbarc

# returns mu for mode k at lattice spacing A fm, on lattice L
def get_energy_scale(k, a, L):
    return 2 * (hbarc / a) * np.linalg.norm(np.sin(np.pi * k / L.LL))

# TODO not sure if the following functions work, should test them
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
