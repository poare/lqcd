
################################################################################
# This script saves a list of common utility functions for general data        #
# processing that I may need to use in my python code. To import the script,   #
# simply add the path with sys before importing. For example, if lqcd/ is in   #
# /Users/theoares:                                                             #
#                                                                              #
# import sys                                                                   #
# sys.path.append('/Users/theoares/lqcd/utilities')                            #
# from pytools import *                                                        #
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

#  Pass in a tensor with the shape (ncfgs, tensor_shape)
def bootstrap(S, seed = 10, weights = None, data_type = np.complex64, Nb = n_boot):
    """
    Bootstraps an input tensor.

    Parameters
    ----------
    S : np.array [ncfgs, ...]
        Input tensor to bootstrap. ncfgs is the number of configurations.
    seed : int (default = 10)
        Seed of random number generator for bootstrapping.
    weights : np.array [Nb] (default = None)
        Weights for bootstrap sample to generate.
    Nb : int (default = n_boot)
        Length of propagator object.

    Returns
    -------
    np.array [Nb, ...]
        Bootstrapped tensor.
    """
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

def invert_props(props, Nb = n_boot):
    """
    Invert propagators to get S^{-1}, required for amputation.

    Parameters
    ----------
    props : np.array [Nb, 3, 4, 3, 4]
        Propagator object for a single momentum.
    Nb : int (default = n_boot)
        Length of propagator object.

    Returns
    -------
    np.array [Nb, 3, 4, 3, 4]
        Inverse propagator object.
    """
    Sinv = np.zeros(props.shape, dtype = np.complex64)
    for b in range(Nb):
        Sinv[b] = np.linalg.tensorinv(props[b])
    return Sinv

def amputate_threepoint(props_inv_L, props_inv_R, threepts, Nb = n_boot):
    """
    Amputate legs of a three-point function to get vertex function \Gamma(p). Uses first
    argument to amputate left-hand side and second argument to amputate right-hand side.

    Parameters
    ----------
    props_inv_L : np.array [Nb, 3, 4, 3, 4]
        Inverse propagator to amputate with on the left of the three-point function.
    props_inv_R : np.array [Nb, 3, 4, 3, 4]
        Inverse propagator to amputate with on the right of the three-point function.
    threepts : np.array [Nb, 3, 4, 3, 4]
        Three-point function for a single momenta to amputate.
    Nb : int (default = n_boot)
        Length of propagator object.

    Returns
    -------
    np.array [Nb, 3, 4, 3, 4]
        Amputated vertex function.
    """
    Gamma = np.zeros(threepts.shape, dtype = np.complex64)
    for b in range(Nb):
        Sinv_L, Sinv_R, G = props_inv_L[b], props_inv_R[b], threepts[b]
        Gamma[b] = np.einsum('aibj,bjck,ckdl->aidl', Sinv_L, G, Sinv_R)
    return Gamma

# amputates the four point function. Assumes the left leg has momentum p1 and right legs have
# momentum p2, so amputates with p1 on the left and p2 on the right
def amputate_fourpoint(props_inv_L, props_inv_R, fourpoints, Nb = n_boot):
    """
    Amputate legs of a four-point function to get vertex function \Gamma(p). Uses first
    argument to amputate left-hand side and second argument to amputate right-hand side.

    Parameters
    ----------
    props_inv_L : np.array [Nb, 3, 4, 3, 4]
        Inverse propagator to amputate with on the left of the four-point function.
    props_inv_R : np.array [Nb, 3, 4, 3, 4]
        Inverse propagator to amputate with on the right of the four-point function.
    fourpoints : np.array [Nb, 3, 4, 3, 4]
        Four-point function for a single momenta to amputate.
    Nb : int (default = n_boot)
        Length of propagator object.

    Returns
    -------
    np.array [Nb, 3, 4, 3, 4]
        Amputated vertex function.
    """
    Gamma = np.zeros(fourpoints.shape, dtype = np.complex64)
    for b in range(Nb):
        Sinv_L, Sinv_R, G = props_inv_L[b], props_inv_R[b], fourpoints[b]
        Gamma[b] = np.einsum('aiem,ckgp,emfngphq,fnbj,hqdl->aibjckdl', Sinv_L, Sinv_L, G, Sinv_R, Sinv_R)
    return Gamma

def quark_renorm(props_inv_q, q, Nb = n_boot):
    """
    Computes the quark field renormalization Zq at momentum q,
        Zq = i Tr[q^mu gamma^mu S^{-1}(q)] / (12 q^2).

    Parameters
    ----------
    props_inv_q : np.array [Nb, 3, 4, 3, 4]
        Inverse propagator to compute Zq with.
    q : np.array [4]
        Lattice momentum to compute Zq at.
    Nb : int (default = n_boot)
        Length of propagator object.

    Returns
    -------
    np.array [Nb]
        Quark field renormalization computed at each bootstrap.
    """
    Zq = np.zeros((Nb), dtype = np.complex64)
    for b in range(Nb):
        Sinv = props_inv_q[b]
        Zq[b] = (1j) * sum([q[mu] * np.einsum('ij,ajai', gamma[mu], Sinv) for mu in range(d)]) / (12 * square(q))
    return Zq

# Returns the adjoint of a bootstrapped propagator object
def adjoint(S):
    """
    Returns the adjoint of a bootstrapped propagator

    Parameters
    ----------
    S : np.array [Nb, 3, 4, 3, 4]
        Propagator to take adjoint of.

    Returns
    -------
    np.array [Nb, 3, 4, 3, 4]
        Adjoint of propagator.
    """
    return np.conjugate(np.einsum('...aibj->...bjai', S))

def antiprop(S):
    """
    Returns the antipropagator of a propagator S, antiprop = gamma_5 S^dagger gamma_5.

    Parameters
    ----------
    S : np.array [Nb, 3, 4, 3, 4]
        Propagator to take adjoint of.

    Returns
    -------
    np.array [Nb, 3, 4, 3, 4]
        Antipropagator of S.
    """
    Sdagger = adjoint(S)
    return np.einsum('ij,zajbk,kl->zaibl', gamma5, Sdagger, gamma5)

def fm_to_GeV(a):
    """
    Returns a in units of GeV^{-1}.

    Parameters
    ----------
    a : float
        Lattice spacing in fm.

    Returns
    -------
    float
        Lattice spacing in GeV^{-1}.
    """
    return a / hbarc

# returns mu for mode k at lattice spacing A fm, on lattice L
def get_energy_scale(k, a, L):
    return 2 * (hbarc / a) * np.linalg.norm(np.sin(np.pi * k / L.LL))

def form_2d_sym_irreps(T):
    """
    Given a rank two tensor T_{mu nu}, returns the 3-dimensional H(4) irrep
    tau_1^{(3)} and 6-dimensional H(4) irrep tau_3^{(6)}.

    Parameters
    ----------
    T : np.array [float or complex]
        Tensor to organize into H(4) irreps. The first two components should be 4 x 4.

    Returns
    -------
    np.array [float or complex]
        Tensor components in the tau_1^{(3)} irrep.
    np.array [float or complex]
        Tensor components in the tau_3^{(6)} irrep.
    """
    tau3 = np.array([
        (T[2, 2] - T[3, 3]) / np.sqrt(2),
        (T[0, 0] - T[1, 1]) / np.sqrt(2),
        (T[0, 0] + T[1, 1] - T[2, 2] - T[3, 3]) / 2.
    ])
    tau6 = np.array([
        (T[0, 1] + T[1, 0]) / np.sqrt(2),
        (T[0, 2] + T[2, 0]) / np.sqrt(2),
        (T[0, 3] + T[3, 0]) / np.sqrt(2),
        (T[1, 2] + T[2, 1]) / np.sqrt(2),
        (T[1, 3] + T[3, 1]) / np.sqrt(2),
        (T[2, 3] + T[3, 2]) / np.sqrt(2)
    ])
    return tau3, tau6

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
