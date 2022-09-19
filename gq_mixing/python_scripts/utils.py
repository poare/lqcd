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

# import all shared utilities
import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from pytools import *         # figure out a way to import pytools without breaking everything
from formattools import *

# n_boot = 50
n_boot = 200

# Tree level values
Gamma_qq_1 = lambda p : (-2j) * np.array([[(p[mu] * gamma[nu] + p[nu] * gamma[mu]) / 2 - delta[mu, nu] * (slash(p)) / 4 for mu in range(4)] for nu in range(4)])
Gamma_qq_2 = lambda p : (-2j) * np.array([[p[mu] * p[nu] * slash(p) / square(p)  - delta[mu, nu] * slash(p) / 4 for mu in range(4)] for nu in range(4)])

def readfiles(cfgs, k):
    props = np.zeros((len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    Gqg = np.zeros((4, 4, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    Gqq3 = np.zeros((3, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    Gqq6 = np.zeros((6, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    for idx, file in enumerate(cfgs):
        # print(file)
        try:
            f = h5py.File(file, 'r')
        except:
            print('File: ' + str(file) + ' not openable.')
        kstr = klist_to_string(k, 'p')
        # print('prop/' + kstr)
        props[idx] = np.einsum('ijab->aibj', f['prop/' + kstr][()])
        # for mu, nu in itertools.product(range(4), repeat = 2):
        #     Gqg[mu, nu, idx] = np.einsum('ijab->aibj', f['Gqg' + str(mu + 1) + str(nu + 1) + '/' + kstr][()])
        for a in range(3):
            Gqq3[a, idx] = np.einsum('ijab->aibj', f['Gqq/3' + str(a + 1) + '/' + kstr][()])
        for a in range(6):
            Gqq6[a, idx] = np.einsum('ijab->aibj', f['Gqq/6' + str(a + 1) + '/' + kstr][()])
        f.close()
    return props, Gqg, Gqq3, Gqq6

def tau13_irrep(O, n):
    """Gets the nth element of tau13 from a tensor O[mu, nu]."""
    assert n in [0, 1, 2]
    return [(O[2, 2] - O[3, 3]) / np.sqrt(2), (O[0, 0] - O[1, 1]) / np.sqrt(2), (O[0, 0] + O[1, 1] - O[2, 2] - O[3, 3]) / 2.][n]

def tau36_irrep(O, n):
    """Gets the nth element of tau36 from a tensor O[mu, nu]."""
    assert n in [0, 1, 2, 3, 4, 5]
    return [(O[0, 1] + O[1, 0]) / np.sqrt(2), (O[0, 2] + O[2, 0]) / np.sqrt(2), (O[0, 3] + O[3, 0]) / np.sqrt(2), \
                (O[1, 2] + O[2, 1]) / np.sqrt(2), (O[1, 3] + O[3, 1]) / np.sqrt(2), (O[2, 3] + O[3, 2]) / np.sqrt(2)][n]

def irrep_label(dim):
    """Gets the irrep label with irrep dimension dim."""
    if dim == 3:
        return '$\\tau_1^{(3)}$'
    elif dim == 6:
        return '$\\tau_3^{(6)}$'
    else:
        raise Exception('Only irreps tau_1^3 and tau_3^6 implemented.')

def get_inner(irrep_dim):
    """
    Returns the inner product for the corresponding H(4) irrep, tau_1^3 or tau_3^6 for operators which have
    Lorentz indices

    Parameters
    ----------
    irrep_dim : int
        3 for the tau_1^3 irrep and 6 for the tau_3^6 irrep. Any other value will raise an Exception.

    Returns
    -------
    function : (np.array [], np.array []) -> np.float64
        Function handle for inner product
    """
    if irrep_dim == 3:
        irrep = tau13_irrep
    elif irrep_dim == 6:
        irrep = tau36_irrep
    else:
        raise Exception('Only irreps tau_1^3 and tau_3^6 implemented.')
    def inner(O1, O2):
        """
        Inner product on the H(4) irrep of dimension irrep_dim. Assumes that O1, O2 are either of shape
        (irrep_dim, 4, 4), or of shape (4, 4, 4, 4), where the last two 4-dimensional indices are spinor
        indices.
        """
        if O1.shape == (4, 4, 4, 4):
            O1 = np.array([irrep(O1, n) for n in range(irrep_dim)], dtype = O1.dtype)
        if O2.shape == (4, 4, 4, 4):
            O2 = np.array([irrep(O2, n) for n in range(irrep_dim)], dtype = O2.dtype)
        return np.einsum('aij,aji->', O1, O2)
    return inner

def A_ab(p, inner):
    """
    Returns the matrix of inner products A_{ab}.
    """
    return np.array([
        [inner(Gamma_qq_1(p), Gamma_qq_1(p)), inner(Gamma_qq_1(p), Gamma_qq_2(p))],
        [inner(Gamma_qq_2(p), Gamma_qq_1(p)), inner(Gamma_qq_2(p), Gamma_qq_2(p))]
    ])

def detA(p, inner):
    return inner(Gamma_qq_1(p), Gamma_qq_1(p)) * inner(Gamma_qq_2(p), Gamma_qq_2(p)) - inner(Gamma_qq_1(p), Gamma_qq_2(p)) ** 2

def A_inv_ab(p, inner):
    A11 = inner(Gamma_qq_1(p), Gamma_qq_1(p))
    A12 = inner(Gamma_qq_1(p), Gamma_qq_2(p))
    A22 = inner(Gamma_qq_2(p), Gamma_qq_2(p))
    det = A11 * A22 - A12 * A12
    return (1 / det) * np.array([
        [A22, -A12],
        [-A12, A11]
    ])
