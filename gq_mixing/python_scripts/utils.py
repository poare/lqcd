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
# import sys
# sys.path.append('/Users/theoares/lqcd/utilities')
# from pytools import *         # figure out a way to import pytools without breaking everything
# from formattools import *

n_boot = 200

# Tree level values
Gamma_qq_1 = lambda p : (-2j) * np.array([[(p[mu] * gamma[nu] + p[nu] * gamma[mu]) / 2 - delta[mu, nu] * (slash(p)) / 4 for mu in range(4)] for nu in range(4)])

def readfiles(cfgs, k):
    props = np.zeros((len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    Gqg = np.zeros((4, 4, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    Gqq3 = np.zeros((3, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    Gqq6 = np.zeros((6, len(cfgs), 3, 4, 3, 4), dtype = np.complex64)
    for idx, file in enumerate(cfgs):
        # print(file)
        f = h5py.File(file, 'r')
        kstr = klist_to_string(k, 'p')
        # print('prop/' + kstr)
        props[idx] = np.einsum('ijab->aibj', f['prop/' + kstr][()])
        for mu, nu in itertools.product(range(4), repeat = 2):
            Gqg[mu, nu, idx] = np.einsum('ijab->aibj', f['Gqg' + str(mu + 1) + str(nu + 1) + '/' + kstr][()])
        for a in range(3):
            Gqq3[a, idx] = np.einsum('ijab->aibj', f['Gqq/3' + str(a + 1) + '/' + kstr][()])
        for a in range(6):
            Gqq6[a, idx] = np.einsum('ijab->aibj', f['Gqq/6' + str(a + 1) + '/' + kstr][()])
    return props, Gqg, Gqq3, Gqq6

def tau13_irrep(O, n):
    """Gets the nth element of tau13 from a tensor O[mu, nu]."""
    assert n in [0, 1, 2]
    return [(O[2, 2] - O[3, 3]) / np.sqrt(2), (O[0, 0] - O[1, 1]) / np.sqrt(2), (O[0, 0] + O[1, 1] - O[2, 2] - O[3, 3]) / 2.][n]

def tau36_irrep(O, n):
    """Gets the nth element of tau36 from a tensor O[mu, nu]."""
    assert n in [0, 1, 2, 3, 4, 5]
    return [(O[0, 1] + O[1, 0]) / np.sqrt(2), (O[0, 2] + O[2, 0]) / np.sqrt(2), (O[0, 3] + O[3, 0]) / np.sqrt(2) \
                (O[1, 2] + O[2, 1]) / np.sqrt(2), (O[1, 3] + O[3, 1]) / np.sqrt(2), (O[2, 3] + O[3, 2]) / np.sqrt(2)][n]

# Assumes O2 can have a tensor structure with Dirac indices on the lect.
def inner(O1, O2):
    traces = [np.einsum('ij,ji', tau13_irrep(O1, n), tau13_irrep(O2, n)) for n in range(3)]
    return sum(traces)

# Returns the matrix A_{ab} discussed in Sergei's thesis.
def A_ab(p):
    return np.array([
        [inner(Lambda1(p), Lambda1(p)), inner(Lambda1(p), Lambda2(p))],
        [inner(Lambda2(p), Lambda1(p)), inner(Lambda2(p), Lambda2(p))]
    ])
