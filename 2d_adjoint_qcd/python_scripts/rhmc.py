################################################################################
# Rational HMC (RHMC) implementation.                                          #
################################################################################
# Note that the file rhmc_coeffs.py has approximations to K^{-1/4} and         #
# K^{+1/8} stored for different numbers of partial fractions P and different   #
# spectral ranges.                                                             #
################################################################################
# Assumptions and shapes of basic objects:                                     #
#   - Spacetime dimensions d = 2.                                              #
#   - Number of colors Nc, which should be easy to vary.                       #
#   - Lattice Lambda of shape [Lx, Lt].                                        #
#   - Fundamental gauge field U of shape [d, Lx, Lt, Nc, Nc].                  #
#   - Adjoint gauge field V of shape [d, Lx, Lt, Nc^2 - 1, Nc^2 - 1].          #
#   - Scalar (pseudofermion) field Phi of shape [d, Lx, Lt].                   #
#   - Adjoint Majorana fermion field of shape [Lx, Lt, Nc, Ns].                #
#   - Dirac operator of shape [Lx, Lt, Nc, Ns, Lx, Lt, Nc, Ns].                #
################################################################################
# Author: Patrick Oare                                                         #
################################################################################

n_boot = 100

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import bsr_matrix
from scipy.optimize import root
from scipy.linalg import block_diag
import h5py
import os
import itertools
import pandas as pd
import gvar as gv
import lsqfit

import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from constants import *
from fittools import *
from formattools import *
import plottools as pt

style = styles['prd_twocol']
pt.set_font()

from rhmc_coeffs import *

################################################################################
############################### INPUT PARAMETERS ###############################
################################################################################

L = 8                                       # Spatial size of the lattice
T = 8                                       # Temporal size of the lattice

DEFAULT_NC = 3                              # Default number of colors

eps = 1e-8                                  # Round-off for floating points.

cg_tol = 1e-12                              # CG error tolerance
cg_max_iter = 1000                          # Max number of iterations for CG

P = 15                                      # Degree of rational approximation for Dirac operator 
lambda_low = 1e-5                           # Smallest eigenvalue possible for K for valid rational approximation
lambda_high = 1000                          # Largest eigenvalue possible for K for valid rational approximation

alpha_m4, beta_m4 = rhmc_m4_15()            # Approximation coefficients for K^{-1/4}
alpha_8, beta_8 = rhmc_8_15()               # Approximation coefficients for K^{+1/8}

bcs = np.array([1, -1])                     # Boundary conditions for fermion fields

################################################################################
############################## UTILITY FUNCTIONS ###############################
################################################################################

def kron_delta(a, b):
    """Returns the Kronecker delta between two objects a and b."""
    if type(a) == np.ndarray:
        if np.array_equal(a, b):
            return 1
        return 0
    if a == b:
        return 1
    return 0

def hat(mu):
    """Returns the position vector \hat{\mu} for \mu = 0, 1, ..., d."""
    return np.array([kron_delta(mu, nu) for nu in range(d)])

class Lattice:
    """
    Represents a 2d lattice.
    """
    
    def __init__(self, l, t):
        self.L = l
        self.T = t
        self.LL = [l, t]
        self.vol = l * t

    def to_linear_momentum(self, k, datatype = np.complex128):
        return np.array([datatype(2 * np.pi * k[mu] / self.LL[mu]) for mu in range(d)])

    def to_lattice_momentum(self, k):
        return np.array([np.complex64(2 * np.sin(np.pi * k[mu] / self.LL[mu])) for mu in range(d)])

    def k_to_mu_p(self, k, A = .1167):
        aGeV = fm_to_GeV(A)
        return (2 * np.pi / aGeV) * np.sqrt(sum([(k[mu] / self.LL[mu]) ** 2 for mu in range(4)]))
    
    def mod(self, x):
        """Rounds a 2-position x to the range (0:L, 0:T)."""
        return np.array([x[0] % self.L, x[1] % self.T])
    
    def __str__(self):
        return f'{self.L} x {self.T} dimensional Lattice'

def id_field(Nc):
    U = np.zeros((d, LAT.L, LAT.T, Nc, Nc))
    for a in range(Nc):
        U[:, :, :, a, a] = 1
    return U

# def set_lat(l, t):
#     global LAT
#     LAT = Lattice(L, T)

################################################################################
################################## CONSTANTS ###################################
################################################################################

d = 2                                       # Spacetime dimensions
Ns = 2                                      # Spinor dimensions
LAT = Lattice(L, T)                         # Lattice object to use

delta = np.eye(Ns, dtype = np.complex128)    # Identity in spinor space
gamma = np.array([
    paulis[0], 
    paulis[2]
], dtype=np.complex128)                       # Euclidean gamma matrices gamma1, gamma2
gamma5 = paulis[1]
Pplus = (delta + gamma5) / 2                  # Positive chirality projector
Pminus = (delta - gamma5) / 2                 # Negative chirality projector

################################################################################
############################ GAUGE FIELD FUNCTIONS #############################
################################################################################

def get_generators(Nc):
    """Returns dimSUN generators of SU(N_c)"""
    if Nc == 2:
        return paulis / 2.
    if Nc == 3:
        return gell_mann / 2.
    if Nc == 4:
        return su4_paulis / 2.
    if Nc > 4:
        raise NotImplementedError('Need to implement more SU(N) generators.')

def trace(U):
    """
    Takes the trace over color indices of a gauge field U. Note this 
    works for a gauge field in the fundamental or adjoint representation, as well 
    as individual SU(N) matrices: the only criterion is that the shape of the array 
    must be square in its last 2 indices. 

    Parameters
    ----------
    U : np.array [... N, N]
        Gauge field array.
    
    Returns
    -------
    np.array [...]
        Trace of gauge field array (scalar field).
    """
    return np.einsum('...aa->...', U)

def dagger(U):
    """
    Takes the hermitian conjugate over color indices of a gauge field U. Note this 
    works for a gauge field in the fundamental or adjoint representation, as well 
    as individual SU(N) matrices: the only criterion is that the shape of the array 
    must be square in its last 2 indices. 

    Parameters
    ----------
    U : np.array [... N, N]
        Gauge field array.
    
    Returns
    -------
    np.array [... N, N]
        Hermitian conjugate of gauge field array.
    """
    return np.conjugate(np.einsum('...ab->...ba', U))

def plaquette(U, Nc):
    """
    Computes the plaquette of U (note that in a 2d lattice, there is only one direction 
    possible for the plaquette, P_{01}(x)) with Nc colors. Note the plaquette is normalized 
    by 1 / Nc, i.e. this function returns (1/Nc) Tr P(x).

    Parameters
    ----------
    U : np.array [2, Lx, Lt, Nc, Nc]
        Gauge field array.
    Nc : int
        Number of colors for the gauge field.

    Returns
    -------
    np.array [Lx, Lt]
        Wilson loop field (1/N_c) Tr P(x).
    """
    mu, nu = 0, 1
    Un_mu = U[mu]
    Unpmu_nu = np.roll(U[nu], -1, axis = mu)
    Unpnu_mu = np.roll(U[mu], -1, axis = nu)
    Un_nu = U[nu]

    return np.real(np.einsum('...ab,...bc,...cd,...da->...', 
        Un_mu, Unpmu_nu, dagger(Unpnu_mu), dagger(Un_nu)
    )) / Nc

def wilson_gauge_action(U, beta, Nc):
    """
    Computes the Wilson gauge action,
    $$
        S_g[U] = \beta \sum_{P} (1 - (1/Nc) Re Tr [P])
    $$
    where {P} is the set of plaquettes.

    Parameters
    ----------
    U : np.array [2, Lx, Lt, Nc, Nc]
        Gauge field array.
    beta : int
        Gauge coupling.
    Nc : int
        Number of colors for SU(Nc).
    
    Returns
    -------
    np.float64
        Value of the action for the configuration U.
    """
    plaqs = np.real(plaquette(U, Nc))
    return beta * np.sum(1 - plaqs)

def construct_adjoint_links(U, gens, lat = LAT):
    """
    Constructs the adjoint link variables
    $$
        V_\mu^{ab}(x) = 2 Tr [U_\mu^\dagger(x) t^a U_\mu(x) t^b]
    $$
    from the fundamental gauge links U.

    Parameters
    ----------
    U : np.array [2, Lx, Lt, Nc, Nc]
        Gauge field array.
    gens : np.array [Nc^2 - 1, Nc, Nc]
        Generators {t^a} of SU(Nc).
    """
    Nc = U.shape[-1]
    dimSUN = Nc**2 - 1
    V = np.zeros((d, lat.L, lat.T, dimSUN, dimSUN), dtype = np.complex128)
    for mu, x, t, a, b in itertools.product(*[range(zz) for zz in U.shape]):
        V[mu, x, t, a, b] = 2 * trace(dagger(U[mu, x, t]) @ gens[a] @ U[mu, x, t] @ gens[b])
    return V

################################################################################
############################## FERMION FUNCTIONS ###############################
################################################################################

def get_dirac_op_idxs(kappa, V, lat = LAT):
    """
    Returns the Dirac operator between two sets of indices ((a, alpha, x), (b, beta, y)), for a 
    configuration V in the adjoint representation. 

    Parameters
    ----------
    kappa : np.float64
        Hopping parameter.
    
    Returns
    -------
    function (V, x, y, a, b, alpha, beta)
        Dirac operator index function. Here x and y should be 2-positions (xx, tx) and (yy, ty).
    """
    # TODO bug here, Dirac operator should give zero if the sites aren't next to each other
    def dirac_op_idxs(x, a, alpha, y, b, beta):
        return kron_delta(a, b) * kron_delta(alpha, beta) - kappa * np.sum([
            V[mu, x[0], x[1], a, b] * (1 + gamma[mu][alpha, beta] * kron_delta(x, lat.mod(y + hat(mu))))
            + V[mu, x[0], x[1], b, a] * (1 - gamma[mu][alpha, beta] * kron_delta(x, lat.mod(y - hat(mu))))
        for mu in range(d)])
    return dirac_op_idxs

def get_dirac_op_full(kappa, V, lat = LAT):
    Nc = int(np.sqrt(V.shape[-1] + 1) + eps)
    dimSUN = Nc**2 - 1
    dirac_op_full = np.zeros((lat.L, lat.T, dimSUN, Ns, lat.L, lat.T, dimSUN, Ns), dtype = np.complex128)
    dirac_op_idxs = get_dirac_op_idxs(kappa, V, lat = LAT)
    for lx, tx, a, alpha, ly, ty, b, beta in itertools.product(*[range(zz) for zz in dirac_op_full.shape]):
        x, y = np.array([lx, tx]), np.array([ly, ty])
        dirac_op_full[lx, tx, a, alpha, ly, ty, b, beta] = dirac_op_idxs(x, a, alpha, y, b, beta)
    return dirac_op_full

def get_dirac_op_block(kappa, V, lat = LAT):
    """
    Returns a function D(x, y) which gives the Dirac operator from sites x to y as a 
    (Nc^2-1)*Ns by (Nc^2-1)*Ns matrix for a given configuration and kappa.
    """
    Nc = int(np.sqrt(V.shape[-1] + 1) + eps)
    dimSUN = Nc**2 - 1
    dirac_op_idxs = get_dirac_op_idxs(kappa, V, lat = LAT)
    def dirac_block(x, y):
        """x and y are spacetime 2-vectors. Returns the corresponding color-spin block of 
        the Dirac operator."""
        blk = np.zeros((dimSUN * Ns, dimSUN * Ns), dtype = np.complex128)
        for a, alpha, b, beta in itertools.product(*[range(zz) for zz in blk.shape]):
            ii, jj = flatten_colspin_idx(a, alpha), flatten_colspin_idx(b, beta)
            blk[ii, jj] = dirac_op_idxs(x, a, alpha, y, b, beta)
        return blk
    return dirac_block

def dirac_op_sparse(cfg, kappa, ):
    """
    Returns the Dirac operator D as a sparse matrix in the Block Compressed Row (BSR) format.

    TODO: check Arthur's implementations of dirac_op and hermitize_dirac_op. 

    Parameters
    ----------

    """
    return

"""
For sparse implementation of the Dirac operator, we'll store it in blocks indexed by each set of sites (n, m) 
that have non-zero hopping (or are diagonal). Each block will have size Nc * Ns, since it must have a spinor 
index and a color index.
"""

def flatten_spacetime_idx(idx, lat = LAT):
    """
    Flattens a 2d spatial index idx = (x, t) based on the lattice LAT. Counts first in the 
    spatial direction, then in the temporal direction. For example, (x, t) = (1, 2) on a 
    lattice of size (4, 8) gives flat index 1 + 2 * 4 = 9.
    """
    x, t = idx
    flat_idx = x + t * lat.L
    return flat_idx

def unflatten_spacetime_idx(flat_idx, lat = LAT):
    """
    Unflattens a spatial index flat_idx to return a tuple (x, t), where x is the corresponding 
    spatial index and t is the temporal index, corresponding to the lattice LAT. 
    """
    x, t = flat_idx % lat.L, flat_idx // lat.L
    return x, t

def flatten_colspin_idx(idx, dimSUN_rep):
    """Flattens a color-spin index idx = (a, alpha) based SU(N) representation with dimension dimSUN_rep."""
    a, alpha = idx
    flat_idx = a + alpha * dimSUN_rep
    return flat_idx

def unflatten_colspin_idx(flat_idx, dimSUN_rep):
    """Unflattens a color-spin index flat_idx based SU(N) representation with dimension dimSUN_rep."""
    a, alpha = flat_idx % dimSUN_rep, flat_idx // dimSUN_rep
    return a, alpha

# class SparseOperator(bsr_matrix):
#     """
#     Represents a sparse operator with two adjoint SU(N) indices, two spinor indices, and 
#     two 2d spacetime indices.
#     """

#     def __init__(self, lat, Nc):
#         # TODO method stub
#         return
    
#     def __getitem__(self, key):
#         """Gets an item from the SparseOperator. Should be indexed as [lx, tx, a, alpha, ly, ty, b, beta]."""
#         lx, tx, a, alpha, ly, ty, b, beta = key
#         # TODO method stub
#         return
    
#     def __setitem__(self, key, val):
#         lx, tx, a, alpha, ly, ty, b, beta = key
#         # TODO method stub
#         return
    
#     def flatten_index(idx):
#         # TODO method stub
#         return
    
#     def unflatten_index(flat_idx):
#         # TODO method stub
#         return

################################################################################
############################### RHMC FUNCTIONS #################################
################################################################################

def r_m4(K):
    """
    Rational approximation to K^{-1/4}.

    # TODO determine if we want this to work for matrix valued K
    """
    return alpha_m4[0] + np.sum(alpha_m4[1:] / (K + beta_m4[1:]))

def r_8(K):
    """
    Rational approximation to K^{+1/8}.
    """
    return alpha_8[0] + np.sum(alpha_8[1:] / (K + beta_8[1:]))

def init_fields():
    """
    Initializes pseudofermion and gauge fields. Pseudofermion fields should be initialized as \Phi = K^{1/8} g, 
    where g is a Gaussian random vector of dimension TODO.
    """


    return

def main(args):

    # Initialize gauge group
    if len(args) > 1:
        Nc = int(args[1])                       # Gauge group is SU(Nc)
    else:
        Nc = DEFAULT_NC
    dimSUN = Nc**2 - 1                          # Dimension of SU(N)
    tSUN = get_generators(Nc)                   # Generators of SU(N)

    print(Nc)
    print('TODO IMPLEMENT')

if __name__ == '__main__':
    main(sys.argv)