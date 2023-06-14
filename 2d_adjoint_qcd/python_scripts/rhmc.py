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
#   - Dimension of adjoint of SU(N) dNc = Nc^2 - 1.                            #
#   - Lattice Lambda of shape [Lx, Lt].                                        #
#   - Fundamental gauge field U of shape [d, Lx, Lt, Nc, Nc].                  #
#   - Adjoint gauge field V of shape [d, Lx, Lt, dNc, dNc].                    #
#   - Scalar (pseudofermion) field Phi of shape [d, Lx, Lt].                   #
#   - Adjoint Majorana fermion field of shape [Lx, Lt, dNc, Ns].               #
#   - Dirac operator of shape [dNc, Ns, Lx, Lt, dNc, Ns, Lx, Lt].              #
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
    
    def taxicab_distance(self, x, y):
        """Computes the periodic taxicab distance between x and y."""
        dx = np.abs(x - y)
        if dx[0] > self.L // 2:
            dx[0] = self.L - dx[0]
        if dx[1] > self.T // 2:
            dx[1] = self.T - dx[1]
        return np.sum(dx)

    def next_to(self, x, y):
        """Returns True if the 2-positions x and y are nearest neighbors."""
        # return np.sum(np.abs(x - y)) == 1
        # return np.sum(np.abs(x % self.LL - y % self.LL)) == 1
        return self.taxicab_distance(x, y) == 1
    
    def next_to_equal(self, x, y):
        """Returns true if the 2-positions x and y are either nearest neighbors or are equal."""
        return self.taxicab_distance(x, y) <= 1

    def __str__(self):
        return f'{self.L} x {self.T} dimensional Lattice'

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

def id_field(Nc, lat = LAT):
    """Constructs an identity SU(Nc) field in the fundamental representation."""
    U = np.zeros((d, lat.L, lat.T, Nc, Nc))
    for a in range(Nc):
        U[:, :, :, a, a] = 1
    return U

def id_field_adjoint(Nc, lat = LAT):
    """Returns an identity field in the adjoint representation of SU(N)."""
    gens = get_generators(Nc)
    U = id_field(Nc, lat = lat)
    return construct_adjoint_links(U, gens, lat = LAT)

def get_generators(Nc):
    """Returns dNc generators of SU(N_c)"""
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
    dNc = Nc**2 - 1
    V = np.zeros((d, lat.L, lat.T, dNc, dNc), dtype = np.complex128)
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
    V : np.array [d, Lx, Lt, dNc, dNc]
        Adjoint gauge field
    TODO: should we add a mass parameter?
    
    Returns
    -------
    function (a, alpha, x, b, beta, y) --> np.complex128
        Dirac operator index function. Here x and y should be 2-positions (xx, tx) and (yy, ty).
    """
    def dirac_op_idxs(a, alpha, x, b, beta, y):
        return kron_delta(a, b) * kron_delta(alpha, beta) * kron_delta(x, y) - kappa * np.sum([
                V[mu, x[0], x[1], a, b] * (1 + gamma[mu][alpha, beta]) * kron_delta(x, lat.mod(y + hat(mu)))
                + V[mu, x[0], x[1], b, a] * (1 - gamma[mu][alpha, beta]) * kron_delta(x, lat.mod(y - hat(mu)))
            for mu in range(d)])
    return dirac_op_idxs

def get_dirac_op_full(kappa, V, lat = LAT):
    """
    Returns the full Dirac operator as a numpy array. This should only be used to check the sparse 
    Dirac operator, or when the lattice is very small. 

    Parameters
    ----------
    kappa : np.float64
        Hopping parameter.
    V : np.array [d, Lx, Lt, dNc, dNc]
        Adjoint gauge field
    
    Returns
    -------
    np.array [dNc, Ns, L, T, dNc, Ns, L, T]
        Dirac operator.
    """
    dNc = V.shape[-1]                       # dimension of adjoint rep of SU(Nc)
    dirac_op_full = np.zeros((dNc, Ns, lat.L, lat.T, dNc, Ns, lat.L, lat.T), dtype = np.complex128)
    dirac_op_idxs = get_dirac_op_idxs(kappa, V, lat = LAT)
    for a, alpha, lx, tx, b, beta, ly, ty in itertools.product(*[range(zz) for zz in dirac_op_full.shape]):
        x, y = np.array([lx, tx]), np.array([ly, ty])
        dirac_op_full[a, alpha, lx, tx, b, beta, ly, ty] = dirac_op_idxs(a, alpha, x, b, beta, y)
    return dirac_op_full

def get_dirac_op_block(kappa, V, lat = LAT):
    """
    Returns a function D(x, y) which gives the Dirac operator from sites x to y as a 
    dNc*Ns by dNc*Ns matrix for a given configuration and kappa.

    Parameters
    ----------
    kappa : np.float64
        Hopping parameter.
    V : np.array [d, Lx, Lt, dNc, dNc]
        Adjoint gauge field
    
    Returns
    -------
    function (x, y) --> np.array [dNc*Ns, dNc*Ns]
        Function to extract the blocked Dirac operator at x, y.
    """
    dNc = V.shape[-1]
    dirac_op_idxs = get_dirac_op_idxs(kappa, V, lat = LAT)
    def dirac_block(x, y):
        """x and y are spacetime 2-vectors. Returns the corresponding color-spin block of 
        the Dirac operator."""
        blk = np.zeros((dNc * Ns, dNc * Ns), dtype = np.complex128)
        for a, alpha, b, beta in itertools.product(range(dNc), range(Ns), repeat = 2):
            ii, jj = flatten_colspin_idx((a, alpha), dNc), flatten_colspin_idx((b, beta), dNc)
            blk[ii, jj] = dirac_op_idxs(a, alpha, x, b, beta, y)
        return blk
    return dirac_block

def dirac_op_sparse(kappa, V, lat = LAT):
    """
    Returns the Dirac operator D as a sparse matrix in the Block Compressed Row (BSR) format.
    This is done by blocking the Dirac operator by spacetime dimension, since it only connects 
    x and y if they have taxicab metric <= 1 (in lattice units). We have lat.vol blocks 
    (for concreteness, 16) of size dNc*Ns (for concreteness for SU(2), size 6).

    Parameters
    ----------
    kappa : np.float64
        Hopping parameter.
    V : np.array [d, Lx, Lt, dNc, dNc]
        Adjoint gauge field
    
    Returns
    -------
    bsr_matrix
        Sparse Dirac operator.
    """
    
    # TODO deal with boundary conditions

    dNc = V.shape[-1]
    dim_dirac = dNc * Ns * lat.vol
    dirac_block = get_dirac_op_block(kappa, V, lat = lat)
    indptr = [0]
    indices = []
    data = []
    for flat_x in range(lat.vol):                   # Traverse x in lexigraphical order
        x = np.array(unflatten_spacetime_idx(flat_x, lat = lat))
        for flat_y in range(lat.vol):
            y = np.array(unflatten_spacetime_idx(flat_y, lat = lat))
            if lat.next_to_equal(x, y):             # then fill the block in
                block = dirac_block(x, y)           # (dNc*Ns) x (dNc*Ns) block
                data.append(block)
                y_flat = flatten_spacetime_idx(y, lat = lat)
                indices.append(y_flat)
        indptr.append(len(indices))                 # next row starts at next index.
    dirac_op = bsr_matrix((data, indices, indptr), shape = (dim_dirac, dim_dirac))
    return dirac_op

def flatten_full_idx(idx, dNc, lat = LAT):
    """Flattens a full color-spin-spacetime index idx = (a, alpha, x, t)."""
    a, alpha, x, t = idx
    flat_idx = a + dNc * (alpha + Ns * (x + lat.L * t))
    return flat_idx

def unflatten_full_idx(flat_idx, dNc, lat = LAT):
    """Unflattens a flat color-spin-spacetime index flat_idx."""
    t = flat_idx // (dNc * Ns * lat.L)
    flat_idx -= t * (dNc * Ns * lat.L)
    x = flat_idx // (dNc * Ns)
    flat_idx -= x * (dNc * Ns)
    alpha = flat_idx // dNc
    a = flat_idx % dNc
    return a, alpha, x, t

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

def flatten_colspin_idx(idx, dNc):
    """Flattens a color-spin index idx = (a, alpha) based SU(N) representation with dimension dNc."""
    a, alpha = idx
    flat_idx = a + alpha * dNc
    return flat_idx

def unflatten_colspin_idx(flat_idx, dNc):
    """Unflattens a color-spin index flat_idx based SU(N) representation with dimension dNc."""
    a, alpha = flat_idx % dNc, flat_idx // dNc
    return a, alpha

def flatten_operator(op, lat = LAT):
    """
    Flattens an operator of shape [dNc, Ns, L, T, dNc, Ns, L, T] into a matrix 
    of shape [dNc*Ns*L*T, dNc*Ns*L*T]
    """
    dNc = op.shape[0]
    flat_op = np.zeros((dNc*Ns*lat.vol, dNc*Ns*lat.vol), dtype = np.complex128)
    for a, alpha, lx, tx, b, beta, ly, ty in itertools.product(*[range(zz) for zz in op.shape]):
        i = flatten_full_idx((a, alpha, lx, tx), dNc, lat = lat)
        j = flatten_full_idx((b, beta, ly, ty), dNc, lat = lat)
        flat_op[i, j] = op[a, alpha, lx, tx, b, beta, ly, ty]
    return flat_op

def unflatten_operator(mat, dNc, lat = LAT):
    """
    Unflattens a matrix of shape [dNc*Ns*L*T, dNc*Ns*L*T] into an operator
    of shape [dNc, Ns, L, T, dNc, Ns, L, T].
    """
    unflattened = np.zeros((dNc, Ns, lat.L, lat.T, dNc, Ns, lat.L, lat.T), dtype = np.complex128)
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        a, alpha, lx, tx = unflatten_full_idx(i, dNc, lat = lat)
        b, beta, ly, ty = unflatten_full_idx(j, dNc, lat = lat)

        # TODO test this tomorrow to make sure it works
        # print(j, (b, beta, ly, ty))

        unflattened[a, alpha, lx, tx, b, beta, ly, ty] = mat[i, j]
    return unflattened

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
    dNc = Nc**2 - 1                          # Dimension of SU(N)
    tSUN = get_generators(Nc)                   # Generators of SU(N)

    print(Nc)
    print('TODO IMPLEMENT')

if __name__ == '__main__':
    main(sys.argv)