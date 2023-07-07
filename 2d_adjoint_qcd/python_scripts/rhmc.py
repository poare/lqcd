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
#   - Lattice Lambda of shape [L, T].                                          #
#   - Fundamental gauge field U of shape [d, L, T, Nc, Nc].                    #
#   - Adjoint gauge field V of shape [d, L, T, dNc, dNc].                      #
#   - Pseudofermion field Phi of shape [dNc, Ns, L, T].                        #
#   - Adjoint Majorana fermion field of shape [dNc, Ns, L, T].                 #
#   - Dirac operator of shape [dNc, Ns, L, T, dNc, Ns, L, T].                  #
################################################################################
# Author: Patrick Oare                                                         #
################################################################################

n_boot = 100

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import bsr_matrix, csr_matrix
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

EPS = 1e-8                                  # Round-off for floating points.

CG_TOL = 1e-12                              # CG error tolerance
CG_MAX_ITER = 1000                          # Max number of iterations for CG

P = 15                                      # Degree of rational approximation for Dirac operator 
lambda_low = 1e-5                           # Smallest eigenvalue possible for K for valid rational approximation
lambda_high = 1000                          # Largest eigenvalue possible for K for valid rational approximation

alpha_m4, beta_m4 = rhmc_m4_15()            # Approximation coefficients for K^{-1/4}
alpha_8, beta_8 = rhmc_8_15()               # Approximation coefficients for K^{+1/8}

DEFAULT_BCS = (1, -1)                     # Boundary conditions for fermion fields

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
    
    def get_delta_bcs(self, bcs = DEFAULT_BCS):
        """
        Returns the Kronecker delta for spacetime points x, y adjusted with a 
        sign for boundary conditions.
        
        Parameters
        ----------
        self : Lattice
            Lattice instance to use.
        bcs : tuple (int, int) (default = DEFAULT_BCS)
            Boundary conditions to use.

        Returns
        -------
        function
            Delta function, adjusted to boundary conditions.
        """
        def delta_fn(x, y):
            """Kronecker delta for spacetime points, adjusted with given boundary conditions."""
            if np.array_equal(x, y):                            # x and y are equal
                return 1
            if np.array_equal(self.mod(x), self.mod(y)):        # x and y are equal up to boundaries
                if x[0] % self.L == y[0] % self.L and x[1] == y[1]:
                    return bcs[0]
                if x[0] == y[0] and x[1] % self.T == y[1] % self.T:
                    return bcs[1]
            return 0                                            # x and y are not equal
        return delta_fn

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
############################# INDEXING FUNCTIONS ###############################
################################################################################

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

def flatten_colspin_vec(vec):
    """
    Flattens a color-spin vector of shape [dNc, Ns] into shape [dNc*Ns].
    """
    dNc = vec.shape[0]
    flat_vec = np.zeros((dNc*Ns), dtype = np.complex128)
    for a, alpha in itertools.product(*[range(zz) for zz in vec.shape]):
        idx = flatten_colspin_idx((a, alpha), dNc)
        flat_vec[idx] = vec[a, alpha]
    return flat_vec

def unflatten_colspin_vec(vec, dNc):
    """
    Unflattens a color-spin vector of shape [dNc*Ns] into shape [dNc, Ns].
    """
    unflattened = np.zeros((dNc, Ns), dtype = np.complex128)
    for idx in range(vec.shape[0]):
        a, alpha = unflatten_colspin_idx(idx, dNc)
        unflattened[a, alpha] = vec[idx]
    return unflattened

def flatten_ferm_field(vec, lat = LAT):
    """
    Flattens a fermion field of shape [dNc, Ns, L, T] into a fermion field 
    of shape [dNc*Ns*L*T].
    """
    dNc = vec.shape[0]
    flat_vec = np.zeros((dNc*Ns*lat.vol), dtype = np.complex128)
    for a, alpha, lx, tx in itertools.product(*[range(zz) for zz in vec.shape]):
        i = flatten_full_idx((a, alpha, lx, tx), dNc, lat = lat)
        flat_vec[i] = vec[a, alpha, lx, tx]
    return flat_vec

def unflatten_ferm_field(vec, dNc, lat = LAT):
    """
    Unflattens a vector of shape [dNc*Ns*L*T] into an fermion field
    of shape [dNc, Ns, L, T].
    """
    unflattened = np.zeros((dNc, Ns, lat.L, lat.T), dtype = np.complex128)
    for i in range(vec.shape[0]):
        a, alpha, lx, tx = unflatten_full_idx(i, dNc, lat = lat)
        unflattened[a, alpha, lx, tx] = vec[i]
    return unflattened

def flatten_operator(op, lat = LAT):
    """
    Flattens an operator of shape [dNc, Ns, L, T, dNc, Ns, L, T] into a matrix 
    of shape [dNc*Ns*L*T, dNc*Ns*L*T].
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
        unflattened[a, alpha, lx, tx, b, beta, ly, ty] = mat[i, j]
    return unflattened

def spin_to_spincol(spin_mat, dNc):
    """Converts a spin matrix (i.e. like gamma5) to a spin-color matrix."""
    assert spin_mat.shape == (Ns, Ns), 'Wrong dimension for spin matrix.'
    id_col = np.eye(dNc)
    return np.kron(spin_mat, id_col)

def col_to_spincol(col_mat):
    """Converts a color matrix (like t^a) to a spin-color matrix."""
    id_spin = np.eye(Ns)
    return np.kron(id_spin, col_mat)

def flat_field_evalat(psi, x, t, dNc, lat = LAT):
    """
    Evaluates the flattened fermion field psi (size dNc*Ns*L*T) at position (x, t).

    Parameters
    ----------
    psi : np.array [dNc*Ns*L*T]
        Fermion field to evaluate.
    x : int
        x coordinate to evaluate psi at.
    t : int
        t coordinate to evaluate psi at.

    Returns
    -------
    np.array [dNc*Ns]
        Flattened color-spin matrix psi(x, t).
    """
    ferm_block = np.zeros((dNc*Ns), dtype = psi.dtype)
    for colspin_idx in itertools.product(range(dNc), range(Ns)):
        a, alpha = colspin_idx
        ferm_block[flatten_colspin_idx(colspin_idx, dNc)] = psi[flatten_full_idx((a, alpha, x, t), dNc, lat = lat)]
    return ferm_block

def flat_field_putat(psi, blk, x, t, dNc, mutate = False, lat = LAT):
    """
    Puts the color-spin vector blk into the flat fermion field psi (size dNc*Ns*L*T) at position (x, t).

    Parameters
    ----------
    psi : np.array [dNc*Ns*L*T]
        Fermion field to put color-spin matrix into.
    blk : np.array [dNc*Ns]
        Color-spin matrix to put into psi.
    x : int
        x coordinate to put blk into psi at
    t : int
        t coordinate to put blk into psi at
    dNc: int
        Dimension of adjoint representation.
    mutate : bool (default = False)
        Whether or not to mutate the original matrix.

    Returns
    -------
    np.array [dNc*Ns]
        Flattened color-spin matrix psi(x, t).
    """
    if blk.shape == (dNc, Ns):
        blk = flatten_colspin_vec(blk)
    if mutate:
        new_psi = psi
    else:
        new_psi = np.copy(psi)
    for colspin_idx in itertools.product(range(dNc), range(Ns)):
        a, alpha = colspin_idx
        new_psi[flatten_full_idx((a, alpha, x, t), dNc, lat = lat)] = blk[flatten_colspin_idx(colspin_idx, dNc)]
    return new_psi

def get_colspin_blocks(D, dNc, lat = LAT):
    """
    Given a Dirac operator D, returns a matrix of color-spin blocks composing D. 
    D can be any of the following formats:
        1. A dense, unflattened Dirac operator.
        2. A dense, flattened Dirac operator.
        3. A sparse bsr_matrix.
    
    Parameters
    ----------
    D : np.array [dNc, Ns, L, T, dNc, Ns, L, T] or np.array [dNc*Ns*L*T, dNc*Ns*L*T] or bsr_matrix
        Dirac operator to return blocks of.
    dNc : int
        Dimension of adjoint representation.
    
    Returns
    -------
    np.array [L*T, L*T, dNc*Ns, dNc*Ns]
        Color-spin blocks of Dirac operator.
    """
    if len(D.shape) == 8:               # D is dense and unflattened
        D = flatten_operator(D, lat = lat)
    elif type(D) == bsr_matrix:
        D = D.toarray()
    block_size = dNc*Ns
    blocks = np.zeros((lat.vol, lat.vol, block_size, block_size), dtype = D.dtype)
    for i, j in itertools.product(range(lat.vol), repeat = 2):
        blocks[i, j] = D[i*block_size : (i + 1)*block_size, j*block_size : (j + 1)*block_size]
    return blocks

def get_permutation_D(dNc, lat = LAT):
    """
    Gets the permutation matrix P that transforms between color-spin blocking and 
    spacetime blocking for the original Dirac matrix D. Note that this should not be 
    used in the Pfaffian computation, since the Hermitian Dirac matrix Q = gamma_5 D 
    has a different structure than D. 

    Parameters
    ----------
    dNc : int
        Dimension of adjoint representation.
    lat : Lattice
        Lattice object to use.
    
    Returns
    -------
    csr_matrix [dNc*Ns*lat.vol, dNc*Ns*lat.vol]
        Permutation matrix P_{spacetime\leftarrow colspin} as a sparse matrix.
    """
    N_cs = dNc * Ns         # colspin block size
    N_sp = lat.vol          # spacetime block size
    N = N_cs * N_sp
    rows = list(range(N))
    cols = [(i % N_sp) * N_cs + (i // N_sp) for i in rows]
    data = np.ones(N, dtype = np.int8)
    return csr_matrix((data, (rows, cols)), shape = (N, N))

def get_permutation_Q(dNc, lat = LAT):
    """
    Gets the permutation matrix \tilde{P} that transforms between color-spin blocking and 
    spacetime blocking for the Hermitian Dirac matrix Q. This should be used in the 
    Pfaffian computation.

    Parameters
    ----------
    dNc : int
        Dimension of adjoint representation.
    lat : Lattice
        Lattice object to use.
    
    Returns
    -------
    csr_matrix [dNc*Ns*lat.vol, dNc*Ns*lat.vol]
        Permutation matrix P_{spacetime\leftarrow colspin} as a sparse matrix.
    csr_matrix [dNc*Ns*lat.vol, dNc*Ns*lat.vol]
        Inverse permutation matrix P_{spacetime\leftarrow colspin}^{-1} as a sparse matrix.
    """
    omega = (1/2)*(1 - 1j) * np.array([[1j, -1j], [1, 1]])
    omega_inv = (1/2)*(1 + 1j) * np.array([[-1j, 1], [1j, 1]])
    omega_colspin = np.kron(omega, np.eye(dNc))
    omega_inv_colspin = np.kron(omega_inv, np.eye(dNc))
    Omega = scipy.sparse.block_diag([omega_colspin for i in range(lat.vol)], format = 'csc')
    Omega_inv = scipy.sparse.block_diag([omega_inv_colspin for i in range(lat.vol)], format = 'csc')
    P = get_permutation_D(dNc, lat = lat)
    return Omega @ P.transpose(), P @ Omega_inv

def check_sparse_equal(A, B):
    """
    Checks that two scipy.sparse.bsr_matrix A and B are equal. Note that you cannot 
    simply use np.array_equal(A, B); you can compare the dense matrices with 
    np.array_equal(A.toarray(), B.toarray()), which should have the same output as 
    check_sparse_equal(A, B).
    """
    return (A != B).nnz == 0

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
    return construct_adjoint_links(U, gens, lat = lat)

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
    U : np.array [2, L, T, Nc, Nc]
        Gauge field array.
    Nc : int
        Number of colors for the gauge field.

    Returns
    -------
    np.array [L, T]
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
    U : np.array [2, L, T, Nc, Nc]
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
    U : np.array [2, L, T, Nc, Nc]
        Gauge field array.
    gens : np.array [Nc^2 - 1, Nc, Nc]
        Generators {t^a} of SU(Nc).
    """
    Nc = U.shape[-1]
    dNc = Nc**2 - 1
    V = np.zeros((d, lat.L, lat.T, dNc, dNc), dtype = np.complex128)
    for mu, x, t, a, b in itertools.product(*[range(zz) for zz in V.shape]):
        V[mu, x, t, a, b] = 2 * trace(dagger(U[mu, x, t]) @ gens[a] @ U[mu, x, t] @ gens[b])
    return V

################################################################################
############################## FERMION FUNCTIONS ###############################
################################################################################

def get_dirac_op_idxs(kappa, V, bcs = DEFAULT_BCS, lat = LAT):
    """
    Returns the Dirac operator between two sets of indices ((a, alpha, x), (b, beta, y)), for a 
    configuration V in the adjoint representation. 

    Parameters
    ----------
    kappa : np.float64
        Hopping parameter.
    V : np.array [d, L, T, dNc, dNc]
        Adjoint gauge field.
    bcs : tuple (int, int) (default = DEFAULT_BCS)
        Boundary conditions to satisfy. 1 for periodic and -1 for antiperiodic.
    lat : Lattice (default = LAT)
        Lattice to work with.
    
    Returns
    -------
    function (a, alpha, x, b, beta, y) --> np.complex128
        Dirac operator index function. Here x and y should be 2-positions (xx, tx) and (yy, ty).
    """
    delta_bcs = lat.get_delta_bcs(bcs)
    def dirac_op_idxs(a, alpha, x, b, beta, y):
        # return kron_delta(a, b) * kron_delta(alpha, beta) * kron_delta(x, y) - kappa * np.sum([
        #         V[mu, x[0], x[1], a, b] * (kron_delta(alpha, beta) + gamma[mu][alpha, beta]) * kron_delta(x, lat.mod(y + hat(mu)))
        #         + V[mu, x[0], x[1], b, a] * (kron_delta(alpha, beta) - gamma[mu][alpha, beta]) * kron_delta(x, lat.mod(y - hat(mu)))
        #     for mu in range(d)])
        return kron_delta(a, b) * kron_delta(alpha, beta) * kron_delta(x, y) - kappa * np.sum([
                V[mu, x[0], x[1], a, b] * (kron_delta(alpha, beta) + gamma[mu][alpha, beta]) * delta_bcs(x, y + hat(mu))
                + V[mu, x[0], x[1], b, a] * (kron_delta(alpha, beta) - gamma[mu][alpha, beta]) * delta_bcs(x, y - hat(mu))
            for mu in range(d)])
    return dirac_op_idxs

def get_dirac_op_full(kappa, V, bcs = DEFAULT_BCS, lat = LAT):
    """
    Returns the full Dirac operator as a numpy array. This should only be used to check the sparse 
    Dirac operator, or when the lattice is very small. 

    Parameters
    ----------
    kappa : np.float64
        Hopping parameter.
    V : np.array [d, L, T, dNc, dNc]
        Adjoint gauge field
    
    Returns
    -------
    np.array [dNc, Ns, L, T, dNc, Ns, L, T]
        Dirac operator.
    """
    dNc = V.shape[-1]                       # dimension of adjoint rep of SU(Nc)
    dirac_op_full = np.zeros((dNc, Ns, lat.L, lat.T, dNc, Ns, lat.L, lat.T), dtype = np.complex128)
    dirac_op_idxs = get_dirac_op_idxs(kappa, V, bcs = bcs, lat = lat)
    for a, alpha, lx, tx, b, beta, ly, ty in itertools.product(*[range(zz) for zz in dirac_op_full.shape]):
        x, y = np.array([lx, tx]), np.array([ly, ty])
        dirac_op_full[a, alpha, lx, tx, b, beta, ly, ty] = dirac_op_idxs(a, alpha, x, b, beta, y)
    return dirac_op_full

def get_dirac_op_block(kappa, V, bcs = DEFAULT_BCS, lat = LAT):
    """
    Returns a function D(x, y) which gives the Dirac operator from sites x to y as a 
    dNc*Ns by dNc*Ns matrix for a given configuration and kappa.

    Parameters
    ----------
    kappa : np.float64
        Hopping parameter.
    V : np.array [d, L, T, dNc, dNc]
        Adjoint gauge field.
    
    Returns
    -------
    function (x, y) --> np.array [dNc*Ns, dNc*Ns]
        Function to extract the blocked Dirac operator at x, y.
    """
    dNc = V.shape[-1]
    dirac_op_idxs = get_dirac_op_idxs(kappa, V, bcs = bcs, lat = lat)
    def dirac_block(x, y):
        """x and y are spacetime 2-vectors. Returns the corresponding color-spin block of 
        the Dirac operator."""
        blk = np.zeros((dNc * Ns, dNc * Ns), dtype = np.complex128)
        for a, alpha, b, beta in itertools.product(range(dNc), range(Ns), repeat = 2):
            ii, jj = flatten_colspin_idx((a, alpha), dNc), flatten_colspin_idx((b, beta), dNc)
            blk[ii, jj] = dirac_op_idxs(a, alpha, x, b, beta, y)
        return blk
    return dirac_block

def dirac_op_sparse(kappa, V, bcs = DEFAULT_BCS, lat = LAT):
    """
    Returns the Dirac operator D as a sparse matrix in the Block Compressed Row (BSR) format.
    This is done by blocking the Dirac operator by spacetime dimension, since it only connects 
    x and y if they have taxicab metric <= 1 (in lattice units). We have lat.vol blocks 
    (for concreteness, 16) of size dNc*Ns (for concreteness for SU(2), size 6).

    TODO deal with boundary conditions

    Parameters
    ----------
    kappa : np.float64
        Hopping parameter.
    V : np.array [d, L, T, dNc, dNc]
        Adjoint gauge field.
    bcs : tuple (int, int) (default = DEFAULT_BCS)
        Boundary conditions to satisfy. 1 for periodic and -1 for antiperiodic.
    lat : Lattice (default = LAT)
        Lattice to work with.
    
    Returns
    -------
    bsr_matrix
        Sparse Dirac operator.
    """
    dNc = V.shape[-1]
    dim_dirac = dNc * Ns * lat.vol
    dirac_block = get_dirac_op_block(kappa, V, bcs = bcs, lat = lat)
    indptr = [0]
    indices = []
    data = []
    for flat_x in range(lat.vol):                   # Traverse x in lexigraphical order
        x = np.array(unflatten_spacetime_idx(flat_x, lat = lat))
        for flat_y in range(lat.vol):
            y = np.array(unflatten_spacetime_idx(flat_y, lat = lat))
            # Here add the sign for bcs
            if lat.next_to_equal(x, y):             # then fill the block in
                block = dirac_block(x, y)           # (dNc*Ns) x (dNc*Ns) block
                data.append(block)
                y_flat = flatten_spacetime_idx(y, lat = lat)
                indices.append(y_flat)
        indptr.append(len(indices))                 # next row starts at next index.
    dirac_op = bsr_matrix((data, indices, indptr), shape = (dim_dirac, dim_dirac))
    return dirac_op

def dagger_op(dirac):
    """
    Returns the Hermitian conjugate of the input operator. Note that the transpose is taken 
    over all indices.

    Parameters
    ----------
    dirac : np.array [dNc, Ns, L, T, dNc, Ns, L, T] or scipy.sparse.bsr_matrix
        Input operator. Can either be a numpy array or a sparse matrix.

    Returns
    -------
    np.array [dNc, Ns, L, T, dNc, Ns, L, T] or scipy.sparse.bsr_matrix
        Hermitian conjugate of input operator.
    """
    if type(dirac) == bsr_matrix:
        return dirac.conj().transpose()
    return np.einsum('aixtbjys->bjysaixt', dirac.conj())

def hermitize_dirac(dirac):
    """
    Returns the Hermitian Dirac operator Q = \gamma_5 D. Note that because we can take C = gamma5, 
    this operator should also be the skew-symmetric Dirac matrix M = C D. 

    Parameters
    ----------
    dirac : np.array [dNc, Ns, L, T, dNc, Ns, L, T] or scipy.sparse.bsr_matrix
        Input Dirac operator. Can either be a numpy array or a sparse matrix.

    Returns
    -------
    np.array [dNc, Ns, L, T, dNc, Ns, L, T] or scipy.sparse.bsr_matrix
        Hermitian Dirac operator Q = gamma_5 D.
    """
    if type(dirac) == bsr_matrix:               # Then dirac is a sparse matrix
        dNc = dirac.blocksize[0] // Ns
        g5_spincol = spin_to_spincol(gamma5, dNc)
        return bsr_matrix((
            [g5_spincol @ blk for blk in dirac.data],
            dirac.indices, 
            dirac.indptr
        ), shape = dirac.shape)
    return np.einsum('ij,ajxtbkys->aixtbkys', gamma5, dirac)

def construct_K(dirac):
    """
    Constructs the squared Dirac operator K = D^\dagger D from the Dirac operator D.

    Parameters
    ----------
    dirac : np.array [dNc, Ns, L, T, dNc, Ns, L, T] or scipy.sparse.bsr_matrix
        Input Dirac operator or Hermitian Dirac operator. Can either be a numpy array or a sparse matrix.
    
    Returns
    -------
    np.array [dNc, Ns, L, T, dNc, Ns, L, T] or scipy.sparse.bsr_matrix
        Squared Dirac operator K = D^\dagger D.
    """
    dirac_dagger = dagger_op(dirac)
    if type(dirac) == bsr_matrix:
        return dirac_dagger @ dirac
    return np.einsum('aixtbjys,bjysclzr->aixtclzr', dirac_dagger, dirac)

################################################################################
############################### RHMC FUNCTIONS #################################
################################################################################

def r_m4(K):
    """
    Rational approximation to K^{-1/4}. Note that for matrices, this will likely be very slow 
    and should not be used. Instead, whenever r(K) appears, it should be applied to a vector and 
    solved using CG.
    """
    if type(K) == np.ndarray:
        return alpha_m4[0] + np.sum(alpha_m4[1:] * np.linalg.inv(K + beta_m4[1:]))
    return alpha_m4[0] + np.sum(alpha_m4[1:] / (K + beta_m4[1:]))

def r_8(K):
    """
    Rational approximation to K^{+1/8}.
    """
    if type(K) == np.ndarray:
        return alpha_8[0] + np.sum(alpha_8[1:] * np.linalg.inv(K + beta_8[1:]))
    return alpha_8[0] + np.sum(alpha_8[1:] / (K + beta_8[1:]))

def cg_shift(K, phi, beta_i, cg_tol = CG_TOL, max_iter = CG_MAX_ITER):
    """
    Solves the linear equation (K + beta_i) psi = phi with a CG solver for a sparse 
    matrix K.

    Parameters
    ----------
    K : scipy.sparse.bsr_matrix [dNc*Ns*L*T, dNc*Ns*L*T]
        Sparse squared Dirac operator K = D^\dagger D.
    phi : np.array [dNc*Ns*L*T]
        Input source to solve (K + \beta_i)^{-1} phi.
    beta_i : float
        Shift for the CG solver.
    cg_tol : float (default = CG_TOL)
        Relative tolerance for CG solver.
    max_iter : int (default = CG_MAX_ITER)
        Maximum iterations for CG solver. 
    
    Returns
    -------
    np.array [dNc*Ns*L*T]
        Solution psi to the equation (K + \beta_i) \psi = \phi.
    
    Raises
    ------
    Exception
        Raised if the CG solver does not converge, or has illegal input.
    """
    dim = K.shape[0]
    assert phi.shape[0] == dim, 'Phi is the wrong dimension.'
    shifted_K = K + beta_i * scipy.sparse.identity(dim, dtype = K.dtype)
    psi, info = scipy.sparse.linalg.cg(shifted_K, phi, tol = cg_tol, maxiter = max_iter)
    if info > 0:
        raise Exception('Convergence not achieved.')
    elif info < 0:
        raise Exception('Illegal input to CG solver.')
    return psi

def apply_rational_approx(K, phi, alphas, betas, cg_tol = CG_TOL, max_iter = CG_MAX_ITER):
    """
    Applies the rational approximation r(K) to the vector phi. Note this is valid for 
    rational approximations of K^{-1/4} and K^{1/8}, it only depends on the input alpha 
    parameters.

    Parameters
    ----------
    K : scipy.sparse.bsr_matrix [dNc*Ns*L*T, dNc*Ns*L*T]
        Sparse squared Dirac operator K = D^\dagger D.
    phi : np.array [dNc*Ns*L*T]
        Input source to solve (K + \beta_i)^{-1} phi.
    alphas : np.array [P + 1]
        Alpha coefficients for CG solver.
    betas : np.array [P + 1]
        Beta coefficients for CG solver. beta[0] should equal 0.
    cg_tol : float (default = CG_TOL)
        Relative tolerance for CG solver.
    max_iter : int (default = CG_MAX_ITER)
        Maximum iterations for CG solver. 
    
    Returns
    -------
    np.array [dNc*Ns*L*T]
        Result of r(K) \Phi, where r(K) is a rational approximation given in terms of alphas and betas.
    """
    rKphi = alphas[0] * phi
    for i in range(1, len(betas)):
        psi_i = cg_shift(K, phi, betas[i], cg_tol, max_iter)
        rKphi += alphas[i] * psi_i
    return rKphi

def pf_force(dirac, U, phi, gens, kappa, alphas = alpha_m4, betas = beta_m4, lat = LAT, cg_tol = CG_TOL, max_iter = CG_MAX_ITER, bcs = DEFAULT_BCS):
    """
    Computes the pseudofermion force d/dU (Phi^\dagger r(K) \Phi) \approx d/dU (Phi^\dagger K^{-1/4} \Phi).

    Parameters
    ----------
    dirac : scipy.sparse.bsr_matrix [dNc*Ns*L*T, dNc*Ns*L*T]
        Sparse Dirac operator D.
    U : np.array [d, L, T, Nc, Nc]
        Fundamental gauge field configuration.
    phi : np.array [dNc*Ns*L*T]
        Input (flattened) pseudofermion force. 
    gens : np.array [dNc, Nc, Nc]
        SU(Nc) generators t^a.
    kappa : float
        Hopping parameter for action. TODO change to a dictionary "action_args"
    alphas : np.array [P + 1] (default = alpha_m4)
        Alpha coefficients for approximation of K^{-1/4}.
    betas : np.array [P + 1] (default = beta_m4)
        Beta coefficients for approximation of K^{-1/4}. beta[0] should equal 0.
    cg_tol : float (default = CG_TOL)
        Relative tolerance for CG solver.
    max_iter : int (default = CG_MAX_ITER)
        Maximum iterations for CG solver. 
    
    Returns
    -------
    dKdU : np.array [d, L, T, Nc, Nc]
        Derivative of pseudofermion part of action by the fundamental gauge field.
    """
    Q = hermitize_dirac(dirac)
    K = construct_K(dirac)
    force = np.zeros(U.shape, U.dtype)
    for i in range(1, len(betas)):
        psi_i = cg_shift(K, phi, betas[i], cg_tol, max_iter)
        psi_dKdU_psi = form_dKdU_bilinear(U, Q, psi_i, gens, kappa, lat = lat, bcs = bcs)
        force += alphas[i] * psi_dKdU_psi
    return force

def form_dKdU_bilinear(U, Q, psi, gens, kappa, lat = LAT, bcs = DEFAULT_BCS):
    """
    For given psi = (K + \beta_i)^{-1}\Phi, evaluates the bilinear \psi^\dagger dK / dU_\mu^{k\ell}(z) psi, 
    where psi = (K + \beta_i)^{-1}\Phi. Returns a field in the same shape as U_\mu(x).
    
    Parameters
    ----------
    U : np.array [d, L, T, Nc, Nc]
        Fundamental gauge field.
    psi : np.array [dNc*Ns*L*T]
        Solution psi to the equation (K + \beta_i) \psi = \phi.
    gens : np.array [dNc, Nc, Nc]
        SU(Nc) generators t^a.
    kappa : float
        Hopping parameter for action. TODO change to a dictionary

    Returns
    -------
    np.array [d, L, T, Nc, Nc]
        Derivative dK/dU contracted with psi^\dagger and psi. Shape should be that 
        of a fundamental gauge field.
    """
    # kappa = action_args['kappa']
    dKdU_bilinear = np.zeros(U.shape, U.dtype)
    Qpsi = Q @ psi
    Qpsi_dagger = Qpsi.conj().transpose()
    tUt = np.einsum('aik,mxtkl,blj->abmxtij', gens, U, gens)
    for deriv_coord in itertools.product(*[range(ii) for ii in U.shape]):
        Mmu_psi_i = form_Mmu_psi(deriv_coord, tUt, psi, gens, kappa, lat = lat, bcs = bcs)
        dKdU_bilinear[deriv_coord] = 2 * Qpsi_dagger @ Mmu_psi_i
    return dKdU_bilinear

def form_Mmu_psi(deriv_coord, tUt, psi, gens, kappa, lat = LAT, bcs = DEFAULT_BCS):
    """
    Evaluate M_\mu psi_i, which is formally dQ/dU_\mu contracted with psi_i, at the point deriv_coord. 
    This must be modified for a given Dirac operator.

    Parameters
    ----------
    deriv_coord : tuple [int, int, int, int, int]
        Coordinates to take derivative at, passed in as a tuple (mu, x, t, k, ell). Here mu
        is the Lorentz index, (k, \ell) are fundamental color indices, and (x, t) is a spacetime 
        index.
    tUt : np.array [dNc, dNc, d, L, T, Nc, Nc]
        Fundamental gauge field conjugated by t^a and t^b.
    psi : np.array [dNc*Ns*L*T]
        Solution psi to the equation (K + \beta_i) \psi = \phi.
    gens : np.array [dNc, Nc, Nc]
        SU(Nc) generators t^a.
    kappa : float
        Hopping parameter for action. TODO change to a dictionary

    Returns
    -------
    np.array [dNc*Ns*L*T]
        M psi, evaluated at the point (mu, k, \ell, xz, tz) = deriv_coord.
    """
    dNc = tUt.shape[0]
    mu, xz, tz, k, ell = deriv_coord
    z = np.array([xz, tz])

    # TODO might need to make this more efficient
    # get psi at z \pm \hat\mu
    tUt_coords = tUt[:, :, mu, xz, tz, k, ell]
    zpmu = lat.mod(z + hat(mu))
    psi_zpmu = unflatten_colspin_vec(
        flat_field_evalat(psi, zpmu[0], zpmu[1], dNc, lat = lat),
    dNc)
    psi_z = unflatten_colspin_vec(
        flat_field_evalat(psi, z[0], z[1], dNc, lat = lat),
    dNc)
    Mpsi_blk1 = np.einsum('ab,ij,bjxt->aixt', tUt_coords, delta - gamma[mu], psi_zpmu)
    Mpsi_blk2 = np.einsum('ba,ij,bjxt->aixt', tUt_coords, delta + gamma[mu], psi_z)
    
    Mpsi = np.zeros((dNc*Ns*lat.vol), dtype = np.complex128)
    flat_field_putat(Mpsi, Mpsi_blk1, z[0], z[1], dNc, mutate = True)
    flat_field_putat(Mpsi, Mpsi_blk2, zpmu[0], zpmu[1], dNc, mutate = True)

    return (-2*kappa) * Mpsi

def pseudofermion_action(K, phi):
    """
    Returns the pseudofermion action, 
    """
    return

def init_fields(Nc, lat = LAT):
    """
    Initializes pseudofermion and gauge fields. Pseudofermion fields should be initialized as \Phi = K^{1/8} g, 
    where g is a Gaussian random vector of dimension TODO.
    """

    U = id_field(Nc, lat = lat)
    
    
    return

def pfaffian(Q):
    """
    Computes the Pfaffian of a flattened (antisymmetric) Dirac operator Q. Q can be sparse or dense. 
    Uses the LU decomposition of Q = P J P^T, where J has trivial Pfaffian, and computes pf(Q) = pf(P)^2
    = det(P) = \prod_i P_{ii}. 

    Parameters
    ----------
    Q : np.array [n, n] or bsr_matrix or csr_matrix
        Input Hermitian Dirac operator, or arbitrary antisymmetric matrix. 
        Can either be a numpy array or a sparse matrix.
    
    Returns
    -------
    np.complex128
        Pfaffian of Q.
    """
    if type(Q) == bsr_matrix:
        assert check_sparse_equal(Q.transpose(), -Q), 'Pfaffian only defined for antisymmetric matrices.'
        Q = csr_matrix(Q)               # CSR format is better for indexing
    elif type(Q) == csr_matrix:
        assert check_sparse_equal(Q.transpose(), -Q), 'Pfaffian only defined for antisymmetric matrices.'
    else:
        assert np.array_equal(Q.transpose(), -Q), 'Pfaffian only defined for antisymmetric matrices.'
    n = Q.shape[0]
    P, _, Pi = lu_decomp(Q, compute_J = False)
    pf = np.prod([P[i, i] for i in range(n)])           # Pfaffian is product of diagonal elements.
    return pf

def cyclic_perm_matrix(perm, N = -1):
    """
    Returns the permutation matrix associated to the string perm = (i, j, ..., k). 
    This matrix is constructed as a sparse scipy matrix, and should be orthogonal. 
    Embeds perm into an N x N array. For example, cyclic_perm_matrix((1, 2), 4) 
    embeds the 2x2 matrix [[0, 1], [1, 0]] into a 4x4 array.

    Parameters
    ----------
    perm : tuple (int)
        Permutation to use. Should be a tuple of non-repeating integers, as in 
        the standard notation for cyclic permutations in the symmetric group (see 
        also https://en.wikipedia.org/wiki/Cyclic_permutation).
    N : int (default = -1)
        Dimension of matrix to embed permutation matrix into. If -1, N will 
        default to len(perm)
    
    Returns
    -------
    P : scipy.sparse.csr_matrix (dtype = np.int8)
    """
    if N == -1:
        N = len(perm)
    data = np.ones(N, dtype = np.int8)                  # Always have N ones
    row = [i for i in range(N) if i not in perm]
    col = row.copy()
    m = len(perm)
    for i in range(m):
        row.append(perm[i])
        ip1 = (i + 1) % m
        col.append(perm[ip1])
    return scipy.sparse.csr_matrix((data, (row, col)), shape = (N, N))

def lu_decomp(A, compute_J = True):
    """
    Returns the LU decomposition of the n x n matrix A. This follows the algorithm 
    presented in hep-lat/1102.3576v2. A is decomposed as A = P J P^T, where P is a 
    lower triangular matrix and J is a tridiagonal matrix with non-zero entries either
    +1 or -1, with trivial Pfaffian. In the case where the columns of A do not allow 
    the algorithm to be used directly, partial pivoting is used on A to form A' = 
    Pi @ A @ Pi.transpose(), where Pi is a permutation matrix. After partial pivoting, 
    we also consider P' = Pi @ P, which has permuted rows compared to the original P.

    Note that since the resulting P is lower triangular, its density is around 50%, 
    hence a sparse representation is actually not a useful way to implement this decomposition. 
    However, J is tridiagonal, hence has density ~ 3 / n, hence will be returned as a sparse matrix. 

    Parameters
    ----------
    A : np.array [n, n] or csr_matrix or bsr_matrix
        Input matrix. Can either be a numpy matrix or a sparse matrix. Must be skew-symmetric 
        and non-singular.
    compute_J : bool (default = True)
        Flag to specify whether or not to compute J. If compute_J is False, returns None for J.
    
    Returns
    -------
    P : np.array [n, n]
        The lower triangular matrix P in the decomposition A = P J P^T.
    J : scipy.sparse.csr_matrix [n, n]
        The trivial tridiagonal matrix J in the decomposition A = P J P^T
    Pi : scipy.sparse.csr_matrix [n, n]
        Pivoting matrix. Equals the identity if no pivoting is used. 
    """
    if type(A) == bsr_matrix:
        A = csr_matrix(A)
    if type(A) == csr_matrix:
        assert check_sparse_equal(A.transpose(), -A), 'A is not skew-symmetric.'
    else:
        assert np.array_equal(A.transpose(), -A), 'A is not skew-symmetric.'
    n = A.shape[0]
    A0 = A.copy()
    p = np.zeros((n, n), dtype = A.dtype)
    for i in np.arange(0, n - 1, 2):
        p[i, i] = 1
    Pi = scipy.sparse.identity(n, dtype = np.int8, format = 'csr')
    for i in np.arange(0, n - 1, 2):                    # Cycle over column pairs
        for j in np.arange(i + 1, n):                   # Update column i + 1
            p[j, i + 1] = A[i, j]
            for k in np.arange(0, i - 1, 2):
                p[j, i + 1] -= p[i, k] * p[j, k + 1] - p[i, k + 1] * p[j, k]
        pip1 = p[i + 1, i + 1]
        if np.abs(pip1) < EPS:                          # PIVOT
            jmax = np.argmax(np.abs(p[:, i + 1]))       # Pivot i+1 <--> jmax
            pip1 = p[jmax, i + 1]                       # Reset pip1

            # swap rows. Note that to form A' and P', we take A' = Pi @ A @ Pi.transpose() and P' = Pi @ P
            tau = cyclic_perm_matrix((i + 1, jmax), n)
            Pi = tau @ Pi                               # keep track of full permutation (product of all transpositions)
            A = tau @ A @ tau.transpose()
            p = tau @ p
        for j in np.arange(i + 2, n):                   # Update column i
            p[j, i] = A[i + 1, j]
            for k in np.arange(0, i - 1, 2):
                p[j, i] -= p[i + 1, k] * p[j, k + 1] - p[i + 1, k + 1] * p[j, k]
            p[j, i] = - p[j, i] / pip1
    if compute_J:
        data, rows, cols = [], [], []
        for i in range(n):
            data.append((-1)**i)
            rows.append(i)
            cols.append(i + (-1)**i)
        J = scipy.sparse.csr_matrix((data, (rows, cols)), shape = (n, n))
    else:
        J = None
    return p, J, Pi

# def lu_decomp(Q, dNc, lat = LAT):
#     """
#     Returns the LU decomposition of Q. This follows the algorithm 
#     presented in Sec. 4.4 of hep-lat/1410.6971. Note that the naive Hermitian Wilson-Dirac 
#     operator satisfies Q[i, i + 1] = 0 in the color-spin blocking defined above, which is 
#     incompatible with the algorithm presented in the paper. Q is thus basis transformed to a 
#     spacetime-blocked matrix Q_sp, and Q_sp is decomposed as Gamma Q_sp Gamma^T = T, where 
#     where T is a tri-diagonal matrix and Gamma is the inverse of a lower triangular matrix L.

#     Note that since the resulting Q is lower triangular, its density is around 50%, 
#     hence a sparse representation is actually not a useful way to implement this decomposition.

#     Parameters
#     ----------
#     Q : np.array [dNc*Ns*L*T, dNc*Ns*L*T] or scipy.sparse.bsr_matrix
#         Input Hermitian Dirac operator. Can either be a numpy matrix or a sparse matrix.
#     dNc : int
#         Dimension of adjoint representation.
#     lat : Lattice
#         Lattice to work on.
    
#     Returns
#     -------
#     Gamma : np.array [dNc*Ns*L*T, dNc*Ns*L*T]
#         The inverse of L.
#     Q_sp : np.array [dNc*Ns*L*T, dNc*Ns*L*T] or scipy.sparse.bsr_matrix
#         The Hermitian Dirac operator in the spacetime blocking. Note this satisfies 
#         the decomposition Gamma Q_sp Gamma^T = T.
#     """
#     N = Q.shape[0]
#     Gamma = np.eye(N, dtype = Q.dtype)
#     Ptilde, Ptilde_inv = get_permutation_Q(dNc, lat = lat)
#     Q_sp = Ptilde_inv @ Q @ Ptilde
#     print(Q_sp.toarray()[:10, :10])
#     for i in np.arange(0, N - 2, 2):                    # Cycle over column pairs
#         Gam_i, Gam_ip1 = Gamma[:, i], Gamma[:, i + 1]     # Columns of Gamma
#         if i < 10:
#             print(f'Gamma_{i}:')
#             print(Gam_i)
#             print(f'Gamma_{i+1}:')
#             print(Gam_ip1)
#         norm = Gam_ip1 @ Q_sp @ Gam_i
#         print(f'Norm for col {i}: {norm}')
#         Gam_ip1 = Gam_ip1 / norm
#         Gamma[:, i + 1] = Gam_ip1
#         for j in range(i + 2, N):
#             Gam_j = Gamma[:, j]
#             Gamma[:, j] = Gam_j - (Gam_ip1 @ Q_sp @ Gam_j) * Gam_i - (Gam_i @ Q_sp @ Gam_j) * Gam_ip1
#     return Gamma, Q_sp

################################################################################
############################## __MAIN__ FUNCTION ###############################
################################################################################

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