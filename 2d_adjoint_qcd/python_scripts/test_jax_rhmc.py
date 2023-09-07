################################################################################
# Tests an implementation of RHMC. Note that tests related to derivatives will #
# be performed in the file test_autodiff.py, as they require a slightly more   # 
# involved setup. Tests to perform:                                            #
#   0. Test accept-reject of each possibility.                                 #
#   1. Gauge coupling g --> 0 limit (free theory)                              #
#   2. g --> and m --> infinity (pure gauge theory)                            #
################################################################################
# Author: Patrick Oare                                                         #
################################################################################

import numpy
import jax.numpy as np
from jax import grad, jit, vmap
import itertools
import scipy
from scipy.sparse import bsr_matrix, csr_matrix

# np.random.seed(20)

from jax import random
seed = 20

import jax_rhmc as rhmc

def print_line():
    print('-'*50)

def is_equal(a, b, eps = rhmc.EPS):
    """Returns whether a and b are equal up to precision eps."""
    return np.abs(a - b) < eps

# def check_sparse_equal(A, B):
#     """
#     Checks that two scipy.sparse.bsr_matrix A and B are equal. Note that you cannot 
#     simply use np.array_equal(A, B); you can compare the dense matrices with 
#     np.array_equal(A.toarray(), B.toarray()), which should have the same output as 
#     check_sparse_equal(A, B).
#     """
#     return (A != B).nnz == 0

def test_gauge_tools():
    """Tests utility functions for gauge fields."""
    Nc = 3
    dNc = Nc**2 - 1                                  # Dimension of SU(N)
    tSUN = rhmc.get_generators(Nc)                   # Generators of SU(N)
    delta_ab = np.eye(dNc)
    for a, b in itertools.product(range(dNc), repeat = 2):
        tr_ab = rhmc.trace(tSUN[a] @ tSUN[b])
        assert is_equal(tr_ab, (1/2) * delta_ab[a, b]), f'Tr[t^a t^b] != (1/2)\delta_ab at (a, b) = ({a}, {b}).'
    print('test_gauge_tools : Pass')

def test_gauge_field_properties(L = 4, T = 4, Nc = 2, U = None, V = None):
    """Tests that the gauge field satisfies assumed properties. A fundamental gauge field U 
    should be unitary with determinant 1. An adjoint gauge field should be real
    """
    dNc = Nc**2 - 1
    gens = rhmc.get_generators(Nc)
    Lat = rhmc.Lattice(L, T)
    # print(V[0, 0, 0])
    if U is None:
        U = rhmc.id_field(Nc)
    if V is None:
        V = rhmc.construct_adjoint_links(U, gens)
    for mu, x, t in itertools.product(range(rhmc.d), range(L), range(T)):
        assert np.allclose(U[mu, x, t] @ rhmc.dagger(U)[mu, x, t], np.eye(Nc, dtype = np.complex64)), f'U is not unitary at ({x}, {t}).'
        assert np.allclose(rhmc.dagger(U)[mu, x, t] @ U[mu, x, t], np.eye(Nc, dtype = np.complex64)), f'U is not unitary at ({x}, {t}).'
        assert np.allclose(V.imag, np.zeros((rhmc.d, L, T, dNc, dNc))), 'V is not real.'
        assert np.allclose(V[mu, x, t] @ rhmc.dagger(V)[mu, x, t], np.eye(dNc, dtype = np.complex64), atol = 1e-5), f'V is not unitary at ({x}, {t}).'
        assert np.allclose(rhmc.dagger(V)[mu, x, t] @ V[mu, x, t], np.eye(dNc, dtype = np.complex64), atol = 1e-5), f'V is not unitary at ({x}, {t}).'
    print('test_gauge_field_properties : Pass')

################################################################################
###################### TEST SPARSE MATRIX IMPLEMENTATIONS ######################
################################################################################

def test_next_to():
    Lat = rhmc.Lattice(4, 4)
    x1 = np.array([2, 2])
    valid_y1 = [(2, 2), (2, 3), (3, 2), (2, 1), (1, 2)]
    x2 = np.array([0, 0])
    valid_y2 = [(0, 0), (0, 1), (1, 0), (0, 3), (3, 0)]
    for ly, ty in itertools.product(range(4), repeat = 2):
        y = (ly, ty)
        if Lat.next_to_equal(x1, np.array(y)):
            assert y in valid_y1, f'{x1} and {np.array(y)} are not next to one another.'
        else:
            assert y not in valid_y1, f'{x1} and {np.array(y)} are next to one another.'
        if Lat.next_to_equal(x2, np.array(y)):
            assert y in valid_y2, f'{x2} and {np.array(y)} are not next to one another.'
        else:
            assert y not in valid_y2, f'{x2} and {np.array(y)} are next to one another.'
    print('test_next_to : Pass')


def test_flatten_spacetime():
    """Tests the flatten and unflatten functions for spacetime indices."""
    Lat = rhmc.Lattice(4, 8)
    assert rhmc.flatten_spacetime_idx((2, 1), lat = Lat) == 6, 'flatten_spacetime_idx not correct.'
    assert rhmc.unflatten_spacetime_idx(11, lat = Lat) == (3, 2), 'unflatten_spacetime_idx not correct.'
    for flat_idx in range(Lat.vol):
        assert rhmc.flatten_spacetime_idx(rhmc.unflatten_spacetime_idx(flat_idx, lat = Lat), lat = Lat) == flat_idx, \
            f'Flatten and unflatten not inverses at flat_idx = {flat_idx}.'
    print('test_flatten_spacetime : Pass')

def test_flatten_colspin():
    """Tests the flatten and unflatten functions for color-spin indices."""
    Nc = 3
    dNc = Nc**2 - 1
    assert rhmc.flatten_colspin_idx((6, 1), dNc) == 14, 'flatten_colspin_idx not correct.'
    assert rhmc.unflatten_colspin_idx(3, dNc) == (3, 0), 'unflatten_colspin_idx not correct.'
    assert rhmc.unflatten_colspin_idx(10, dNc) == (2, 1), 'unflatten_colspin_idx not correct.'
    for flat_idx in range(dNc * rhmc.Ns):
        assert rhmc.flatten_colspin_idx(rhmc.unflatten_colspin_idx(flat_idx, dNc), dNc) == flat_idx, \
            f'Flatten and unflatten not inverses at flat_idx = {flat_idx}.'
    print('test_flatten_colspin : Pass')

def test_flatten_full():
    """Tests the flatten and unflatten functions for full color-spin-spacetime indices."""
    Lat = rhmc.Lattice(4, 8)
    Nc = 2
    dNc = Nc**2 - 1
    assert rhmc.flatten_full_idx((2, 0, 3, 5), dNc, lat = Lat) == 140, 'flatten_full_idx not correct.'
    assert rhmc.unflatten_full_idx(82, dNc, lat = Lat) == (1, 1, 1, 3), 'unflatten_full_idx not correct.'
    for flat_idx in range(dNc * rhmc.Ns * Lat.vol):
        assert rhmc.flatten_full_idx(rhmc.unflatten_full_idx(flat_idx, dNc, lat = Lat), dNc, lat = Lat) == flat_idx, \
            f'Flatten and unflatten not inverses at flat_idx = {flat_idx}.'
    print('test_flatten_full : Pass')

def test_zeros_full_dirac(L = 4, T = 4, Nc = 2, kappa = 0.1, V = None):
    """Tests that the full Dirac operator has zeros in the correct places."""
    Lat = rhmc.Lattice(L, T)
    if V is None:
        V = rhmc.id_field_adjoint(Nc, lat = Lat)
    dirac_op = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
    for lx, tx, ly, ty in itertools.product(range(L), range(T), repeat = 2):
        x, y = np.array([lx, tx]), np.array([ly, ty])
        if Lat.next_to_equal(x, y):
            continue
        submat = dirac_op[:, :, lx, tx, :, :, ly, ty]
        assert not np.any(submat), f'Dirac operator is nonzero outside of nearest neighbors.'
    print('test_zeros_full_dirac : Pass')

def test_zeros_sparse_dirac(L = 4, T = 4, Nc = 2, kappa = 0.1, V = None):
    """Tests that the sparse Dirac operator has zeros in the correct places."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    if V is None:
        V = rhmc.id_field_adjoint(Nc, lat = Lat)
    sparse_dirac_op = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
    dirac_op = rhmc.unflatten_operator(sparse_dirac_op.toarray(), dNc, lat = Lat)
    for lx, tx, ly, ty in itertools.product(range(L), range(T), repeat = 2):
        x, y = np.array([lx, tx]), np.array([ly, ty])
        if Lat.next_to_equal(x, y):
            continue
        submat = dirac_op[:, :, lx, tx, :, :, ly, ty]
        assert not np.any(submat), f'Dirac operator is nonzero outside of nearest neighbors at x = {x}, y = {y}.'
    print('test_zeros_sparse_dirac : Pass')

def test_bcs(L = 4, T = 4, Nc = 2, kappa = 0.1, V = None):
    """Tests that the sparse Dirac operator has zeros in the correct places."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    if V is None:
        V = rhmc.id_field_adjoint(Nc, lat = Lat)
    dirac0 = rhmc.dirac_op_sparse(kappa, V, bcs = (1, -1), lat = Lat).toarray()
    dirac1 = rhmc.dirac_op_sparse(kappa, V, bcs = (1, 1), lat = Lat).toarray()
    dirac2 = rhmc.dirac_op_sparse(kappa, V, bcs = (-1, -1), lat = Lat).toarray()
    for i, j in itertools.product(range(dirac0.shape[0]), repeat = 2):
        x = np.array(rhmc.unflatten_full_idx(i, dNc, lat = Lat))[2:]
        y = np.array(rhmc.unflatten_full_idx(j, dNc, lat = Lat))[2:]
        if not Lat.next_to_equal(x, y):
            continue
        if np.abs(x[0] - y[0]) > 1:         # then we have crossed the spatial boundary
            assert dirac0[i, j] == dirac1[i, j], f'Periodic spatial bcs not working at ({x}, {y}).'
            assert dirac0[i, j] == -dirac2[i, j], f'Antiperiodic spatial bcs not working at ({x}, {y}).'
        elif np.abs(x[1] - y[1]) > 1:
            assert dirac0[i, j] == -dirac1[i, j], f'Periodic temporal bcs not working at ({x}, {y}).'
            assert dirac0[i, j] == dirac2[i, j], f'Antiperiodic temporal bcs not working at ({x}, {y}).'
        else:
            assert dirac0[i, j] == dirac1[i, j], f'Bulk values not equal for dirac0 and dirac1 at ({x}, {y}).'
            assert dirac0[i, j] == dirac2[i, j], f'Bulk values not equal for dirac0 and dirac2 at ({x}, {y}).'
    print('test_bcs : Pass')

def test_sparse_dirac(L = 4, T = 4, Nc = 2, kappa = 0.1, V = None):
    """
    Compares the sparse Dirac operator against a full Dirac operator on a small lattice.
    TODO this runs very slowly with JAX, despite being basically the same operation. Find out why.
    """
    Lat = rhmc.Lattice(L, T)
    if V is None:
        V = rhmc.id_field_adjoint(Nc, lat = Lat)
    # print(V)
    sparse_dirac = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
    full_dirac = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
    full_dirac_flat = rhmc.flatten_operator(full_dirac, lat = Lat)
    assert np.array_equal(sparse_dirac.toarray(), full_dirac_flat), \
        'Sparse matrix construction disagrees with full Dirac operator.'
    print('test_sparse_dirac : Pass')

def test_herm_dirac_full(L = 4, T = 4, Nc = 2, kappa = 0.1, V = None):
    """Confirms that the full Dirac operator D is gamma5-Hermitian."""
    Lat = rhmc.Lattice(L, T)
    if V is None:
        V = rhmc.id_field_adjoint(Nc, lat = Lat)
    dirac = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
    dirac_conj = np.einsum('ij,ajxtbkys,kl->aixtblys', rhmc.gamma5, dirac, rhmc.gamma5)
    dirac_dagger = rhmc.dagger_op(dirac)
    # # assert np.array_equal(dirac_conj, dirac_dagger), 'Full Dirac operator is not gamma5-hermitian.'
    # # print(np.max(np.abs(dirac_conj - dirac_dagger)))
    assert np.allclose(dirac_conj, dirac_dagger), 'Full Dirac operator is not gamma5-hermitian.'

    Q = rhmc.hermitize_dirac(dirac)
    Q_dagger = rhmc.dagger_op(Q)
    # assert np.array_equal(Q, Q_dagger), f'Q is not Hermitian.'
    # print(rhmc.flatten_operator(Q, lat = Lat)[:6, 24:30])
    # print(rhmc.flatten_operator(Q, lat = Lat)[24:30, :6])
    assert np.allclose(Q, Q_dagger), f'Q is not Hermitian.'

    print('test_herm_dirac_full : Pass')

def test_herm_dirac_sparse(L = 4, T = 4, Nc = 2, kappa = 0.1, V = None):
    """Confirms that the sparse Hermitian Dirac operator Q = gamma5 D is, in fact, Hermitian."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    if V is None:
        V = rhmc.id_field_adjoint(Nc, lat = Lat)
    sparse_dirac = rhmc.dirac_op_sparse(kappa, V, lat = Lat)

    g5_spincol = rhmc.spin_to_spincol(rhmc.gamma5, dNc)
    sparse_dirac_conj = bsr_matrix((
        [g5_spincol @ blk @ g5_spincol for blk in sparse_dirac.data], 
        sparse_dirac.indices, 
        sparse_dirac.indptr
        ), sparse_dirac.shape)
    sparse_dirac_dagger = rhmc.dagger_op(sparse_dirac)
    dirac = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
    dirac_conj = np.einsum('ij,ajxtbkys,kl->aixtblys', rhmc.gamma5, dirac, rhmc.gamma5)
    dirac_dagger = rhmc.dagger_op(dirac)
    assert np.allclose(sparse_dirac_conj.toarray(), rhmc.flatten_operator(dirac_conj, lat = Lat)), 'Sparse gamma5-conjugate != full gamma5-conjugate.'
    assert np.allclose(sparse_dirac_dagger.toarray(), rhmc.flatten_operator(dirac_dagger, lat = Lat)), 'Sparse dagger != full dagger.'
    assert np.allclose(sparse_dirac_conj.toarray(), sparse_dirac_dagger.toarray()), 'Sparse gamma5-conjugate != sparse Hermitian conjugate'

    Q = rhmc.hermitize_dirac(sparse_dirac)
    assert np.allclose(Q.toarray(), Q.conj().transpose().toarray()), 'Q is not Hermitian.'
    assert np.allclose(Q.toarray(), rhmc.dagger_op(Q).toarray()), 'Q is not Hermitian.'
    assert rhmc.check_sparse_allclose(Q, Q.conj().transpose()), 'Q is not Hermitian.'
    print('test_herm_dirac_sparse : Pass')

def test_skew_sym_sparse(L = 4, T = 4, Nc = 2, kappa = 0.1, V = None):
    """Tests the skew-symmetric Dirac operator is skew-symmetric."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    if V is None:
        V = rhmc.id_field_adjoint(Nc, lat = Lat)
    sparse_dirac = rhmc.dirac_op_sparse(kappa, V, lat = Lat)

    Q = rhmc.hermitize_dirac(sparse_dirac)
    assert rhmc.check_sparse_allclose(Q.transpose(), -Q), 'Q is not skew-symmetric.'
    print('test_skew_sym_sparse : Pass')

def test_squared_dirac(L = 4, T = 4, Nc = 2, kappa = 0.1, V = None):
    """Tests that the sparse construction and full construction of the squared Dirac 
    operator are equal."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    if V is None:
        V = rhmc.id_field_adjoint(Nc, lat = Lat)
    sparse_dirac = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
    dense_dirac = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
    sparse_K = rhmc.construct_K(sparse_dirac)
    dense_K = rhmc.construct_K(dense_dirac)
    dense_K_flat = rhmc.flatten_operator(dense_K, lat = Lat)
    assert np.allclose(sparse_K.toarray(), dense_K_flat), 'Sparse K != dense K.'
    assert rhmc.check_sparse_allclose(sparse_K, rhmc.dagger_op(sparse_K)), 'K is not Hermitian.'
    print('test_squared_dirac : Pass')

def test_eval_flat_ferm(L = 4, T = 4, Nc = 2):
    """Tests that the eval_flat_ferm_field and flat_field_putat functions work as intended."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    # psi_dense = np.random.rand(dNc, rhmc.Ns, L, T) + 1j*np.random.rand(dNc, rhmc.Ns, L, T)

    key = random.PRNGKey(seed)
    k0, k1, k2, k3, k4, k5 = random.split(key, 6)
    psi_dense = random.uniform(k0, shape = (dNc, rhmc.Ns, L, T)) + 1j*random.uniform(k1, shape = (dNc, rhmc.Ns, L, T))
    psi_flat = rhmc.flatten_ferm_field(psi_dense, lat = Lat)
    assert np.array_equal(rhmc.unflatten_ferm_field(psi_flat, dNc, lat = Lat), psi_dense), 'Sparse psi != dense psi.'

    # colspin_dense = np.random.rand(dNc, rhmc.Ns) + 1j*np.random.rand(dNc, rhmc.Ns)
    colspin_dense = random.uniform(k2, shape = (dNc, rhmc.Ns)) + 1j*random.uniform(k3, shape = (dNc, rhmc.Ns))
    colspin_flat = rhmc.flatten_colspin_vec(colspin_dense)
    assert np.array_equal(rhmc.unflatten_colspin_vec(colspin_flat, dNc), colspin_dense), 'Sparse colspin != dense colspin.'

    x, t = 1, 2
    psi_blk_dense = psi_dense[:, :, x, t]
    psi_blk_flat = rhmc.flat_field_evalat(psi_flat, x, t, dNc, lat = Lat)
    assert np.array_equal(rhmc.unflatten_colspin_vec(psi_blk_flat, dNc), psi_blk_dense), 'Sparse psi @ (x, t) != dense psi @ (x, t).'

    # rand_colspin = np.random.rand(dNc, rhmc.Ns) + 1j*np.random.rand(dNc, rhmc.Ns)
    rand_colspin = random.uniform(k4, shape = (dNc, rhmc.Ns)) + 1j*random.uniform(k5, shape = (dNc, rhmc.Ns))
    psi2_dense = np.zeros((dNc, rhmc.Ns, L, T), dtype = np.complex64)
    psi2_flat0 = np.zeros((dNc*rhmc.Ns*L*T), dtype = np.complex64)
    psi2_dense.at[:, :, x, t].set(rand_colspin)
    psi2_flat = rhmc.flat_field_putat(psi2_flat0, rand_colspin, x, t, dNc, mutate = False, lat = Lat)
    assert np.array_equal(rhmc.unflatten_ferm_field(psi2_flat, dNc, lat = Lat), psi2_dense), 'Evalat1: sparse psi @ (x, t) != dense psi @ (x, t).'
    psi3_flat = rhmc.flat_field_putat(psi2_flat0, rhmc.flatten_colspin_vec(rand_colspin), x, t, dNc, mutate = False, lat = Lat)
    assert np.array_equal(rhmc.unflatten_ferm_field(psi3_flat, dNc, lat = Lat), psi2_dense), 'Evalat2: sparse psi @ (x, t) != dense psi @ (x, t).'

    # Test mutation
    assert np.array_equal(psi2_flat0, np.zeros((dNc*rhmc.Ns*L*T), dtype = np.complex64)), 'Original field has been mutated.'
    rhmc.flat_field_putat(psi2_flat0, rand_colspin, x, t, dNc, mutate = True, lat = Lat)
    assert np.array_equal(rhmc.unflatten_ferm_field(psi2_flat0, dNc, lat = Lat), psi2_dense), 'Original field not mutated.'

    print('test_eval_flat_ferm : Pass')


################################################################################
############################### TEST CG SOLVER #################################
################################################################################
def test_shift_cg(L = 4, T = 4, Nc = 2, kappa = 0.1, V = None):
    """Tests the shifted CG solver by explicitly computing (K + \beta_i)^{-1} \Phi on 
    a small lattice."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    if V is None:
        V = rhmc.id_field_adjoint(Nc, lat = Lat)
    dirac = csr_matrix(rhmc.dirac_op_sparse(kappa, V, lat = Lat))
    dim = dirac.shape[0]

    _, betas = rhmc.rhmc_m4_5()         # Test on some of the actual betas.
    for beta in betas:
        K = rhmc.construct_K(dirac)
        K_shift = K + beta * scipy.sparse.identity(K.shape[0], dtype = K.dtype)
        K_shift_inv = scipy.sparse.linalg.inv(K_shift)

        phi = np.random.rand(dim) + 1j*np.random.rand(dim)
        psi = K_shift_inv @ phi
        psi_cg = rhmc.cg_shift(K, phi, beta)
        # TODO maybe at some point try to edit the CG solver to drop the warnings.
        assert np.allclose(psi, psi_cg, rtol = rhmc.CG_TOL), 'Actual inverse and CG inverse disagree.'
    print('test_shift_cg : Pass')

def test_rK_application(L = 4, T = 4, Nc = 2, kappa = 0.1, V = None):
    """Tests an application of r(K) to a random vector phi on a small lattice."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    if V is None:
        V = rhmc.id_field_adjoint(Nc, lat = Lat)
    dirac = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
    K = rhmc.construct_K(dirac)
    dim = K.shape[0]
    phi = np.random.rand(dim) + 1j*np.random.rand(dim)

    # Exact K^{-1/4} Phi
    K_dense = K.toarray()
    eigs = np.abs(np.linalg.eig(K_dense)[0])
    assert np.max(eigs) < 50 and np.min(eigs) > 0.1, 'Spectral range of K is too large.'
    K_m4 = scipy.linalg.fractional_matrix_power(K_dense, -1/4)
    Km4_phi_exact = K_m4 @ phi

    # Rational approximation
    alphas, betas = rhmc.rhmc_m4_5()
    rKphi = rhmc.apply_rational_approx(K, phi, alphas, betas)
    assert np.allclose(rKphi, Km4_phi_exact, rtol = 1e-4, atol = 2e-5), 'Rational approximation disagrees with K^{-1/4} Phi.'

    print('test_rK_application : Pass')

def test_init_fields(L = 4, T = 4, Nc = 2, kappa = 0.1, V = None):
    """Tests that the fields are initialized correctly."""
    Lat = rhmc.Lattice(L, T)
    gens = rhmc.get_generators(Nc)
    dNc = Nc**2 - 1
    if V is None:
        V = rhmc.id_field_adjoint(Nc, lat = Lat)
    dirac = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
    K = rhmc.construct_K(dirac)

    # TODO method stub, finish implementing tomorrow
    phi, U, V, Pi = rhmc.init_fields(K, Nc, gens, hot_start = False, lat = Lat)

    print('test_init_fields : Pass')

def test_pseudoferm_action(L = 4, T = 4, Nc = 2, kappa = 0.1, V = None):
    """Tests that the pseudofermion action Phi^\dagger K^{-1/4} \Phi obtained using 
    a CG solver is accurate by explicitly computing this quantity with a full matrix."""

    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    if V is None:
        V = rhmc.id_field_adjoint(Nc, lat = Lat)
    dirac = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
    K = rhmc.construct_K(dirac)
    dim = K.shape[0]
    phi = np.random.rand(dim) + 1j*np.random.rand(dim)

    K_m4 = scipy.linalg.fractional_matrix_power(K.toarray(), -1/4)
    S_full = - phi.conj().transpose() @ (K_m4 @ phi)

    S_cg = rhmc.pseudofermion_action(dirac, phi)
    assert np.allclose(S_cg, S_full, rtol = 1e-4, atol = 2e-5), 'Rational approximation disagrees with exact pseudofermion action.'

    print('test_pseudoferm_action : Pass')

def test_dDdw():
    """Uses autodiff to check that \partial D / \partial\omega is computed correctly for the 
    Wilson action."""

    # TODO method stub

    print('test_dDdw : Pass')

def test_pseudoferm_force(L = 4, T = 4, Nc = 2, kappa = 0.1, V = None):
    """Tests that the pseudofermion derivative of Phi^\dagger K^{-1/4} \Phi is accurate 
    by using automatic differentation on U_mu(x). Note that we can test this by setting a 
    bunch of elements to a fixed number, and using autodiff on a specific parameter."""

    # TODO method stub

    print('test_pseudoferm_force : Pass')

def test_gauge_force_wilson(L = 4, T = 4, Nc = 2, kappa = 0.1, U = None):
    """Tests that the derivative of the Wilson gauge action with respect to \omega_\mu^a(n) 
    is computed correctly by using autodifferentiation."""

    d = 2
    Lat = rhmc.Lattice(L, T)
    gens = rhmc.get_generators(Nc)
    dNc = Nc**2 - 1
    if U is None:
        U = rhmc.id_field(Nc, lat = Lat)
    var_idx = tuple([np.random.choice(sz) for sz in [d, L, T, dNc]])        # index for \omega_\mu^a(n)

    
    # TODO method stub

    S = rhmc.wilson_gauge_action()

    print('test_gauge_force_wilson : Pass')


################################################################################
######################### TEST RATIONAL APPROXIMATION ##########################
################################################################################
def test_rational_approx_small(n_samps = 100, delta = 2e-5):
    """
    Tests that |r(K) - K^{-1/4}| < \delta, where \delta is the predicted error for 
    the eigenvalue range of K. Uses the smallest spectral range available, which has 
    P = 5 partial fractions and spectral range [0.1, 50].

    Parameters
    ----------
    n_samps : int (default = 1000)
        Number of sample K matrices to test maximum error of approximation with.
    delta : float (default = 2e-5)
        Error tolerance on rational approximation.
    """
    alpha4, beta4 = rhmc.rhmc_m4_5()       # Spectral range: [0.1, 50]
    alpha8, beta8 = rhmc.rhmc_8_5()
    def r4(x):
        return alpha4[0] + np.sum(alpha4[1:] / (x + beta4[1:]))
    def pw4(x):
        return x**(-1/4)
    def r8(x):
        return alpha8[0] + np.sum(alpha8[1:] / (x + beta8[1:]))
    def pw8(x):
        return x**(1/8)
    for x in np.linspace(0.1, 50, num = n_samps):
        assert np.abs((r4(x) - pw4(x)) / r4(x)) < delta, f'x = {x}, x^(-1/4) = {r4(x)}, r(x) = {pw4(x)}'
        assert np.abs((r8(x) - pw8(x)) / r8(x)) < delta, f'x = {x}, x^(1/8) = {r8(x)}, r(x) = {pw8(x)}'
    print('test_rational_approx_small : Pass')

def test_rational_approx_large(n_samps = 100, delta = 2e-5):
    """
    Tests that |r(K) - K^{-1/4}| < \delta, where \delta is the predicted error for 
    the eigenvalue range of K. Uses the smallest spectral range available, which has 
    P = 15 partial fractions and spectral range [1e-7, 1000].

    Parameters
    ----------
    n_samps : int (default = 1000)
        Number of sample K matrices to test maximum error of approximation with.
    """
    alpha4, beta4 = rhmc.rhmc_m4_15()       # Spectral range: [1e-7, 1000]
    alpha8, beta8 = rhmc.rhmc_8_15()
    def r4(x):
        return alpha4[0] + np.sum(alpha4[1:] / (x + beta4[1:]))
    def pw4(x):
        return x**(-1/4)
    def r8(x):
        return alpha8[0] + np.sum(alpha8[1:] / (x + beta8[1:]))
    def pw8(x):
        return x**(1/8)
    for x in np.linspace(1e-7, 1000, num = n_samps):
        assert np.abs((r4(x) - pw4(x)) / r4(x)) < delta, f'x = {x}, x^(-1/4) = {r4(x)}, r(x) = {pw4(x)}'
        assert np.abs((r8(x) - pw8(x)) / r8(x)) < delta, f'x = {x}, x^(1/8) = {r8(x)}, r(x) = {pw8(x)}'
    print('test_rational_approx_large : Pass')


################################################################################
################################ TEST PFAFFIAN #################################
################################################################################
def is_tridiagonal(T):
    """Returns True if T is tridiagonal."""
    N = T.shape[0]
    for i in range(N):
        for j in range(N):
            if i == j or i == j - 1 or i == j + 1:
                continue
            if np.abs(T[i, j]) > rhmc.EPS:
                return False
    return True

def is_upper_diagonal(U, eps = 1e-10):
    """Returns True if U is upper diagonal."""
    N = U.shape[0]
    for i in range(N):
        for j in range(0, i):
            if np.abs(U[i, j]) > eps:
                return False
    return True

def is_lower_diagonal(L, eps = 1e-10):
    """Returns True if L is lower diagonal."""
    N = L.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            if np.abs(L[i, j]) > eps:
                return False
    return True

def test_permutation(L = 4, T = 4, Nc = 2):
    """Tests that the permutation matrix is orthogonal, and that it changes the basis on the Dirac operator correctly."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    N = dNc*rhmc.Ns*Lat.vol
    kappa = 0.1
    V = rhmc.id_field_adjoint(Nc, lat = Lat)
    sparse_dirac = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
    sparse_Q = rhmc.hermitize_dirac(sparse_dirac)

    # Test permutation matrix for original Dirac operator
    P = rhmc.get_permutation_D(dNc, lat = Lat)
    assert np.array_equal((P @ P.transpose()).toarray(), np.eye(N, dtype = np.int8)), 'P matrix is not orthogonal.'
    D = P @ sparse_dirac @ P.transpose()
    for i in range(0, N - 2, 2):
        assert np.abs(D[i, i]) > rhmc.EPS and np.abs(D[i, i + 1]) > rhmc.EPS, \
            'Even columns of basis-changed Dirac operator have zero (i, i + 1) component.'

    # Test change-of-basis matrix for Hermitian Dirac operator
    Ptilde, Ptilde_inv = rhmc.get_permutation_Q(dNc, lat = Lat)
    assert np.array_equal((Ptilde @ Ptilde_inv).toarray(), np.eye(N, dtype = np.complex64)), 'Ptilde matrix is not orthogonal.'
    Qtilde = Ptilde_inv @ sparse_Q @ Ptilde
    for i in range(0, N - 2, 2):
        assert np.abs(Qtilde[i, i]) > rhmc.EPS and np.abs(Qtilde[i, i + 1]) > rhmc.EPS, \
            'Even columns of basis-changed Hermitian Dirac operator have zero (i, i + 1) component.'
    print('test_permutation : Pass')

def test_perm_matrix():
    """Tests that the correct permutation matrix is returned from rhmc.cyclic_perm_matrix."""
    tau = rhmc.cyclic_perm_matrix((0, 1), 2)
    assert np.array_equal(tau.toarray(), np.array([[0, 1], [1, 0]], dtype = np.int8)), 'Matrix for tau is incorrect.'

    sigma = rhmc.cyclic_perm_matrix((1, 3, 5), 6)
    sigma_mat = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0]
    ])
    assert np.array_equal(sigma.toarray(), sigma_mat), 'Matrix for sigma is incorrect.'

    assert np.array_equal(rhmc.cyclic_perm_matrix((3, 7), N = 10).toarray(), rhmc.cyclic_perm_matrix((7, 3), N = 10).toarray()), \
        'cyclic_perm_matrix is not commutative.'

    sig1, sig2 = rhmc.cyclic_perm_matrix((2, 3), 5), rhmc.cyclic_perm_matrix((2, 1), 5)
    sig3 = rhmc.cyclic_perm_matrix((1, 2, 3), 5)
    assert np.array_equal((sig1 @ sig2).toarray(), sig3.toarray()), 'cyclic_perm_matrix is not a group homomorphism.'

    print('test_perm_matrix : Pass')

def test_covariant_perms():
    """Tests that transforming P, A to P' and A' are really ``covariant'', in the sense of Eq. (11) of 
    the Pfaffian paper. Test case uses pi = (1 3 4) with a 6 x 6 matrix."""
    n = 6
    Pi = rhmc.cyclic_perm_matrix((1, 3, 4), n)
    def this_perm(k):
        if k == 1:
            return 3
        if k == 3:
            return 4
        if k == 4:
            return 1
        return k

    p = np.random.rand(n, n)
    A = np.random.rand(n, n)
    p_prime1 = Pi @ p
    A_prime1 = Pi @ A @ Pi.transpose()
    p_prime2 = np.zeros(p.shape, p.dtype)
    A_prime2 = np.zeros(A.shape, A.dtype)
    for i, j in itertools.product(range(n), repeat = 2):
        p_prime2[i, j] = p[this_perm(i), j]
        A_prime2[i, j] = A[this_perm(i), this_perm(j)]
    assert np.array_equal(p_prime1, p_prime2), 'P did not transform covariantly under permutation.'
    assert np.array_equal(A_prime1, A_prime2), 'A did not transform covariantly under permutation.'
    print('test_covariant_perms : Pass')

def test_LU_small_mat():
    """Tests the pivoting mechanism by explicitly pivoting the matrix for LU decomposition prior 
    to calling the lu_decomp function. Test matrix is a 6 x 6 color-spin block from the Dirac 
    matrix on a 4 x 4 lattice."""
    Lat = rhmc.Lattice(4, 4)
    Nc = 2
    dNc = Nc**2 - 1
    V = rhmc.id_field_adjoint(Nc, lat = Lat)
    dirac_op_sparse = rhmc.dirac_op_sparse(0.1, V, lat = Lat)
    Q = rhmc.hermitize_dirac(dirac_op_sparse).toarray()[:6, :6]

    assert np.array_equal(Q.transpose(), -Q), 'Q is not antisymmetric.'
    assert np.abs(np.linalg.det(Q)) > rhmc.EPS, 'Q is singular.'
    P, J, Pi = rhmc.lu_decomp(Q)
    Qprime = Pi @ Q @ Pi.transpose()
    Qdecomp = P @ J @ P.transpose()
    assert np.array_equal(Qprime, Qdecomp), 'LU decomposition for Q failed.'

    print('test_LU_small_mat : Pass')

def test_LU_random_mat(n_max = 21):
    """Tests the LU decomposition by decomposing a random, antisymmetric, non-singular matrix. 
    Constructs random matrices of size 4, 6, 8, ..., 20 for the test."""
    for n in np.arange(6, n_max, 2):
        Q = np.zeros((n, n), dtype = np.complex64)
        for i in range(n):
            for j in range(i):
                Q[i, j] = np.random.rand()
                Q[j, i] = -Q[i, j]
        assert np.array_equal(Q.transpose(), -Q), 'Q is not antisymmetric.'
        assert np.abs(np.linalg.det(Q)) > rhmc.EPS, 'Q is singular.'
        P, J, Pi = rhmc.lu_decomp(Q)
        Qprime = Pi @ Q @ Pi.transpose()
        Qdecomp = P @ J @ P.transpose()
        assert np.allclose(Qprime, Qdecomp), f'LU decomposition for Q failed for matrix of size ({n}, {n}).'
    print('test_LU_random_mat : Pass')

def test_pfsq_random_mat(n_max = 21):
    """Tests the Pfaffian computation by decomposing a random, antisymmetric, non-singular matrix. 
    Constructs random matrices of size 4, 6, 8, ..., 20 for the test."""
    for n in np.arange(6, n_max, 2):
        Q = np.zeros((n, n), dtype = np.complex64)
        for i in range(n):
            for j in range(i):
                Q[i, j] = np.random.rand()
                Q[j, i] = -Q[i, j]
        assert np.array_equal(Q.transpose(), -Q), 'Q is not antisymmetric.'
        assert np.abs(np.linalg.det(Q)) > rhmc.EPS, 'Q is singular.'
        pf = rhmc.pfaffian(Q)
        dev = pf**2 - np.linalg.det(Q)
        assert np.abs(dev) < rhmc.EPS, f'Pfaffian^2 != determinant with deviation {dev} for matrix of size ({n}, {n}).'
    print('test_pfsq_random_mat : Pass')

def test_LU_decomp(L = 4, T = 4, Nc = 2, kappa = 0.25, V = None):
    """Tests the LU decomposition. For a skew-symmetric matrix Q, the LU decomposition 
    is D = PJP^T, where P is lower triangular and J is tridiagonal with trivial Pfaffian. 
    """
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    if V is None:
        V = rhmc.id_field_adjoint(Nc, lat = Lat)
    sparse_dirac_op = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
    Q = rhmc.hermitize_dirac(sparse_dirac_op)
    # print(Q.shape)

    # TODO this isn't working either... HOWEVER IT WORKS WITH NC = 2, JUST NOT NC = 3--- FIGURE OUT WHY
    # for nc = 3, it seemed to have a few non-zero components in the last couple of blocks
    # The structure of the permutation matrix Pi seems to be different in the Nc = 2 and Nc = 3 cases...
    # It only has transpositions for Nc = 3 (each of the odd rows / cols are swapped with one single other one)
    # while for Nc = 2 there are a number of 3-cycles like (1 3 5) and others. Can't figure out why they would be different.
    # TODO ALSO WORKS FOR NC = 4, seems like it breaks when Nc = ODD for some reason??? Maybe there's a dimensional thing going on...
    # NOTE for Nc = 3, L, T = 4 we have n = 256 which is a power of 2... not sure if this is notable or not
    P, J, Pi = rhmc.lu_decomp(Q)
    # print(P)
    # print(Pi[:6, :6].toarray())
    # print(Pi)
    Qprime = Pi @ Q @ Pi.transpose()
    Qdecomp = P @ J @ P.transpose()
    # print(Qprime.toarray() - Qdecomp)
    # print((Qprime.toarray() - Qdecomp)[1])
    # print(np.where(np.abs(Qprime.toarray() - Qdecomp) > 0.05))
    assert np.allclose(Qprime.toarray(), Qdecomp), f'LU decomposition for Dirac operator failed.'
    print('test_LU_decomp : Pass')


################################################################################
############################# TEST RHMC SPECIFICS ##############################
################################################################################
def test_thermalization():
    """
    Tests that a plaquette has thermalized to the same value, after both doing a cold start 
    and a hot start. 
    """

    # TODO method stub

    print('test_thermalization : Pass')


################################################################################
################################## RUN TESTS ###################################
################################################################################
L, T = 2, 2
# L, T = 4, 4             # lattice size to test on
# L, T = 6, 6
# L, T = 8, 8
# L, T = 10, 10

Lat = rhmc.Lattice(L, T)
Nc = 2
# Nc = 3          
# Nc = 4
dNc = Nc**2 - 1

eps = 0.2
gens = rhmc.get_generators(Nc)
# V = np.random.rand(2, L, T, dNc, dNc) + (1j) * np.random.rand(2, L, T, dNc, dNc)
U = rhmc.gen_random_fund_field(Nc, lat = Lat)
V = rhmc.construct_adjoint_links(U, gens, lat = Lat)

print_line()
print('RUNNING TESTS')
print_line()

test_gauge_tools()
test_next_to()
test_gauge_field_properties(L = L, T = T, Nc = Nc, U = U, V = V)

test_flatten_spacetime()
test_flatten_colspin()
test_flatten_full()
# test_zeros_full_dirac(L = L, T = T, Nc = Nc)     # passes but runs slow
test_zeros_sparse_dirac(L = L, T = T, Nc = Nc)
# test_bcs(L = L, T = T, Nc = Nc)     # passes but runs slow

test_sparse_dirac(L = L, T = T, Nc = Nc)                 # free field test
test_sparse_dirac(L = L, T = T, Nc = Nc, V = V)          # random field test

test_herm_dirac_full(L = L, T = T, Nc = Nc, V = V)
test_herm_dirac_sparse(L = L, T = T, Nc = Nc, V = V)
test_skew_sym_sparse(L = L, T = T, Nc = Nc, V = V)
test_squared_dirac(L = L, T = T, Nc = Nc, V = V)
test_eval_flat_ferm(L = L, T = T, Nc = Nc)

test_shift_cg(L = L, T = T, Nc = Nc)
test_rK_application(L = L, T = T, Nc = Nc)

# TODO test these next!
test_init_fields(L = L, T = T, Nc = Nc, V = V)
test_pseudoferm_action(L = L, T = T, Nc = Nc)
test_pseudoferm_force(L = L, T = T, Nc = Nc)
test_gauge_force_wilson(L = L, T = T, Nc = Nc, U = U)

test_rational_approx_small()
test_rational_approx_large()

test_perm_matrix()
test_covariant_perms()
test_permutation(L = L, T = T, Nc = Nc)
test_LU_small_mat()
test_LU_random_mat()
test_pfsq_random_mat()
test_LU_decomp(L = L, T = T, Nc = Nc)

print_line()
print('ALL TESTS PASSED')
print_line()


################################################################################
################################# SCRATCH WORK #################################
################################################################################
# Lat = rhmc.Lattice(4, 4)
# Nc = 2
# dNc = Nc**2 - 1
# V = rhmc.id_field_adjoint(Nc, lat = Lat)
# kappa = 0.1
# dirac_op_full = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
# dirac_op_sparse = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
# dirac_op_sparse_apbc = rhmc.dirac_op_sparse(kappa, V, bcs = (-1, -1), lat = Lat)

# Q = rhmc.hermitize_dirac(dirac_op_sparse)
# P = rhmc.get_permutation_D(dNc, lat = Lat)
# Q_sp = P @ Q @ P.transpose()
# Q_blk = Q.toarray()[:6, :6]
# Qsp_blk = Q_sp.toarray()[:4, :4]

