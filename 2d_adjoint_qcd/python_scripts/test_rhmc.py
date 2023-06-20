################################################################################
# Tests an implementation of RHMC.                                             #
# Tests to perform:                                                            #
#   0. Test accept-reject of each possibility.                                 #
#   1. Gauge coupling g --> 0 limit (free theory)                              #
#   2. g --> and m --> infinity (pure gauge theory)                            #
################################################################################
# Author: Patrick Oare                                                         #
################################################################################

n_boot = 200
import numpy as np
import itertools
import scipy
from scipy.sparse import bsr_matrix

np.random.seed(20)

import rhmc

def print_line():
    print('-'*50)

def is_equal(a, b, eps = rhmc.EPS):
    """Returns whether a and b are equal up to precision eps."""
    return np.abs(a - b) < eps

def check_sparse_equal(A, B):
    """
    Checks that two scipy.sparse.bsr_matrix A and B are equal. Note that you cannot 
    simply use np.array_equal(A, B); you can compare the dense matrices with 
    np.array_equal(A.toarray(), B.toarray()), which should have the same output as 
    check_sparse_equal(A, B).
    """
    return (A != B).nnz == 0

def test_gauge_tools():
    """Tests utility functions for gauge fields."""
    Nc = 3
    dNc = Nc**2 - 1                                  # Dimension of SU(N)
    tSUN = rhmc.get_generators(Nc)                   # Generators of SU(N)
    delta_ab = np.eye(dNc)
    for a, b in itertools.product(range(dNc), repeat = 2):
        tr_ab = rhmc.trace(tSUN[a] @ tSUN[b])
        assert is_equal(tr_ab, (1/2) * delta_ab[a, b]), f'Tr[t^a t^b] != (1/2)\delta_ab at (a, b) = ({a}, {b}).'
    print('test_gauge tools : Pass')

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

def test_zeros_full_dirac(L = 4, T = 4, Nc = 2, kappa = 0.1):
    """Tests that the full Dirac operator has zeros in the correct places."""
    Lat = rhmc.Lattice(L, T)
    V = rhmc.id_field_adjoint(Nc, lat = Lat)
    dirac_op = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
    for lx, tx, ly, ty in itertools.product(range(L), range(T), repeat = 2):
        x, y = np.array([lx, tx]), np.array([ly, ty])
        if Lat.next_to_equal(x, y):
            continue
        submat = dirac_op[:, :, lx, tx, :, :, ly, ty]
        assert not np.any(submat), f'Dirac operator is nonzero outside of nearest neighbors.'
    print('test_zeros_full_dirac : Pass')

def test_zeros_sparse_dirac(L = 4, T = 4, Nc = 2, kappa = 0.1):
    """Tests that the sparse Dirac operator has zeros in the correct places."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
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

def test_sparse_dirac(L = 4, T = 4, Nc = 2, kappa = 0.1):
    """Compares the sparse Dirac operator against a full Dirac operator on a small lattice."""
    Lat = rhmc.Lattice(L, T)
    V = rhmc.id_field_adjoint(Nc, lat = Lat)
    sparse_dirac = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
    full_dirac = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
    full_dirac_flat = rhmc.flatten_operator(full_dirac, lat = Lat)
    assert np.array_equal(sparse_dirac.toarray(), full_dirac_flat), \
        'Sparse matrix construction disagrees with full Dirac operator.'
    print('test_sparse_dirac : Pass')

def test_herm_dirac_full(L = 4, T = 4, Nc = 2, kappa = 0.1):
    """Confirms that the full Dirac operator D is gamma5-Hermitian."""
    Lat = rhmc.Lattice(L, T)
    V = rhmc.id_field_adjoint(Nc, lat = Lat)
    dirac = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
    dirac_conj = np.einsum('ij,ajxtbkys,kl->aixtblys', rhmc.gamma5, dirac, rhmc.gamma5)
    dirac_dagger = rhmc.dagger_op(dirac)
    assert np.array_equal(dirac_conj, dirac_dagger), 'Full Dirac operator is not gamma5-hermitian.'

    Q = rhmc.hermitize_dirac(dirac)
    Q_dagger = rhmc.dagger_op(Q)
    assert np.array_equal(Q, Q_dagger), f'Q is not Hermitian.'

    print('test_herm_dirac_full : Pass')

def test_herm_dirac_sparse(L = 4, T = 4, Nc = 2, kappa = 0.1):
    """Confirms that the sparse Hermitian Dirac operator Q = gamma5 D is, in fact, Hermitian."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
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
    assert np.array_equal(sparse_dirac_conj.toarray(), rhmc.flatten_operator(dirac_conj, lat = Lat)), 'Sparse gamma5-conjugate != full gamma5-conjugate.'
    assert np.array_equal(sparse_dirac_dagger.toarray(), rhmc.flatten_operator(dirac_dagger, lat = Lat)), 'Sparse dagger != full dagger.'
    assert np.array_equal(sparse_dirac_conj.toarray(), sparse_dirac_dagger.toarray()), 'Sparse gamma5-conjugate != sparse Hermitian conjugate'

    Q = rhmc.hermitize_dirac(sparse_dirac)
    assert np.array_equal(Q.toarray(), Q.conj().transpose().toarray()), 'Q is not Hermitian.'
    assert np.array_equal(Q.toarray(), rhmc.dagger_op(Q).toarray()), 'Q is not Hermitian.'
    assert check_sparse_equal(Q, Q.conj().transpose()), 'Q is not Hermitian.'
    print('test_herm_dirac_sparse : Pass')

def test_skew_sym_sparse(L = 4, T = 4, Nc = 2, kappa = 0.1):
    """Tests the skew-symmetric Dirac operator is skew-symmetric."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    V = rhmc.id_field_adjoint(Nc, lat = Lat)
    sparse_dirac = rhmc.dirac_op_sparse(kappa, V, lat = Lat)

    Q = rhmc.hermitize_dirac(sparse_dirac)
    assert check_sparse_equal(Q.transpose(), -Q), 'Q is not skew-symmetric.'
    print('test_skew_sym_sparse : Pass')

def test_squared_dirac(L = 4, T = 4, Nc = 2, kappa = 0.1):
    """Tests that the sparse construction and full construction of the squared Dirac 
    operator are equal."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    V = rhmc.id_field_adjoint(Nc, lat = Lat)
    sparse_dirac = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
    dense_dirac = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
    sparse_K = rhmc.construct_K(sparse_dirac)
    dense_K = rhmc.construct_K(dense_dirac)
    dense_K_flat = rhmc.flatten_operator(dense_K, lat = Lat)
    assert np.array_equal(sparse_K.toarray(), dense_K_flat), 'Sparse K != dense K.'
    assert check_sparse_equal(sparse_K, rhmc.dagger_op(sparse_K)), 'K is not Hermitian.'
    print('test_squared_dirac : Pass')

def test_eval_flat_ferm(L = 4, T = 4, Nc = 2):
    """Tests that the eval_flat_ferm_field and flat_field_putat functions work as intended."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    psi_dense = np.random.rand(dNc, rhmc.Ns, L, T) + 1j*np.random.rand(dNc, rhmc.Ns, L, T)
    psi_flat = rhmc.flatten_ferm_field(psi_dense, lat = Lat)
    assert np.array_equal(rhmc.unflatten_ferm_field(psi_flat, dNc, lat = Lat), psi_dense), 'Sparse psi != dense psi.'

    colspin_dense = np.random.rand(dNc, rhmc.Ns) + 1j*np.random.rand(dNc, rhmc.Ns)
    colspin_flat = rhmc.flatten_colspin_vec(colspin_dense)
    assert np.array_equal(rhmc.unflatten_colspin_vec(colspin_flat, dNc), colspin_dense), 'Sparse colspin != dense colspin.'

    x, t = 1, 2
    psi_blk_dense = psi_dense[:, :, x, t]
    psi_blk_flat = rhmc.flat_field_evalat(psi_flat, x, t, dNc, lat = Lat)
    assert np.array_equal(rhmc.unflatten_colspin_vec(psi_blk_flat, dNc), psi_blk_dense), 'Sparse psi @ (x, t) != dense psi @ (x, t).'

    rand_colspin = np.random.rand(dNc, rhmc.Ns) + 1j*np.random.rand(dNc, rhmc.Ns)
    psi2_dense = np.zeros((dNc, rhmc.Ns, L, T), dtype = np.complex128)
    psi2_flat0 = np.zeros((dNc*rhmc.Ns*L*T), dtype = np.complex128)
    psi2_dense[:, :, x, t] = rand_colspin
    psi2_flat = rhmc.flat_field_putat(psi2_flat0, rand_colspin, x, t, dNc, mutate = False, lat = Lat)
    assert np.array_equal(rhmc.unflatten_ferm_field(psi2_flat, dNc, lat = Lat), psi2_dense), 'Evalat1: sparse psi @ (x, t) != dense psi @ (x, t).'
    psi3_flat = rhmc.flat_field_putat(psi2_flat0, rhmc.flatten_colspin_vec(rand_colspin), x, t, dNc, mutate = False, lat = Lat)
    assert np.array_equal(rhmc.unflatten_ferm_field(psi3_flat, dNc, lat = Lat), psi2_dense), 'Evalat2: sparse psi @ (x, t) != dense psi @ (x, t).'

    # Test mutation
    assert np.array_equal(psi2_flat0, np.zeros((dNc*rhmc.Ns*L*T), dtype = np.complex128)), 'Original field has been mutated.'
    rhmc.flat_field_putat(psi2_flat0, rand_colspin, x, t, dNc, mutate = True, lat = Lat)
    assert np.array_equal(rhmc.unflatten_ferm_field(psi2_flat0, dNc, lat = Lat), psi2_dense), 'Original field not mutated.'

    print('test_eval_flat_ferm : Pass')

################################################################################
############################### TEST CG SOLVER #################################
################################################################################

def test_shift_cg(L = 4, T = 4, Nc = 2, kappa = 0.1):
    """Tests the shifted CG solver by explicitly computing (K + \beta_i)^{-1} \Phi on 
    a small lattice."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
    V = rhmc.id_field_adjoint(Nc, lat = Lat)
    dirac = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
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

def test_rK_application(L = 4, T = 4, Nc = 2, kappa = 0.1):
    """Tests an application of r(K) to a random vector phi on a small lattice."""
    Lat = rhmc.Lattice(L, T)
    dNc = Nc**2 - 1
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

def test_pseudoferm_force():
    """Tests that the pseudofermion derivative of Phi^\dagger K^{-1/4} \Phi is accurate 
    by using automatic differentation on U_mu(x)."""

    print('test_pseudoferm_force : Pass')

################################################################################
############################# TEST RHMC SPECIFICS ##############################
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
################################## RUN TESTS ###################################
################################################################################

L, T = 4, 4             # lattice size to test on
# L, T = 8, 8
# L, T = 10, 10
print_line()

test_gauge_tools()
test_next_to()

test_flatten_spacetime()
test_flatten_colspin()
test_flatten_full()
test_zeros_full_dirac(L, T)
test_zeros_sparse_dirac(L, T)
test_sparse_dirac(L, T)
test_herm_dirac_full(L, T)
test_herm_dirac_sparse(L, T)
test_skew_sym_sparse(L, T)
test_squared_dirac(L, T)
test_eval_flat_ferm(L, T)

test_shift_cg(L, T)
test_rK_application(L, T)

test_rational_approx_small()
test_rational_approx_large()

print_line()

################################################################################
################################# SCRATCH WORK #################################
################################################################################

# row = np.array([0, 0, 1, 2, 2, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3 ,4, 5, 6])
# sparse = bsr_matrix((data, (row, col)), shape=(3, 3))
# print(sparse.toarray())
# print(np.array([[1, 0, 2],
#        [0, 0, 3],
#        [4, 5, 6]]))

# indptr = np.array([0, 2, 3, 6])
# # indices = np.array([0, 2, 2, 0, 1, 2])
# indices = np.array([2, 0, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
# mat = bsr_matrix((data,indices,indptr), shape=(6, 6))
# print(mat.toarray())

# dirac = test_zeros_full_dirac()

Lat = rhmc.Lattice(4, 4)
Nc = 2
V = rhmc.id_field_adjoint(Nc, lat = Lat)
kappa = 0.1
dirac_op_full = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
dirac_op_sparse = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
