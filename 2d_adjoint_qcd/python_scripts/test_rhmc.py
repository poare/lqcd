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
from scipy.sparse import bsr_matrix

import rhmc

def print_line():
    print('-'*50)

def is_equal(a, b, eps = 1e-10):
    """Returns whether a and b are equal up to precision eps."""
    return np.abs(a - b) < eps

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

def test_zeros_full_dirac():
    """Tests that the full Dirac operator has zeros in the correct places."""
    Lat = rhmc.Lattice(4, 4)
    Nc = 2
    kappa = 0.1
    V = rhmc.id_field_adjoint(Nc)
    dirac_op = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
    for lx, tx, ly, ty in itertools.product(range(4), repeat = 4):
        x, y = np.array([lx, tx]), np.array([ly, ty])
        if Lat.next_to_equal(x, y):
            continue
        submat = dirac_op[:, :, lx, tx, :, :, ly, ty]
        assert not np.any(submat), f'Dirac operator is nonzero outside of nearest neighbors.'
    print('test_zeros_full_dirac : Pass')

def test_zeros_sparse_dirac():
    """Tests that the sparse Dirac operator has zeros in the correct places."""
    Lat = rhmc.Lattice(4, 4)
    Nc = 2
    dNc = Nc**2 - 1
    kappa = 0.1
    V = rhmc.id_field_adjoint(Nc, lat = Lat)
    sparse_dirac_op = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
    dirac_op = rhmc.unflatten_operator(sparse_dirac_op.toarray(), dNc, lat = Lat)
    for lx, tx, ly, ty in itertools.product(range(4), repeat = 4):
        x, y = np.array([lx, tx]), np.array([ly, ty])
        if Lat.next_to_equal(x, y):
            continue
        submat = dirac_op[:, :, lx, tx, :, :, ly, ty]
        assert not np.any(submat), f'Dirac operator is nonzero outside of nearest neighbors at x = {x}, y = {y}.'
    print('test_zeros_sparse_dirac : Pass')

def test_sparse_dirac():
    """Compares the sparse Dirac operator against a full Dirac operator on a small lattice."""
    Lat = rhmc.Lattice(4, 4)
    Nc = 2
    kappa = 0.1
    V = rhmc.id_field_adjoint(Nc)
    sparse_dirac = rhmc.dirac_op_sparse(kappa, V, lat = Lat)
    full_dirac = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
    full_dirac_flat = rhmc.flatten_operator(full_dirac, lat = Lat)
    assert np.array_equal(sparse_dirac.toarray(), full_dirac_flat), \
        'Sparse matrix construction disagrees with full Dirac operator.'
    
    print('test_sparse_dirac : Pass')

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

print_line()

test_gauge_tools()
test_next_to()

test_flatten_spacetime()
test_flatten_colspin()
test_flatten_full()
test_zeros_full_dirac()
test_zeros_sparse_dirac()
test_sparse_dirac()

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

# Lat = rhmc.Lattice(4, 4)
# Nc = 2
# V = rhmc.id_field_adjoint(Nc)
# kappa = 0.1
# dirac_op = rhmc.get_dirac_op_full(kappa, V, lat = Lat)
