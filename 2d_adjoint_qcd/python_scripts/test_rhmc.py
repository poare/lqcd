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
    dimSUN = Nc**2 - 1                          # Dimension of SU(N)
    tSUN = rhmc.get_generators(Nc)                   # Generators of SU(N)
    delta_ab = np.eye(dimSUN)
    for a, b in itertools.product(range(dimSUN), repeat = 2):
        tr_ab = rhmc.trace(tSUN[a] @ tSUN[b])
        assert is_equal(tr_ab, (1/2) * delta_ab[a, b]), f'Tr[t^a t^b] != (1/2)\delta_ab at (a, b) = ({a}, {b}).'
    print('test_gauge tools : Pass')

################################################################################
###################### TEST SPARSE MATRIX IMPLEMENTATIONS ######################
################################################################################

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
    dimSUN = Nc**2 - 1
    assert rhmc.flatten_colspin_idx((6, 1), dimSUN) == 14, 'flatten_colspin_idx not correct.'
    assert rhmc.unflatten_colspin_idx(3, dimSUN) == (3, 0), 'unflatten_colspin_idx not correct.'
    assert rhmc.unflatten_colspin_idx(10, dimSUN) == (2, 1), 'unflatten_colspin_idx not correct.'
    for flat_idx in range(dimSUN * rhmc.Ns):
        assert rhmc.flatten_colspin_idx(rhmc.unflatten_colspin_idx(flat_idx, dimSUN), dimSUN) == flat_idx, \
            f'Flatten and unflatten not inverses at flat_idx = {flat_idx}.'
    print('test_flatten_colspin : Pass')

def test_sparse_dirac():
    """Compares the sparse Dirac operator against a full Dirac operator on a small lattice."""
    Lat = rhmc.Lattice(4, 4)
    Nc = 2
    # TODO method stub

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

# Run tests
print_line()
test_gauge_tools()
test_flatten_spacetime()
test_flatten_colspin()
test_rational_approx_small()
test_rational_approx_large()
print_line()

# row = np.array([0, 0, 1, 2, 2, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3 ,4, 5, 6])
# sparse = bsr_matrix((data, (row, col)), shape=(3, 3))
# print(sparse.toarray())
# print(np.array([[1, 0, 2],
#        [0, 0, 3],
#        [4, 5, 6]]))

indptr = np.array([0, 2, 3, 6])
# indices = np.array([0, 2, 2, 0, 1, 2])
indices = np.array([2, 0, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
mat = bsr_matrix((data,indices,indptr), shape=(6, 6))
print(mat.toarray())