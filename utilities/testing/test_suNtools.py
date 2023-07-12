################################################################################
# Tests an implementation of RHMC.                                             #
# Tests to perform:                                                            #
#   0. Test accept-reject of each possibility.                                 #
#   1. Gauge coupling g --> 0 limit (free theory)                              #
#   2. g --> and m --> infinity (pure gauge theory)                            #
################################################################################
# Author: Patrick Oare                                                         #
################################################################################

# n_boot = 200
import numpy as np
import itertools
import scipy
from scipy.sparse import bsr_matrix

np.random.seed(20)

import sys
sys.path.append('/Users/theoares/lqcd/utilities')
import suNtools

EPS = 1e-8

def print_line():
    print('-'*50)

def is_equal(a, b, eps = EPS):
    """Returns whether a and b are equal up to precision eps."""
    return np.abs(a - b) < eps

def test_proj_suN(n_samps = 500):
    """Tests that matrices projected with suNtools.proj_fund_suN are in fact, in SU(N)."""
    for N in range(2, 10):
        for i in range(n_samps):
            M = np.random.rand(N, N) + 1j * np.random.rand(N, N)
            U = suNtools.proj_fund_suN(M, prec = EPS)
            assert np.allclose(U @ suNtools.dagger(U), np.eye(N), atol = EPS), 'U is not unitary.'
            assert np.allclose(suNtools.dagger(U) @ U, np.eye(N), atol = EPS), 'U is not unitary.'
            assert is_equal(np.linalg.det(U), 1), 'U does not have unit determinant.'
    print('test_proj_suN : Pass')

def test_gen_suN(n_samps = 100):
    """Tests that matrices generated with suNtools.rand_suN_matrix are in fact, in SU(N)."""
    for N in range(2, 10):
        for eps in np.arange(0.1, 0.9, 0.1):
            for i in range(n_samps):
                U = suNtools.rand_suN_matrix(N)
                assert np.allclose(U @ suNtools.dagger(U), np.eye(N), atol = EPS), 'U is not unitary.'
                assert np.allclose(suNtools.dagger(U) @ U, np.eye(N), atol = EPS), 'U is not unitary.'
                assert is_equal(np.linalg.det(U), 1), 'U does not have unit determinant.'
    print('test_gen_suN : Pass')

################################################################################
################################## RUN TESTS ###################################
################################################################################

print_line()
print('RUNNING TESTS')
print_line()

test_proj_suN()
test_gen_suN()

print_line()
print('ALL TESTS PASSED')
print_line()

