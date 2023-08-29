"""
Generates a spectral function to simulate and saves it to an hdf5 file.

Arguments
---------
string : path to save the spectral function at.
"""

import sys
import numpy as np
import gmpy2 as gmp

sys.path.append('/Users/theoares/inverse_problems/inverse_problems')
import fileio
import scipy.interpolate as interpolate
import pylab as plt
# from nevanlinna import *

# sys.path.append('/Users/theoares/lqcd/spectral/python_scripts')
# import poare_utils

# Set precision for gmpy2 and initialize complex numbers
# prec = 128
prec = 256
# prec = 1028
gmp.get_context().allow_complex = True
gmp.get_context().precision = prec
ONE = gmp.mpc(1, 0)
I = gmp.mpc(0, 1)

def main():

    fname = str(sys.argv[1])
    print('Writing data to: ' + fname)

    # Construct desired spectral function
    # beta = 48
    beta = 64
    # freqs = matsubara(beta, boson=True)
    freqs = matsubara(beta, boson = False)

    ngs = np.zeros(len(freqs), dtype=object)
    # for mm in [gmp.mpfr('0.5')]:
    for mm in [gmp.mpfr('0.2'), gmp.mpfr('0.5'), gmp.mpfr('0.8')]:
        ngs = ngs + analytic_ft(freqs, mm, beta)
    freqs, ngs = freqs[1:], ngs[1:]

    # sub_idxs = list(range(20))
    # sub_idxs = list(range(40))
    # sub_idxs = list(range(len(ngs)))
    sub_idxs = list(range(12))
    # sub_idxs.append(20)

    start = 0.0
    stop = 1.0
    num = 1000

    fileio.write_gmp_input_h5(fname, beta, freqs[sub_idxs], ngs[sub_idxs], start, stop, num)
    print('Green\'s function data written to: ' + fname)

def build_bspline(x, y):
    t, c, k = interpolate.splrep(x, y, s=0, k=4)
    bspline = interpolate.BSpline(t, c, k, extrapolate=True)
    return bspline

def matsubara(beta, boson = False):
    rng = np.arange(beta)
    if boson:
        return np.array([2*gmp.const_pi()*I*n/beta for n in rng])
    # return np.array([(2*n + 1)*gmp.const_pi()*I/beta for n in rng])
    return np.array([((2*n+1)*ONE)  * gmp.const_pi() * I / (beta * ONE) for n in rng])

def analytic_ft(z, m, beta):
    # prefactor = -1*(1 + gmp.exp(-m*beta))
    # prefactor = gmp.mpfr('-1')
    # pole1 = 1/(z - m)
    # pole2 = 1/(z + m)
    prefactor = -ONE
    pole1 = ONE / (z - m)
    pole2 = ONE / (z + m)
    return prefactor * (pole1 + pole2)

if __name__ == '__main__':
    main()