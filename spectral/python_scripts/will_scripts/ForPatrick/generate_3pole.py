"""
Generates a spectral function to simulate and saves it to an hdf5 file.

Arguments
---------
string : path to save the spectral function at.
"""

import sys
import numpy as np
import gmpy2 as gmp
import fileio
import scipy.interpolate as interpolate
import pylab as plt
# from nevanlinna import *

# sys.path.append('/Users/theoares/lqcd/spectral/python_scripts')
# import poare_utils

# Set precision for gmpy2 and initialize complex numbers
prec = 128
gmp.get_context().allow_complex = True
gmp.get_context().precision = prec
I = gmp.mpc(0, 1)

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <output_file_name>")
        exit(1)
    fname = str(sys.argv[1])
    print('Writing data to: ' + fname)

    # Construct desired spectral function
    beta = 64
    freqs = matsubara(beta, boson=False)

    masses = [
        gmp.mpc(2, 0)/gmp.mpc(10, 0),
        gmp.mpc(1, 0)/gmp.mpc(2, 0),
        gmp.mpc(4, 0)/gmp.mpc(5, 0)]

    green_fcn = pole(masses[0], freqs) + pole(masses[1], freqs) + pole(masses[2], freqs)
    freqs = freqs[1:]
    green_fcn = green_fcn[1:]
    fileio.write_gmp_input_h5(fname, beta, freqs, green_fcn, start=0, stop=1, num=1000)


def matsubara(beta, boson = False):
    rng = np.arange(beta)
    if boson:
        return np.array([2*gmp.const_pi()*I*n/beta for n in rng])
    return np.array([(2*n + 1)*gmp.const_pi()*I/beta for n in rng])

def pole(m, z):
    return -gmp.mpc(1, 0) / (z - m) - gmp.mpc(1, 0) / (m + z)

if __name__ == '__main__':
    main()