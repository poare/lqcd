
################################################################################
# This script reads in Npts values of the Matsubara Green's function at        #
# frequencies i\omega and performs an analytic continuation of these points to #
# the upper half plane by exploiting properties of Nevanlinna / contractive    #
# functions. This is an implementation following the paper:                    #
#       Fei, J., Yeh, C.N. & Gull, E. Nevanlinna Analytical Continuation.      #
#                       Phys Rev Lett 126, 056402 (2021).                      #
# A note about the implementation: the notation used in this script is the     #
# same as that used in the paper, for clarity. Namely, the Matsubara freqs are #
# stored in a numpy array Y, and the Nevanlinna function values NG are stored  #
# in a numpy array C.
################################################################################

import numpy as np
import gmpy2 as gmp
import h5py
import os
import time
import re
import itertools
import io
import random

# Set precision for gmpy2 and initialize complex numbers
prec = 128
gmp.get_context().allow_complex = True
gmp.get_context().precision = prec

# Mobius transform
I = gmp.mpc(0, 1)
h = lambda z : (z - I) / (z + I)
hinv = lambda q : I * (gmp.mpc(1, 0) + q) / (gmp.mpc(1, 0) - q)
# for some reason mpc segfaults when it uses the built in conjugate values, so just use this
def conj(z):
    return gmp.mpc(z.real, -z.imag)

# Data input with frequencies and values of G
data_path = '/Users/theoares/lqcd/spectral/hardy_optim_clean_submission/test/freqs_1.txt'
Npts = 0
Y = []    # Matsubara frequencies i\omega
C = []    # Negative of G, NG
with open(data_path) as f:
    for line in f:
        line = line.strip()
        tokens = line.split()
        # TODO depending on the input precision we may need to be more careful about the next line
        freq, reG, imG = float(tokens[0]), float(tokens[1]), float(tokens[2])
        Y.append(gmp.mpc(0, freq))
        C.append(gmp.mpc(-reG, -imG))
        Npts += 1
    f.close()
Y.reverse()         # code states that the algorithm is more robust with reverse order
C.reverse()
Y = np.array(Y)
C = np.array(C)
lam = [h(z) for z in C]
print('Read in ' + str(Npts) + ' Matsubara modes at frequencies ~ ' + str(["{0:1.8f}".format(x) for x in Y]) + '.')

# Confirm that we're using a Nevanlinna function
for z in C:
    assert z.imag > 0.0, 'Negative of input function is not Nevanlinna.'

# Construct Pick matrix to check that the Nevanlinna interpolant exists
Pick = np.empty((Npts, Npts), dtype = object)
for i, j in itertools.product(range(Npts), repeat = 2):
    num = 1 - lam[i] * conj(lam[j])
    denom = 1 - h(Y[i]) * conj(h(Y[j]))
    Pick[i, j] = num / denom
print('Pick matrix: ')
print(Pick)

# TODO implement check for positive semidefinite-ness

# Construct phi[k] = theta_k(Y_k)
abcd_bar_lst = []
for k in range(Npts - 1):
    id = np.array([
        [gmp.mpc(1, 0), gmp.mpc(0, 0)],
        [gmp.mpc(0, 0), gmp.mpc(1, 0)]
    ])
    abcd_bar_lst.append(id)
phi = np.full((Npts), gmp.mpc(0, 0), dtype = object)
phi[0] = lam[0]
for k in range(Npts - 1):
    for j in range(k, Npts - 1):
        xik = (Y[j + 1] - Y[k]) / (Y[j + 1] - conj(Y[k]))
        factor = np.array([
            [xik, phi[k]],
            [conj(phi[k]) * xik, gmp.mpc(1, 0)]
        ])
        abcd_bar_lst[j] = abcd_bar_lst[j] @ factor
    num = lam[k + 1] * abcd_bar_lst[k][1, 1] - abcd_bar_lst[k][0, 1]
    denom = abcd_bar_lst[k][0, 0] - lam[k + 1] * abcd_bar_lst[k][1, 0]
    phi[k + 1] = num / denom
print('Phi[k] is: ' + str(phi))

# Choose interpolant function theta_{M + 1}, and the grid to evaluate on. These
# parameters can all be varied.
Nreal = 6000
omega_min = -10
omega_max = 10
eta = 1e-3
theta_mp1 = lambda z : 0            # change this based on priors

# Perform interpolation by constructing a, b, c, d at each z-point.
zmesh = np.linspace(omega_min, omega_max, num = Nreal)
zspace = np.array([gmp.mpc(z, eta) for z in zmesh])

# confirm that NGreal extends NG-- the last Npts values of NGreal should be C
# zspace = np.append(zspace, Y)
# Nreal += len(Y)

NGreal = np.empty((Nreal), dtype = object)
for idx, z in enumerate(zspace):
    abcd = np.array([
        [gmp.mpc(1, 0), gmp.mpc(0, 0)],
        [gmp.mpc(0, 0), gmp.mpc(1, 0)]
    ])
    for k in range(Npts):
        xikz = (z - Y[k]) / (z - conj(Y[k]))
        factor = np.array([
            [xikz, phi[k]],
            [conj(phi[k]) * xikz, gmp.mpc(1, 0)]
        ])
        abcd = abcd @ factor
    num = abcd[0, 0] * theta_mp1(z) + abcd[0, 1]
    denom = abcd[1, 0] * theta_mp1(z) + abcd[1, 1]
    theta = num / denom         # contractive function theta(z)
    NGreal[idx] = hinv(theta)

out_path = '/Users/theoares/Dropbox (MIT)/research/spectral/testing/output_1.txt'
f = open(out_path, 'w')
header = f'{Nreal:.0f} {omega_min:.0f} {omega_max:.0f} {eta:.5f}\n'
f.write(header)
for i, NGval in enumerate(NGreal):
    s =  NGval.__str__() + '\n'
    f.write(s)
f.close()
