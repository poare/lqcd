import numpy as np
import gmpy2 as gmp
import h5py
import os
import time
import re
import itertools
import io
import random
import matplotlib.pyplot as plt

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
    """Returns the conjugate of a gmp.mpc value z. For some reason, the built in mpc function .conj
    breaks when it is called."""
    return gmp.mpc(z.real, -z.imag)

def to_mpfr(z):
    """Reformats a np.real number into a gmpy2.mpfr number"""
    return gmp.mpfr(z)

def to_mpc(z):
    """Reformats a np.complex number into a gmpy2.mpc number"""
    return gmp.mpc(z)

def is_zero(z, epsilon = 1e-10):
    """
    Returns whether a gmp.mpc number z is consistent with 0 to the precision epsilon in both the
    real and imaginary parts.
    """
    return np.abs(np.float64(z.real)) < epsilon and np.abs(np.float64(z.imag)) < epsilon

def hardy(k):
    """
    Returns the kth basis element f^k for the standard Hardy space basis.

    Parameters
    ----------
    k : int
        Element of the Hardy basis to return.

    Returns
    -------
    function fk : gmp.mpc -> gmp.mpc
        kth basis function for the Hardy space.
    """
    def fk(z):
        """kth basis element for the standard Hardy space basis. Acts on gmpy2.mpc numbers."""
        return 1 / (gmp.sqrt(gmp.const_pi()) * (z + I)) * ((z - I) / (z + I)) ** k
    return fk

def read_txt_input(data_path):
    """
    Reads correlator input from a text file and outputs the frequencies the correlator is
    evaluated at, the correlator values at these frequencies, and the Mobius transform of the
    correlator values.

    Parameters
    ----------
    datapath : string
        Path to input. Should be a text file.

    Returns
    -------
    np.array[gmp.mpc]
        Matsubara frequencies Y.
    np.array[gmp.mpc]
        Correlator values C.
    np.array[gmp.mpc]
        Mobius transformed correlator values lambda.
    int
        Number of points the correlator is evaluated at, i.e. card(C)
    """
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
    return Y, C, lam, Npts

def construct_Pick(Y, lam):
    """
    Constructs the Pick matrix (1 - lambda_i lambda_j^*) / (1 - h(Y_i) h(Y_j)^*) from correlator input
    to check that the Nevanlinna interpolant exists.

    Parameters
    ----------
    Y : np.array[gmp.mpc]
        Matsubara frequencies the correlator is measured at.
    lam : np.array[gmp.mpc]
        Mobius transform of the input correlator.

    Returns
    -------
    np.array[gmp.mpc]
        len(lam) by len(lam) array for the Pick matrix.
    """
    Npts = len(Y)
    Pick = np.empty((Npts, Npts), dtype = object)
    for i, j in itertools.product(range(Npts), repeat = 2):
        num = 1 - lam[i] * conj(lam[j])
        denom = 1 - h(Y[i]) * conj(h(Y[j]))
        print(num)
        print(denom)
        Pick[i, j] = num / denom
    return Pick

def is_soluble(Y, lam, prec = 1e-10):
    """
    Determines if there is a Nevanlinna interpolant to this set of Y and lambda values.

    Parameters
    ----------
    Y : np.array[gmp.mpc]
        Matsubara frequencies the correlator is measured at.
    lam : np.array[gmp.mpc]
        Mobius transform of the input correlator.

    Returns
    -------
    bool
        True if there is a Nevanlinna interpolant, False otherwise.
    """
    Pick = construct_Pick(Y, lam)
    Pick_np = np.complex64(Pick)
    eigs, _ = np.linalg.eigh(Pick_np)
    soluble = True
    for eig in eigs:
        if eig < 0 and not np.abs(eig) < prec:
            soluble = False
    return soluble

def construct_phis(Y, lam):
    """
    This is the indexing I've been using.
    Constructs the phi[k] values needed to perform the Nevanlinna analytic continuation.
    At each iterative step k, phi[k] is defined as theta_k(Y_k), where theta_k is the kth
    iterative approximation to the continuation.

    Parameters
    ----------
    Y : np.array[gmp.mpc]
        Mobius transform of Matsubara frequencies the correlator is measured at.
    lam : np.array[gmp.mpc]
        Mobius transform of the input correlator.

    Returns
    -------
    np.array[gmp.mpc]
        Values of phi[k] = theta_k(Y_k).
    """
    Npts = len(Y)
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
        print('k: ' + str(k))
        print('abcd|_{Y_k} is: ' + str(abcd_bar_lst[k]))
        print('num is: ' + str(num))
        print('denom is: ' + str(denom))
        if is_zero(num):
            phi[k + 1] = gmp.mpc(0, 0)
        else:
            phi[k + 1] = num / denom
        print('phi_{k + 1}: ' + str(phi[k + 1]))
    return phi

def construct_phis_theirs(Y, lam):
    """
    This is the indexing used in the code from the Nevanlinna paper.
    Constructs the phi[k] values needed to perform the Nevanlinna analytic continuation.
    At each iterative step k, phi[k] is defined as theta_k(Y_k), where theta_k is the kth
    iterative approximation to the continuation.

    Parameters
    ----------
    Y : np.array[gmp.mpc]
        Mobius transform of Matsubara frequencies the correlator is measured at.
    lam : np.array[gmp.mpc]
        Mobius transform of the input correlator.

    Returns
    -------
    np.array[gmp.mpc]
        Values of phi[k] = theta_k(Y_k).
    """
    Npts = len(Y)
    abcd_bar_lst = []
    for k in range(Npts):
        id = np.array([
            [gmp.mpc(1, 0), gmp.mpc(0, 0)],
            [gmp.mpc(0, 0), gmp.mpc(1, 0)]
        ])
        abcd_bar_lst.append(id)
    phi = np.full((Npts), gmp.mpc(0, 0), dtype = object)
    phi[0] = lam[0]
    for j in range(Npts - 1):
        for k in range(j, Npts):
            xik = (Y[k] - Y[j]) / (Y[k] - conj(Y[j]))
            factor = np.array([
                [xik, phi[j]],
                [conj(phi[j]) * xik, gmp.mpc(1, 0)]
            ])
            abcd_bar_lst[k] = abcd_bar_lst[k] @ factor
        num = -abcd_bar_lst[j + 1][1, 1] * lam[j + 1] + abcd_bar_lst[j + 1][0, 1]
        denom = abcd_bar_lst[j + 1][1, 0] * lam[j + 1] - abcd_bar_lst[j + 1][0, 0]
        if is_zero(num):
            phi[j + 1] = gmp.mpc(0, 0)
        else:
            phi[j + 1] = num / denom
    return phi

def analytic_continuation(Y, phi, zspace, theta_mp1 = lambda z : 0):
    """
    Performs an analytic continuation assuming the function is Nevanlinna according to the
    algorithm prescribed in the paper.

    Parameters
    ----------
    Y : np.array[gmp.mpc]
        Matsubara frequencies the correlator is measured at.
    phi : np.array[gmp.mpc]
        Vector of phi[k] = theta_k(Y_k) values to input.
    zspace : np.array[np.float64]
        Linspace to evaluate the analytic continuation at, defined on C+
    theta_mp1 : Function
        Free function theta_{M + 1}(z) to fix in the algorithm.

    Returns
    -------
    np.array[gmp.mpc]
        len(zspace) array of values for the analytic continuation performed on zspace.
    """
    Nreal, Npts = len(zspace), len(Y)
    cont = np.empty((Nreal), dtype = object)
    for idx, z in enumerate(zspace):
        abcd = np.array([
            [gmp.mpc(1, 0), gmp.mpc(0, 0)],
            [gmp.mpc(0, 0), gmp.mpc(1, 0)]
        ])
        for k in range(Npts):
            # print('k: ' + str(k))
            xikz = (z - Y[k]) / (z - conj(Y[k]))
            # print('xi' + str(k) + ' is ' + str(xikz))
            factor = np.array([
                [xikz, phi[k]],
                [conj(phi[k]) * xikz, gmp.mpc(1, 0)]
            ])
            abcd = abcd @ factor
        num = abcd[0, 0] * theta_mp1(z) + abcd[0, 1]
        denom = abcd[1, 0] * theta_mp1(z) + abcd[1, 1]
        if is_zero(num):
            theta = gmp.mpc(0, 0)
        else:
            theta = num / denom         # contractive function theta(z)
        print(theta)    # theta(z) should have norm <= 1
        cont[idx] = hinv(theta)
    return cont

def write_txt_output(out_path, data, Nreal, omega, eta):
    """
    Writes output from analytic continuation to a text file.

    Parameters
    ----------
    out_path : string
        Output file path to write to. Should be a .txt file.
    data : np.array[gmp.mpc]
        Datapoints for function to write to file.
    Nreal : int
        Size of linspace that the function is evaluated on.
    omega : list[int]
        Pair of values [omega_min, omega_max] with the minimum and maximum values of omega.
    eta : float
        Imaginary part of evaluation line, i.e. the function is evaluated at f(x + i\eta) with x real.

    Returns
    -------
    """
    f = open(out_path, 'w')
    header = f'{Nreal:.0f} {omega[0]:.0f} {omega[1]:.0f} {eta:.5f}\n'
    f.write(header)
    for i, val in enumerate(data):
        s =  val.__str__() + '\n'
        f.write(s)
    f.close()
    return True
