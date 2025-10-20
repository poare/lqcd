import numpy as np
# import gmpy2 as gmp
import h5py
import os
import time
import re
import itertools
import io
import random
import matplotlib.pyplot as plt
import mpmath as mp
import scipy
from abc import ABC

################################################################################
###################################### IO ######################################
################################################################################

def write_params(fname, rho, G, precision = 64):
    """
    Writes parameters to a .txt file to be saved. 
    """
    return

def read_param_file(fname):
    """
    Reads a .txt parameter file which specifies the number of omega points, tau points, ground truth spectral function, 
    and Green's function. 

    Parameters
    ----------
    fname : string
        File name to read from.
    
    Returns
    -------
    int
        Number of points on the omega line.
    int
        Number of temporal points.
    np.array[float64 or mp.mpc]
        Spectral function

    """
    # return n_omega, n_tau, rho, G
    return


################################################################################
################################## Nevanlinna ##################################
################################################################################

# Set precision for mp and define some basic functions / constants.
mp.dps = 100

# Basic data type for manipulation: np.ndarray with objects of type mp.mpc or mp.mpf. Note that 
# given either an mp.matrix or np.ndarray A, one can cast it to this type with `to_np(to_mp(A))`.
def to_np(A):
    """Converts a mp.matrix to an np.array."""
    return np.array(A.tolist())
def to_mp(A):
    """Converts a np.ndarray to a mp.matrix. Note this only works for square matrices."""
    return mp.matrix(A.tolist())

ZERO = mp.mpf("0")
ONE  = mp.mpf("1")
I    = mp.mpc("0", "1")
PI   = mp.pi()

def zeros(n):
    """Creates a zero matrix of dimension n with mpmath."""
    # return np.array(mp.zeros(n)).reshape(n, n)
    return to_np(mp.zeros(n))

def eye(n):
    """Creates a zero matrix of dimension n with mpmath."""
    # return np.array(mp.eye(n)).reshape(n, n)
    return to_np(mp.eye(n))

def minv(A):
    """Inverts a matrix represented as a np.ndarray of mp.mpc numbers."""
    return to_np(mp.inverse(to_mp(A)))

def msqrt(A):
    """Evaluates the matrix square root of a matrix."""
    return to_np(mp.sqrtm(to_mp(A)))

def mlog(A):
    """Evaluates the matrix logarithm of a matrix."""
    return to_np(mp.logm(to_mp(A)))

def twoNorm(v):
    """Evaluates the L2 norm of a vector of mp.mpc objects."""
    return mp.sqrt(sum([z*z.conjugate() for z in v]))

def factor_2by2(A):
    """Unpacks a 2x2 block matrix A into its four components."""
    n = A.shape[0] // 2
    return np.array([
        [A[:n, :n], A[:n, n:]],
        [A[n:, :n], A[n:, n:]]
    ])

def unit_vector(i, N):
    """Constructs the ith unit vector in R^n with mp.mpc numbers."""
    ei = np.array([ZERO for _ in range(N)])
    ei[i] = ONE
    return ei

SIGMA2 = to_np(mp.matrix([[0, -I], [I, 0]]))

# Cayley transform
h = lambda z : (z - I) / (z + I)
hinv = lambda q : I * (ONE + q) / (ONE - q)

# for some reason mpc segfaults when it uses the built in conjugate values, so just use this
# def conj(z):
#     """Returns the conjugate of a mp.mpc value z. For some reason, the built in mpc function .conj
#     breaks when it is called."""
#     return mp.mpc(z.real, -z.imag)

def to_mpfr(z):
    """Reformats a np.real number into a mp.mpf number"""
    return mp.mpf(z)

def to_mpc(z):
    """Reformats a np.complex number into a gmpy2.mpc number"""
    return mp.mpc(z)

# def is_zero(z, epsilon = 1e-10):
def is_zero(z, epsilon = 1e-20):
    """
    Returns whether a mp.mpc number z is consistent with 0 to the precision epsilon in both the
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
    function fk : mp.mpc -> mp.mpc
        kth basis function for the Hardy space.
    """
    def fk(z):
        """kth basis element for the standard Hardy space basis. Acts on gmpy2.mpc numbers."""
        return 1 / (mp.sqrt(PI) * (z + I)) * ((z - I) / (z + I)) ** k
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
    np.array[mp.mpc]
        Matsubara frequencies Y.
    np.array[mp.mpc]
        Correlator values C.
    np.array[mp.mpc]
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
            Y.append(mp.mpc(0, freq))
            C.append(mp.mpc(-reG, -imG))
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
    Y : np.array[mp.mpc]
        Matsubara frequencies the correlator is measured at.
    lam : np.array[mp.mpc]
        Mobius transform of the input correlator.

    Returns
    -------
    np.array[mp.mpc]
        len(lam) by len(lam) array for the Pick matrix.
    """
    Npts = len(Y)
    Pick = np.empty((Npts, Npts), dtype = object)
    for i in range(Npts):
        for j in range(Npts):
            num = 1 - lam[i] * mp.conj(lam[j])
            denom = 1 - h(Y[i]) * mp.conj(h(Y[j]))
            Pick[i, j] = num / denom
    return Pick

def is_soluble(Y, lam, prec = 1e-10):
    """
    Determines if there is a Nevanlinna interpolant to this set of Y and lambda values.

    Parameters
    ----------
    Y : np.array[mp.mpc]
        Matsubara frequencies the correlator is measured at.
    lam : np.array[mp.mpc]
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
    This is the indexing used in the code from the Nevanlinna paper.
    Constructs the phi[k] values needed to perform the Nevanlinna analytic continuation.
    At each iterative step k, phi[k] is defined as theta_k(Y_k), where theta_k is the kth
    iterative approximation to the continuation.

    Parameters
    ----------
    Y : np.array[mp.mpc]
        Matsubara frequencies the correlator is measured at.
    lam : np.array[mp.mpc]
        Mobius transform of the input correlator.

    Returns
    -------
    np.array[mp.mpc]
        Values of phi[k] = theta_k(Y_k).
    """
    Npts = len(Y)
    phi = np.full((Npts), ZERO, dtype = object)
    phi[0] = lam[0]
    for j in range(1, Npts):
        arr = np.array([
            [ONE, ZERO],
            [ZERO, ONE]
        ])
        for k in range(0, j):
            xik = (Y[j] - Y[k]) / (Y[j] - mp.conj(Y[k]))
            factor = np.array([
                [xik, phi[k]],
                [mp.conj(phi[k]) * xik, ONE]
            ])
            arr = arr @ factor
        [[a, b], [c, d]] = arr
        # print('Num: ' + str(lam[j] * d - b))
        # print('Denom: ' + str(a - lam[j] * c))
        phi[j] = (lam[j] * d - b) / (a - lam[j] * c)
    return phi

def construct_phis_non_lex(Y, lam):
    """
    Constructs the phi[k] values needed to perform the Nevanlinna analytic continuation.
    At each iterative step k, phi[k] is defined as theta_k(Y_k), where theta_k is the kth
    iterative approximation to the continuation.

    This function uses a different ordering to construct each \phi_k value. Instead of computing phi_1, phi_2, phi_3, ... 
    sequentially, it stores intermediate steps in the product for all \phi_k values at y_1 first, then does y_2, 
    and so on. 

    Parameters
    ----------
    Y : np.array[mp.mpc]
        Mobius transform of Matsubara frequencies the correlator is measured at.
    lam : np.array[mp.mpc]
        Mobius transform of the input correlator.

    Returns
    -------
    np.array[mp.mpc]
        Values of phi[k] = theta_k(Y_k).
    """
    Npts = len(Y)
    abcd_bar_lst = []
    for k in range(Npts - 1):
        id = np.array([
            [ONE, ZERO],
            [ZERO, ONE]
        ])
        abcd_bar_lst.append(id)
    phi = np.full((Npts), ZERO, dtype = object)         # TODO make sure this doesn't break anything if it's by reference
    phi[0] = lam[0]
    for k in range(Npts - 1):
        for j in range(k, Npts - 1):
            xik = (Y[j + 1] - Y[k]) / (Y[j + 1] - mp.conj(Y[k]))
            factor = np.array([
                [xik, phi[k]],
                [mp.conj(phi[k]) * xik, ONE]
            ])
            abcd_bar_lst[j] = abcd_bar_lst[j] @ factor
        num = lam[k + 1] * abcd_bar_lst[k][1, 1] - abcd_bar_lst[k][0, 1]
        denom = abcd_bar_lst[k][0, 0] - lam[k + 1] * abcd_bar_lst[k][1, 0]
        print('num: ' + str(num))
        print('denom: ' + str(denom))
        if is_zero(num):
            phi[k + 1] = ZERO
        else:
            phi[k + 1] = num / denom
    return phi

def analytic_continuation(Y, phi, zspace, theta_mp1 = lambda z : 0):
    """
    Performs an analytic continuation assuming the function is Nevanlinna according to the
    algorithm prescribed in the paper.

    Parameters
    ----------
    Y : np.array[mp.mpc]
        Matsubara frequencies the correlator is measured at.
    phi : np.array[mp.mpc]
        Vector of phi[k] = theta_k(Y_k) values to input.
    zspace : np.array[np.float64]
        Linspace to evaluate the analytic continuation at, defined on C+
    theta_mp1 : Function
        Free function theta_{M + 1}(z) to fix in the algorithm.

    Returns
    -------
    np.array[mp.mpc]
        len(zspace) array of values for the analytic continuation performed on zspace.
    """
    Nreal, Npts = len(zspace), len(Y)
    cont = np.empty((Nreal), dtype = object)
    for idx, z in enumerate(zspace):
        abcd = np.array([
            [ONE, ZERO],
            [ZERO, ONE]
        ])
        for k in range(Npts):
            xikz = (z - Y[k]) / (z - mp.conj(Y[k]))
            factor = np.array([
                [xikz, phi[k]],
                [mp.conj(phi[k]) * xikz, ONE]
            ])
            abcd = abcd @ factor
        num = abcd[0, 0] * theta_mp1(z) + abcd[0, 1]
        denom = abcd[1, 0] * theta_mp1(z) + abcd[1, 1]
        if is_zero(num):
            theta = ZERO
        else:
            theta = num / denom         # contractive function theta(z)
        # print(theta)    # theta(z) should have norm <= 1
        cont[idx] = hinv(theta)
    return cont

def write_txt_output(out_path, data, Nreal, omega, eta):
    """
    Writes output from analytic continuation to a text file.

    Parameters
    ----------
    out_path : string
        Output file path to write to. Should be a .txt file.
    data : np.array[mp.mpc]
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

################################################################################
################################## Kovalishina #################################
################################################################################

class WeylBall:
    """
    Implements a Weyl matrix ball, defined as,
    $$
        B(C; L, R) := \{C + LXR : 1 - X^\dagger X\succeq 0 \}.
    $$
    The Weyl matrix ball generalizes the disk to matrices 
    of dimension n. 

    Fields
    ------
    n : int
        Dimension of the Weyl matrix ball
    center : np.array [N, N], dtype = np.complex128 or mp.mpc
        Center of the matrix ball.
    lRad : np.array [N, N], dtype = mp.mpc or np.complex128
        Left (gauche) radius for the matrix ball.
    rRad : np.array [N, N], dtype = np.complex128 or mp.mpc
        Right (droit) radius for the matrix ball. 
    """
    n      : int
    center : np.ndarray
    lRad   : np.ndarray
    rRad   : np.ndarray

    def __init__(self, _center, _lRad, _rRad):
        """
        Initializes a Weyl matrix ball. 
        """
        self.n      = _center.shape[0]
        self.center = _center
        self.lRad   = _lRad
        self.rRad   = _rRad
        return
    
    def elementof(self, Z : np.ndarray):
        """
        Returns true if Z is a member of the given Weyl matrix ball.
        """
        # eye(n) - mp.inverse(self.lRad) * (Z - self.center) * mp.inverse(self.rRad)

        return
    
    def projection(self, a : np.ndarray, b : np.ndarray):
        """
        Projects the Weyl ball into a disk.
        """
        return

class Hankel:
    """
    Fields
    ------
    n : int
        Number of moments n. The Hankel matrix stores information about the moments 0, 1, ..., n, n+1, ..., 2n.
    k : int
        Dimension of the problem (size of each matrix / moment).
    moments : np.ndarray [2n+1, k], dtype = np.complex128 or mp.mpc
        Moments [C0, ..., C_{2n}] for the Hankel matrix.
    hankel : np.ndarray [(n+1)k, (n+1)k], dtype = np.complex128 or mp.mpc
        Hankel matrix,
        $$
            [
                [C0, C1, ..., C_n],
                [C1, C2, ..., C_n+1],
                ...
                [Cn, C_n+1, ..., C_2n],
            ].
        $$
    """
    n       : int
    k       : int
    moments : np.ndarray
    hankel  : np.ndarray

    def __init__(self, _mvec : np.ndarray):
        """
        Parameters
        ----------
        _mvec : np.ndarray (2n+1, k, k), dtype = mp.mpc or np.complex128
            Vector of 2n+1 moments [C0, C1, ..., C_{2n}], each of which is a k-dimensional matrix.
        """
        # _mvec = to_np(to_mp(_mvec))                 # cast _mvec to np.ndarray of mp.mpc elements
        # self.n       = (_mvec.shape[0] - 1) // 2
        self.n       = _mvec.shape[0] // 2
        self.k       = _mvec.shape[1]
        self.moments = _mvec
        _hankel = []
        # for i in range(self.n + 1):                 # populate ith row
        for i in range(self.n):                 # populate ith row
            row = []
            # for j in range(i, self.n + 1 + i):      # get N+1 moments i, i+1, ..., i+1+N
            for j in range(i, self.n + i):      # get N+1 moments i, i+1, ..., i+1+N
                # row[i].append(_mvec[j])
                row.append(_mvec[j])
            _hankel.append(row)
        self.hankel = np.block(_hankel)
    
    def isPSD(self):
        """
        Returns true if self.hankel is positive semi-definite, false otherwise. 
        """
        try:
            mp.cholesky(to_mp(self.hankel))
        except ValueError as e:
            print(e)
            return False
        return True

# TODO: make an abstract parent class Kovalishina that can 
# then have specific implementations for NevanlinnaPick, 
# Caratheodory, and Hamburger.
# class Kovalishina (ABC):
#     """
#     Abstract class to generalize the Kovalishina interpolation.
#     """
#     # moments : mp.matrix
#     def __init__(self, _moments):
#         self.moments = _moments
#         return

# class Hamburger (Kovalishina):
class Hamburger:
    """
    Computes the solution to the Hamburger moment problem, given either 
        (a) A list of moments [C0, C1, ..., C_{2n}], or
        (b) A Hankel matrix of type 'Hankel'.
    The 'Hamburger' class can take in an optional parameter _eval_line, which is the 
    evaluation axis for the problem. In this case, it will store the matrix U(z) and 
    the Weyl matrix W(z), computed for each z in _eval_line. If _eval_line is not input 
    when instantiating a 'Hamburger' object, one can still compute U(z) and W(z) for a 
    given z or pass in an evaluation axis later. 

    hankel    : Hankel
        Hankel matrix of moments H (or allow a vector to be input).
    n         : int
        n parameter from paper, moments C0, C1, ..., C_{2n} calculated.
    k         : int
        Size of the block problem (each moment is a k by k matrix).
    J         : np.ndarray (2k, 2k), dtype = np.complex128 or mp.mpc
        k by k J-matrix. For Hamburger moment problem, J is block sigma2.
    eval_axis : None or np.ndarray (M), dtype = np.complex128 or mp.mpc
        Axis to evaluate the Weyl matrix at. Can be None and computed later. 
    M         : int
        Size of evaluation axis
    U         : np.ndarray (2k, 2k, M), dtype = np.complex128 or mp.mpc
        U matrix from Kovalishina. 
    W         : np.ndarray (2k, 2k, M), dtype = np.complex128 or mp.mpc
        Weyl matrix from Kovalishina. 
    center    : np.ndarray (k, k, M), dtype = np.complex128 or mp.mpc
        Center of the component-wise Wertevorrat. 
    radius    : np.ndarray (k, k, M), dtype = np.complex128 or mp.mpc
        Radius of the component-wise Wertevorrat. 
    """
    hankel      : Hankel
    n           : int
    k           : int
    J           : np.ndarray
    eval_axis   : np.ndarray
    M           : int
    U           : np.ndarray
    W           : np.ndarray
    center      : np.ndarray            # [k, k, N] center of component-wise Wertevorrat
    radius      : np.ndarray            # radius of component-wise Wertevorrat

    def __init__(self, _moments, _eval_axis = None):
        # super().__init__(_moments)
        if isinstance(_moments, np.ndarray):
            self.hankel = Hankel(_moments)
        elif isinstance(_moments, Hankel):
            self.hankel = _moments
        else:
            raise Exception('_moments must either be a vector of moments or a Hankel matrix.')
        
        self.n       = self.hankel.n
        self.k       = self.hankel.k
        self.J = np.block([
            [zeros(self.k),  I*eye(self.k)],
            [-I*eye(self.k), zeros(self.k)]
        ])

        self.eval_axis = None
        self.M = 0
        self.U = None; self.W = None
        self.center = None; self.radius = None
        if _eval_axis:                  # if eval_line is passed in, then compute coeffs
            self(_eval_axis)
        return

    # def set_evaluation_axis(self, _eval_axis):
    def __call__(self, _eval_axis):
        self.M = len(_eval_axis)
        self.eval_axis = _eval_axis
        U = []; W = []
        # center = []; rad = []
        self.center = np.full((self.k, self.k, self.M), ZERO)
        self.radius = np.full((self.k, self.k, self.M), ZERO)
        for idx, z in enumerate(_eval_axis):
            Uz = self.compute_U(z)
            print(Uz)
            Wz = self.compute_Weyl(Uz)
            U.append(Uz); W.append(Wz)

            # unpack Weyl matrix. W = block matrix of [[-R, S], [S^dag, -T]]
            blocks = factor_2by2(Wz)
            Rz = -blocks[0, 0]
            Sz =  blocks[0, 1]
            Tz = -blocks[1, 1]

            # Compute radius and center of Wertevorrat
            left = minv(Rz)
            C = left @ Sz
            right = Sz.conjugate().transpose() @ C - Tz

            # project Wertevorrat onto components
            left_sqrt = msqrt(left)
            right_sqrt = msqrt(right)
            for i in range(self.k):
                for j in range(i, self.k):
                    ei = unit_vector(i, self.k); ej = unit_vector(j, self.k)
                    self.center[i, j, idx] = ei.transpose() @ C @ ej
                    self.radius[i, j, idx] = twoNorm(left_sqrt.transpose().T @ ei) \
                                                * twoNorm(right_sqrt @ ej)
                    if i != j:          # populate bottom triangular elements as well
                        self.center[j, i, idx] = self.center[i, j, idx]
                        self.radius[j, i, idx] = self.radius[i, j, idx]

        self.U = np.array(U); self.W = np.array(W)
        return
    
    def map_back(self, phases = np.array([mp.expjpi(n) for n in mp.linspace(0, 4, 50)])):
        """
        Maps the Wertevorrat back to the complex energy plane. 
        """
        upper = np.full((self.k, self.k, self.M), ZERO)
        lower = np.full((self.k, self.k, self.M), ZERO)
        for i in range(self.k):
            for j in range(i, self.k):
                for zidx in range(self.M):
                    # TODO figure out if I need the - or not
                    wv_imag = np.array([(self.center[i, j, zidx] + p * self.radius[i, j, zidx]).imag for p in phases])
                    upper[i, j, zidx] = np.max(wv_imag)
                    lower[i, j, zidx] = np.min(wv_imag)
                    if i != j:
                        upper[j, i, zidx] = upper[i, j, zidx]
                        lower[j, i, zidx] = lower[i, j, zidx]
        return upper, lower

    def compute_Weyl(self, U):
        """
        Computes the Weyl matrix given U(z). To compute the Weyl matrix at z, simply 
        run compute_U(z) first. 
        ```
        # ham is an instance of Hamburger class
        U = ham.compute_U(z)
        W = ham.compute_W(U)
        ```
        """
        # U = self.compute_U(z)
        Uinv = minv(U)
        W = Uinv.conjugate().transpose() @ self.J @ Uinv
        return W

    def compute_U(self, z):
        """
        Computes the U(z) matrix containing the Kovalishina coefficients using Eq. (A.22).
        
        Parameters
        ----------
        z : np.complex128 or mp.mpc
            Complex number z to evaluate U(z) at.
        
        Returns
        -------
        U(z) : np.ndarray [2k, 2k], dtype = mp.mpc
            2x2 block matrix U solving the FMI. 
        """
        vec1 = np.block(
            [-self.compute_c(ZERO), self.compute_b(ZERO)]
        )           # (nk, 2k) dimensional
        vec2 = np.block(
            [
                [-self.compute_c(mp.conj(z)).conjugate().transpose()],
                [ self.compute_b(mp.conj(z)).conjugate().transpose()]
            ]
        )           # (2k, nk) dimensional
        # return eye(2*self.k) - I * (vec2 @ minv(self.hankel.hankel) @ vec1)
        # return eye(2*self.k) - I * z* (vec2 @ minv(self.hankel.hankel) @ vec1)
        vec1 = np.block(
            [self.compute_b(ZERO), -self.compute_c(ZERO)]
        )           # (nk, 2k) dimensional
        vec2 = np.block(
            [
                [self.compute_b(mp.conj(z)).conjugate().transpose()],
                [-self.compute_c(mp.conj(z)).conjugate().transpose()],
            ]
        )           # (2k, nk) dimensional
        return eye(2*self.k) - I * z * self.J @ (vec2 @ minv(self.hankel.hankel) @ vec1)

    def compute_b(self, z):
        """
        Computes the b block vector b(z) = (I zI z^2I ... z^{n-1} I).

        Parameters
        ----------
        z : np.complex128 or mp.mpc
            Complex number z to evaluate b(z) at.
        
        Returns
        -------
        b(z) : np.ndarray [nk, k], dtype = mp.mpc
            n-dimensional block vector b(z).
        """
        return np.block(
            [[mp.power(z, i)*eye(self.k)] for i in range(self.n)]
        )
    
    def compute_c(self, z):
        """
        Computes the c block vector evaluated at z,
        $$
            -c(z) = [
                0 
                C_0 
                C_1+zC_0
                ...
                C_{n-2} + z C_{n-1} + ... + z^{n-2} C_0
            ]
        $$

        Parameters
        ----------
        z : np.complex128 or mp.mpc
            Complex number z to evaluate c(z) at.

        Returns
        -------
        c(z) : np.ndarray [nk, k], dtype = mp.mpc
            n-dimensional block vector c(z).
        """
        c_lst = []
        for i in range(self.n):
            tmp = zeros(self.k)
            for j in range(i):
                tmp += mp.power(z, j) * self.hankel.moments[i-j-1]
            c_lst.append([tmp])
        return -np.block(c_lst)

################################################################################
################################ Sparse Modeling ###############################
################################################################################

def supnorm(x):
    """
    Returns the supnorm ||x||_\infty of a matrix x, defined as the maximum element of the absolute
    value of x.
    """
    return np.max(np.abs(x))

def lpnorm(p):
    """
    Returns the L^p norm function of a vector.

    Parameters
    ----------
    p : int
        p-norm to return

    Returns
    -------
    function : np.array --> float
        Function which returns the p-norm of a vector x, defined as ||x||_p^p = \sum_k | x_k |^p
    """
    def pnorm(x):
        return np.sum(np.abs(x) ** p) ** (1 / float(p))
    return pnorm

def check_mat_equal(A, B, eps = 1e-8, norm = supnorm):
    """
    Checks that two matrices are equal element-wise up to tolerance epsilon. Returns if they are equal,
    and either the deviation or the indices where they are not equal.

    Parameters
    ----------
    A, B : np.array
        Matrices to compare.
    eps : float (default = 1e-8)
        Tolerance to compare to
    norm : function (default = supnorm)
        Norm to use for the comparison.

    Returns
    -------
    bool
        True if equal, False if not equal.
    float
        Deviation of A - B from 0.
    """
    dev = norm(A - B)
    return dev < eps, dev

def hc(M):
    """
    Returns the hermitian conjugate of a numpy array M.
    """
    return M.conj().T

def svals_to_mat(svals, Ntau, Nomega):
    """
    Embeds a vector of singular values into the diagonal of an Ntau x Nomega matrix S.
    """
    return np.pad(np.diag(svals), [(0, 0), (0, Nomega - Ntau)])

def laplace_kernel(taus, omegas):
    """
    Returns the Laplace kernel at discretized tau and omega points, K_ij = e^{-omega_j tau_i}.

    Parameters
    ----------
    taus : np.array [Ntau]
        Times to evaluate the kernel at.
    omegas : np.array [Nomega]
        Freqencies to evaluate the kernel at.

    Returns
    -------
    np.array [Ntau, Nomega]
        Laplace kernel, evaluated at the given times and frequencies.
    """
    Ntau, Nomega = len(taus), len(omegas)
    kernel = np.zeros((Ntau, Nomega), np.float64)
    for n, tau in enumerate(taus):
        for k, omega in enumerate(omegas):
            kernel[n, k] = np.exp(- omega * tau)
    return kernel

def lin_combo(*args):
    """
    Returns a function representing the linear combination of passed in functions. Input should be
    of the form [c1, f1], ..., [cn, fn] and returns the linear combination c1 * f1(x) + ... + cn * fn(x).
    """
    def fn(x):
        res = 0.
        for (cn, fn) in args:
            res += cn * fn(x)
        return res
    return fn

def soft_threshold(x, beta):
    """
    Soft thresholding operator with parameter beta. This function either returns the
    soft threshold of x, or performs the operation element-wise. Formally, this is:
    S_beta(x) = \begin{cases}
        x - \beta       && x > \beta \\
        0               && \beta > x > -\beta \\
        x + \beta       && x < -\beta
    \end{cases}

    Parameters
    ----------
    x : np.array
        Function value for soft-thresholding
    beta : float
        Parameter for soft-threshold function.

    Returns
    -------
    float
        Value of the function.
    """
    thresh = np.maximum(np.abs(x) - beta, 0)
    return thresh * np.sign(x)

def proj_nneg(z):
    """
    Projection P_+ of a vector onto the non-negative quadrant of R^n.

    Parameters
    ----------
    z : np.array
        Vector to project.

    Returns
    -------
    np.array
        Projection of vector, with components max(z_j, 0).
    """
    return np.array([zk if zk >= 0.0 else 0.0 for zk in z])

class ADMMParams:
    """
    Class to hold parameters for ADMM. Holds the following parameters:

    Fields
    ------
    lam : float
        Multiplier for the L_1 regularizer.
    mu : float
        Lagrange multiplier for non-negativity constraint.
    mup : float
        Lagrange multiplier for the sum rule constraint.
    max_iters : int
        Maximum number of iterations.
    eps : float
        Tolerance for convergance of the algorithm.
    xp0 : np.array
        Vector to optimize over, i.e. rho'.
    zp0 : np.array
        Auxiliary vector for the constraint z' = x', which implements the sum rule.
    up0 : np.array
        Vector to implement Lagrange multiplier on sum rule constraint.
    z0 : np.array
        Auxiliary vector for the constraint z = V x', which implements non-negativity.
    u0 : np.array
        Vector to implement Lagrange multiplier on non-negativity constraint.
    dim : [2]
        Dimension (Ntau, Nomega) of optimization problem.
    """

    default_constraints = {
        'nneg'          : True,
        'sum_rule'      : True,
        'states'        : []
    }

    def __init__(self, lam, mu, mup, max_iters, eps, xp0, zp0, up0, z0, u0, dim = None, constraints = default_constraints):
        self.lam = lam
        self.mu = mu
        self.mup = mup

        self.max_iters = max_iters
        self.eps = eps

        self.xp0 = xp0
        self.zp0 = zp0
        self.up0 = up0
        self.z0 = z0
        self.u0 = u0

        self.dim = dim
        self.constraints = constraints

    def set_dim(self, dim):
        self.dim = dim

    @staticmethod
    def default_params(Nomega, d = None):
        """
        Default parameters for ADMM.
        """
        return ADMMParams(1., 1., 1., 10000, 1e-8, np.zeros((Nomega), dtype = np.float64), np.zeros((Nomega), dtype = np.float64), \
                    np.zeros((Nomega), dtype = np.float64), np.zeros((Nomega), dtype = np.float64), \
                    np.zeros((Nomega), dtype = np.float64), dim = d)

def admm(G, taus, omegas, params, resid_norm = lpnorm(2), disp_iters = 100):
    """
    Implements the Alternating Direction Method of Multipliers (ADMM) to solve the
    minimization:
        \hat{\rho} = argmin_x || G - K x ||_2^2 + lambda || x' ||_1
    where x' = V^dagger x is the IR representation of x. Convergence is defined
    by z converging to V x' with respect to the resid_norm, which can be specified.

    Parameters
    ----------
    G : np.array [Ntau]
        Input data for the Green's function.
    taus : np.array [Ntau]
        Euclidean times the correlator is evaluated at.
    omegas : np.array [Nomega]
        Frequencies to evaluate spectral function at.
    params : ADMMParams
        Parameters for initialization for ADMM algorithm.
    resid_norm : function (default = lpnorm(2))
        Norm function to use for residual term ||z - V x'||.
    disp_iters : int
        Number of iterations to display an update to residual and time.

    Returns
    -------
    np.array
        Projection of vector, with components max(z_j, 0).
    """
    # initialize kernel
    Ntau, Nomega = params.dim
    K = laplace_kernel(taus, omegas)
    U, svals, Vdag = np.linalg.svd(K)
    S = svals_to_mat(svals, Ntau, Nomega)       # K = U S Vdag
    Udag, V = hc(U), hc(Vdag)
    Gp = Udag @ G
    DelOmega = (omegas[-1] - omegas[0]) / Nomega

    # initialize vectors and parameters
    lam, mu, mup = params.lam, params.mu, params.mup
    max_iters, eps = params.max_iters, params.eps
    xp = params.xp0
    zp, up = params.zp0, params.up0
    z, u = params.z0, params.u0

    # run optimization
    ii = 0
    dual_resid = eps + 1          # start with something > epsilon
    start = time.time()
    print('Starting solver.')
    while (ii < max_iters) and dual_resid > eps:
        xp, zp, up, z, u = admm_update(Gp, xp, zp, up, z, u, params, svals, V, params.constraints, DelOmega)
        primal_resid = resid_norm(Gp - S @ xp)    # G' - S rho'
        dual_resid = resid_norm(z - (V @ xp))
        ii += 1
        if ii % disp_iters == 0:
            print('Iteration ' + str(ii) + ': primal residual = ' + str(primal_resid) + ', dual resid = ' \
                    + str(dual_resid) + ', elapsed time = ' + str(time.time() - start))
    print('Run complete. \n   Iterations: ' + str(ii) + '\n   Primal residual: ' + str(primal_resid) + '\n   Dual residual: ' \
            + str(dual_resid) + '\n   Elapsed time: ' + str(time.time() - start))
    # rho = V @ xp
    # rho = proj_nneg(V @ xp)
    rho = z    # TODO: what's the difference between these three ways to get rho?
    return rho, xp, primal_resid, dual_resid, ii

def admm_update(Gp, xp, zp, up, z, u, params, svals, V, constraints, DelOmega = 1.):
    Ntau, Nomega = params.dim
    lam, mu, mup = params.lam, params.mu, params.mup
    VT = V.T
    e = np.ones((Nomega), dtype = np.float64)
    # inverse_mat = np.zeros((Nomega, Nomega), dtype = np.float64)
    inverse_mat_vec = np.zeros((Nomega), dtype = np.float64)
    StGp = np.zeros((Nomega), dtype = np.float64)
    for idx in range(Nomega):
        mat_val = mu + mup
        if idx < len(svals):
            mat_val += svals[idx] * svals[idx].conj() / lam
            StGp[idx] = svals[idx] * Gp[idx]
        # inverse_mat[idx, idx] = 1 / mat_val
        inverse_mat_vec[idx] = 1 / mat_val
        # Cheaper to implement diagonal matrix product as vector elementwise multiplication.
    xi1 = inverse_mat_vec * (StGp / lam + mup * (zp - up) + mu * (VT @ (z - u)))
    xi2 = inverse_mat_vec * (VT @ e)
    # c = 1 / DelOmega
    c = 1
    nu = (c - np.sum(V @ xi1)) / np.sum(V @ xi2)
    xp = xi1 + nu * xi2
    zp = soft_threshold(xp + up, 1 / mup)
    up = up + xp - zp
    # z = proj_nneg(V @ xp + u)
    z = V @ xp + u
    if constraints['nneg']:
        z = proj_nneg(z)
    u = u + (V @ xp) - z
    return xp, zp, up, z, u

def parameter_scan(G, taus, omegas, lam_list, mu_list, mup_list, max_iters, eps = 1e-5):
    """
    Runs the ADMM algorithm over the range of (lambda, mu, mup) given as input.

    Parameters
    ----------
    G : np.array [Ntau]
        Input data for the Green's function.
    taus : np.array [Ntau]
        Euclidean times the correlator is evaluated at.
    omegas : np.array [Nomega]
        Frequencies to evaluate spectral function at.
    lam_list : np.array [Nlam]
        Values of hyperparameter lambda to scan over.
    mu_list : np.array [Nmu]
        Values of hyperparameter mu to scan over.
    mup_list : np.array [Nmup]
        Values of hyperparameter mup to scan over.
    max_iters : int
        Maximum number of iterations for each ADMM run.

    Returns
    -------
    np.array [Nlam, Nmu, Nmup, Nomega]
        Spectral function recons for each triple of parameters (lambda, mu, mup).
    """
    Nomega, Ntau = len(omegas), len(taus)
    rho_recons = np.zeros((len(lam_list), len(mu_list), len(mup_list), Nomega))
    p_resids = np.zeros((len(lam_list), len(mu_list), len(mup_list)))
    d_resids = np.zeros((len(lam_list), len(mu_list), len(mup_list)))
    for lidx, lam in enumerate(lam_list):
        for muidx, mu in enumerate(mu_list):
            for mupidx, mup in enumerate(mup_list):
                print('(lambda, mu, mup) = ' + str((lam, mu, mup)))
                params = ADMMParams.default_params(Nomega, d = (Ntau, Nomega))
                params.lam = lam
                params.mu = mu
                params.mup = mup
                params.max_iters = max_iters
                params.eps = eps
                tmp = admm(G, taus, omegas, params, disp_iters = max_iters / 10)
                rho_recons[lidx, muidx, mupidx] = tmp[0]
                p_resids[lidx, muidx, mupidx] = tmp[2]
                d_resids[lidx, muidx, mupidx] = tmp[3]
    return rho_recons, p_resids, d_resids

def min_indices(A, k):
    """Returns the indices corresponding to the k minimum values of a multi-dimensional array A."""
    return np.array([np.unravel_index(x, A.shape) for x in np.argsort(A.flatten())[:k]])

def parse_resids(rho_recons, resids, Nbest = 8):
    """
    Returns the indices of the Nbest minimal residuals in the list resids after a parameter scan.
    Assumes that resids has the same shape as output by parameter_scan, i.e. is of shape (N_lam, N_mu, N_mup).

    Parameters
    ----------
    resids : np.array [Nlam, Nmu, Nmup]
        Residual list to parse.
    Nbest : int (default = 8)
        Returns the indices of the Nbest smallest residuals.

    Returns
    -------
    np.array [Nbest]
        Indices of the Nbest smallest residuals.
    np.array [Nbest]
        Rho recon of Nbest smallest residuals.
    """
    idxs = min_indices(resids, Nbest)
    print('Lowest residuals: ' + str([resids[idx[0], idx[1], idx[2]] for idx in idxs]))
    best_rhos = np.array([rho_recons[idx[0], idx[1], idx[2]] for idx in idxs])
    return best_rhos, idxs
