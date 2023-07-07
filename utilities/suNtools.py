################################################################################
# This script saves a list of common utility functions for manipulations of    #
# SU(N) valued group and algebra elements. To import the script, simply add    #
# the path with sys before importing. For example, if lqcd/ is in              #
# /Users/theoares:                                                             #
#                                                                              #
# import sys                                                                   #
# sys.path.append('/Users/theoares/lqcd/utilities')                            #
# from suNtools import *                                                       #
#                                                                              #
# Author: Patrick Oare                                                         #
################################################################################

from __main__ import *
# n_boot = n_boot
n_boot = 200

import numpy as np
import h5py
import os
import time
import re
import itertools
import io
import random

def vec_dot(u, v):
    return np.dot(u, np.conjugate(v))

def vec_norm(u):
    return np.sqrt(np.abs(vec_dot(u, u)))

def dagger(M):
    """Takes the Hermitian conjugate of a complex matrix M."""
    return M.transpose().conj()

def proj_vec(u, v):
    """Projects v onto linear subspace spanned by u."""
    return vec_dot(v, u) / vec_dot(u, u) * u

# def proj_SU3(M):
#     """
#     Projects a matrix M to the group SU(3) by orthonormalizing the first two columns, then 
#     taking a cross product.
#     """
#     [v1, v2, v3] = M.T
#     u1 = v1 / vec_norm(v1)              # normalize
#     u2 = v2 - proj_vec(u1, v2)
#     u2 = u2 / vec_norm(u2)
#     u3 = np.cross(u1.conj(), u2.conj())
#     return np.array([u1, u2, u3], dtype = np.complex128).T

# def rand_su3_matrix(eps):
#     """
#     Generates a random SU(3) matrix for the metropolis update with parameter eps. 
#     Follows Peter Lepage's notes.

#     Parameters
#     ----------
#     eps : float
#         Metropolis parameter for update candidate.
    
#     Returns
#     -------
#     np.array [Nc, Nc]
#         Metropolis update candidate.
#     """
#     mat_elems = np.random.uniform(low = -1, high = 1, size = 6)
#     H = np.array([
#         [mat_elems[0], mat_elems[1], mat_elems[2]], 
#         [mat_elems[1], mat_elems[3], mat_elems[4]], 
#         [mat_elems[2], mat_elems[4], mat_elems[5]]
#     ], dtype = np.complex64)
#     return proj_SU3(np.eye(3) + 1j*eps*H)

def proj_fund_suN(M, prec = 1e-8):
    """
    Projects a N x N matrix onto the fundamental representation of SU(N) by using the Gram-Schmidt procedure. 

    Parameters
    ----------
    M : np.array [N, N], dtype = np.complex128
        Matrix to project to SU(N).
    
    Returns
    -------
    proj(M) : np.array [N, N], dtype = np.complex128
        Projection of mat. 
    """
    N = M.shape[0]
    assert M.shape == (N, N), 'Matrix to project must be square.'
    cols = [M[:, i] for i in range(N)]
    for i in range(N):
        vi = cols[i]
        ui = vi.copy()
        for k in range(i):
            # cols[i] = cols[i] - proj_vec(u, v)
            ui -= proj_vec(cols[k], vi)
        cols[i] = ui / vec_norm(ui)
    U = np.array(cols, dtype = np.complex128).T
    detU = np.linalg.det(U)
    if np.abs(detU - 1) < prec:
        return U
    assert np.abs(np.abs(detU) - 1) < prec, '|det U| is not 1.'
    V = detU**(-1/N) * U
    return V

def rand_suN_matrix(N, eps):
    """
    Generates a random SU(N) matrix for the metropolis update with parameter eps. 
    Follows Peter Lepage's notes.

    Parameters
    ----------
    eps : float
        Metropolis parameter for update candidate.
    
    Returns
    -------
    np.array [N, N]
        Metropolis update candidate.
    """
    num = N * (N + 1) // 2               # number of real parameters for su(N)
    mat_elems = list(np.random.uniform(low = -1, high = 1, size = num))
    H = np.zeros((N, N), dtype = np.complex128)
    for i in range(N):
        elem = mat_elems.pop()
        H[i, i] = elem
    for i in range(N):
        for j in range(i):
            elem = mat_elems.pop()
            H[i, j] = elem
            H[j, i] = elem
    assert len(mat_elems) == 0
    return proj_fund_suN(np.eye(N) + 1j*eps*H)
