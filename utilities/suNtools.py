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
n_boot = n_boot

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

def proj_vec(u, v):
    """Projects v onto linear subspace spanned by u."""
    return vec_dot(v, u) / vec_dot(u, u) * u

def proj_SU3(M):
    """
    Projects a matrix M to the group SU(3) by orthonormalizing the first two columns, then 
    taking a cross product.
    """
    N = M.copy()
    [v1, v2, v3] = M.T
    u1 = v1 / vec_norm(v1)              # normalize
    u2 = v2 - proj_vec(u1, v2)
    u2 = u2 / vec_norm(u2)
    u3 = np.cross(u1.conj(), u2.conj())
    return np.array([u1, u2, u3], dtype = np.complex64).T

def rand_su3_matrix(eps):
    """
    Generates a random SU(3) matrix for the metropolis update with parameter eps. 
    Follows Peter Lepage's notes.

    Parameters
    ----------
    eps : float
        Metropolis parameter for update candidate.
    
    Returns
    -------
    np.array [Nc, Nc]
        Metropolis update candidate.
    """
    mat_elems = np.random.uniform(low = -1, high = 1, size = 6)
    H = np.array([
        [mat_elems[0], mat_elems[1], mat_elems[2]], 
        [mat_elems[1], mat_elems[3], mat_elems[4]], 
        [mat_elems[2], mat_elems[4], mat_elems[5]]
    ], dtype = np.complex64)
    return proj_SU3(np.eye(3) + 1j*eps*H)