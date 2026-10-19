"""
Utility code for the moment problem. At the moment, it is mostly just trying to understand the problem a bit better. 
"""

import numpy as np
import numpy.polynomial.polynomial as poly
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
import utils

################################################################################
############################ ORTHOGONAL POLYNOMIALS ############################
################################################################################

class RieszFunctional:
    """
    Given a truncated moment sequence (s_j)_{j=0}^{m}, represents the Riesz functional on C[x]_m
    and its Hilbert space completion H_s. Note that here C[x]_m denotes the finite-dimensional algebra 
    of polynomials with degree <= m with complex coefficients, implemented as instances of the class 
    numpy.polynomial.polynomial.Polynomial. 
    """
    s       : np.ndarray
    m       : int

    def __init__(self, _s : np.ndarray):
        """
        Constructs a Riesz functional, given a moment sequence _s. 

        Parameters
        ----------
        s : np.ndarray [m+1], dtype = np.complex128 or mp.mpc
            Moment sequence to use to realize the functional. 
        """
        self.s = _s
        self.m = len(_s) - 1
    
    def __call__(self, p):
        """
        Calls the Riesz functional on a polynomial p\in C[x]_m. 

        Parameters
        ----------
        p : poly.Polynomial 
            Polynomial to evaluate functional on. 
        """
        n = p.degree()
        assert n <= self.m, 'Polynomial has too high of a degree.'
        Ls = mp.mpf("0")
        for i in range(self.m + 1):
            ci = p.coef[i] if i < len(p.coef) else mp.mpc("0")
            print(ci)
            Ls += ci * self.s[i]
        return Ls


