
################################################################################
# This script saves a list of common utility functions for gauge field         #
# simulations that I may need to use in my python code. To import the script,  #
# simply add the path with sys before importing. For example, if lqcd/ is in   #
# /Users/theoares:                                                             #
#                                                                              #
# import sys                                                                   #
# sys.path.append('/Users/theoares/lqcd/utilities')                            #
# from gaugetools import *                                                     #
#                                                                              #
# Author: Patrick Oare                                                         #
################################################################################

from __main__ import *

import numpy as np
import h5py
import os
import time
import re
import itertools
import io
import random

import gvar as gv
import lsqfit
import pandas as pd

class GaugeField:
    """
    Container class for a gauge field. A gauge field will be stored as a numpy 
    array of size (d, Lx, Ly, Lt, Lz, Nc, Nc), where LL = [Lx, Ly, Lz, Lt] is 
    the lattice geometry.

    Fields
    ------
    _U : np.array [d, Lx, Ly, Lz, Lt, Nc, Nc] (dtype = np.complex64)
        Gauge field. Defaults to initializing a zero gauge field.
    _tol : double
        Tolerance to use to compare different gauge fields.
    _LL : np.array [d] CONSTANT
        Dimensions of lattice. 
    _D : int (default = 4) CONSTANT
        Number of spacetime dimensions.
    _NDIMS : int >= 0 (default = 1)
        Number of Lorentz indices for the GaugeField to have. _NDIMS = 0 is a color 
        matrix field, _NDIMS = 1 is a standard gauge field U_mu(n), and _NDIMS = 2 
        is a plaquette field. 
    _NC : int (default = 3) CONSTANT
        Number of colors.
    
    _SHAPE : tuple (_D, *_LL, _NC, _NC)
        Shape of gauge field
    """

    DTYPE = np.complex64

    def __init__(self, LL, U = None, n_dims = 1, tol = 1e-5, d = 4, Nc = 3):
        if U is None:
            # self._U = np.zeros((d, *LL, Nc, Nc), dtype = np.complex64)
            self._U = np.zeros((*([d]*n_dims), *LL, Nc, Nc), dtype = GaugeField.DTYPE)
        else:
            # assert U.shape == (d, *LL, Nc, Nc)
            assert U.shape == (*([d]*n_dims), *LL, Nc, Nc)
            self._U = U
        self._tol = tol
        self._LL = LL
        self._D = d
        self._NDIMS = n_dims
        self._NC = Nc
        self._SHAPE = (*([d]*n_dims), *LL, Nc, Nc)
    
    def set_U(self, Unew):
        assert Unew.shape == self._SHAPE
        self._U = Unew
        return self
    
    def get_U(self):
        return self._U
    def get_tol(self):
        return self._tol
    def get_d(self):
        return self._D
    def get_ndims(self):
        return self._NDIMS
    def get_Nc(self):
        return self._NC
    def shape(self):
        return self._SHAPE

    def update_link(self, n, mu, Up):
        """
        Updates the link at (n, mu) to a color matrix Up.

        Parameters
        ----------
        self : GaugeField
            Gauge field to update.
        n : np.array[d]
            4-position to update the link at.
        """
        self._U[(mu, *n)] = Up
        return self
    
    def __getitem__(self, key):
        """
        key can either be a set of Lorentz indices + spacetime index 
        (to get a color matrix) or a full index.
        """
        key_bc = tuple([key[ii] % self._LL[ii] for ii in range(len(key))])
        return self._U[key_bc]

    def __setitem__(self, key, item):
        """
        key can either be a set of Lorentz indices + spacetime index 
        (to get a color matrix) or a full index.
        """
        key_bc = tuple([key[ii] % self._LL[ii] for ii in range(len(key))])
        self._U[key_bc] = item
    
    def __mul__(self, other):
        return self.__lmul__(other)

    def __rmul__(self, other):
        """
        Right-multiplies gauge degrees of freedom. other can be a gauge field or 
        a color matrix.
        """
        if type(other) == GaugeField:
            if self._NDIMS == other._NDIMS:
                tmp = np.einsum('...ab,...bc->...ac', self._U, other._U)
            else:
                raise Exception('Not yet implemented') # TODO np.einsum with ... expects the same shape
        else:
            assert type(other) == np.ndarray and other.shape == (self._NC, self._NC)
            tmp = np.einsum('...ab,bc->...ac', self._U, other)
        return GaugeField(self._LL, U = tmp, tol = self._tol, d = self._D, n_dims = self._NDIMS, Nc = self._NC)
    
    def __lmul__(self, other):
        """
        Left-multiplies gauge degrees of freedom. other can be a gauge field or 
        a color matrix.
        """
        if type(other) == GaugeField:
            if self._NDIMS == other._NDIMS:
                tmp = np.einsum('...ab,...bc->...ac', other._U, self._U)
            else:
                raise Exception('Not yet implemented') # TODO np.einsum with ... expects the same shape
        else:
            assert type(other) == np.ndarray and other.shape == (self._NC, self._NC)
            tmp = np.einsum('ab,...bc->...ac', other, self._U)
        return GaugeField(self._LL, U = tmp, tol = self._tol, d = self._D, n_dims = self._NDIMS, Nc = self._NC)
    
    def __add__(self, other):
        if type(other) == GaugeField:
            tmp = self._U + other._U
        else:
            assert type(other) == np.ndarray and other.shape == (self._NC, self._NC)
            tmp = copy(self._U)
            tmp[..., :, :] = tmp[..., :, :] + other
        return GaugeField(self._LL, U = tmp, tol = self._tol, d = self._D, n_dims = self._NDIMS, Nc = self._NC)
    
    def __eq__(self, other):
        if self._SHAPE != other._SHAPE:
            return False
        return np.allclose(self._U, other._U, self._tol)
    
    def real(self):
        return GaugeField(self._LL, U = np.real(self._U) + (1j) * np.zeros(self._SHAPE), \
            tol = self._tol, d = self._D, n_dims = self._NDIMS, Nc = self._NC)

    def imag(self):
        return GaugeField(self._LL, U = (1j) * np.imag(self._U), \
            tol = self._tol, d = self._D, n_dims = self._NDIMS, Nc = self._NC)

    def dagger(self):
        return GaugeField(self._LL, U = np.einsum('...ab->...ba', np.conjugate(self._U)), \
            tol = self._tol, d = self._D, n_dims = self._NDIMS, Nc = self._NC)

    def trace(self):
        return np.einsum('...aa->...', self._U)
    
    def shift(self, delta, mu):
        """
        Shifts a gauge field by delta units in the \hat{\mu} direction.

        Parameters
        ----------
        delta : int
            Number of lattice sites to shift by.
        mu : {0, 1, 2, 3}
            Direction to shift.
        """
        return GaugeField(self._LL, U = np.roll(self._U, delta, axis = self._NDIMS + mu), \
            tol = self._tol, d = self._D, n_dims = self._NDIMS, Nc = self._NC)

    def site_plaquette(self, n, mu, nu):
        """Evaluates the plaquette of the gauge field at site n. This returns a (Nc x Nc) matrix."""
        mu_hat = np.array([1 if rho == mu else 0 for rho in range(self._D)])
        nu_hat = np.array([1 if rho == nu else 0 for rho in range(self._D)])
        return self[(mu, *n)] @ self[(nu, *(n + mu_hat))] @ \
            dagger(self[(mu, *(n + nu_hat))]) @ dagger(self[(nu, *n)])

    def plaquette(self, mu, nu):
        """
        Returns the plaquette of the gauge field. Note this differs from 
        site_plaquette because it returns the entire plaquette field, rather 
        than the plaquette evaluated at a single site n.
        """
        assert self._NDIMS == 1, 'Not a vector gauge field.'
        Un_mu = GaugeField(self._LL, U = self._U[mu], \
            n_dims = 0, tol = self._tol, d = self._D, Nc = self._NC)
        Unpmu_nu = GaugeField(self._LL, self.shift(1, mu)._U[nu], \
            n_dims = 0, tol = self._tol, d = self._D, Nc = self._NC)
        Unpnu_mu = GaugeField(self._LL, self.shift(1, nu)._U[nu], \
            n_dims = 0, tol = self._tol, d = self._D, Nc = self._NC)
        Un_nu = GaugeField(self._LL, U = self._U[nu], \
            n_dims = 0, tol = self._tol, d = self._D, Nc = self._NC)
        tmp = Un_mu * Unpmu_nu
        return Un_mu * Unpmu_nu * Unpnu_mu.dagger() * Un_nu.dagger()
    
    def wilson_loop(U, mu, nu, Lmu, Lnu):
        """
        Returns the Lmu x Lnu site Wilson loop in the (mu, nu) plane. Note that 
        for Lmu = Lnu = 1, equals plaquette(self, mu, nu).
        """
        
        return

    def staple(self, n, mu):
        """
        Computes the staple of self in the \hat{\mu} direction.
        """
        A = np.zeros((self._NC, self._NC), dtype = GaugeField.DTYPE)
        mu_hat = np.array([1 if rho == mu else 0 for rho in range(self._D)])
        for nu in range(self._D):
            if nu == mu:
                continue
            nu_hat = np.array([1 if rho == nu else 0 for rho in range(self._D)])
            A += self[(nu, *(n + mu_hat))] @ dagger(self[(mu, *(n + nu_hat))]) @ dagger(self[(nu, *n)])
            A += dagger(self[(nu, *(n + mu_hat - nu_hat))]) @ dagger(self[(mu, *(n - nu_hat))]) @ self[(nu, *(n - nu_hat))]
        return A
    
    @staticmethod
    def zeros(LL, n_dims = 1, tol = 1e-5, d = 4, Nc = 3):
        return GaugeField(LL, n_dims = n_dims, tol = tol, d = d, Nc = Nc)
    
    @staticmethod
    def id(LL, n_dims = 1, tol = 1e-5, d = 4, Nc = 3):
        """
        Returns the identity color gauge field with the given parameters.
        """
        Uid = np.zeros((*([d]*n_dims), *LL, Nc, Nc), dtype = GaugeField.DTYPE)
        Uid[..., :, :] = np.eye(Nc, dtype = GaugeField.DTYPE)
        return GaugeField(LL, U = Uid, n_dims = n_dims, tol = tol, d = d, Nc = Nc)