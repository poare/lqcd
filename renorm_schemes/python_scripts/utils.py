###############################################################################
# Utility functions for renormalization.                                      #
###############################################################################

import numpy as np
import h5py
import os
import time
import itertools
import io
import random
import sys
import gvar as gv

# Dirac gamma matrices.
delta = np.identity(4, dtype = np.complex64)
gamma = np.zeros((4, 4, 4),dtype=complex)
gamma[0] = gamma[0] + np.array([[0,0,0,1j],[0,0,1j,0],[0,-1j,0,0],[-1j,0,0,0]])
gamma[1] = gamma[1] + np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]])
gamma[2] = gamma[2] + np.array([[0,0,1j,0],[0,0,0,-1j],[-1j,0,0,0],[0,1j,0,0]])
gamma[3] = gamma[3] + np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
gamma5 = np.dot(np.dot(np.dot(gamma[0], gamma[1]), gamma[2]), gamma[3])

def bootstrap(S, n_boot):
    """
    Bootstraps an input tensor. Generates each bootstrap sample by averaging 
    over ncfgs data points, where ncfgs is the number of configurations in 
    the original dataset. 

    Parameters
    ----------
    S : np.array [ncfgs, 3, 4, 3, 4]
        Input tensor to bootstrap. ncfgs is the number of configurations.
    n_boot : int
        Number of bootstraps to resample with.

    Returns
    -------
    np.array [n_boot, 3, 4, 3, 4]
        Bootstrapped tensor.
    """
    S_boot = np.zeros((n_boot, 3, 4, 3, 4), dtype = np.complex64)
    raise NotImplementedError('bootstrap needs to be implemented')
    return S_boot

def quark_renorm(props, p, n_boot):
    """
    Computes the quark field renormalization Zq at momentum p,
        Zq = i Tr[\tilde{p}^mu gamma^mu S^{-1}(p)] / (12 \tilde{p}^2),
    where \tilde{p} = (2/a) \sin(ap/2) is the lattice momentum.

    Parameters
    ----------
    props : np.array [Nb, 3, 4, 3, 4]
        Propagator to compute Zq with.
    q : np.array
        Four-momentum to compute Zq at.
    n_boot : int
        Number of bootstrap samples.

    Returns
    -------
    np.array [n_boot]
        Quark field renormalization, computed at each bootstrap.
    """
    Zq = np.zeros((n_boot), dtype = np.complex64)
    raise NotImplementedError('quark_renorm needs to be implemneted.')
    return