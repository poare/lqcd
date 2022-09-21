#####################################################################
# Code for 8.871 Problem Set 1. Note that this uses a variety of    #
# code written for my research which is stored in the               #
# '/Users/theoares/lqcd/utilities' folder. I will include relevant  #
# code snippets in my writeup if they are imported from another     #
# script that I wrote.                                              #
#####################################################################

import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from pytools import *
import plottools as pt

#####################################################################
########################### PROBLEM 2.2.1 ###########################
#####################################################################

potential = lambda x : (x**2) / 2
def action(x, V, m = 1., a = 0.5):
    """
    Computes the action functional S[x] for a 1D path x(t), given a 
    potential V.

    Parameters
    ----------
    x : np.array[N]
        Discretized path of length N.
    V : function
        Potential function for the action.
    m : float (default = 1.0)
        Mass of the particle.
    a : float (default = 0.5)
        Lattice spacing to use.
    
    Returns
    -------
    float
        Value of S[x].
    """
    return np.sum(
        (m / (2*a)) * (np.roll(x, -1) - x)**2 + a * V(x)
    )

def metropolis(S, N, N_cfgs, N_corr = 20, eps = 1.4):
    """
    Implements the 1D Metropolis algorithm to sample the distribution 
    exp(-S[x]).

    Parameters
    ----------
    S : function
        Action to use.
    N : int
        Number of sites for discretization.
    N_cfgs : int
        Number of configurations to generate.
    N_corr : int (default = 20)
        Correlation length between different generated configurations. 
    eps : float (default = 1.4)
        Average fluctuation size for Metropolis algorithm.
    Returns
    -------
    np.array[N_cfgs, N]
        N_cfgs generated configurations.
    float 
        Accept/reject ratio of Metropolis algorithm (after thermalization).
    """
    return


#####################################################################
########################### PROBLEM 2.2.1 ###########################
#####################################################################




#####################################################################
########################### PROBLEM 2.2.1 ###########################
#####################################################################