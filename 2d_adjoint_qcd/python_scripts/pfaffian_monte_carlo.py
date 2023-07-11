################################################################################
# Monte Carlo implementation to see what the Pfaffian of the Dirac operator    #
# D will look like. Note that configurations will be sampled randomly, and     #
# there is a chance that exceptional configurations that make the Pfaffian     #
# appear ill-determined do not really come up in the path integral with any    # 
# non-zero measure.                                                            #
################################################################################
# Author: Patrick Oare                                                         #
################################################################################

n_boot = 100
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import bsr_matrix
from scipy.linalg import block_diag
import h5py
import os
import itertools
import pandas as pd
import gvar as gv
import lsqfit

import sys
sys.path.append('/Users/theoares/lqcd/utilities')
from constants import *
from fittools import *
from formattools import *
import plottools as pt

style = styles['prd_twocol']
pt.set_font()

import rhmc

Nc = 3
eps = 0.5
kappa = 0.2
L, T = 4, 4
bcs = (1, -1)
n_samps = 500

gens = rhmc.get_generators(Nc)
Lat = rhmc.Lattice(L, T)

for i in range(n_samps):
    # if i % 100 == 0:
    print(i)
    U = rhmc.gen_random_fund_field(Nc, eps, lat = Lat)
    V = rhmc.construct_adjoint_links(U, gens, lat = Lat)
    print(V)
    D = rhmc.dirac_op_sparse(kappa, V, bcs = bcs, lat = Lat)
    print(D.toarray()[0])
    Q = rhmc.hermitize_dirac(D)
    # print(Q.toarray()[0])
    # TODO: why isn't Q antisymmetric?
    print((Q + Q.transpose()).toarray()[0])
    pf = rhmc.pfaffian(Q)
