import numpy as np
from scipy.optimize import root
import h5py
import os
from scipy.special import zeta
import time
import re
import itertools

# from analysis import *
# Add parent directory to path so that we can import from it
import sys
sys.path.append('/Users/theoares/lqcd/0nubb/python_scripts')
from utils import *

# read in npr_momfrac format. Used for testing RI/sMOM RCs on RI'-MOM data.
def readfiles_momfrac(cfgs, q):
    props_k1 = np.zeros((len(cfgs), Nc, Nd, Nc, Nd), dtype = np.complex64)
    props_k2 = np.zeros((len(cfgs), Nc, Nd, Nc, Nd), dtype = np.complex64)
    props_q = np.zeros((len(cfgs), Nc, Nd, Nc, Nd), dtype = np.complex64)
    GV = np.zeros((d, len(cfgs), Nc, Nd, Nc, Nd), dtype = np.complex64)
    GA = np.zeros((d, len(cfgs), Nc, Nd, Nc, Nd), dtype = np.complex64)
    GO = np.zeros((16, len(cfgs), Nc, Nd, Nc, Nd, Nc, Nd, Nc, Nd), dtype = np.complex64)

    for idx, file in enumerate(cfgs):
        f = h5py.File(file, 'r')
        qstr = klist_to_string(q, 'p')
        props_q[idx] = np.einsum('ijab->aibj', f['prop/' + qstr][()]) / vol
        for mu in range(d):
            GV[mu, idx] = np.einsum('ijab->aibj', f['GV' + str(mu + 1) + '/' + qstr][()]) / vol
            GA[mu, idx] = np.einsum('ijab->aibj', f['GA' + str(mu + 1) + '/' + qstr][()]) / vol
    return props_q, GV, GA

# Reads in propagator output from free_field_props.qlua
def read_propagators(file, lat, q):
    props_k1 = np.zeros((lat.L, lat.L, lat.L, lat.T, 3, 4, 3, 4), dtype = np.complex64)
    props_k2 = np.zeros((3, 4, 3, 4), dtype = np.complex64)
    props_q = np.zeros((3, 4, 3, 4), dtype = np.complex64)

    f = h5py.File(file, 'r')
    qstr = lat.klist_to_string(q, 'q')
    props_k1 = np.einsum('...ijab->...aibj', f['prop_k1/' + qstr][()])
    props_k2 = np.einsum('...ijab->...aibj', f['prop_k2/' + qstr][()])
    props_q = np.einsum('...ijab->...aibj', f['prop_q/' + qstr][()])
    return props_k1, props_k2, props_q
