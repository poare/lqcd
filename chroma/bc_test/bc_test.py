import numpy as np
from scipy.optimize import root
import h5py
import os
# from test_utils import *
import time
import itertools

import sys
home = '/Users/theoares/'
# home = '/Users/poare/'
sys.path.append(home + 'lqcd/0nubb/python_scripts')
from utils import *

################################## PARAMETERS #################################
# L, T = 4, 8
L, T = 16, 48
# L, T = 8, 8
home = '/Users/theoares/'
# home = '/Users/poare/'
data_dir = home + 'lqcd/chroma/bc_test/output'
L = Lattice(L, T)

k1_list = [[-1, 0, 1, 0]]
k2_list = [[0, 1, 1, 0]]

k1_list = np.array(k1_list)
k2_list = np.array(k2_list)
q_list = k2_list - k1_list
print('Number of total momenta: ' + str(len(q_list)))

############################### PERFORM ANALYSIS ##############################
cfgs = ['chroma_out.h5']
# cfgs = ['chroma_out_isoclover.h5']
# cfgs = ['qlua_out_38113.h5']
# cfgs = ['qlua_out_38108.h5']
for idx, cfg in enumerate(cfgs):
    cfgs[idx] = data_dir + '/' + cfgs[idx]
n_cfgs = len(cfgs)

scheme = 'gamma'                # scheme == 'gamma' or 'qslash'
F = getF(L, scheme)                # tree level projections

start = time.time()
Zq = np.zeros((len(q_list), n_boot), dtype = np.complex64)
ZV, ZA = np.zeros((len(q_list), n_boot), dtype = np.complex64), np.zeros((len(q_list), n_boot), dtype = np.complex64)
Z = np.zeros((5, 5, len(q_list), n_boot), dtype = np.complex64)
for q_idx, q in enumerate(q_list):
    print('Momentum index: ' + str(q_idx))
    # k1, k2, props_k1, props_k2, props_q, GV, GA, GO = readfiles(cfgs, q, op_renorm = True)
    k1, k2, props_k1, props_k2, props_q, GV, GA, GO = readfiles(cfgs, q, op_renorm = False)
    props_k1_b, props_k2_b, props_q_b = bootstrap(props_k1), bootstrap(props_k2), bootstrap(props_q)
    print(props_q[0])
    GV_boot, GA_boot, GO_boot = np.array([bootstrap(GV[mu]) for mu in range(4)]), np.array([bootstrap(GA[mu]) for mu in range(4)]), \
        np.array([bootstrap(GO[n]) for n in range(16)])
    props_k1_inv, props_k2_inv, props_q_inv = invert_props(props_k1_b), invert_props(props_k2_b), invert_props(props_q_b)
    print(props_q_inv[0])

    # q = -q
    # q_lat = np.sin(L.to_linear_momentum(q + bvec))            # choice of lattice momentum will affect how artifacts look, but numerics should look roughly the same
    q = -q                                                  # for chroma
    q_lat = np.sin(L.to_linear_momentum(q))

    Zq[q_idx] = quark_renorm(props_q_inv, q_lat)
    GammaV, GammaA = np.zeros(GV_boot.shape, dtype = np.complex64), np.zeros(GA_boot.shape, dtype = np.complex64)
    qDotV, qDotA = np.zeros(GV_boot.shape[1:]), np.zeros(GA_boot.shape[1:])
    qlat_slash = slash(q_lat)
    print('Computing axial and vector renormalizations.')
    for mu in range(4):
        # GammaV[mu], GammaA[mu] = amputate_threepoint(props_k1_inv, props_k2_inv, GV_boot[mu]), amputate_threepoint(props_k1_inv, props_k2_inv, GA_boot[mu])
        GammaV[mu], GammaA[mu] = amputate_threepoint(props_k2_inv, props_k1_inv, GV_boot[mu]), amputate_threepoint(props_k2_inv, props_k1_inv, GA_boot[mu])
        qDotV, qDotA = qDotV + q_lat[mu] * GammaV[mu], qDotA + q_lat[mu] * GammaA[mu]
    ZV[q_idx] = 12 * Zq[q_idx] * square(q_lat) / np.einsum('zaiaj,ji->z', qDotV, qlat_slash)
    ZA[q_idx] = 12 * Zq[q_idx] * square(q_lat) / np.einsum('zaiaj,jk,ki->z', qDotA, gamma5, qlat_slash)

    print('Zq ~ ' + str(Zq[q_idx, 0]))
    print('ZV ~ ' + str(ZV[q_idx, 0]))
    print('ZA ~ ' + str(ZA[q_idx, 0]))

    # # Amputate and get scalar / vector / pseudoscalar / axial / tensor green's functions from G^n
    # # print(GO[3, 0])
    # GammaO = np.zeros(GO_boot.shape, dtype = np.complex64)
    # for n in range(16):
    #     print('Amputating Green\'s function for n = ' + str(n) + '.')
    #     GammaO[n] = amputate_fourpoint(props_k2_inv, props_k1_inv, GO_boot[n])
    # SS = GammaO[0]
    # PP = GammaO[15]
    # VV = GammaO[1] + GammaO[2] + GammaO[4] + GammaO[8]
    # # AA = GammaO[14] - GammaO[13] + GammaO[11] - GammaO[7]
    # # TT = GammaO[3] + GammaO[5] + GammaO[9] + GammaO[6] + GammaO[10] + GammaO[12]
    # AA = GammaO[14] + GammaO[13] + GammaO[11] + GammaO[7]
    # TT = GammaO[3] + GammaO[5] + GammaO[9] + GammaO[6] + GammaO[10] + GammaO[12]
    #
    # # Get positive parity operator projections
    # print('Projecting onto tree level vertex.')
    # Gamma = [VV + AA, VV - AA, SS - PP, SS + PP, TT]
    # P = projectors(scheme, L.to_linear_momentum(q), L.to_linear_momentum(k1), L.to_linear_momentum(k2))
    # Lambda = np.einsum('nbjaidlck,mzaibjckdl->zmn', P, Gamma)    # Lambda is n_boot x 5 x 5
    # print('Lambda ~ ' + str(Lambda[0, :, :]))       # projected 4 pt function
    # # Lambda_inv = np.array([np.linalg.inv(Lambda[b, :, :]) for b in range(n_boot)])
    # # Z[:, :, q_idx, :] = (Zq[q_idx] ** 2) * np.einsum('ik,zkj->ijz', F, Lambda_inv)
    # for b in range(n_boot):
    #     Lambda_inv = np.linalg.inv(Lambda[b, :, :])
    #     Z[:, :, q_idx, b] = (Zq[q_idx, b] ** 2) * np.einsum('ik,kj->ij', F, Lambda_inv)
    #
    # print('Z_ij ~ ' + str(Z[:, :, q_idx, 0]))

    # Time per iteration
    print('Elapsed time: ' + str(time.time() - start))
