import numpy as np
from scipy.optimize import root
import h5py
import os
import test_utils
from utils import *

# Just do the regular analysis but don't bootstrap, want to see what the errors are (we'll bootstrap afterwards)
################################## PARAMETERS #################################
# cfgbase = 'cl3_16_48_b6p1_m0p2450'
# job_num = 29552
# data_dir = '/Users/theoares/Dropbox (MIT)/research/0nubb/meas/' + cfgbase + '_' + str(job_num)
base = '/Users/theoares/Dropbox (MIT)/research/0nubb/meas/'
ens = '24I/ml0p01'
data_dir = base + ens + '/hdf5'

l = 24
t = 64
# An.set_dimensions(L, T)
L = Lattice(l, t)

k1_list = []
k2_list = []
# for n in range(-6, 7):
for n in range(1, 2):
    k1_list.append([-n, 0, n, 0])
    k2_list.append([0, n, n, 0])
k1_list = np.array(k1_list)
k2_list = np.array(k2_list)
q_list = k2_list - k1_list
print('Number of total momenta: ' + str(len(q_list)))

############################### PERFORM ANALYSIS ##############################
cfgs = []
for (dirpath, dirnames, file) in os.walk(data_dir):
    print(file)
    cfgs.extend(file)
for idx, cfg in enumerate(cfgs):
    cfgs[idx] = data_dir + '/' + cfgs[idx]
print(cfgs)
n_cfgs = len(cfgs)
print('Running on ' + str(n_cfgs) + ' configs.')

scheme = 'gamma'                # scheme == 'gamma' or 'qslash'
F = getF(L, scheme)                # tree level projections

start = time.time()
Zq = np.zeros((len(q_list), n_cfgs), dtype = np.complex64)
ZV, ZA = np.zeros((len(q_list), n_cfgs), dtype = np.complex64), np.zeros((len(q_list), n_cfgs), dtype = np.complex64)
Z = np.zeros((5, 5, len(q_list), n_cfgs), dtype = np.complex64)
for q_idx, q in enumerate(q_list):
    print('Momentum index: ' + str(q_idx))
    k1, k2, props_k1, props_k2, props_q, GV, GA, GO = readfiles(cfgs, q, True)
    # q_lat = np.sin(L.to_linear_momentum(q + bvec))
    q = -q
    q_lat = np.sin(L.to_linear_momentum(q))

    props_k1_b, props_k2_b, props_q_b = props_k1, props_k2, props_q
    GV_boot, GA_boot, GO_boot = GV, GA, GO
    # props_k1_b, props_k2_b, props_q_b = bootstrap(props_k1), bootstrap(props_k2), bootstrap(props_q)
    # GV_boot, GA_boot, GO_boot = np.array([bootstrap(GV[mu]) for mu in range(4)]), np.array([bootstrap(GA[mu]) for mu in range(4)]), np.array([bootstrap(GO[n]) for n in range(16)])
    props_k1_inv, props_k2_inv, props_q_inv = invert_props(props_k1_b), invert_props(props_k2_b), invert_props(props_q_b)
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

    # Amputate and get scalar / vector / pseudoscalar / axial / tensor green's functions from G^n
    GammaO = np.zeros(GO_boot.shape, dtype = np.complex64)
    for n in range(16):
        print('Amputating Green\'s function for n = ' + str(n) + '.')
        GammaO[n] = amputate_fourpoint(props_k2_inv, props_k1_inv, GO_boot[n])
    SS = GammaO[0]
    PP = GammaO[15]
    VV = GammaO[1] + GammaO[2] + GammaO[4] + GammaO[8]
    AA = GammaO[14] + GammaO[13] + GammaO[11] + GammaO[7]
    TT = GammaO[3] + GammaO[5] + GammaO[9] + GammaO[6] + GammaO[10] + GammaO[12]

    # Get positive parity operator projections
    print('Projecting onto tree level vertex.')
    Gamma = [VV + AA, VV - AA, SS - PP, SS + PP, TT]
    P = projectors(scheme, L.to_linear_momentum(q), L.to_linear_momentum(k1), L.to_linear_momentum(k2))
    Lambda = np.einsum('nbjaidlck,mzaibjckdl->zmn', P, Gamma)    # Lambda is n_boot x 5 x 5
    print('Lambda ~ ' + str(Lambda[0, :, :]))       # projected 4 pt function
    # Lambda_inv = np.array([np.linalg.inv(Lambda[b, :, :]) for b in range(n_boot)])
    # Z[:, :, q_idx, :] = (Zq[q_idx] ** 2) * np.einsum('ik,zkj->ijz', F, Lambda_inv)
    for b in range(n_cfgs):
        Lambda_inv = np.linalg.inv(Lambda[b, :, :])
        Z[:, :, q_idx, b] = (Zq[q_idx, b] ** 2) * np.einsum('ik,kj->ij', F, Lambda_inv)

    print('Z_ij ~ ' + str(Z[:, :, q_idx, 0]))
    # print('Z_ij ~ ' + str(np.mean(Z[:, :, q_idx])) + ' \pm ' + str(np.std(Z[:, :, q_idx], ddof = 1)))

    # Time per iteration
    print('Elapsed time: ' + str(time.time() - start))

################################## SAVE DATA ##################################
# out_file = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/job' + str(job_num) + '/Z_' + scheme + '.h5'
out_file = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/' + ens + '/Z_' + scheme + '_error.h5'
f = h5py.File(out_file, 'w')
f['momenta'] = q_list
f['ZV'] = ZV
f['ZA'] = ZA
f['Zq'] = Zq
for i, j in itertools.product(range(5), repeat = 2):
    f['Z' + str(i + 1) + str(j + 1)] = Z[i, j]
f['cfgnum'] = n_cfgs
f.close()
print('Output saved at: ' + out_file)
