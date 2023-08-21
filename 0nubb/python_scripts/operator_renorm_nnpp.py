import numpy as np
from scipy.optimize import root
import h5py
import os
from utils import *
base = '/Users/theoares/Dropbox (MIT)/research/0nubb/meas/'

################################## PARAMETERS #################################
# ens = 'cl3_32_48_b6p1_m0p2450_99999'
ens = 'cl3_32_48_b6p1_m0p2450_113400'
# ens = 'cl3_32_48_b6p1_m0p2450_114105'
data_dir = base + 'nnpp/' + ens
l = 32
t = 48

L = Lattice(l, t)

k1_list = []
k2_list = []
for n in range(1, 19):
    k1_list.append([-n, 0, n, 0])
    k2_list.append([0, n, n, 0])
k1_list = np.array(k1_list)
k2_list = np.array(k2_list)
q_list = k2_list - k1_list
print('Number of total momenta: ' + str(len(q_list)))

# store all the momenta that we already know
q_known = [[k, k, 0, 0] for k in range(1, 14)]
f_known = h5py.File('/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/nnpp/cl3_32_48_b6p1_m0p2450_99999/Z_gamma.h5', 'r')
# f_known = h5py.File('/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/nnpp/cl3_32_48_b6p1_m0p2450_113400/Z_gamma.h5', 'r')
Zq_known = f_known['Zq'][()]
ZV_known = f_known['ZV'][()]
ZA_known = f_known['ZA'][()]
Lambda_known = f_known['Lambda'][()]
Z_known = np.zeros((5, 5, Zq_known.shape[0], n_boot), dtype = np.complex64)
for i, j in itertools.product(range(5), repeat = 2):
    Z_known[i, j] = f_known['Z' + str(i + 1) + str(j + 1)][()]
f_known.close()

############################### PERFORM ANALYSIS ##############################
cfgs = []
for (dirpath, dirnames, file) in os.walk(data_dir):
    cfgs.extend(file)
for idx, cfg in enumerate(cfgs):
    cfgs[idx] = data_dir + '/' + cfgs[idx]
n_cfgs = len(cfgs)
print('Reading ' + str(n_cfgs) + ' configs.')

scheme = 'gamma'                # scheme == 'gamma' or 'qslash'
F = getF(L, scheme)                # tree level projections

start = time.time()
Zq = np.zeros((len(q_list), n_boot), dtype = np.complex64)
ZV, ZA = np.zeros((len(q_list), n_boot), dtype = np.complex64), np.zeros((len(q_list), n_boot), dtype = np.complex64)
Z = np.zeros((5, 5, len(q_list), n_boot), dtype = np.complex64)
Lambda_list = np.zeros((5, 5, len(q_list), n_boot), dtype = np.complex64)
for q_idx, q in enumerate(q_list):
    print('Momentum index: ' + str(q_idx))
    print('Momentum is: ' + str(q))

    if q.tolist() in q_known:
        print('Momentum q = ' + str(q) + ' already computed. Saving arrays.')
        Zq[q_idx, :] = Zq_known[q_idx, :]
        ZV[q_idx, :] = ZV_known[q_idx, :]
        ZA[q_idx, :] = ZA_known[q_idx, :]
        Lambda_list[:, :, q_idx, :] = Lambda_known[:, :, q_idx, :]
        Z[:, :, q_idx, :] = Z_known[:, :, q_idx, :]
        continue

    k1, k2, props_k1, props_k2, props_q, GV, GA, GO = readfiles(cfgs, q, True, chroma = False)
    q_lat = np.sin(L.to_linear_momentum(q + bvec))          # for qlua

    props_k1_b, props_k2_b, props_q_b = bootstrap(props_k1), bootstrap(props_k2), bootstrap(props_q)
    GV_boot, GA_boot, GO_boot = np.array([bootstrap(GV[mu]) for mu in range(4)]), np.array([bootstrap(GA[mu]) for mu in range(4)]), np.array([bootstrap(GO[n]) for n in range(16)])
    props_k1_inv, props_k2_inv, props_q_inv = invert_props(props_k1_b), invert_props(props_k2_b), invert_props(props_q_b)
    # Zq[q_idx] = quark_renorm(props_q_inv, q_lat)
    Zq_qslash = quark_renorm(props_q_inv, q_lat)
    GammaV, GammaA = np.zeros(GV_boot.shape, dtype = np.complex64), np.zeros(GA_boot.shape, dtype = np.complex64)
    qDotV, qDotA = np.zeros(GV_boot.shape[1:]), np.zeros(GA_boot.shape[1:])
    qlat_slash = slash(q_lat)
    print('Computing axial and vector renormalizations.')
    for mu in range(4):
        GammaV[mu], GammaA[mu] = amputate_threepoint(props_k2_inv, props_k1_inv, GV_boot[mu]), amputate_threepoint(props_k2_inv, props_k1_inv, GA_boot[mu])
        qDotV, qDotA = qDotV + q_lat[mu] * GammaV[mu], qDotA + q_lat[mu] * GammaA[mu]
    ZV[q_idx] = 12 * Zq_qslash * square(q_lat) / np.einsum('zaiaj,ji->z', qDotV, qlat_slash)
    ZA[q_idx] = 12 * Zq_qslash * square(q_lat) / np.einsum('zaiaj,jk,ki->z', qDotA, gamma5, qlat_slash)
    Zq[q_idx] = np.einsum('mij,mzajai->z', gamma, GammaV) / 48.

    print('Zq ~ ' + str(np.mean(Zq[q_idx])) + ' \pm ' + str(np.std(Zq[q_idx], ddof = 1)))
    print('ZV ~ ' + str(np.mean(ZV[q_idx])) + ' \pm ' + str(np.std(ZV[q_idx], ddof = 1)))
    print('ZA ~ ' + str(np.mean(ZA[q_idx])) + ' \pm ' + str(np.std(ZA[q_idx], ddof = 1)))

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
    for b in range(n_boot):
        Lambda_list[:, :, q_idx, b] = Lambda[b]         # save Lambda for chiral extrapolation
        Lambda_inv = np.linalg.inv(Lambda[b, :, :])
        Z[:, :, q_idx, b] = (Zq[q_idx, b] ** 2) * np.einsum('ik,kj->ij', F, Lambda_inv)

    print('Z_ij ~ ' + str(Z[:, :, q_idx, 0]))

    # Time per iteration
    print('Elapsed time: ' + str(time.time() - start))

################################## SAVE DATA ##################################
out_file = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/nnpp/' + ens + '/Z_' + scheme + '_0_19.h5'    # nnpp output
f = h5py.File(out_file, 'w')
f['momenta'] = q_list
f['ZV'] = ZV
f['ZA'] = ZA
f['Zq'] = Zq                                            # Zq / ZV
f['Lambda'] = Lambda_list
for i, j in itertools.product(range(5), repeat = 2):
    f['Z' + str(i + 1) + str(j + 1)] = Z[i, j]          # Z / ZV^2
f['cfgnum'] = n_cfgs
f.close()
print('Output saved at: ' + out_file)
