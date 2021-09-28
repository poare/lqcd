import numpy as np
from scipy.optimize import root
import h5py
import os
from utils import *

################################## PARAMETERS #################################
base = '/Users/theoares/Dropbox (MIT)/research/0nubb/meas/'
ens = '24I/ml0p01'
# ens = '24I/ml0p005'
# ens = '32I/ml0p006'
# ens = '32I/ml0p004'
data_dir = base + ens + '/hdf5'

l = 24
# l = 32
t = 64
L = Lattice(l, t)

k1_list = []
k2_list = []
for n in range(2, 10):
    k1_list.append([-n, 0, n, 0])
    k2_list.append([0, n, n, 0])
k1_list = np.array(k1_list)
k2_list = np.array(k2_list)
q_list = k2_list - k1_list
print('Number of total momenta: ' + str(len(q_list)))

############################### PERFORM ANALYSIS ##############################
cfgs = []
for (dirpath, dirnames, file) in os.walk(data_dir):
    cfgs.extend(file)
for idx, cfg in enumerate(cfgs):
    cfgs[idx] = data_dir + '/' + cfgs[idx]
n_cfgs = len(cfgs)

GammaVList = np.zeros((len(q_list), 4, n_boot, 3, 4, 3, 4), dtype = np.complex64)
GammaAList = np.zeros((len(q_list), 4, n_boot, 3, 4, 3, 4), dtype = np.complex64)
qDotVList = np.zeros((len(q_list), n_boot, 3, 4, 3, 4), dtype = np.complex64)
qDotAList = np.zeros((len(q_list), n_boot, 3, 4, 3, 4), dtype = np.complex64)
start = time.time()
Zq = np.zeros((len(q_list), n_boot), dtype = np.complex64)
ZV, ZA = np.zeros((len(q_list), n_boot), dtype = np.complex64), np.zeros((len(q_list), n_boot), dtype = np.complex64)
for q_idx, q in enumerate(q_list):
    print('Momentum index: ' + str(q_idx))
    k1, k2, props_k1, props_k2, props_q, GV, GA, GO = readfiles(cfgs, q, False)
    q = -q
    q_lat = np.sin(L.to_linear_momentum(q))
    props_k1_b, props_k2_b, props_q_b = bootstrap(props_k1), bootstrap(props_k2), bootstrap(props_q)
    GV_boot, GA_boot = np.array([bootstrap(GV[mu]) for mu in range(4)]), np.array([bootstrap(GA[mu]) for mu in range(4)])
    props_k1_inv, props_k2_inv, props_q_inv = invert_props(props_k1_b), invert_props(props_k2_b), invert_props(props_q_b)
    # props_k1_inv, props_k2_inv, props_q_inv = invert_props(antiprop(props_k1_b)), invert_props(props_k2_b), invert_props(props_q_b)    # try antipropagator
    # props_k1_inv, props_k2_inv, props_q_inv = invert_props(props_k1_b), invert_props(antiprop(props_k2_b)), invert_props(props_q_b)
    Zq[q_idx] = quark_renorm(props_q_inv, q_lat)
    GammaV, GammaA = np.zeros(GV_boot.shape, dtype = np.complex64), np.zeros(GA_boot.shape, dtype = np.complex64)
    qDotV, qDotA = np.zeros(GV_boot.shape[1:]), np.zeros(GA_boot.shape[1:])
    qlat_slash = slash(q_lat)
    for mu in range(4):
        # GammaV[mu], GammaA[mu] = amputate_threepoint(props_k1_inv, props_k2_inv, GV_boot[mu]), amputate_threepoint(props_k1_inv, props_k2_inv, GA_boot[mu])
        GammaV[mu], GammaA[mu] = amputate_threepoint(props_k2_inv, props_k1_inv, GV_boot[mu]), amputate_threepoint(props_k2_inv, props_k1_inv, GA_boot[mu])
        qDotV, qDotA = qDotV + q_lat[mu] * GammaV[mu], qDotA + q_lat[mu] * GammaA[mu]
        GammaVList[q_idx, mu] = GammaV[mu]
        GammaAList[q_idx, mu] = GammaA[mu]
    qDotVList[q_idx] = qDotV
    qDotAList[q_idx] = qDotA
    ZV[q_idx] = 12 * Zq[q_idx] * square(q_lat) / np.einsum('zaiaj,ji->z', qDotV, qlat_slash)
    ZA[q_idx] = 12 * Zq[q_idx] * square(q_lat) / np.einsum('zaiaj,jk,ki->z', qDotA, gamma5, qlat_slash)

    # Time per iteration
    print('Elapsed time: ' + str(time.time() - start))


################################## SAVE DATA ##################################
out_file = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/' + ens + '/AVcurrents.h5'
f = h5py.File(out_file, 'w')
f['momenta'] = q_list
f['GammaV'] = GammaVList
f['GammaA'] = GammaAList
f['qDotV'] = qDotVList
f['qDotA'] = qDotAList
f['ZV'] = ZV
f['ZA'] = ZA
f['Zq'] = Zq
f['cfgnum'] = n_cfgs
f.close()
print('Output saved at: ' + out_file)
