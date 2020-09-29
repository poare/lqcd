import numpy as np
from scipy.optimize import root
import h5py
import os
from analysis import *

################################## PARAMETERS #################################
cfgbase = 'cl3_16_48_b6p1_m0p2450'
job_num = 28230
data_dir = '/Users/theoares/Dropbox (MIT)/research/0nubb/meas/' + cfgbase + '_' + str(job_num)

L = 16
T = 48
set_dimensions(L, T)

k1_list = []
k2_list = []
for n in range(-6, 7):
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
n_cfgs = len(cfgs)

Zq = np.zeros((len(q_list), n_boot), dtype = np.complex64)
ZV, ZA = np.zeros((len(q_list), n_boot), dtype = np.complex64), np.zeros((len(q_list), n_boot), dtype = np.complex64)
for q_idx, q in enumerate(q_list):
    q_lat = to_lattice_momentum(q)
    k1, k2, props_k1, props_k2, props_q, GV, GA, GO = analysis.readfiles(cfgs, q, False)
    props_k1_inv, props_k2_inv, props_q_inv = invert_props(props_k1), invert_props(props_k2), invert_props(props_q)
    Zq[q_idx] = quark_renorm(props_q_inv, q)
    GammaV, GammaA = np.zeros(GV.shape, dtype = np.complex64), np.zeros(GA.shape, dtype = np.complex64)
    GammaVslash, GammaAslash = np.zeros(GV.shape[1:]), np.zeros(GA.shape[1:])
    qlat_slash = slash(q_lat)
    for mu in range(4):
        GammaV[mu], GammaA[mu] = amputate(props_k1_inv, props_k2_inv, GV[mu]), amputate(props_k1_inv, props_k2_inv, GA[mu])
        GammaVslash, GammaAslash = GammaVslash + q_lat[mu] * GammaV[mu], GammaAslash + q_lat[mu] * GammaA[mu]
    ZV[q_idx] = 12 * Zq[q_idx] * square(q_lat) * np.einsum('zaiaj,ji->z', GammaVslash, qlat_slash)
    ZA[q_idx] = 12 * Zq[q_idx] * square(q_lat) * np.einsum('zaiaj,jk,ki->z', GammaVslash, gamma5, qlat_slash)


################################## SAVE DATA ##################################
out_file = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/currents' + str(job_num) + '/Z.h5'
f = h5py.File(out_file, 'w')
f['momenta'] = mom_list
f['ZV'] = ZV
f['ZA'] = ZA
f['Zq'] = Zq
f['cfgnum'] = n_cfgs
f.close()
print('Output saved at: ' + out_file)
