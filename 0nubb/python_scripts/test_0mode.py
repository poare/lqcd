import numpy as np
from scipy.optimize import root
import h5py
import os
from analysis import *

################################## PARAMETERS #################################
cfgbase = 'cl3_16_48_b6p1_m0p2450'
# job_num = 28337
job_num = 28357
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
for idx, cfg in enumerate(cfgs):
    cfgs[idx] = data_dir + '/' + cfgs[idx]
n_cfgs = len(cfgs)

Zq1 = np.zeros((len(q_list), n_boot), dtype = np.complex64)
ZV1, ZA1 = np.zeros((len(q_list), n_boot), dtype = np.complex64), np.zeros((len(q_list), n_boot), dtype = np.complex64)
for q_idx, q in enumerate(q_list):
    print('Momentum index: ' + str(q_idx))
    # q_lat = to_lattice_momentum(q + bvec)
    q_lat = np.sin(to_linear_momentum(q + bvec))            # choice of lattice momentum will affect how artifacts look, but numerics should look roughly the same
    # q_lat = to_linear_momentum(q + bvec)

    k1, k2, props_k1, props_k2, props_q, GV, GA, GO = readfiles(cfgs, q, False)
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
    ZV1[q_idx] = 12 * Zq[q_idx] * square(q_lat) / np.einsum('zaiaj,ji->z', qDotV, qlat_slash)
    ZA1[q_idx] = 12 * Zq[q_idx] * square(q_lat) / np.einsum('zaiaj,jk,ki->z', qDotA, gamma5, qlat_slash)
    # print([12 * Zq[q_idx] / np.einsum('zaiaj,ji->z', GammaV[mu], gamma[mu]) for mu in range(4)])    # this is what I had in the RI'-MOM code; it's the same order of mag


################################## SAVE DATA ##################################
out_file1 = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/currents' + str(job_num) + '/Z_0nubb.h5'
f1 = h5py.File(out_file1, 'w')
f1['momenta'] = q_list
f1['ZV'] = ZV
f1['ZA'] = ZA
f1['Zq'] = Zq
f1['cfgnum'] = n_cfgs
f1.close()
print('Onubb output saved at: ' + out_file1)

out_file2 = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/currents' + str(job_num) + '/Z_npr_momfrac.h5'
f2 = h5py.File(out_file2, 'w')
f2['momenta'] = q_list
f2['ZV'] = ZV
f2['ZA'] = ZA
f2['Zq'] = Zq
f2['cfgnum'] = n_cfgs
f2.close()
print('npr_momfrac output saved at: ' + out_file2)
