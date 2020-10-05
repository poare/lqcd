import numpy as np
from scipy.optimize import root
import h5py
import os
from analysis import *

################################## PARAMETERS #################################
cfgbase = 'cl3_16_48_b6p1_m0p2450'
job_num = 28509
data_dir1 = '/Users/theoares/Dropbox (MIT)/research/0nubb/meas/' + cfgbase + '_test0mode_' + str(job_num) + '/0nubb_output'
data_dir2 = '/Users/theoares/Dropbox (MIT)/research/0nubb/meas/' + cfgbase + '_test0mode_' + str(job_num) + '/npr_momfrac_output'

L = 16
T = 48
set_dimensions(L, T)

q = np.array([0, 0, 0, 0])
# q_lat = to_lattice_momentum(q + bvec)
q_lat = np.sin(to_linear_momentum(q + bvec))            # choice of lattice momentum will affect how artifacts look, but numerics should look roughly the same
# q_lat = to_linear_momentum(q + bvec)
qlat_slash = slash(q_lat)

Zq1, Zq2 = np.zeros((n_boot), dtype = np.complex64), np.zeros((n_boot), dtype = np.complex64)
ZV1, ZV2 = np.zeros((n_boot), dtype = np.complex64), np.zeros((n_boot), dtype = np.complex64)
ZA1, ZA2 = np.zeros((n_boot), dtype = np.complex64), np.zeros((n_boot), dtype = np.complex64)

################################# 0NUBB ANALYSIS ################################
cfgs1 = []
for (dirpath, dirnames, file) in os.walk(data_dir1):
    cfgs1.extend(file)
for idx, cfg in enumerate(cfgs1):
    cfgs1[idx] = data_dir1 + '/' + cfgs1[idx]
n_cfgs1 = len(cfgs1)

k1, k2, props_k1, props_k2, props_q, GV1, GA1, GO = readfiles(cfgs1, q, False)
props_k1_b, props_k2_b, props_q_b = bootstrap(props_k1), bootstrap(props_k2), bootstrap(props_q)
GV1_boot, GA1_boot = np.array([bootstrap(GV1[mu]) for mu in range(4)]), np.array([bootstrap(GA1[mu]) for mu in range(4)])
props_k1_inv, props_k2_inv, props_q_inv = invert_props(props_k1_b), invert_props(props_k2_b), invert_props(props_q_b)
# props_k1_inv, props_k2_inv, props_q_inv = invert_props(antiprop(props_k1_b)), invert_props(props_k2_b), invert_props(props_q_b)    # try antipropagator
# props_k1_inv, props_k2_inv, props_q_inv = invert_props(props_k1_b), invert_props(antiprop(props_k2_b)), invert_props(props_q_b)
Zq1 = quark_renorm(props_q_inv, q_lat)
GammaV1, GammaA1 = np.zeros(GV1_boot.shape, dtype = np.complex64), np.zeros(GA1_boot.shape, dtype = np.complex64)
qDotV1, qDotA1 = np.zeros(GV1_boot.shape[1:]), np.zeros(GA1_boot.shape[1:])
for mu in range(4):
    # GammaV[mu], GammaA[mu] = amputate_threepoint(props_k1_inv, props_k2_inv, GV1_boot[mu]), amputate_threepoint(props_k1_inv, props_k2_inv, GA1_boot[mu])
    GammaV1[mu], GammaA1[mu] = amputate_threepoint(props_k2_inv, props_k1_inv, GV1_boot[mu]), amputate_threepoint(props_k2_inv, props_k1_inv, GA1_boot[mu])
    qDotV1, qDotA1 = qDotV1 + q_lat[mu] * GammaV1[mu], qDotA1 + q_lat[mu] * GammaA1[mu]
ZV1 = 12 * Zq1 * square(q_lat) / np.einsum('zaiaj,ji->z', qDotV1, qlat_slash)
ZA1 = 12 * Zq1 * square(q_lat) / np.einsum('zaiaj,jk,ki->z', qDotA1, gamma5, qlat_slash)

out_file1 = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/currents_test' + str(job_num) + '/Z_0nubb.h5'
f1 = h5py.File(out_file1, 'w')
f1['momenta'] = q
f1['ZV'] = ZV1
f1['ZA'] = ZA1
f1['Zq'] = Zq1
f1['cfgnum'] = n_cfgs1
f1.close()
print('Onubb output saved at: ' + out_file1)

############################## NPR_MOMFRAC ANALYSIS #############################
cfgs2 = []
for (dirpath, dirnames, file) in os.walk(data_dir2):
    cfgs2.extend(file)
for idx, cfg in enumerate(cfgs2):
    cfgs2[idx] = data_dir2 + '/' + cfgs2[idx]
n_cfgs2 = len(cfgs2)

props, GV, GA = readfiles_momfrac(cfgs2, q)
props_b = bootstrap(props)
GV2_boot, GA2_boot = np.array([bootstrap(GV[mu]) for mu in range(4)]), np.array([bootstrap(GA[mu]) for mu in range(4)])
props_inv = invert_props(props_b)
# props_q_inv = invert_props(antiprop(props_q_b))
Zq2 = quark_renorm(props_inv, q_lat)
GammaV2, GammaA2 = np.zeros(GV2_boot.shape, dtype = np.complex64), np.zeros(GA2_boot.shape, dtype = np.complex64)
qDotV2, qDotA2 = np.zeros(GV2_boot.shape[1:]), np.zeros(GA2_boot.shape[1:])
qlat_slash = slash(q_lat)
for mu in range(4):
    # GammaV2[mu], GammaA2[mu] = amputate_threepoint(props_k1_inv, props_k2_inv, GV2_boot[mu]), amputate_threepoint(props_k1_inv, props_k2_inv, GA2_boot[mu])
    GammaV2[mu], GammaA2[mu] = amputate_threepoint(props_inv, props_inv, GV2_boot[mu]), amputate_threepoint(props_inv, props_inv, GA2_boot[mu])
    qDotV2, qDotA2 = qDotV2 + q_lat[mu] * GammaV2[mu], qDotA2 + q_lat[mu] * GammaA2[mu]
ZV2 = 12 * Zq2 * square(q_lat) / np.einsum('zaiaj,ji->z', qDotV2, qlat_slash)
ZA2 = 12 * Zq2 * square(q_lat) / np.einsum('zaiaj,jk,ki->z', qDotA2, gamma5, qlat_slash)

out_file2 = '/Users/theoares/Dropbox (MIT)/research/0nubb/analysis_output/currents_test' + str(job_num) + '/Z_npr_momfrac.h5'
f2 = h5py.File(out_file2, 'w')
f2['momenta'] = q
f2['ZV'] = ZV2
f2['ZA'] = ZA2
f2['Zq'] = Zq2
f2['cfgnum'] = n_cfgs2
f2.close()
print('npr_momfrac output saved at: ' + out_file2)
