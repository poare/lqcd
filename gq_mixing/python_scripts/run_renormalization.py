import numpy as np
from scipy.optimize import root
import h5py
import os

n_boot = 50
from utils import *

################################## PARAMETERS #################################
jobid = 89127
base = '/Users/theoares/Dropbox (MIT)/research/gq_mixing/meas/'
stem = 'cl21_48_96_b6p3_m0p2416_m0p2050_' + str(jobid)
data_dir = base + stem

l = 48
t = 96
L = Lattice(l, t)

k_list = [[2, 2, 2, 2], [4, 4, 4, 4], [6, 6, 6, 6], [8, 8, 8, 8], [3, 0, 0, 0], [6, 0, 0, 0], [9, 0, 0, 0]]
k_list = np.array(k_list)
print('Number of total momenta: ' + str(len(k_list)))

############################### PERFORM ANALYSIS ##############################
cfgs = []
for (dirpath, dirnames, file) in os.walk(data_dir):
    cfgs.extend(file)
for idx, cfg in enumerate(cfgs):
    cfgs[idx] = data_dir + '/' + cfgs[idx]
n_cfgs = len(cfgs)
print('Reading ' + str(n_cfgs) + ' configs.')

start = time.time()
Zq = np.zeros((len(k_list), n_boot), dtype = np.complex64)
Gamma_qg_list = np.zeros((len(q_list), 4, 4, n_boot, 3, 4, 3, 4), dtype = np.complex64)
Gamma_qq3_list = np.zeros((len(q_list), 3, n_boot, 3, 4, 3, 4), dtype = np.complex64)
Gamma_qq6_list = np.zeros((len(q_list), 6, n_boot, 3, 4, 3, 4), dtype = np.complex64)
for k_idx, k in enumerate(k_list):
    print('Momentum index: ' + str(k_idx))
    k_lat = to_lattice_momentum(k + bvec)
    props, Gqg, Gqq3, Gqq6 = readfiles(cfgs, k)
    props_b = bootstrap(props)
    Gqg_boot = np.array([[bootstrap(Gqq[mu][nu]) for nu in range(4)] for mu in range(4)])    # TODO figure out order of (mu, nu) (although won't matter bc symmetrizing)
    Gqq3_boot, Gqq6_boot = np.array([bootstrap(Gqq3[a]) for a in range(3)]), np.array([bootstrap(Gqq6[a]) for a in range(6)])
    props_inv = invert_props(props_b)
    Zq[k_idx] = quark_renorm(props_inv, k_lat)
    GammaV, GammaA = np.zeros(GV_boot.shape, dtype = np.complex64), np.zeros(GA_boot.shape, dtype = np.complex64)
    Gamma_qg, Gamma_qq3, Gamma_qq6 = np.zeros(Gqg_boot.shape, dtype = np.complex64), np.zeros(Gqq3_boot.shape, np.complex64), np.zeros(Gqq6_boot.shape, np.complex64)
    for mu, nu in itertools.product(range(4), repeat = 2):
        Gamma_qg[mu, nu] = amputate_threepoint(props_inv, props_inv, Gqg_boot[mu, nu])
    for a in range(3):
        Gamma_qq3[a] = amputate_threepoint(props_inv, props_inv, Gqq3_boot[a])
    for a in range(6):
        Gamma_qq6[a] = amputate_threepoint(props_inv, props_inv, Gqq6_boot[a])
    Gamma_qg_list[k_idx] = Gamma_qg
    Gamma_qq3_list[k_idx] = Gamma_qq3
    Gamma_qq6_list[k_idx] = Gamma_qq6

    # Time per iteration
    print('Elapsed time: ' + str(time.time() - start))


################################## SAVE DATA ##################################
out_file = '/Users/theoares/Dropbox (MIT)/research/gq_mixing/analysis_output/Z_' + str(jobid) + '.h5'
f = h5py.File(out_file, 'w')
f['cfgnum'] = n_cfgs
f['momenta'] = k_list
f['Zq'] = Zq
f['Gamma_qg'] = Gamma_qg_list
f['Gamma_qq3'] = Gamma_qq3_list
f['Gamma_qq6'] = Gamma_qq6_list
f.close()
print('Output saved at: ' + out_file)
