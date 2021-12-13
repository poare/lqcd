import numpy as np
from scipy.optimize import root
import h5py
import os

n_boot = 50
from utils import *

################################## PARAMETERS #################################
# jobid = 89346
jobid = 90712
base = '/Users/theoares/Dropbox (MIT)/research/gq_mixing/meas/'
stem = 'cl21_48_96_b6p3_m0p2416_m0p2050_' + str(jobid)
data_dir = base + stem

l = 48
t = 96
L = Lattice(l, t)

k_list = [[2, 2, 2, 2], [4, 4, 4, 4], [6, 6, 6, 6], [8, 8, 8, 8], [3, 0, 0, 0], [6, 0, 0, 0], [9, 0, 0, 0]]
k_list = np.array(k_list)
print('Number of total momenta: ' + str(len(k_list)))

################################# READ IN MEAS #################################
unsorted_cfgs = []
unsorted_cfgids = []
for (dirpath, dirnames, file) in os.walk(data_dir):
    unsorted_cfgs.extend(file)
    for fi in file:
        unsorted_cfgids.append(int(fi[3 : 7]))    # slicing only good for cfgs in the 1000's
for idx, cfg in enumerate(unsorted_cfgs):
    unsorted_cfgs[idx] = data_dir + '/' + unsorted_cfgs[idx]
n_cfgs = len(unsorted_cfgs)
print('Reading ' + str(n_cfgs) + ' configs.')
print(unsorted_cfgids)

# glue pieces
glue_dir = '/Users/theoares/Dropbox (MIT)/research/gq_mixing/glue_ops/cfgs_1100_1500/'
path_glue_ids = glue_dir + 'ops_for_patrick.txt'
path_glue_emt = glue_dir + 'ops_for_patrick.npy'
glue_emt_all = np.load(path_glue_emt)
glue_ids = []
with open(path_glue_ids, 'r') as f:
    for line in f:
        glue_ids.append(int(line.split('cfg_')[1][:-1]))

cfgids = sorted(unsorted_cfgids)
cfgs = [x for _, x in sorted(zip(unsorted_cfgids, unsorted_cfgs), key = lambda pair: pair[0])]
glue_emt = np.array([glue_emt_all[ii] for ii in range(len(glue_ids)) if glue_ids[ii] in cfgids])
glue_emt_b = bootstrap(glue_emt)

# cfgs = []
# for (dirpath, dirnames, file) in os.walk(data_dir):
#     cfgs.extend(file)
# for idx, cfg in enumerate(cfgs):
#     cfgs[idx] = data_dir + '/' + cfgs[idx]
# n_cfgs = len(cfgs)
# print('Reading ' + str(n_cfgs) + ' configs.')

############################### PERFORM ANALYSIS ###############################
start = time.time()
Zq = np.zeros((len(k_list), n_boot), dtype = np.complex64)
props_list = np.zeros((len(k_list), n_boot, 3, 4, 3, 4), dtype = np.complex64)
Gamma_qg3_list = np.zeros((len(k_list), 3, n_boot, 3, 4, 3, 4), dtype = np.complex64)
Gamma_qg6_list = np.zeros((len(k_list), 6, n_boot, 3, 4, 3, 4), dtype = np.complex64)
Gamma_qq3_list = np.zeros((len(k_list), 3, n_boot, 3, 4, 3, 4), dtype = np.complex64)
Gamma_qq6_list = np.zeros((len(k_list), 6, n_boot, 3, 4, 3, 4), dtype = np.complex64)
for k_idx, k in enumerate(k_list):
    print('Momentum index: ' + str(k_idx))
    k_lat = L.to_lattice_momentum(k + bvec)
    props, Gqg_qlua, Gqq3, Gqq6 = readfiles(cfgs, k)
    Gqg = np.einsum('zmn,zaibj->mnzaibj', glue_emt, props)
    Gqg3, Gqg6 = form_2d_sym_irreps(Gqg)
    props_b = bootstrap(props)
    props_list[k_idx] = props_b
    # Gqg_boot = np.array([[bootstrap(Gqg[mu][nu]) for nu in range(4)] for mu in range(4)])
    Gqg3_boot, Gqg6_boot = np.array([bootstrap(Gqg3[a]) for a in range(3)]), np.array([bootstrap(Gqg6[a]) for a in range(6)])
    Gqq3_boot, Gqq6_boot = np.array([bootstrap(Gqq3[a]) for a in range(3)]), np.array([bootstrap(Gqq6[a]) for a in range(6)])
    props_inv = invert_props(props_b)
    Zq[k_idx] = quark_renorm(props_inv, k_lat)
    print(Zq[k_idx])
    Gamma_qg3, Gamma_qg6 = np.zeros(Gqg3_boot.shape, dtype = np.complex64), np.zeros(Gqg6_boot.shape, dtype = np.complex64)
    Gamma_qq3, Gamma_qq6 = np.zeros(Gqq3_boot.shape, dtype = np.complex64), np.zeros(Gqq6_boot.shape, dtype = np.complex64)
    for a in range(3):
        Gamma_qg3[a] = amputate_threepoint(props_inv, props_inv, Gqg3_boot[a])
        Gamma_qq3[a] = amputate_threepoint(props_inv, props_inv, Gqq3_boot[a])
    for a in range(6):
        Gamma_qg6[a] = amputate_threepoint(props_inv, props_inv, Gqg6_boot[a])
        Gamma_qq6[a] = amputate_threepoint(props_inv, props_inv, Gqq6_boot[a])
    Gamma_qg3_list[k_idx] = Gamma_qg3
    Gamma_qg6_list[k_idx] = Gamma_qg6
    Gamma_qq3_list[k_idx] = Gamma_qq3
    Gamma_qq6_list[k_idx] = Gamma_qq6

    # run analysis with no mixing, just for Zqq


    # Time per iteration
    print('Elapsed time: ' + str(time.time() - start))


################################## SAVE DATA ##################################
out_file = '/Users/theoares/Dropbox (MIT)/research/gq_mixing/analysis_output/Z_' + str(jobid) + '.h5'
f = h5py.File(out_file, 'w')
f['cfgnum'] = n_cfgs
f['momenta'] = k_list
f['props'] = props_list
f['glue_emt'] = glue_emt_b
f['Zq'] = Zq
f['Gamma_qg3'] = Gamma_qg3_list
f['Gamma_qg6'] = Gamma_qg6_list
f['Gamma_qq3'] = Gamma_qq3_list
f['Gamma_qq6'] = Gamma_qq6_list
f.close()
print('Output saved at: ' + out_file)
