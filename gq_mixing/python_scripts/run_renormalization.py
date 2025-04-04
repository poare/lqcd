import numpy as np
from scipy.optimize import root
import h5py
import os

# n_boot = 50
n_boot = 200
from utils import *

################################## PARAMETERS #################################
base = '/Users/theoares/Dropbox (MIT)/research/gq_mixing/meas/'

# jobid = 114141                                                                 # 12^3, beta 6.1, heavy ensemble 
# jobid = 117649
# stem = 'cl21_12_24_b6p1_m0p2800m0p2450-a_' + str(jobid)
# l = 12
# t = 24

# jobid = 106539                                                                # old 12^3 heavy (wrong mass for quark propagators)
# stem = 'cl21_12_24_b6p1_m0p2800m0p2450_' + str(jobid)
# l = 12
# t = 24

# jobid = 113030                                                                # light 12^3
# stem = 'cl21_12_24_b6p3_m0p2416m0p2050-b_' + str(jobid)
# l = 12
# t = 24

# jobid = 113356                                                                # 16^4, need more configs though
# stem = 'cl21_16_16_b6p3_m0p2416m0p2050-a_' + str(jobid)
# l = 16
# t = 16

jobid = 115282
stem = 'cl21_48_96_b6p3_m0p2416_m0p2050_' + str(jobid)
l = 48
t = 96

data_dir = base + stem
L = Lattice(l, t)

eps = 1e-8
kmin, kmax = 1, 7
# k_range = range(kmin, kmax)
k_range = range(2, 13, 2)
k_list = []
for i, j, k, l in itertools.product(k_range, repeat = 4):
    k_list.append([i, j, k, l])
    kk = np.array([i, j, k, l])
    p_lat = L.to_lattice_momentum(kk)
    if np.abs(detA(p_lat, get_inner(3))) < eps:
        print(kk)
    # if np.abs(detA) >= eps:     # if A(p) is invertible
    #     mom_tot.append(k)
    # else:
    #     removed.append(k)
k_list = np.array(k_list)
# k_list = np.array([[1, 1, 1, 6], [1, 1, 6, 1], [1, 6, 1, 1], [6, 1, 1, 1]])
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
# unsorted_cfgs = unsorted_cfgs[:30]
n_cfgs = len(unsorted_cfgs)
print('Reading ' + str(n_cfgs) + ' configs.')
print(unsorted_cfgids)

# glue pieces. Comment out for now
"""
glue_dir = '/Users/theoares/Dropbox (MIT)/research/gq_mixing/glue_ops/cfgs_1100_1500/'
path_glue_ids = glue_dir + 'ops_for_patrick.txt'
path_glue_emt = glue_dir + 'ops_for_patrick.npy'
glue_emt_all = np.load(path_glue_emt)
glue_ids = []
with open(path_glue_ids, 'r') as f:
    for line in f:
        glue_ids.append(int(line.split('cfg_')[1][:-1]))
"""

cfgids = sorted(unsorted_cfgids)
cfgs = [x for _, x in sorted(zip(unsorted_cfgids, unsorted_cfgs), key = lambda pair: pair[0])]
"""
glue_emt = np.array([glue_emt_all[ii] for ii in range(len(glue_ids)) if glue_ids[ii] in cfgids])
glue_emt_b = bootstrap(glue_emt)
"""

############################### PERFORM ANALYSIS ###############################
start = time.time()
Zq = np.zeros((len(k_list), n_boot), dtype = np.complex64)

Zqq3 = np.zeros((len(k_list), 3, n_boot), dtype = np.complex64)
Zqq6 = np.zeros((len(k_list), 6, n_boot), dtype = np.complex64)

# results with tensor structure mixing. The first index specifies the irrep, [0] is tau_1^3 and [1] is tau_3^6
Pi1 = np.zeros((2, len(k_list), n_boot), dtype = np.complex64)
Pi2 = np.zeros((2, len(k_list), n_boot), dtype = np.complex64)
Zqq_mix = np.zeros((2, len(k_list), n_boot), dtype = np.complex64)

props_list = np.zeros((len(k_list), n_boot, 3, 4, 3, 4), dtype = np.complex64)
Gamma_qg3_list = np.zeros((len(k_list), 3, n_boot, 3, 4, 3, 4), dtype = np.complex64)
Gamma_qg6_list = np.zeros((len(k_list), 6, n_boot, 3, 4, 3, 4), dtype = np.complex64)
Gamma_qq3_list = np.zeros((len(k_list), 3, n_boot, 3, 4, 3, 4), dtype = np.complex64)
Gamma_qq6_list = np.zeros((len(k_list), 6, n_boot, 3, 4, 3, 4), dtype = np.complex64)
for k_idx, k in enumerate(k_list):
    print('Momentum index = ' + str(k_idx) + '; mom = ' + str(k))
    p_lat = L.to_lattice_momentum(k + bvec)

    props, Gqg_qlua, Gqq3, Gqq6 = readfiles(cfgs, k)
    """Gqg = np.einsum('zmn,zaibj->mnzaibj', glue_emt, props)
    Gqg3, Gqg6 = form_2d_sym_irreps(Gqg)"""
    Gqg3, Gqg6 = form_2d_sym_irreps(Gqg_qlua)
    props_b = bootstrap(props)
    props_list[k_idx] = props_b
    Gqg3_boot, Gqg6_boot = np.array([bootstrap(Gqg3[a]) for a in range(3)]), np.array([bootstrap(Gqg6[a]) for a in range(6)])
    Gqq3_boot, Gqq6_boot = np.array([bootstrap(Gqq3[a]) for a in range(3)]), np.array([bootstrap(Gqq6[a]) for a in range(6)])
    props_inv = invert_props(props_b)
    Zq[k_idx] = quark_renorm(props_inv, p_lat)
    print('Zq = ' + export_float_latex(np.mean(np.real(Zq[k_idx])), np.std(np.real(Zq[k_idx]), ddof = 1)))
    Gamma_qg3, Gamma_qg6 = np.zeros(Gqg3_boot.shape, dtype = np.complex64), np.zeros(Gqg6_boot.shape, dtype = np.complex64)
    Gamma_qq3, Gamma_qq6 = np.zeros(Gqq3_boot.shape, dtype = np.complex64), np.zeros(Gqq6_boot.shape, dtype = np.complex64)
    for t in range(3):      # t = irrep index, used for both tau_1^3 and tau_3^6
        Gamma_qg3[t] = amputate_threepoint(props_inv, props_inv, Gqg3_boot[t])
        Gamma_qq3[t] = amputate_threepoint(props_inv, props_inv, Gqq3_boot[t])
    for t in range(6):
        Gamma_qg6[t] = amputate_threepoint(props_inv, props_inv, Gqg6_boot[t])
        Gamma_qq6[t] = amputate_threepoint(props_inv, props_inv, Gqq6_boot[t])
    Gamma_qg3_list[k_idx] = Gamma_qg3
    Gamma_qg6_list[k_idx] = Gamma_qg6
    Gamma_qq3_list[k_idx] = Gamma_qq3
    Gamma_qq6_list[k_idx] = Gamma_qq6

    # run analysis with no mixing, just for Zqq
    Gamma_qq_tree = Gamma_qq_1(p_lat)
    Gamma_qq3_tree, Gamma_qq6_tree = form_2d_sym_irreps(Gamma_qq_tree)
    Gamma_qq3_tree_inv = np.array([np.linalg.inv(Gamma_qq3_tree[t]) for t in range(3)])
    Gamma_qq6_tree_inv = np.array([np.linalg.inv(Gamma_qq6_tree[t]) for t in range(6)])
    Tr_Zqq3 = np.einsum('tzaiaj,tji->tz', Gamma_qq3, Gamma_qq3_tree_inv)
    Zqq3[k_idx] = 12 * np.einsum('z,tz->tz', Zq[k_idx], 1 / Tr_Zqq3)
    Tr_Zqq6 = np.einsum('tzaiaj,tji->tz', Gamma_qq6, Gamma_qq6_tree_inv)
    Zqq6[k_idx] = 12 * np.einsum('z,tz->tz', Zq[k_idx], 1 / Tr_Zqq6)

    # print Zqq with no tensor mixing
    print('Zqq3, no tensor mixing = ' + str(np.mean(np.real(Zqq3[k_idx]), axis = 1)))
    print('Zqq6, no tensor mixing = ' + str(np.mean(np.real(Zqq6[k_idx]), axis = 1)))

    # run analysis with tensor structure mixing for Zqq on both irreps
    irrep_dims = [3, 6]
    Gamma_qq_irreps = [Gamma_qq3, Gamma_qq6]
    L1, L2 = Gamma_qq_1(p_lat), Gamma_qq_2(p_lat)
    for ii, dim in enumerate(irrep_dims):
        irr_label = irrep_label(dim)
        # print('Evaluating Zqq with tensor mixing on irrep: ' + irr_label)
        TrC_Gammaqq = np.einsum('tzaiaj->ztij', Gamma_qq_irreps[ii]) / 3.
        inner = get_inner(dim)
        Ainv = A_inv_ab(p_lat, inner)
        for b in range(n_boot):
            vb = np.array([
                inner(L1, TrC_Gammaqq[b]),
                inner(L2, TrC_Gammaqq[b])
            ])
            Pi1[ii, k_idx, b], Pi2[ii, k_idx, b] = Ainv @ vb
            Zqq_mix[ii, k_idx, b] = Zq[k_idx, b] / Pi1[ii, k_idx, b]
        # print('Zqq on irrep ' + irr_label + ' = ' + str(np.mean(np.real(Zqq_mix[ii, k_idx]))))
        print('Zqq on irrep ' + irr_label + ' = ' + \
            export_float_latex(np.mean(np.real(Zqq_mix[ii, k_idx])), np.std(np.real(Zqq_mix[ii, k_idx]), ddof = 1)))

    # Time per iteration
    print('Elapsed time: ' + str(time.time() - start))

################################## SAVE DATA ##################################
# out_file = '/Users/theoares/Dropbox (MIT)/research/gq_mixing/analysis_output/Zqq_' + str(jobid) + '_30cfgs.h5'
out_file = '/Users/theoares/Dropbox (MIT)/research/gq_mixing/analysis_output/Zqq_' + str(jobid) + '.h5'
f = h5py.File(out_file, 'w')
f['cfgnum'] = n_cfgs
f['momenta'] = k_list
f['props'] = props_list
"""f['glue_emt'] = glue_emt_b"""
f['Zq'] = Zq
f['Zqq3'] = Zqq3
f['Zqq6'] = Zqq6

f['Pi1'] = Pi1
f['Pi2'] = Pi2
f['Zqq_mix'] = Zqq_mix

f['Gamma_qg3'] = Gamma_qg3_list
f['Gamma_qg6'] = Gamma_qg6_list
f['Gamma_qq3'] = Gamma_qq3_list
f['Gamma_qq6'] = Gamma_qq6_list
f.close()
print('Output saved at: ' + out_file)
