import numpy as np
from scipy.optimize import root
import h5py
import os
import analysis

cfgbase = 'cl3_16_48_b6p1_m0p2450'

################################# PARAMETERS ################################
# job_num = 20401
job_num = 22454    # only run a subset of momentum right now
mom_tot = []
eps = 1e-10    # tolerance for being close to 0
# mixing_list = [[], [], []]
removed = []
# for i in range(0, 6):
#     for j in range(0, 6):
#         for l in range(0, 6):
#             for m in range(0, 6):
for i in range(-6, 7):
    for j in range(-6, 7):
        for l in range(-6, 7):
            for m in range(-6, 7):
                k = [i, j, l, m]
                if k == [0, 0, 0, 0]:
                    removed.append(k)
                    continue
                p_lat = analysis.to_lattice_momentum(k)
                L1, L2 = analysis.Lambda1(p_lat), analysis.Lambda2(p_lat)
                A11, A12, A22 = analysis.inner(L1, L1), analysis.inner(L1, L2), analysis.inner(L2, L2)
                detA = A11 * A22 - A12 * A12
                if np.abs(detA) >= eps:     # if A(p) is invertible
                    mom_tot.append(k)
                    # A_inv = (1 / detA) * np.array([
                    #     [A22, -A12],
                    #     [-A12, A11]
                    # ])
                    # mixing_list[0].append(A_inv)
                    # mixing_list[1].append(L1)
                    # mixing_list[2].append(L2)
                else:
                    removed.append(k)
mom_list = mom_tot
# determine which momenta make A not invertible

# TEST AT THIS MOMENTUM
# p = [1, 0, 1, 0]
# mom_list = [p]

print('Number of total (invertible) sink momenta: ' + str(len(mom_list)))
#############################################################################

data_dir = './output/' + cfgbase + '_' + str(job_num)

mom_str_list = [analysis.plist_to_string(p) for p in mom_list]
analysis.mom_list = mom_list
analysis.mom_str_list = mom_str_list

# Z11, Z12, Zq = analysis.run_analysis_mixing(data_dir, mom_list, mixing_list)    #make sure to uncomment to get the mixing_list
# Z11, Z12, Zq = analysis.run_analysis_mixing(data_dir, mom_list)

Pi_11, Pi_12, Zq, Gamma = analysis.run_analysis_mixing(data_dir, mom_list)

# Z11_mean = np.mean(Z11, axis = 1)
# Z11_std = np.std(Z11, axis = 1)
#
# Z12_mean = np.mean(Z12, axis = 1)
# Z12_std = np.std(Z12, axis = 1)

# print('Z11 is:')
# print(Z11)
#
# print('Z12 is')
# print(Z12)

# print('Z11 mean and std: ' + str(Z11_mean) + ' pm ' + str(Z11_std))
# print('Z12 mean and std: ' + str(Z12_mean) + ' pm ' + str(Z12_std))

out_file = '/Users/theoares/lqcd/npr_momfrac/analysis_output/mixing_job' + str(job_num) + '/Pi.h5'
f = h5py.File(out_file, 'w')
f['momenta'] = mom_list
# f['Z11'] = Z11
# f['Z12'] = Z12
f['Pi11'] = Pi_11
f['Pi12'] = Pi_12
f['Gamma'] = Gamma
f['Zq'] = Zq
# f['Gamma'] = Gamma_D
f['cfgnum'] = analysis.num_cfgs
f.close()
print('Output saved at: ' + out_file)
