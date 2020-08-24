import numpy as np
from scipy.optimize import root
import h5py
import os
import analysis

# cfgbase = 'cl3_16_48_b6p1_m0p2450'
cfgbase = 'cl3_24_24_b6p1_m0p2450'

################################# PARAMETERS ################################
# job_num = 20401
# job_num = 22454    # only run a subset of momentum right now
job_num = 25291
mom_tot = []
eps = 1e-10    # tolerance for being close to 0
removed = []
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
                else:
                    removed.append(k)
mom_list = mom_tot

print('Number of total (invertible) sink momenta: ' + str(len(mom_list)))
#############################################################################

# data_dir = './output/' + cfgbase + '_' + str(job_num)
data_dir = '/Users/theoares/Dropbox (MIT)/research/npr_momfrac/meas/' + cfgbase + '_' + str(job_num)

mom_str_list = [analysis.plist_to_string(p) for p in mom_list]
analysis.mom_list = mom_list
analysis.mom_str_list = mom_str_list

# Z11, Z12, Zq, Gamma = analysis.run_analysis_mixing(data_dir, mom_list)
Pi_11, Pi_12, Zq, Gamma = analysis.run_analysis_mixing(data_dir, mom_list)
Z11, Z12 = Zq / Pi_11, Zq / Pi_12

# out_file = '/Users/theoares/lqcd/npr_momfrac/analysis_output/mixing_job' + str(job_num) + '/Pi.h5'
out_file = '/Users/theoares/Dropbox (MIT)/research/npr_momfrac/analysis_output/mixing_job' + str(job_num) + '/Pi.h5'
f = h5py.File(out_file, 'w')
f['momenta'] = mom_list
f['Z11'] = Z11
f['Z12'] = Z12
f['Pi11'] = Pi_11
f['Pi12'] = Pi_12
f['Gamma'] = Gamma
f['Zq'] = Zq
f['cfgnum'] = analysis.num_cfgs
f.close()
print('Output saved at: ' + out_file)
