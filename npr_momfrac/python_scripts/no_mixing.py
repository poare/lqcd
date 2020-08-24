import numpy as np
from scipy.optimize import root
import h5py
import os
import analysis

cfgbase = 'cl3_16_48_b6p1_m0p2450'

################################# PARAMETERS ################################
job_num = 20401
mom_tot = []
for i in range(0, 6):
    for j in range(0, 6):
        if i == 0 and j == 0:
            continue
        for k in range(0, 6):
            for l in range(0, 6):
                if k == 0 and l == 0:
                    continue
                mom_tot.append([i, j, k, l])
mom_list = mom_tot
print('Number of total sink momenta: ' + str(len(mom_list)))

# for job 17951, delete after testing is done
# mom_list = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]]
#############################################################################
# print('Sink momenta: ' + str(mom_list))

data_dir = './output/' + cfgbase + '_' + str(job_num)

# Uncomment here for irreps
# mom_sub_list = [p for p in mom_list if not (p[0] == 0 and p[1] == 0) and not (p[2] == 0 and p[3] == 0) and not (p[0] == 0 and p[2] == 0)]
# # mom_sub_list.remove([0, 1, 0, 1])
# mom_str_list = [analysis.plist_to_string(p) for p in mom_sub_list]
# print('Number of momenta with p[mu] != 0: ' + str(len(mom_sub_list)))
# analysis.mom_list = mom_sub_list
# analysis.mom_str_list = mom_str_list
# Z, Zq = analysis.run_analysis_irreps(data_dir, mom_sub_list)
# out_file = '/Users/theoares/lqcd/npr_momfrac/analysis_output/job' + str(job_num) + '/Z_irreps.h5'
# f = h5py.File(out_file, 'w')
# f['momenta'] = mom_sub_list
# f['Z1'] = Z[0]
# f['Z2'] = Z[1]
# f['Z3'] = Z[2]
# f['Zq'] = Zq
# f['cfgnum'] = analysis.num_cfgs
# f.close()
# print('Output saved at: ' + out_file)


# Uncomment here for vector components
for mu in range(0, 4):
    print('Running for mu = ' + str(mu + 1))
    mom_sub_list = [p for p in mom_list if p[mu] != 0]    # remove 0 terms so Born term is well defined.
    print('Number of momenta with p[mu] != 0: ' + str(len(mom_sub_list)))
    mom_str_list = [analysis.plist_to_string(p) for p in mom_sub_list]
    analysis.mom_list = mom_sub_list
    analysis.mom_str_list = mom_str_list
    Z, Zq = analysis.run_analysis_point_sources(data_dir, mom_sub_list, mu = mu)
    out_file = '/Users/theoares/lqcd/npr_momfrac/analysis_output/job' + str(job_num) + \
            '/O' + str(mu + 1) + str(mu + 1) + '.h5'
    f = h5py.File(out_file, 'w')
    f['momenta'] = mom_sub_list
    print(Z)
    f['Z'] = Z
    f['Zq'] = Zq
    # f['pts'] = pt_list
    f['cfgnum'] = analysis.num_cfgs
    f.close()
    print('Output saved at: ' + out_file)
