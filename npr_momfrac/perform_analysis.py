import numpy as np
from scipy.optimize import root
import h5py
import os
import analysis

cfgbase = 'cl3_16_48_b6p1_m0p2450'

################################# PARAMETERS ################################
# job_num = 17999    # less momenta
# mom_tot = []
# for i in range(-5, 6):
#     for j in range(-5, 6):
#         for k in range(-5, 6):
#             for l in range(1, 6):
#                 mom_tot.append([i, j, k, l])
# mom_list = analysis.cylinder(mom_tot, 2)

# job_num = 19910
job_num = 20401
mom_tot = []
for i in range(0, 6):
    for j in range(0, 6):
        for k in range(0, 6):
            for l in range(0, 6):
                mom_tot.append([i, j, k, l])
mom_list = mom_tot

print('Number of total sink momenta: ' + str(len(mom_list)))

# for job 17951, delete after testing is done
# mom_list = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]]
#############################################################################
# print('Sink momenta: ' + str(mom_list))

data_dir = './output/' + cfgbase + '_' + str(job_num)

for mu in range(0, 4):
    print('Running for mu = ' + str(mu + 1))
    mom_sub_list = [p for p in mom_list if p[mu] != 0]    # remove 0 terms so Born term is well defined.
    print('Number of momenta with p[mu] != 0: ' + str(len(mom_sub_list)))
    mom_str_list = [analysis.plist_to_string(p) for p in mom_sub_list]
    analysis.mom_list = mom_sub_list
    analysis.mom_str_list = mom_str_list
    Z, Zq = analysis.run_analysis_point_sources(data_dir, mom_sub_list, mu = mu)
    # Z, Zq, pt_list = analysis.run_analysis_untied(data_dir, mu = mu)
    out_file = '/Users/theoares/lqcd/npr_momfrac/analysis_output/job' + str(job_num) + \
            'born/O' + str(mu + 1) + str(mu + 1) + '.h5'
                # 'born/O' + str(mu + 1) + str(mu + 1) + '.h5'
    f = h5py.File(out_file, 'w')
    f['momenta'] = mom_sub_list
    f['Z'] = Z
    f['Zq'] = Zq
    # f['pts'] = pt_list
    f['cfgnum'] = analysis.num_cfgs
    f.close()
    print('Output saved at: ' + out_file)
