import numpy as np
from scipy.optimize import root
import h5py
import os
import analysis

cfgbase = 'cl3_16_48_b6p1_m0p2450'

################################# PARAMETERS ################################
job_num = 23014
# job_num = 23476
mom_list = []
for i in range(0, 6):
    for j in range(0, 6):
        for k in range(0, 6):
            for l in range(0, 6):
                if [i, j, k, l] == [0, 0, 0, 0]:
                    continue
                mom_list.append([i, j, k, l])
print('Number of total sink momenta: ' + str(len(mom_list)))
#############################################################################

# mom_list = [[1, 1, 1, 1], [2, 2, 2, 2]]

data_dir = './output/' + cfgbase + '_' + str(job_num)
# data_dir = './output/free_field_currents'

# Uncomment here for irreps
mom_str_list = [analysis.plist_to_string(p) for p in mom_list]
analysis.mom_list = mom_list
analysis.mom_str_list = mom_str_list
ZV, ZA, Zq = analysis.current_analysis(data_dir, mom_list)

# GammaV, GammaA, Zq = analysis.current_analysis(data_dir, mom_list)
out_file = '/Users/theoares/lqcd/npr_momfrac/analysis_output/currents' + str(job_num) + '/Z.h5'
# out_file = '/Users/theoares/lqcd/npr_momfrac/analysis_output/free_field_currents/Z.h5'
f = h5py.File(out_file, 'w')
f['momenta'] = mom_list
f['ZV'] = ZV
f['ZA'] = ZA
f['Zq'] = Zq
f['cfgnum'] = analysis.num_cfgs
f.close()
print('Output saved at: ' + out_file)
