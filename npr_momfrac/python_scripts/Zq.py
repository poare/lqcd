import numpy as np
from scipy.optimize import root
import h5py
import os
import analysis

cfgbase = 'cl3_16_48_b6p1_m0p2450'
mom_list = []

################################# PARAMETERS ################################

# for the wall source
# job_num = 19214
# mom_list = [[i, i, i, i] for i in range(1, 6)]

job_num = 22454
for i in range(-6, 7):
    for j in range(-6, 7):
        for k in range(-6, 7):
            for l in range(-6, 7):
                mom_list.append([i, j, k, l])

# for i in range(5):
#     for j in range(5):
#         for k in range(5):
#             for l in range(5):
#                 mom_list.append([i, j, k, l])
print('Number of total sink momenta: ' + str(len(mom_list)))
#############################################################################

data_dir = './output/' + cfgbase + '_' + str(job_num)
# data_dir = './phiala_code/d10a_data'
V = (16 ** 3) * 48
# for N in [1]:
mom_str_list = [analysis.plist_to_string(p) for p in mom_list]
analysis.mom_list = mom_list
analysis.mom_str_list = mom_str_list
Zq = analysis.Zq_analysis(data_dir, mom_list)

# print(Zq)

out_file = '/Users/theoares/lqcd/npr_momfrac/analysis_output/jobZq' + str(job_num) + '/Zq.h5'
# out_file = '/Users/theoares/lqcd/npr_momfrac/phiala_code/analysis_output/my_Zq_analysis.h5'
f = h5py.File(out_file, 'w')
f['momenta'] = mom_list
f['Zq'] = Zq
f.close()
print('Output saved at: ' + out_file)
