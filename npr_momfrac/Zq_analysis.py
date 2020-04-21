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
#
# job_num = 20361
# mom_list = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5]]

# for the wall source
# job_num = 19214
# mom_list = [[i, i, i, i] for i in range(1, 6)]

# mom_list = []
# for i in range(-7, 8):
#     for j in range(-7, 8):
#         for k in range(-7, 8):
#             for l in range(-7, 8):
#                 mom_list.append([i, j, k, l])

# TODO use this one
# job_num = 19910
# job_num = 21664
mom_list = []
for i in range(5):
    for j in range(5):
        for k in range(5):
            for l in range(5):
                mom_list.append([i, j, k, l])


mom_list = [[2, 3, 1, 4]]
print('Number of total sink momenta: ' + str(len(mom_list)))
#############################################################################

# data_dir = './output/' + cfgbase + '_' + str(job_num)
data_dir = './phiala_code/d10a_data'
V = (16 ** 3) * 48
# for N in [1]:
mom_str_list = [analysis.plist_to_string(p) for p in mom_list]
analysis.mom_list = mom_list
analysis.mom_str_list = mom_str_list
Zq = analysis.Zq_analysis(data_dir, mom_list) / V

print(Zq)

# # out_file = '/Users/theoares/lqcd/npr_momfrac/analysis_output/jobZq' + str(job_num) + '/Zq_confirmation.h5'
# out_file = '/Users/theoares/lqcd/npr_momfrac/phiala_code/analysis_output/my_Zq_analysis.h5'
# f = h5py.File(out_file, 'w')
# f['momenta'] = mom_list
# f['Zq'] = Zq / V
# print(Zq / V)
# f.close()
# print('Output saved at: ' + out_file)
