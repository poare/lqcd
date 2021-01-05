import numpy as np
from scipy.optimize import root
import h5py
import os
import analysis

cfgbase = 'cl3_16_48_b6p1_m0p2450'

################################# PARAMETERS ################################
job_num = 28390
mom_list = []
for i in range(-6, 7):
    mom_list.append([i, i, 0, 0])
#     for j in range(-6, 7):
#         for l in range(-6, 7):
#             for m in range(-6, 7):
# for i in range(6):
#     for j in range(6):
#         for l in range(6):
#             for m in range(6):
#                 mom_list.append([i, j, l, m])
print('Number of total sink momenta: ' + str(len(mom_list)))
#############################################################################
# data_dir = './output/' + cfgbase + '_' + str(job_num)
data_dir = '/Users/theoares/Dropbox (MIT)/research/npr_momfrac/meas/' + cfgbase + '_' + str(job_num)

mom_str_list = [analysis.plist_to_string(p) for p in mom_list]
analysis.mom_list = mom_list
analysis.mom_str_list = mom_str_list
ZV, ZA, Zq = analysis.current_analysis(data_dir, mom_list)
# ZA, Zq = analysis.current_analysis(data_dir, mom_list)

# out_file = '/Users/theoares/lqcd/npr_momfrac/analysis_output/currents' + str(job_num) + '/ZA.h5'
out_file = '/Users/theoares/Dropbox (MIT)/research/npr_momfrac/analysis_output/currents' + str(job_num) + '/Z.h5'
f = h5py.File(out_file, 'w')
f['momenta'] = mom_list
f['ZV'] = ZV
f['ZA'] = ZA
f['Zq'] = Zq
f['cfgnum'] = analysis.num_cfgs
f.close()
print('Output saved at: ' + out_file)
