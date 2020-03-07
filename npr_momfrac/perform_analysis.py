import numpy as np
from scipy.optimize import root
import h5py
import os
import analysis

cfgbase = 'cl3_16_48_b6p1_m0p2450'

################################# PARAMETERS ################################
# job_num = 16583
job_num = 16636

# Momentum the propagators are run at.
prop_mom_list = [[0, 0, 0, 0], [2, 2, 2, 2], [-2, -2, -2, -2], [2, 2, -2, -2], [-2, -2, 2, 2]]

# Sink momentum that we tie up at.
mom_list = []
for i in range(1, 4 + 1):
    for j in range(1, 8 + 1):
        if j % 2 == 0:
            mom_list.extend([[i, i, i, j], [i, i, -i, j], [i, -i, i, j], [i, -i, -i, j], [-i, i, i, j], [-i, i, -i, j], [-i, -i, i, j], [-i, -i, -i, j]])
#############################################################################
print('Propagator momenta: ' + str(prop_mom_list))
print('Sink momenta: ' + str(mom_list))

mom_str_list = [analysis.plist_to_string(p) for p in mom_list]
analysis.prop_mom_list = prop_mom_list
analysis.mom_list = mom_list
analysis.mom_str_list = mom_str_list

out_folder = '/Users/theoares/lqcd/npr_momfrac/analysis_output'
out_file = out_folder + '/job' + str(job_num)
data_directory = './output/' + cfgbase + '_' + str(job_num)

mu, sigma = analysis.test_analysis_propagators(data_directory)

os.execute('mkdir ' + out_folder)
analysis.save_mu_sigma(mu, sigma, analysis_out_file)
